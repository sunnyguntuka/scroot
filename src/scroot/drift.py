# Apache-2.0. OSS drift / regression detection — real implementation.
from __future__ import annotations

import pathlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._entitlements import get_enterprise

if TYPE_CHECKING:
    from .result import EntailmentResult


@dataclass
class RegressionReport:
    """Result of regression_check().

    Attributes:
        passed: True if no metric regressed beyond its threshold.
        regressions: List of dicts describing each regression, sorted by
            magnitude (largest delta first).
        summary: Human-readable one-liner.
    """

    passed: bool
    regressions: list[dict] = field(default_factory=list)
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.summary:
            if self.passed:
                self.summary = "No regressions detected."
            else:
                names = ", ".join(r["metric"] for r in self.regressions)
                self.summary = f"Regression detected in: {names}."


_DEFAULT_THRESHOLDS = {
    "iqs": 0.05,
    "groundedness": 0.10,
    "completeness": 0.10,
    "relevance": 0.10,
    "consistency": 0.10,
    "confidence": 0.10,
}


def regression_check(
    current: "EntailmentResult | list[EntailmentResult] | dict",
    baseline: "EntailmentResult | dict | str | pathlib.Path",
    *,
    thresholds: dict[str, float] | None = None,
) -> RegressionReport:
    """Point-in-time regression check against a baseline for CI gating.

    Fully OSS — pure Python, no external API, no scroot-cloud required.

    Args:
        current: The result(s) to check. Accepts an ``EntailmentResult``,
            a list of them (averaged), or a plain dict from ``result.to_dict()``.
        baseline: The reference to compare against. Accepts an
            ``EntailmentResult``, a dict (from ``result.to_dict()``), or a
            path to a JSON file produced by a previous ``result.to_dict()``
            serialization.
        thresholds: Per-metric delta thresholds. A drop larger than the
            threshold for that metric is a regression. Defaults:
            IQS -0.05; all individual metrics -0.10.

    Returns:
        RegressionReport with passed/failed status and regression details.
    """
    thr = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    current_dict = _to_dict(current)
    baseline_dict = _to_dict(baseline)

    metrics = ["iqs", "groundedness", "completeness", "relevance", "consistency", "confidence"]
    regressions = []

    for metric in metrics:
        cur_val = current_dict.get(metric)
        base_val = baseline_dict.get(metric)
        if cur_val is None or base_val is None:
            continue
        delta = cur_val - base_val  # negative = regression
        limit = thr.get(metric, 0.10)
        if delta < -limit:
            regressions.append({
                "metric": metric,
                "current": round(cur_val, 4),
                "baseline": round(base_val, 4),
                "delta": round(delta, 4),
                "threshold": -limit,
            })

    regressions.sort(key=lambda r: r["delta"])
    return RegressionReport(passed=not regressions, regressions=regressions)


def _to_dict(obj: object) -> dict:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (str, pathlib.Path)):
        with open(obj, encoding="utf-8") as f:
            return json.load(f)
    if isinstance(obj, list):
        dicts = [_to_dict(item) for item in obj]
        if not dicts:
            return {}
        keys = ["iqs", "groundedness", "completeness", "relevance", "consistency", "confidence"]
        averaged: dict = {}
        for k in keys:
            vals = [d[k] for d in dicts if d.get(k) is not None]
            averaged[k] = round(sum(vals) / len(vals), 4) if vals else None
        return averaged
    # EntailmentResult
    return obj.to_dict()


def continuous(*args, **kwargs) -> object:
    """Cloud handoff to Ampulla for longitudinal drift monitoring.

    scroot does not implement continuous drift monitoring — that is Ampulla's
    domain. This seam passes the call through to scroot-cloud, which registers
    the Ampulla integration. See https://scroot.dev/cloud/drift for details.
    """
    return get_enterprise("drift.continuous").start(*args, **kwargs)
