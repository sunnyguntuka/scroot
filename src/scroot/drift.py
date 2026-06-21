# Apache-2.0. OSS drift / regression detection: real implementation.
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
    min_effect_size: float | None = None,
    confidence: float | None = None,
) -> RegressionReport:
    """Point-in-time regression check against a baseline for CI gating.

    Fully OSS: pure Python, no external API, no scroot-cloud required.

    When ``current`` is a list and ``min_effect_size`` / ``confidence`` are
    set, the check uses Mann-Whitney U (requires scipy) to distinguish true
    regressions from sampling noise. A difference is flagged only when the
    effect is both practically significant (``min_effect_size``) **and**
    statistically significant at the given ``confidence`` level.

    Args:
        current: The result(s) to check. Accepts an ``EntailmentResult``,
            a list of them (averaged), or a plain dict from ``result.to_dict()``.
        baseline: The reference to compare against. Accepts an
            ``EntailmentResult``, a dict (from ``result.to_dict()``), or a
            path to a JSON file produced by a previous ``result.to_dict()``
            serialization. For statistical comparison, pass a list of
            ``EntailmentResult`` objects as the baseline too.
        thresholds: Per-metric delta thresholds. A drop larger than the
            threshold for that metric is a regression. Defaults:
            IQS -0.05; all individual metrics -0.10.
        min_effect_size: Minimum absolute delta to consider practically
            significant (e.g. ``0.03`` ignores drops smaller than 0.03).
            Only used when ``confidence`` is also set and ``current`` is a
            list. Ignored otherwise.
        confidence: Statistical significance level (0–1) for Mann-Whitney U.
            ``0.95`` means flag only when p-value < 0.05. Requires scipy.
            When scipy is not installed, falls back to the raw-average
            comparison with a logged warning.

    Returns:
        RegressionReport with passed/failed status and regression details.
    """
    thr = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    # Statistical path: list vs list with Mann-Whitney U
    use_stats = (
        confidence is not None
        and isinstance(current, list)
        and isinstance(baseline, list)
        and len(current) >= 2
        and len(baseline) >= 2
    )

    if use_stats:
        return _regression_check_statistical(
            current=current,
            baseline=baseline,
            thresholds=thr,
            min_effect_size=min_effect_size or 0.0,
            confidence=confidence,
        )

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


def _regression_check_statistical(
    current: "list",
    baseline: "list",
    thresholds: dict,
    min_effect_size: float,
    confidence: float,
) -> RegressionReport:
    """Internal: Mann-Whitney U regression check for list vs list."""
    import logging
    _log = logging.getLogger(__name__)

    try:
        from scipy.stats import mannwhitneyu  # type: ignore[import]
        has_scipy = True
    except ImportError:
        has_scipy = False
        _log.warning(
            "scipy not installed; regression_check() falling back to raw-average "
            "comparison. Install scipy for statistical significance testing: "
            "pip install scipy"
        )

    metrics_names = ["iqs", "groundedness", "completeness", "relevance", "consistency", "confidence"]
    regressions = []
    alpha = 1.0 - confidence

    def _extract(items: list, metric: str) -> list[float]:
        vals = []
        for item in items:
            d = _to_dict(item)
            v = d.get(metric)
            if v is not None:
                vals.append(float(v))
        return vals

    for metric in metrics_names:
        cur_vals = _extract(current, metric)
        bas_vals = _extract(baseline, metric)
        if not cur_vals or not bas_vals:
            continue

        cur_mean = sum(cur_vals) / len(cur_vals)
        bas_mean = sum(bas_vals) / len(bas_vals)
        delta = cur_mean - bas_mean
        limit = thresholds.get(metric, 0.10)

        if abs(delta) < min_effect_size:
            continue  # effect too small to care about

        if delta >= -limit:
            continue  # not a regression by threshold

        # Statistical gating
        statistically_significant = True
        p_value: float | None = None
        if has_scipy and len(cur_vals) >= 2 and len(bas_vals) >= 2:
            try:
                # alternative="less": tests if current is stochastically smaller
                stat, p_value = mannwhitneyu(cur_vals, bas_vals, alternative="less")
                statistically_significant = p_value < alpha
            except Exception:
                statistically_significant = True  # fail open → conservative

        if not statistically_significant:
            continue  # within sampling variance, not a real regression

        reg: dict = {
            "metric": metric,
            "current": round(cur_mean, 4),
            "baseline": round(bas_mean, 4),
            "delta": round(delta, 4),
            "threshold": -limit,
        }
        if p_value is not None:
            reg["p_value"] = round(p_value, 4)
        regressions.append(reg)

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

    scroot does not implement continuous drift monitoring; that is Ampulla's
    domain. This seam passes the call through to scroot-cloud, which registers
    the Ampulla integration. See https://scroot.dev/cloud/drift for details.
    """
    return get_enterprise("drift.continuous").start(*args, **kwargs)
