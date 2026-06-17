# Apache-2.0. OSS calibration — real implementation, no cloud dependency.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._entitlements import get_enterprise

if TYPE_CHECKING:
    from .result import EntailmentResult


@dataclass
class CalibrationResult:
    """Output of calibrate().

    Attributes:
        threshold: IQS threshold that maximises the objective on the labeled data.
        weights: Per-metric weight overrides to pass to Auditor(weights=...).
            ``None`` means the default weights are already optimal.
        precision: Precision at the chosen threshold.
        recall: Recall at the chosen threshold.
        f1: F1 score at the chosen threshold.
        confusion_matrix: Counts at the chosen threshold (tp, fp, tn, fn).
        n_samples: Number of labeled samples used.
    """

    threshold: float
    weights: dict[str, float] | None
    precision: float
    recall: float
    f1: float
    confusion_matrix: dict[str, int]
    n_samples: int


def calibrate(
    labeled_data: list[tuple["EntailmentResult", bool]],
    *,
    target_precision: float | None = None,
    threshold_step: float = 0.05,
) -> CalibrationResult:
    """Fit an IQS threshold (and optionally per-metric weights) from labeled data.

    Fully OSS — pure Python, no external API, no scroot-cloud required.

    Args:
        labeled_data: List of ``(EntailmentResult, passed)`` pairs where
            ``passed`` is the human judgment (True = acceptable response).
        target_precision: If set, select the lowest threshold that achieves
            at least this precision. Useful for compliance deployments where
            false-positives (approving a bad response) must be bounded.
            Falls back to best-F1 if no threshold meets the target.
        threshold_step: Grid resolution. Default 0.05 (thresholds 0.30..0.95).

    Returns:
        CalibrationResult with the optimal threshold, precision, recall, F1,
        and confusion-matrix counts. ``weights`` is always None in this OSS
        implementation (weight search is the managed lifecycle in scroot Cloud).
    """
    if not labeled_data:
        raise ValueError("calibrate() requires at least one labeled sample.")

    scores = [r.iqs for r, _ in labeled_data]
    labels = [bool(p) for _, p in labeled_data]
    n = len(labeled_data)

    thresholds = [
        round(0.30 + i * threshold_step, 4)
        for i in range(int((0.95 - 0.30) / threshold_step) + 1)
    ]

    best: dict = {"f1": -1.0, "threshold": 0.70, "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for thr in thresholds:
        tp = fp = tn = fn = 0
        for score, label in zip(scores, labels):
            pred_pass = score >= thr
            if pred_pass and label:
                tp += 1
            elif pred_pass and not label:
                fp += 1
            elif not pred_pass and not label:
                tn += 1
            else:
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        if target_precision is not None:
            if prec >= target_precision and (
                best["f1"] < 0 or thr < best["threshold"]
            ):
                best = {"f1": f1, "threshold": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        else:
            if f1 > best["f1"]:
                best = {"f1": f1, "threshold": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    # Fall back to best-F1 if target_precision was requested but not achievable.
    if best["f1"] < 0:
        return calibrate(labeled_data, threshold_step=threshold_step)

    tp, fp, tn, fn = best["tp"], best["fp"], best["tn"], best["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = best["f1"]

    return CalibrationResult(
        threshold=best["threshold"],
        weights=None,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        n_samples=n,
    )


def schedule_recalibration(agent: object, cadence: str) -> object:
    """Cloud: stored, scheduled, versioned, audit-grade calibration lifecycle."""
    return get_enterprise("calibration.schedule").schedule(agent, cadence)
