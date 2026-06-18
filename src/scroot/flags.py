"""Flag detection: identify specific quality issues.

Flags are string labels attached to a EntailmentResult when
specific patterns indicate known quality problems.
"""

from __future__ import annotations


# Built-in defaults for each flag threshold.
# These match the values used before F1.4 was implemented and are the
# reference values emitted by ``calibrate()`` as the starting point.
DEFAULT_FLAG_THRESHOLDS: dict[str, float] = {
    # hallucination_risk: fires when groundedness is low AND confidence is high
    "hallucination_risk_groundedness": 0.5,
    "hallucination_risk_confidence": 0.7,
    # off_topic: relevance below threshold
    "off_topic": 0.3,
    # self_contradictory: consistency below threshold
    "self_contradictory": 0.7,
    # incomplete: completeness below threshold
    "incomplete": 0.3,
    # ungrounded: groundedness below threshold (strict)
    "ungrounded": 0.3,
}


def detect_flags(
    groundedness: float | None,
    completeness: float,
    relevance: float,
    consistency: float,
    confidence: float,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Detect quality flags based on metric scores.

    Args:
        groundedness: Groundedness score (0–1) or ``None`` if no context.
        completeness: Completeness score (0–1).
        relevance: Relevance score (0–1).
        consistency: Consistency score (0–1).
        confidence: Confidence score (0–1).
        thresholds: Optional per-flag threshold overrides. Keys match those
            in ``DEFAULT_FLAG_THRESHOLDS``; unset keys fall back to defaults.
            Pass via ``Auditor(flag_thresholds=...)`` rather than calling
            this function directly.

    Returns:
        List of flag strings. Empty list = no issues detected.
    """
    thr = {**DEFAULT_FLAG_THRESHOLDS, **(thresholds or {})}
    flags = []

    if (
        groundedness is not None
        and groundedness < thr["hallucination_risk_groundedness"]
        and confidence > thr["hallucination_risk_confidence"]
    ):
        flags.append("hallucination_risk")

    if relevance < thr["off_topic"]:
        flags.append("off_topic")

    if consistency < thr["self_contradictory"]:
        flags.append("self_contradictory")

    if completeness < thr["incomplete"]:
        flags.append("incomplete")

    if groundedness is not None and groundedness < thr["ungrounded"]:
        flags.append("ungrounded")

    return flags
