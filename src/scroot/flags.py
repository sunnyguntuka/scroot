"""Flag detection: identify specific quality issues.

Flags are string labels attached to a EntailmentResult when
specific patterns indicate known quality problems.
"""

from __future__ import annotations


def detect_flags(
    groundedness: float | None,
    completeness: float,
    relevance: float,
    consistency: float,
    confidence: float,
) -> list[str]:
    """Detect quality flags based on metric scores.

    Args:
        All metric scores (0-1 or None).

    Returns:
        List of flag strings.
    """
    flags = []

    if groundedness is not None and groundedness < 0.5 and confidence > 0.7:
        flags.append("hallucination_risk")

    if relevance < 0.3:
        flags.append("off_topic")

    if consistency < 0.7:
        flags.append("self_contradictory")

    if completeness < 0.3:
        flags.append("incomplete")

    if groundedness is not None and groundedness < 0.3:
        flags.append("ungrounded")

    return flags
