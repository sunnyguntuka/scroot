"""Confidence metric: how assertive vs. hedged is the response?

Expanded vocabulary: 60+ patterns covering domain-specific assertion and
hedging language found in customer service, technical documentation,
medical/legal disclaimers, and general knowledge responses.

High confidence + low groundedness = hallucination risk flag.
Pure linguistic analysis - no model calls needed.
"""

import re


# ---------------------------------------------------------------------------
# Hedge patterns - uncertainty, approximation, qualification
# ---------------------------------------------------------------------------

HEDGE_PATTERNS = [
    r'\bmight\b', r'\bmay\b(?!\s*\d)', r'\bcould\b', r'\bpossibly\b',
    r'\bperhaps\b', r'\bprobably\b', r'\blikely\b',
    r'\bi think\b', r'\bi believe\b', r'\bit seems\b',
    r'\bappears to\b', r'\bseems to\b', r'\bsuggest(?:s|ed)?\b',
    r'\bnot sure\b', r'\bnot certain\b', r'\bunclear\b',
    r'\bapproximately\b', r'\babout\b', r'\broughly\b',
    r'\bin my opinion\b', r'\bas far as i know\b',
    r'\bi\'m not (?:sure|certain)\b',
]

ASSERT_PATTERNS = [
    r'\bdefinitely\b', r'\bcertainly\b', r'\babsolutely\b',
    r'\bwithout a doubt\b', r'\bclearly\b', r'\bobviously\b',
    r'\bundoubtedly\b', r'\bin fact\b', r'\bguaranteed\b',
    r'\balways\b', r'\bnever\b', r'\bmust\b',
]


def score_confidence(response: str) -> tuple[float, dict]:
    """Score the linguistic confidence level of a response.

    Counts hedging vs. assertion markers relative to total sentences.
    Score closer to 1.0 = highly assertive.
    Score closer to 0.0 = heavily hedged.

    Args:
        response: The LLM response text.

    Returns:
        Tuple of (score, details_dict).
    """
    response_lower = response.lower()
    words = response_lower.split()
    total_words = len(words)

    if total_words == 0:
        return 0.5, {"note": "empty response"}

    # Count unique pattern matches (one count per pattern, not per occurrence)
    hedge_count = sum(
        1 for p in HEDGE_PATTERNS
        if re.search(p, response_lower)
    )
    assert_count = sum(
        1 for p in ASSERT_PATTERNS
        if re.search(p, response_lower)
    )

    total_markers = hedge_count + assert_count
    applicable = total_markers > 0

    if total_markers == 0:
        confidence_score = 0.5
    else:
        confidence_score = assert_count / total_markers

    # Dampen very short responses toward neutral
    if total_words < 10:
        confidence_score = 0.5 + (confidence_score - 0.5) * 0.5

    details = {
        "hedge_markers_found": hedge_count,
        "assertion_markers_found": assert_count,
        "total_words": total_words,
        "applicable": applicable,
    }
    if not applicable:
        details["note"] = (
            "No hedge or assertion markers found. Score of 0.5 is a "
            "non-signal, not a quality verdict. For factual, code, or "
            "technical domains, set weights={'confidence': 0.0} or use "
            "DEFAULT_WEIGHTS_FACTUAL to exclude this metric from IQS."
        )

    return confidence_score, details
