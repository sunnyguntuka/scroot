"""Dimension applicability gating for the IQS composite.

The IQS is a weighted harmonic mean of five metrics. The harmonic mean is
deliberately unforgiving: any single near-zero metric drags the composite toward
zero. That is the right behaviour when the metric *is* measuring a real failure
(a hallucinated claim -> low groundedness -> low IQS). It is the *wrong*
behaviour when a metric is structurally **inapplicable** to the task.

The motivating case is summarization. The task query is generic
("Summarize the following article."), so:

  - **relevance** - there is no specific information need for the response to be
    "relevant" to; the relevance metric returns a pathologically low score
    (~0.003 on SummEval) for every sample, which then collapses IQS via the
    harmonic mean even when groundedness is high (~0.9). This destroys IQS's
    correlation with human judgement (rho ~0.12) despite groundedness tracking
    consistency well (rho ~0.36).

  - **consistency** - a single-sentence response has no second clause to
    contradict, so the contradiction check has nothing to measure.

This module provides cheap, deterministic, input-based predicates that detect
those cases. When a dimension is inapplicable, the caller sets its score to
``None`` so :func:`scroot.composite.compute_iqs_detailed` excludes it and
renormalises the remaining weights (groundedness is never gated out).
"""

from __future__ import annotations

import re

# A generic query expresses no specific information need. Matches the common
# summarise / explain / describe templates at the start of the query.
_GENERIC_QUERY_PATTERNS = re.compile(
    r"^\s*(summari[sz]e|"
    r"what (does|is|are|was) (this|the|it)\b|"
    r"give (me )?(a |the )?summary|"
    r"explain (this|the)\b|"
    r"describe (this|the)\b|"
    r"tell me about (this|the)\b|"
    r"provide (a |the )?summary)",
    re.IGNORECASE,
)

_QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "which",
    "did", "does", "is", "are", "was", "were", "can", "could",
    "will", "would", "should", "whom", "whose",
}

# Generic-query word-count ceiling: a short query with no question word and no
# specific named entity carries no information need to be relevant to.
_GENERIC_WORD_CEILING = 8

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def is_generic_query(query: str | None) -> bool:
    """True when the query expresses no specific information need.

    A query is generic when EITHER:
      - it matches a summarise/explain/describe template, OR
      - it is short (<= 8 words) AND contains no question word AND contains no
        specific named entity (a capitalised token that is not the first word).

    Generic queries make the ``relevance`` metric meaningless: there is nothing
    specific for the response to be relevant to.
    """
    if not query:
        return True
    q = query.strip()
    if not q:
        return True
    if _GENERIC_QUERY_PATTERNS.match(q):
        return True

    words = q.split()
    has_question_word = any(w.lower().strip("?,.") in _QUESTION_WORDS for w in words)
    if has_question_word:
        return False
    # A specific named entity (capitalised mid-sentence token) signals a real
    # information need even in a short query ("Tesla earnings").
    has_named_entity = any(
        w[:1].isupper() and not w.isupper() for w in words[1:]
    )
    return len(words) <= _GENERIC_WORD_CEILING and not has_named_entity


def response_sentence_count(response: str | None) -> int:
    """Number of non-trivial sentences in the response.

    Used to gate ``consistency``: a response with < 2 sentences has no second
    clause for the contradiction check to compare against.
    """
    if not response:
        return 0
    parts = [s for s in _SENTENCE_SPLIT.split(response.strip()) if s.strip()]
    return len(parts)


def inapplicable_dimensions(
    query: str | None,
    response: str | None,
) -> set[str]:
    """Return the set of IQS dimensions that are inapplicable for this input.

    ``groundedness`` and ``completeness`` are never gated here:
      - groundedness is the primary faithfulness signal and must always count;
      - completeness self-zeroes only on a genuine omission, which is a real
        measurement, not an inapplicability.

    Args:
        query: The user query / task prompt.
        response: The model response.

    Returns:
        A subset of ``{"relevance", "consistency"}``.
    """
    out: set[str] = set()
    if is_generic_query(query):
        out.add("relevance")
    if response_sentence_count(response) < 2:
        out.add("consistency")
    return out
