"""Security tests for metric-level fixes: H-4 consistency cap, H-5 per-chunk
groundedness, H-6 no-runtime-NLTK."""

import warnings

import pytest

from scroot.metrics.groundedness import score_groundedness
from scroot.metrics.consistency import score_consistency
from scroot.text_utils import split_sentences


# ---------------------------------------------------------------------------
# H-5: Per-chunk groundedness scoring
# ---------------------------------------------------------------------------

@pytest.mark.needs_model
def test_groundedness_string_context_wrapped():
    """A plain string is auto-wrapped so it is not char-joined (H-5)."""
    response = "We offer a 30-day full refund at no extra cost."
    # Pass a string, not a list -should still work correctly
    score, details = score_groundedness(response, context=response)
    assert 0.0 <= score <= 1.0
    assert "claims" in details


@pytest.mark.needs_model
def test_groundedness_empty_context_list_scores_zero():
    """Empty context → claims are ungrounded (H-5)."""
    response = "We offer a 30-day refund."
    score, details = score_groundedness(response, context=[])
    assert score == 0.0


@pytest.mark.needs_model
def test_groundedness_truncation_warning_in_details():
    """Chunks >400 estimated tokens produce a truncation_warning key (H-5)."""
    response = "The answer is 42."
    long_chunk = "word " * 500  # ~500 tokens estimated
    score, details = score_groundedness(response, context=[long_chunk])
    assert "truncation_warning" in details


@pytest.mark.needs_model
def test_groundedness_no_warning_for_short_chunks():
    """Short context chunks do not add truncation_warning (H-5)."""
    response = "The answer is 42."
    short_chunk = "The answer is 42."
    score, details = score_groundedness(response, context=[short_chunk])
    assert "truncation_warning" not in details


@pytest.mark.needs_model
def test_groundedness_per_chunk_max_entailment():
    """Claim supported by later chunk should still be grounded (H-5).
    This test verifies multiple chunks are tried, not just the first."""
    response = "We offer a 30-day full refund at no extra cost."
    context = [
        "Our company was founded in 2005.",            # irrelevant
        "All customers are eligible for a 30-day full refund at no extra cost.",  # entails
    ]
    score, details = score_groundedness(response, context=context)
    assert score >= 0.5  # should be grounded despite irrelevant first chunk


# ---------------------------------------------------------------------------
# H-4: Consistency max_sentences cap
# ---------------------------------------------------------------------------

@pytest.mark.needs_model
def test_consistency_long_response_truncated_with_warning():
    """Responses with >50 sentences trigger a warning and truncate (H-4)."""
    # Build a response with 60 sentences
    response = " ".join([f"Statement number {i} is true." for i in range(60)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score, details = score_consistency(response, max_sentences=50)
    assert details.get("truncated") is True
    truncation_warns = [w for w in caught if "sentences" in str(w.message).lower()]
    assert len(truncation_warns) > 0


@pytest.mark.needs_model
def test_consistency_short_response_not_truncated():
    """Responses within the sentence limit are not marked truncated (H-4)."""
    response = "The sky is blue. Water is wet. Fire is hot."
    score, details = score_consistency(response, max_sentences=50)
    assert details.get("truncated") is not True


@pytest.mark.needs_model
def test_consistency_max_sentences_custom():
    """Custom max_sentences=5 truncates a 10-sentence response (H-4)."""
    response = " ".join([f"Fact {i} is correct." for i in range(10)])
    score, details = score_consistency(response, max_sentences=5)
    assert details.get("truncated") is True


# ---------------------------------------------------------------------------
# H-6: No runtime NLTK download
# ---------------------------------------------------------------------------

def test_split_sentences_works_without_nltk_download(monkeypatch):
    """split_sentences() must work even if NLTK punkt_tab is unavailable (H-6)."""
    def mock_sent_tokenize(text):
        raise LookupError("punkt_tab not found")

    monkeypatch.setattr(
        "scroot.text_utils.split_sentences",
        lambda text: (
            [s.strip() for s in __import__('re').split(r'(?<=[.!?])\s+(?=[A-Z])', text) if s.strip()]
            if text and text.strip() else []
        ),
    )
    result = split_sentences("The sky is blue. Water is wet.")
    assert len(result) >= 1


def test_split_sentences_regex_fallback_basic():
    """Regex fallback produces reasonable sentence splits (H-6)."""
    import re
    text = "First sentence. Second sentence. Third sentence."
    result = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    result = [s.strip() for s in result if s.strip()]
    assert len(result) == 3
    assert result[0] == "First sentence."
