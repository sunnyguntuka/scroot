import pytest
from scroot.metrics.completeness import score_completeness

pytestmark = pytest.mark.needs_model


def test_complete_response():
    query = "What is your refund policy?"
    response = "We offer a 30-day full refund at no extra cost. You can return any item within 30 days of purchase."
    score, details = score_completeness(query, response)
    assert score >= 0.5
    assert "segments" in details


def test_empty_response():
    query = "What is your refund policy?"
    response = ""
    score, details = score_completeness(query, response)
    assert score == 0.0


def test_empty_query():
    score, details = score_completeness("", "Some response here.")
    assert score == 0.0


def test_returns_tuple():
    result = score_completeness("What is X?", "X is a thing.")
    assert isinstance(result, tuple)
    score, details = result
    assert 0.0 <= score <= 1.0
    assert isinstance(details, dict)


def test_details_structure():
    query = "What is your policy?"
    response = "We have a 30-day refund policy."
    score, details = score_completeness(query, response)
    assert "segments" in details
    assert "total_segments" in details
    assert "covered_segments" in details


def test_single_word_query():
    score, details = score_completeness("Refund?", "We offer a 30-day refund.")
    assert 0.0 <= score <= 1.0
