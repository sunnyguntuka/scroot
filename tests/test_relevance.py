import pytest
from scroot.metrics.relevance import score_relevance

pytestmark = pytest.mark.needs_model


def test_relevant_response():
    query = "What is the refund policy?"
    response = "We offer a 30-day full refund at no extra cost."
    score, details = score_relevance(query, response)
    assert score > 0.3
    assert "raw_cosine_similarity" in details


def test_off_topic_response():
    query = "What is the refund policy?"
    response = "The weather in San Francisco is typically foggy in summer."
    score, details = score_relevance(query, response)
    # Should be lower than a relevant response
    relevant_score, _ = score_relevance(query, "We offer a 30-day refund.")
    assert score < relevant_score


def test_empty_query():
    score, details = score_relevance("", "Some response.")
    assert score == 0.0


def test_empty_response():
    score, details = score_relevance("What is X?", "")
    assert score == 0.0


def test_returns_tuple():
    result = score_relevance("What is X?", "X is a concept.")
    assert isinstance(result, tuple)
    score, details = result
    assert 0.0 <= score <= 1.0
    assert isinstance(details, dict)


def test_score_in_range():
    score, _ = score_relevance("Explain quantum computing", "Quantum computing uses qubits in superposition.")
    assert 0.0 <= score <= 1.0
