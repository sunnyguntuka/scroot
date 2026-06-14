import pytest
from scroot.metrics.groundedness import score_groundedness

pytestmark = pytest.mark.needs_model


def test_fully_grounded():
    response = "We offer a 30-day full refund at no extra cost."
    context = ["All customers are eligible for a 30-day full refund at no extra cost."]
    score, details = score_groundedness(response, context)
    assert score >= 0.5
    assert "claims" in details


def test_completely_ungrounded():
    response = "We offer a 90-day money-back guarantee with free worldwide shipping and a lifetime warranty."
    context = ["The sky is blue on clear days."]
    score, details = score_groundedness(response, context)
    assert score < 0.8  # should be lower than a grounded response


def test_no_claims_in_response():
    # Greeting only → no claims → returns 1.0
    response = "Hi there! Thanks for your question."
    context = ["Some context."]
    score, details = score_groundedness(response, context)
    assert score == 1.0
    assert details.get("note") == "no claims detected"


def test_empty_context_list():
    # Context is provided but empty list -joined context is ""
    response = "We offer a 30-day refund."
    context = []
    score, details = score_groundedness(response, context)
    assert 0.0 <= score <= 1.0


def test_returns_tuple():
    response = "The product is available."
    context = ["Product is in stock."]
    result = score_groundedness(response, context)
    assert isinstance(result, tuple)
    assert len(result) == 2
    score, details = result
    assert 0.0 <= score <= 1.0
    assert isinstance(details, dict)


def test_details_structure():
    response = "We offer a 30-day refund. Returns are free."
    context = ["Customers get a 30-day refund. No return shipping cost."]
    score, details = score_groundedness(response, context)
    assert "claims" in details
    assert "total_claims" in details
    assert "grounded_claims" in details
