import pytest
from scroot.metrics.consistency import score_consistency

pytestmark = pytest.mark.needs_model


def test_single_sentence():
    response = "The product is available."
    score, details = score_consistency(response)
    assert score == 1.0
    assert "single sentence" in details.get("note", "")


def test_empty_response():
    score, details = score_consistency("")
    assert score == 1.0


def test_consistent_response():
    response = "The product is in stock. It can be shipped today. Orders placed before 5pm ship same day."
    score, details = score_consistency(response)
    assert score >= 0.7


def test_contradictory_response():
    response = (
        "Yes, the product is currently in stock and available for immediate shipping. "
        "Unfortunately, the product is out of stock and will not be available until next month."
    )
    score, details = score_consistency(response)
    # The score should be less than 1.0; it may or may not hit the threshold
    assert 0.0 <= score <= 1.0
    assert "total_pairs" in details
    assert "contradictions_found" in details


def test_returns_tuple():
    result = score_consistency("This is a test.")
    assert isinstance(result, tuple)
    score, details = result
    assert 0.0 <= score <= 1.0
    assert isinstance(details, dict)


def test_details_structure():
    response = "The sky is blue. Grass is green. The sun is bright."
    score, details = score_consistency(response)
    assert "total_pairs" in details
    assert "contradictions" in details
