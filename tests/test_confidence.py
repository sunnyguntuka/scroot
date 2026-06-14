from scroot.metrics.confidence import score_confidence


def test_empty_response():
    score, details = score_confidence("")
    assert score == 0.5
    assert "empty response" in details.get("note", "")


def test_assertive_response():
    text = "The product is definitely in stock. It is clearly the best option. It must be ordered today."
    score, details = score_confidence(text)
    assert score > 0.5
    assert details["assertion_markers_found"] > 0


def test_hedged_response():
    text = "I think it might be available. Perhaps you could check. It possibly could work."
    score, details = score_confidence(text)
    assert score < 0.5
    assert details["hedge_markers_found"] > 0


def test_neutral_no_markers():
    # A response with no hedge or assert markers → 0.5
    text = "Blue sky green grass."
    score, details = score_confidence(text)
    assert score == 0.5


def test_short_response_dampened():
    # Very short responses get dampened toward 0.5
    text = "Yes definitely."
    score, _ = score_confidence(text)
    assert 0.4 <= score <= 0.9


def test_returns_tuple():
    result = score_confidence("The answer is clear.")
    assert isinstance(result, tuple)
    assert len(result) == 2
    score, details = result
    assert 0.0 <= score <= 1.0
    assert isinstance(details, dict)
