from scroot.flags import detect_flags


def test_hallucination_risk():
    flags = detect_flags(
        groundedness=0.2,
        completeness=0.8,
        relevance=0.8,
        consistency=0.9,
        confidence=0.9,
    )
    assert "hallucination_risk" in flags


def test_no_hallucination_when_grounded():
    flags = detect_flags(
        groundedness=0.9,
        completeness=0.8,
        relevance=0.8,
        consistency=0.9,
        confidence=0.9,
    )
    assert "hallucination_risk" not in flags


def test_off_topic():
    flags = detect_flags(
        groundedness=0.8,
        completeness=0.8,
        relevance=0.1,
        consistency=0.9,
        confidence=0.5,
    )
    assert "off_topic" in flags


def test_self_contradictory():
    flags = detect_flags(
        groundedness=0.8,
        completeness=0.8,
        relevance=0.8,
        consistency=0.3,
        confidence=0.5,
    )
    assert "self_contradictory" in flags


def test_incomplete():
    flags = detect_flags(
        groundedness=0.8,
        completeness=0.1,
        relevance=0.8,
        consistency=0.9,
        confidence=0.5,
    )
    assert "incomplete" in flags


def test_ungrounded():
    flags = detect_flags(
        groundedness=0.1,
        completeness=0.8,
        relevance=0.8,
        consistency=0.9,
        confidence=0.3,
    )
    assert "ungrounded" in flags


def test_no_flags_clean_response():
    flags = detect_flags(
        groundedness=0.9,
        completeness=0.9,
        relevance=0.9,
        consistency=0.9,
        confidence=0.7,
    )
    assert flags == []


def test_no_context_no_hallucination_flag():
    # groundedness=None means no context -hallucination_risk should not fire
    flags = detect_flags(
        groundedness=None,
        completeness=0.8,
        relevance=0.8,
        consistency=0.9,
        confidence=0.95,
    )
    assert "hallucination_risk" not in flags
    assert "ungrounded" not in flags
