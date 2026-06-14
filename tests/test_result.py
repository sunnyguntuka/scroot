from scroot.evidence import EvidenceEntry, EvidenceMap
from scroot.result import EntailmentResult


def test_to_dict():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.8,
        relevance=0.85,
        consistency=1.0,
        confidence=0.7,
        iqs=0.87,
        flags=[],
    )
    d = r.to_dict()
    assert d["iqs"] == 0.87
    assert d["groundedness"] == 0.9
    assert d["flags"] == []


def test_repr_with_groundedness():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.8,
        relevance=0.85,
        consistency=1.0,
        confidence=0.7,
        iqs=0.87,
        flags=["hallucination_risk"],
    )
    rep = repr(r)
    assert "iqs=0.87" in rep
    assert "groundedness=0.90" in rep
    assert "hallucination_risk" in rep


def test_repr_without_groundedness():
    r = EntailmentResult(
        groundedness=None,
        completeness=0.8,
        relevance=0.85,
        consistency=1.0,
        confidence=0.7,
        iqs=0.82,
        flags=[],
    )
    rep = repr(r)
    assert "groundedness" not in rep
    assert "iqs=0.82" in rep


def test_to_dict_no_groundedness():
    r = EntailmentResult(
        groundedness=None,
        completeness=0.8,
        relevance=0.85,
        consistency=1.0,
        confidence=0.7,
        iqs=0.82,
    )
    d = r.to_dict()
    assert d["groundedness"] is None


def test_to_dict_includes_derived_fields():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.8,
        relevance=0.85,
        consistency=1.0,
        confidence=0.7,
        iqs=0.87,
    )
    d = r.to_dict()
    assert d["weakest_metric"] == "confidence"
    assert d["score_variance"] == r.score_variance
    assert d["iqs_explanation"] == r.iqs_explanation()


def test_weakest_metric():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.8,
        relevance=0.85,
        consistency=0.31,
        confidence=0.7,
        iqs=0.6,
    )
    assert r.weakest_metric == "consistency"


def test_weakest_metric_excludes_none_groundedness():
    r = EntailmentResult(
        groundedness=None,
        completeness=0.2,
        relevance=0.85,
        consistency=0.9,
        confidence=0.95,
        iqs=0.5,
    )
    assert r.weakest_metric == "completeness"


def test_score_variance_uniform_scores():
    r = EntailmentResult(
        groundedness=0.8,
        completeness=0.8,
        relevance=0.8,
        consistency=0.8,
        confidence=0.8,
        iqs=0.8,
    )
    assert r.score_variance == 0.0


def test_score_variance_with_outlier():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.9,
        relevance=0.9,
        consistency=0.9,
        confidence=0.1,
        iqs=0.5,
    )
    assert r.score_variance > 0.30


def test_iqs_explanation_above_threshold():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.85,
        relevance=0.9,
        consistency=0.95,
        confidence=0.8,
        iqs=0.88,
    )
    explanation = r.iqs_explanation()
    assert explanation == "IQS 0.88 - all metrics above threshold."


def test_iqs_explanation_below_threshold_names_weakest_metric():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.9,
        relevance=0.9,
        consistency=0.31,
        confidence=0.9,
        iqs=0.54,
    )
    explanation = r.iqs_explanation()
    assert "IQS 0.54" in explanation
    assert "consistency (0.31)" in explanation
    assert "contradicts itself" in explanation


def test_iqs_explanation_custom_threshold():
    r = EntailmentResult(
        groundedness=0.9,
        completeness=0.9,
        relevance=0.9,
        consistency=0.75,
        confidence=0.9,
        iqs=0.80,
    )
    # Passes the default 0.70 threshold but not a stricter 0.85 one.
    assert r.iqs_explanation(threshold=0.70) == "IQS 0.80 - all metrics above threshold."
    explanation = r.iqs_explanation(threshold=0.85)
    assert "consistency (0.75)" in explanation


def test_passes_gate_iqs_only():
    passing = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
    )
    failing = EntailmentResult(
        groundedness=0.4, completeness=0.5, relevance=0.5,
        consistency=0.5, confidence=0.5, iqs=0.5,
    )
    assert passing.passes_gate(0.70) is True
    assert failing.passes_gate(0.70) is False


def test_passes_gate_per_metric_floor():
    r = EntailmentResult(
        groundedness=0.85, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.88,
    )
    # IQS passes, but groundedness floor is not met.
    assert r.passes_gate(0.80, require_groundedness=0.95) is False
    assert r.passes_gate(0.80, require_groundedness=0.80) is True


def test_passes_gate_none_groundedness_with_floor_fails_open():
    # Spec: a groundedness floor with groundedness=None fails OPEN (the floor
    # cannot be evaluated) and emits GroundednessUnavailableWarning. The IQS
    # threshold still applies.
    import warnings

    from scroot.exceptions import GroundednessUnavailableWarning

    r = EntailmentResult(
        groundedness=None, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        passed = r.passes_gate(0.70, require_groundedness=0.95)
    assert passed is True  # fail-open
    assert any(issubclass(x.category, GroundednessUnavailableWarning) for x in w)


def test_passes_gate_iqs_floor_still_applies_when_groundedness_none():
    r = EntailmentResult(
        groundedness=None, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.55,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert r.passes_gate(0.70, require_groundedness=0.95) is False


def test_gate_reason_none_when_passing():
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
    )
    assert r.gate_reason(0.70) is None


def test_gate_reason_iqs_below_threshold():
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.31, confidence=0.9, iqs=0.54,
    )
    reason = r.gate_reason(0.70)
    assert "IQS 0.54" in reason
    assert "consistency" in reason
    assert "required >= 0.70" in reason


def test_gate_reason_per_metric_floor():
    r = EntailmentResult(
        groundedness=0.85, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.88,
    )
    reason = r.gate_reason(0.80, require_groundedness=0.95)
    assert reason == "groundedness 0.85 below required floor 0.95."


def test_gate_reason_none_groundedness_with_floor_fails_open():
    # Fail-open: the unmeasurable groundedness floor is skipped (reason is
    # None when IQS passes), with a warning.
    import warnings

    from scroot.exceptions import GroundednessUnavailableWarning

    r = EntailmentResult(
        groundedness=None, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reason = r.gate_reason(0.70, require_groundedness=0.95)
    assert reason is None
    assert any(issubclass(x.category, GroundednessUnavailableWarning) for x in w)


def test_metric_explanations_empty_when_no_flags():
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9, flags=[],
    )
    assert r.metric_explanations == {}


def test_metric_explanations_maps_each_flag():
    cases = {
        "hallucination_risk": "groundedness",
        "ungrounded": "groundedness",
        "off_topic": "relevance",
        "self_contradictory": "consistency",
        "incomplete": "completeness",
    }
    for flag, metric in cases.items():
        r = EntailmentResult(
            groundedness=0.5, completeness=0.5, relevance=0.5,
            consistency=0.5, confidence=0.5, iqs=0.5, flags=[flag],
        )
        assert metric in r.metric_explanations
        assert r.metric_explanations[metric]


def test_metric_explanations_dedupes_groundedness():
    r = EntailmentResult(
        groundedness=0.5, completeness=0.5, relevance=0.5,
        consistency=0.5, confidence=0.5, iqs=0.5,
        flags=["hallucination_risk", "ungrounded"],
    )
    assert list(r.metric_explanations.keys()) == ["groundedness"]


def test_metric_explanations_ignores_unknown_flags():
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
        flags=["some_unmapped_flag"],
    )
    assert r.metric_explanations == {}


def test_metric_explanations_in_to_dict():
    r = EntailmentResult(
        groundedness=0.5, completeness=0.5, relevance=0.5,
        consistency=0.5, confidence=0.5, iqs=0.5,
        flags=["off_topic"],
    )
    d = r.to_dict()
    assert d["metric_explanations"] == {"relevance": r.metric_explanations["relevance"]}


def _evidence_map():
    entry = EvidenceEntry(
        response_sentence="Paris is the capital of France.",
        best_matching_chunk="Paris is the capital city of France.",
        entailment_score=0.95,
        supported=True,
        chunk_source="retrieval",
        chunk_index=0,
    )
    return EvidenceMap(
        entries=[entry],
        supported_count=1,
        unsupported_count=0,
        contradiction_count=0,
        coverage_ratio=1.0,
        weakest_sentence=None,
    )


def test_to_dict_evidence_map_none_by_default():
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
    )
    assert r.evidence_map is None
    assert r.to_dict()["evidence_map"] is None


def test_to_dict_evidence_map_round_trips():
    evidence_map = _evidence_map()
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
        evidence_map=evidence_map,
    )
    assert r.to_dict()["evidence_map"] == evidence_map.to_dict()


def test_repr_includes_evidence_coverage_when_set():
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
        evidence_map=_evidence_map(),
    )
    assert "evidence_coverage=1.00" in repr(r)


def test_repr_omits_evidence_coverage_when_unset():
    r = EntailmentResult(
        groundedness=0.9, completeness=0.9, relevance=0.9,
        consistency=0.9, confidence=0.9, iqs=0.9,
    )
    assert "evidence_coverage" not in repr(r)
