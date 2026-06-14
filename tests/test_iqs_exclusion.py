"""IQS groundedness-exclusion behaviour (SCROOT_IQS_GROUNDEDNESS_EXCLUSION_SPEC).

Covers compute_iqs_detailed() None-exclusion + weight redistribution, the new
EntailmentResult transparency fields, fail-open passes_gate, and the
empty/whitespace-context handling + NoContextWarning in Auditor.score().

Auditor.score() tests mock all metric scorers so they run under
``-m "not needs_model"`` (no model downloads).
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from scroot.composite import DEFAULT_WEIGHTS, compute_iqs, compute_iqs_detailed
from scroot.exceptions import (
    GroundednessComputationError,
    GroundednessUnavailableWarning,
    NoContextWarning,
)
from scroot.result import EntailmentResult


def make_scores(**kwargs):
    """All five metrics at 0.80, overridden by kwargs."""
    base = {"groundedness": 0.80, "completeness": 0.80,
            "relevance": 0.80, "consistency": 0.80, "confidence": 0.80}
    return {**base, **kwargs}


# ─── compute_iqs_detailed: None groundedness ──────────────────────────────

class TestNoneGroundedness:
    def test_none_groundedness_excluded_from_iqs(self):
        iqs, eff = compute_iqs_detailed(make_scores(groundedness=None), DEFAULT_WEIGHTS)
        assert "groundedness" not in eff
        assert abs(sum(eff.values()) - 1.0) < 1e-6

    def test_none_groundedness_uniform_gives_same_value(self):
        iqs, _ = compute_iqs_detailed(make_scores(groundedness=None), DEFAULT_WEIGHTS)
        assert iqs == pytest.approx(0.80, abs=1e-4)

    def test_weights_redistributed_proportionally(self):
        _, eff = compute_iqs_detailed(make_scores(groundedness=None), DEFAULT_WEIGHTS)
        # completeness:relevance ratio preserved at 0.25:0.20
        assert eff["completeness"] / eff["relevance"] == pytest.approx(0.25 / 0.20, rel=1e-4)

    def test_effective_weights_sum_to_one(self):
        _, eff = compute_iqs_detailed(make_scores(groundedness=None), DEFAULT_WEIGHTS)
        assert abs(sum(eff.values()) - 1.0) < 1e-6

    def test_none_not_same_as_zero(self):
        iqs_none, _ = compute_iqs_detailed(make_scores(groundedness=None), DEFAULT_WEIGHTS)
        iqs_zero, _ = compute_iqs_detailed(make_scores(groundedness=0.0), DEFAULT_WEIGHTS)
        assert iqs_none > 0.0
        assert iqs_zero == 0.0

    def test_expected_redistributed_weights(self):
        _, eff = compute_iqs_detailed(make_scores(groundedness=None), DEFAULT_WEIGHTS)
        assert eff["completeness"] == pytest.approx(0.25 / 0.65, abs=1e-4)
        assert eff["confidence"] == pytest.approx(0.05 / 0.65, abs=1e-4)


# ─── compute_iqs_detailed: genuine zero ───────────────────────────────────

class TestZeroGroundedness:
    def test_zero_groundedness_produces_iqs_zero(self):
        iqs, eff = compute_iqs_detailed(make_scores(groundedness=0.0), DEFAULT_WEIGHTS)
        assert iqs == 0.0
        assert "groundedness" in eff  # included, not redistributed

    def test_zero_groundedness_keeps_full_weight(self):
        _, eff = compute_iqs_detailed(make_scores(groundedness=0.0), DEFAULT_WEIGHTS)
        assert eff["groundedness"] == pytest.approx(DEFAULT_WEIGHTS["groundedness"])

    def test_zero_other_metric_also_zero(self):
        iqs, _ = compute_iqs_detailed(make_scores(completeness=0.0), DEFAULT_WEIGHTS)
        assert iqs == 0.0

    def test_zero_geometric_also_collapses(self):
        iqs, _ = compute_iqs_detailed(make_scores(groundedness=0.0), DEFAULT_WEIGHTS, mode="geometric")
        assert iqs == 0.0


# ─── compute_iqs_detailed: weight redistribution edge cases ───────────────

class TestWeightRedistribution:
    def test_two_metrics_none(self):
        iqs, eff = compute_iqs_detailed(
            make_scores(groundedness=None, completeness=None), DEFAULT_WEIGHTS)
        assert "groundedness" not in eff and "completeness" not in eff
        assert len(eff) == 3
        assert abs(sum(eff.values()) - 1.0) < 1e-6

    def test_all_none_raises(self):
        scores = {k: None for k in DEFAULT_WEIGHTS}
        with pytest.raises(ValueError, match="All metrics are None"):
            compute_iqs_detailed(scores, DEFAULT_WEIGHTS)

    def test_zero_weight_metric_excluded_even_with_value(self):
        weights = {**DEFAULT_WEIGHTS, "groundedness": 0.0}
        _, eff = compute_iqs_detailed(make_scores(groundedness=0.50), weights)
        assert "groundedness" not in eff

    def test_backward_compatible_compute_iqs_matches(self):
        # The positional wrapper delegates to compute_iqs_detailed.
        iqs_positional = compute_iqs(None, 0.80, 0.80, 0.80, 0.80)
        iqs_detailed, _ = compute_iqs_detailed(make_scores(groundedness=None), DEFAULT_WEIGHTS)
        assert iqs_positional == iqs_detailed


# ─── EntailmentResult transparency fields ─────────────────────────────────

class TestResultFields:
    def test_context_used_false_when_none(self):
        r = _result(groundedness=None)
        assert r.context_used is False
        assert r.iqs_metric_count == 4
        assert "groundedness" not in r.effective_weights

    def test_context_used_true_when_scored(self):
        r = _result(groundedness=0.80)
        assert r.context_used is True
        assert r.iqs_metric_count == 5
        assert "groundedness" in r.effective_weights

    def test_effective_weights_sum_to_one(self):
        for g in (None, 0.80, 0.0):
            r = _result(groundedness=g)
            assert abs(sum(r.effective_weights.values()) - 1.0) < 1e-6

    def test_to_dict_includes_new_fields(self):
        d = _result(groundedness=None).to_dict()
        assert d["context_used"] is False
        assert d["iqs_metric_count"] == 4
        assert "groundedness" not in d["effective_weights"]


# ─── passes_gate fail-open ────────────────────────────────────────────────

class TestPassesGateNoneGroundedness:
    def test_require_groundedness_floor_fails_open_when_none(self):
        r = _result(groundedness=None, iqs=0.85)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            passed = r.passes_gate(threshold=0.70, require_groundedness=0.90)
        assert passed is True
        assert any(issubclass(x.category, GroundednessUnavailableWarning) for x in w)

    def test_floor_enforced_when_value_present(self):
        r = _result(groundedness=0.60, iqs=0.80)
        assert r.passes_gate(require_groundedness=0.90) is False

    def test_iqs_floor_still_applies(self):
        r = _result(groundedness=None, iqs=0.55)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert r.passes_gate(threshold=0.70) is False


# ─── Auditor.score: empty-context handling + NoContextWarning ─────────────

def _mock_metrics():
    """Patch the four always-on metric scorers (groundedness handled separately)."""
    return [
        patch("scroot.core.score_completeness", return_value=(0.8, {})),
        patch("scroot.core.score_relevance", return_value=(0.8, {})),
        patch("scroot.core.score_consistency", return_value=(0.8, {})),
        patch("scroot.core.score_confidence", return_value=(0.8, {})),
    ]


def _score(context, **kw):
    from scroot.core import Auditor
    auditor = Auditor()
    patches = _mock_metrics()
    patches.append(patch("scroot.core.score_groundedness", return_value=(0.8, {})))
    for p in patches:
        p.start()
    try:
        return auditor.score(query="q", response="r", context=context, **kw) \
            if kw else auditor.score(query="q", response="r", context=context)
    finally:
        for p in patches:
            p.stop()


class TestNoContextWarning:
    def test_no_context_emits_warning_and_none_groundedness(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _score(None)
        assert result.groundedness is None
        assert result.context_used is False
        assert any(issubclass(x.category, NoContextWarning) for x in w)

    def test_empty_string_context_treated_as_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _score("")
        assert result.groundedness is None
        assert any(issubclass(x.category, NoContextWarning) for x in w)

    def test_whitespace_context_treated_as_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _score("   \n\t  ")
        assert result.groundedness is None
        assert any(issubclass(x.category, NoContextWarning) for x in w)

    def test_empty_list_context_treated_as_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _score([])
        assert result.groundedness is None
        assert any(issubclass(x.category, NoContextWarning) for x in w)

    def test_real_context_scores_groundedness_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _score(["Paris is the capital of France."])
        assert result.groundedness == 0.8
        assert result.context_used is True
        assert not any(issubclass(x.category, NoContextWarning) for x in w)

    def test_no_warning_when_groundedness_weight_zero(self):
        from scroot.core import Auditor
        weights = {**DEFAULT_WEIGHTS, "groundedness": 0.0}
        auditor = Auditor(weights=weights)
        patches = _mock_metrics()
        for p in patches:
            p.start()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                auditor.score(query="q", response="r")
            assert not any(issubclass(x.category, NoContextWarning) for x in w)
        finally:
            for p in patches:
                p.stop()


class TestGroundednessComputationError:
    def test_nli_error_degrades_gracefully(self):
        from scroot.core import Auditor
        auditor = Auditor()
        patches = _mock_metrics()
        patches.append(patch(
            "scroot.core.score_groundedness",
            side_effect=RuntimeError("model exploded"),
        ))
        # evidence map also calls models; disable it for this unit test
        auditor.compute_evidence_map = False
        for p in patches:
            p.start()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = auditor.score(query="q", response="r", context=["some ctx"])
            assert result.groundedness is None
            assert result.context_used is False
            assert any(issubclass(x.category, GroundednessComputationError) for x in w)
        finally:
            for p in patches:
                p.stop()


def _result(groundedness, iqs=0.80, completeness=0.80,
            relevance=0.80, consistency=0.80, confidence=0.80):
    """Build an EntailmentResult; transparency fields auto-derive in __post_init__."""
    return EntailmentResult(
        groundedness=groundedness,
        completeness=completeness,
        relevance=relevance,
        consistency=consistency,
        confidence=confidence,
        iqs=iqs,
    )
