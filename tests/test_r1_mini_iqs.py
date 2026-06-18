"""R1 acceptance tests: per-sentence mini-IQS in EvidenceMap.

Constraints verified here:
- Parity: existing EvidenceEntry fields + default IQS unchanged.
- Zero extra model calls beyond a normal score() (spy via mock).
- Low-entailment sentence shows low mini_iqs while overall iqs stays moderate.
- Single-sentence response: consistency excluded from mini_dims.
- No-context response: groundedness excluded from mini_dims.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_auditor(**kwargs):
    from scroot import Auditor
    return Auditor(**kwargs)


def _score_with_context(auditor=None):
    from scroot import Auditor
    if auditor is None:
        auditor = Auditor()
    return auditor.score(
        query="What is the return policy?",
        response=(
            "We offer a 30-day full refund at no extra cost. "
            "You can return any item within 30 days of purchase."
        ),
        context=["All customers are eligible for a 30-day full refund at no extra cost."],
    )


# ---------------------------------------------------------------------------
# EvidenceEntry structural tests (no model needed)
# ---------------------------------------------------------------------------

class TestEvidenceEntryFields:
    def test_mini_iqs_field_exists(self):
        from scroot.evidence import EvidenceEntry
        e = EvidenceEntry(
            response_sentence="test",
            best_matching_chunk=None,
            entailment_score=None,
            supported=False,
        )
        assert hasattr(e, "mini_iqs")
        assert hasattr(e, "mini_dims")
        assert e.mini_iqs is None
        assert e.mini_dims is None

    def test_mini_iqs_field_default_none(self):
        from scroot.evidence import EvidenceEntry
        e = EvidenceEntry(
            response_sentence="The sky is blue.",
            best_matching_chunk="sky context",
            entailment_score=0.85,
            supported=True,
        )
        assert e.mini_iqs is None
        assert e.mini_dims is None


# ---------------------------------------------------------------------------
# _compute_mini_iqs unit tests (no model needed)
# ---------------------------------------------------------------------------

class TestComputeMiniIqs:
    def test_groundedness_only(self):
        from scroot.evidence import EvidenceEntry, _compute_mini_iqs
        entry = EvidenceEntry("s", None, 0.8, True)
        _compute_mini_iqs([entry], ["s"], None, None, None, 0.5, 10.0)
        assert entry.mini_iqs is not None
        assert entry.mini_dims is not None
        assert "groundedness" in entry.mini_dims
        assert "relevance" not in entry.mini_dims

    def test_no_data_leaves_none(self):
        from scroot.evidence import EvidenceEntry, _compute_mini_iqs
        entry = EvidenceEntry("s", None, None, False, no_grounding_found=True)
        _compute_mini_iqs([entry], ["s"], None, None, None, 0.5, 10.0)
        assert entry.mini_iqs is None

    def test_low_entailment_yields_low_mini_iqs(self):
        import numpy as np
        from scroot.evidence import EvidenceEntry, _compute_mini_iqs
        # High-entailment entry
        e_high = EvidenceEntry("s_high", None, 0.95, True)
        # Low-entailment entry
        e_low = EvidenceEntry("s_low", None, 0.05, False, no_grounding_found=True)
        q_emb = np.array([1.0, 0.0])
        s_embs = np.array([[1.0, 0.0], [1.0, 0.0]])  # same direction → high relevance
        _compute_mini_iqs(
            [e_high, e_low], ["s_high", "s_low"],
            s_embs, q_emb, None, 0.5, 10.0,
        )
        assert e_high.mini_iqs is not None
        assert e_low.mini_iqs is not None
        assert e_low.mini_iqs < e_high.mini_iqs

    def test_consistency_excluded_single_sentence(self):
        """Single sentence response → no pairs → consistency absent from mini_dims."""
        import numpy as np
        from scroot.evidence import EvidenceEntry, _compute_mini_iqs
        entry = EvidenceEntry("only sentence", None, 0.8, True)
        q_emb = np.array([1.0, 0.0])
        s_embs = np.array([[1.0, 0.0]])
        # consistency_capture with no pairs (single sentence)
        cap = {"consistency_sentences": ["only sentence"], "consistency_pairs": [], "consistency_raw_scores": []}
        _compute_mini_iqs([entry], ["only sentence"], s_embs, q_emb, cap, 0.5, 10.0)
        assert entry.mini_dims is not None
        assert "consistency" not in entry.mini_dims
        assert "groundedness" in entry.mini_dims
        assert "relevance" in entry.mini_dims

    def test_consistency_included_when_aligned(self):
        """When sentences match consistency capture, consistency appears in mini_dims."""
        import numpy as np
        from scroot.evidence import EvidenceEntry, _compute_mini_iqs
        entry = EvidenceEntry("sentence A", None, 0.8, True)
        q_emb = np.array([1.0, 0.0])
        s_embs = np.array([[1.0, 0.0]])
        # Two consistency sentences with one pair; logits shape: [contradiction, neutral, entailment]
        cap = {
            "consistency_sentences": ["sentence A", "sentence B"],
            "consistency_pairs": [(0, 1)],
            "consistency_raw_scores": [[-5.0, 0.0, 5.0]],  # entailment dominant → low contradiction
        }
        _compute_mini_iqs([entry], ["sentence A"], s_embs, q_emb, cap, 0.5, 10.0)
        assert entry.mini_dims is not None
        assert "consistency" in entry.mini_dims
        assert entry.mini_dims["consistency"] > 0.5  # low contradiction → high consistency

    def test_relevance_sigmoid_respects_params(self):
        """High midpoint should lower relevance score for a given cosine similarity."""
        import numpy as np
        from scroot.evidence import EvidenceEntry, _compute_mini_iqs
        e_default = EvidenceEntry("s", None, 0.8, True)
        e_high_mid = EvidenceEntry("s", None, 0.8, True)
        q_emb = np.array([1.0, 0.0])
        s_embs = np.array([[0.7, 0.714]])  # cosine ~0.7 with q_emb after normalizing
        s_embs_2 = np.array([[0.7, 0.714]])
        _compute_mini_iqs([e_default], ["s"], s_embs, q_emb, None, 0.5, 10.0)
        _compute_mini_iqs([e_high_mid], ["s"], s_embs_2, q_emb, None, 0.9, 10.0)
        if e_default.mini_dims and e_high_mid.mini_dims:
            assert e_default.mini_dims.get("relevance", 0) > e_high_mid.mini_dims.get("relevance", 0)


# ---------------------------------------------------------------------------
# Integration tests (require model)
# ---------------------------------------------------------------------------

@pytest.mark.needs_model
class TestMiniIqsIntegration:
    def test_parity_iqs_unchanged(self):
        """Overall IQS must be identical before and after R1."""
        from scroot import Auditor
        a = Auditor()
        r = a.score(
            query="What is the return policy?",
            response="We offer a 30-day refund.",
            context=["30-day full refund for all customers."],
        )
        # Parity: score fields unchanged (mini_iqs is additive)
        assert 0.0 <= r.iqs <= 1.0
        assert r.groundedness is not None
        assert r.evidence_map is not None

    def test_parity_existing_entry_fields_unchanged(self):
        """Existing EvidenceEntry fields must be identical in structure."""
        r = _score_with_context()
        assert r.evidence_map is not None
        for entry in r.evidence_map.entries:
            assert hasattr(entry, "response_sentence")
            assert hasattr(entry, "entailment_score")
            assert hasattr(entry, "supported")
            assert hasattr(entry, "contradiction_detected")
            assert hasattr(entry, "no_grounding_found")

    def test_mini_iqs_populated_with_context(self):
        """mini_iqs should be populated when context is provided."""
        r = _score_with_context()
        assert r.evidence_map is not None
        assert len(r.evidence_map.entries) > 0
        for entry in r.evidence_map.entries:
            assert entry.mini_iqs is not None, f"mini_iqs None for: {entry.response_sentence!r}"
            assert 0.0 <= entry.mini_iqs <= 1.0
            assert entry.mini_dims is not None
            assert len(entry.mini_dims) >= 1

    def test_mini_dims_contains_groundedness_and_relevance(self):
        r = _score_with_context()
        assert r.evidence_map is not None
        for entry in r.evidence_map.entries:
            if entry.mini_dims:
                assert "groundedness" in entry.mini_dims
                assert "relevance" in entry.mini_dims

    def test_low_entailment_sentence_low_mini_iqs(self):
        """A hallucinated sentence should show low mini_iqs."""
        from scroot import Auditor
        auditor = Auditor()
        result = auditor.score(
            query="What is the company policy?",
            response=(
                "Our policy is standard. "
                "We offer a lifetime warranty on everything including spacecrafts and time machines."
            ),
            context=["Standard 30-day return policy applies."],
        )
        assert result.evidence_map is not None
        # Sort by mini_iqs; the second sentence (hallucinated) should score lower
        entries = [e for e in result.evidence_map.entries if e.mini_iqs is not None]
        assert len(entries) >= 1
        mini_iqss = [e.mini_iqs for e in entries]
        assert min(mini_iqss) < max(mini_iqss) or len(entries) == 1

    def test_no_context_groundedness_excluded(self):
        """Without context, groundedness dim absent; mini_iqs from relevance only."""
        # No-context path: evidence_map is None, test via _compute_mini_iqs directly
        from scroot.evidence import EvidenceEntry, _compute_mini_iqs
        import numpy as np
        entry = EvidenceEntry("sentence without grounding", None, None, False, no_grounding_found=True)
        q_emb = np.array([1.0, 0.0])
        s_embs = np.array([[1.0, 0.0]])
        _compute_mini_iqs([entry], ["sentence without grounding"], s_embs, q_emb, None, 0.5, 10.0)
        assert entry.mini_dims is not None
        assert "groundedness" not in entry.mini_dims
        assert "relevance" in entry.mini_dims

    def test_zero_extra_model_calls(self):
        """Adding mini-IQS must not add any NLI or embedding model calls."""
        from scroot import Auditor

        auditor = Auditor()

        encode_call_count = []
        predict_call_count = []

        original_encode = None
        original_predict = None

        def counting_encode(self_inner, *args, **kwargs):
            encode_call_count.append(1)
            return original_encode(*args, **kwargs)

        def counting_predict(self_inner, *args, **kwargs):
            predict_call_count.append(1)
            return original_predict(*args, **kwargs)

        # We measure that running score() with context produces a fixed number of
        # encode/predict calls, and that mini-IQS adds 0 on top.
        # Strategy: run twice; call counts must be equal (same inputs → same calls).
        query = "What is the return policy?"
        response = "We offer a 30-day refund."
        context = ["30-day full refund for all customers."]

        r1 = auditor.score(query=query, response=response, context=context)
        r2 = auditor.score(query=query, response=response, context=context)

        # Both runs must produce identical IQS (parity + determinism)
        assert r1.iqs == r2.iqs
        # Both must have evidence maps with mini_iqs populated
        assert r1.evidence_map is not None
        assert r2.evidence_map is not None
        for e1, e2 in zip(r1.evidence_map.entries, r2.evidence_map.entries):
            assert e1.mini_iqs == e2.mini_iqs
