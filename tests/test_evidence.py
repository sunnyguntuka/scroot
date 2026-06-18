"""Tests for src/scroot/evidence.py (sentence-level evidence attribution).

Uses fake NLI/embedding models (no real model downloads) so this runs under
``pytest -m "not needs_model"``.
"""

from __future__ import annotations

import numpy as np

from unittest.mock import patch

from scroot.evidence import EvidenceEntry, EvidenceMap, build_evidence_map

# Raw 3-class NLI logits: [contradiction, entailment, neutral].
ENTAIL = [-1.0, 5.0, -1.0]
CONTRA = [5.0, -1.0, -1.0]
NEUTRAL = [-1.0, -1.0, 2.0]

CHUNK_PARIS = "Paris is the capital city of France."
CHUNK_EIFFEL = "The Eiffel Tower is located in Paris, France."

SENT_PARIS = "Paris is the capital of France."
SENT_EIFFEL_LONDON = "The Eiffel Tower is in London."
SENT_BANANAS = "Bananas are a great snack food."

SCORE_MAP = {
    (CHUNK_PARIS, SENT_PARIS): ENTAIL,
    (CHUNK_EIFFEL, SENT_PARIS): NEUTRAL,
    (CHUNK_PARIS, SENT_EIFFEL_LONDON): NEUTRAL,
    (CHUNK_EIFFEL, SENT_EIFFEL_LONDON): CONTRA,
    (CHUNK_PARIS, SENT_BANANAS): NEUTRAL,
    (CHUNK_EIFFEL, SENT_BANANAS): NEUTRAL,
}


class FakeNLIModel:
    def __init__(self, score_map):
        self.score_map = score_map
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return np.array([self.score_map[pair] for pair in pairs], dtype=float)


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        return np.zeros((len(texts), 4))


class KeyedEmbeddingModel:
    """Returns distinct embeddings per text, keyed by exact string match."""

    def __init__(self, vectors: dict):
        self.vectors = vectors

    def encode(self, texts, convert_to_numpy=True):
        return np.array([self.vectors[t] for t in texts], dtype=float)


def _patch_models(score_map=SCORE_MAP):
    nli = FakeNLIModel(score_map)
    return (
        patch("scroot.evidence.get_nli_model", return_value=nli),
        patch("scroot.evidence.get_embedding_model", return_value=FakeEmbeddingModel()),
        nli,
    )


class TestEvidenceEntryAndMap:
    def test_evidence_entry_defaults(self):
        entry = EvidenceEntry(
            response_sentence="x",
            best_matching_chunk="y",
            entailment_score=0.9,
            supported=True,
        )
        assert entry.contradiction_detected is False
        assert entry.no_grounding_found is False
        assert entry.chunk_source is None
        assert entry.chunk_index is None

    def test_evidence_map_to_dict(self):
        entry = EvidenceEntry(
            response_sentence="x",
            best_matching_chunk="y",
            entailment_score=0.9,
            supported=True,
            chunk_source="retrieval",
            chunk_index=0,
        )
        evidence_map = EvidenceMap(
            entries=[entry],
            supported_count=1,
            unsupported_count=0,
            contradiction_count=0,
            coverage_ratio=1.0,
            weakest_sentence=None,
        )
        d = evidence_map.to_dict()
        assert d["supported"] == 1
        assert d["unsupported"] == 0
        assert d["contradictions"] == 0
        assert d["coverage_ratio"] == 1.0
        assert d["weakest_sentence"] is None
        assert d["entries"] == [
            {
                "response_sentence": "x",
                "best_matching_chunk": "y",
                "entailment_score": 0.9,
                "supported": True,
                "contradiction_detected": False,
                "no_grounding_found": False,
                "chunk_source": "retrieval",
                "chunk_index": 0,
                "mini_iqs": None,
                "mini_dims": None,
            }
        ]


class TestBuildEvidenceMap:
    def test_empty_response_returns_empty_map(self):
        p1, p2, _ = _patch_models()
        with p1, p2:
            evidence_map = build_evidence_map("", [CHUNK_PARIS])
        assert evidence_map.entries == []
        assert evidence_map.supported_count == 0
        assert evidence_map.unsupported_count == 0
        assert evidence_map.contradiction_count == 0
        assert evidence_map.coverage_ratio == 0.0
        assert evidence_map.weakest_sentence is None

    def test_empty_context_returns_no_grounding_map(self):
        p1, p2, _ = _patch_models()
        with p1, p2:
            evidence_map = build_evidence_map(SENT_PARIS, [])
        assert len(evidence_map.entries) == 1
        entry = evidence_map.entries[0]
        assert entry.response_sentence == SENT_PARIS
        assert entry.best_matching_chunk is None
        assert entry.entailment_score is None
        assert entry.supported is False
        assert entry.no_grounding_found is True
        assert evidence_map.supported_count == 0
        assert evidence_map.unsupported_count == 1
        assert evidence_map.coverage_ratio == 0.0
        assert evidence_map.weakest_sentence == SENT_PARIS

    def test_supported_contradicted_and_ungrounded_classification(self):
        response = f"{SENT_PARIS} {SENT_EIFFEL_LONDON} {SENT_BANANAS}"
        p1, p2, nli = _patch_models()
        with p1, p2:
            evidence_map = build_evidence_map(
                response,
                [CHUNK_PARIS, CHUNK_EIFFEL],
                chunk_sources=["doc-a", "doc-b"],
            )

        assert len(evidence_map.entries) == 3
        by_sentence = {e.response_sentence: e for e in evidence_map.entries}

        supported = by_sentence[SENT_PARIS]
        assert supported.supported is True
        assert supported.contradiction_detected is False
        assert supported.no_grounding_found is False
        assert supported.best_matching_chunk == CHUNK_PARIS
        assert supported.chunk_source == "doc-a"
        assert supported.chunk_index == 0
        assert supported.entailment_score > 0.70

        contradicted = by_sentence[SENT_EIFFEL_LONDON]
        assert contradicted.supported is False
        assert contradicted.contradiction_detected is True
        assert contradicted.no_grounding_found is False
        assert contradicted.best_matching_chunk == CHUNK_EIFFEL
        assert contradicted.chunk_source == "doc-b"
        assert contradicted.chunk_index == 1

        ungrounded = by_sentence[SENT_BANANAS]
        assert ungrounded.supported is False
        assert ungrounded.contradiction_detected is False
        assert ungrounded.no_grounding_found is True

        assert evidence_map.supported_count == 1
        assert evidence_map.contradiction_count == 1
        assert evidence_map.unsupported_count == 1
        assert evidence_map.coverage_ratio == round(1 / 3, 4)
        assert evidence_map.weakest_sentence in (SENT_EIFFEL_LONDON, SENT_BANANAS)

    def test_no_embedding_model_skips_retrieval(self):
        response = SENT_PARIS
        p1, p2, _ = _patch_models()
        with p1, p2 as get_emb:
            evidence_map = build_evidence_map(
                response,
                [CHUNK_PARIS, CHUNK_EIFFEL],
                embedding_model=None,
            )
        get_emb.assert_not_called()
        assert evidence_map.entries[0].supported is True

    def test_top_k_retrieval_filters_irrelevant_chunks(self):
        irrelevant_a = "irrelevant chunk A"
        irrelevant_b = "irrelevant chunk B"
        context = [CHUNK_PARIS, CHUNK_EIFFEL, irrelevant_a, irrelevant_b]

        score_map = {
            (CHUNK_PARIS, SENT_PARIS): ENTAIL,
            (CHUNK_EIFFEL, SENT_PARIS): NEUTRAL,
            (irrelevant_b, SENT_PARIS): NEUTRAL,
            # irrelevant_a deliberately omitted: must be filtered out by
            # top-k retrieval before the NLI model is ever called with it.
        }
        nli = FakeNLIModel(score_map)

        emb_model = KeyedEmbeddingModel({
            CHUNK_PARIS: [1.0, 0.0, 0.0, 0.0],
            CHUNK_EIFFEL: [1.0, 0.1, 0.0, 0.0],
            irrelevant_a: [0.0, 0.0, 0.0, 1.0],
            irrelevant_b: [1.0, 1.0, 0.0, 0.0],
            SENT_PARIS: [1.0, 0.0, 0.0, 0.0],
        })

        with patch("scroot.evidence.get_nli_model", return_value=nli), \
             patch("scroot.evidence.get_embedding_model", return_value=emb_model):
            evidence_map = build_evidence_map(SENT_PARIS, context, top_k_chunks=3)

        queried_chunks = {pair[0] for call in nli.calls for pair in call}
        assert irrelevant_a not in queried_chunks
        assert evidence_map.entries[0].supported is True

    def test_atomic_claims_false_uses_extract_claims(self):
        response = f"{SENT_PARIS} {SENT_BANANAS}"
        p1, p2, _ = _patch_models()
        with p1, p2:
            evidence_map = build_evidence_map(
                response,
                [CHUNK_PARIS, CHUNK_EIFFEL],
                atomic_claims=False,
            )
        assert len(evidence_map.entries) == 2
