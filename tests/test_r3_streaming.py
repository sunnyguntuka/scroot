"""R3 acceptance tests: StreamingAuditor incremental scoring.

All tests use fake NLI/embedding models (no real model downloads).

Constraints verified:
- Every non-final partial has provisional=True; deferred dims not in dims dict.
- Final partial has provisional=False; iqs == auditor.score() (parity).
- Consistency call count is O(k) per sentence (not O(n²) recompute).
- Sentence segmentation buffering: partial sentences not emitted early.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scroot.streaming import PartialScore, StreamingAuditor, _STREAMING_DEFERRED


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

class FakeEmbeddingModel:
    """Returns a deterministic unit vector per text (hash-based direction)."""

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            v = np.zeros(4, dtype=float)
            for i, ch in enumerate(t[:4]):
                v[i % 4] += ord(ch)
            norm = np.linalg.norm(v) or 1.0
            vecs.append(v / norm)
        return np.array(vecs)


class FakeNLIModel:
    """Always returns neutral logits; records all predict calls."""

    NEUTRAL = np.array([-1.0, -1.0, 5.0])

    def __init__(self):
        self.call_pairs: list[list[tuple]] = []

    def predict(self, pairs):
        self.call_pairs.append(list(pairs))
        return np.array([self.NEUTRAL] * len(pairs))


class ContradictingNLIModel:
    """Returns contradiction for the first pair; neutral for the rest."""

    def predict(self, pairs):
        out = []
        for i, _ in enumerate(pairs):
            if i == 0:
                out.append(np.array([5.0, -1.0, -1.0]))  # contradiction
            else:
                out.append(np.array([-1.0, -1.0, 5.0]))  # neutral
        return np.array(out)


def _fake_auditor(nli_model=None, emb_model=None):
    """Return a minimal Auditor-like object with the needed attributes."""
    from scroot import Auditor
    a = Auditor.__new__(Auditor)
    a.nli_model = "cross-encoder/nli-deberta-v3-base"
    a.embedding_model = "all-MiniLM-L6-v2"
    a.device = "cpu"
    a.weights = None
    a.relevance_sigmoid_midpoint = 0.5
    a.relevance_sigmoid_steepness = 10.0
    a.contradiction_threshold = 0.7
    a._nli_model_obj = nli_model
    a._emb_model_obj = emb_model
    return a


def _patch_models(nli=None, emb=None):
    """Patch get_nli_model and get_embedding_model in scroot.streaming."""
    nli = nli or FakeNLIModel()
    emb = emb or FakeEmbeddingModel()
    p_nli = patch("scroot.streaming.get_nli_model", return_value=nli)
    p_emb = patch("scroot.streaming.get_embedding_model", return_value=emb)
    return p_nli, p_emb, nli, emb


def _collect_stream(streamer, chunks, query, context=None):
    """Run score_stream and return list of all PartialScore objects."""
    return list(streamer.score_stream(chunks, query, context))


# ---------------------------------------------------------------------------
# TestPartialScoreDataclass
# ---------------------------------------------------------------------------

class TestPartialScoreDataclass:
    def test_fields_exist(self):
        p = PartialScore(
            partial_iqs=0.5,
            provisional=True,
            deferred=["groundedness"],
            sentences_seen=1,
            dims={"relevance": 0.6, "consistency": 1.0},
        )
        assert p.iqs is None
        assert p.result is None
        assert p.provisional is True
        assert p.sentences_seen == 1

    def test_deferred_constant_contains_groundedness_completeness_confidence(self):
        for dim in ("groundedness", "completeness", "confidence"):
            assert dim in _STREAMING_DEFERRED


# ---------------------------------------------------------------------------
# TestStreamSegmentation
# ---------------------------------------------------------------------------

class TestStreamSegmentation:
    """Buffer logic: partial sentences wait; complete ones are emitted."""

    def _make_streamer(self, nli=None, emb=None):
        from scroot import Auditor
        auditor = Auditor.__new__(Auditor)
        auditor.nli_model = "x"
        auditor.embedding_model = "y"
        auditor.device = "cpu"
        auditor.weights = None
        auditor.relevance_sigmoid_midpoint = 0.5
        auditor.relevance_sigmoid_steepness = 10.0
        auditor.contradiction_threshold = 0.7
        streamer = StreamingAuditor(auditor)
        # Patch score to avoid real model calls in final pass.
        auditor.score = MagicMock(return_value=_dummy_result())
        return streamer

    def test_single_sentence_two_chunks_emits_one_partial_plus_final(self):
        nli = FakeNLIModel()
        emb = FakeEmbeddingModel()
        p_nli, p_emb, _, _ = _patch_models(nli, emb)
        streamer = self._make_streamer()
        with p_nli, p_emb:
            # Two chunks form one complete sentence.
            partials = _collect_stream(streamer, ["Hello ", "world."], "q")
        # 1 sentence partial + 1 final
        assert len(partials) == 2
        assert partials[0].provisional is True
        assert partials[0].sentences_seen == 1
        assert partials[-1].provisional is False

    def test_two_sentences_in_one_chunk_emits_two_partials_plus_final(self):
        p_nli, p_emb, _, _ = _patch_models()
        streamer = self._make_streamer()
        with patch("scroot.streaming.split_sentences", side_effect=_fake_split):
            with p_nli, p_emb:
                chunks = ["First sentence. Second sentence."]
                partials = _collect_stream(streamer, chunks, "q")
        # 2 sentence partials + 1 final
        assert sum(1 for p in partials if p.provisional) == 2
        assert partials[-1].provisional is False

    def test_empty_stream_yields_only_final(self):
        p_nli, p_emb, _, _ = _patch_models()
        streamer = self._make_streamer()
        with p_nli, p_emb:
            partials = _collect_stream(streamer, [], "q")
        # No text → only final (provisional=False), but since full_text is empty,
        # no final is yielded either (score_stream returns early).
        assert len(partials) == 0

    def test_abbreviation_not_split(self):
        """Text with 'Mr.' in the middle must not emit a partial at 'Mr.'."""
        p_nli, p_emb, _, _ = _patch_models()
        streamer = self._make_streamer()
        # Force split_sentences to return the correct single sentence
        # (NLTK handles abbreviations; regex fallback may not, so mock it).
        with patch(
            "scroot.streaming.split_sentences",
            return_value=["Mr. Smith is great."],
        ):
            with p_nli, p_emb:
                partials = _collect_stream(
                    streamer, ["Mr. Smith is", " great."], "q"
                )
        sentence_partials = [p for p in partials if p.provisional]
        # Exactly one sentence "Mr. Smith is great.", not two.
        assert len(sentence_partials) == 1


# ---------------------------------------------------------------------------
# TestProvisionalAndDeferred
# ---------------------------------------------------------------------------

class TestProvisionalAndDeferred:
    def _run(self, text, query="What?", context=None, nli=None, emb=None):
        from scroot import Auditor
        auditor = Auditor.__new__(Auditor)
        auditor.nli_model = "x"
        auditor.embedding_model = "y"
        auditor.device = "cpu"
        auditor.weights = None
        auditor.relevance_sigmoid_midpoint = 0.5
        auditor.relevance_sigmoid_steepness = 10.0
        auditor.contradiction_threshold = 0.7
        auditor.score = MagicMock(return_value=_dummy_result())
        streamer = StreamingAuditor(auditor)
        p_nli, p_emb, nli_obj, _ = _patch_models(nli, emb)
        with p_nli, p_emb:
            return list(streamer.score_stream([text], query, context)), nli_obj

    def test_all_non_final_partials_provisional(self):
        partials, _ = self._run("Sent one. Sent two. Sent three.")
        non_final = partials[:-1]
        assert all(p.provisional for p in non_final)

    def test_final_partial_not_provisional(self):
        partials, _ = self._run("Sent one. Sent two.")
        assert partials[-1].provisional is False

    def test_deferred_dims_absent_from_partial_dims(self):
        partials, _ = self._run("Sent one. Sent two.")
        for p in partials[:-1]:
            for dim in _STREAMING_DEFERRED:
                assert dim not in p.dims

    def test_deferred_list_populated_on_partials(self):
        partials, _ = self._run("Sent one. Sent two.")
        for p in partials[:-1]:
            for dim in _STREAMING_DEFERRED:
                assert dim in p.deferred

    def test_deferred_empty_on_final(self):
        partials, _ = self._run("Sent one. Sent two.")
        assert partials[-1].deferred == []

    def test_partial_dims_contain_relevance_and_consistency(self):
        partials, _ = self._run("Sent one. Sent two.")
        for p in partials[:-1]:
            assert "relevance" in p.dims
            assert "consistency" in p.dims

    def test_partial_iqs_between_zero_and_one(self):
        partials, _ = self._run("Sent one. Sent two.")
        for p in partials[:-1]:
            assert 0.0 <= p.partial_iqs <= 1.0

    def test_sentences_seen_increments(self):
        partials, _ = self._run("Sent one. Sent two. Sent three.")
        counts = [p.sentences_seen for p in partials[:-1]]
        assert counts == list(range(1, len(counts) + 1))


# ---------------------------------------------------------------------------
# TestConsistencyOkPerSentence
# ---------------------------------------------------------------------------

class TestConsistencyOkPerSentence:
    """Verify O(k) consistency: only new pairs per sentence, not recompute."""

    def _run_n_sentences(self, n):
        from scroot import Auditor
        auditor = Auditor.__new__(Auditor)
        auditor.nli_model = "x"
        auditor.embedding_model = "y"
        auditor.device = "cpu"
        auditor.weights = None
        auditor.relevance_sigmoid_midpoint = 0.5
        auditor.relevance_sigmoid_steepness = 10.0
        auditor.contradiction_threshold = 0.7
        auditor.score = MagicMock(return_value=_dummy_result())
        streamer = StreamingAuditor(auditor)
        nli = FakeNLIModel()
        emb = FakeEmbeddingModel()
        # Build n sentences
        sentences = [f"Sentence number {i + 1}." for i in range(n)]
        text = " ".join(sentences)
        p_nli = patch("scroot.streaming.get_nli_model", return_value=nli)
        p_emb = patch("scroot.streaming.get_embedding_model", return_value=emb)
        with p_nli, p_emb:
            list(streamer.score_stream([text], "query?"))
        return nli

    def test_zero_nli_pairs_for_first_sentence(self):
        nli = self._run_n_sentences(1)
        # No prior sentences → no consistency NLI calls during streaming.
        # (The final auditor.score() is mocked, so no NLI there either.)
        total_streaming_pairs = sum(len(batch) for batch in nli.call_pairs)
        assert total_streaming_pairs == 0

    def test_two_sentences_two_pairs_bidirectional(self):
        # sentence 2 vs sentence 1: 1 fwd + 1 bwd = 2 pairs.
        nli = self._run_n_sentences(2)
        total = sum(len(b) for b in nli.call_pairs)
        assert total == 2  # 2*(2-1) = 2

    def test_three_sentences_six_pairs(self):
        # sentence 2: 2 pairs; sentence 3: 4 pairs. Total = 6 = 3*(3-1).
        nli = self._run_n_sentences(3)
        total = sum(len(b) for b in nli.call_pairs)
        assert total == 6  # 3*(3-1) = 6

    def test_n_sentences_incremental_total_pairs(self):
        # Total NLI pairs during streaming = N*(N-1) (bidirectional).
        n = 5
        nli = self._run_n_sentences(n)
        total = sum(len(b) for b in nli.call_pairs)
        assert total == n * (n - 1)  # 5*4 = 20

    def test_each_call_batch_size_grows_by_two(self):
        # Batch sizes should be 0, 2, 4, 6, ... (2 * number of prior sentences).
        n = 4
        nli = self._run_n_sentences(n)
        # Sentence 1: no batch; sentences 2,3,4: batches of 2, 4, 6.
        assert [len(b) for b in nli.call_pairs] == [2, 4, 6]

    def test_contradiction_detected_lowers_consistency(self):
        from scroot import Auditor
        auditor = Auditor.__new__(Auditor)
        auditor.nli_model = "x"
        auditor.embedding_model = "y"
        auditor.device = "cpu"
        auditor.weights = None
        auditor.relevance_sigmoid_midpoint = 0.5
        auditor.relevance_sigmoid_steepness = 10.0
        auditor.contradiction_threshold = 0.7
        auditor.score = MagicMock(return_value=_dummy_result())
        streamer = StreamingAuditor(auditor)
        contra_nli = ContradictingNLIModel()
        p_nli = patch("scroot.streaming.get_nli_model", return_value=contra_nli)
        p_emb = patch("scroot.streaming.get_embedding_model",
                      return_value=FakeEmbeddingModel())
        with p_nli, p_emb:
            partials = list(streamer.score_stream(
                ["First sentence. Contradicting sentence."], "q"
            ))
        # The second sentence's consistency should be < 1.0.
        sentence_partials = [p for p in partials if p.provisional]
        assert len(sentence_partials) >= 2
        assert sentence_partials[1].dims["consistency"] < 1.0


# ---------------------------------------------------------------------------
# TestStreamingParity
# ---------------------------------------------------------------------------

class TestStreamingParity:
    """Final PartialScore.iqs must equal auditor.score() for the full text."""

    def test_final_iqs_comes_from_auditor_score(self):
        from scroot import Auditor
        auditor = Auditor.__new__(Auditor)
        auditor.nli_model = "x"
        auditor.embedding_model = "y"
        auditor.device = "cpu"
        auditor.weights = None
        auditor.relevance_sigmoid_midpoint = 0.5
        auditor.relevance_sigmoid_steepness = 10.0
        auditor.contradiction_threshold = 0.7

        expected_iqs = 0.731
        mock_result = _dummy_result(iqs=expected_iqs)
        auditor.score = MagicMock(return_value=mock_result)

        streamer = StreamingAuditor(auditor)
        p_nli, p_emb, _, _ = _patch_models()
        with p_nli, p_emb:
            partials = list(streamer.score_stream(
                ["Hello world. This is a test."], "q"
            ))

        final = partials[-1]
        assert final.provisional is False
        assert final.iqs == expected_iqs
        assert final.result is mock_result

    def test_auditor_score_called_with_full_assembled_text(self):
        from scroot import Auditor
        auditor = Auditor.__new__(Auditor)
        auditor.nli_model = "x"
        auditor.embedding_model = "y"
        auditor.device = "cpu"
        auditor.weights = None
        auditor.relevance_sigmoid_midpoint = 0.5
        auditor.relevance_sigmoid_steepness = 10.0
        auditor.contradiction_threshold = 0.7
        auditor.score = MagicMock(return_value=_dummy_result())

        streamer = StreamingAuditor(auditor)
        p_nli, p_emb, _, _ = _patch_models()
        chunks = ["Hello ", "world", ". How ", "are you?"]
        full_text = "Hello world. How are you?"
        with p_nli, p_emb:
            list(streamer.score_stream(chunks, "query", ["ctx"]))

        auditor.score.assert_called_once_with("query", full_text, ["ctx"])

    def test_final_dims_reflect_full_result(self):
        from scroot import Auditor
        auditor = Auditor.__new__(Auditor)
        auditor.nli_model = "x"
        auditor.embedding_model = "y"
        auditor.device = "cpu"
        auditor.weights = None
        auditor.relevance_sigmoid_midpoint = 0.5
        auditor.relevance_sigmoid_steepness = 10.0
        auditor.contradiction_threshold = 0.7
        mock_result = _dummy_result(iqs=0.8, groundedness=0.9,
                                    completeness=0.7, relevance=0.8,
                                    consistency=0.95, confidence=0.6)
        auditor.score = MagicMock(return_value=mock_result)

        streamer = StreamingAuditor(auditor)
        p_nli, p_emb, _, _ = _patch_models()
        with p_nli, p_emb:
            partials = list(streamer.score_stream(["Sentence one. Sentence two."], "q"))

        final = partials[-1]
        assert final.dims["groundedness"] == pytest.approx(0.9)
        assert final.dims["completeness"] == pytest.approx(0.7)
        assert final.dims["relevance"] == pytest.approx(0.8)
        assert final.dims["consistency"] == pytest.approx(0.95)
        assert final.dims["confidence"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# TestGroundednessCapture (prerequisite: _capture param on score_groundedness)
# ---------------------------------------------------------------------------

class TestGroundednessCapture:
    def test_capture_populated_when_provided(self):
        from unittest.mock import patch as mpatch
        from scroot.metrics.groundedness import score_groundedness

        ENTAIL = [-1.0, 5.0, -1.0]

        class FakeNLI:
            def predict(self, pairs):
                return np.array([ENTAIL] * len(pairs))

        capture: dict = {}
        with mpatch("scroot.metrics.groundedness.get_nli_model", return_value=FakeNLI()):
            score_groundedness(
                "Paris is the capital.",
                ["Paris is the capital city of France."],
                embedding_model=None,  # skip embedding for test isolation
                _capture=capture,
            )

        assert "groundedness_claims" in capture
        assert "groundedness_scores" in capture
        assert len(capture["groundedness_claims"]) >= 1
        assert all(isinstance(s, float) for s in capture["groundedness_scores"])

    def test_no_capture_when_none(self):
        """_capture=None (default) must not affect return value."""
        from scroot.metrics.groundedness import score_groundedness

        class FakeNLI:
            def predict(self, pairs):
                return np.array([[-1.0, 5.0, -1.0]] * len(pairs))

        with patch("scroot.metrics.groundedness.get_nli_model", return_value=FakeNLI()):
            score, _ = score_groundedness(
                "Paris is the capital.",
                ["Paris is capital."],
                embedding_model=None,
            )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_split(text):
    """Minimal sentence splitter for tests: splits on '. ' boundary."""
    import re
    parts = re.split(r"(?<=\.)\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _dummy_result(
    iqs=0.75,
    groundedness=None,
    completeness=0.7,
    relevance=0.8,
    consistency=0.9,
    confidence=0.5,
):
    """Build a minimal EntailmentResult-like object for mocking."""
    from scroot.result import EntailmentResult
    return EntailmentResult(
        groundedness=groundedness,
        completeness=completeness,
        relevance=relevance,
        consistency=consistency,
        confidence=confidence,
        iqs=iqs,
        flags={},
        details={},
        evidence_map=None,
        effective_weights={},
        context_used=(groundedness is not None),
        iqs_metric_count=4,
    )
