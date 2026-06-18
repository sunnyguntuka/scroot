"""Wave 0 acceptance tests: P0, F0.1, F0.2.

P0  — keep_intermediates: scores identical with/without; intermediates populated.
F0.1 — DatabaseConnector: bad identifier raises; dry_run returns SQL with no I/O.
F0.2 — Relevance sigmoid params: parity with defaults; override changes score.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# P0 — Surface cached intermediates
# ---------------------------------------------------------------------------

pytestmark_needs_model = pytest.mark.needs_model


@pytest.mark.needs_model
class TestKeepIntermediates:
    """P0: keep_intermediates=True must not change any scoring output."""

    _query = "What is the return policy?"
    _response = (
        "We offer a 30-day full refund at no extra cost. "
        "You can return any item within 30 days of purchase."
    )

    def _score(self, keep: bool):
        from scroot import Auditor
        auditor = Auditor(keep_intermediates=keep)
        return auditor.score(query=self._query, response=self._response)

    def test_parity_iqs(self):
        """IQS must be byte-for-byte identical regardless of keep_intermediates."""
        r_off = self._score(keep=False)
        r_on = self._score(keep=True)
        assert r_on.iqs == r_off.iqs

    def test_parity_all_metrics(self):
        """Every metric score must be identical."""
        r_off = self._score(keep=False)
        r_on = self._score(keep=True)
        assert r_on.completeness == r_off.completeness
        assert r_on.relevance == r_off.relevance
        assert r_on.consistency == r_off.consistency
        assert r_on.confidence == r_off.confidence

    def test_parity_flags(self):
        r_off = self._score(keep=False)
        r_on = self._score(keep=True)
        assert r_on.flags == r_off.flags

    def test_intermediates_none_by_default(self):
        r = self._score(keep=False)
        assert r.intermediates is None

    def test_intermediates_populated(self):
        import numpy as np
        r = self._score(keep=True)
        assert r.intermediates is not None
        interm = r.intermediates
        assert "query_embedding" in interm
        assert "response_embeddings" in interm
        assert "response_sentences" in interm
        assert isinstance(interm["query_embedding"], np.ndarray)
        assert isinstance(interm["response_embeddings"], np.ndarray)
        assert isinstance(interm["response_sentences"], list)

    def test_intermediates_consistency_fields(self):
        """Consistency NLI capture fields must be present when response has >1 sentence."""
        r = self._score(keep=True)
        interm = r.intermediates
        assert interm is not None
        # Multi-sentence response → _capture fields populated
        assert "consistency_sentences" in interm
        assert "consistency_pairs" in interm
        assert "consistency_raw_scores" in interm

    def test_intermediates_embedding_dim_matches(self):
        """query_embedding and response_embeddings must share the same dimension."""
        import numpy as np
        r = self._score(keep=True)
        interm = r.intermediates
        assert interm is not None
        q_emb = interm["query_embedding"]
        r_embs = interm["response_embeddings"]
        assert isinstance(q_emb, np.ndarray)
        assert isinstance(r_embs, np.ndarray)
        if r_embs.ndim == 2 and r_embs.shape[0] > 0:
            assert q_emb.shape[-1] == r_embs.shape[-1]

    def test_intermediates_sentence_count_matches_embeddings(self):
        """Number of response_sentences must equal the first dim of response_embeddings."""
        import numpy as np
        r = self._score(keep=True)
        interm = r.intermediates
        assert interm is not None
        sentences = interm["response_sentences"]
        r_embs = interm["response_embeddings"]
        assert isinstance(r_embs, np.ndarray)
        if len(sentences) > 0 and r_embs.ndim == 2:
            assert r_embs.shape[0] == len(sentences)


# ---------------------------------------------------------------------------
# F0.1 — DatabaseConnector SQL-injection hardening
# ---------------------------------------------------------------------------

pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")


class TestDatabaseConnectorHardening:
    """F0.1: identifier allowlist raises; dry_run returns SQL with no I/O."""

    def test_bad_table_name_raises(self):
        from scroot.connectors import DatabaseConnector
        with pytest.raises(ValueError, match="source_table"):
            DatabaseConnector(
                connection_string="sqlite:///:memory:",
                source_table="users; DROP TABLE x",
                column_map={"query": "query", "response": "response"},
            )

    def test_bad_column_name_raises(self):
        from scroot.connectors import DatabaseConnector
        with pytest.raises(ValueError):
            DatabaseConnector(
                connection_string="sqlite:///:memory:",
                source_table="responses",
                column_map={"query": "query'; DROP TABLE--", "response": "response"},
            )

    def test_valid_identifiers_accepted(self):
        from scroot.connectors import DatabaseConnector
        # Should not raise
        conn = DatabaseConnector(
            connection_string="sqlite:///:memory:",
            source_table="my_responses_2024",
            column_map={"query": "q_text", "response": "r_text"},
            dry_run=True,
        )
        assert conn is not None

    def test_dry_run_fetch_returns_sql_string(self):
        from scroot.connectors import DatabaseConnector
        conn = DatabaseConnector(
            connection_string="sqlite:///:memory:",
            source_table="responses",
            column_map={"query": "query", "response": "response"},
            dry_run=True,
        )
        sql = conn.fetch()
        assert isinstance(sql, str)
        assert "responses" in sql.lower()

    def test_dry_run_write_returns_sql_string(self):
        from scroot.connectors import DatabaseConnector
        from scroot.result import EntailmentResult
        conn = DatabaseConnector(
            connection_string="sqlite:///:memory:",
            source_table="responses",
            result_table="results",
            column_map={"query": "query", "response": "response"},
            dry_run=True,
        )
        result = EntailmentResult(
            groundedness=0.8,
            completeness=0.8,
            relevance=0.8,
            consistency=0.8,
            confidence=0.8,
            iqs=0.8,
        )
        sql = conn.write_result(row_id=1, result=result)
        assert isinstance(sql, str)

    def test_dry_run_no_engine_created(self):
        """dry_run=True must not instantiate a database engine."""
        from scroot.connectors import DatabaseConnector
        conn = DatabaseConnector(
            connection_string="sqlite:///this_file_must_not_be_created.db",
            source_table="responses",
            column_map={"query": "query", "response": "response"},
            dry_run=True,
        )
        # If engine were created and the file created, the test would still pass,
        # but the connector spec says dry_run performs no I/O.
        # We verify by checking there is no engine attribute set.
        assert not hasattr(conn, "_engine") or conn._engine is None


# ---------------------------------------------------------------------------
# F0.2 — Relevance sigmoid parameters
# ---------------------------------------------------------------------------

@pytest.mark.needs_model
class TestRelevanceSigmoidParams:
    """F0.2: defaults produce parity; overrides change score deterministically."""

    _query = "What is the refund policy?"
    _response = "We offer a 30-day full refund at no extra cost."

    def _score_direct(self, midpoint=0.5, steepness=10.0):
        from scroot.metrics.relevance import score_relevance
        return score_relevance(
            self._query, self._response,
            midpoint=midpoint,
            steepness=steepness,
        )

    def _score_via_auditor(self, midpoint=0.5, steepness=10.0):
        from scroot import Auditor
        auditor = Auditor(
            relevance_sigmoid_midpoint=midpoint,
            relevance_sigmoid_steepness=steepness,
        )
        result = auditor.score(query=self._query, response=self._response)
        return result.relevance, result.details.get("relevance", {})

    def test_parity_direct_defaults(self):
        """score_relevance with explicit defaults == score_relevance with no params."""
        from scroot.metrics.relevance import score_relevance
        s1, _ = score_relevance(self._query, self._response)
        s2, _ = self._score_direct(midpoint=0.5, steepness=10.0)
        assert s1 == s2

    def test_parity_auditor_defaults(self):
        """Auditor with explicit defaults produces same relevance as default Auditor."""
        from scroot import Auditor
        a_default = Auditor()
        a_explicit = Auditor(relevance_sigmoid_midpoint=0.5, relevance_sigmoid_steepness=10.0)
        r_default = a_default.score(query=self._query, response=self._response)
        r_explicit = a_explicit.score(query=self._query, response=self._response)
        assert r_default.relevance == r_explicit.relevance

    def test_higher_midpoint_lowers_score(self):
        """Raising the midpoint above raw cosine similarity should reduce relevance."""
        s_default, _ = self._score_direct(midpoint=0.5, steepness=10.0)
        s_high_mid, _ = self._score_direct(midpoint=0.95, steepness=10.0)
        assert s_high_mid < s_default

    def test_lower_steepness_compresses_range(self):
        """Lower steepness should produce a score closer to 0.5."""
        s_steep, _ = self._score_direct(midpoint=0.5, steepness=10.0)
        s_flat, _ = self._score_direct(midpoint=0.5, steepness=1.0)
        # With steepness=1, the sigmoid is very flat; score should be closer to 0.5
        assert abs(s_flat - 0.5) < abs(s_steep - 0.5)

    def test_details_include_sigmoid_params(self):
        """Details must expose sigmoid_midpoint and sigmoid_steepness."""
        _, details = self._score_direct(midpoint=0.7, steepness=5.0)
        assert details.get("sigmoid_midpoint") == 0.7
        assert details.get("sigmoid_steepness") == 5.0

    def test_auditor_threads_params_into_relevance(self):
        """Auditor must pass its sigmoid params into score_relevance."""
        from scroot import Auditor
        auditor = Auditor(relevance_sigmoid_midpoint=0.7, relevance_sigmoid_steepness=5.0)
        result = auditor.score(query=self._query, response=self._response)
        rel_details = result.details.get("relevance", {})
        assert rel_details.get("sigmoid_midpoint") == 0.7
        assert rel_details.get("sigmoid_steepness") == 5.0
