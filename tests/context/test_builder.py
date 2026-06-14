"""ContextBuilder unit tests."""

import warnings

import pytest

from scroot import ContextBuilder, ContextPayload
from scroot.exceptions import (
    ContextAssemblyWarning,
    ContextEmptyWarning,
    ContextSealedError,
    ContextTooLargeWarning,
    SecurityWarning,
)


@pytest.fixture(autouse=True)
def quiet_audit():
    from scroot import configure_audit_log
    configure_audit_log(destination="disabled")
    yield
    configure_audit_log(destination="stderr")


class TestConstruction:
    def test_auto_session_id(self):
        ctx = ContextBuilder()
        assert ctx.session_id.startswith("cb-")

    def test_custom_session_id(self):
        assert ContextBuilder(session_id="trace-42").session_id == "trace-42"

    def test_session_id_length_limit(self):
        with pytest.raises(ValueError):
            ContextBuilder(session_id="x" * 129)

    def test_pii_scrub_off_in_production_warns(self, monkeypatch):
        monkeypatch.setenv("SCROOT_ENV", "production")
        with pytest.warns(SecurityWarning):
            ContextBuilder(pii_scrub=False)

    def test_pii_scrub_off_outside_production_silent(self, monkeypatch):
        monkeypatch.delenv("SCROOT_ENV", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ContextBuilder(pii_scrub=False)


class TestAddMethods:
    def test_chaining(self):
        ctx = ContextBuilder()
        out = ctx.add_query("q").add_retrieved(["a"]).add_system_prompt("s")
        assert out is ctx

    def test_add_retrieved_str_and_list(self):
        ctx = ContextBuilder()
        ctx.add_retrieved("single chunk")
        ctx.add_retrieved(["chunk one", "chunk two"])
        assert ctx.snapshot()["total_entries"] == 3

    def test_add_tool_output_records_tool_name(self):
        ctx = ContextBuilder()
        ctx.add_tool_output("rows: 5", tool_name="sql_query")
        payload = ctx.build()
        assert payload.sources[0].metadata["tool_name"] == "sql_query"

    def test_unrecognised_type_warns_not_raises(self):
        ctx = ContextBuilder()
        with pytest.warns(ContextAssemblyWarning):
            ctx.add_retrieved([42])
        assert ctx.snapshot()["total_entries"] == 0

    def test_over_500_chunks_dropped_with_warning(self):
        ctx = ContextBuilder(max_tokens=1_000_000)
        with pytest.warns(ContextAssemblyWarning):
            ctx.add_retrieved([f"chunk {i}" for i in range(501)])
        assert ctx.snapshot()["total_entries"] == 500

    def test_giant_chunk_truncated(self):
        ctx = ContextBuilder(max_tokens=1_000_000)
        with pytest.warns(ContextAssemblyWarning):
            ctx.add_retrieved("z" * 60_000)
        payload = ctx.build()
        assert payload.sources[0].content.endswith("[TRUNCATED]")
        assert len(payload.sources[0].content) <= 50_000 + len(" [TRUNCATED]")

    def test_metadata_limits(self):
        ctx = ContextBuilder()
        with pytest.raises(ValueError):
            ctx.add_retrieved("x", metadata={f"k{i}": i for i in range(21)})
        with pytest.raises(ValueError):
            ctx.add_retrieved("x", metadata={"k": "v" * 1001})

    def test_empty_text_skipped(self):
        ctx = ContextBuilder()
        ctx.add_retrieved(["", "   ", "real"])
        assert ctx.snapshot()["total_entries"] == 1


class TestBuild:
    def test_empty_build_returns_none_with_warning(self):
        with pytest.warns(ContextEmptyWarning):
            assert ContextBuilder().build() is None

    def test_build_returns_payload(self):
        ctx = ContextBuilder()
        ctx.add_query("what is the policy?")
        ctx.add_retrieved(["Policy: 30-day refunds."])
        payload = ctx.build()
        assert isinstance(payload, ContextPayload)
        assert payload.session_id == ctx.session_id
        assert "Policy: 30-day refunds." in payload.assembled_text
        assert payload.total_tokens > 0

    def test_checksum_is_sha256_of_assembled_text(self):
        import hashlib
        ctx = ContextBuilder(dedup=False)
        ctx.add_retrieved(["alpha", "beta"])
        payload = ctx.build()
        expected = "sha256:" + hashlib.sha256(
            payload.assembled_text.encode("utf-8")
        ).hexdigest()
        assert payload.checksum == expected

    def test_build_seals_builder(self):
        ctx = ContextBuilder()
        ctx.add_retrieved(["x"])
        ctx.build()
        with pytest.raises(ContextSealedError):
            ctx.add_retrieved(["more"])
        with pytest.raises(ContextSealedError):
            ctx.add_query("q")

    def test_empty_build_also_seals(self):
        ctx = ContextBuilder()
        with pytest.warns(ContextEmptyWarning):
            ctx.build()
        with pytest.raises(ContextSealedError):
            ctx.add_retrieved(["late"])

    def test_reset_unseals_and_clears(self):
        ctx = ContextBuilder()
        ctx.add_retrieved(["x"])
        ctx.build()
        ctx.reset()
        ctx.add_retrieved(["fresh"])
        snap = ctx.snapshot()
        assert snap["sealed"] is False
        assert snap["total_entries"] == 1

    def test_max_tokens_truncation_keeps_highest_weight(self):
        # reranked (weight 1.0) must survive; lower-weight sources dropped
        ctx = ContextBuilder(max_tokens=30, dedup=False)
        ctx.add_system_prompt("system prompt words " * 10)
        ctx.add_retrieved(["retrieved words " * 10])
        ctx.add_reranked(["short reranked chunk"])
        with pytest.warns(ContextTooLargeWarning):
            payload = ctx.build()
        assert payload.was_truncated is True
        sources = [e.source for e in payload.sources]
        assert "reranker" in sources

    def test_source_priority_order(self):
        ctx = ContextBuilder(dedup=False)
        ctx.add_query("the query")
        ctx.add_system_prompt("the system prompt")
        ctx.add_retrieved(["the retrieved chunk"])
        ctx.add_reranked(["the reranked chunk"])
        ctx.add_tool_output("the tool output", tool_name="t")
        payload = ctx.build()
        weights = [e.source_weight for e in payload.sources]
        assert weights == sorted(weights, reverse=True)
        assert payload.sources[0].source == "reranker"

    def test_dedup_in_build(self):
        ctx = ContextBuilder()
        ctx.add_retrieved(["Same text here.", "Same text here."])
        payload = ctx.build()
        assert len(payload.sources) == 1


class TestSnapshot:
    def test_snapshot_counts_and_sources(self):
        ctx = ContextBuilder()
        ctx.add_query("q")
        ctx.add_retrieved(["a", "b"])
        snap = ctx.snapshot()
        assert snap["total_entries"] == 3
        assert snap["sources"] == ["query", "retrieval", "retrieval"]
        assert snap["total_tokens"] > 0
        assert snap["sealed"] is False
        assert snap["pii_scrub_enabled"] is True

    def test_snapshot_does_not_seal(self):
        ctx = ContextBuilder()
        ctx.add_retrieved(["x"])
        ctx.snapshot()
        ctx.add_retrieved(["y"])  # must not raise
        assert ctx.snapshot()["total_entries"] == 2


class TestEncryptionKey:
    def test_valid_fernet_key_accepted(self):
        Fernet = pytest.importorskip("cryptography.fernet").Fernet
        ctx = ContextBuilder(encryption_key=Fernet.generate_key())
        ctx.add_retrieved(["x"])
        assert ctx.build() is not None

    def test_invalid_key_rejected(self):
        pytest.importorskip("cryptography.fernet")
        with pytest.raises(ValueError):
            ContextBuilder(encryption_key=b"not-a-fernet-key")


class TestInputShapes:
    def test_bare_dict_chunk(self):
        ctx = ContextBuilder()
        ctx.add_retrieved({"text": "from a single dict"})
        assert ctx.snapshot()["total_entries"] == 1

    def test_bare_non_iterable_chunk_warns(self):
        ctx = ContextBuilder()
        with pytest.warns(ContextAssemblyWarning):
            ctx.add_retrieved(42)
        assert ctx.snapshot()["total_entries"] == 0


class TestScrubberFailure:
    def test_scrubber_exception_degrades_gracefully(self, monkeypatch):
        from scroot.context import builder as builder_mod

        def broken_scrub(text):
            raise RuntimeError("regex engine exploded")

        monkeypatch.setattr(builder_mod, "scrub", broken_scrub)
        ctx = ContextBuilder()
        with pytest.warns(ContextAssemblyWarning, match="scrubber failed"):
            ctx.add_retrieved(["content with john@acme.com"])
        payload = ctx.build()
        # Content passes through unscrubbed rather than crashing the pipeline
        assert "john@acme.com" in payload.assembled_text
        assert payload.pii_scrubbed is False


class TestPIIIntegration:
    def test_pii_scrubbed_by_default(self):
        ctx = ContextBuilder()
        ctx.add_retrieved(["Email john@acme.com about the refund."])
        payload = ctx.build()
        assert "[EMAIL]" in payload.assembled_text
        assert "john@acme.com" not in payload.assembled_text
        assert payload.pii_scrubbed is True
        assert payload.scrub_summary["EMAIL"] == 1

    def test_pii_scrub_disabled(self):
        ctx = ContextBuilder(pii_scrub=False)
        ctx.add_retrieved(["Email john@acme.com about the refund."])
        payload = ctx.build()
        assert "john@acme.com" in payload.assembled_text
        assert payload.pii_scrubbed is False
