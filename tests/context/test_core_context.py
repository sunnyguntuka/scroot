"""Model-free tests for Auditor.score() context handling.

Metric scorers are stubbed so the ContextPayload/str/list conversion
logic in core.py is exercised without loading any models.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import scroot.core as core_mod
from scroot import Auditor, ContextBuilder, configure_audit_log
from scroot.context.payload import ContextPayload


@pytest.fixture(autouse=True)
def stub_metrics(monkeypatch):
    configure_audit_log(destination="disabled")
    captured = {}

    def fake_groundedness(response, context, **kwargs):
        captured["context"] = context
        return 0.9, {"stub": True}

    monkeypatch.setattr(core_mod, "score_groundedness", fake_groundedness)
    monkeypatch.setattr(
        core_mod, "score_completeness", lambda *a, **k: (0.8, {})
    )
    monkeypatch.setattr(core_mod, "score_relevance", lambda *a, **k: (0.7, {}))
    monkeypatch.setattr(core_mod, "score_consistency", lambda *a, **k: (1.0, {}))
    monkeypatch.setattr(core_mod, "score_confidence", lambda *a, **k: (0.6, {}))
    yield captured
    configure_audit_log(destination="stderr")


class TestContextPayloadHandling:
    def test_payload_converted_to_chunks(self, stub_metrics):
        ctx = ContextBuilder(session_id="trace-x")
        ctx.add_retrieved(["chunk one", "chunk two"])
        payload = ctx.build()

        result = Auditor().score("q", "r", context=payload)
        assert result.groundedness == 0.9
        # Each ContextEntry became one grounding chunk
        assert stub_metrics["context"] == ["chunk one", "chunk two"]

    def test_audit_trail_recorded_in_details(self, stub_metrics):
        ctx = ContextBuilder(session_id="trace-y")
        ctx.add_retrieved(["grounding text"])
        payload = ctx.build()

        result = Auditor().score("q", "r", context=payload)
        audit = result.details["context"]
        assert audit["session_id"] == "trace-y"
        assert audit["checksum"] == payload.checksum
        assert audit["total_tokens"] == payload.total_tokens
        assert audit["was_truncated"] is False
        assert audit["pii_scrubbed"] is False
        # Raw grounding text never appears in the audit trail
        assert "grounding text" not in str(audit)

    def test_payload_with_no_sources_skips_groundedness(self, stub_metrics):
        empty = ContextPayload(
            session_id="cb-empty",
            sources=[],
            assembled_text="",
            total_tokens=0,
            was_truncated=True,
            pii_scrubbed=False,
            scrub_summary={},
            built_at=datetime.now(timezone.utc),
            checksum="sha256:none",
        )
        result = Auditor().score("q", "r", context=empty)
        assert result.groundedness is None
        assert "context" not in stub_metrics
        # Audit trail still recorded even though scoring was skipped
        assert result.details["context"]["session_id"] == "cb-empty"


class TestStringAndListContext:
    def test_plain_string_wrapped_as_single_chunk(self, stub_metrics):
        result = Auditor().score("q", "r", context="one grounding string")
        assert result.groundedness == 0.9
        assert stub_metrics["context"] == ["one grounding string"]

    def test_list_passthrough(self, stub_metrics):
        Auditor().score("q", "r", context=["a", "b"])
        assert stub_metrics["context"] == ["a", "b"]

    def test_none_skips_groundedness(self, stub_metrics):
        result = Auditor().score("q", "r", context=None)
        assert result.groundedness is None
        assert "context" not in result.details
