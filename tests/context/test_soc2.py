"""SOC II compliance tests.

Verifies the architectural guarantees: audit log is content-free, the
builder seals after build(), and PII never reaches the ContextPayload.
"""

import json

import pytest

from scroot import ContextBuilder, configure_audit_log
from scroot.exceptions import ContextSealedError

SENSITIVE_CHUNK = (
    "Customer john@acme.com (SSN 123-45-6789) called from 192.168.1.1 "
    "about order 42. API key sk-abcdefghij1234567890abcd was leaked."
)


@pytest.fixture
def audit_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    configure_audit_log(destination="file", path=str(path))
    yield path
    configure_audit_log(destination="disabled")


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestAuditLogContentFree:
    def test_entry_added_event_has_no_content(self, audit_file):
        ctx = ContextBuilder()
        ctx.add_retrieved([SENSITIVE_CHUNK])
        events = read_events(audit_file)
        added = [e for e in events if e["event"] == "context_entry_added"]
        assert len(added) == 1
        raw = json.dumps(added[0])
        assert "john@acme.com" not in raw
        assert "123-45-6789" not in raw
        assert "sk-abcdefghij" not in raw
        assert "order 42" not in raw
        # But the metadata trail is present
        assert added[0]["source"] == "retrieval"
        assert added[0]["pii_scrubbed"] is True
        assert added[0]["scrub_summary"]["EMAIL"] == 1

    def test_built_event_has_no_content(self, audit_file):
        ctx = ContextBuilder()
        ctx.add_retrieved([SENSITIVE_CHUNK])
        payload = ctx.build()
        events = read_events(audit_file)
        built = [e for e in events if e["event"] == "context_built"]
        assert len(built) == 1
        raw = json.dumps(built[0])
        assert "john@acme.com" not in raw
        assert "Customer" not in raw
        assert built[0]["checksum"] == payload.checksum
        assert built[0]["sources_used"] == ["retrieval"]

    def test_audit_events_carry_session_id_and_version(self, audit_file):
        ctx = ContextBuilder(session_id="soc2-session")
        ctx.add_retrieved(["some grounding text"])
        ctx.build()
        for event in read_events(audit_file):
            assert event["session_id"] == "soc2-session"
            assert "scroot_version" in event
            assert "timestamp" in event


class TestPayloadHygiene:
    def test_pii_not_in_payload(self):
        configure_audit_log(destination="disabled")
        ctx = ContextBuilder()
        ctx.add_retrieved([SENSITIVE_CHUNK])
        payload = ctx.build()
        assert "john@acme.com" not in payload.assembled_text
        assert "123-45-6789" not in payload.assembled_text
        assert "sk-abcdefghij1234567890abcd" not in payload.assembled_text
        for entry in payload.sources:
            assert "john@acme.com" not in entry.content

    def test_scrub_summary_counts_only(self):
        configure_audit_log(destination="disabled")
        ctx = ContextBuilder()
        ctx.add_retrieved([SENSITIVE_CHUNK])
        payload = ctx.build()
        for key, value in payload.scrub_summary.items():
            assert isinstance(key, str)
            assert isinstance(value, int)


class TestSealing:
    def test_sealed_after_build(self):
        configure_audit_log(destination="disabled")
        ctx = ContextBuilder()
        ctx.add_retrieved(["x"])
        ctx.build()
        for call in (
            lambda: ctx.add_query("q"),
            lambda: ctx.add_retrieved(["r"]),
            lambda: ctx.add_reranked(["rr"]),
            lambda: ctx.add_system_prompt("s"),
            lambda: ctx.add_tool_output("t", tool_name="tool"),
        ):
            with pytest.raises(ContextSealedError):
                call()


class TestAuditConfig:
    def test_invalid_destination_rejected(self):
        with pytest.raises(ValueError):
            configure_audit_log(destination="webhook")

    def test_file_requires_path(self):
        with pytest.raises(ValueError):
            configure_audit_log(destination="file")

    def test_retention_rotation(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        old = {"event": "old", "timestamp": "2020-01-01T00:00:00+00:00"}
        fresh = {"event": "fresh", "timestamp": "2099-01-01T00:00:00+00:00"}
        path.write_text(
            json.dumps(old) + "\n" + json.dumps(fresh) + "\n", encoding="utf-8"
        )
        configure_audit_log(destination="file", path=str(path), retention_days=90)
        events = read_events(path)
        configure_audit_log(destination="disabled")
        assert [e["event"] for e in events] == ["fresh"]
