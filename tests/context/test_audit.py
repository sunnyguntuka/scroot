"""Audit logger edge cases - stderr destination, rotation, bad input."""

from __future__ import annotations

import json

import pytest

from scroot import audit
from scroot.audit import configure_audit_log, emit


@pytest.fixture(autouse=True)
def restore_config():
    yield
    configure_audit_log(destination="disabled")


class TestStderrDestination:
    def test_emit_writes_json_line_to_stderr(self, capsys):
        configure_audit_log(destination="stderr")
        emit("test_event", session_id="s1", count=3)
        err = capsys.readouterr().err
        event = json.loads(err.strip().splitlines()[-1])
        assert event["event"] == "test_event"
        assert event["session_id"] == "s1"
        assert event["count"] == 3
        assert "timestamp" in event
        assert "scroot_version" in event

    def test_disabled_emits_nothing(self, capsys):
        configure_audit_log(destination="disabled")
        emit("test_event")
        assert capsys.readouterr().err == ""


class TestEmitRobustness:
    def test_circular_structure_swallowed(self, capsys):
        configure_audit_log(destination="stderr")
        circular = {}
        circular["self"] = circular
        emit("bad_event", data=circular)  # must not raise
        assert capsys.readouterr().err == ""

    def test_non_json_types_stringified(self, tmp_path):
        path = tmp_path / "a.jsonl"
        configure_audit_log(destination="file", path=str(path))
        emit("typed_event", when={1, 2})  # set is not JSON - default=str
        event = json.loads(path.read_text(encoding="utf-8").strip())
        assert event["event"] == "typed_event"


class TestRotation:
    def test_malformed_and_blank_lines_dropped(self, tmp_path):
        path = tmp_path / "a.jsonl"
        fresh = {"event": "fresh", "timestamp": "2099-01-01T00:00:00+00:00"}
        path.write_text(
            "not json at all\n\n" + json.dumps(fresh) + "\n", encoding="utf-8"
        )
        configure_audit_log(destination="file", path=str(path), retention_days=90)
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "fresh"

    def test_rotate_missing_file_is_noop(self, tmp_path):
        configure_audit_log(
            destination="file", path=str(tmp_path / "missing.jsonl")
        )  # must not raise

    def test_rotate_never_raises_on_os_error(self, tmp_path, monkeypatch):
        path = tmp_path / "a.jsonl"
        path.write_text("{}\n", encoding="utf-8")

        def boom(*a, **k):
            raise OSError("locked")

        monkeypatch.setattr("builtins.open", boom)
        audit._rotate(str(path), 90)  # must not raise


class TestConfigureValidation:
    def test_tilde_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        configure_audit_log(destination="file", path="~/audit.jsonl")
        emit("home_event")
        assert (tmp_path / "audit.jsonl").exists()
