"""Tests for /api/guardrails endpoints via FastAPI TestClient."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scroot.dashboard.routers.guardrails import guardrails_router  # noqa: E402
from scroot.feedback.store import CorrectionRecord, FeedbackStore  # noqa: E402


def make_record(rid, guardrail_applied_count=0):
    return CorrectionRecord(
        id=rid,
        timestamp=datetime.now(timezone.utc).isoformat(),
        query="What is the refund policy?",
        response="Some response text.",
        scores={"iqs": 0.5},
        flags=[],
        correction="",
        reason="",
        context_used=[],
        corrected_by="agent-a",
        guardrail_applied_count=guardrail_applied_count,
    )


@pytest.fixture
def client(tmp_path):
    store = FeedbackStore(path=str(tmp_path / "guardrails.jsonl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.add(make_record("r-unused", guardrail_applied_count=0))
        store.add(make_record("r-used-once", guardrail_applied_count=1))
        store.add(make_record("r-used-many", guardrail_applied_count=5))
    app = FastAPI()
    app.include_router(guardrails_router(store), prefix="/api/guardrails")
    yield TestClient(app)


def test_stats_shape(client):
    data = client.get("/api/guardrails/stats").json()
    assert data["active_guardrails"] == 2
    assert data["total_applications"] == 6
    assert [r["id"] for r in data["records"]] == ["r-used-many", "r-used-once"]


def test_stats_only_includes_records_with_count_above_zero(client):
    data = client.get("/api/guardrails/stats").json()
    ids = [r["id"] for r in data["records"]]
    assert "r-unused" not in ids


def test_stats_empty_store(tmp_path):
    store = FeedbackStore(path=str(tmp_path / "empty.jsonl"))
    app = FastAPI()
    app.include_router(guardrails_router(store), prefix="/api/guardrails")
    data = TestClient(app).get("/api/guardrails/stats").json()
    assert data["active_guardrails"] == 0
    assert data["total_applications"] == 0
    assert data["records"] == []
