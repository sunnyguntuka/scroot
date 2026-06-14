"""Tests for /api/queue endpoints via FastAPI TestClient."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scroot.dashboard.routers import queue as queue_mod  # noqa: E402
from scroot.dashboard.routers.queue import queue_router  # noqa: E402
from scroot.feedback.store import CorrectionRecord, FeedbackStore  # noqa: E402


def make_record(
    rid,
    iqs=0.5,
    status="pending",
    flags=None,
    query="What is the refund policy?",
    timestamp=None,
    session_id=None,
    context_checksum=None,
):
    return CorrectionRecord(
        id=rid,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        query=query,
        response="Some response text.",
        scores={"iqs": iqs},
        flags=flags or [],
        correction="",
        reason="",
        context_used=[],
        corrected_by="agent-a",
        status=status,
        session_id=session_id,
        context_checksum=context_checksum,
    )


@pytest.fixture
def client(tmp_path):
    store = FeedbackStore(path=str(tmp_path / "queue.jsonl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.add(make_record("r-low", iqs=0.20, flags=["ungrounded"]))
        store.add(make_record("r-warn", iqs=0.55, flags=["incomplete"]))
        store.add(make_record(
            "r-high", iqs=0.92, status="reviewed",
            query="How do I reset my password?",
            session_id="cb-q1", context_checksum="sha256:abc",
        ))
    app = FastAPI()
    app.include_router(queue_router(store), prefix="/api/queue")
    queue_mod._claims.clear()
    yield TestClient(app)
    queue_mod._claims.clear()


class TestListQueue:
    def test_returns_all(self, client):
        data = client.get("/api/queue").json()
        assert data["total"] == 3
        assert len(data["records"]) == 3
        assert data["page"] == 1

    def test_iqs_status_filters(self, client):
        fail = client.get("/api/queue", params={"status": "fail"}).json()
        assert [r["id"] for r in fail["records"]] == ["r-low"]
        warn = client.get("/api/queue", params={"status": "warn"}).json()
        assert [r["id"] for r in warn["records"]] == ["r-warn"]
        passed = client.get("/api/queue", params={"status": "pass"}).json()
        assert [r["id"] for r in passed["records"]] == ["r-high"]

    def test_workflow_status_filter(self, client):
        data = client.get("/api/queue", params={"status": "reviewed"}).json()
        assert [r["id"] for r in data["records"]] == ["r-high"]

    def test_applied_status_does_not_500(self, tmp_path):
        # Regression: 'applied' is a documented CorrectionRecord status, but the
        # QueueItem Literal omitted it, so any applied record 500'd the whole
        # queue listing.
        store = FeedbackStore(path=str(tmp_path / "applied.jsonl"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.add(make_record("r-applied", iqs=0.9, status="applied"))
        app = FastAPI()
        app.include_router(queue_router(store), prefix="/api/queue")
        resp = TestClient(app).get("/api/queue")
        assert resp.status_code == 200
        assert resp.json()["records"][0]["status"] == "applied"

    def test_search(self, client):
        data = client.get("/api/queue", params={"search": "password"}).json()
        assert [r["id"] for r in data["records"]] == ["r-high"]

    def test_flag_filter(self, client):
        data = client.get("/api/queue", params={"flag": "ungrounded"}).json()
        assert [r["id"] for r in data["records"]] == ["r-low"]

    def test_iqs_range_filter(self, client):
        data = client.get(
            "/api/queue", params={"min_iqs": 0.5, "max_iqs": 0.9}
        ).json()
        assert [r["id"] for r in data["records"]] == ["r-warn"]

    def test_sort_iqs(self, client):
        asc = client.get("/api/queue", params={"sort": "iqs_asc"}).json()
        assert [r["id"] for r in asc["records"]] == ["r-low", "r-warn", "r-high"]
        desc = client.get("/api/queue", params={"sort": "iqs_desc"}).json()
        assert [r["id"] for r in desc["records"]] == ["r-high", "r-warn", "r-low"]

    def test_sort_created(self, client):
        oldest = client.get("/api/queue", params={"sort": "created_asc"}).json()
        assert oldest["records"][0]["id"] == "r-low"
        newest = client.get("/api/queue", params={"sort": "newest"}).json()
        assert newest["records"][0]["id"] == "r-high"

    def test_pagination(self, client):
        page1 = client.get(
            "/api/queue", params={"limit": 2, "page": 1, "sort": "created_asc"}
        ).json()
        page2 = client.get(
            "/api/queue", params={"limit": 2, "page": 2, "sort": "created_asc"}
        ).json()
        assert len(page1["records"]) == 2
        assert len(page2["records"]) == 1
        assert page1["total"] == 3
        assert page2["page"] == 2

    def test_session_id_and_checksum_exposed(self, client):
        data = client.get("/api/queue", params={"search": "password"}).json()
        record = data["records"][0]
        assert record["session_id"] == "cb-q1"
        assert record["context_checksum"] == "sha256:abc"

    def test_session_fields_default_none(self, client):
        data = client.get("/api/queue", params={"status": "fail"}).json()
        record = data["records"][0]
        assert record["session_id"] is None
        assert record["context_checksum"] is None


class TestClaims:
    def test_claim_and_release(self, client):
        resp = client.post("/api/queue/claim/r-low")
        assert resp.status_code == 200
        assert resp.json()["status"] == "claimed"

        # Claimed record carries claimed_at in the queue listing
        data = client.get("/api/queue", params={"status": "fail"}).json()
        assert data["records"][0]["claimed_at"] is not None

        resp = client.delete("/api/queue/claim/r-low")
        assert resp.json()["status"] == "released"

    def test_double_claim_conflicts(self, client):
        client.post("/api/queue/claim/r-low")
        resp = client.post("/api/queue/claim/r-low")
        assert resp.status_code == 409

    def test_unclaim_unknown_is_idempotent(self, client):
        resp = client.delete("/api/queue/claim/never-claimed")
        assert resp.status_code == 200


class TestStats:
    def test_stats_shape(self, client):
        data = client.get("/api/queue/stats").json()
        assert data["pending"] == 2
        assert 0.0 <= data["avg_iqs"] <= 1.0
        assert data["oldest_pending_hours"] >= 0.0

    def test_reviewed_today_counted(self, client):
        # r-high was reviewed with a today timestamp
        data = client.get("/api/queue/stats").json()
        assert data["reviewed_today"] == 1

    def test_stats_empty_store(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "empty.jsonl"))
        app = FastAPI()
        app.include_router(queue_router(store), prefix="/api/queue")
        data = TestClient(app).get("/api/queue/stats").json()
        assert data["pending"] == 0
        assert data["avg_iqs"] == 0.0
        assert data["oldest_pending_hours"] == 0.0
