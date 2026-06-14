"""Tests for /api/records/:id via FastAPI TestClient."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scroot.dashboard.routers.records import records_router  # noqa: E402
from scroot.feedback.store import CorrectionRecord, FeedbackStore  # noqa: E402


def make_record(rid, scores=None, flags=None, guardrail_applied_count=0):
    return CorrectionRecord(
        id=rid,
        timestamp=datetime.now(timezone.utc).isoformat(),
        query="What is the refund policy?",
        response="Refunds anytime, no receipt needed.",
        scores=scores or {"iqs": 0.5},
        flags=flags or [],
        correction="",
        reason="",
        context_used=["Refunds within 30 days with a receipt."],
        corrected_by="agent-a",
        guardrail_applied_count=guardrail_applied_count,
    )


@pytest.fixture
def client(tmp_path):
    store = FeedbackStore(path=str(tmp_path / "records.jsonl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.add(make_record(
            "r1",
            scores={
                "iqs": 0.5,
                "groundedness": 0.3,
                "completeness": 0.8,
                "relevance": 0.9,
                "consistency": 0.9,
                "confidence": 0.9,
                "weakest_metric": "groundedness",
                "score_variance": 0.35,
                "iqs_explanation": "IQS 0.50 - primary driver: groundedness (0.30).",
                "metric_explanations": {
                    "groundedness": "The response makes claims that are not supported by the provided context.",
                },
                "evidence_map": {
                    "supported": 1,
                    "unsupported": 1,
                    "contradictions": 0,
                    "coverage_ratio": 0.5,
                    "weakest_sentence": "Refunds are available anytime.",
                    "entries": [
                        {
                            "response_sentence": "Refunds anytime, no receipt needed.",
                            "best_matching_chunk": "Refunds within 30 days with a receipt.",
                            "entailment_score": 0.9,
                            "supported": True,
                            "contradiction_detected": False,
                            "no_grounding_found": False,
                            "chunk_source": "retrieval",
                            "chunk_index": 0,
                        },
                    ],
                },
            },
            flags=["ungrounded"],
            guardrail_applied_count=3,
        ))
        store.add(make_record("r2"))
    app = FastAPI()
    app.include_router(records_router(store), prefix="/api/records")
    yield TestClient(app)


def test_get_record_includes_metric_explanations(client):
    data = client.get("/api/records/r1").json()
    assert data["metric_explanations"] == {
        "groundedness": "The response makes claims that are not supported by the provided context.",
    }
    assert data["weakest_metric"] == "groundedness"
    assert data["score_variance"] == 0.35
    assert data["iqs_explanation"] == "IQS 0.50 - primary driver: groundedness (0.30)."


def test_get_record_includes_guardrail_applied_count(client):
    data = client.get("/api/records/r1").json()
    assert data["guardrail_applied_count"] == 3


def test_get_record_includes_evidence_map(client):
    data = client.get("/api/records/r1").json()
    assert data["evidence_map"]["coverage_ratio"] == 0.5
    assert data["evidence_map"]["supported"] == 1
    assert len(data["evidence_map"]["entries"]) == 1


def test_get_record_evidence_map_none_without_scores(client):
    data = client.get("/api/records/r2").json()
    assert data["evidence_map"] is None


def test_get_record_metrics_excludes_derived_fields(client):
    data = client.get("/api/records/r1").json()
    assert set(data["metrics"]) == {
        "groundedness", "completeness", "relevance", "consistency", "confidence",
    }


def test_get_record_defaults_for_record_without_extra_scores(client):
    data = client.get("/api/records/r2").json()
    assert data["weakest_metric"] is None
    assert data["score_variance"] is None
    assert data["iqs_explanation"] is None
    assert data["metric_explanations"] == {}
    assert data["guardrail_applied_count"] == 0


def test_get_record_context_used_true_when_groundedness_scored(client):
    data = client.get("/api/records/r1").json()
    assert data["context_used"] is True
    assert data["iqs_metric_count"] == 5


def test_get_record_context_used_false_without_groundedness(client):
    # r2's scores have no groundedness key → derived as no-context.
    data = client.get("/api/records/r2").json()
    assert data["context_used"] is False
    assert data["iqs_metric_count"] == 4


def test_get_record_not_found(client):
    resp = client.get("/api/records/missing")
    assert resp.status_code == 404
