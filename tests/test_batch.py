import pytest
from scroot import Auditor, EntailmentResult

pytestmark = pytest.mark.needs_model


@pytest.fixture(scope="module")
def auditor():
    return Auditor()


def test_batch_returns_list(auditor):
    items = [
        {"query": "What is AI?", "response": "AI is artificial intelligence."},
        {"query": "What is ML?", "response": "ML is machine learning."},
    ]
    results = auditor.score_batch(items)
    assert isinstance(results, list)
    assert len(results) == 2


def test_batch_all_audit_results(auditor):
    items = [
        {"query": "What is AI?", "response": "AI stands for artificial intelligence."},
        {"query": "What is the refund policy?", "response": "30 days.", "context": ["We have a 30-day refund."]},
    ]
    results = auditor.score_batch(items)
    assert all(isinstance(r, EntailmentResult) for r in results)


def test_batch_preserves_context(auditor):
    items = [
        {"query": "q1", "response": "r1", "context": ["ctx1"]},
        {"query": "q2", "response": "r2"},  # no context
    ]
    results = auditor.score_batch(items)
    assert results[0].groundedness is not None
    assert results[1].groundedness is None


def test_batch_empty_list(auditor):
    results = auditor.score_batch([])
    assert results == []


def test_batch_single_item(auditor):
    items = [{"query": "What is Python?", "response": "Python is a programming language."}]
    results = auditor.score_batch(items)
    assert len(results) == 1
