"""Tests for Auditor input length limits (H-3)."""

import pytest
from unittest.mock import patch
from scroot.core import Auditor


def make_auditor(**kwargs):
    return Auditor(**kwargs)


def _mock_all_metrics(score_value=0.8):
    """Patch all metric functions to return a fixed score quickly."""
    details = {}
    patches = [
        patch("scroot.core.score_groundedness", return_value=(score_value, details)),
        patch("scroot.core.score_completeness", return_value=(score_value, details)),
        patch("scroot.core.score_relevance",    return_value=(score_value, details)),
        patch("scroot.core.score_consistency",  return_value=(score_value, details)),
        patch("scroot.core.score_confidence",   return_value=(score_value, details)),
    ]
    return patches


def test_query_truncated_to_max_length():
    auditor = make_auditor(max_query_length=10)
    captured = {}

    def fake_completeness(query, response, **kw):
        captured["query"] = query
        return 0.8, {}

    with patch("scroot.core.score_completeness", side_effect=fake_completeness), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  return_value=(0.8, {})), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})):
        auditor.score(query="A" * 100, response="some response")

    assert len(captured["query"]) == 10


def test_response_truncated_to_max_length():
    auditor = make_auditor(max_response_length=20)
    captured = {}

    def fake_consistency(response, **kw):
        captured["response"] = response
        return 0.8, {}

    with patch("scroot.core.score_completeness", return_value=(0.8, {})), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  side_effect=fake_consistency), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})):
        auditor.score(query="q", response="R" * 100)

    assert len(captured["response"]) == 20


def test_context_items_truncated_to_max():
    auditor = make_auditor(max_context_items=3)
    captured = {}

    def fake_groundedness(response, context, **kw):
        captured["context"] = context
        return 0.8, {}

    with patch("scroot.core.score_groundedness", side_effect=fake_groundedness), \
         patch("scroot.core.score_completeness", return_value=(0.8, {})), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  return_value=(0.8, {})), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})):
        auditor.score(query="q", response="r", context=["chunk"] * 10)

    assert len(captured["context"]) == 3


def test_context_item_length_truncated():
    auditor = make_auditor(max_context_item_length=5)
    captured = {}

    def fake_groundedness(response, context, **kw):
        captured["context"] = context
        return 0.8, {}

    with patch("scroot.core.score_groundedness", side_effect=fake_groundedness), \
         patch("scroot.core.score_completeness", return_value=(0.8, {})), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  return_value=(0.8, {})), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})):
        auditor.score(query="q", response="r", context=["A" * 100])

    assert all(len(c) <= 5 for c in captured["context"])


def test_score_batch_exceeding_max_raises():
    auditor = make_auditor(max_batch_size=5)
    items = [{"query": "q", "response": "r"}] * 6
    with pytest.raises(ValueError, match="max_batch_size"):
        auditor.score_batch(items)


def test_evidence_map_built_when_context_provided():
    auditor = make_auditor()
    sentinel = object()

    with patch("scroot.core.score_groundedness", return_value=(0.8, {})), \
         patch("scroot.core.score_completeness", return_value=(0.8, {})), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  return_value=(0.8, {})), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})), \
         patch("scroot.core.build_evidence_map", return_value=sentinel) as mock_build:
        result = auditor.score(query="q", response="r", context=["chunk"])

    mock_build.assert_called_once()
    args, kwargs = mock_build.call_args
    assert args[0] == "r"
    assert args[1] == ["chunk"]
    assert result.evidence_map is sentinel


def test_evidence_map_not_built_without_context():
    auditor = make_auditor()

    with patch("scroot.core.score_completeness", return_value=(0.8, {})), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  return_value=(0.8, {})), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})), \
         patch("scroot.core.build_evidence_map") as mock_build:
        result = auditor.score(query="q", response="r")

    mock_build.assert_not_called()
    assert result.evidence_map is None


def test_evidence_map_disabled_via_compute_evidence_map_flag():
    auditor = make_auditor(compute_evidence_map=False)

    with patch("scroot.core.score_groundedness", return_value=(0.8, {})), \
         patch("scroot.core.score_completeness", return_value=(0.8, {})), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  return_value=(0.8, {})), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})), \
         patch("scroot.core.build_evidence_map") as mock_build:
        result = auditor.score(query="q", response="r", context=["chunk"])

    mock_build.assert_not_called()
    assert result.evidence_map is None


def test_score_batch_at_limit_succeeds():
    auditor = make_auditor(max_batch_size=3)
    items = [{"query": "q", "response": "r"}] * 3
    with patch("scroot.core.score_completeness", return_value=(0.8, {})), \
         patch("scroot.core.score_relevance",    return_value=(0.8, {})), \
         patch("scroot.core.score_consistency",  return_value=(0.8, {})), \
         patch("scroot.core.score_confidence",   return_value=(0.8, {})):
        results = auditor.score_batch(items)
    assert len(results) == 3
