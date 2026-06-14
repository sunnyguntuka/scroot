import pytest
from scroot import Auditor, EntailmentResult
from .conftest import (
    GROUNDED_EXAMPLE,
    HALLUCINATED_EXAMPLE,
    OFF_TOPIC_EXAMPLE,
    CONTRADICTORY_EXAMPLE,
)

pytestmark = pytest.mark.needs_model


@pytest.fixture(scope="module")
def auditor():
    return Auditor()


class TestAuditor:
    def test_score_with_context(self, auditor):
        ex = GROUNDED_EXAMPLE
        result = auditor.score(
            query=ex["query"],
            response=ex["response"],
            context=ex["context"],
        )
        assert isinstance(result, EntailmentResult)
        assert result.groundedness is not None
        assert result.iqs > 0.0
        assert 0.0 <= result.iqs <= 1.0

    def test_score_without_context(self, auditor):
        result = auditor.score(
            query="Explain quantum computing",
            response="Quantum computing uses qubits that can be in superposition...",
        )
        assert result.groundedness is None
        assert result.iqs > 0.0

    def test_hallucinated_response(self, auditor):
        ex = HALLUCINATED_EXAMPLE
        result = auditor.score(
            query=ex["query"],
            response=ex["response"],
            context=ex["context"],
        )
        assert result.groundedness is not None
        # Hallucinated response should have lower groundedness
        assert result.groundedness < 0.9

    def test_off_topic_response(self, auditor):
        ex = OFF_TOPIC_EXAMPLE
        result = auditor.score(
            query=ex["query"],
            response=ex["response"],
            context=ex["context"],
        )
        assert isinstance(result, EntailmentResult)
        assert 0.0 <= result.relevance <= 1.0

    def test_contradictory_response(self, auditor):
        ex = CONTRADICTORY_EXAMPLE
        result = auditor.score(
            query=ex["query"],
            response=ex["response"],
            context=ex["context"],
        )
        assert isinstance(result, EntailmentResult)
        assert 0.0 <= result.consistency <= 1.0

    def test_empty_response(self, auditor):
        result = auditor.score(
            query="What is the policy?",
            response="",
        )
        assert isinstance(result, EntailmentResult)
        assert result.iqs >= 0.0

    def test_batch_scoring(self, auditor):
        items = [
            {"query": GROUNDED_EXAMPLE["query"], "response": GROUNDED_EXAMPLE["response"], "context": GROUNDED_EXAMPLE["context"]},
            {"query": HALLUCINATED_EXAMPLE["query"], "response": HALLUCINATED_EXAMPLE["response"], "context": HALLUCINATED_EXAMPLE["context"]},
            {"query": "Explain AI", "response": "AI stands for artificial intelligence."},
        ]
        results = auditor.score_batch(items)
        assert len(results) == 3
        assert all(isinstance(r, EntailmentResult) for r in results)

    def test_result_has_details(self, auditor):
        result = auditor.score(
            query=GROUNDED_EXAMPLE["query"],
            response=GROUNDED_EXAMPLE["response"],
            context=GROUNDED_EXAMPLE["context"],
        )
        assert isinstance(result.details, dict)
        assert "completeness" in result.details
        assert "relevance" in result.details

    def test_result_flags_list(self, auditor):
        result = auditor.score(
            query=GROUNDED_EXAMPLE["query"],
            response=GROUNDED_EXAMPLE["response"],
            context=GROUNDED_EXAMPLE["context"],
        )
        assert isinstance(result.flags, list)

    def test_evidence_map_with_context(self, auditor):
        result = auditor.score(
            query=GROUNDED_EXAMPLE["query"],
            response=GROUNDED_EXAMPLE["response"],
            context=GROUNDED_EXAMPLE["context"],
        )
        assert result.evidence_map is not None
        assert 0.0 <= result.evidence_map.coverage_ratio <= 1.0
        assert len(result.evidence_map.entries) > 0

    def test_evidence_map_none_without_context(self, auditor):
        result = auditor.score(
            query="Explain quantum computing",
            response="Quantum computing uses qubits that can be in superposition...",
        )
        assert result.evidence_map is None
