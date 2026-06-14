"""ContextBuilder → Auditor.score() end-to-end integration tests."""

import pytest

from scroot import Auditor, ContextBuilder, EntailmentResult
from scroot.exceptions import ContextEmptyWarning

pytestmark = pytest.mark.needs_model

QUERY = "What is our return policy?"
RESPONSE = (
    "We offer a 30-day full refund at no extra cost. You can return any "
    "item within 30 days of purchase."
)
CONTEXT_CHUNK = (
    "All customers are eligible for a 30-day full refund at no extra cost. "
    "Items must be returned within 30 days of the original purchase date."
)


@pytest.fixture(scope="module")
def auditor():
    return Auditor()


class TestScoreWithPayload:
    def test_payload_scores_groundedness(self, auditor):
        ctx = ContextBuilder()
        ctx.add_query(QUERY)
        ctx.add_retrieved([CONTEXT_CHUNK])
        result = auditor.score(QUERY, RESPONSE, context=ctx.build())
        assert isinstance(result, EntailmentResult)
        assert result.groundedness is not None
        assert result.groundedness > 0.5

    def test_context_beats_no_context(self, auditor):
        ctx = ContextBuilder()
        ctx.add_retrieved([CONTEXT_CHUNK])
        with_ctx = auditor.score(QUERY, RESPONSE, context=ctx.build())
        without = auditor.score(QUERY, RESPONSE, context=None)
        assert without.groundedness is None
        assert with_ctx.groundedness is not None
        # A grounded factual response must score higher with context present
        assert with_ctx.groundedness > 0.0

    def test_audit_trail_in_details(self, auditor):
        ctx = ContextBuilder(session_id="trace-integration")
        ctx.add_retrieved([CONTEXT_CHUNK])
        payload = ctx.build()
        result = auditor.score(QUERY, RESPONSE, context=payload)
        assert result.details["context"]["session_id"] == "trace-integration"
        assert result.details["context"]["checksum"] == payload.checksum
        # Raw context text never flows into the result details
        assert CONTEXT_CHUNK not in str(result.details["context"])

    def test_none_payload_from_empty_builder(self, auditor):
        ctx = ContextBuilder()
        with pytest.warns(ContextEmptyWarning):
            payload = ctx.build()
        result = auditor.score(QUERY, RESPONSE, context=payload)
        assert result.groundedness is None


class TestBackwardCompat:
    def test_plain_string_context(self, auditor):
        result = auditor.score(QUERY, RESPONSE, context=CONTEXT_CHUNK)
        assert result.groundedness is not None
        assert result.groundedness > 0.5

    def test_list_context_unchanged(self, auditor):
        result = auditor.score(QUERY, RESPONSE, context=[CONTEXT_CHUNK])
        assert result.groundedness is not None
        assert result.groundedness > 0.5

    def test_none_context_unchanged(self, auditor):
        result = auditor.score(QUERY, RESPONSE, context=None)
        assert result.groundedness is None
        assert result.iqs > 0.0


class TestFeedbackStoreFields:
    def test_session_id_and_checksum_on_correction_record(self, tmp_path):
        from scroot.feedback.store import CorrectionRecord, FeedbackStore

        store = FeedbackStore(path=str(tmp_path / "fb.jsonl"))
        record = CorrectionRecord(
            id="r1",
            timestamp="2026-06-09T00:00:00+00:00",
            query=QUERY,
            response=RESPONSE,
            scores={"iqs": 0.42},
            flags=["ungrounded"],
            correction="",
            reason="",
            context_used=[],
            corrected_by="test",
            session_id="cb-abc",
            context_checksum="sha256:deadbeef",
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.add(record)
        loaded = store.get_all()[0]
        assert loaded.session_id == "cb-abc"
        assert loaded.context_checksum == "sha256:deadbeef"
