import pytest
from scroot.feedback.store import FeedbackStore, CorrectionRecord

pytestmark = pytest.mark.filterwarnings("ignore:FeedbackStore writing unencrypted:UserWarning")


def make_record(id="abc123", query="What is the policy?"):
    return CorrectionRecord(
        id=id,
        timestamp="2026-05-28T00:00:00Z",
        query=query,
        response="Wrong response",
        scores={"iqs": 0.3},
        flags=["hallucination_risk"],
        correction="Correct response",
        reason="Made up facts",
        context_used=["Real context"],
        corrected_by="human",
    )


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(path=str(tmp_path / "test.jsonl"))


def test_add_and_get_all(store):
    record = make_record()
    store.add(record)
    records = store.get_all()
    assert len(records) == 1
    assert records[0].id == "abc123"


def test_count(store):
    assert store.count() == 0
    store.add(make_record("r1"))
    store.add(make_record("r2"))
    assert store.count() == 2


def test_get_recent(store):
    for i in range(5):
        store.add(make_record(id=f"r{i}"))
    recent = store.get_recent(3)
    assert len(recent) == 3
    assert recent[-1].id == "r4"


def test_empty_store_returns_empty_list(store):
    assert store.get_all() == []
    assert store.get_recent() == []
    assert store.count() == 0


def test_nonexistent_path_returns_empty():
    store = FeedbackStore(path="/nonexistent/path/file.jsonl")
    assert store.get_all() == []
    assert store.count() == 0


def test_record_roundtrip(store):
    original = make_record()
    store.add(original)
    loaded = store.get_all()[0]
    assert loaded.query == original.query
    assert loaded.flags == original.flags
    assert loaded.scores == original.scores


def test_metadata_optional(store):
    record = make_record()
    record.metadata = {"agent": "support_bot"}
    store.add(record)
    loaded = store.get_all()[0]
    assert loaded.metadata == {"agent": "support_bot"}


def test_increment_guardrail_count(store):
    store.add(make_record("r1"))
    store.add(make_record("r2"))

    store.increment_guardrail_count(["r1"])

    records = {r.id: r for r in store.get_all()}
    assert records["r1"].guardrail_applied_count == 1
    assert records["r2"].guardrail_applied_count == 0

    store.increment_guardrail_count(["r1"])
    records = {r.id: r for r in store.get_all()}
    assert records["r1"].guardrail_applied_count == 2


def test_increment_guardrail_count_ignores_unknown_ids(store):
    store.add(make_record("r1"))
    store.increment_guardrail_count(["unknown-id"])
    assert store.get_all()[0].guardrail_applied_count == 0


def test_increment_guardrail_count_empty_list_noop(store):
    store.add(make_record("r1"))
    store.increment_guardrail_count([])
    assert store.get_all()[0].guardrail_applied_count == 0


def test_increment_guardrail_count_persists_across_reload(store, tmp_path):
    store.add(make_record("r1"))
    store.increment_guardrail_count(["r1"])

    reloaded = FeedbackStore(path=str(tmp_path / "test.jsonl"))
    assert reloaded.get_all()[0].guardrail_applied_count == 1


def test_default_guardrail_applied_count(store):
    store.add(make_record("r1"))
    assert store.get_all()[0].guardrail_applied_count == 0
