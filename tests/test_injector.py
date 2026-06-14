import pytest
from scroot.feedback.store import FeedbackStore, CorrectionRecord
from scroot.feedback.injector import GuardrailInjector

pytestmark = pytest.mark.filterwarnings("ignore:FeedbackStore writing unencrypted:UserWarning")


def make_record(id, query, response, correction, reason):
    return CorrectionRecord(
        id=id,
        timestamp="2026-05-28T00:00:00Z",
        query=query,
        response=response,
        scores={"iqs": 0.3},
        flags=["hallucination_risk"],
        correction=correction,
        reason=reason,
        context_used=[],
        corrected_by="human",
    )


@pytest.fixture
def populated_store(tmp_path):
    store = FeedbackStore(path=str(tmp_path / "test.jsonl"))
    store.add(make_record("r1", "What is refund policy?", "90-day guarantee", "30-day refund", "Made up 90-day policy"))
    store.add(make_record("r2", "How do I cancel?", "You can't cancel", "Cancel via settings", "Incorrect cancellation info"))
    return store


@pytest.fixture
def empty_store(tmp_path):
    return FeedbackStore(path=str(tmp_path / "empty.jsonl"))


def test_build_context_recent(populated_store):
    injector = GuardrailInjector(populated_store)
    context = injector.build_context(strategy="recent", max_corrections=5)
    assert "KNOWN CORRECTIONS" in context
    assert len(context) > 0


def test_build_context_rules(populated_store):
    injector = GuardrailInjector(populated_store)
    context = injector.build_context(strategy="rules")
    assert "GUARDRAILS" in context


def test_build_context_empty_store(empty_store):
    injector = GuardrailInjector(empty_store)
    context = injector.build_context(strategy="recent")
    assert context == ""


def test_invalid_strategy(populated_store):
    injector = GuardrailInjector(populated_store)
    with pytest.raises(ValueError, match="Unknown strategy"):
        injector.build_context(strategy="invalid")


def test_relevant_requires_query(populated_store):
    injector = GuardrailInjector(populated_store)
    with pytest.raises(ValueError, match="query required"):
        injector.build_context(strategy="relevant", query=None)


def test_token_budget_respected(populated_store):
    injector = GuardrailInjector(populated_store)
    context = injector.build_context(strategy="recent", max_tokens=50)
    # Should be truncated to roughly 200 chars
    assert len(context) <= 500


def test_rules_deduplicates(tmp_path):
    store = FeedbackStore(path=str(tmp_path / "dedup.jsonl"))
    # Add two records with identical reason
    for i in range(3):
        store.add(make_record(f"r{i}", f"query {i}", "wrong", "correct", "Same reason text here"))
    injector = GuardrailInjector(store)
    context = injector.build_context(strategy="rules")
    # Should appear only once
    assert context.count("Same reason text here") == 1


def test_build_context_recent_increments_guardrail_count(populated_store):
    injector = GuardrailInjector(populated_store)
    injector.build_context(strategy="recent", max_corrections=5)

    records = {r.id: r for r in populated_store.get_all()}
    assert records["r1"].guardrail_applied_count == 1
    assert records["r2"].guardrail_applied_count == 1


@pytest.mark.needs_model
def test_build_context_relevant_increments_guardrail_count(populated_store):
    injector = GuardrailInjector(populated_store)
    injector.build_context(strategy="relevant", query="What is refund policy?")

    records = {r.id: r for r in populated_store.get_all()}
    assert records["r1"].guardrail_applied_count == 1


def test_build_context_rules_increments_guardrail_count(populated_store):
    injector = GuardrailInjector(populated_store)
    injector.build_context(strategy="rules")

    records = {r.id: r for r in populated_store.get_all()}
    assert records["r1"].guardrail_applied_count == 1
    assert records["r2"].guardrail_applied_count == 1


def test_build_context_empty_store_does_not_increment(empty_store):
    injector = GuardrailInjector(empty_store)
    injector.build_context(strategy="recent")
    assert empty_store.get_all() == []


def test_token_budget_excludes_records_from_guardrail_count(populated_store):
    injector = GuardrailInjector(populated_store)
    injector.build_context(strategy="recent", max_tokens=1)

    records = {r.id: r for r in populated_store.get_all()}
    assert records["r1"].guardrail_applied_count == 0
    assert records["r2"].guardrail_applied_count == 0
