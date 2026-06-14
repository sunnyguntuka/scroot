"""Feedback loop example: score → flag → store → inject guardrails."""

import uuid
from datetime import datetime, timezone

from scroot import Auditor
from scroot.feedback.store import FeedbackStore, CorrectionRecord
from scroot.feedback.injector import GuardrailInjector

auditor = Auditor()
store = FeedbackStore("corrections.jsonl")
injector = GuardrailInjector(store)

# Simulate a bad response being logged for review
bad_query = "What is our refund policy?"
bad_response = "We offer a 90-day money-back guarantee with free worldwide shipping."
context = ["All customers are eligible for a 30-day full refund at no extra cost."]

result = auditor.score(query=bad_query, response=bad_response, context=context)
print(f"Scored bad response: IQS={result.iqs:.3f}, flags={result.flags}")

if result.iqs < 0.7 or result.flags:
    record = CorrectionRecord(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=bad_query,
        response=bad_response,
        scores=result.to_dict(),
        flags=result.flags,
        correction="We offer a 30-day full refund at no extra cost.",
        reason="Response fabricated 90-day guarantee and free shipping.",
        context_used=context,
        corrected_by="human",
        metadata={"model": "gpt-4o"},
    )
    store.add(record)
    print(f"Logged correction. Store now has {store.count()} record(s).")

# Build guardrail context for the next inference
guardrails = injector.build_context(
    query=bad_query,
    strategy="recent",
    max_corrections=5,
)

print("\n=== Guardrail context for next LLM call ===")
print(guardrails if guardrails else "(no guardrails yet)")

# Show rules-based extraction
rules = injector.build_context(strategy="rules")
print("\n=== Rules extracted from corrections ===")
print(rules if rules else "(no rules yet)")

print(f"\nTotal corrections stored: {store.count()}")
