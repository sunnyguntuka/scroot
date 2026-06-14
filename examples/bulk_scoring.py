"""Bulk scoring example: score a batch of stored query/response pairs.

Demonstrates ``Auditor.score_batch()`` over a larger dataset and
summarises the results - the kind of pass you'd run over a logged
sample of production responses.
"""

from scroot import Auditor

auditor = Auditor()

# Stand-in for 100 logged (query, response, context) records pulled from
# your own store (database, JSONL file, data warehouse, ...).
base_items = [
    {
        "query": "What is our refund policy?",
        "response": "We offer a 30-day full refund at no extra cost.",
        "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
    },
    {
        "query": "How do I reset my password?",
        "response": "Go to Settings > Security and click 'Reset password'.",
        "context": ["Users can reset their password from Settings > Security."],
    },
    {
        "query": "What is our refund policy?",
        "response": "We offer a 90-day money-back guarantee with free worldwide shipping.",
        "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
    },
]
items = base_items * 34  # 102 items, simulating a bulk batch

results = auditor.score_batch(items)

mean_iqs = sum(r.iqs for r in results) / len(results)
flag_counts: dict[str, int] = {}
for r in results:
    for flag in r.flags:
        flag_counts[flag] = flag_counts.get(flag, 0) + 1

print(f"Scored {len(results)} responses")
print(f"Mean IQS: {mean_iqs:.3f}")
print(f"Flag counts: {flag_counts}")

low_scoring = [r for r in results if r.iqs < 0.7]
print(f"Below 0.70 IQS: {len(low_scoring)}")
