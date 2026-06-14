"""Basic usage example for scroot."""

from scroot import Auditor

auditor = Auditor()

# --- Example 1: RAG response with context ---
result = auditor.score(
    query="What is our refund policy?",
    response="We offer a 30-day full refund at no extra cost.",
    context=["All customers are eligible for a 30-day full refund at no extra cost."],
)

print("=== Grounded RAG Response ===")
print(result)
print(f"IQS: {result.iqs}")
print(f"Groundedness: {result.groundedness}")
print(f"Completeness: {result.completeness}")
print(f"Relevance: {result.relevance}")
print(f"Consistency: {result.consistency}")
print(f"Confidence: {result.confidence}")
print(f"Flags: {result.flags}")
print()

# --- Example 2: No context (general chatbot) ---
result2 = auditor.score(
    query="Explain quantum computing",
    response="Quantum computing uses qubits that can be in superposition, allowing parallel computation.",
)

print("=== No-Context Mode ===")
print(result2)
print(f"Groundedness (should be None): {result2.groundedness}")
print(f"IQS: {result2.iqs}")
print()

# --- Example 3: Hallucinated response ---
result3 = auditor.score(
    query="What is our refund policy?",
    response="We offer a 90-day money-back guarantee with free return shipping worldwide.",
    context=["All customers are eligible for a 30-day full refund at no extra cost."],
)

print("=== Hallucinated Response ===")
print(result3)
print(f"Groundedness: {result3.groundedness}")
print(f"Flags: {result3.flags}")
print()

# --- Example 4: Batch scoring ---
results = auditor.score_batch([
    {"query": "What is AI?", "response": "AI is artificial intelligence."},
    {"query": "What is our policy?", "response": "30-day refund.", "context": ["30-day refund policy."]},
])

print("=== Batch Scoring ===")
for i, r in enumerate(results):
    print(f"Item {i+1}: IQS={r.iqs:.3f}, flags={r.flags}")
