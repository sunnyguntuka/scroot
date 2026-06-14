"""RAG pipeline example: carry retrieval context through to scoring with ContextBuilder.

ContextBuilder accumulates grounding material as it becomes available at each
step of a retrieval-augmented pipeline, then hands it to ``auditor.score()``
without restructuring the rest of the pipeline.
"""

from scroot import Auditor, ContextBuilder

auditor = Auditor()

query = "What is our refund policy?"

# --- Step 1: record the query as soon as it arrives ---
ctx = ContextBuilder()
ctx.add_query(query)

# --- Step 2: record retrieved documents (the most important step for groundedness) ---
retrieved_docs = [
    "All customers are eligible for a 30-day full refund at no extra cost.",
    "Refund requests must be submitted via the support portal.",
]
ctx.add_retrieved(retrieved_docs, source="vector_db")

# --- Step 3: record reranked results, if your pipeline reranks ---
reranked_docs = [
    "All customers are eligible for a 30-day full refund at no extra cost.",
]
ctx.add_reranked(reranked_docs)

# --- Step 4: your LLM call happens here, untouched ---
response = "We offer a 30-day full refund at no extra cost."

# --- Step 5: build the context payload and score ---
result = auditor.score(query=query, response=response, context=ctx.build())

print(f"IQS:          {result.iqs:.2f}")
print(f"Groundedness: {result.groundedness:.2f}")
print(f"Completeness: {result.completeness:.2f}")
print(f"Flags:        {result.flags}")
