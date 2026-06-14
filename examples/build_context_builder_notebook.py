"""Generate examples/context_builder.ipynb - the ContextBuilder demo notebook.

Run:  python examples/build_context_builder_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace examples/context_builder.ipynb
"""

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []


def md(text):
    cells.append(new_markdown_cell(text.strip()))


def code(text):
    cells.append(new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# ContextBuilder - carrying grounding context to `auditor.score()`

In a typical RAG pipeline, grounding documents exist at **step 3 (retrieval)**
but the LLM response isn't available until **step 6 (generation)**. By the time
you call `auditor.score()`, the retrieved chunks are out of scope - and without
them, **groundedness** (scroot's strongest, hallucination-catching metric)
cannot be computed.

`ContextBuilder` is a lightweight, request-scoped accumulator that fixes this:

1. Create one at the start of a request
2. Add grounding material as it becomes available at each pipeline step
3. Pass `ctx.build()` to `auditor.score()` at the end

Your LLM call is **never touched** - context assembly is fully separated from
scoring. And it's SOC II-minded by default: PII scrubbing on, content held in
memory only, content-free audit trail.

This notebook is fully runnable **without any credentials or API keys**.
""")

# --- Cell 1: installation and imports --------------------------------------
md("""
## 1. Installation and imports

`ContextBuilder` ships with the core package - no extras needed.
""")
code("""
# pip install scroot

import warnings
import scroot
from scroot import Auditor, ContextBuilder

# Keep notebook output clean: audit events go to stderr by default.
scroot.configure_audit_log(destination="disabled")

auditor = Auditor()   # loads NLI + embedding models once, then cached
print("scroot", scroot.__version__, "ready")
""")

# --- Cell 2: minimal example - plain string (existing behaviour) ------------
md("""
## 2. The existing behaviour - plain string context

Before reaching for `ContextBuilder`: if you already have your grounding text
in hand as a string (or list of strings), `auditor.score()` accepts it
directly. This behaviour is unchanged.
""")
code("""
result = auditor.score(
    query="What is our refund policy?",
    response="We offer a 30-day full refund at no extra cost.",
    context="All customers are eligible for a 30-day full refund at no extra cost.",
)
print(f"groundedness = {result.groundedness:.2f}   iqs = {result.iqs:.2f}   flags = {result.flags}")
""")

# --- Cell 3: basic ContextBuilder -------------------------------------------
md("""
## 3. Basic `ContextBuilder` - query + retrieval

The 4-line integration. Add the query first, then drop retrieved chunks in
right after your retrieval step. `build()` assembles everything into a
`ContextPayload` that `auditor.score()` consumes.
""")
code("""
user_query = "What is the warranty period for the X200 laptop?"

# --- your existing retrieval step (simulated here) ---
retrieved_chunks = [
    "The X200 laptop carries a 24-month manufacturer warranty covering parts and labour.",
    "Warranty claims for the X200 require the original proof of purchase.",
    "The X150 tablet carries a 12-month warranty.",
]

ctx = ContextBuilder()
ctx.add_query(user_query)
ctx.add_retrieved(retrieved_chunks)

# --- your existing LLM call (simulated here) ---
llm_response = "The X200 laptop comes with a 24-month warranty on parts and labour."

result = auditor.score(user_query, llm_response, context=ctx.build())
print(f"groundedness = {result.groundedness:.2f}   iqs = {result.iqs:.2f}   flags = {result.flags}")
""")

# --- Cell 4: full RAG pipeline ----------------------------------------------
md("""
## 4. Full RAG pipeline - retrieval + reranking

Each `add_*` method labels its content with a **source weight** used at
assembly time:

| Source | Weight | Why |
|---|---|---|
| `add_reranked()` | 1.00 | what the LLM actually used |
| `add_retrieved()` | 0.85 | raw retrieval signal |
| `add_tool_output()` | 0.70 | facts from tools |
| `add_system_prompt()` | 0.50 | instructions, not facts |
| `add_query()` | 0.30 | trace context |

When the `max_tokens` budget forces truncation, the **highest-weight sources
survive**. Methods chain, so the integration stays compact.
""")
code("""
ctx = ContextBuilder()
ctx.add_query(user_query) \\
   .add_retrieved(retrieved_chunks, source="vector_store") \\
   .add_reranked(retrieved_chunks[:2]) \\
   .add_system_prompt("Answer using only the provided documentation.")

# snapshot() shows the state mid-pipeline without building (no content text)
snap = ctx.snapshot()
print("sources:", snap["sources"])
print("entries:", snap["total_entries"], "  est. tokens:", snap["total_tokens"])

payload = ctx.build()
result = auditor.score(user_query, llm_response, context=payload)
print(f"\\ngroundedness = {result.groundedness:.2f}   iqs = {result.iqs:.2f}")
""")

# --- Cell 5: agentic pipeline -----------------------------------------------
md("""
## 5. Agentic pipeline - multiple tool outputs

In an agent loop, grounding facts arrive as **tool outputs** (database rows,
API responses, calculator results). Record each one with its tool name —
the name flows into the audit trail and the dashboard's provenance display.
""")
code("""
task = "How many open support tickets does ACME Corp have, and what is their plan?"

ctx = ContextBuilder(session_id="agent-trace-0042")
ctx.add_query(task)

# --- simulated agent steps ---
ctx.add_tool_output(
    "Ticket database query result: ACME Corp has 7 open support tickets.",
    tool_name="sql_query",
)
ctx.add_tool_output(
    "CRM lookup result: ACME Corp is on the Enterprise plan with 250 seats.",
    tool_name="crm_lookup",
)

final_answer = "ACME Corp has 7 open support tickets. They are on the Enterprise plan."

result = auditor.score(task, final_answer, context=ctx.build())
print(f"groundedness = {result.groundedness:.2f}   iqs = {result.iqs:.2f}   flags = {result.flags}")
print("trace:", result.details["context"]["session_id"])
""")

# --- Cell 6: PII scrubbing demo ----------------------------------------------
md("""
## 6. PII scrubbing - on by default

Every addition is scrubbed **before** it is stored, entirely locally (regex,
no external API call). Nine entity types are replaced with typed placeholders;
the originals are never retained anywhere. The scrub summary records **counts
only** - safe for audit logs.

The synthetic PII below never leaves this cell unredacted.
""")
code("""
dirty_chunk = (
    "Customer John Smith (john.smith@acme.com, +1-555-867-5309, SSN 123-45-6789) "
    "reported the issue from 192.168.1.50. "
    "Payment card 4111-1111-1111-1111. Support key sk-abcdefghij1234567890abcd."
)

ctx = ContextBuilder()
ctx.add_retrieved([dirty_chunk])
payload = ctx.build()

print("BEFORE:", dirty_chunk[:80], "...")
print()
print("AFTER: ", payload.assembled_text)
print()
print("scrub summary (counts only, no original values):")
for entity, count in payload.scrub_summary.items():
    if count:
        print(f"  {entity:>24}: {count}")
""")

# --- Cell 7: ContextPayload fields -------------------------------------------
md("""
## 7. What `build()` returns - the `ContextPayload`

The payload carries the assembled text plus a full audit trail: provenance,
token accounting, truncation flag, PII summary, and a SHA-256 checksum of the
assembled text for integrity verification. Note what it does **not** carry:
the raw pre-scrub additions.
""")
code("""
ctx = ContextBuilder(max_tokens=4096)
ctx.add_query("What is the refund policy?")
ctx.add_retrieved(["Refunds are available within 30 days of purchase.",
                   "Refunds are processed to the original payment method."])
payload = ctx.build()

print(f"session_id    : {payload.session_id}")
print(f"total_tokens  : {payload.total_tokens}")
print(f"was_truncated : {payload.was_truncated}")
print(f"pii_scrubbed  : {payload.pii_scrubbed}")
print(f"built_at      : {payload.built_at.isoformat()}")
print(f"checksum      : {payload.checksum[:30]}...")
print(f"sources       : {[ (e.source, round(e.source_weight, 2)) for e in payload.sources ]}")
""")

# --- Cell 8: graceful degradation --------------------------------------------
md("""
## 8. Graceful degradation - `build()` with no content

Forgot to add anything? `build()` returns `None` with a warning - **not a
crash** - and groundedness scores as `None` while the other four metrics
still work. Partial integration always beats no integration.

`build()` also **seals** the builder: adding content afterwards raises
`ContextSealedError`, which keeps the payload checksum a final audit boundary.
""")
code("""
from scroot.exceptions import ContextEmptyWarning, ContextSealedError

empty_ctx = ContextBuilder()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    payload = empty_ctx.build()
print("payload:", payload)
print("warning:", caught[0].category.__name__)

result = auditor.score("What is the refund policy?",
                       "We offer a 30-day refund.", context=payload)
print(f"groundedness = {result.groundedness}   (other metrics still score: iqs = {result.iqs:.2f})")

try:
    empty_ctx.add_retrieved(["too late"])
except ContextSealedError as e:
    print("sealed:", e)
""")

# --- Cell 9: LangChain integration --------------------------------------------
md("""
## 9. LangChain integration

`add_retrieved()` understands LangChain `Document` objects directly —
`.page_content` is extracted automatically (the same goes for ChromaDB query
results, LlamaIndex nodes, Pinecone scored vectors, and dicts with a `text`
key). The mock retriever below has the same interface as a real LangChain
retriever - swap it for yours.
""")
code("""
class MockDocument:
    \"\"\"Stands in for langchain_core.documents.Document.\"\"\"
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MockRetriever:
    \"\"\"Same interface as a LangChain retriever - swap for the real thing.\"\"\"
    def get_relevant_documents(self, query):
        return [
            MockDocument("Standard shipping takes 3-5 business days within the US.",
                         {"source": "shipping_faq.md"}),
            MockDocument("Express shipping (1-2 business days) is available for $15.",
                         {"source": "shipping_faq.md"}),
        ]

question = "How long does standard shipping take?"
retriever = MockRetriever()

ctx = ContextBuilder()
ctx.add_query(question)
ctx.add_retrieved(retriever.get_relevant_documents(question))   # Documents, not strings

answer = "Standard shipping takes 3 to 5 business days within the US."
result = auditor.score(question, answer, context=ctx.build())
print(f"groundedness = {result.groundedness:.2f}   iqs = {result.iqs:.2f}")
""")

# --- Cell 10: with vs without context ----------------------------------------
md("""
## 10. The payoff - scoring with vs. without context

The same query/response pair, scored both ways. Without context, groundedness
is `None` and a hallucination sails through unflagged. With context, the
fabricated response is caught immediately.

*This is the cell to screenshot.*
""")
code("""
query = "What is our return policy?"
context_docs = [
    "All customers are eligible for a 30-day full refund at no extra cost. "
    "Items must be returned within 30 days of the original purchase date."
]

faithful     = ("You can return any item within 30 days of purchase "
                "for a full refund at no extra cost.")
hallucinated = ("We offer a 90-day money-back guarantee with free worldwide "
                "return shipping and a lifetime warranty on all products.")

rows = []
for label, response in [("faithful", faithful), ("hallucinated", hallucinated)]:
    # WITHOUT context - groundedness is blind
    no_ctx = auditor.score(query, response)

    # WITH context via ContextBuilder
    ctx = ContextBuilder()
    ctx.add_query(query)
    ctx.add_retrieved(context_docs)
    with_ctx = auditor.score(query, response, context=ctx.build())

    rows.append((label, no_ctx.groundedness, no_ctx.iqs,
                 with_ctx.groundedness, with_ctx.iqs, with_ctx.flags))

print(f"{'response':<14}{'g/ness (no ctx)':>16}{'iqs (no ctx)':>14}{'g/ness (ctx)':>14}{'iqs (ctx)':>11}   flags")
print("-" * 92)
for label, g0, i0, g1, i1, flags in rows:
    g0s = "None" if g0 is None else f"{g0:.2f}"
    print(f"{label:<14}{g0s:>16}{i0:>14.2f}{g1:>14.2f}{i1:>11.2f}   {flags}")
""")

md("""
---

### Where to go next

- **API reference**: [`docs/context_builder.md`](../docs/context_builder.md) —
  constructor options, accepted chunk types, size limits, SOC II data-flow table
- **Audit logging**: `scroot.configure_audit_log(destination="file",
  path="~/.scroot/audit.jsonl", retention_days=90)` - structured,
  content-free events for every context operation
- **Review Console**: `scroot serve` - `session_id` and `context_checksum`
  now flow into the review queue for trace reconstruction
- **Full feature tour**: [`scroot_interactive_demo.ipynb`](scroot_interactive_demo.ipynb)
""")

# ---------------------------------------------------------------------------
nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out_path = os.path.join(os.path.dirname(__file__), "context_builder.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path} with {len(cells)} cells")
