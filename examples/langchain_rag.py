"""LangChain RAG integration: score an existing chain's output with scroot.

This shows the pattern for adding scroot to a LangChain RAG pipeline
with minimal changes to existing code:

1. A `BaseCallbackHandler` captures the documents returned by the
   retriever and the LLM's final response as the chain runs.
2. After the chain finishes, the captured documents are fed to
   `ContextBuilder` and scored with `auditor.score()`.

The retriever and LLM below are toy in-memory/fake implementations so
this example runs with no external services or API keys. In a real
chain, swap in your actual retriever (e.g. a vector store) and LLM --
the callback handler doesn't need to change.

Requires: pip install langchain-core
"""

from __future__ import annotations

from typing import Any

from scroot import Auditor, ContextBuilder

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.documents import Document
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example requires langchain-core: pip install langchain-core"
    ) from exc


class InMemoryRetriever(BaseRetriever):
    """Toy retriever that always returns the same documents."""

    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.documents


class ScrootCallbackHandler(BaseCallbackHandler):
    """Captures retrieved context and the LLM response for scroot scoring."""

    def __init__(self, query: str) -> None:
        self.ctx = ContextBuilder()
        self.ctx.add_query(query)
        self.response: str | None = None

    def on_retriever_end(self, documents: list[Document], **kwargs: Any) -> None:
        self.ctx.add_retrieved([d.page_content for d in documents], source="retriever")

    def on_llm_end(self, response, **kwargs: Any) -> None:
        self.response = response.generations[0][0].text


# --- A toy RAG chain (replace with your real retriever + LLM) ---
docs = [
    Document(page_content=(
        "All customers are eligible for a 30-day full refund at no extra cost."
    )),
    Document(page_content="Refund requests must be submitted via the support portal."),
]
retriever = InMemoryRetriever(documents=docs)
llm = FakeListLLM(responses=["We offer a 30-day full refund at no extra cost."])

query = "What is our refund policy?"

# --- Run the chain with the scroot callback handler attached ---
handler = ScrootCallbackHandler(query)
retriever.invoke(query, config={"callbacks": [handler]})
llm.invoke(query, config={"callbacks": [handler]})

# --- Score the chain's output with scroot ---
auditor = Auditor()
result = auditor.score(
    query=query,
    response=handler.response,
    context=handler.ctx.build(),
)

print(f"Response:     {handler.response}")
print(f"IQS:          {result.iqs:.2f}")
print(f"Groundedness: {result.groundedness:.2f}")
print(f"Completeness: {result.completeness:.2f}")
print(f"Flags:        {result.flags}")
