"""Type adapters - extract plain text from common document/result types.

Adapters are intentionally defensive: they never raise, only return
None when extraction fails. The caller logs a warning and skips the
chunk so the client's pipeline is never crashed by an exotic type.
"""

from __future__ import annotations

from typing import Any

_TEXT_KEYS = ('text', 'content', 'page_content', 'body', 'chunk')


def extract_text(chunk: Any, source_hint: str = "") -> str | None:
    """Extract plain text from any common document type.

    Handles, in order: plain strings, LangChain ``Document``
    (``.page_content``), dicts with a known text key, LlamaIndex
    ``TextNode``/``NodeWithScore`` (``.text`` / ``.node.text``), ChromaDB
    ``QueryResult``-style objects (``.documents`` list of lists), and
    Pinecone ``ScoredVector`` (``.metadata['text']``).

    Args:
        chunk: The object to extract text from.
        source_hint: Optional label, reserved for diagnostics.

    Returns:
        The extracted text, or None if extraction fails (caller logs
        and skips - never raises).
    """
    try:
        if isinstance(chunk, str):
            return chunk

        # LangChain Document
        page_content = getattr(chunk, 'page_content', None)
        if isinstance(page_content, str):
            return page_content

        # Dict with known text keys
        if isinstance(chunk, dict):
            for key in _TEXT_KEYS:
                if key in chunk and isinstance(chunk[key], str):
                    return chunk[key]
            # ChromaDB-style dict result: {'documents': [[...]]}
            docs = chunk.get('documents')
            if isinstance(docs, list):
                flat = [
                    d
                    for sub in docs
                    for d in (sub if isinstance(sub, list) else [sub])
                    if isinstance(d, str)
                ]
                return '\n\n'.join(flat) if flat else None
            return None

        # LlamaIndex TextNode / NodeWithScore
        text = getattr(chunk, 'text', None)
        if isinstance(text, str):
            return text
        node = getattr(chunk, 'node', None)
        if node is not None:
            node_text = getattr(node, 'text', None)
            if isinstance(node_text, str):
                return node_text

        # ChromaDB QueryResult object (list of lists)
        documents = getattr(chunk, 'documents', None)
        if isinstance(documents, list):
            flat = [
                d
                for sub in documents
                for d in (sub if isinstance(sub, list) else [sub])
                if isinstance(d, str)
            ]
            return '\n\n'.join(flat) if flat else None

        # Pinecone ScoredVector with metadata
        metadata = getattr(chunk, 'metadata', None)
        if isinstance(metadata, dict) and isinstance(metadata.get('text'), str):
            return metadata['text']

        return None
    except Exception:
        return None
