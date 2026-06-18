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
            # Milvus hit dict (pymilvus returns dicts with an 'entity' sub-dict)
            entity = chunk.get('entity')
            if isinstance(entity, dict):
                for key in _TEXT_KEYS:
                    if isinstance(entity.get(key), str):
                        return entity[key]
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

        # Weaviate v4 Object (.properties dict with known text keys)
        # Soft-import guard: check type name to avoid importing weaviate.
        chunk_type = type(chunk).__name__
        if chunk_type == 'Object':
            properties = getattr(chunk, 'properties', None)
            if isinstance(properties, dict):
                for key in _TEXT_KEYS:
                    if isinstance(properties.get(key), str):
                        return properties[key]

        # Qdrant ScoredPoint (.payload dict with known text keys)
        if chunk_type == 'ScoredPoint':
            payload = getattr(chunk, 'payload', None)
            if isinstance(payload, dict):
                for key in _TEXT_KEYS:
                    if isinstance(payload.get(key), str):
                        return payload[key]

        return None
    except Exception:
        return None
