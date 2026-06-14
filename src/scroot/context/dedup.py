"""Chunk deduplication for context assembly.

Near-identical chunks frequently appear when the same document is picked
up by multiple retrieval steps (raw retrieval + reranking, or two vector
stores indexing the same corpus). Scoring duplicate text wastes the
token budget and skews groundedness weighting, so build() merges them.

Similarity backend: cosine similarity over sentence-transformers
embeddings when available (the model is shared with the Auditor's cache),
falling back to ``difflib.SequenceMatcher`` ratio when
sentence-transformers is not installed. Both use the same threshold.
"""

from __future__ import annotations

from .payload import ContextEntry


def _exact_key(text: str) -> str:
    return " ".join(text.lower().split())


def _similarity_matrix(texts: list[str], embedding_model: str, device: str):
    """Pairwise cosine similarity via embeddings, or None if unavailable."""
    try:
        import numpy as np
        from ..models import get_embedding_model
        model = get_embedding_model(embedding_model, device=device)
        embs = model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        normalised = embs / norms
        return normalised @ normalised.T
    except Exception:
        return None


def _fallback_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def deduplicate(
    entries: list[ContextEntry],
    threshold: float = 0.92,
    embedding_model: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> list[ContextEntry]:
    """Remove near-duplicate entries, keeping the first occurrence.

    Entries should be pre-sorted by source weight descending so the most
    authoritative copy of duplicated content survives.

    Args:
        entries: Context entries to deduplicate.
        threshold: Cosine similarity (or fallback ratio) at or above
            which two entries are considered duplicates. Default 0.92.
        embedding_model: Sentence-transformers model name for the
            embedding backend. Shares the Auditor's model cache.
        device: "cpu" or "cuda".

    Returns:
        Entries with duplicates removed, original order preserved.
    """
    if len(entries) <= 1:
        return list(entries)

    # Pass 1: exact duplicates after whitespace/case normalisation.
    seen: set[str] = set()
    unique: list[ContextEntry] = []
    for entry in entries:
        key = _exact_key(entry.content)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)

    if len(unique) <= 1:
        return unique

    # Pass 2: near-duplicates by similarity.
    texts = [e.content for e in unique]
    matrix = _similarity_matrix(texts, embedding_model, device)

    kept: list[ContextEntry] = []
    kept_idx: list[int] = []
    for i, entry in enumerate(unique):
        is_dup = False
        for j in kept_idx:
            if matrix is not None:
                sim = float(matrix[i][j])
            else:
                sim = _fallback_similarity(texts[i], texts[j])
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(entry)
            kept_idx.append(i)
    return kept
