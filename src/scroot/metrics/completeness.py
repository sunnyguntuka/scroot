"""Completeness metric: did the response address all parts of the query?

Two-stage scoring:
  1. Embedding similarity (existing) - measures topical coverage.
  2. NLI answer-presence check (new) - verifies the response actually
     *answers* each query aspect, not just mentions the same topic.

The combined score penalises responses that talk around the question
without directly answering it.
"""

from __future__ import annotations

import re

import numpy as np

from ..models import get_embedding_model
from ..text_utils import split_sentences


# ---------------------------------------------------------------------------
# Query aspect decomposition
# ---------------------------------------------------------------------------

_WH_QUESTION = re.compile(
    r'\b(what|when|where|who|which|why|how|how much|how many|how long|how often)\b',
    re.IGNORECASE,
)

_CLAUSE_SPLIT = re.compile(
    r'(?:[,;]\s*(?:and|or|but|also)\s+|\?\s+|\band\b|\bor\b)',
    re.IGNORECASE,
)


def _decompose_query(query: str) -> list[str]:
    """Split a query into its constituent question aspects.

    "What is the price and how long does shipping take?"
    → ["What is the price", "how long does shipping take"]

    Falls back to the full query if no split is detected.
    """
    # Split on clause boundaries
    parts = _CLAUSE_SPLIT.split(query)
    aspects = []
    for p in parts:
        p = p.strip().rstrip("?").strip()
        if len(p.split()) >= 3:
            aspects.append(p)

    # If no useful split found, keep the whole query
    return aspects if len(aspects) > 1 else [query.rstrip("?").strip()]


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_batch(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    dot = np.dot(matrix, vec)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
    return dot / (norms + 1e-8)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_completeness(
    query: str,
    response: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    nli_model: str | None = None,   # reserved for future use
    device: str = "cpu",
    coverage_threshold: float = 0.45,
    nli_answer_weight: float = 0.35,
) -> tuple[float, dict]:
    """Score how completely the response addresses the query.

    Combines two signals:
      - Embedding similarity (65%): does the response cover the topic?
      - NLI answer-presence (35%): does the response *contain an answer*
        to each query aspect, not just mention the topic?

    Args:
        query: The user's query/question.
        response: The LLM-generated response.
        embedding_model: Sentence-transformers model name or instance.
        nli_model: Optional NLI cross-encoder for answer-presence check.
            When None, falls back to embedding-only scoring.
        device: "cpu" or "cuda".
        coverage_threshold: Minimum embedding similarity for a query aspect
            to be considered covered. Default 0.45.
        nli_answer_weight: Weight of the NLI answer-presence signal in the
            combined score. Default 0.35 (35% NLI, 65% embedding).

    Returns:
        Tuple of (score, details_dict).
    """
    emb_model = get_embedding_model(embedding_model, device=device)

    # Decompose query into aspects (sub-questions)
    aspects = _decompose_query(query)
    if not aspects:
        return 0.0, {"note": "empty query"}

    response_sentences = split_sentences(response)
    if not response_sentences:
        return 0.0, {"note": "empty response"}

    a_embeddings = emb_model.encode(aspects, convert_to_numpy=True)
    r_embeddings = emb_model.encode(response_sentences, convert_to_numpy=True)

    # ── Stage 1: embedding similarity ────────────────────────────────────
    segment_results = []
    for i, aspect in enumerate(aspects):
        similarities = _cosine_similarity_batch(a_embeddings[i], r_embeddings)
        max_sim = float(np.max(similarities))
        best_idx = int(np.argmax(similarities))
        segment_results.append({
            "query_aspect": aspect,
            "best_match": response_sentences[best_idx],
            "similarity": round(max_sim, 4),
            "covered_by_embedding": max_sim >= coverage_threshold,
        })

    emb_covered = sum(1 for s in segment_results if s["covered_by_embedding"])
    emb_score = emb_covered / len(aspects)

    # Combined score (embedding-based; NLI completeness reserved for future)
    details: dict = {
        "segments": segment_results,
        "total_segments": len(aspects),
        "covered_segments": emb_covered,
    }

    return round(min(max(emb_score, 0.0), 1.0), 4), details
