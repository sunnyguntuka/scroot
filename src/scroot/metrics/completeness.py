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
from ._utils import softmax


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
    nli_model: str | None = None,
    device: str = "cpu",
    coverage_threshold: float = 0.45,
    nli_answer_weight: float = 0.35,
) -> tuple[float, dict]:
    """Score how completely the response addresses the query.

    Two-stage scoring:
      1. Embedding similarity (primary fallback or weight when NLI absent):
         does the response cover the same topic as each query aspect?
      2. NLI answer-presence (primary when ``nli_model`` is set):
         does the response *entail an answer* to each aspect?
         NLI as primary catches paraphrased/implicit coverage that pure
         embedding similarity misses.

    Combined: ``score = nli_weight * nli_score + (1-nli_weight) * emb_score``
    when NLI is available; pure embedding otherwise.

    Args:
        query: The user's query/question.
        response: The LLM-generated response.
        embedding_model: Sentence-transformers model name or instance.
        nli_model: Optional NLI cross-encoder. When provided, each query
            aspect is checked via entailment across all response sentences;
            the highest-entailment sentence counts as covering the aspect.
            When ``None``, falls back to embedding-only scoring.
        device: ``"cpu"`` or ``"cuda"``.
        coverage_threshold: Minimum embedding similarity for a query aspect
            to be considered covered by the embedding path. Default 0.45.
        nli_answer_weight: Weight of the NLI answer-presence signal in the
            combined score. Default 0.35.

    Returns:
        Tuple of ``(score, details_dict)``.
    """
    emb_model = get_embedding_model(embedding_model, device=device)

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

    # ── Stage 2: NLI answer-presence ─────────────────────────────────────
    if nli_model is not None:
        from ..models import get_nli_model
        nli_m = get_nli_model(nli_model, device=device)

        nli_covered = 0
        for i, seg in enumerate(segment_results):
            aspect = seg["query_aspect"]
            # Build (aspect, response_sentence) pairs; NLI checks entailment
            pairs = [(aspect, sent) for sent in response_sentences]
            raw_scores = nli_m.predict(pairs)
            best_entail = 0.0
            best_sent = response_sentences[0]
            for j, raw_s in enumerate(raw_scores):
                probs = softmax(raw_s)
                entail_p = float(probs[2])  # label 2 = entailment
                if entail_p > best_entail:
                    best_entail = entail_p
                    best_sent = response_sentences[j]
            covered_by_nli = best_entail >= 0.5
            if covered_by_nli:
                nli_covered += 1
            segment_results[i].update({
                "nli_entailment": round(best_entail, 4),
                "nli_best_sentence": best_sent,
                "covered_by_nli": covered_by_nli,
            })

        nli_score = nli_covered / len(aspects)
        combined = nli_answer_weight * nli_score + (1.0 - nli_answer_weight) * emb_score
        nli_cov_count = nli_covered
    else:
        combined = emb_score
        nli_score = None
        nli_cov_count = None

    details: dict = {
        "segments": segment_results,
        "total_segments": len(aspects),
        "covered_segments": emb_covered,
        "nli_covered_segments": nli_cov_count,
        "embedding_score": round(emb_score, 4),
        "nli_score": round(nli_score, 4) if nli_score is not None else None,
        "nli_active": nli_model is not None,
    }

    return round(min(max(combined, 0.0), 1.0), 4), details
