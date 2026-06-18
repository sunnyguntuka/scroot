"""Relevance metric: is the response actually about what was asked?

Simple but important -detects off-topic responses, topic drift, or
responses that are factual but don't address the actual question.
"""

import math
import numpy as np
from ..models import get_embedding_model


def score_relevance(
    query: str,
    response: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    midpoint: float = 0.5,
    steepness: float = 10.0,
) -> tuple[float, dict]:
    """Score the semantic relevance of response to query.

    Uses cosine similarity between query and response embeddings, then
    applies sigmoid scaling to map raw similarity to a 0-1 score that
    better reflects human perception of relevance.

    Args:
        query: The user's query.
        response: The LLM response.
        embedding_model: Sentence-transformers model name or pre-instantiated
            SentenceTransformer instance.
        device: ``"cpu"`` or ``"cuda"``.
        midpoint: Cosine similarity value that maps to 0.5 relevance.
            Default 0.5. Override for retrievers with a higher baseline
            similarity (e.g. 0.7 for dense retrievers on same-domain data).
        steepness: Controls how sharply the sigmoid rises. Default 10.0.
            Higher values → sharper transition around the midpoint.

    Returns:
        Tuple of ``(score, details_dict)``.
        score: float 0–1.
        details_dict: raw_cosine_similarity, scaled_score, midpoint, steepness.
    """
    if not query.strip() or not response.strip():
        return 0.0, {"note": "empty query or response"}

    model = get_embedding_model(embedding_model, device=device)

    q_emb = model.encode(query, convert_to_numpy=True)
    r_emb = model.encode(response, convert_to_numpy=True)

    raw_similarity = float(np.dot(q_emb, r_emb) / (
        np.linalg.norm(q_emb) * np.linalg.norm(r_emb) + 1e-8
    ))

    scaled_score = _scale_similarity(raw_similarity, midpoint=midpoint, steepness=steepness)

    details = {
        "raw_cosine_similarity": raw_similarity,
        "scaled_score": scaled_score,
        "sigmoid_midpoint": midpoint,
        "sigmoid_steepness": steepness,
    }

    return scaled_score, details


def _scale_similarity(
    sim: float,
    midpoint: float = 0.5,
    steepness: float = 10.0,
) -> float:
    """Scale raw cosine similarity to a perceptually-aligned 0-1 score.

    Args:
        sim: Raw cosine similarity value.
        midpoint: Similarity value that maps to 0.5 output. Default 0.5.
        steepness: Controls how sharply the sigmoid rises. Default 10.0.

    Returns:
        float 0-1.
    """
    return 1.0 / (1.0 + math.exp(-steepness * (sim - midpoint)))
