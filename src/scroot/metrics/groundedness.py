"""Groundedness metric: NLI-based faithfulness scoring with semantic retrieval.

Pipeline:
  1. Extract atomic claims from the response (compound sentences split).
  2. Embed all claims and all context chunks once.
  3. For each claim, retrieve the top-k most semantically similar chunks
     (semantic retrieval) - avoids diluting NLI with irrelevant context.
  4. Run NLI only on the retrieved top-k (chunk, claim) pairs.
  5. Bi-encoder similarity fallback when NLI confidence is uncertain (0.3-0.7)
     to catch paraphrases that exact NLI entailment misses.
"""

from __future__ import annotations

import re
import numpy as np

from ..models import get_nli_model
from ..text_utils import extract_atomic_claims, extract_claims
from ._utils import softmax

LABEL_CONTRADICTION = 0
LABEL_ENTAILMENT = 1
LABEL_NEUTRAL = 2

_CHUNK_TOKEN_WARN_THRESHOLD = 400
_UNCERTAIN_LOW = 0.30
_UNCERTAIN_HIGH = 0.70


def _cosine_batch(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single vector and a matrix of row vectors."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec) + 1e-8
    return np.dot(matrix, vec) / norms


def _top_k_indices(claim_emb: np.ndarray, chunk_embs: np.ndarray, k: int) -> list[int]:
    """Return indices of the k most similar chunks for a given claim."""
    sims = _cosine_batch(claim_emb, chunk_embs)
    k = min(k, len(sims))
    return list(np.argsort(sims)[::-1][:k])


def score_groundedness(
    response: str,
    context: list[str],
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    embedding_model: str | None = None,
    device: str = "cpu",
    entailment_threshold: float = 0.5,
    atomic_claims: bool = True,
    similarity_fallback: bool = True,
    similarity_threshold: float = 0.82,
    top_k_chunks: int = 3,
    top_k_premises: int | None = None,
    _capture: "dict | None" = None,
    backbone_scorer=None,
) -> tuple[float, dict]:
    """Score how well the response is grounded in the context.

    Args:
        response: The LLM-generated response text.
        context: List of source context strings. A plain string is
            automatically wrapped in a list.
        nli_model: HuggingFace NLI cross-encoder model name or instance.
        embedding_model: Sentence-transformers model for semantic retrieval
            and similarity fallback. Pass None to disable both.
        device: "cpu" or "cuda".
        entailment_threshold: Minimum entailment probability to consider a
            claim grounded. Default 0.5.
        atomic_claims: If True (default), split compound sentences into
            sub-claims before scoring.
        similarity_fallback: If True (default), use cosine similarity when
            NLI confidence is uncertain (0.3-0.7) to catch paraphrases.
        similarity_threshold: Cosine similarity threshold for paraphrase
            credit. Default 0.82.
        top_k_chunks: Number of most semantically similar context chunks to
            retrieve per claim before running NLI. Reduces noise from
            irrelevant chunks and focuses NLI on the most relevant context.
            Default 3. Set to None to use all chunks (original behaviour).

    Returns:
        Tuple of (score, details_dict).
    """
    if isinstance(context, str):
        context = [context]
    context = [str(c) for c in context if c is not None]

    # backbone_scorer, when provided, bypasses the standard NLI cross-encoder.
    # It must expose score_pairs(pairs) -> list[float] returning P(supported)
    # directly (no softmax needed). Similarity fallback is skipped for
    # alternative backbones; it is tuned for the deberta uncertain-zone.
    _use_backbone = backbone_scorer is not None
    model = None if _use_backbone else get_nli_model(nli_model, device=device)
    claims = extract_atomic_claims(response) if atomic_claims else extract_claims(response)

    if not claims:
        return 1.0, {"claims": [], "note": "no claims detected"}

    chunky_chunks = [
        chunk[:60] + "..."
        for chunk in context
        if len(chunk) // 4 > _CHUNK_TOKEN_WARN_THRESHOLD
    ]

    claim_results = []

    if not context:
        for claim in claims:
            claim_results.append({
                "claim": claim,
                "grounded": False,
                "entailment_prob": 0.0,
                "contradiction_prob": 0.0,
            })
    else:
        # Load embedding model once if needed (retrieval or fallback)
        emb_model = None
        if embedding_model and (top_k_chunks is not None or similarity_fallback):
            from ..models import get_embedding_model
            emb_model = get_embedding_model(embedding_model, device=device)

        # Pre-encode chunks and claims once for retrieval + fallback
        chunk_embs: np.ndarray | None = None
        claim_embs: np.ndarray | None = None
        if emb_model is not None:
            chunk_embs = emb_model.encode(context, convert_to_numpy=True)
            claim_embs = emb_model.encode(claims, convert_to_numpy=True)

        for c_idx, claim in enumerate(claims):
            # --- Semantic retrieval: pick top-k most relevant chunks ---
            if (chunk_embs is not None
                    and claim_embs is not None
                    and top_k_chunks is not None
                    and len(context) > top_k_chunks):
                indices = _top_k_indices(claim_embs[c_idx], chunk_embs, top_k_chunks)
                selected_chunks = [context[i] for i in indices]
                selected_chunk_embs = chunk_embs[indices]
            else:
                selected_chunks = context
                indices = list(range(len(context)))
                selected_chunk_embs = chunk_embs if chunk_embs is not None else None

            # --- NLI on selected chunks ---
            # Sentence-split each chunk so the NLI cross-encoder receives
            # single-sentence premises. The model degrades to NEUTRAL on
            # long multi-sentence paragraphs even for verbatim content.
            nli_pairs: list[tuple[str, str]] = []
            pair_chunk_idx: list[int] = []  # maps each pair back to its chunk
            for ci, chunk in enumerate(selected_chunks):
                sents = re.split(r'(?<=[.!?])\s+', chunk.strip())
                sents = [s.strip() for s in sents if len(s.split()) >= 4]
                if not sents:
                    sents = [chunk]
                for s in sents:
                    nli_pairs.append((s, claim))
                    pair_chunk_idx.append(ci)

            # --- Premise pre-filtering: keep only the top-k premise sentences
            # most semantically similar to THIS claim before running the NLI
            # cross-encoder. top_k_chunks bounds retrieval at the chunk level;
            # a single retained chunk can still sentence-split into many
            # premises, so NLI cost grows with total sentence count. Ranking
            # premises by claim-similarity and keeping the top-k caps the NLI
            # batch size per claim, cutting latency on large contexts while
            # retaining the premises most likely to entail the claim. Requires
            # an embedding model; no-op when k is None, k >= len(pairs), or no
            # embedder is available.
            if (top_k_premises is not None
                    and emb_model is not None
                    and claim_embs is not None
                    and len(nli_pairs) > top_k_premises):
                premise_texts = [p[0] for p in nli_pairs]
                premise_embs = emb_model.encode(premise_texts,
                                                convert_to_numpy=True)
                sims = _cosine_batch(claim_embs[c_idx], premise_embs)
                keep = sorted(
                    np.argsort(sims)[::-1][:top_k_premises].tolist()
                )
                nli_pairs = [nli_pairs[j] for j in keep]
                pair_chunk_idx = [pair_chunk_idx[j] for j in keep]

            if _use_backbone:
                support_probs = backbone_scorer.score_pairs(nli_pairs)
                best_entailment = max(support_probs) if support_probs else 0.0
                best_contradiction = 0.0
                best_similarity = 0.0
            else:
                raw_scores = model.predict(nli_pairs)

                best_entailment = 0.0
                best_contradiction = 0.0
                best_similarity = 0.0

                for pair_idx, score_row in enumerate(raw_scores):
                    probs = softmax(score_row)
                    ep = float(probs[LABEL_ENTAILMENT])
                    cp = float(probs[LABEL_CONTRADICTION])

                    # Bi-encoder similarity fallback in uncertain zone
                    chunk_idx = pair_chunk_idx[pair_idx]
                    if (emb_model is not None
                            and claim_embs is not None
                            and selected_chunk_embs is not None
                            and similarity_fallback
                            and _UNCERTAIN_LOW < ep < _UNCERTAIN_HIGH):
                        sim = float(
                            _cosine_batch(
                                claim_embs[c_idx],
                                selected_chunk_embs[chunk_idx:chunk_idx + 1]
                            )[0]
                        )
                        best_similarity = max(best_similarity, sim)
                        if sim >= similarity_threshold:
                            ep = max(ep, entailment_threshold + 0.01)

                    if ep > best_entailment:
                        best_entailment = ep
                        best_contradiction = cp

            grounded = best_entailment >= entailment_threshold
            result: dict = {
                "claim": claim,
                "grounded": grounded,
                "entailment_prob": round(best_entailment, 4),
                "contradiction_prob": round(best_contradiction, 4),
            }
            if best_similarity > 0:
                result["similarity"] = round(best_similarity, 4)
            claim_results.append(result)

    if _capture is not None:
        _capture["groundedness_claims"] = [c["claim"] for c in claim_results]
        _capture["groundedness_scores"] = [c["entailment_prob"] for c in claim_results]

    grounded_count = sum(1 for c in claim_results if c["grounded"])
    groundedness_score = grounded_count / len(claims)

    details: dict = {
        "claims": claim_results,
        "total_claims": len(claims),
        "grounded_claims": grounded_count,
    }
    if chunky_chunks:
        details["truncation_warning"] = (
            f"{len(chunky_chunks)} context chunk(s) exceed ~400 estimated tokens. "
            f"Affected chunk prefixes: {chunky_chunks}"
        )

    return groundedness_score, details
