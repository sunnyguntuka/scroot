"""Evidence map: sentence-level NLI attribution against retrieved context.

For each sentence (atomic claim) in a response, finds the context chunk that
best supports or contradicts it, and classifies the sentence as supported,
contradicted, or ungrounded. Mirrors the retrieval + NLI pipeline used by
:func:`~scroot.metrics.groundedness.score_groundedness`, but reports
per-sentence detail intended for the Review Console's Evidence Map panel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .composite import compute_iqs_detailed
from .metrics._utils import softmax
from .metrics.groundedness import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    _top_k_indices,
)
from .models import get_embedding_model, get_nli_model
from .text_utils import extract_atomic_claims, extract_claims


@dataclass
class EvidenceEntry:
    """Attribution of a single response sentence to its best-matching context chunk."""

    response_sentence: str
    best_matching_chunk: "str | None"
    entailment_score: "float | None"
    supported: bool
    contradiction_detected: bool = False
    no_grounding_found: bool = False
    chunk_source: "str | None" = None
    chunk_index: "int | None" = None
    # Per-sentence quality signal (R1). None when inputs are unavailable.
    # Excludes completeness and confidence (response-level only).
    mini_iqs: "float | None" = None
    mini_dims: "dict | None" = None


@dataclass
class EvidenceMap:
    """Sentence-level evidence attribution for an entire response."""

    entries: "list[EvidenceEntry]"
    supported_count: int
    unsupported_count: int
    contradiction_count: int
    coverage_ratio: float
    weakest_sentence: "str | None"

    def to_dict(self) -> dict:
        """Convert to plain dict for logging/serialization."""
        return {
            "supported": self.supported_count,
            "unsupported": self.unsupported_count,
            "contradictions": self.contradiction_count,
            "coverage_ratio": round(self.coverage_ratio, 3),
            "weakest_sentence": self.weakest_sentence,
            "entries": [vars(e) for e in self.entries],
        }


def _per_sentence_consistency(
    consistency_capture: dict,
) -> "dict[int, float]":
    """Aggregate pairwise NLI logits into a per-sentence consistency score.

    Returns a dict mapping consistency_sentence_index → score in [0, 1],
    where 1.0 means no contradictions involving that sentence.
    """
    cons_pairs = consistency_capture.get("consistency_pairs", [])
    cons_raw = consistency_capture.get("consistency_raw_scores", [])
    pair_cp: "dict[int, list[float]]" = {}
    for pair_idx, (i, j) in enumerate(cons_pairs):
        raw = cons_raw[pair_idx]
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            fwd_p = softmax(raw[0])
            bwd_p = softmax(raw[1])
            cp = max(float(fwd_p[LABEL_CONTRADICTION]), float(bwd_p[LABEL_CONTRADICTION]))
        else:
            probs = softmax(raw)
            cp = float(probs[LABEL_CONTRADICTION])
        pair_cp.setdefault(i, []).append(cp)
        pair_cp.setdefault(j, []).append(cp)
    return {idx: 1.0 - (sum(cps) / len(cps)) for idx, cps in pair_cp.items()}


def _find_consistency_idx(sentence: str, cons_sentences: "list[str]") -> "int | None":
    """Find the index of the best-matching consistency sentence."""
    for idx, cs in enumerate(cons_sentences):
        if sentence == cs or sentence in cs or cs in sentence:
            return idx
    return None


def _compute_mini_iqs(
    entries: "list[EvidenceEntry]",
    sentences: "list[str]",
    sentence_embs: "np.ndarray | None",
    query_embedding: "np.ndarray | None",
    consistency_capture: "dict | None",
    sigmoid_midpoint: float,
    sigmoid_steepness: float,
) -> None:
    """Populate mini_iqs and mini_dims on each EvidenceEntry in-place.

    Uses pre-computed embeddings and consistency NLI logits; no extra model
    calls. Completeness and confidence are response-level and excluded here.
    """
    import numpy as np

    # Build per-sentence consistency scores from capture if available
    per_cons: "dict[int, float]" = {}
    cons_sentences: "list[str]" = []
    if consistency_capture:
        cons_sentences = consistency_capture.get("consistency_sentences", [])
        per_cons = _per_sentence_consistency(consistency_capture)

    for s_idx, entry in enumerate(entries):
        dims: "dict[str, float]" = {}

        # groundedness_i from existing entailment_score
        if entry.entailment_score is not None:
            dims["groundedness"] = float(entry.entailment_score)

        # relevance_i = sigmoid(cosine(query_emb, sentence_embs[s_idx]))
        if (
            query_embedding is not None
            and sentence_embs is not None
            and s_idx < len(sentence_embs)
        ):
            q = query_embedding
            s = sentence_embs[s_idx]
            denom = float(np.linalg.norm(q) * np.linalg.norm(s)) + 1e-8
            cos_sim = float(np.dot(q, s)) / denom
            dims["relevance"] = 1.0 / (1.0 + math.exp(
                -sigmoid_steepness * (cos_sim - sigmoid_midpoint)
            ))

        # consistency_i from in-process NLI capture via text alignment
        if cons_sentences:
            c_idx = _find_consistency_idx(entry.response_sentence, cons_sentences)
            if c_idx is not None and c_idx in per_cons:
                dims["consistency"] = per_cons[c_idx]

        if not dims:
            continue

        # mini_iqs = weighted harmonic mean over present dims
        mini_iqs, _ = compute_iqs_detailed(dims)
        entry.mini_iqs = round(mini_iqs, 4)
        entry.mini_dims = {k: round(v, 4) for k, v in dims.items()}


def _no_grounding_map(sentences: "list[str]") -> EvidenceMap:
    entries = [
        EvidenceEntry(
            response_sentence=sentence,
            best_matching_chunk=None,
            entailment_score=None,
            supported=False,
            no_grounding_found=True,
        )
        for sentence in sentences
    ]
    return EvidenceMap(
        entries=entries,
        supported_count=0,
        unsupported_count=len(entries),
        contradiction_count=0,
        coverage_ratio=0.0,
        weakest_sentence=sentences[0] if sentences else None,
    )


def build_evidence_map(
    response: str,
    context: "list[str]",
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    embedding_model: "str | None" = "all-MiniLM-L6-v2",
    device: str = "cpu",
    entailment_threshold: float = 0.70,
    contradiction_threshold: float = 0.30,
    top_k_chunks: int = 3,
    chunk_sources: "list[str | None] | None" = None,
    atomic_claims: bool = True,
    query_embedding: "np.ndarray | None" = None,
    consistency_capture: "dict | None" = None,
    sigmoid_midpoint: float = 0.5,
    sigmoid_steepness: float = 10.0,
) -> EvidenceMap:
    """Build a sentence-level evidence map for a response against its context.

    For each sentence in ``response``, retrieves the most semantically
    similar context chunks, runs NLI to find the chunk with the highest
    entailment probability and the chunk with the highest contradiction
    probability, and classifies the sentence as supported (entailment >=
    ``entailment_threshold``), contradicted (contradiction >=
    ``contradiction_threshold``), or ungrounded (neither).

    Args:
        response: The LLM-generated response text.
        context: List of source context chunk strings.
        nli_model: HuggingFace NLI cross-encoder model name.
        embedding_model: Sentence-transformers model for top-k chunk
            retrieval. Pass ``None`` to score every sentence against all
            chunks (no retrieval pre-filter).
        device: ``"cpu"`` or ``"cuda"``.
        entailment_threshold: Minimum entailment probability for a sentence
            to be considered supported. Default 0.70.
        contradiction_threshold: Minimum contradiction probability for a
            sentence to be flagged as contradicted. Default 0.30.
        top_k_chunks: Number of most semantically similar chunks to examine
            per sentence. Default 3.
        chunk_sources: Optional per-chunk provenance labels (e.g.
            ``"retrieval"``, ``"reranker"``), aligned by index with
            ``context``. Used to populate ``EvidenceEntry.chunk_source``.
        atomic_claims: If True (default), split compound sentences into
            sub-claims via :func:`~scroot.text_utils.extract_atomic_claims`.

    Returns:
        :class:`EvidenceMap` with one :class:`EvidenceEntry` per sentence.
    """
    sentences = extract_atomic_claims(response) if atomic_claims else extract_claims(response)
    if not sentences:
        return EvidenceMap(
            entries=[],
            supported_count=0,
            unsupported_count=0,
            contradiction_count=0,
            coverage_ratio=0.0,
            weakest_sentence=None,
        )

    context = [str(c) for c in context if c is not None]
    if not context:
        return _no_grounding_map(sentences)

    model = get_nli_model(nli_model, device=device)

    emb_model = None
    chunk_embs = None
    sentence_embs = None
    if embedding_model:
        emb_model = get_embedding_model(embedding_model, device=device)
        chunk_embs = emb_model.encode(context, convert_to_numpy=True)
        sentence_embs = emb_model.encode(sentences, convert_to_numpy=True)

    entries: "list[EvidenceEntry]" = []
    for s_idx, sentence in enumerate(sentences):
        if (
            chunk_embs is not None
            and sentence_embs is not None
            and len(context) > top_k_chunks
        ):
            indices = _top_k_indices(sentence_embs[s_idx], chunk_embs, top_k_chunks)
        else:
            indices = list(range(len(context)))

        nli_pairs = [(context[i], sentence) for i in indices]
        raw_scores = model.predict(nli_pairs)

        best_entail = -1.0
        best_entail_pos = 0
        best_contra = -1.0
        best_contra_pos = 0
        for pos, score_row in enumerate(raw_scores):
            probs = softmax(score_row)
            ep = float(probs[LABEL_ENTAILMENT])
            cp = float(probs[LABEL_CONTRADICTION])
            if ep > best_entail:
                best_entail = ep
                best_entail_pos = pos
            if cp > best_contra:
                best_contra = cp
                best_contra_pos = pos

        if best_entail >= entailment_threshold:
            chosen_pos = best_entail_pos
            supported = True
            contradiction_detected = False
            no_grounding_found = False
        elif best_contra >= contradiction_threshold:
            chosen_pos = best_contra_pos
            supported = False
            contradiction_detected = True
            no_grounding_found = False
        else:
            chosen_pos = best_entail_pos
            supported = False
            contradiction_detected = False
            no_grounding_found = True

        chosen_idx = indices[chosen_pos]
        entries.append(
            EvidenceEntry(
                response_sentence=sentence,
                best_matching_chunk=context[chosen_idx],
                entailment_score=round(best_entail, 4),
                supported=supported,
                contradiction_detected=contradiction_detected,
                no_grounding_found=no_grounding_found,
                chunk_source=chunk_sources[chosen_idx] if chunk_sources else None,
                chunk_index=chosen_idx,
            )
        )

    # Per-sentence mini-IQS (R1). Reuses in-process embeddings and consistency
    # NLI logits; no additional model calls required.
    # Completeness and confidence are response-level → excluded per-sentence.
    _compute_mini_iqs(
        entries, sentences, sentence_embs, query_embedding,
        consistency_capture, sigmoid_midpoint, sigmoid_steepness,
    )

    supported_count = sum(1 for e in entries if e.supported)
    contradiction_count = sum(1 for e in entries if e.contradiction_detected)
    unsupported_count = len(entries) - supported_count - contradiction_count
    coverage_ratio = supported_count / len(entries)

    non_supported = [e for e in entries if not e.supported]
    weakest_sentence = None
    if non_supported:
        weakest = min(non_supported, key=lambda e: e.entailment_score or 0.0)
        weakest_sentence = weakest.response_sentence

    return EvidenceMap(
        entries=entries,
        supported_count=supported_count,
        unsupported_count=unsupported_count,
        contradiction_count=contradiction_count,
        coverage_ratio=round(coverage_ratio, 4),
        weakest_sentence=weakest_sentence,
    )
