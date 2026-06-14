"""Evidence map: sentence-level NLI attribution against retrieved context.

For each sentence (atomic claim) in a response, finds the context chunk that
best supports or contradicts it, and classifies the sentence as supported,
contradicted, or ungrounded. Mirrors the retrieval + NLI pipeline used by
:func:`~scroot.metrics.groundedness.score_groundedness`, but reports
per-sentence detail intended for the Review Console's Evidence Map panel.
"""

from __future__ import annotations

from dataclasses import dataclass

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
