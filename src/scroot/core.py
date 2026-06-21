"""Auditor: main orchestrator class.

Loads models once, runs all metrics, computes IQS, returns result.
"""

from __future__ import annotations

import logging
import os
import time
import warnings

from .result import EntailmentResult
from .context.payload import ContextPayload
from .evidence import build_evidence_map
from .exceptions import GroundednessComputationError, NoContextWarning
from .metrics.groundedness import score_groundedness
from .metrics.completeness import score_completeness
from .metrics.relevance import score_relevance
from .metrics.consistency import score_consistency
from .metrics.confidence import score_confidence
from .metrics.numeric_groundedness import score_numeric_groundedness
from .composite import DEFAULT_WEIGHTS, compute_iqs_detailed
from .flags import detect_flags
from .models import get_embedding_model
from .text_utils import split_sentences

logger = logging.getLogger(__name__)


class Auditor:
    """LLM-free response quality scorer.

    Scores LLM responses using NLI models and embedding similarity.
    No LLM API calls required. Runs locally, deterministic, fast.

    Args:
        nli_model: NLI cross-encoder model name or pre-instantiated instance.
            Upgrade to ``cross-encoder/nli-deberta-v3-large`` for ~4% better
            accuracy at the cost of ~2x latency.
        embedding_model: Sentence-transformers model name or instance.
        device: Inference device. ``"cpu"`` or ``"cuda"``.
        weights: Optional custom IQS component weights dict. Missing keys
            default to the standard weights. Use ``scroot.RAG_WEIGHTS``
            for RAG-optimised scoring (higher groundedness weight).
        iqs_mode: IQS formula. ``"harmonic"`` (default) uses the weighted
            harmonic mean: ``IQS = n / sum(w_i / s_i)``. Any metric near
            zero drives IQS to zero. ``"geometric"`` uses weighted geometric
            mean - does not collapse to zero on partial hallucination.
        atomic_claims: If True (default), split compound sentences into
            sub-claims before groundedness scoring. Prevents one wrong fact
            from zeroing an entire multi-fact sentence.
        similarity_fallback: If True (default), use bi-encoder cosine
            similarity as a fallback when NLI confidence is uncertain (0.3-0.7).
            Catches paraphrases that exact NLI entailment misses.
        similarity_threshold: Cosine similarity threshold for paraphrase
            credit in the similarity fallback. Default 0.82.
        max_query_length: Truncate query to this many characters (H-3).
        max_response_length: Truncate response to this many characters (H-3).
        max_context_items: Maximum number of context chunks (H-3).
        max_context_item_length: Truncate each context chunk to this length (H-3).
        max_batch_size: ``score_batch()`` raises ValueError above this limit (H-3).
        entailment_threshold: Minimum entailment probability for a claim to
            be grounded. Default 0.5.
        coverage_threshold: Minimum embedding similarity for a query segment
            to be considered covered by the response. Default 0.45.
        contradiction_threshold: Minimum contradiction probability to flag a
            sentence pair as contradictory. Default 0.7.
        max_sentences: Maximum sentences evaluated by consistency scorer. Default 25.
        compute_evidence_map: If True (default) and context is provided,
            attach a sentence-level :class:`~scroot.EvidenceMap` to
            ``result.evidence_map`` showing which response sentences are
            supported, contradicted, or ungrounded.
        evidence_entailment_threshold: Minimum entailment probability for a
            sentence to be marked "supported" in the evidence map. Default
            0.70 (stricter than the groundedness ``entailment_threshold``).
        evidence_contradiction_threshold: Minimum contradiction probability
            for a sentence to be marked "contradicted" in the evidence map.
            Default 0.30.
        relevance_sigmoid_midpoint: Cosine similarity value that maps to 0.5
            relevance. Default 0.5. Override for retrievers with higher
            baseline similarity (e.g. 0.7 for dense retrievers on same-domain
            corpora).
        relevance_sigmoid_steepness: Controls how sharply the relevance
            sigmoid rises. Default 10.0. Higher values → sharper transition.
        compute_numeric_groundedness: If True (default) and context is
            provided, run the numeric grounding verifier and expose the
            per-claim breakdown in ``result.details["numeric_groundedness"]``.
        flag_thresholds: Per-flag threshold overrides. Keys are flag names
            (``"hallucination_risk"``, ``"off_topic"``, ``"self_contradictory"``,
            ``"incomplete"``, ``"ungrounded"``). Unset keys fall back to
            built-in defaults.
        keep_intermediates: If True, attach intermediate computation details to
            ``result.details``. Default False.
        gate_inapplicable_dimensions: If True (default), dimensions that cannot
            meaningfully apply to a given input are excluded from the IQS
            composite rather than scored at near-zero. This prevents inapplicable
            metrics (e.g. relevance on a single-sentence extractive summary) from
            deflating the composite score. Disable only if you need all five
            dimensions scored unconditionally.
        groundedness_backbone: Grounding model to use.

            - ``"minicheck-roberta-large"`` (default): MiniCheck-RoBERTa-Large,
              355M, purpose-built factuality classifier. AUC 0.991 on NQ-500,
              Spearman ρ=0.47 on SummEval. Higher accuracy, ~1.75× latency
              vs deberta.
            - ``"deberta-base"``: cross-encoder/nli-deberta-v3-base, 184M,
              3-class NLI. Faster (3.2s vs 4.8s groundedness harness), lower
              accuracy (AUC 0.875, ρ=0.43). Use when latency is the primary
              constraint.
        top_k_premises: Number of premises (context sentences) to pass to the
            NLI cross-encoder per claim, pre-selected by embedding similarity.
            Reduces grounding inference from O(R×C) to O(R×k) with zero score
            delta on benchmarks (6,000 checks, 0 deviations). Speedup is
            negligible at <10 context sentences and 3.5× at 40 sentences.
            Set to None to disable and run the full O(R×C) cross-product
            (not recommended for long contexts). Default 8.
    """

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        weights: dict | None = None,
        iqs_mode: str = "harmonic",
        atomic_claims: bool = True,
        similarity_fallback: bool = True,
        similarity_threshold: float = 0.82,
        top_k_chunks: int = 3,
        top_k_premises: int = 8,
        bidirectional_consistency: bool = True,
        nli_completeness: bool = True,
        max_query_length: int = 10_000,
        max_response_length: int = 50_000,
        max_context_items: int = 50,
        max_context_item_length: int = 10_000,
        max_batch_size: int = 1_000,
        entailment_threshold: float = 0.5,
        coverage_threshold: float = 0.45,
        contradiction_threshold: float = 0.7,
        max_sentences: int = 25,
        compute_evidence_map: bool = True,
        evidence_entailment_threshold: float = 0.70,
        evidence_contradiction_threshold: float = 0.30,
        relevance_sigmoid_midpoint: float = 0.5,
        relevance_sigmoid_steepness: float = 10.0,
        compute_numeric_groundedness: bool = True,
        flag_thresholds: dict | None = None,
        keep_intermediates: bool = False,
        gate_inapplicable_dimensions: bool = True,
        groundedness_backbone: str = "minicheck-roberta-large",
    ):
        self.nli_model = nli_model
        self.embedding_model = embedding_model
        self.device = device
        self.weights = weights
        self.iqs_mode = iqs_mode
        self.atomic_claims = atomic_claims
        self.similarity_fallback = similarity_fallback
        self.similarity_threshold = similarity_threshold
        self.top_k_chunks = top_k_chunks
        self.top_k_premises = top_k_premises
        self.bidirectional_consistency = bidirectional_consistency
        self.nli_completeness = nli_completeness
        self.max_query_length = max_query_length
        self.max_response_length = max_response_length
        self.max_context_items = max_context_items
        self.max_context_item_length = max_context_item_length
        self.max_batch_size = max_batch_size
        self.entailment_threshold = entailment_threshold
        self.coverage_threshold = coverage_threshold
        self.contradiction_threshold = contradiction_threshold
        self.max_sentences = max_sentences
        self.compute_evidence_map = compute_evidence_map
        self.evidence_entailment_threshold = evidence_entailment_threshold
        self.evidence_contradiction_threshold = evidence_contradiction_threshold
        self.relevance_sigmoid_midpoint = relevance_sigmoid_midpoint
        self.relevance_sigmoid_steepness = relevance_sigmoid_steepness
        self.compute_numeric_groundedness = compute_numeric_groundedness
        self.flag_thresholds = flag_thresholds
        self.keep_intermediates = keep_intermediates
        self.gate_inapplicable_dimensions = gate_inapplicable_dimensions
        self.groundedness_backbone = groundedness_backbone
        self._backbone_scorer = None  # loaded lazily; _BACKBONE_LOADED sentinel set after first load
        self._backbone_loaded = False

    def score(
        self,
        query: str,
        response: str,
        context: "ContextPayload | str | list[str] | None" = None,
    ) -> EntailmentResult:
        """Score a single LLM response.

        Inputs are silently truncated to the configured length limits before
        processing. Non-string context items are coerced to ``str``; ``None``
        items are dropped.

        Args:
            query: The user's query/question.
            response: The LLM-generated response.
            context: Grounding context. Accepts:

                - :class:`~scroot.context.ContextPayload` - built by
                  :class:`~scroot.ContextBuilder`. Consumed here: the
                  assembled chunks feed the NLI scorer locally and the
                  payload is not retained. Only ``session_id`` and
                  ``checksum`` flow into ``details["context"]`` for the
                  audit trail.
                - ``str`` - a single grounding string.
                - ``list[str]`` - source context chunks.
                - ``None`` - groundedness is skipped entirely.

                If provided (even an empty list), groundedness is scored.

        Returns:
            :class:`EntailmentResult` with all metric scores and flags.
            ``result.iqs`` is computed as ``IQS = n / sum(w_i / s_i)``
            (the weighted harmonic mean of the five metrics, where
            ``n = sum(w_i)``) by default; see ``iqs_mode``.
        """
        context_audit: dict | None = None
        chunk_sources: list[str | None] | None = None
        if isinstance(context, ContextPayload):
            payload = context
            # Per-source chunks preserve top-k retrieval behaviour in the
            # groundedness scorer; the payload itself is consumed here.
            context = [e.content for e in payload.sources] or None
            chunk_sources = [e.source for e in payload.sources] or None
            context_audit = {
                "session_id": payload.session_id,
                "checksum": payload.checksum,
                "total_tokens": payload.total_tokens,
                "was_truncated": payload.was_truncated,
                "pii_scrubbed": payload.pii_scrubbed,
            }
        elif isinstance(context, str):
            context = [context]

        query = query[: self.max_query_length]
        response = response[: self.max_response_length]
        if context is not None:
            context = context[: self.max_context_items]
            context = [
                str(c)[: self.max_context_item_length]
                for c in context
                if c is not None and str(c).strip()  # drop empty/whitespace chunks
            ]
            if chunk_sources is not None:
                chunk_sources = chunk_sources[: self.max_context_items]
            # Empty / whitespace-only context is equivalent to no context:
            # groundedness cannot be computed (spec: treat "", "  ", [], None
            # identically).
            if not context:
                context = None
                chunk_sources = None

        # Lazy-load the groundedness backbone on first call.
        if not self._backbone_loaded:
            from .models import get_groundedness_backbone
            self._backbone_scorer = get_groundedness_backbone(
                self.groundedness_backbone, self.device)
            self._backbone_loaded = True

        _debug_timing = os.environ.get("SCROOT_DEBUG_TIMING") == "1"
        _t0 = time.perf_counter() if _debug_timing else 0.0

        def _elapsed(label: str) -> None:
            if _debug_timing:
                logger.debug("[timing] %s: %.3fs", label, time.perf_counter() - _t0)

        details = {}
        if context_audit is not None:
            details["context"] = context_audit

        if context is not None:
            try:
                groundedness, g_details = score_groundedness(
                    response, context,
                    nli_model=self.nli_model,
                    embedding_model=self.embedding_model,
                    device=self.device,
                    entailment_threshold=self.entailment_threshold,
                    atomic_claims=self.atomic_claims,
                    similarity_fallback=self.similarity_fallback,
                    similarity_threshold=self.similarity_threshold,
                    top_k_chunks=self.top_k_chunks,
                    top_k_premises=self.top_k_premises,
                    backbone_scorer=self._backbone_scorer,
                )
                details["groundedness"] = g_details
                _elapsed("groundedness")
            except Exception as e:
                # Context was provided but groundedness scoring failed
                # unexpectedly. Degrade gracefully: exclude groundedness from
                # IQS rather than failing the whole call.
                logger.error("Groundedness computation failed: %s", e)
                groundedness = None
                warnings.warn(
                    f"Groundedness computation failed due to an unexpected "
                    f"error. IQS will be computed from the remaining metrics. "
                    f"Error: {e}",
                    GroundednessComputationError,
                    stacklevel=2,
                )
        else:
            groundedness = None
            # Encourage adding context - but stay silent if the caller has
            # explicitly opted out by zeroing the groundedness weight.
            ground_weight = (self.weights or DEFAULT_WEIGHTS).get(
                "groundedness", DEFAULT_WEIGHTS["groundedness"]
            )
            if ground_weight > 0.0:
                warnings.warn(
                    "auditor.score() called without context. groundedness "
                    "will be None and is excluded from IQS (the remaining "
                    "metrics' weights are redistributed). To score "
                    "groundedness, pass context= or use ContextBuilder.",
                    NoContextWarning,
                    stacklevel=2,
                )

        # Pre-compute query embedding once; reused by score_relevance and
        # build_evidence_map (per-sentence relevance) — zero extra encode calls.
        _query_emb = None
        if query.strip():
            _emb_model = get_embedding_model(self.embedding_model, device=self.device)
            _query_emb = _emb_model.encode(query, convert_to_numpy=True)

        completeness, c_details = score_completeness(
            query, response,
            embedding_model=self.embedding_model,
            nli_model=self.nli_model if self.nli_completeness else None,
            device=self.device,
            coverage_threshold=self.coverage_threshold,
        )
        details["completeness"] = c_details
        _elapsed("completeness")

        relevance, r_details = score_relevance(
            query, response,
            embedding_model=self.embedding_model,
            device=self.device,
            midpoint=self.relevance_sigmoid_midpoint,
            steepness=self.relevance_sigmoid_steepness,
            query_embedding=_query_emb,
        )
        details["relevance"] = r_details
        _elapsed("relevance")

        # _cap collects consistency NLI logits; always-on for mini-IQS.
        # The intermediates block reads it when keep_intermediates=True.
        _cap: dict = {}
        consistency, cons_details = score_consistency(
            response,
            nli_model=self.nli_model,
            device=self.device,
            contradiction_threshold=self.contradiction_threshold,
            max_sentences=self.max_sentences,
            bidirectional=self.bidirectional_consistency,
            _capture=_cap,
        )
        details["consistency"] = cons_details
        _elapsed("consistency")

        evidence_map = None
        if context is not None and self.compute_evidence_map:
            evidence_map = build_evidence_map(
                response, context,
                nli_model=self.nli_model,
                embedding_model=self.embedding_model,
                device=self.device,
                entailment_threshold=self.evidence_entailment_threshold,
                contradiction_threshold=self.evidence_contradiction_threshold,
                top_k_chunks=self.top_k_chunks,
                chunk_sources=chunk_sources,
                atomic_claims=self.atomic_claims,
                query_embedding=_query_emb,
                consistency_capture=_cap if _cap else None,
                sigmoid_midpoint=self.relevance_sigmoid_midpoint,
                sigmoid_steepness=self.relevance_sigmoid_steepness,
            )

        confidence, conf_details = score_confidence(response)
        details["confidence"] = conf_details
        _elapsed("confidence")

        iqs_scores: dict = {
            "completeness": completeness,
            "relevance": relevance,
            "consistency": consistency,
            "confidence": confidence,
        }
        if groundedness is not None:
            iqs_scores["groundedness"] = groundedness

        # Custom metrics from register_metric(). Scored after built-ins;
        # weight=0 records the score in details without affecting IQS.
        custom_weights: dict[str, float] = {}
        from .metrics._registry import _CUSTOM_METRICS
        if _CUSTOM_METRICS:
            context_list = list(context) if context is not None else None
            custom_scores: dict[str, float] = {}
            for name, (fn, weight) in _CUSTOM_METRICS.items():
                try:
                    val = float(fn(query, response, context_list))
                    val = max(0.0, min(1.0, val))
                    iqs_scores[name] = val
                    custom_scores[name] = val
                    if weight > 0:
                        custom_weights[name] = weight
                except Exception as exc:  # noqa: BLE001
                    logger.warning("custom metric %r failed: %s", name, exc)
            if custom_scores:
                details["custom_metrics"] = custom_scores

        # Applicability gating: when a dimension is structurally inapplicable to
        # the task (e.g. relevance under a generic "summarise this" query, or
        # consistency on a single-sentence response), set its score to None so
        # compute_iqs_detailed excludes it and renormalises the remaining
        # weights, rather than letting a pathologically low non-signal collapse
        # IQS via the harmonic mean. groundedness/completeness are never gated.
        if self.gate_inapplicable_dimensions:
            from .applicability import inapplicable_dimensions
            gated = inapplicable_dimensions(query, response)
            # Never gate every dimension away; keep at least groundedness.
            for dim in gated:
                if dim in iqs_scores:
                    iqs_scores[dim] = None
            if gated:
                details["gated_dimensions"] = sorted(gated)

        merged_weights = {**(self.weights or {}), **custom_weights} or None
        iqs, effective_weights = compute_iqs_detailed(
            iqs_scores, weights=merged_weights, mode=self.iqs_mode,
        )

        if context is not None and self.compute_numeric_groundedness:
            _, numeric_details = score_numeric_groundedness(
                response, context,
                nli_model=self.nli_model if self.nli_completeness else None,
                device=self.device,
            )
            details["numeric_groundedness"] = numeric_details
            _elapsed("numeric_groundedness")

        flags = detect_flags(
            groundedness, completeness, relevance,
            consistency, confidence,
            thresholds=self.flag_thresholds,
        )

        intermediates: dict | None = None
        if self.keep_intermediates:
            import numpy as np
            emb_model = get_embedding_model(self.embedding_model, device=self.device)
            response_sentences = split_sentences(response)
            query_embedding = emb_model.encode(query, convert_to_numpy=True)
            response_embeddings = emb_model.encode(response_sentences, convert_to_numpy=True) if response_sentences else np.empty((0,))
            intermediates = {
                "query_embedding": query_embedding,
                "response_embeddings": response_embeddings,
                "response_sentences": response_sentences,
            }
            if _cap:
                intermediates.update(_cap)
            _elapsed("intermediates")

        return EntailmentResult(
            groundedness=groundedness,
            completeness=completeness,
            relevance=relevance,
            consistency=consistency,
            confidence=confidence,
            iqs=iqs,
            flags=flags,
            details=details,
            evidence_map=evidence_map,
            effective_weights=effective_weights,
            context_used=(groundedness is not None),
            iqs_metric_count=len(effective_weights),
            intermediates=intermediates,
        )

    def score_batch(
        self,
        items: list[dict],
    ) -> list[EntailmentResult]:
        """Score a batch of responses.

        Args:
            items: List of dicts with keys ``"query"``, ``"response"``,
                and optionally ``"context"`` (list[str]).

        Returns:
            List of :class:`EntailmentResult`, one per item, in order.

        Raises:
            ValueError: If ``len(items)`` exceeds ``max_batch_size`` (H-3).
        """
        if len(items) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(items)} exceeds max_batch_size={self.max_batch_size}. "
                f"Split into smaller batches or increase max_batch_size."
            )
        return [
            self.score(
                query=item["query"],
                response=item["response"],
                context=item.get("context"),
            )
            for item in items
        ]
