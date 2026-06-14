from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .evidence import EvidenceMap

# One-sentence, deterministic descriptions of what a low score on each
# metric means. Used by iqs_explanation() - no LLM involved.
_METRIC_DESCRIPTIONS: dict[str, str] = {
    "groundedness": "the response makes claims not supported by the context",
    "completeness": "the response does not address all parts of the query",
    "relevance": "the response drifts from the topic of the query",
    "consistency": "the response contradicts itself",
    "confidence": "the response expresses inappropriate certainty relative to the evidence",
}

# Full-sentence explanations shown alongside a flagged metric in the
# dashboard. Used by metric_explanations - no LLM involved.
_METRIC_FLAG_EXPLANATIONS: dict[str, str] = {
    "groundedness": "The response makes claims that are not supported by the provided context.",
    "completeness": "The response does not address all parts of the query.",
    "relevance": "The response drifts from the topic of the query.",
    "consistency": "The response contradicts itself.",
    "confidence": "The response expresses inappropriate certainty relative to the evidence.",
}

# detect_flags() returns semantic flag names, not metric names. Map each
# flag back to the metric it explains.
_FLAG_TO_METRIC: dict[str, str] = {
    "hallucination_risk": "groundedness",
    "ungrounded": "groundedness",
    "off_topic": "relevance",
    "self_contradictory": "consistency",
    "incomplete": "completeness",
}


@dataclass
class EntailmentResult:
    """Result of scoring a single LLM response."""

    groundedness: float | None
    completeness: float
    relevance: float
    consistency: float
    confidence: float
    iqs: float
    flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    evidence_map: EvidenceMap | None = None

    # IQS transparency: which metrics actually contributed and at what weight.
    # Defaulted to None and derived in __post_init__ so results constructed
    # without them (e.g. in tests) still serialise correctly; Auditor.score()
    # passes the authoritative values computed alongside the IQS.
    effective_weights: "dict[str, float] | None" = None
    context_used: "bool | None" = None
    iqs_metric_count: "int | None" = None

    def __post_init__(self) -> None:
        active = self._metric_scores()  # excludes groundedness when None
        if self.context_used is None:
            self.context_used = self.groundedness is not None
        if self.iqs_metric_count is None:
            self.iqs_metric_count = len(active)
        if self.effective_weights is None:
            from .composite import DEFAULT_WEIGHTS
            total = sum(DEFAULT_WEIGHTS[k] for k in active) or 1.0
            self.effective_weights = {
                k: round(DEFAULT_WEIGHTS[k] / total, 4) for k in active
            }

    def _metric_scores(self) -> dict[str, float]:
        """Return the scored metrics as a dict.

        Excludes ``groundedness`` when it is ``None`` (no context provided),
        matching how :func:`compute_iqs` redistributes its weight.
        """
        scores = {
            "completeness": self.completeness,
            "relevance": self.relevance,
            "consistency": self.consistency,
            "confidence": self.confidence,
        }
        if self.groundedness is not None:
            scores["groundedness"] = self.groundedness
        return scores

    @property
    def weakest_metric(self) -> str:
        """Name of the lowest-scoring metric.

        ``groundedness`` is excluded when it was not scored (no context).
        """
        scores = self._metric_scores()
        return min(scores, key=scores.get)

    @property
    def score_variance(self) -> float:
        """Population standard deviation of the scored metrics.

        A value above ~0.30 indicates one metric is an outlier rather than
        uniform mediocrity across all metrics.
        """
        scores = self._metric_scores()
        return round(statistics.pstdev(scores.values()), 4)

    @property
    def metric_explanations(self) -> dict[str, str]:
        """Map of metric name to a one-sentence explanation, for flagged metrics.

        Derived from ``self.flags`` via ``_FLAG_TO_METRIC``. Multiple flags
        that map to the same metric (e.g. ``hallucination_risk`` and
        ``ungrounded`` both map to ``groundedness``) produce a single entry.
        """
        explanations = {}
        for flag in self.flags:
            metric = _FLAG_TO_METRIC.get(flag)
            if metric is None:
                continue
            explanations[metric] = _METRIC_FLAG_EXPLANATIONS[metric]
        return explanations

    def iqs_explanation(self, threshold: float = 0.70) -> str:
        """Deterministic one-sentence explanation of the IQS score.

        No LLM is used - the same inputs always produce the same explanation.

        Args:
            threshold: IQS value above which the response is considered to
                pass with no caveats. Default 0.70.

        Returns:
            A human-readable explanation naming the weakest metric when the
            IQS is below ``threshold``, or a short pass message otherwise.
        """
        if self.iqs >= threshold:
            return f"IQS {self.iqs:.2f} - all metrics above threshold."
        weakest = self.weakest_metric
        weakest_score = self._metric_scores()[weakest]
        desc = _METRIC_DESCRIPTIONS.get(weakest, f"{weakest} is low")
        return (
            f"IQS {self.iqs:.2f} - primary driver: {weakest} ({weakest_score:.2f}). "
            f"The {desc}."
        )

    def passes_gate(
        self,
        threshold: float = 0.70,
        require_groundedness: float | None = None,
        require_completeness: float | None = None,
        require_relevance: float | None = None,
        require_consistency: float | None = None,
        require_confidence: float | None = None,
    ) -> bool:
        """Return True if this response meets all quality thresholds.

        Args:
            threshold: Minimum IQS required. Default 0.70.
            require_*: Optional per-metric floors. If set, that metric must
                meet its floor even if IQS passes. If the metric was not
                scored (``groundedness`` is ``None`` when no context was
                provided) and a floor is requested for it, the gate **fails
                open** on that floor: it is skipped with a
                :class:`~scroot.GroundednessUnavailableWarning` rather than
                rejecting the response, since an unmeasured metric cannot be
                evaluated. The IQS threshold still applies.

        Usage:
            # Simple IQS gate
            if not result.passes_gate(0.85):
                return "I'm not confident in this answer."

            # Legal bot - groundedness must be very high
            if not result.passes_gate(0.80, require_groundedness=0.95):
                return fallback_response
        """
        return self.gate_reason(
            threshold=threshold,
            require_groundedness=require_groundedness,
            require_completeness=require_completeness,
            require_relevance=require_relevance,
            require_consistency=require_consistency,
            require_confidence=require_confidence,
        ) is None

    def gate_reason(
        self,
        threshold: float = 0.70,
        require_groundedness: float | None = None,
        require_completeness: float | None = None,
        require_relevance: float | None = None,
        require_consistency: float | None = None,
        require_confidence: float | None = None,
    ) -> str | None:
        """Return a human-readable reason the gate failed, or None if passed.

        Args mirror :meth:`passes_gate`.

        Usage:
            reason = result.gate_reason(threshold=0.85, require_groundedness=0.95)
            if reason:
                log.warning(f"Quality gate: {reason}")
        """
        if self.iqs < threshold:
            return f"{self.iqs_explanation(threshold)} (required >= {threshold:.2f})"

        floors = {
            "groundedness": require_groundedness,
            "completeness": require_completeness,
            "relevance": require_relevance,
            "consistency": require_consistency,
            "confidence": require_confidence,
        }
        for metric, floor in floors.items():
            if floor is None:
                continue
            value = getattr(self, metric)
            if value is None:
                # Metric inapplicable (no context) - cannot evaluate the floor.
                # Fail open: warn and skip rather than reject every no-context
                # response. The IQS threshold above still gates.
                import warnings

                from .exceptions import GroundednessUnavailableWarning
                warnings.warn(
                    f"require_{metric}={floor:.2f} was specified but {metric} "
                    f"is None (no context provided). The {metric} floor was "
                    f"not evaluated. Provide context to enforce this "
                    f"requirement.",
                    GroundednessUnavailableWarning,
                    stacklevel=2,
                )
                continue
            if value < floor:
                return f"{metric} {value:.2f} below required floor {floor:.2f}."
        return None

    def to_dict(self) -> dict:
        """Convert to plain dict for logging/serialization."""
        return {
            "iqs": self.iqs,
            "groundedness": self.groundedness,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "consistency": self.consistency,
            "confidence": self.confidence,
            "flags": self.flags,
            "details": self.details,
            "weakest_metric": self.weakest_metric,
            "score_variance": self.score_variance,
            "iqs_explanation": self.iqs_explanation(),
            "metric_explanations": self.metric_explanations,
            "evidence_map": self.evidence_map.to_dict() if self.evidence_map else None,
            "effective_weights": self.effective_weights,
            "context_used": self.context_used,
            "iqs_metric_count": self.iqs_metric_count,
        }

    def __repr__(self) -> str:
        """Return a concise string representation showing all metric scores."""
        parts = [f"iqs={self.iqs:.2f}"]
        if self.groundedness is not None:
            parts.append(f"groundedness={self.groundedness:.2f}")
        parts.extend([
            f"completeness={self.completeness:.2f}",
            f"relevance={self.relevance:.2f}",
            f"consistency={self.consistency:.2f}",
            f"confidence={self.confidence:.2f}",
        ])
        if self.flags:
            parts.append(f"flags={self.flags}")
        if self.evidence_map is not None:
            parts.append(f"evidence_coverage={self.evidence_map.coverage_ratio:.2f}")
        return f"EntailmentResult({', '.join(parts)})"
