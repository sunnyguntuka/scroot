"""scroot - LLM-free response quality scoring."""

from __future__ import annotations

from .core import Auditor
from .result import EntailmentResult
from .agents import AgentRegistry, AgentConfig
from .sampling import SamplingResult, SamplingStrategy, sample_and_score
from .composite import RAG_WEIGHTS, DEFAULT_WEIGHTS_FACTUAL
from .flags import DEFAULT_FLAG_THRESHOLDS
from .metrics.numeric_groundedness import score_numeric_groundedness
from .context import ContextBuilder, ContextEntry, ContextPayload
from .evidence import EvidenceEntry, EvidenceMap, build_evidence_map
from .streaming import PartialScore, StreamingAuditor
from .audit import configure_audit_log, export_evidence_bundle
from .calibrate import CalibrationResult, calibrate, schedule_recalibration
from .drift import RegressionReport, regression_check
from .pii import scrub
from .metrics._registry import register_metric
from ._entitlements import EnterpriseFeatureError
from .exceptions import (
    GroundednessComputationError,
    GroundednessUnavailableWarning,
    NoContextWarning,
)

__version__ = "0.4.0"
__all__ = [
    "Auditor",
    "ContextBuilder",
    "ContextPayload",
    "ContextEntry",
    "EntailmentResult",
    "EvidenceEntry",
    "EvidenceMap",
    "build_evidence_map",
    "PartialScore",
    "StreamingAuditor",
    "AgentRegistry",
    "AgentConfig",
    "SamplingResult",
    "SamplingStrategy",
    "sample_and_score",
    "RAG_WEIGHTS",
    "DEFAULT_WEIGHTS_FACTUAL",
    "DEFAULT_FLAG_THRESHOLDS",
    "score_numeric_groundedness",
    "configure_audit_log",
    "export_evidence_bundle",
    "calibrate",
    "CalibrationResult",
    "schedule_recalibration",
    "regression_check",
    "RegressionReport",
    "scrub",
    "register_metric",
    "EnterpriseFeatureError",
    "setup_nltk",
    "score",
    "verify",
    "NoContextWarning",
    "GroundednessUnavailableWarning",
    "GroundednessComputationError",
]


def setup_nltk() -> None:
    """Download NLTK punkt_tab tokenizer data for improved sentence splitting.

    Call this once after installation to enable NLTK-backed sentence
    splitting (more accurate than the built-in regex fallback).
    This is a one-time deployment step - not called at runtime.

    Example:
        python -c "import scroot; scroot.setup_nltk()"
    """
    import nltk
    nltk.download("punkt_tab", quiet=False)


def score(
    query: str,
    response: str,
    context: "ContextPayload | str | list[str] | None" = None,
    **kwargs,
) -> EntailmentResult:
    """Score a single LLM response using default settings.

    Convenience wrapper around Auditor().score(). Creates a fresh Auditor
    instance on each call. For repeated scoring, instantiate Auditor once
    and reuse it to avoid reloading models.

    Args:
        query: The user's query/question.
        response: The LLM-generated response.
        context: Grounding context - a ContextPayload from
            ContextBuilder.build(), a plain string, a list of source
            context strings, or None.
        **kwargs: Passed through to Auditor().

    Returns:
        EntailmentResult with all metric scores and flags.
    """
    auditor = Auditor(**kwargs)
    return auditor.score(query=query, response=response, context=context)


def verify(
    query: str,
    response: str,
    context: "ContextPayload | str | list[str] | None" = None,
    threshold: float = 0.7,
    **kwargs,
) -> bool:
    """Check whether a response meets a minimum quality threshold.

    Convenience wrapper that returns True if the IQS score meets or
    exceeds the threshold.

    Args:
        query: The user's query/question.
        response: The LLM-generated response.
        context: Grounding context - ContextPayload, str, list[str], or None.
        threshold: Minimum IQS score to pass. Default 0.7.
        **kwargs: Passed through to Auditor().

    Returns:
        True if IQS >= threshold, False otherwise.
    """
    result = score(query=query, response=response, context=context, **kwargs)
    return result.iqs >= threshold
