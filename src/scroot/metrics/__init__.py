"""Metrics subpackage for scroot."""

from .groundedness import score_groundedness
from .completeness import score_completeness
from .relevance import score_relevance
from .consistency import score_consistency
from .confidence import score_confidence
from ._registry import register_metric


def open_builder(*args, **kwargs) -> object:
    """Cloud: hosted no-code visual metric builder."""
    from .._entitlements import get_enterprise

    return get_enterprise("metrics.builder", "No-code custom metric builder").open(
        *args, **kwargs
    )


__all__ = [
    "score_groundedness",
    "score_completeness",
    "score_relevance",
    "score_consistency",
    "score_confidence",
    "register_metric",
    "open_builder",
]
