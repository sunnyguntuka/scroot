# Apache-2.0. Custom metric registry for register_metric().
from __future__ import annotations

from typing import Callable

# name -> (fn, weight)
_CUSTOM_METRICS: dict[str, tuple[Callable, float]] = {}


def register_metric(name: str, fn: Callable, *, weight: float = 0.1) -> None:
    """Register a custom scoring metric.

    Args:
        name: Unique metric name. Must not collide with built-in metric names
            (groundedness, completeness, relevance, consistency, confidence).
        fn: Callable ``(query: str, response: str, context: list[str] | None) -> float``.
            Must return a value in [0.0, 1.0]. Values outside that range are clamped.
            Exceptions raised by fn are caught and logged; the metric is skipped.
        weight: Contribution to IQS. The built-in metric weights are renormalised
            proportionally so the total stays at 1.0. Use weight=0 to score the
            metric and record it in details without affecting IQS.
    """
    if not callable(fn):
        raise TypeError(f"register_metric: fn must be callable, got {type(fn)!r}")
    if name in ("groundedness", "completeness", "relevance", "consistency", "confidence"):
        raise ValueError(
            f"register_metric: {name!r} collides with a built-in metric name."
        )
    _CUSTOM_METRICS[name] = (fn, weight)


def clear_custom_metrics() -> None:
    _CUSTOM_METRICS.clear()
