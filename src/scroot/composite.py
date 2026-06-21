"""Information Quality Score (IQS) - composite metric.

IQS = n / sum(w_i / s_i) -- the weighted harmonic mean of the five
metric scores, where n = sum(w_i).

Two scoring modes:
  harmonic (default): weighted harmonic mean.  Any metric near zero drives
      IQS to zero.  A response with groundedness=0.1 and all others at 0.9
      scores ~0.31. Zero tolerance: a single quality failure dominates the
      score, which matches the goal of flagging unreliable responses.

  geometric: weighted geometric mean.  Penalizes low scores but does not
      collapse to zero unless a metric is literally zero. Reflects partial
      quality more gently: 9 correct claims + 1 wrong claim -> ~0.8 IQS
      (not near 0).

Harmonic is the default and the formula documented in the README: any
metric near zero (e.g. a hallucinated claim) should drive the composite
score down hard rather than being averaged away.

Default weights:
    groundedness  0.35  (most important: is it faithful to the source?)
    completeness  0.25  (did it answer the full question?)
    relevance     0.20  (is it on topic?)
    consistency   0.15  (does it contradict itself?)
    confidence    0.05  (calibration signal, low weight)

When context is not provided, groundedness weight is redistributed
proportionally across the remaining metrics.
"""

from __future__ import annotations

import math


DEFAULT_WEIGHTS = {
    "groundedness": 0.35,
    "completeness": 0.25,
    "relevance": 0.20,
    "consistency": 0.15,
    "confidence": 0.05,
}

# RAG-optimised preset: boost groundedness, reduce completeness weight.
# Use when the source context IS the ground truth and faithfulness is
# the primary concern.
RAG_WEIGHTS = {
    "groundedness": 0.50,
    "completeness": 0.15,
    "relevance": 0.20,
    "consistency": 0.10,
    "confidence": 0.05,
}

# Factual/code/legal/technical preset: confidence excluded (weight 0.0).
# The confidence metric counts hedge/assertion markers. On declarative or
# code responses with no hedging language it always returns 0.5 (no signal).
# Use this preset for domains where responses are expected to be assertive
# and the 0.5 non-signal would dilute IQS. Redistribution is proportional:
# groundedness 0.37, completeness 0.26, relevance 0.21, consistency 0.16.
DEFAULT_WEIGHTS_FACTUAL = {
    "groundedness": 0.35,
    "completeness": 0.25,
    "relevance": 0.20,
    "consistency": 0.15,
    "confidence": 0.0,
}


def compute_iqs(
    groundedness: float | None,
    completeness: float,
    relevance: float,
    consistency: float,
    confidence: float,
    weights: dict | None = None,
    mode: str = "harmonic",
) -> float:
    """Compute the Information Quality Score.

    IQS = n / sum(w_i / s_i), where n = sum(w_i) and s_i are the metric
    scores. This is the weighted harmonic mean.

    Args:
        groundedness: 0-1 or None if no context was provided.
        completeness: 0-1.
        relevance: 0-1.
        consistency: 0-1.
        confidence: 0-1.
        weights: Optional custom weight dict. Missing keys default to
            DEFAULT_WEIGHTS.
        mode: Scoring formula.
            "harmonic" (default) - weighted harmonic mean. Zero tolerance:
                any metric near zero drives IQS toward zero.
            "geometric" - weighted geometric mean. Gracefully handles
                partial quality; does not collapse to zero unless a metric
                is literally zero.

    Returns:
        IQS score in [0, 1].
    """
    scores: dict[str, float] = {
        "completeness": completeness,
        "relevance": relevance,
        "consistency": consistency,
        "confidence": confidence,
    }
    # None groundedness (no context) is excluded so its weight is
    # redistributed proportionally rather than counted as a zero.
    if groundedness is not None:
        scores["groundedness"] = groundedness

    iqs, _ = compute_iqs_detailed(scores, weights=weights, mode=mode)
    return iqs


def compute_iqs_detailed(
    scores: "dict[str, float | None]",
    weights: "dict | None" = None,
    mode: str = "harmonic",
) -> "tuple[float, dict[str, float]]":
    """Compute IQS and report the effective weights actually used.

    The dict-based companion to :func:`compute_iqs`. A metric is *active* when
    it has a non-``None`` score **and** a positive weight; only active metrics
    contribute to IQS, and their weights are renormalised to sum to 1.0
    (proportional redistribution). This is how an inapplicable metric -
    typically ``groundedness`` when no context was provided - is excluded
    without being treated as a catastrophic zero.

    A metric value of exactly ``0.0`` is a *real* measurement (not missing
    data) and collapses IQS to ``0.0`` under both means.

    Args:
        scores: Metric name -> score (float) or ``None`` (inapplicable).
        weights: Metric name -> weight. Missing keys fall back to
            ``DEFAULT_WEIGHTS``. Need not sum to 1.0; active weights are
            renormalised.
        mode: ``"harmonic"`` (default) or ``"geometric"``.

    Returns:
        ``(iqs, effective_weights)`` - the IQS in ``[0.0, 1.0]`` and the
        normalised weights of the active metrics (sums to 1.0).

    Raises:
        ValueError: if no metric is active (all ``None`` or zero-weighted).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Active = scored (non-None) AND positively weighted. Excluding
    # zero-weighted metrics lets a caller opt a metric out explicitly (e.g.
    # groundedness weight 0.0 when context is never available).
    active = {
        k: v for k, v in scores.items()
        if v is not None and w.get(k, 0.0) > 0.0
    }
    if not active:
        raise ValueError(
            "All metrics are None (or zero-weighted) - nothing to compute IQS "
            "from. At least one metric must have a non-None score and a "
            "positive weight."
        )

    total_active_weight = sum(w[k] for k in active)
    effective_weights = {k: w[k] / total_active_weight for k in active}

    # A genuine zero score is a real failure: collapse IQS to 0.0 under both
    # means (harmonic does this via eps anyway; this makes geometric match).
    if any(v == 0.0 for v in active.values()):
        return 0.0, effective_weights

    eps = 1e-6
    if mode == "geometric":
        # Weighted geometric mean: exp(sum(w_i * log(s_i))) == prod(s_i ^ w_i)
        log_iqs = sum(effective_weights[k] * math.log(max(active[k], eps)) for k in active)
        iqs = math.exp(log_iqs)
    else:
        # Weighted harmonic mean: 1 / sum(w_i / s_i), weights summing to 1.
        iqs = 1.0 / sum(effective_weights[k] / max(active[k], eps) for k in active)

    return round(min(max(iqs, 0.0), 1.0), 4), effective_weights
