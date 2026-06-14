"""Sampling strategies for scoring subsets of responses.

Provides random, percentage, stratified, confidence-interval, and
priority-based sampling. All strategies are reproducible via seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


class SamplingStrategy:
    """String constants for sampling strategy names."""

    RANDOM = "random"
    PERCENTAGE = "percentage"
    STRATIFIED = "stratified"
    CONFIDENCE = "confidence"
    PRIORITY = "priority"


@dataclass
class SamplingResult:
    """Result of a sampled scoring run."""

    scored_items: list[dict]
    total_population: int
    sample_size: int
    sampling_rate: float
    strategy: str
    seed: int | None
    mean_iqs: float
    median_iqs: float
    std_iqs: float
    min_iqs: float
    max_iqs: float
    flag_counts: dict
    flag_rate: dict
    iqs_confidence_interval: tuple[float, float] | None = None
    stratum_stats: dict | None = None

    def summary(self) -> str:
        """Human-readable summary of the sampling run."""
        ci = ""
        if self.iqs_confidence_interval:
            lo, hi = self.iqs_confidence_interval
            ci = f", 95% CI: [{lo:.3f}, {hi:.3f}]"
        flags = ", ".join(
            f"{k}: {v} ({self.flag_rate[k]:.1%})"
            for k, v in self.flag_counts.items()
            if v > 0
        )
        return (
            f"Sampled {self.sample_size}/{self.total_population} "
            f"({self.sampling_rate:.1%}) using {self.strategy} strategy.\n"
            f"Mean IQS: {self.mean_iqs:.3f} (±{self.std_iqs:.3f}){ci}\n"
            f"Flags: {flags or 'none'}"
        )

    def to_dict(self) -> dict:
        """Serialize for logging or API response."""
        return {
            "total_population": self.total_population,
            "sample_size": self.sample_size,
            "sampling_rate": self.sampling_rate,
            "strategy": self.strategy,
            "seed": self.seed,
            "mean_iqs": self.mean_iqs,
            "median_iqs": self.median_iqs,
            "std_iqs": self.std_iqs,
            "min_iqs": self.min_iqs,
            "max_iqs": self.max_iqs,
            "flag_counts": self.flag_counts,
            "flag_rate": self.flag_rate,
            "iqs_confidence_interval": self.iqs_confidence_interval,
            "stratum_stats": self.stratum_stats,
        }


def _compute_confidence_sample_size(
    population: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.03,
) -> int:
    """Compute required sample size using Cochran's formula with FPC.

    Assumes maximum variance (p=0.5) for a conservative estimate.
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level)
    if z is None:
        raise ValueError(
            f"Unsupported confidence_level={confidence_level}. "
            "Use 0.90, 0.95, or 0.99."
        )
    p = 0.5
    e = margin_of_error
    n0 = (z ** 2 * p * (1 - p)) / (e ** 2)
    n = n0 / (1 + (n0 - 1) / population)
    return min(int(math.ceil(n)), population)


def _compute_confidence_interval(
    scores: list[float],
    population_size: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for the estimated population mean IQS."""
    import numpy as np
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0)
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n)
    fpc = math.sqrt((population_size - n) / max(population_size - 1, 1))
    margin = z * se * fpc
    return (
        round(max(0.0, mean - margin), 4),
        round(min(1.0, mean + margin), 4),
    )


def _select_indices(
    items: list[dict],
    strategy: str,
    sample_size: int | None = None,
    sample_pct: float | None = None,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.03,
    stratify_by: str | None = None,
    priority_fn=None,
    seed: int | None = None,
) -> list[int]:
    """Return sorted list of item indices to score based on strategy."""
    n = len(items)
    rng = random.Random(seed)

    if strategy == "random":
        if sample_size is None:
            raise ValueError("sample_size required for 'random' strategy")
        size = min(sample_size, n)
        return sorted(rng.sample(range(n), size))

    elif strategy == "percentage":
        if sample_pct is None:
            raise ValueError("sample_pct required for 'percentage' strategy")
        size = min(max(1, int(math.ceil(n * sample_pct))), n)
        return sorted(rng.sample(range(n), size))

    elif strategy == "stratified":
        if stratify_by is None:
            raise ValueError("stratify_by required for 'stratified' strategy")
        if sample_size is None:
            raise ValueError(
                "sample_size required for 'stratified' strategy (per stratum)"
            )
        strata: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            key = str(item.get(stratify_by, "_unknown"))
            strata.setdefault(key, []).append(i)
        selected = []
        for indices in strata.values():
            per_stratum = min(sample_size, len(indices))
            selected.extend(rng.sample(indices, per_stratum))
        return sorted(selected)

    elif strategy == "confidence":
        size = _compute_confidence_sample_size(n, confidence_level, margin_of_error)
        return sorted(rng.sample(range(n), size))

    elif strategy == "priority":
        if priority_fn is None:
            raise ValueError("priority_fn required for 'priority' strategy")
        if sample_size is None:
            raise ValueError("sample_size required for 'priority' strategy")
        ranked = sorted(enumerate(items), key=lambda x: priority_fn(x[1]), reverse=True)
        return sorted([i for i, _ in ranked[:sample_size]])

    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            "Use 'random', 'percentage', 'stratified', 'confidence', or 'priority'."
        )


def sample_and_score(
    auditor,
    items: list[dict],
    strategy: str = "random",
    sample_size: int | None = None,
    sample_pct: float | None = None,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.03,
    stratify_by: str | None = None,
    priority_fn=None,
    seed: int | None = 42,
) -> SamplingResult:
    """Score a sampled subset of responses.

    Args:
        auditor: Auditor instance.
        items: Full list of response dicts, each with "query", "response",
            and optionally "context" and any extra fields.
        strategy: One of "random", "percentage", "stratified",
            "confidence", "priority".
        sample_size: Items to sample (random, stratified, priority).
        sample_pct: Fraction to sample (percentage strategy).
        confidence_level: Target confidence level (confidence strategy).
        margin_of_error: Acceptable margin of error (confidence strategy).
        stratify_by: Key in item dicts to group strata by.
        priority_fn: Callable(item) -> numeric; higher = higher priority.
        seed: Random seed for reproducibility.

    Returns:
        SamplingResult with scored items and aggregate statistics.
    """
    import numpy as np
    if not items:
        return SamplingResult(
            scored_items=[],
            total_population=0,
            sample_size=0,
            sampling_rate=0.0,
            strategy=strategy,
            seed=seed,
            mean_iqs=0.0,
            median_iqs=0.0,
            std_iqs=0.0,
            min_iqs=0.0,
            max_iqs=0.0,
            flag_counts={},
            flag_rate={},
        )

    indices = _select_indices(
        items, strategy, sample_size, sample_pct,
        confidence_level, margin_of_error,
        stratify_by, priority_fn, seed,
    )

    scored_items = []
    iqs_scores: list[float] = []
    all_flags: list[str] = []

    for idx in indices:
        item = items[idx]
        result = auditor.score(
            query=item["query"],
            response=item["response"],
            context=item.get("context"),
        )
        scored_items.append({"item": item, "result": result, "index": idx})
        iqs_scores.append(result.iqs)
        all_flags.extend(result.flags)

    iqs_arr = np.array(iqs_scores) if iqs_scores else np.array([0.0])

    flag_counts: dict[str, int] = {}
    for f in all_flags:
        flag_counts[f] = flag_counts.get(f, 0) + 1

    n_scored = len(scored_items) or 1
    flag_rate = {k: v / n_scored for k, v in flag_counts.items()}

    ci = _compute_confidence_interval(iqs_scores, len(items), confidence_level)

    stratum_stats = None
    if strategy == "stratified" and stratify_by:
        stratum_stats = {}
        for si in scored_items:
            key = str(si["item"].get(stratify_by, "_unknown"))
            if key not in stratum_stats:
                stratum_stats[key] = {"scores": [], "flags": []}
            stratum_stats[key]["scores"].append(si["result"].iqs)
            stratum_stats[key]["flags"].extend(si["result"].flags)
        for key, data in stratum_stats.items():
            s = np.array(data["scores"])
            stratum_stats[key] = {
                "count": len(data["scores"]),
                "mean_iqs": float(np.mean(s)),
                "std_iqs": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
                "flag_counts": {f: data["flags"].count(f) for f in set(data["flags"])},
            }

    return SamplingResult(
        scored_items=scored_items,
        total_population=len(items),
        sample_size=len(scored_items),
        sampling_rate=len(scored_items) / len(items),
        strategy=strategy,
        seed=seed,
        mean_iqs=float(np.mean(iqs_arr)),
        median_iqs=float(np.median(iqs_arr)),
        std_iqs=float(np.std(iqs_arr, ddof=1)) if len(iqs_arr) > 1 else 0.0,
        min_iqs=float(np.min(iqs_arr)),
        max_iqs=float(np.max(iqs_arr)),
        flag_counts=flag_counts,
        flag_rate=flag_rate,
        iqs_confidence_interval=ci,
        stratum_stats=stratum_stats,
    )
