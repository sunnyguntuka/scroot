"""Tests for entail.sampling - all strategies, statistics, edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scroot.result import EntailmentResult
from scroot.sampling import SamplingStrategy, sample_and_score


def _fake_result(iqs: float = 0.8, flags: list[str] | None = None) -> EntailmentResult:
    return EntailmentResult(
        groundedness=0.8,
        completeness=0.8,
        relevance=0.8,
        consistency=0.8,
        confidence=0.8,
        iqs=iqs,
        flags=flags or [],
        details={},
    )


def _mock_auditor(iqs: float = 0.8, flags: list[str] | None = None):
    auditor = MagicMock()
    auditor.score.return_value = _fake_result(iqs=iqs, flags=flags)
    return auditor


def _items(n: int, extra: dict | None = None) -> list[dict]:
    base = {"query": "q", "response": "r"}
    if extra:
        base.update(extra)
    return [dict(base, id=i) for i in range(n)]


# --- strategy: random ---

def test_random_sampling():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(1000), strategy="random", sample_size=100, seed=42)
    assert result.sample_size == 100
    assert len(result.scored_items) == 100


def test_random_sampling_no_duplicates():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(1000), strategy="random", sample_size=100, seed=42)
    indices = [si["index"] for si in result.scored_items]
    assert len(indices) == len(set(indices))


# --- strategy: percentage ---

def test_percentage_sampling():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(1000), strategy="percentage", sample_pct=0.1, seed=1)
    assert result.sample_size == 100
    assert len(result.scored_items) == 100


def test_percentage_rounds_up():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(7), strategy="percentage", sample_pct=0.5, seed=0)
    assert result.sample_size == 4  # ceil(3.5) = 4


# --- strategy: stratified ---

def test_stratified_sampling():
    items = []
    for agent in ["a", "b", "c"]:
        for _ in range(100):
            items.append({"query": "q", "response": "r", "agent": agent})
    auditor = _mock_auditor()
    result = sample_and_score(
        auditor, items, strategy="stratified",
        stratify_by="agent", sample_size=10, seed=7,
    )
    assert result.sample_size == 30
    for si in result.scored_items:
        assert si["item"]["agent"] in ["a", "b", "c"]


def test_stratum_stats_populated():
    items = []
    for agent in ["x", "y"]:
        for _ in range(20):
            items.append({"query": "q", "response": "r", "agent": agent})
    auditor = _mock_auditor()
    result = sample_and_score(
        auditor, items, strategy="stratified",
        stratify_by="agent", sample_size=5, seed=0,
    )
    assert result.stratum_stats is not None
    assert "x" in result.stratum_stats
    assert "y" in result.stratum_stats
    assert result.stratum_stats["x"]["count"] == 5


# --- strategy: confidence ---

def test_confidence_sampling_auto_size():
    auditor = _mock_auditor()
    result = sample_and_score(
        auditor, _items(10_000),
        strategy="confidence", confidence_level=0.95, margin_of_error=0.03, seed=0,
    )
    # Cochran's formula for N=10000, z=1.96, p=0.5, e=0.03 -> ~964
    assert 900 <= result.sample_size <= 1100


def test_confidence_sampling_caps_at_population():
    auditor = _mock_auditor()
    result = sample_and_score(
        auditor, _items(10),
        strategy="confidence", confidence_level=0.95, margin_of_error=0.03, seed=0,
    )
    assert result.sample_size == 10


# --- strategy: priority ---

def test_priority_sampling_selects_highest():
    items = [{"query": "q", "response": "r" * (i + 1)} for i in range(200)]
    auditor = _mock_auditor()
    result = sample_and_score(
        auditor, items, strategy="priority",
        priority_fn=lambda item: len(item["response"]),
        sample_size=50, seed=0,
    )
    assert result.sample_size == 50
    # All selected items should be from the top 50 longest responses
    selected_indices = {si["index"] for si in result.scored_items}
    top_50_indices = set(range(150, 200))  # items 150-199 are the longest
    assert selected_indices == top_50_indices


# --- reproducibility ---

def test_reproducible_with_seed():
    auditor = _mock_auditor()
    r1 = sample_and_score(auditor, _items(500), strategy="random", sample_size=50, seed=99)
    r2 = sample_and_score(auditor, _items(500), strategy="random", sample_size=50, seed=99)
    assert [si["index"] for si in r1.scored_items] == [si["index"] for si in r2.scored_items]


def test_different_seeds_differ():
    auditor = _mock_auditor()
    r1 = sample_and_score(auditor, _items(500), strategy="random", sample_size=50, seed=1)
    r2 = sample_and_score(auditor, _items(500), strategy="random", sample_size=50, seed=2)
    assert [si["index"] for si in r1.scored_items] != [si["index"] for si in r2.scored_items]


# --- edge cases ---

def test_empty_items():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, [], strategy="random", sample_size=10)
    assert result.sample_size == 0
    assert result.total_population == 0
    assert result.scored_items == []


def test_sample_size_exceeds_population():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(100), strategy="random", sample_size=2000, seed=0)
    assert result.sample_size == 100


def test_unknown_strategy_raises():
    auditor = _mock_auditor()
    with pytest.raises(ValueError, match="Unknown strategy"):
        sample_and_score(auditor, _items(10), strategy="invalid", sample_size=5)


def test_random_missing_sample_size_raises():
    auditor = _mock_auditor()
    with pytest.raises(ValueError, match="sample_size required"):
        sample_and_score(auditor, _items(10), strategy="random")


def test_percentage_missing_sample_pct_raises():
    auditor = _mock_auditor()
    with pytest.raises(ValueError, match="sample_pct required"):
        sample_and_score(auditor, _items(10), strategy="percentage")


# --- statistics ---

def test_statistics_correct():
    # All results return iqs=0.8, so mean/median/min/max should all be 0.8
    auditor = _mock_auditor(iqs=0.8)
    result = sample_and_score(auditor, _items(100), strategy="random", sample_size=20, seed=0)
    assert abs(result.mean_iqs - 0.8) < 1e-6
    assert abs(result.median_iqs - 0.8) < 1e-6
    assert abs(result.min_iqs - 0.8) < 1e-6
    assert abs(result.max_iqs - 0.8) < 1e-6
    assert result.std_iqs < 1e-10


def test_sampling_rate_correct():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(200), strategy="random", sample_size=50, seed=0)
    assert abs(result.sampling_rate - 0.25) < 1e-9


def test_total_population_recorded():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(500), strategy="random", sample_size=10, seed=0)
    assert result.total_population == 500


def test_confidence_interval_computed():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(1000), strategy="random", sample_size=100, seed=0)
    assert result.iqs_confidence_interval is not None
    lo, hi = result.iqs_confidence_interval
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert 0.0 <= lo <= hi <= 1.0


def test_flag_counts_aggregated():
    auditor = _mock_auditor(flags=["hallucination_risk"])
    result = sample_and_score(auditor, _items(100), strategy="random", sample_size=10, seed=0)
    assert result.flag_counts.get("hallucination_risk") == 10
    assert abs(result.flag_rate.get("hallucination_risk", 0) - 1.0) < 1e-9


# --- summary / to_dict ---

def test_summary_string():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(100), strategy="random", sample_size=10, seed=0)
    summary = result.summary()
    assert "10/100" in summary
    assert "random" in summary
    assert "Mean IQS" in summary


def test_to_dict_keys():
    auditor = _mock_auditor()
    result = sample_and_score(auditor, _items(50), strategy="random", sample_size=5, seed=0)
    d = result.to_dict()
    for key in ("total_population", "sample_size", "sampling_rate", "strategy",
                "mean_iqs", "median_iqs", "std_iqs", "flag_counts", "flag_rate"):
        assert key in d


# --- SamplingStrategy constants ---

def test_strategy_constants():
    assert SamplingStrategy.RANDOM == "random"
    assert SamplingStrategy.PERCENTAGE == "percentage"
    assert SamplingStrategy.STRATIFIED == "stratified"
    assert SamplingStrategy.CONFIDENCE == "confidence"
    assert SamplingStrategy.PRIORITY == "priority"
