"""Tests for entail.agents.AgentRegistry."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from scroot.agents import AgentRegistry
from scroot.result import EntailmentResult
from scroot.composite import compute_iqs


def _fake_result(
    groundedness: float = 0.8,
    completeness: float = 0.8,
    relevance: float = 0.8,
    consistency: float = 0.8,
    confidence: float = 0.8,
    flags: list[str] | None = None,
) -> EntailmentResult:
    iqs = compute_iqs(groundedness, completeness, relevance, consistency, confidence)
    return EntailmentResult(
        groundedness=groundedness,
        completeness=completeness,
        relevance=relevance,
        consistency=consistency,
        confidence=confidence,
        iqs=iqs,
        flags=flags or [],
        details={},
    )


def _mock_auditor(**metric_kwargs) -> MagicMock:
    """Return a mock Auditor whose score() creates a fresh result each call."""
    auditor = MagicMock()
    auditor.weights = None
    auditor.score.side_effect = lambda query, response, context=None: _fake_result(**metric_kwargs)
    return auditor


def _registry(**kw) -> tuple[AgentRegistry, MagicMock]:
    auditor = _mock_auditor(**kw)
    return AgentRegistry(auditor), auditor


# ── Registration ─────────────────────────────────────────────────────────────

def test_register_agent():
    reg, _ = _registry()
    reg.register("bot")
    assert "bot" in reg.list_agents()


def test_register_duplicate_raises():
    reg, _ = _registry()
    reg.register("bot")
    with pytest.raises(ValueError, match="already registered"):
        reg.register("bot")


def test_register_with_custom_weights():
    reg, _ = _registry()
    w = {"groundedness": 0.5, "completeness": 0.5}
    reg.register("bot", weights=w)
    assert reg.get_config("bot").weights == w


def test_register_with_defaults():
    reg, _ = _registry()
    reg.register("bot")
    assert reg.get_config("bot").iqs_threshold == 0.7
    assert reg.get_config("bot").weights is None
    assert reg.get_config("bot").context_required is False


def test_register_custom_threshold():
    reg, _ = _registry()
    reg.register("bot", iqs_threshold=0.85)
    assert reg.get_config("bot").iqs_threshold == 0.85


def test_register_zero_threshold_not_replaced_by_default():
    reg, _ = _registry()
    reg.register("bot", iqs_threshold=0.0)
    assert reg.get_config("bot").iqs_threshold == 0.0


def test_unregister():
    reg, _ = _registry()
    reg.register("bot")
    reg.unregister("bot")
    assert "bot" not in reg.list_agents()


def test_unregister_nonexistent_raises():
    reg, _ = _registry()
    with pytest.raises(ValueError, match="not registered"):
        reg.unregister("ghost")


def test_update_threshold():
    reg, _ = _registry()
    reg.register("bot", iqs_threshold=0.7)
    reg.update("bot", iqs_threshold=0.9)
    assert reg.get_config("bot").iqs_threshold == 0.9


def test_update_weights():
    reg, _ = _registry()
    reg.register("bot")
    new_w = {"completeness": 0.9}
    reg.update("bot", weights=new_w)
    assert reg.get_config("bot").weights == new_w


def test_update_unknown_field_raises():
    reg, _ = _registry()
    reg.register("bot")
    with pytest.raises(ValueError, match="Unknown config field"):
        reg.update("bot", nonexistent_field=True)


def test_update_unregistered_raises():
    reg, _ = _registry()
    with pytest.raises(ValueError, match="not registered"):
        reg.update("ghost", iqs_threshold=0.5)


def test_list_agents_empty():
    reg, _ = _registry()
    assert reg.list_agents() == []


def test_list_agents_multiple():
    reg, _ = _registry()
    reg.register("a")
    reg.register("b")
    reg.register("c")
    agents = reg.list_agents()
    assert set(agents) == {"a", "b", "c"}


def test_get_config_unregistered_raises():
    reg, _ = _registry()
    with pytest.raises(ValueError, match="not registered"):
        reg.get_config("ghost")


# ── Scoring ───────────────────────────────────────────────────────────────────

def test_score_registered_agent():
    reg, _ = _registry()
    reg.register("bot")
    result = reg.score("bot", query="q", response="r")
    assert result.details["agent"] == "bot"


def test_score_unregistered_default_mode():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor, strict=False)
    result = reg.score("unknown", query="q", response="r")
    assert result.details["agent"] == "unknown"


def test_score_unregistered_strict_raises():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor, strict=True)
    with pytest.raises(ValueError, match="not registered"):
        reg.score("unknown", query="q", response="r")


def test_agent_info_in_result_details():
    reg, _ = _registry()
    w = {"completeness": 0.5, "relevance": 0.5}
    reg.register("bot", weights=w, iqs_threshold=0.8)
    result = reg.score("bot", query="q", response="r")
    assert result.details["agent"] == "bot"
    assert result.details["agent_config"]["iqs_threshold"] == 0.8
    assert "completeness" in result.details["agent_config"]["weights"]


def test_custom_weights_affect_iqs():
    """Two agents with different weights on same response → different IQS."""
    # groundedness is low (0.3), completeness is high (0.9)
    auditor = _mock_auditor(groundedness=0.3, completeness=0.9,
                            relevance=0.8, consistency=0.8, confidence=0.8)
    reg = AgentRegistry(auditor)
    # agent1 heavily weights completeness (high)
    reg.register("completeness_heavy",
                 weights={"groundedness": 0.0, "completeness": 0.8,
                          "relevance": 0.1, "consistency": 0.05, "confidence": 0.05})
    # agent2 heavily weights groundedness (low)
    reg.register("groundedness_heavy",
                 weights={"groundedness": 0.8, "completeness": 0.0,
                          "relevance": 0.1, "consistency": 0.05, "confidence": 0.05})
    r1 = reg.score("completeness_heavy", query="q", response="r", context=["ctx"])
    r2 = reg.score("groundedness_heavy", query="q", response="r", context=["ctx"])
    assert r1.iqs > r2.iqs


def test_score_uses_effective_weights_not_auditor_weights():
    """IQS in result matches what compute_iqs gives with agent weights."""
    from scroot.composite import DEFAULT_WEIGHTS
    auditor = _mock_auditor(groundedness=0.6, completeness=0.7,
                            relevance=0.5, consistency=0.9, confidence=0.4)
    reg = AgentRegistry(auditor)
    custom = {"groundedness": 0.5, "completeness": 0.3, "relevance": 0.1,
              "consistency": 0.05, "confidence": 0.05}
    reg.register("bot", weights=custom)
    result = reg.score("bot", query="q", response="r", context=["ctx"])
    effective = dict(DEFAULT_WEIGHTS)
    effective.update(custom)
    expected_iqs = compute_iqs(0.6, 0.7, 0.5, 0.9, 0.4, weights=effective)
    assert abs(result.iqs - expected_iqs) < 1e-6


def test_score_does_not_mutate_auditor_weights():
    auditor = _mock_auditor()
    auditor.weights = {"completeness": 0.5}  # some initial value
    reg = AgentRegistry(auditor)
    reg.register("bot", weights={"groundedness": 0.9})
    reg.score("bot", query="q", response="r")
    assert auditor.weights == {"completeness": 0.5}  # unchanged


def test_context_required_warning():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("strict_bot", context_required=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.score("strict_bot", query="q", response="r", context=None)
        assert len(w) == 1
        assert "requires context" in str(w[0].message)


def test_context_required_no_warning_when_context_provided():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("strict_bot", context_required=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.score("strict_bot", query="q", response="r", context=["ctx"])
        assert len(w) == 0


def test_score_without_agent_uses_default():
    """Duck-type compatibility: score() without agent arg uses '_default'."""
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    result = reg.score(query="q", response="r")
    assert result.details["agent"] == "_default"


# ── Batch ─────────────────────────────────────────────────────────────────────

def test_score_batch_routes_by_agent():
    auditor = MagicMock()
    auditor.score.side_effect = lambda query, response, context=None: _fake_result()
    reg = AgentRegistry(auditor)
    reg.register("a", iqs_threshold=0.6)
    reg.register("b", iqs_threshold=0.8)
    items = [
        {"agent": "a", "query": "q", "response": "r"},
        {"agent": "b", "query": "q", "response": "r"},
        {"agent": "a", "query": "q", "response": "r"},
    ]
    results = reg.score_batch(items)
    assert len(results) == 3
    assert results[0].details["agent"] == "a"
    assert results[1].details["agent"] == "b"
    assert results[2].details["agent"] == "a"


def test_score_batch_missing_agent_key():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    items = [{"query": "q", "response": "r"}]
    results = reg.score_batch(items)
    assert results[0].details["agent"] == "_default"


def test_score_batch_preserves_order():
    auditor = MagicMock()
    call_order = []
    def side(query, response, context=None):
        call_order.append(query)
        return _fake_result()
    auditor.score.side_effect = side
    reg = AgentRegistry(auditor)
    items = [{"agent": "x", "query": f"q{i}", "response": "r"} for i in range(5)]
    results = reg.score_batch(items)
    assert len(results) == 5
    assert call_order == [f"q{i}" for i in range(5)]


# ── Statistics ────────────────────────────────────────────────────────────────

def test_stats_accumulate():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("bot")
    for _ in range(5):
        reg.score("bot", query="q", response="r")
    assert reg.get_stats("bot")["count"] == 5


def test_stats_mean_iqs():
    auditor = _mock_auditor(groundedness=0.8, completeness=0.8,
                            relevance=0.8, consistency=0.8, confidence=0.8)
    reg = AgentRegistry(auditor)
    reg.register("bot")
    for _ in range(4):
        reg.score("bot", query="q", response="r")
    stats = reg.get_stats("bot")
    assert stats["count"] == 4
    assert abs(stats["mean_iqs"] - stats["min_iqs"]) < 1e-4  # all same


def test_stats_flag_counts():
    auditor = _mock_auditor(flags=["hallucination_risk"])
    reg = AgentRegistry(auditor)
    reg.register("bot")
    reg.score("bot", query="q", response="r")
    stats = reg.get_stats("bot")
    assert stats["flag_counts"].get("hallucination_risk") == 1
    assert stats["flagged_count"] == 1


def test_stats_below_threshold():
    # IQS will be around 0.8; set threshold=0.9 so it's below
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("bot", iqs_threshold=0.9)
    reg.score("bot", query="q", response="r")
    stats = reg.get_stats("bot")
    assert stats["below_threshold_count"] == 1
    assert stats["below_threshold_rate"] == 1.0


def test_stats_not_below_threshold_when_above():
    auditor = _mock_auditor(groundedness=0.99, completeness=0.99,
                            relevance=0.99, consistency=0.99, confidence=0.99)
    reg = AgentRegistry(auditor)
    reg.register("bot", iqs_threshold=0.5)
    reg.score("bot", query="q", response="r")
    assert reg.get_stats("bot")["below_threshold_count"] == 0


def test_get_stats_all_agents():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    for name in ["a", "b", "c"]:
        reg.register(name)
        reg.score(name, query="q", response="r")
    stats = reg.get_stats()
    assert set(stats.keys()) == {"a", "b", "c"}
    for name in ["a", "b", "c"]:
        assert stats[name]["count"] == 1


def test_get_stats_unknown_agent_empty():
    reg, _ = _registry()
    assert reg.get_stats("nonexistent") == {}


def test_reset_stats_single():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("bot")
    for _ in range(5):
        reg.score("bot", query="q", response="r")
    reg.reset_stats("bot")
    assert reg.get_stats("bot")["count"] == 0


def test_reset_stats_all():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("a")
    reg.register("b")
    reg.score("a", query="q", response="r")
    reg.score("b", query="q", response="r")
    reg.reset_stats()
    assert reg.get_stats("a")["count"] == 0
    assert reg.get_stats("b")["count"] == 0


def test_reset_stats_nonexistent_agent_no_error():
    reg, _ = _registry()
    reg.reset_stats("ghost")  # should not raise


# ── Thread Safety ─────────────────────────────────────────────────────────────

def test_concurrent_scoring_different_agents():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    for name in ["x", "y", "z"]:
        reg.register(name)

    errors = []

    def score_many(agent_name):
        try:
            for _ in range(20):
                reg.score(agent_name, query="q", response="r")
        except Exception as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(score_many, name) for name in ["x", "y", "z"]]
        for f in futures:
            f.result()

    assert errors == []
    for name in ["x", "y", "z"]:
        assert reg.get_stats(name)["count"] == 20


def test_concurrent_stats_access():
    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("bot")
    errors = []

    def score_loop():
        try:
            for _ in range(10):
                reg.score("bot", query="q", response="r")
        except Exception as e:
            errors.append(e)

    def stats_loop():
        try:
            for _ in range(10):
                reg.get_stats("bot")
        except Exception as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(score_loop) for _ in range(2)]
        futures += [pool.submit(stats_loop) for _ in range(2)]
        for f in futures:
            f.result()

    assert errors == []


# ── Integration with Sampling ─────────────────────────────────────────────────

def test_registry_works_with_sample_and_score():
    """Registry is duck-type compatible with Auditor for sample_and_score."""
    from scroot.sampling import sample_and_score

    auditor = _mock_auditor()
    reg = AgentRegistry(auditor)
    reg.register("support_bot")
    reg.register("code_assistant")

    items = (
        [{"agent": "support_bot", "query": "q", "response": "r"} for _ in range(50)]
        + [{"agent": "code_assistant", "query": "q", "response": "r"} for _ in range(50)]
    )

    result = sample_and_score(
        auditor=reg,
        items=items,
        strategy="random",
        sample_size=20,
        seed=42,
    )

    assert result.sample_size == 20
    assert result.total_population == 100
    # All scored items should have agent info in details
    for si in result.scored_items:
        assert "agent" in si["result"].details
