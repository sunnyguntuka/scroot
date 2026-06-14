"""AgentRegistry: per-agent configuration and statistics tracking.

Routes scoring calls through agent-specific configs (custom weights,
thresholds, context requirements) while sharing a single Auditor
instance and its loaded models. One process, one model load, per-agent
metrics.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field

from .composite import DEFAULT_WEIGHTS, compute_iqs
from .result import EntailmentResult


@dataclass
class AgentConfig:
    """Configuration for a single agent.

    Args:
        name: Unique agent identifier.
        weights: Custom IQS weights. Missing keys filled from defaults.
        iqs_threshold: IQS below this value triggers below_threshold tracking.
        context_required: If True, warn when score() called without context.
        metadata: Optional free-form dict (model name, team, description).
    """

    name: str
    weights: dict | None = None
    iqs_threshold: float = 0.7
    context_required: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentStats:
    """Accumulated scoring statistics for a single agent."""

    count: int = 0
    iqs_sum: float = 0.0
    iqs_min: float = 1.0
    iqs_max: float = 0.0
    flagged_count: int = 0
    below_threshold_count: int = 0
    flag_counts: dict = field(default_factory=dict)

    def record(self, result: EntailmentResult, threshold: float) -> None:
        """Update stats with a new scoring result.

        Args:
            result: EntailmentResult from a scoring call.
            threshold: IQS threshold for below_threshold tracking.
        """
        self.count += 1
        self.iqs_sum += result.iqs
        self.iqs_min = min(self.iqs_min, result.iqs)
        self.iqs_max = max(self.iqs_max, result.iqs)
        if result.flags:
            self.flagged_count += 1
            for f in result.flags:
                self.flag_counts[f] = self.flag_counts.get(f, 0) + 1
        if result.iqs < threshold:
            self.below_threshold_count += 1

    def to_dict(self) -> dict:
        """Serialize stats for API or logging."""
        count = max(self.count, 1)
        return {
            "count": self.count,
            "mean_iqs": round(self.iqs_sum / count, 4),
            "min_iqs": self.iqs_min if self.count > 0 else None,
            "max_iqs": self.iqs_max if self.count > 0 else None,
            "flagged_count": self.flagged_count,
            "flag_rate": round(self.flagged_count / count, 4),
            "below_threshold_count": self.below_threshold_count,
            "below_threshold_rate": round(self.below_threshold_count / count, 4),
            "flag_counts": dict(self.flag_counts),
        }


class AgentRegistry:
    """Per-agent configuration and statistics routing layer.

    Wraps an Auditor instance. Each registered agent can have custom
    IQS weights, thresholds, and metadata. Unregistered agents use
    default config unless strict=True.

    The registry is duck-type compatible with Auditor: ``score()`` can be
    called with only ``query``, ``response``, and ``context`` kwargs (the
    ``agent`` parameter defaults to ``"_default"``), so it works as a
    drop-in for ``sample_and_score`` and ``DatabaseConnector``.

    Args:
        auditor: Auditor instance (shared across all agents).
        strict: If True, scoring an unregistered agent raises ValueError.
        default_iqs_threshold: Threshold for unregistered / default agents.
    """

    def __init__(
        self,
        auditor,
        strict: bool = False,
        default_iqs_threshold: float = 0.7,
    ):
        self._auditor = auditor
        self._strict = strict
        self._default_threshold = default_iqs_threshold
        self._configs: dict[str, AgentConfig] = {}
        self._stats: dict[str, AgentStats] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        weights: dict | None = None,
        iqs_threshold: float | None = None,
        context_required: bool = False,
        metadata: dict | None = None,
    ) -> None:
        """Register an agent with optional custom configuration.

        Args:
            name: Unique agent identifier string.
            weights: Custom IQS weights dict. Partial dicts OK -
                missing keys filled from DEFAULT_WEIGHTS.
            iqs_threshold: Custom IQS threshold. Defaults to registry default.
            context_required: If True, warn when score() is called without context.
            metadata: Optional dict (model, team, description, etc).

        Raises:
            ValueError: If an agent with this name is already registered.
        """
        with self._lock:
            if name in self._configs:
                raise ValueError(
                    f"Agent {name!r} already registered. "
                    "Use update() to modify or unregister() first."
                )
            self._configs[name] = AgentConfig(
                name=name,
                weights=weights,
                iqs_threshold=iqs_threshold if iqs_threshold is not None else self._default_threshold,
                context_required=context_required,
                metadata=metadata or {},
            )
            self._stats[name] = AgentStats()

    def update(self, name: str, **kwargs) -> None:
        """Update a registered agent's configuration.

        Args:
            name: Agent identifier.
            **kwargs: AgentConfig fields to update (weights, iqs_threshold,
                context_required, metadata).

        Raises:
            ValueError: If agent is not registered or field name is invalid.
        """
        with self._lock:
            if name not in self._configs:
                raise ValueError(f"Agent {name!r} not registered.")
            config = self._configs[name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    raise ValueError(f"Unknown config field: {key!r}")

    def unregister(self, name: str) -> None:
        """Remove an agent and its accumulated stats.

        Args:
            name: Agent identifier to remove.

        Raises:
            ValueError: If agent is not registered.
        """
        with self._lock:
            if name not in self._configs:
                raise ValueError(f"Agent {name!r} not registered.")
            del self._configs[name]
            self._stats.pop(name, None)

    def list_agents(self) -> list[str]:
        """Return names of all registered agents."""
        with self._lock:
            return list(self._configs.keys())

    def get_config(self, name: str) -> AgentConfig:
        """Get an agent's configuration.

        Args:
            name: Agent identifier.

        Returns:
            AgentConfig for the named agent.

        Raises:
            ValueError: If agent is not registered.
        """
        with self._lock:
            if name not in self._configs:
                raise ValueError(f"Agent {name!r} not registered.")
            return self._configs[name]

    def score(
        self,
        agent: str = "_default",
        *,
        query: str,
        response: str,
        context: list[str] | None = None,
    ) -> EntailmentResult:
        """Score a response using agent-specific configuration.

        The ``agent`` parameter defaults to ``"_default"``, making this
        method duck-type compatible with ``Auditor.score()`` so that the
        registry can be passed to ``sample_and_score()`` or
        ``DatabaseConnector`` directly.

        IQS is recomputed from the raw metric scores using the agent's
        custom weights. The auditor's own weights attribute is never
        mutated, so concurrent calls for different agents are safe.

        Args:
            agent: Agent identifier. Defaults to "_default".
            query: User query.
            response: LLM-generated response.
            context: Optional source context list.

        Returns:
            EntailmentResult with agent-specific IQS and details["agent"] set.

        Raises:
            ValueError: If strict=True and agent is not registered.
        """
        with self._lock:
            config = self._configs.get(agent)

        if config is None:
            if self._strict:
                raise ValueError(
                    f"Agent {agent!r} not registered. "
                    "Call registry.register() first."
                )
            config = AgentConfig(name=agent, iqs_threshold=self._default_threshold)

        if config.context_required and context is None:
            warnings.warn(
                f"Agent {agent!r} requires context but none was provided. "
                "Groundedness will be skipped.",
                stacklevel=2,
            )

        raw = self._auditor.score(query=query, response=response, context=context)

        # Recompute IQS with agent-specific weights (no mutation of auditor state).
        effective_weights = dict(DEFAULT_WEIGHTS)
        if config.weights:
            effective_weights.update(config.weights)

        iqs = compute_iqs(
            raw.groundedness, raw.completeness, raw.relevance,
            raw.consistency, raw.confidence,
            weights=effective_weights,
            mode=self._auditor.iqs_mode,
        )

        result = EntailmentResult(
            groundedness=raw.groundedness,
            completeness=raw.completeness,
            relevance=raw.relevance,
            consistency=raw.consistency,
            confidence=raw.confidence,
            iqs=iqs,
            flags=list(raw.flags),
            details={
                **raw.details,
                "agent": agent,
                "agent_config": {
                    "weights": effective_weights,
                    "iqs_threshold": config.iqs_threshold,
                },
            },
        )

        with self._lock:
            if agent not in self._stats:
                self._stats[agent] = AgentStats()
            self._stats[agent].record(result, config.iqs_threshold)

        return result

    def score_batch(self, items: list[dict]) -> list[EntailmentResult]:
        """Score a batch of responses, each routed to its agent config.

        Items without an "agent" key are scored under "_default".

        Args:
            items: List of dicts with "agent", "query", "response",
                and optionally "context".

        Returns:
            List of EntailmentResult, one per item, in order.
        """
        return [
            self.score(
                agent=item.get("agent", "_default"),
                query=item["query"],
                response=item["response"],
                context=item.get("context"),
            )
            for item in items
        ]

    def get_stats(self, agent: str | None = None) -> dict:
        """Get accumulated scoring statistics.

        Args:
            agent: If provided, return stats for this agent only.
                Returns ``{}`` if agent has no stats yet.
                If None, return stats for all agents.
        """
        with self._lock:
            if agent is not None:
                stats = self._stats.get(agent)
                return stats.to_dict() if stats is not None else {}
            return {name: s.to_dict() for name, s in self._stats.items()}

    def reset_stats(self, agent: str | None = None) -> None:
        """Reset accumulated statistics.

        Args:
            agent: If provided, reset only this agent. If None, reset all.
        """
        with self._lock:
            if agent is not None:
                if agent in self._stats:
                    self._stats[agent] = AgentStats()
            else:
                for name in self._stats:
                    self._stats[name] = AgentStats()
