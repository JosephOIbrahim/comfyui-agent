"""Hyperagent Meta-Agent — three-tier self-evolution with ratchet validation.

Tier 1: Auto-evolve (recipes, routing weights, memory patterns)
Tier 2: Ratchet-validated (agent prompt tuning, optimization params)
Tier 3: Human-gate (constitution, roles, scoring function, anchors)

The scoring function is a constitutional anchor — it cannot be modified
by the meta-agent itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

EvolutionTier = Literal[1, 2, 3]


# ---------------------------------------------------------------------------
# Tier classification rules
# ---------------------------------------------------------------------------

_TIER_1_CATEGORIES: frozenset[str] = frozenset({
    "recipe", "routing_weight", "memory_pattern",
    "prompt_template", "keyword_mapping",
})

_TIER_2_CATEGORIES: frozenset[str] = frozenset({
    "agent_prompt_tuning", "optimization_param",
    "chain_sequence", "retry_strategy",
    "modification_strategy",
})

_TIER_3_CATEGORIES: frozenset[str] = frozenset({
    "constitution", "role_definition", "scoring_function",
    "anchor_parameter", "authority_rule",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Improvement:
    """A proposed improvement to the agent system."""

    category: str
    description: str
    proposed_change: dict[str, Any]
    tier: EvolutionTier
    rationale: str
    timestamp: float = field(default_factory=time.time)
    status: Literal[
        "proposed", "testing", "accepted", "rejected", "blocked",
    ] = "proposed"
    test_result: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "description": self.description,
            "proposed_change": self.proposed_change,
            "rationale": self.rationale,
            "status": self.status,
            "test_result": self.test_result,
            "tier": self.tier,
            "timestamp": self.timestamp,
        }


@dataclass
class CalibrationRecord:
    """Record of a calibration/evolution event."""

    improvement_id: str
    tier: EvolutionTier
    before: dict[str, Any]
    after: dict[str, Any]
    outcome: Literal["improved", "degraded", "neutral"]
    score_delta: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "after": self.after,
            "before": self.before,
            "improvement_id": self.improvement_id,
            "outcome": self.outcome,
            "score_delta": self.score_delta,
            "tier": self.tier,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Meta-Agent
# ---------------------------------------------------------------------------

class MetaAgent:
    """Three-tier self-evolution meta-agent.

    Tier 1 — auto-evolve: recipes, routing weights, memory patterns.
    Tier 2 — ratchet-validated: agent prompt tuning, optimization params.
    Tier 3 — human-gate: constitution, roles, scoring function, anchors.
    """

    def __init__(self):
        self._history: list[Improvement] = []
        self._calibrations: list[CalibrationRecord] = []
        self._strategy: dict[str, Any] = {
            "max_concurrent_tests": 1,
            "min_improvement_threshold": 0.01,
            "auto_accept_tier1": True,
        }

    @property
    def history(self) -> list[Improvement]:
        return list(self._history)

    @property
    def calibrations(self) -> list[CalibrationRecord]:
        return list(self._calibrations)

    def classify_change(self, category: str) -> EvolutionTier:
        """Determine which tier a proposed change belongs to.

        Tier 1: Auto-evolve (no validation needed)
        Tier 2: Requires ratchet A/B test
        Tier 3: Requires human approval (constitutional anchor)
        """
        if category in _TIER_1_CATEGORIES:
            return 1
        if category in _TIER_2_CATEGORIES:
            return 2
        if category in _TIER_3_CATEGORIES:
            return 3
        # Unknown category defaults to highest tier (safest)
        return 3

    def propose_improvement(
        self,
        category: str,
        description: str,
        proposed_change: dict[str, Any],
        rationale: str = "",
    ) -> Improvement:
        """Propose a system improvement. Tier is auto-classified."""
        tier = self.classify_change(category)
        improvement = Improvement(
            category=category,
            description=description,
            proposed_change=proposed_change,
            tier=tier,
            rationale=rationale,
        )
        self._history.append(improvement)
        return improvement

    def propose_prompt_tuning(
        self,
        agent_name: str,
        current_fragment: str,
        proposed_fragment: str,
        rationale: str = "",
    ) -> Improvement:
        """Propose a prompt tuning change (Tier 2 — requires ratchet test)."""
        return self.propose_improvement(
            category="agent_prompt_tuning",
            description=f"Tune prompt for agent '{agent_name}'",
            proposed_change={
                "agent_name": agent_name,
                "current": current_fragment,
                "proposed": proposed_fragment,
            },
            rationale=rationale,
        )

    def evaluate_improvement(
        self,
        improvement: Improvement,
        score_before: float,
        score_after: float,
    ) -> CalibrationRecord:
        """Evaluate an improvement by comparing before/after scores.

        For Tier 2: this is the ratchet A/B test.
        The scoring function itself is a Tier 3 anchor — cannot be modified.
        """
        delta = score_after - score_before

        if delta > self._strategy["min_improvement_threshold"]:
            outcome = "improved"
            improvement.status = "accepted"
        elif delta < -self._strategy["min_improvement_threshold"]:
            outcome = "degraded"
            improvement.status = "rejected"
        else:
            outcome = "neutral"
            improvement.status = "rejected"

        record = CalibrationRecord(
            improvement_id=id(improvement).__str__(),
            tier=improvement.tier,
            before={"score": score_before},
            after={"score": score_after},
            outcome=outcome,
            score_delta=delta,
        )

        improvement.test_result = record.to_dict()
        self._calibrations.append(record)
        return record

    def can_auto_apply(self, improvement: Improvement) -> bool:
        """Check if an improvement can be auto-applied (Tier 1 only)."""
        return (
            improvement.tier == 1
            and self._strategy.get("auto_accept_tier1", True)
        )

    def requires_human_gate(self, improvement: Improvement) -> bool:
        """Check if an improvement requires human approval (Tier 3)."""
        return improvement.tier == 3

    def requires_ratchet_test(self, improvement: Improvement) -> bool:
        """Check if an improvement requires A/B ratchet testing (Tier 2)."""
        return improvement.tier == 2

    def improve_own_strategy(
        self,
        key: str,
        value: Any,
        rationale: str = "",
    ) -> Improvement:
        """Modify the meta-agent's own strategy (Tier 2 — self-referential).

        The modification strategy itself is a Tier 2 change, ensuring
        the meta-agent can't silently change its own rules without testing.
        """
        current_value = self._strategy.get(key)
        improvement = self.propose_improvement(
            category="modification_strategy",
            description=f"Change strategy '{key}': {current_value} → {value}",
            proposed_change={
                "key": key,
                "current": current_value,
                "proposed": value,
            },
            rationale=rationale,
        )
        return improvement

    def apply_strategy_change(self, improvement: Improvement) -> bool:
        """Apply a strategy change after it's been accepted.

        Only applies if status is 'accepted'.
        """
        if improvement.status != "accepted":
            return False
        change = improvement.proposed_change
        key = change.get("key")
        if key is None:
            return False
        self._strategy[key] = change["proposed"]
        return True

    def get_calibration_stats(self) -> dict[str, Any]:
        """Aggregate statistics over calibration history."""
        if not self._calibrations:
            return {
                "total": 0,
                "improved": 0,
                "degraded": 0,
                "neutral": 0,
                "avg_delta": 0.0,
            }

        outcomes = [c.outcome for c in self._calibrations]
        deltas = [c.score_delta for c in self._calibrations]
        return {
            "total": len(self._calibrations),
            "improved": outcomes.count("improved"),
            "degraded": outcomes.count("degraded"),
            "neutral": outcomes.count("neutral"),
            "avg_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        }

    def get_history_by_tier(self, tier: EvolutionTier) -> list[Improvement]:
        """Return improvements filtered by tier."""
        return [i for i in self._history if i.tier == tier]

    def get_pending(self) -> list[Improvement]:
        """Return all proposed (not yet evaluated) improvements."""
        return [i for i in self._history if i.status == "proposed"]
