"""Tests for agent/stage/hyperagent.py — all mocked, no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.hyperagent import (
    CalibrationRecord,
    Improvement,
    MetaAgent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def meta():
    return MetaAgent()


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

class TestTierClassification:
    def test_tier_1_recipe(self, meta: MetaAgent):
        assert meta.classify_change("recipe") == 1

    def test_tier_1_routing_weight(self, meta: MetaAgent):
        assert meta.classify_change("routing_weight") == 1

    def test_tier_1_memory_pattern(self, meta: MetaAgent):
        assert meta.classify_change("memory_pattern") == 1

    def test_tier_2_agent_prompt(self, meta: MetaAgent):
        assert meta.classify_change("agent_prompt_tuning") == 2

    def test_tier_2_optimization(self, meta: MetaAgent):
        assert meta.classify_change("optimization_param") == 2

    def test_tier_2_modification_strategy(self, meta: MetaAgent):
        assert meta.classify_change("modification_strategy") == 2

    def test_tier_3_constitution(self, meta: MetaAgent):
        assert meta.classify_change("constitution") == 3

    def test_tier_3_scoring_function(self, meta: MetaAgent):
        assert meta.classify_change("scoring_function") == 3

    def test_tier_3_anchor(self, meta: MetaAgent):
        assert meta.classify_change("anchor_parameter") == 3

    def test_unknown_defaults_tier_3(self, meta: MetaAgent):
        assert meta.classify_change("unknown_category") == 3


# ---------------------------------------------------------------------------
# Propose improvement
# ---------------------------------------------------------------------------

class TestProposeImprovement:
    def test_basic_proposal(self, meta: MetaAgent):
        imp = meta.propose_improvement(
            category="recipe",
            description="Add new recipe",
            proposed_change={"name": "test_recipe"},
        )
        assert imp.tier == 1
        assert imp.status == "proposed"
        assert imp.category == "recipe"
        assert len(meta.history) == 1

    def test_proposal_with_rationale(self, meta: MetaAgent):
        imp = meta.propose_improvement(
            category="constitution",
            description="Add new rule",
            proposed_change={"rule": "test"},
            rationale="Safety improvement",
        )
        assert imp.tier == 3
        assert imp.rationale == "Safety improvement"


class TestProposePromptTuning:
    def test_prompt_tuning_is_tier_2(self, meta: MetaAgent):
        imp = meta.propose_prompt_tuning(
            agent_name="scout",
            current_fragment="You are the Scout.",
            proposed_fragment="You are the Scout. Be thorough.",
        )
        assert imp.tier == 2
        assert imp.category == "agent_prompt_tuning"
        assert "scout" in imp.description

    def test_prompt_tuning_records_change(self, meta: MetaAgent):
        imp = meta.propose_prompt_tuning(
            agent_name="forge",
            current_fragment="old",
            proposed_fragment="new",
        )
        assert imp.proposed_change["current"] == "old"
        assert imp.proposed_change["proposed"] == "new"


# ---------------------------------------------------------------------------
# Evaluate improvement
# ---------------------------------------------------------------------------

class TestEvaluateImprovement:
    def test_improved(self, meta: MetaAgent):
        imp = meta.propose_improvement(
            "recipe", "test", {"x": 1},
        )
        record = meta.evaluate_improvement(imp, 0.5, 0.7)
        assert record.outcome == "improved"
        assert imp.status == "accepted"
        assert record.score_delta == pytest.approx(0.2)

    def test_degraded(self, meta: MetaAgent):
        imp = meta.propose_improvement(
            "recipe", "test", {"x": 1},
        )
        record = meta.evaluate_improvement(imp, 0.7, 0.5)
        assert record.outcome == "degraded"
        assert imp.status == "rejected"

    def test_neutral(self, meta: MetaAgent):
        imp = meta.propose_improvement(
            "recipe", "test", {"x": 1},
        )
        record = meta.evaluate_improvement(imp, 0.5, 0.505)
        assert record.outcome == "neutral"
        assert imp.status == "rejected"

    def test_calibration_recorded(self, meta: MetaAgent):
        imp = meta.propose_improvement("recipe", "test", {})
        meta.evaluate_improvement(imp, 0.5, 0.7)
        assert len(meta.calibrations) == 1

    def test_test_result_attached(self, meta: MetaAgent):
        imp = meta.propose_improvement("recipe", "test", {})
        meta.evaluate_improvement(imp, 0.5, 0.7)
        assert imp.test_result is not None
        assert imp.test_result["outcome"] == "improved"


# ---------------------------------------------------------------------------
# Tier checks
# ---------------------------------------------------------------------------

class TestTierChecks:
    def test_can_auto_apply_tier1(self, meta: MetaAgent):
        imp = meta.propose_improvement("recipe", "test", {})
        assert meta.can_auto_apply(imp) is True

    def test_cannot_auto_apply_tier2(self, meta: MetaAgent):
        imp = meta.propose_improvement("agent_prompt_tuning", "test", {})
        assert meta.can_auto_apply(imp) is False

    def test_requires_human_gate_tier3(self, meta: MetaAgent):
        imp = meta.propose_improvement("constitution", "test", {})
        assert meta.requires_human_gate(imp) is True

    def test_no_human_gate_tier1(self, meta: MetaAgent):
        imp = meta.propose_improvement("recipe", "test", {})
        assert meta.requires_human_gate(imp) is False

    def test_requires_ratchet_tier2(self, meta: MetaAgent):
        imp = meta.propose_improvement("optimization_param", "test", {})
        assert meta.requires_ratchet_test(imp) is True

    def test_no_ratchet_tier1(self, meta: MetaAgent):
        imp = meta.propose_improvement("recipe", "test", {})
        assert meta.requires_ratchet_test(imp) is False


# ---------------------------------------------------------------------------
# Self-modification
# ---------------------------------------------------------------------------

class TestSelfModification:
    def test_improve_own_strategy_is_tier2(self, meta: MetaAgent):
        imp = meta.improve_own_strategy(
            "max_concurrent_tests", 2,
            rationale="Need more parallelism",
        )
        assert imp.tier == 2
        assert imp.category == "modification_strategy"

    def test_apply_strategy_accepted(self, meta: MetaAgent):
        imp = meta.improve_own_strategy("max_concurrent_tests", 5)
        imp.status = "accepted"
        assert meta.apply_strategy_change(imp) is True

    def test_apply_strategy_rejected(self, meta: MetaAgent):
        imp = meta.improve_own_strategy("max_concurrent_tests", 5)
        imp.status = "rejected"
        assert meta.apply_strategy_change(imp) is False


# ---------------------------------------------------------------------------
# Stats and queries
# ---------------------------------------------------------------------------

class TestStatsAndQueries:
    def test_empty_calibration_stats(self, meta: MetaAgent):
        stats = meta.get_calibration_stats()
        assert stats["total"] == 0
        assert stats["avg_delta"] == 0.0

    def test_populated_calibration_stats(self, meta: MetaAgent):
        for score_after in [0.7, 0.3, 0.5]:
            imp = meta.propose_improvement("recipe", "test", {})
            meta.evaluate_improvement(imp, 0.5, score_after)
        stats = meta.get_calibration_stats()
        assert stats["total"] == 3
        assert stats["improved"] == 1
        assert stats["degraded"] == 1

    def test_get_history_by_tier(self, meta: MetaAgent):
        meta.propose_improvement("recipe", "t1", {})
        meta.propose_improvement("constitution", "t3", {})
        meta.propose_improvement("recipe", "t1b", {})
        assert len(meta.get_history_by_tier(1)) == 2
        assert len(meta.get_history_by_tier(3)) == 1

    def test_get_pending(self, meta: MetaAgent):
        imp1 = meta.propose_improvement("recipe", "t1", {})
        imp2 = meta.propose_improvement("recipe", "t2", {})
        meta.evaluate_improvement(imp1, 0.5, 0.7)
        pending = meta.get_pending()
        assert len(pending) == 1
        assert pending[0].description == "t2"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_improvement_to_dict(self):
        imp = Improvement(
            category="recipe",
            description="test",
            proposed_change={"x": 1},
            tier=1,
            rationale="because",
        )
        d = imp.to_dict()
        assert d["category"] == "recipe"
        assert d["tier"] == 1
        assert d["status"] == "proposed"

    def test_calibration_to_dict(self):
        rec = CalibrationRecord(
            improvement_id="abc",
            tier=2,
            before={"score": 0.5},
            after={"score": 0.7},
            outcome="improved",
            score_delta=0.2,
        )
        d = rec.to_dict()
        assert d["outcome"] == "improved"
        assert d["score_delta"] == 0.2
