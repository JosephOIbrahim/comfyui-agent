"""Tests for agent/stage/hyperagent_tools.py — all mocked, no real I/O."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.stage.hyperagent_tools import TOOLS, handle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: str) -> dict:
    return json.loads(result)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level MetaAgent singleton between tests."""
    import agent.stage.hyperagent_tools as mod
    mod._meta_agent = None
    yield
    mod._meta_agent = None


# ---------------------------------------------------------------------------
# TOOLS schema
# ---------------------------------------------------------------------------

class TestToolSchemas:
    EXPECTED_NAMES = {
        "propose_improvement",
        "check_evolution_tier",
        "get_meta_history",
        "get_calibration_stats",
    }

    def test_tools_is_list(self):
        assert isinstance(TOOLS, list)

    def test_four_tools(self):
        assert len(TOOLS) == 4

    def test_all_have_name(self):
        for t in TOOLS:
            assert "name" in t

    def test_all_have_description(self):
        for t in TOOLS:
            assert "description" in t
            assert len(t["description"]) > 10

    def test_all_have_input_schema(self):
        for t in TOOLS:
            assert "input_schema" in t
            assert t["input_schema"]["type"] == "object"

    def test_expected_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == self.EXPECTED_NAMES


# ---------------------------------------------------------------------------
# propose_improvement
# ---------------------------------------------------------------------------

class TestProposeImprovement:
    def test_tier_1_recipe(self):
        result = _parse(handle("propose_improvement", {
            "category": "recipe",
            "description": "Add SDXL recipe",
            "proposed_change": {"model": "sdxl"},
        }))
        assert result["tier"] == 1
        assert result["can_auto_apply"] is True
        assert result["requires_human_gate"] is False

    def test_tier_3_constitution(self):
        result = _parse(handle("propose_improvement", {
            "category": "constitution",
            "description": "Add new rule",
            "proposed_change": {"rule": "no_delete_all"},
        }))
        assert result["tier"] == 3
        assert result["requires_human_gate"] is True
        assert result["can_auto_apply"] is False

    def test_with_rationale(self):
        result = _parse(handle("propose_improvement", {
            "category": "recipe",
            "description": "test",
            "proposed_change": {},
            "rationale": "Testing rationale",
        }))
        assert result["rationale"] == "Testing rationale"


# ---------------------------------------------------------------------------
# check_evolution_tier
# ---------------------------------------------------------------------------

class TestCheckEvolutionTier:
    def test_tier_1(self):
        result = _parse(handle("check_evolution_tier", {"category": "recipe"}))
        assert result["tier"] == 1
        assert result["tier_label"] == "auto-evolve"

    def test_tier_2(self):
        result = _parse(handle("check_evolution_tier", {
            "category": "agent_prompt_tuning",
        }))
        assert result["tier"] == 2
        assert result["tier_label"] == "ratchet-validated"

    def test_tier_3(self):
        result = _parse(handle("check_evolution_tier", {
            "category": "scoring_function",
        }))
        assert result["tier"] == 3
        assert result["tier_label"] == "human-gate"


# ---------------------------------------------------------------------------
# get_meta_history
# ---------------------------------------------------------------------------

class TestGetMetaHistory:
    def test_empty_history(self):
        result = _parse(handle("get_meta_history", {}))
        assert result["count"] == 0
        assert result["items"] == []

    def test_after_proposal(self):
        handle("propose_improvement", {
            "category": "recipe",
            "description": "test",
            "proposed_change": {},
        })
        result = _parse(handle("get_meta_history", {}))
        assert result["count"] == 1

    def test_filter_by_tier(self):
        handle("propose_improvement", {
            "category": "recipe",
            "description": "tier1",
            "proposed_change": {},
        })
        handle("propose_improvement", {
            "category": "constitution",
            "description": "tier3",
            "proposed_change": {},
        })
        result = _parse(handle("get_meta_history", {"tier": 1}))
        assert result["count"] == 1
        assert result["items"][0]["tier"] == 1

    def test_filter_by_status(self):
        handle("propose_improvement", {
            "category": "recipe",
            "description": "proposed",
            "proposed_change": {},
        })
        result = _parse(handle("get_meta_history", {"status": "proposed"}))
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# get_calibration_stats
# ---------------------------------------------------------------------------

class TestGetCalibrationStats:
    def test_empty_stats(self):
        result = _parse(handle("get_calibration_stats", {}))
        assert result["total"] == 0

    def test_returns_dict(self):
        result = _parse(handle("get_calibration_stats", {}))
        assert "total" in result
        assert "improved" in result
        assert "degraded" in result
        assert "neutral" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_tool(self):
        result = _parse(handle("nonexistent_tool", {}))
        assert "error" in result
