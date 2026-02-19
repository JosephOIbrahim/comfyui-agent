"""Tests for the iterative_refine brain tool — MoE pipeline orchestrator.

This module previously tested the heuristic-based iterative refinement loop.
Phase 5 replaced the implementation with the MoE pipeline orchestrator that
wires Router, Intent Agent, and Verify Agent together. Comprehensive tests
for the new implementation live in test_moe_integration.py.

These tests verify backward-compatible module-level interface and SDK class.
"""

import json

import pytest

from agent.brain.iterative_refine import (
    IterativeRefineAgent,
    TOOLS,
    handle,
)
from agent.brain._sdk import BrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    """Create an IterativeRefineAgent with standalone config."""
    cfg = BrainConfig()
    return IterativeRefineAgent(cfg=cfg)


# ---------------------------------------------------------------------------
# Module-level interface
# ---------------------------------------------------------------------------


class TestModuleLevelInterface:
    """Module-level TOOLS and handle() backward compatibility."""

    def test_tools_exported(self):
        """Module-level TOOLS should be accessible."""
        assert len(TOOLS) == 2
        tool_names = [t["name"] for t in TOOLS]
        assert "iterative_refine" in tool_names
        assert "classify_intent" in tool_names

    def test_handle_iterative_refine(self):
        """Module-level handle() dispatches iterative_refine."""
        result = json.loads(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        assert "status" in result
        assert result["status"] == "planned"

    def test_handle_classify_intent(self):
        """Module-level handle() dispatches classify_intent."""
        result = json.loads(handle("classify_intent", {
            "user_intent": "dreamier",
        }))
        assert "intent_type" in result

    def test_handle_unknown_tool(self):
        """Unknown tool name should return error."""
        result = handle("nonexistent", {})
        parsed = json.loads(result)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# SDK class
# ---------------------------------------------------------------------------


class TestIterativeRefineAgentSDK:
    """IterativeRefineAgent SDK class tests."""

    def test_instantiation_default(self):
        agent = IterativeRefineAgent()
        assert agent is not None

    def test_instantiation_custom_config(self):
        cfg = BrainConfig()
        agent = IterativeRefineAgent(cfg=cfg)
        assert agent.cfg is cfg

    def test_tools_class_attribute(self):
        assert IterativeRefineAgent.TOOLS == TOOLS

    def test_handle_dispatches(self, agent):
        result = json.loads(agent.handle("iterative_refine", {
            "user_intent": "sharper",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "planned"

    def test_handle_classify(self, agent):
        result = json.loads(agent.handle("classify_intent", {
            "user_intent": "sharper",
        }))
        assert result["intent_type"] == "modification"

    def test_handle_unknown(self, agent):
        result = json.loads(agent.handle("bogus", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool schema validation
# ---------------------------------------------------------------------------


class TestToolSchemas:
    """Validate tool schemas are well-formed."""

    def test_iterative_refine_schema(self):
        schema = TOOLS[0]
        assert schema["name"] == "iterative_refine"
        assert "description" in schema
        props = schema["input_schema"]["properties"]
        assert "user_intent" in props
        assert "model_id" in props
        assert "workflow_state" in props
        assert "output_analysis" in props
        assert "max_iterations" in props
        assert "schemas" in props
        assert set(schema["input_schema"]["required"]) == {"user_intent", "model_id"}

    def test_classify_intent_schema(self):
        schema = TOOLS[1]
        assert schema["name"] == "classify_intent"
        props = schema["input_schema"]["properties"]
        assert "user_intent" in props
        assert "workflow_state" in props
        assert schema["input_schema"]["required"] == ["user_intent"]


# ---------------------------------------------------------------------------
# Brain registration
# ---------------------------------------------------------------------------


class TestBrainRegistration:
    """Verify module is properly registered in brain/__init__."""

    def test_registered_in_all_brain_tools(self):
        from agent.brain import ALL_BRAIN_TOOLS
        names = [t["name"] for t in ALL_BRAIN_TOOLS]
        assert "iterative_refine" in names
        assert "classify_intent" in names

    def test_handle_from_brain_dispatch(self):
        from agent.brain import handle as brain_handle
        result = json.loads(brain_handle("classify_intent", {
            "user_intent": "dreamier",
        }))
        assert "intent_type" in result

    def test_agent_class_exported(self):
        from agent.brain import IterativeRefineAgent as Exported
        assert Exported is IterativeRefineAgent
