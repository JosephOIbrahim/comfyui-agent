"""Tests for the iterative_refine brain tool — MoE pipeline orchestrator.

This module previously tested the heuristic-based iterative refinement loop.
Phase 5 replaced the implementation with the MoE pipeline orchestrator that
wires Router, Intent Agent, and Verify Agent together. Comprehensive tests
for the new implementation live in test_moe_integration.py.

These tests verify backward-compatible module-level interface, SDK class,
and the P4 validation layer (MoE → Tools grounding).
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from agent.brain.iterative_refine import (
    IterativeRefineAgent,
    TOOLS,
    handle,
    _is_comfyui_available,
    _validate_intent_mutations,
    _extract_parameters_from_workflow,
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


# ---------------------------------------------------------------------------
# P4: Validation helpers
# ---------------------------------------------------------------------------

def _make_mock_intent_spec(mutations=None):
    """Build a mock IntentSpecification with given parameter mutations."""
    spec = MagicMock()
    if mutations is None:
        spec.parameter_mutations = []
    else:
        muts = []
        for target in mutations:
            m = MagicMock()
            m.target = target
            muts.append(m)
        spec.parameter_mutations = muts
    return spec


class TestIsComfyuiAvailable:
    """_is_comfyui_available() tests."""

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_running_true(self, mock_th):
        mock_th.return_value = json.dumps({"running": True})
        assert _is_comfyui_available() is True
        mock_th.assert_called_once_with("is_comfyui_running", {})

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_running_false(self, mock_th):
        mock_th.return_value = json.dumps({"running": False})
        assert _is_comfyui_available() is False

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_exception_returns_false(self, mock_th):
        mock_th.side_effect = ConnectionError("nope")
        assert _is_comfyui_available() is False


class TestValidateIntentMutations:
    """_validate_intent_mutations() tests."""

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_valid_node_and_input(self, mock_th):
        mock_th.return_value = json.dumps({
            "input": {
                "required": {"cfg": ["FLOAT", {"default": 7.0}]},
                "optional": {},
            },
        })
        spec = _make_mock_intent_spec(["KSampler.cfg"])
        results = _validate_intent_mutations(spec)
        assert len(results) == 1
        assert results[0]["status"] == "ok"
        assert results[0]["node_exists"] is True
        assert results[0]["input_exists"] is True

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_unknown_node(self, mock_th):
        mock_th.return_value = json.dumps({"error": "Node type 'Bogus' not found."})
        spec = _make_mock_intent_spec(["Bogus.cfg"])
        results = _validate_intent_mutations(spec)
        assert len(results) == 1
        assert results[0]["status"] == "warning"
        assert results[0]["node_exists"] is False

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_unknown_input(self, mock_th):
        mock_th.return_value = json.dumps({
            "input": {
                "required": {"cfg": ["FLOAT", {}]},
                "optional": {},
            },
        })
        spec = _make_mock_intent_spec(["KSampler.nonexistent_param"])
        results = _validate_intent_mutations(spec)
        assert len(results) == 1
        assert results[0]["status"] == "warning"
        assert results[0]["node_exists"] is True
        assert results[0]["input_exists"] is False

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_caches_node_info(self, mock_th):
        """Same node_class should only call get_node_info once."""
        mock_th.return_value = json.dumps({
            "input": {
                "required": {"cfg": ["FLOAT", {}], "steps": ["INT", {}]},
                "optional": {},
            },
        })
        spec = _make_mock_intent_spec(["KSampler.cfg", "KSampler.steps"])
        results = _validate_intent_mutations(spec)
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)
        # get_node_info called only once despite two mutations
        assert mock_th.call_count == 1

    def test_exception_returns_empty(self):
        """If accessing parameter_mutations raises, return empty list."""
        class BadSpec:
            @property
            def parameter_mutations(self):
                raise RuntimeError("boom")

        results = _validate_intent_mutations(BadSpec())
        assert results == []

    def test_empty_mutations(self):
        spec = _make_mock_intent_spec([])
        results = _validate_intent_mutations(spec)
        assert results == []

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_invalid_target_format(self, mock_th):
        """Target without a dot should produce a warning."""
        spec = _make_mock_intent_spec(["cfg_only"])
        results = _validate_intent_mutations(spec)
        assert len(results) == 1
        assert results[0]["status"] == "warning"
        assert "Invalid target format" in results[0]["message"]
        # No tools_handle call needed for malformed target
        mock_th.assert_not_called()

    @patch("agent.brain.iterative_refine.tools_handle")
    def test_optional_input_found(self, mock_th):
        """Inputs in 'optional' section should also be found."""
        mock_th.return_value = json.dumps({
            "input": {
                "required": {},
                "optional": {"denoise": ["FLOAT", {"default": 1.0}]},
            },
        })
        spec = _make_mock_intent_spec(["KSampler.denoise"])
        results = _validate_intent_mutations(spec)
        assert results[0]["status"] == "ok"
        assert results[0]["input_exists"] is True


class TestExtractParametersFromWorkflow:
    """_extract_parameters_from_workflow() tests."""

    def test_extracts_params(self):
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 7.0,
                    "steps": 20,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.75,
                    "model": ["1", 0],  # connection — should be skipped
                },
            },
        }
        result = _extract_parameters_from_workflow(workflow)
        assert result == {
            "cfg": 7.0,
            "steps": 20,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.75,
        }

    def test_skips_connections(self):
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": ["4", 0],  # connection, not literal
                    "steps": 20,
                },
            },
        }
        result = _extract_parameters_from_workflow(workflow)
        assert result is not None
        assert "cfg" not in result
        assert result["steps"] == 20

    def test_none_input(self):
        assert _extract_parameters_from_workflow(None) is None

    def test_empty_workflow(self):
        assert _extract_parameters_from_workflow({}) is None

    def test_no_extractable_inputs(self):
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "hello", "clip": ["2", 0]},
            },
        }
        assert _extract_parameters_from_workflow(workflow) is None


# ---------------------------------------------------------------------------
# P4: Validation integration tests
# ---------------------------------------------------------------------------

class TestValidationIntegration:
    """Validation wiring into pipeline results."""

    def test_planned_result_has_validation_key(self):
        """Standard planned result should include 'validation' key."""
        result = json.loads(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        assert "validation" in result
        assert isinstance(result["validation"], list)

    def test_error_result_has_validation_key(self):
        """Error results should also include 'validation' key."""
        result = json.loads(handle("iterative_refine", {
            "user_intent": "",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "error"
        assert "validation" in result

    @patch("agent.brain.iterative_refine._is_comfyui_available", return_value=False)
    def test_validation_skipped_when_offline(self, mock_avail):
        """When ComfyUI is offline, validation should be empty list."""
        result = json.loads(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        assert result["validation"] == []

    @patch("agent.brain.iterative_refine._validate_intent_mutations")
    @patch("agent.brain.iterative_refine._is_comfyui_available", return_value=True)
    def test_warnings_surface_in_preconditions(self, mock_avail, mock_validate):
        """Validation warnings should appear in precondition_warnings."""
        mock_validate.return_value = [
            {
                "target": "KSampler.bogus",
                "node_class": "KSampler",
                "input_name": "bogus",
                "node_exists": True,
                "input_exists": False,
                "status": "warning",
                "message": "Input 'bogus' not found on node 'KSampler'",
            },
        ]
        result = json.loads(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        assert len(result["validation"]) == 1
        assert any("bogus" in w for w in result["precondition_warnings"])

    @patch("agent.brain.iterative_refine._validate_intent_mutations")
    @patch("agent.brain.iterative_refine._is_comfyui_available", return_value=True)
    def test_validation_non_blocking_on_exception(self, mock_avail, mock_validate):
        """Even if validation throws, pipeline should still produce a result."""
        mock_validate.side_effect = RuntimeError("validation crash")
        # _validate_intent_mutations has internal try/except so this tests the
        # outer behavior — the mock bypasses the internal catch
        result = json.loads(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        # Pipeline should still complete (the mock raises before returning,
        # but the generation path wraps it in a try/except)
        assert result["status"] in ("planned", "error")


# ---------------------------------------------------------------------------
# P4: Verify Agent enrichment tests
# ---------------------------------------------------------------------------

class TestVerifyEnrichment:
    """Verify Agent receives parameters_used from workflow extraction."""

    @patch("agent.brain.iterative_refine.memory_handle")
    @patch("agent.brain.iterative_refine._is_comfyui_available", return_value=False)
    def test_parameters_used_passed_to_evaluate(self, mock_avail, mock_mem):
        """Verify Agent should receive extracted parameters via evaluation path."""
        mock_mem.return_value = json.dumps({})

        # Use "evaluate" / "rate" to trigger evaluation intent classification
        result = json.loads(handle("iterative_refine", {
            "user_intent": "evaluate the quality of this output",
            "model_id": "sdxl-base",
            "workflow_state": {
                "state": "has_output",
                "3": {
                    "class_type": "KSampler",
                    "inputs": {"cfg": 7.0, "steps": 20},
                },
            },
            "output_analysis": {"quality_score": 0.8, "artifacts": []},
        }))
        # Should complete evaluation path (not crash)
        assert result["status"] == "evaluated"
        assert "validation" in result

    def test_parameters_none_without_workflow(self):
        """Without workflow_state, parameters_used should be None."""
        result = _extract_parameters_from_workflow(None)
        assert result is None

    @patch("agent.brain.iterative_refine.memory_handle")
    @patch("agent.brain.iterative_refine._is_comfyui_available", return_value=False)
    def test_evaluation_path_has_validation_key(self, mock_avail, mock_mem):
        """Evaluation results should include validation key."""
        mock_mem.return_value = json.dumps({})

        result = json.loads(handle("iterative_refine", {
            "user_intent": "how does this look",
            "model_id": "sdxl-base",
            "workflow_state": {"state": "has_output"},
            "output_analysis": {"quality_score": 0.8, "artifacts": []},
        }))
        assert "validation" in result
