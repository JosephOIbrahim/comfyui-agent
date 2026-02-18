"""Tests for context engineering features:
- Session-aware system prompt
- Structured compaction
- Tool result formats
- Semantic composition tools
- Pre-execution validation
- Dynamic knowledge loading
- Observation masking
- Workflow templates
- Parallel tool execution
"""

import copy
import json

import pytest

from agent.system_prompt import build_system_prompt, _detect_relevant_knowledge
from agent.main import _summarize_dropped, _mask_processed_results
from agent.tools import workflow_patch, workflow_templates, comfy_api, comfy_inspect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_workflow_state():
    """Reset workflow_patch state between tests."""
    original = copy.deepcopy(workflow_patch._state)
    yield
    workflow_patch._state.update(original)


SAMPLE_WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello", "clip": ["1", 1]}},
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 42, "steps": 20, "cfg": 7.0,
        "model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
        "latent_image": ["4", 0],
    }},
    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
}


def _load_sample_workflow():
    """Helper: load sample workflow into workflow_patch state."""
    workflow_patch._state["loaded_path"] = "test.json"
    workflow_patch._state["base_workflow"] = copy.deepcopy(SAMPLE_WORKFLOW)
    workflow_patch._state["current_workflow"] = copy.deepcopy(SAMPLE_WORKFLOW)
    workflow_patch._state["history"] = []
    workflow_patch._state["format"] = "api"


# ---------------------------------------------------------------------------
# 1. Session-aware system prompt
# ---------------------------------------------------------------------------

class TestSessionAwarePrompt:
    def test_no_session(self):
        prompt = build_system_prompt()
        assert "ComfyUI co-pilot" in prompt
        assert "Session Context" not in prompt

    def test_with_recommendations(self):
        from unittest.mock import patch as mock_patch
        ctx = {
            "name": "test-session",
            "notes": [],
            "workflow": {},
        }
        mock_recs = json.dumps({
            "recommendations": [
                {"category": "sampler", "recommendation": "DPM++ 2M Karras works best", "confidence": 0.9},
                {"category": "model", "recommendation": "SDXL base produces best results", "confidence": 0.8},
                {"category": "low", "recommendation": "Try Euler", "confidence": 0.3},
            ],
        })
        with mock_patch("agent.system_prompt.memory_handle", create=True):
            # Patch at import location inside the function
            with mock_patch("agent.brain.memory.handle", return_value=mock_recs):
                prompt = build_system_prompt(session_context=ctx)
        assert "Recommendations from Past Sessions" in prompt
        assert "DPM++ 2M Karras works best" in prompt
        assert "SDXL base produces best results" in prompt
        # Low confidence should be filtered out
        assert "Try Euler" not in prompt

    def test_without_recommendations(self):
        # No session name → no recommendations injected
        prompt = build_system_prompt(session_context=None)
        assert "Recommendations from Past Sessions" not in prompt

    def test_with_session_notes(self):
        ctx = {
            "name": "test-session",
            "notes": [{"text": "User prefers SDXL"}],
            "workflow": {},
        }
        prompt = build_system_prompt(session_context=ctx)
        assert "Session Context" in prompt
        assert "test-session" in prompt
        assert "User prefers SDXL" in prompt

    def test_with_workflow_info(self):
        ctx = {
            "name": "test",
            "notes": [],
            "workflow": {
                "loaded_path": "/path/to/workflow.json",
                "format": "api",
                "history_depth": 3,
            },
        }
        prompt = build_system_prompt(session_context=ctx)
        assert "/path/to/workflow.json" in prompt
        assert "Patches applied: 3" in prompt

    def test_notes_limited_to_10(self):
        ctx = {
            "name": "test",
            "notes": [{"text": f"Note {i}"} for i in range(20)],
            "workflow": {},
        }
        prompt = build_system_prompt(session_context=ctx)
        # Should only include last 10
        assert "Note 19" in prompt
        assert "Note 10" in prompt
        assert "Note 9" not in prompt


# ---------------------------------------------------------------------------
# 2. Structured compaction
# ---------------------------------------------------------------------------

class TestStructuredCompaction:
    def test_summarize_extracts_user_requests(self):
        messages = [
            {"role": "user", "content": "What models do I have?"},
            {"role": "user", "content": "Search for SDXL LoRAs"},
        ]
        summary = _summarize_dropped(messages)
        assert "What models do I have?" in summary
        assert "Search for SDXL LoRAs" in summary

    def test_summarize_extracts_tool_names(self):
        messages = [
            {
                "role": "assistant",
                "content": [type("Block", (), {"type": "tool_use", "name": "get_all_nodes"})()],
            },
        ]
        summary = _summarize_dropped(messages)
        assert "get_all_nodes" in summary

    def test_summarize_extracts_workflow_path(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": json.dumps({"loaded_path": "/my/workflow.json"})},
                ],
            },
        ]
        summary = _summarize_dropped(messages)
        assert "/my/workflow.json" in summary

    def test_summarize_empty_messages(self):
        summary = _summarize_dropped([])
        assert "Context Summary" in summary


# ---------------------------------------------------------------------------
# 3. Tool result formats (get_all_nodes)
# ---------------------------------------------------------------------------

class TestToolResultFormats:
    def test_get_all_nodes_names_only(self):
        from unittest.mock import patch
        mock_info = {
            "KSampler": {"category": "sampling", "display_name": "KSampler"},
            "CLIPTextEncode": {"category": "conditioning", "display_name": "CLIP Text Encode"},
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_info):
            result = json.loads(comfy_api.handle("get_all_nodes", {"format": "names_only"}))
        assert result["count"] == 2
        assert isinstance(result["nodes"], list)
        assert "KSampler" in result["nodes"]

    def test_get_all_nodes_summary(self):
        from unittest.mock import patch
        mock_info = {
            "KSampler": {"category": "sampling", "display_name": "KSampler"},
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_info):
            result = json.loads(comfy_api.handle("get_all_nodes", {"format": "summary"}))
        assert isinstance(result["nodes"], dict)
        assert "category" in result["nodes"]["KSampler"]

    def test_get_all_nodes_full(self):
        from unittest.mock import patch
        mock_info = {
            "KSampler": {
                "category": "sampling",
                "display_name": "KSampler",
                "description": "Sampler node",
                "input": {"required": {"seed": ["INT"], "steps": ["INT"]}},
                "output": ["LATENT"],
            },
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_info):
            result = json.loads(comfy_api.handle("get_all_nodes", {"format": "full"}))
        assert "input_types" in result["nodes"]["KSampler"]
        assert "output_types" in result["nodes"]["KSampler"]

    def test_list_models_names_only(self, tmp_path):
        from unittest.mock import patch
        model_dir = tmp_path / "checkpoints"
        model_dir.mkdir()
        (model_dir / "model1.safetensors").write_bytes(b"x" * 100)
        (model_dir / "model2.safetensors").write_bytes(b"x" * 200)

        with patch("agent.tools.comfy_inspect.MODELS_DIR", tmp_path):
            result = json.loads(comfy_inspect.handle(
                "list_models", {"model_type": "checkpoints", "format": "names_only"}
            ))
        assert result["count"] == 2
        assert isinstance(result["models"], list)
        assert "model1.safetensors" in result["models"]


# ---------------------------------------------------------------------------
# 4. Semantic composition tools
# ---------------------------------------------------------------------------

class TestAddNode:
    def test_add_node(self):
        _load_sample_workflow()
        result = json.loads(workflow_patch.handle("add_node", {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0]},
        }))
        assert result["added"] is True
        assert result["class_type"] == "VAEDecode"
        # Should get a new ID (max existing is 4, so next is 5)
        assert result["node_id"] == "5"
        assert result["total_nodes"] == 5

    def test_add_node_no_workflow(self):
        result = json.loads(workflow_patch.handle("add_node", {
            "class_type": "VAEDecode",
        }))
        assert "error" in result

    def test_add_node_undoable(self):
        _load_sample_workflow()
        workflow_patch.handle("add_node", {"class_type": "VAEDecode"})
        assert len(workflow_patch._state["current_workflow"]) == 5
        workflow_patch.handle("undo_workflow_patch", {})
        assert len(workflow_patch._state["current_workflow"]) == 4


class TestConnectNodes:
    def test_connect(self):
        _load_sample_workflow()
        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "3",
            "from_output": 0,
            "to_node": "2",
            "to_input": "conditioning",
        }))
        assert result["connected"] is True
        # Verify the connection was set
        assert workflow_patch._state["current_workflow"]["2"]["inputs"]["conditioning"] == ["3", 0]

    def test_connect_invalid_source(self):
        _load_sample_workflow()
        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "99",
            "from_output": 0,
            "to_node": "2",
            "to_input": "clip",
        }))
        assert "error" in result
        assert "99" in result["error"]

    def test_connect_undoable(self):
        _load_sample_workflow()
        old_clip = workflow_patch._state["current_workflow"]["2"]["inputs"]["clip"]
        workflow_patch.handle("connect_nodes", {
            "from_node": "3",
            "from_output": 0,
            "to_node": "2",
            "to_input": "clip",
        })
        workflow_patch.handle("undo_workflow_patch", {})
        assert workflow_patch._state["current_workflow"]["2"]["inputs"]["clip"] == old_clip


class TestSetInput:
    def test_set_input(self):
        _load_sample_workflow()
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "3",
            "input_name": "seed",
            "value": 999,
        }))
        assert result["set"] is True
        assert result["old_value"] == 42
        assert result["new_value"] == 999
        assert workflow_patch._state["current_workflow"]["3"]["inputs"]["seed"] == 999

    def test_set_input_invalid_node(self):
        _load_sample_workflow()
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "99",
            "input_name": "seed",
            "value": 1,
        }))
        assert "error" in result

    def test_set_input_undoable(self):
        _load_sample_workflow()
        workflow_patch.handle("set_input", {
            "node_id": "3",
            "input_name": "steps",
            "value": 50,
        })
        workflow_patch.handle("undo_workflow_patch", {})
        assert workflow_patch._state["current_workflow"]["3"]["inputs"]["steps"] == 20


# ---------------------------------------------------------------------------
# 6. Dynamic knowledge loading
# ---------------------------------------------------------------------------

class TestDynamicKnowledge:
    def test_no_triggers(self):
        triggers = _detect_relevant_knowledge(None)
        assert len(triggers) == 0

    def test_controlnet_detected_from_workflow(self):
        ctx = {
            "workflow": {
                "current_workflow": {
                    "1": {"class_type": "ControlNetLoader"},
                    "2": {"class_type": "ControlNetApply"},
                },
            },
            "notes": [],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "controlnet_patterns" in triggers

    def test_flux_detected_from_notes(self):
        ctx = {
            "workflow": {},
            "notes": [{"text": "Using Flux dev model for generation"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "flux_specifics" in triggers

    def test_video_detected(self):
        ctx = {
            "workflow": {
                "current_workflow": {
                    "1": {"class_type": "ADE_AnimateDiffLoaderWithContext"},
                },
            },
            "notes": [],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "video_workflows" in triggers

    def test_recipe_detected(self):
        ctx = {
            "workflow": {},
            "notes": [{"text": "create workflow from scratch for portraits"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "common_recipes" in triggers

    def test_3d_detected_from_workflow(self):
        ctx = {
            "workflow": {
                "current_workflow": {
                    "1": {"class_type": "Hunyuan3DLoader"},
                    "2": {"class_type": "SaveGLB"},
                },
            },
            "notes": [],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers

    def test_3d_detected_from_notes(self):
        ctx = {
            "workflow": {},
            "notes": [{"text": "generate a 3D mesh of a column"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers

    def test_audio_detected_from_workflow(self):
        ctx = {
            "workflow": {
                "current_workflow": {
                    "1": {"class_type": "CosyVoiceLoader"},
                    "2": {"class_type": "SaveAudio"},
                },
            },
            "notes": [],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "audio_workflows" in triggers

    def test_audio_detected_from_notes(self):
        ctx = {
            "workflow": {},
            "notes": [{"text": "add TTS narration to the scene"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "audio_workflows" in triggers

    def test_wan_triggers_3d(self):
        ctx = {
            "workflow": {},
            "notes": [{"text": "use Wan2.1 for multi-view generation"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers


# ---------------------------------------------------------------------------
# 7. Observation masking
# ---------------------------------------------------------------------------

class TestObservationMasking:
    def test_masks_old_large_results(self):
        messages = [
            # Old turn: user sent tool results
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "x" * 3000},
            ]},
            # Model processed it
            {"role": "assistant", "content": "I see the results."},
            # New turn: fresh tool results (should NOT be masked)
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": "y" * 3000},
            ]},
        ]
        masked = _mask_processed_results(messages)
        # Old result should be masked
        old_content = masked[0]["content"][0]["content"]
        assert "Processed result" in old_content
        assert len(old_content) < 3000
        # New result should be preserved
        new_content = masked[2]["content"][0]["content"]
        assert new_content == "y" * 3000

    def test_preserves_small_results(self):
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "small result"},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": "another"},
            ]},
        ]
        masked = _mask_processed_results(messages)
        assert masked[0]["content"][0]["content"] == "small result"

    def test_no_masking_with_few_messages(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _mask_processed_results(messages)
        assert result is messages

    def test_does_not_mutate_original(self):
        original_content = "x" * 3000
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": original_content},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": "new"},
            ]},
        ]
        _mask_processed_results(messages)
        assert messages[0]["content"][0]["content"] == original_content


# ---------------------------------------------------------------------------
# 8. Workflow templates
# ---------------------------------------------------------------------------

class TestWorkflowTemplates:
    def test_list_templates(self):
        result = json.loads(workflow_templates.handle("list_workflow_templates", {}))
        assert result["count"] >= 4
        names = [t["name"] for t in result["templates"]]
        assert "txt2img_sdxl" in names
        assert "img2img" in names

    def test_load_template(self):
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "txt2img_sdxl",
        }))
        assert result["loaded"] == "txt2img_sdxl"
        assert result["node_count"] == 7
        # Should be loaded in workflow_patch state
        assert workflow_patch._state["current_workflow"] is not None
        assert workflow_patch._state["loaded_path"] == "template:txt2img_sdxl"

    def test_load_template_not_found(self):
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "nonexistent",
        }))
        assert "error" in result
        assert "available" in result

    def test_template_is_editable(self):
        """After loading a template, can we edit it with semantic tools?"""
        workflow_templates.handle("get_workflow_template", {"template": "txt2img_sdxl"})
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "5",
            "input_name": "seed",
            "value": 123,
        }))
        assert result["set"] is True
        assert workflow_patch._state["current_workflow"]["5"]["inputs"]["seed"] == 123

    def test_template_nodes_have_editable_inputs(self):
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "txt2img_sd15",
        }))
        # KSampler (node 5) should list its editable inputs
        assert "editable_inputs" in result["nodes"]["5"]
        assert "seed" in result["nodes"]["5"]["editable_inputs"]
        assert "steps" in result["nodes"]["5"]["editable_inputs"]
