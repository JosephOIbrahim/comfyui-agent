"""Tests for auto_wire module — intelligent model wiring into workflows."""

import copy
import json

import pytest

from agent.tools import auto_wire
from agent.tools.workflow_patch import _get_state


# ---------------------------------------------------------------------------
# Fixtures: mock workflows
# ---------------------------------------------------------------------------

_SIMPLE_CHECKPOINT_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "2": {
        "class_type": "KSampler",
        "inputs": {"seed": 42, "steps": 20, "cfg": 7.0, "model": ["1", 0]},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a cat", "clip": ["1", 1]},
    },
}

_LORA_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sdxl_base.safetensors"},
    },
    "2": {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "old_lora.safetensors",
            "strength_model": 1.0,
            "strength_clip": 1.0,
            "model": ["1", 0],
            "clip": ["1", 1],
        },
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {"model": ["2", 0]},
    },
}

_MULTI_LOADER_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sdxl_base.safetensors"},
    },
    "2": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "sdxl_vae.safetensors"},
    },
    "3": {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "detail_lora.safetensors", "model": ["1", 0]},
    },
    "4": {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": "control_sdxl_depth.safetensors"},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {"model": ["3", 0]},
    },
}

_EMPTY_WORKFLOW = {}

_NO_LOADER_WORKFLOW = {
    "1": {
        "class_type": "KSampler",
        "inputs": {"seed": 42, "steps": 20},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "hello"},
    },
}


def _load_mock_workflow(workflow: dict):
    """Inject a mock workflow into the patch state."""
    s = _get_state()
    with s._lock:
        s["loaded_path"] = "<test>"
        s["format"] = "api"
        s["base_workflow"] = copy.deepcopy(workflow)
        s["current_workflow"] = copy.deepcopy(workflow)
        s["history"] = []
        s["_engine"] = None


def _clear_workflow():
    """Clear any loaded workflow."""
    s = _get_state()
    with s._lock:
        s["loaded_path"] = None
        s["format"] = None
        s["base_workflow"] = None
        s["current_workflow"] = None
        s["history"] = []
        s["_engine"] = None


@pytest.fixture(autouse=True)
def _clean_state():
    """Ensure clean state before and after each test."""
    _clear_workflow()
    yield
    _clear_workflow()


# ---------------------------------------------------------------------------
# wire_model tests
# ---------------------------------------------------------------------------

class TestWireModel:
    def test_wire_checkpoint_simple(self):
        """wire_model with CheckpointLoaderSimple sets ckpt_name."""
        _load_mock_workflow(_SIMPLE_CHECKPOINT_WORKFLOW)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "dreamshaperXL_v2.safetensors",
            "model_type": "checkpoints",
        }))
        assert result["wired"] is True
        assert result["node_id"] == "1"
        assert result["class_type"] == "CheckpointLoaderSimple"
        assert result["input_field"] == "ckpt_name"
        assert result["previous_value"] == "sd_xl_base_1.0.safetensors"
        assert result["new_value"] == "dreamshaperXL_v2.safetensors"

    def test_wire_lora(self):
        """wire_model with LoraLoader sets lora_name."""
        _load_mock_workflow(_LORA_WORKFLOW)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "new_style_lora.safetensors",
            "model_type": "loras",
        }))
        assert result["wired"] is True
        assert result["node_id"] == "2"
        assert result["class_type"] == "LoraLoader"
        assert result["input_field"] == "lora_name"
        assert result["previous_value"] == "old_lora.safetensors"
        assert result["new_value"] == "new_style_lora.safetensors"

    def test_wire_no_loader_returns_error(self):
        """wire_model with no matching loader returns helpful error."""
        _load_mock_workflow(_NO_LOADER_WORKFLOW)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "some_model.safetensors",
            "model_type": "checkpoints",
        }))
        assert "error" in result
        assert "CheckpointLoaderSimple" in result["recommended_node"]

    def test_wire_no_workflow_loaded(self):
        """wire_model with no workflow returns error."""
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "model.safetensors",
            "model_type": "checkpoints",
        }))
        assert "error" in result
        assert "No workflow" in result["error"]

    def test_wire_invalid_model_type(self):
        """wire_model with invalid model_type returns error."""
        _load_mock_workflow(_SIMPLE_CHECKPOINT_WORKFLOW)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "model.safetensors",
            "model_type": "nonexistent",
        }))
        assert "error" in result
        assert "valid_types" in result

    def test_wire_embeddings_returns_guidance(self):
        """wire_model for embeddings explains inline usage."""
        _load_mock_workflow(_SIMPLE_CHECKPOINT_WORKFLOW)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "easynegative.safetensors",
            "model_type": "embeddings",
        }))
        assert "error" in result
        assert "inline" in result["error"].lower()

    def test_wire_family_mismatch_warning(self):
        """wire_model warns on family mismatch (SD1.5 model into SDXL workflow)."""
        _load_mock_workflow(_SIMPLE_CHECKPOINT_WORKFLOW)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "realisticVisionV60B1.safetensors",
            "model_type": "checkpoints",
        }))
        # The wire should succeed (warnings don't block)
        assert result["wired"] is True
        # But should include a compatibility warning
        assert "compatibility_warning" in result
        warning = result["compatibility_warning"]
        assert "mismatch" in warning["warning"].lower()

    def test_wire_multiple_loaders_notes_others(self):
        """wire_model with multiple loaders of same type notes the extras."""
        # Create a workflow with two checkpoint loaders
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model_a.safetensors"},
            },
            "2": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model_b.safetensors"},
            },
        }
        _load_mock_workflow(workflow)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "new_model.safetensors",
            "model_type": "checkpoints",
        }))
        assert result["wired"] is True
        assert result["node_id"] == "1"  # first match
        assert "other_loaders" in result
        assert len(result["other_loaders"]) == 1


# ---------------------------------------------------------------------------
# suggest_wiring tests
# ---------------------------------------------------------------------------

class TestSuggestWiring:
    def test_suggest_multi_loaders(self):
        """suggest_wiring lists all loader nodes in a multi-loader workflow."""
        _load_mock_workflow(_MULTI_LOADER_WORKFLOW)
        result = json.loads(auto_wire.handle("suggest_wiring", {}))
        assert result["loader_count"] == 4
        class_types = {ldr["class_type"] for ldr in result["loaders"]}
        assert "CheckpointLoaderSimple" in class_types
        assert "VAELoader" in class_types
        assert "LoraLoader" in class_types
        assert "ControlNetLoader" in class_types

    def test_suggest_empty_workflow(self):
        """suggest_wiring on empty workflow returns empty list."""
        _load_mock_workflow(_EMPTY_WORKFLOW)
        result = json.loads(auto_wire.handle("suggest_wiring", {}))
        assert result["loader_count"] == 0
        assert result["loaders"] == []

    def test_suggest_no_workflow_loaded(self):
        """suggest_wiring with no workflow returns error."""
        result = json.loads(auto_wire.handle("suggest_wiring", {}))
        assert "error" in result

    def test_suggest_missing_vae(self):
        """suggest_wiring identifies missing VAE loader."""
        _load_mock_workflow(_SIMPLE_CHECKPOINT_WORKFLOW)
        result = json.loads(auto_wire.handle("suggest_wiring", {}))
        missing_types = {m["model_type"] for m in result["missing_core_loaders"]}
        assert "vae" in missing_types

    def test_suggest_loader_details(self):
        """suggest_wiring returns current model values for each loader."""
        _load_mock_workflow(_MULTI_LOADER_WORKFLOW)
        result = json.loads(auto_wire.handle("suggest_wiring", {}))
        # Find the checkpoint loader entry
        ckpt_loaders = [ldr for ldr in result["loaders"] if ldr["model_type"] == "checkpoints"]
        assert len(ckpt_loaders) == 1
        assert ckpt_loaders[0]["current_value"] == "sdxl_base.safetensors"


class TestFilenameValidation:
    """Cycle 29 fix: wire_model must reject filenames with path separators."""

    def test_path_traversal_rejected(self):
        """filename containing path traversal must return an error."""
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "../../etc/passwd",
            "model_type": "checkpoints",
        }))
        assert "error" in result
        assert "path" in result["error"].lower() or "filename" in result["error"].lower()

    def test_absolute_path_rejected(self):
        """filename containing a path separator must be rejected."""
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "/etc/shadow",
            "model_type": "checkpoints",
        }))
        assert "error" in result

    def test_null_byte_rejected(self):
        """filename with null byte must be rejected."""
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "model\x00.safetensors",
            "model_type": "checkpoints",
        }))
        assert "error" in result

    def test_simple_filename_accepted(self):
        """A valid simple filename passes filename validation (may fail for other reasons)."""
        # Load a workflow so we get past filename validation
        _load_mock_workflow(_SIMPLE_CHECKPOINT_WORKFLOW)
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "flux-1-dev.safetensors",
            "model_type": "checkpoints",
        }))
        # Error may occur (no matching loader or wrong family), but NOT about filename/path
        if "error" in result:
            assert "path" not in result["error"].lower() and "filename" not in result["error"].lower()  # Cycle 70: or→and (logic bug: OR let one false positive through)


# ---------------------------------------------------------------------------
# Cycle 41: json.loads guard on patch_handle result
# ---------------------------------------------------------------------------

class TestPatchHandleNonJsonGuard:
    """Cycle 41: auto_wire must handle non-JSON from patch_handle gracefully."""

    def test_non_json_from_set_input_returns_error(self):
        """If set_input returns a non-JSON string, wire_model must return error JSON."""
        from unittest.mock import patch

        _load_mock_workflow(_SIMPLE_CHECKPOINT_WORKFLOW)

        with patch("agent.tools.auto_wire.json") as mock_json:
            # Allow loads to work for everything except the patch_handle result
            import json as _real_json
            call_count = [0]

            def selective_loads(s):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ValueError("non-JSON from patch_handle")
                return _real_json.loads(s)

            mock_json.loads = selective_loads
            mock_json.dumps = _real_json.dumps
            mock_json.JSONDecodeError = _real_json.JSONDecodeError

        # Simpler approach: patch set_input to return garbage
        with patch("agent.tools.workflow_patch.handle", return_value="NOT JSON AT ALL"):
            result = json.loads(auto_wire.handle("wire_model", {
                "filename": "sd15_base.safetensors",
                "model_type": "checkpoints",
            }))
        # Should return a JSON error, not raise
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 47 — wire_model required field guards
# ---------------------------------------------------------------------------

class TestWireModelRequiredFields:
    """wire_model must return structured error when filename or model_type is missing."""

    def test_missing_filename_returns_error(self):
        from agent.tools import auto_wire
        result = json.loads(auto_wire.handle("wire_model", {"model_type": "checkpoints"}))
        assert "error" in result
        assert "filename" in result["error"].lower()

    def test_missing_model_type_returns_error(self):
        from agent.tools import auto_wire
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "flux1-dev.safetensors",
        }))
        assert "error" in result
        assert "model_type" in result["error"].lower()

    def test_empty_filename_returns_error(self):
        from agent.tools import auto_wire
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "", "model_type": "checkpoints",
        }))
        assert "error" in result

    def test_none_model_type_returns_error(self):
        from agent.tools import auto_wire
        result = json.loads(auto_wire.handle("wire_model", {
            "filename": "model.safetensors", "model_type": None,
        }))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 49 — catch-all exception user-friendly context
# ---------------------------------------------------------------------------

class TestAutoWireExceptionContext:
    """Unhandled exceptions in auto_wire must include tool name and hint."""

    def test_exception_includes_tool_name(self):
        """When an unhandled exception fires, result must include tool name."""
        import json
        from unittest.mock import patch
        from agent.tools import auto_wire

        with patch("agent.tools.auto_wire._handle_wire_model", side_effect=RuntimeError("boom")):
            result = json.loads(auto_wire.handle("wire_model", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))
        assert "error" in result
        assert "tool" in result
        assert result["tool"] == "wire_model"
        assert "hint" in result
