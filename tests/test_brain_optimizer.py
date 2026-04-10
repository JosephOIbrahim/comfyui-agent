"""Tests for brain/optimizer.py — GPU-aware performance engineering."""

import copy
import json
from unittest.mock import patch

import pytest

from agent.brain import handle
from agent.tools import workflow_patch


SAMPLE_WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl_base.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello", "clip": ["1", 1]}},
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
        "model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
        "latent_image": ["4", 0],
    }},
    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "5": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "test"}},
}


@pytest.fixture(autouse=True)
def reset_workflow_state():
    """Reset workflow_patch state between tests."""
    original = copy.deepcopy(workflow_patch._get_state())
    yield
    workflow_patch._get_state().update(original)


def _load_sample():
    workflow_patch._get_state()["loaded_path"] = "test.json"
    workflow_patch._get_state()["base_workflow"] = copy.deepcopy(SAMPLE_WORKFLOW)
    workflow_patch._get_state()["current_workflow"] = copy.deepcopy(SAMPLE_WORKFLOW)
    workflow_patch._get_state()["history"] = []
    workflow_patch._get_state()["format"] = "api"


class TestProfileWorkflow:
    def test_profile_no_workflow(self):
        result = json.loads(handle("profile_workflow", {}))
        assert "error" in result

    def test_profile_with_workflow(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "NVIDIA GeForce RTX 4090",
            "vram_gb": 24,
            "trt_supported": True,
            "sweet_spots": {"max_resolution_no_tiling": [1536, 1536]},
        }):
            result = json.loads(handle("profile_workflow", {}))
        assert result["workflow"]["total_nodes"] == 6
        assert len(result["workflow"]["gpu_heavy_nodes"]) >= 2  # KSampler + VAEDecode
        assert result["gpu"]["name"] == "NVIDIA GeForce RTX 4090"

    def test_profile_detects_resolution(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "test", "vram_gb": 24, "trt_supported": True,
            "sweet_spots": {"max_resolution_no_tiling": [1536, 1536]},
        }):
            result = json.loads(handle("profile_workflow", {}))
        resolutions = result["workflow"]["resolutions"]
        assert any(r["width"] == 1024 for r in resolutions)


class TestSuggestOptimizations:
    def test_suggest_no_workflow(self):
        result = json.loads(handle("suggest_optimizations", {}))
        assert "error" in result

    def test_suggest_with_workflow(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "NVIDIA GeForce RTX 4090",
            "vram_gb": 24,
            "trt_supported": True,
            "compute_cap": "sm_89",
            "sweet_spots": {"max_resolution_no_tiling": [1536, 1536]},
        }):
            result = json.loads(handle("suggest_optimizations", {}))
        assert result["optimization_count"] >= 5
        # Should include TRT since GPU supports it
        ids = [o["id"] for o in result["optimizations"]]
        assert "tensorrt" in ids
        assert "fp16_precision" in ids

    def test_suggest_with_explicit_gpu(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "manual", "vram_gb": 10, "trt_supported": True,
            "sweet_spots": {},
        }):
            result = json.loads(handle("suggest_optimizations", {
                "gpu_name": "NVIDIA GeForce RTX 4090",
            }))
        assert result["optimization_count"] >= 1


class TestCheckTensorRT:
    def test_check_trt_no_packs(self, tmp_path):
        with patch("agent.brain.optimizer.CUSTOM_NODES_DIR", tmp_path), \
             patch("agent.brain.optimizer.MODELS_DIR", tmp_path), \
             patch("agent.brain.optimizer._detect_gpu", return_value={
                 "detected_name": "RTX 4090", "trt_supported": True,
             }):
            result = json.loads(handle("check_tensorrt_status", {}))
        assert result["any_pack_installed"] is False
        assert result["ready"] is False

    def test_check_trt_with_pack(self, tmp_path):
        # Create fake TRT pack
        trt_dir = tmp_path / "ComfyUI_TensorRT"
        trt_dir.mkdir()
        with patch("agent.brain.optimizer.CUSTOM_NODES_DIR", tmp_path), \
             patch("agent.brain.optimizer.MODELS_DIR", tmp_path), \
             patch("agent.brain.optimizer._detect_gpu", return_value={
                 "detected_name": "RTX 4090", "trt_supported": True,
             }):
            result = json.loads(handle("check_tensorrt_status", {}))
        assert result["any_pack_installed"] is True
        assert result["ready"] is True


class TestApplyOptimization:
    def test_apply_vae_tiling(self):
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "vae_tiling",
        }))
        assert result["applied"] == "vae_tiling"
        assert len(result["nodes_swapped"]) == 1
        # Verify the node was swapped
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["5"]["class_type"] == "VAEDecodeTiled"

    def test_apply_batch_size(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "test", "vram_gb": 24, "trt_supported": True,
            "sweet_spots": {"sdxl_batch": 2, "sd15_batch": 4},
        }):
            result = json.loads(handle("apply_optimization", {
                "optimization_id": "batch_size",
            }))
        assert result["applied"] == "batch_size"
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["4"]["inputs"]["batch_size"] == 2  # SDXL detected

    def test_apply_step_optimization(self):
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": {"steps": 15},
        }))
        assert result["applied"] == "step_optimization"
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["3"]["inputs"]["steps"] == 15

    def test_apply_sampler_efficiency(self):
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "sampler_efficiency",
            "params": {"sampler": "dpmpp_2m", "scheduler": "karras"},
        }))
        assert result["applied"] == "sampler_efficiency"
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["3"]["inputs"]["sampler_name"] == "dpmpp_2m"
        assert wf["3"]["inputs"]["scheduler"] == "karras"

    def test_apply_no_workflow(self):
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "vae_tiling",
        }))
        assert "error" in result

    def test_apply_unknown(self):
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "nonexistent",
        }))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 32: optimizer input validation guards
# ---------------------------------------------------------------------------

def _load_sample():
    """Helper: load the sample workflow into workflow_patch state."""
    workflow_patch._get_state()["current_workflow"] = copy.deepcopy(SAMPLE_WORKFLOW)


class TestOptimizerInputValidation:
    """apply_optimization must validate user-supplied params before applying."""

    def test_batch_size_zero_returns_error(self):
        """batch_size=0 is invalid and must return an error dict."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "batch_size",
            "params": {"batch_size": 0},
        }))
        assert "error" in result
        assert "batch_size" in result["error"]

    def test_batch_size_negative_returns_error(self):
        """Negative batch_size must return error."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "batch_size",
            "params": {"batch_size": -1},
        }))
        assert "error" in result

    def test_batch_size_non_integer_string_returns_error(self):
        """Non-numeric batch_size must return error, not ValueError crash."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "batch_size",
            "params": {"batch_size": "lots"},
        }))
        assert "error" in result

    def test_steps_zero_returns_error(self):
        """steps=0 is invalid and must return an error dict."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": {"steps": 0},
        }))
        assert "error" in result
        assert "steps" in result["error"]

    def test_steps_negative_returns_error(self):
        """Negative steps must return error."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": {"steps": -5},
        }))
        assert "error" in result

    def test_sampler_empty_string_returns_error(self):
        """Empty sampler string must return error."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "sampler_efficiency",
            "params": {"sampler": "", "scheduler": "karras"},
        }))
        assert "error" in result
        assert "sampler" in result["error"]

    def test_scheduler_empty_string_returns_error(self):
        """Empty scheduler string must return error."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "sampler_efficiency",
            "params": {"sampler": "euler", "scheduler": ""},
        }))
        assert "error" in result
        assert "scheduler" in result["error"]

    def test_sampler_none_returns_error(self):
        """None sampler must return error, not AttributeError crash."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "sampler_efficiency",
            "params": {"sampler": None, "scheduler": "karras"},
        }))
        assert "error" in result

    def test_valid_batch_size_still_applies(self):
        """A valid positive batch_size must still apply normally."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "batch_size",
            "params": {"batch_size": 4},
        }))
        assert "error" not in result
        assert result.get("batch_size") == 4

    def test_valid_steps_still_applies(self):
        """A valid positive steps value must still apply normally."""
        _load_sample()
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": {"steps": 25},
        }))
        assert "error" not in result
        assert result.get("steps") == 25


# ---------------------------------------------------------------------------
# Cycle 35: optimizer patch-result validation
# ---------------------------------------------------------------------------

class TestOptimizerPatchResultValidation:
    """When patch_handle returns an error dict, the node must NOT appear in nodes_updated."""

    def test_batch_size_patch_error_not_counted(self):
        """If set_input errors on a node, that node must be excluded from nodes_updated."""
        from agent.brain.optimizer import OptimizerAgent
        from agent.brain._sdk import BrainConfig
        import json as _json

        wf = {
            "1": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1}},
        }

        error_response = _json.dumps({"error": "Node not found"})

        cfg = BrainConfig(
            get_workflow_state=lambda: {"current_workflow": wf},
            patch_handle=lambda name, args: error_response,
        )
        agent = OptimizerAgent(cfg)
        result = _json.loads(agent.handle("apply_optimization", {
            "optimization_id": "batch_size",
            "params": {"batch_size": 4},
        }))
        # No error at the top level — the patch failure is silently skipped
        assert "error" not in result
        # But the failing node must NOT appear in nodes_updated
        assert result.get("nodes_updated") == []

    def test_step_optimization_patch_error_not_counted(self):
        """If set_input errors for steps, that node must be excluded from nodes_updated."""
        from agent.brain.optimizer import OptimizerAgent
        from agent.brain._sdk import BrainConfig
        import json as _json

        wf = {
            "1": {"class_type": "KSampler", "inputs": {"steps": 20}},
        }

        error_response = _json.dumps({"error": "Unknown node"})

        cfg = BrainConfig(
            get_workflow_state=lambda: {"current_workflow": wf},
            patch_handle=lambda name, args: error_response,
        )
        agent = OptimizerAgent(cfg)
        result = _json.loads(agent.handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": {"steps": 30},
        }))
        assert "error" not in result
        assert result.get("nodes_updated") == []

    def test_sampler_efficiency_patch_error_not_counted(self):
        """If sampler set_input errors, that node must be excluded from nodes_updated."""
        from agent.brain.optimizer import OptimizerAgent
        from agent.brain._sdk import BrainConfig
        import json as _json

        wf = {
            "1": {"class_type": "KSampler", "inputs": {"sampler_name": "euler"}},
        }

        error_response = _json.dumps({"error": "Bad input"})

        cfg = BrainConfig(
            get_workflow_state=lambda: {"current_workflow": wf},
            patch_handle=lambda name, args: error_response,
        )
        agent = OptimizerAgent(cfg)
        result = _json.loads(agent.handle("apply_optimization", {
            "optimization_id": "sampler_efficiency",
            "params": {"sampler": "dpmpp_2m", "scheduler": "karras"},
        }))
        assert "error" not in result
        assert result.get("nodes_updated") == []


# ---------------------------------------------------------------------------
# Cycle 46 — apply_optimization required field guard
# ---------------------------------------------------------------------------

class TestApplyOptimizationRequiredField:
    """apply_optimization must return structured error when optimization_id is missing."""

    def test_missing_optimization_id_returns_error(self):
        result = json.loads(handle("apply_optimization", {}))
        assert "error" in result
        assert "optimization_id" in result["error"].lower()

    def test_empty_optimization_id_returns_error(self):
        result = json.loads(handle("apply_optimization", {"optimization_id": ""}))
        assert "error" in result

    def test_none_optimization_id_returns_error(self):
        result = json.loads(handle("apply_optimization", {"optimization_id": None}))
        assert "error" in result

    def test_integer_optimization_id_returns_error(self):
        result = json.loads(handle("apply_optimization", {"optimization_id": 99}))
        assert "error" in result

    def test_valid_optimization_id_passes_guard(self):
        """A valid string optimization_id must not be blocked by the guard."""
        result = json.loads(handle("apply_optimization", {
            "optimization_id": "nonexistent_opt_id_xyz",
        }))
        # Must not be the required-field error
        assert "optimization_id" not in result.get("error", "").lower() or "required" not in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# Cycle 59 — optimization loops log.warning on malformed patch result
# ---------------------------------------------------------------------------

class TestOptimizerSilentExceptionLogging:
    """Cycle 59: when patch_handle returns malformed JSON, optimization loops must log a
    warning and exclude the node from the result (not silently count it as updated)."""

    def _make_cfg(self, wf: dict) -> object:
        from agent.brain._sdk import BrainConfig
        # Return deliberately malformed JSON so _json.loads raises JSONDecodeError
        return BrainConfig(
            get_workflow_state=lambda: {"current_workflow": wf},
            patch_handle=lambda name, args: "not valid json {{{",
        )

    def test_vae_tiling_logs_warning_and_skips_node(self, caplog):
        """vae_tiling: JSONDecodeError on patch result → warning logged, node excluded."""
        import logging
        from agent.brain.optimizer import OptimizerAgent
        wf = {"1": {"class_type": "VAEDecode", "inputs": {"samples": ["0", 0]}}}
        agent = OptimizerAgent(self._make_cfg(wf))
        with caplog.at_level(logging.WARNING, logger="agent.brain.optimizer"):
            result = json.loads(agent.handle("apply_optimization", {"optimization_id": "vae_tiling"}))
        assert result["applied"] == "vae_tiling"
        assert result["nodes_swapped"] == []  # Cycle 59: node must NOT be counted on parse failure
        assert any("vae_tiling" in r.message for r in caplog.records)

    def test_batch_size_logs_warning_and_skips_node(self, caplog):
        """batch_size: JSONDecodeError on patch result → warning logged, node excluded."""
        import logging
        from agent.brain.optimizer import OptimizerAgent
        wf = {"1": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1}}}
        agent = OptimizerAgent(self._make_cfg(wf))
        with caplog.at_level(logging.WARNING, logger="agent.brain.optimizer"):
            result = json.loads(agent.handle("apply_optimization", {
                "optimization_id": "batch_size",
                "params": {"batch_size": 4},
            }))
        assert result["applied"] == "batch_size"
        assert result["nodes_updated"] == []
        assert any("batch_size" in r.message for r in caplog.records)

    def test_step_optimization_logs_warning_and_skips_node(self, caplog):
        """step_optimization: JSONDecodeError on patch result → warning logged, node excluded."""
        import logging
        from agent.brain.optimizer import OptimizerAgent
        wf = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}
        agent = OptimizerAgent(self._make_cfg(wf))
        with caplog.at_level(logging.WARNING, logger="agent.brain.optimizer"):
            result = json.loads(agent.handle("apply_optimization", {
                "optimization_id": "step_optimization",
                "params": {"steps": 15},
            }))
        assert result["applied"] == "step_optimization"
        assert result["nodes_updated"] == []
        assert any("step_optimization" in r.message for r in caplog.records)

    def test_sampler_efficiency_logs_warning_and_skips_node(self, caplog):
        """sampler_efficiency: JSONDecodeError on patch result → warning logged, node excluded."""
        import logging
        from agent.brain.optimizer import OptimizerAgent
        wf = {"1": {"class_type": "KSampler", "inputs": {"sampler_name": "euler"}}}
        agent = OptimizerAgent(self._make_cfg(wf))
        with caplog.at_level(logging.WARNING, logger="agent.brain.optimizer"):
            result = json.loads(agent.handle("apply_optimization", {
                "optimization_id": "sampler_efficiency",
                "params": {"sampler": "dpmpp_2m", "scheduler": "karras"},
            }))
        assert result["applied"] == "sampler_efficiency"
        assert result["nodes_updated"] == []
        assert any("sampler_efficiency" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Cycle 69: apply_optimization params type guard
# ---------------------------------------------------------------------------

class TestApplyOptimizationParamsGuardCycle69:
    """Cycle 69: apply_optimization must reject non-dict params before calling .get()."""

    def _make_cfg(self, wf: dict) -> object:
        from agent.brain._sdk import BrainConfig
        import json as _json
        # Use valid patch_handle so params guard is the only failure path
        def _real_patch(name, args):
            return _json.dumps({"patched": True, "nodes": list(wf.keys())})
        return BrainConfig(
            get_workflow_state=lambda: {"current_workflow": wf},
            patch_handle=_real_patch,
        )

    def test_string_params_returns_error_not_attributeerror(self):
        """params='string' must return JSON error, not AttributeError on .get()."""
        from agent.brain.optimizer import OptimizerAgent
        wf = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}
        agent = OptimizerAgent(self._make_cfg(wf))
        result = json.loads(agent.handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": "15",  # string instead of dict
        }))
        assert "error" in result
        assert "params" in result["error"].lower()

    def test_list_params_returns_error_not_attributeerror(self):
        """params=[1, 2] (list) must return JSON error, not AttributeError on .get()."""
        from agent.brain.optimizer import OptimizerAgent
        wf = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}
        agent = OptimizerAgent(self._make_cfg(wf))
        result = json.loads(agent.handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": [1, 2, 3],  # list instead of dict
        }))
        assert "error" in result
        assert "params" in result["error"].lower()

    def test_dict_params_not_blocked_by_guard(self):
        """Well-formed dict params must not trigger the type guard."""
        from agent.brain.optimizer import OptimizerAgent
        wf = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}
        agent = OptimizerAgent(self._make_cfg(wf))
        result = json.loads(agent.handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": {"steps": 15},
        }))
        assert result.get("error", "") != "params must be a dict (e.g. {\"batch_size\": 2})."
