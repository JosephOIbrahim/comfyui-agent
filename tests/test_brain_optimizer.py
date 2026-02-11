"""Tests for brain/optimizer.py — GPU-aware performance engineering."""

import copy
import json
from unittest.mock import patch

import pytest

from agent.brain import optimizer
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
    original = copy.deepcopy(workflow_patch._state)
    yield
    workflow_patch._state.update(original)


def _load_sample():
    workflow_patch._state["loaded_path"] = "test.json"
    workflow_patch._state["base_workflow"] = copy.deepcopy(SAMPLE_WORKFLOW)
    workflow_patch._state["current_workflow"] = copy.deepcopy(SAMPLE_WORKFLOW)
    workflow_patch._state["history"] = []
    workflow_patch._state["format"] = "api"


class TestProfileWorkflow:
    def test_profile_no_workflow(self):
        result = json.loads(optimizer.handle("profile_workflow", {}))
        assert "error" in result

    def test_profile_with_workflow(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "NVIDIA GeForce RTX 4090",
            "vram_gb": 24,
            "trt_supported": True,
            "sweet_spots": {"max_resolution_no_tiling": [1536, 1536]},
        }):
            result = json.loads(optimizer.handle("profile_workflow", {}))
        assert result["workflow"]["total_nodes"] == 6
        assert len(result["workflow"]["gpu_heavy_nodes"]) >= 2  # KSampler + VAEDecode
        assert result["gpu"]["name"] == "NVIDIA GeForce RTX 4090"

    def test_profile_detects_resolution(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "test", "vram_gb": 24, "trt_supported": True,
            "sweet_spots": {"max_resolution_no_tiling": [1536, 1536]},
        }):
            result = json.loads(optimizer.handle("profile_workflow", {}))
        resolutions = result["workflow"]["resolutions"]
        assert any(r["width"] == 1024 for r in resolutions)


class TestSuggestOptimizations:
    def test_suggest_no_workflow(self):
        result = json.loads(optimizer.handle("suggest_optimizations", {}))
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
            result = json.loads(optimizer.handle("suggest_optimizations", {}))
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
            result = json.loads(optimizer.handle("suggest_optimizations", {
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
            result = json.loads(optimizer.handle("check_tensorrt_status", {}))
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
            result = json.loads(optimizer.handle("check_tensorrt_status", {}))
        assert result["any_pack_installed"] is True
        assert result["ready"] is True


class TestApplyOptimization:
    def test_apply_vae_tiling(self):
        _load_sample()
        result = json.loads(optimizer.handle("apply_optimization", {
            "optimization_id": "vae_tiling",
        }))
        assert result["applied"] == "vae_tiling"
        assert len(result["nodes_swapped"]) == 1
        # Verify the node was swapped
        wf = workflow_patch._state["current_workflow"]
        assert wf["5"]["class_type"] == "VAEDecodeTiled"

    def test_apply_batch_size(self):
        _load_sample()
        with patch("agent.brain.optimizer._detect_gpu", return_value={
            "detected_name": "test", "vram_gb": 24, "trt_supported": True,
            "sweet_spots": {"sdxl_batch": 2, "sd15_batch": 4},
        }):
            result = json.loads(optimizer.handle("apply_optimization", {
                "optimization_id": "batch_size",
            }))
        assert result["applied"] == "batch_size"
        wf = workflow_patch._state["current_workflow"]
        assert wf["4"]["inputs"]["batch_size"] == 2  # SDXL detected

    def test_apply_step_optimization(self):
        _load_sample()
        result = json.loads(optimizer.handle("apply_optimization", {
            "optimization_id": "step_optimization",
            "params": {"steps": 15},
        }))
        assert result["applied"] == "step_optimization"
        wf = workflow_patch._state["current_workflow"]
        assert wf["3"]["inputs"]["steps"] == 15

    def test_apply_sampler_efficiency(self):
        _load_sample()
        result = json.loads(optimizer.handle("apply_optimization", {
            "optimization_id": "sampler_efficiency",
            "params": {"sampler": "dpmpp_2m", "scheduler": "karras"},
        }))
        assert result["applied"] == "sampler_efficiency"
        wf = workflow_patch._state["current_workflow"]
        assert wf["3"]["inputs"]["sampler_name"] == "dpmpp_2m"
        assert wf["3"]["inputs"]["scheduler"] == "karras"

    def test_apply_no_workflow(self):
        result = json.loads(optimizer.handle("apply_optimization", {
            "optimization_id": "vae_tiling",
        }))
        assert "error" in result

    def test_apply_unknown(self):
        _load_sample()
        result = json.loads(optimizer.handle("apply_optimization", {
            "optimization_id": "nonexistent",
        }))
        assert "error" in result
