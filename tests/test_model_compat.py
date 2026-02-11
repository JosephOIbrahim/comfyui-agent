"""Tests for model compatibility matrix."""

import json

from agent.tools import model_compat


class TestIdentifyFamily:
    def test_sd15_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "realisticVisionV60B1.safetensors",
        }))
        assert result["family"] == "sd15"

    def test_sdxl_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "sdxl_base_1.0.safetensors",
        }))
        assert result["family"] == "sdxl"

    def test_flux_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "flux1-dev-fp8.safetensors",
        }))
        assert result["family"] == "flux"

    def test_sd3_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "sd3_medium_incl_clips.safetensors",
        }))
        assert result["family"] == "sd3"

    def test_sd15_controlnet(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "control_v11p_sd15_depth.pth",
        }))
        assert result["family"] == "sd15"

    def test_sdxl_vae(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "sdxl_vae.safetensors",
        }))
        assert result["family"] == "sdxl"

    def test_unknown_model(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "my_custom_model_v3.safetensors",
        }))
        assert result["family"] == "unknown"

    def test_pony_detected_as_sdxl(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "ponyDiffusionV6XL.safetensors",
        }))
        assert result["family"] == "sdxl"

    def test_incompatible_list(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "sdxl_base_1.0.safetensors",
        }))
        assert "Stable Diffusion 1.5" in result["incompatible_with"]
        assert "Flux" in result["incompatible_with"]


class TestCheckCompatibility:
    def test_same_family_compatible(self):
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": [
                "sdxl_base_1.0.safetensors",
                "sdxl_vae.safetensors",
            ],
        }))
        assert result["compatible"] is True
        assert result["family"] == "sdxl"

    def test_mixed_families_incompatible(self):
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": [
                "sdxl_base_1.0.safetensors",
                "control_v11p_sd15_depth.pth",
            ],
        }))
        assert result["compatible"] is False
        assert len(result["conflicts"]) > 0
        assert "sdxl" in result["families_detected"]
        assert "sd15" in result["families_detected"]

    def test_three_family_conflict(self):
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": [
                "sdxl_base.safetensors",
                "control_v11p_sd15_depth.pth",
                "flux1-dev-fp8.safetensors",
            ],
        }))
        assert result["compatible"] is False
        assert len(result["families_detected"]) == 3

    def test_unknown_models_compatible(self):
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": [
                "my_model_a.safetensors",
                "my_model_b.safetensors",
            ],
        }))
        assert result["compatible"] is True
        assert result["family"] == "unknown"

    def test_no_models_error(self):
        from agent.tools import workflow_patch
        workflow_patch._state["current_workflow"] = None
        result = json.loads(model_compat.handle("check_model_compatibility", {}))
        assert "error" in result

    def test_extract_from_workflow(self, tmp_path):
        import json as json_mod
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sdxl_base_1.0.safetensors"},
            },
            "2": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "control_v11p_sd15_depth.pth"},
            },
        }
        path = tmp_path / "test_wf.json"
        path.write_text(json_mod.dumps(wf), encoding="utf-8")

        result = json.loads(model_compat.handle("check_model_compatibility", {
            "workflow_path": str(path),
        }))
        assert result["compatible"] is False
        assert "sdxl" in result["families_detected"]


class TestHelpers:
    def test_extract_models_from_workflow(self):
        wf = {
            "1": {"class_type": "Test", "inputs": {"ckpt_name": "model_a.safetensors"}},
            "2": {"class_type": "Test", "inputs": {"lora_name": "lora_b.safetensors"}},
            "3": {"class_type": "Test", "inputs": {"seed": 42}},
        }
        models = model_compat._extract_models_from_workflow(wf)
        assert "lora_b.safetensors" in models
        assert "model_a.safetensors" in models
        assert len(models) == 2

    def test_all_families_have_required_fields(self):
        for fid, fam in model_compat.MODEL_FAMILIES.items():
            assert "label" in fam, f"{fid} missing label"
            assert "resolution" in fam, f"{fid} missing resolution"
            assert "checkpoint_patterns" in fam, f"{fid} missing patterns"
            assert "incompatible_families" in fam, f"{fid} missing incompatible"


class TestRegistration:
    def test_tools_registered(self):
        names = [t["name"] for t in model_compat.TOOLS]
        assert "check_model_compatibility" in names
        assert "identify_model_family" in names
