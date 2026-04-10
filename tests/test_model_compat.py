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

    def test_hunyuan3d_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "hunyuan3d-v2.safetensors",
        }))
        assert result["family"] == "hunyuan3d"
        assert result["label"] == "Hunyuan3D"

    def test_wan_i2v_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "wan2.1_i2v_720p.safetensors",
        }))
        assert result["family"] == "wan"

    def test_wan_t2v_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "wan_t2v_base.safetensors",
        }))
        assert result["family"] == "wan"

    def test_cosyvoice_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "cosyvoice_base.pt",
        }))
        assert result["family"] == "audio"
        assert result["label"] == "Audio/TTS"

    def test_chattts_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "chattts_model.safetensors",
        }))
        assert result["family"] == "audio"

    def test_xtts_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "xtts_v2.pth",
        }))
        assert result["family"] == "audio"

    def test_bark_checkpoint(self):
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "bark_small.pt",
        }))
        assert result["family"] == "audio"

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
        workflow_patch._get_state()["current_workflow"] = None
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

    def test_3d_audio_model_input_names_extracted(self):
        wf = {
            "1": {"class_type": "Loader", "inputs": {
                "model_path": "hunyuan3d.safetensors",
            }},
            "2": {"class_type": "TTS", "inputs": {
                "tts_model": "cosyvoice.pt",
            }},
            "3": {"class_type": "Video", "inputs": {
                "video_model": "wan2.1.safetensors",
            }},
        }
        models = model_compat._extract_models_from_workflow(wf)
        assert "hunyuan3d.safetensors" in models
        assert "cosyvoice.pt" in models
        assert "wan2.1.safetensors" in models

    def test_3d_vs_image_incompatible(self):
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": [
                "hunyuan3d-v2.safetensors",
                "sdxl_base_1.0.safetensors",
            ],
        }))
        assert result["compatible"] is False

    def test_audio_vs_image_incompatible(self):
        result = json.loads(model_compat.handle("check_model_compatibility", {
            "models": [
                "cosyvoice_base.pt",
                "flux1-dev-fp8.safetensors",
            ],
        }))
        assert result["compatible"] is False

    def test_all_families_have_required_fields(self):
        for fid, fam in model_compat.MODEL_FAMILIES.items():
            assert "label" in fam, f"{fid} missing label"
            assert "resolution" in fam, f"{fid} missing resolution"
            assert "checkpoint_patterns" in fam, f"{fid} missing patterns"
            assert "incompatible_families" in fam, f"{fid} missing incompatible"

    def test_family_count(self):
        assert len(model_compat.MODEL_FAMILIES) == 7


class TestRegistration:
    def test_tools_registered(self):
        names = [t["name"] for t in model_compat.TOOLS]
        assert "check_model_compatibility" in names
        assert "identify_model_family" in names


# ---------------------------------------------------------------------------
# Cycle 32: MODEL_FAMILIES unknown family key guard
# ---------------------------------------------------------------------------

class TestUnknownFamilyKeyGuard:
    """MODEL_FAMILIES[f] must not KeyError on unknown family strings."""

    def test_identify_family_unknown_incompatible_family(self):
        """incompatible_with must use .get() — no KeyError on unknown family."""
        import copy
        from unittest.mock import patch
        # Patch real MODEL_FAMILIES so one family's incompatible_families references a ghost key.
        # Keeps all required fields intact (avoids KeyError from other code paths).
        real_families = copy.deepcopy(model_compat.MODEL_FAMILIES)
        real_families["sd15"]["incompatible_families"] = ["GHOST_FAMILY"]
        with patch.object(model_compat, "MODEL_FAMILIES", real_families):
            result = json.loads(model_compat.handle("identify_model_family", {
                "model_name": "sd1.5_base.safetensors",
            }))
        # Must not raise KeyError; incompatible_with entry falls back to the key itself
        assert "incompatible_with" in result
        assert result["incompatible_with"] == ["GHOST_FAMILY"]

    def test_check_compatibility_unknown_family_in_conflict_message(self):
        """_check_compatibility conflict reason must not KeyError on unknown family string."""
        import copy
        from unittest.mock import patch
        # Patch sd15's incompatible_families to include a ghost key.
        # Real family dicts keep all required fields (vae_patterns, controlnet_patterns, etc.)
        # intact so _identify_family doesn't error. Only the ghost incompatible key matters.
        real_families = copy.deepcopy(model_compat.MODEL_FAMILIES)
        real_families["sd15"]["incompatible_families"] = ["GHOST_FAMILY"]
        real_families["sdxl"]["incompatible_families"] = ["sd15"]
        with patch.object(model_compat, "MODEL_FAMILIES", real_families):
            result = json.loads(model_compat.handle("check_model_compatibility", {
                "models": ["sd1.5_base.safetensors", "sdxl_base_1.0.safetensors"],
            }))
        assert result["compatible"] is False
        assert "conflicts" in result
        assert len(result["conflicts"]) > 0
        # reason and message must not KeyError — GHOST_FAMILY falls back to its key name
        assert isinstance(result["conflicts"][0]["reason"], str)
        assert isinstance(result["message"], str)


# ---------------------------------------------------------------------------
# Cycle 45 — identify_model_family required field guard
# ---------------------------------------------------------------------------

class TestIdentifyFamilyRequiredField:
    """identify_model_family must return a structured error when model_name is
    missing, empty, or not a string — never KeyError or AttributeError.
    """

    def test_missing_model_name_returns_error(self):
        """Omitting model_name must return an error dict, not raise."""
        result = json.loads(model_compat.handle("identify_model_family", {}))
        assert "error" in result
        assert "model_name" in result["error"].lower()

    def test_empty_string_model_name_returns_error(self):
        """Empty string model_name must return an error dict."""
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "",
        }))
        assert "error" in result

    def test_none_model_name_returns_error(self):
        """None model_name must return an error dict, not AttributeError."""
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": None,
        }))
        assert "error" in result

    def test_integer_model_name_returns_error(self):
        """Non-string model_name (int) must return an error dict."""
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": 42,
        }))
        assert "error" in result

    def test_valid_model_name_still_works(self):
        """Guard must not break the happy path for a valid model name."""
        result = json.loads(model_compat.handle("identify_model_family", {
            "model_name": "sdxl_base_1.0.safetensors",
        }))
        assert "error" not in result
        assert result["family"] == "sdxl"
