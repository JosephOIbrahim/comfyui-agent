"""Tests for agent/stage/workflow_signature.py — no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.workflow_signature import (
    WorkflowSignature,
    from_workflow_json,
    from_stage,
    _classify_resolution,
    _classify_sampler,
    _classify_style,
    _classify_model_family,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def usd_stage():
    """Fresh CognitiveWorkflowStage, skipped if usd-core is not installed."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


@pytest.fixture
def sd15_workflow():
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "v1-5-pruned.safetensors"},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {"sampler_name": "euler_ancestral", "steps": 20},
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
    }


@pytest.fixture
def flux_workflow():
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "flux_dev.safetensors"},
        },
        "2": {
            "class_type": "FluxGuidance",
            "inputs": {"guidance": 1.0},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {"sampler_name": "dpm_2m_sde"},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
    }


@pytest.fixture
def controlnet_lora_workflow():
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "realisticVisionV60.safetensors"},
        },
        "2": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "canny.safetensors"},
        },
        "3": {
            "class_type": "ControlNetApply",
            "inputs": {},
        },
        "4": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "detail_v1.safetensors"},
        },
        "5": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "sharpness_v2.safetensors"},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {"sampler_name": "dpm_2m_karras"},
        },
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 768, "height": 768, "batch_size": 1},
        },
    }


# ---------------------------------------------------------------------------
# WorkflowSignature dataclass
# ---------------------------------------------------------------------------

class TestWorkflowSignatureBasic:
    def test_defaults(self):
        sig = WorkflowSignature()
        assert sig.model_family == "unknown"
        assert sig.resolution_band == "other"
        assert sig.style_target == "unknown"
        assert sig.sampler_class == "other"
        assert sig.controlnet is False
        assert sig.lora_count == 0

    def test_custom_fields(self):
        sig = WorkflowSignature(
            model_family="sdxl",
            resolution_band="1024",
            style_target="photorealistic",
            sampler_class="euler",
            controlnet=True,
            lora_count=3,
        )
        assert sig.model_family == "sdxl"
        assert sig.lora_count == 3
        assert sig.controlnet is True

    def test_frozen(self):
        sig = WorkflowSignature()
        with pytest.raises(AttributeError):
            sig.model_family = "sdxl"

    def test_to_dict(self):
        sig = WorkflowSignature(model_family="flux", lora_count=2)
        d = sig.to_dict()
        assert d["model_family"] == "flux"
        assert d["lora_count"] == 2
        assert len(d) == 6

    def test_to_dict_contains_all_fields(self):
        sig = WorkflowSignature()
        d = sig.to_dict()
        for key in ("model_family", "resolution_band", "style_target",
                     "sampler_class", "controlnet", "lora_count"):
            assert key in d


# ---------------------------------------------------------------------------
# signature_hash
# ---------------------------------------------------------------------------

class TestSignatureHash:
    def test_deterministic(self):
        sig = WorkflowSignature(model_family="sdxl", resolution_band="1024")
        assert sig.signature_hash() == sig.signature_hash()

    def test_same_fields_same_hash(self):
        a = WorkflowSignature(model_family="flux", lora_count=1)
        b = WorkflowSignature(model_family="flux", lora_count=1)
        assert a.signature_hash() == b.signature_hash()

    def test_different_fields_different_hash(self):
        a = WorkflowSignature(model_family="flux")
        b = WorkflowSignature(model_family="sdxl")
        assert a.signature_hash() != b.signature_hash()

    def test_hash_is_64_hex_chars(self):
        sig = WorkflowSignature()
        h = sig.signature_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_controlnet_affects_hash(self):
        a = WorkflowSignature(controlnet=False)
        b = WorkflowSignature(controlnet=True)
        assert a.signature_hash() != b.signature_hash()

    def test_lora_count_affects_hash(self):
        a = WorkflowSignature(lora_count=0)
        b = WorkflowSignature(lora_count=1)
        assert a.signature_hash() != b.signature_hash()


# ---------------------------------------------------------------------------
# match_score
# ---------------------------------------------------------------------------

class TestMatchScore:
    def test_identical_returns_one(self):
        sig = WorkflowSignature(model_family="sdxl", resolution_band="1024")
        assert abs(sig.match_score(sig) - 1.0) < 1e-9

    def test_completely_different_low_score(self):
        a = WorkflowSignature(
            model_family="flux", resolution_band="1024",
            style_target="anime", sampler_class="euler",
            controlnet=True, lora_count=5,
        )
        b = WorkflowSignature(
            model_family="sd15", resolution_band="512",
            style_target="photorealistic", sampler_class="dpm",
            controlnet=False, lora_count=0,
        )
        score = a.match_score(b)
        assert score < 0.2

    def test_partial_match(self):
        a = WorkflowSignature(model_family="sdxl", resolution_band="1024")
        b = WorkflowSignature(model_family="sdxl", resolution_band="512")
        score = a.match_score(b)
        assert 0.5 < score < 1.0

    def test_lora_count_similarity_close(self):
        a = WorkflowSignature(lora_count=2)
        b = WorkflowSignature(lora_count=3)
        score = a.match_score(b)
        # lora contributes 1/(1+1) = 0.5 out of 1.0 for that field
        # plus 5 other matching defaults = 5.5/6
        assert score > 0.8

    def test_lora_count_similarity_far(self):
        a = WorkflowSignature(lora_count=0)
        b = WorkflowSignature(lora_count=10)
        score_near = WorkflowSignature(lora_count=0).match_score(
            WorkflowSignature(lora_count=1)
        )
        score_far = a.match_score(b)
        assert score_near > score_far

    def test_score_in_range(self):
        a = WorkflowSignature(model_family="flux")
        b = WorkflowSignature(model_family="sd3")
        score = a.match_score(b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

class TestClassifyResolution:
    def test_512(self):
        assert _classify_resolution(512, 512) == "512"

    def test_576_still_512_band(self):
        assert _classify_resolution(576, 576) == "512"

    def test_768(self):
        assert _classify_resolution(768, 768) == "768"

    def test_1024(self):
        assert _classify_resolution(1024, 1024) == "1024"

    def test_2048(self):
        assert _classify_resolution(2048, 2048) == "2048"

    def test_4096_is_other(self):
        assert _classify_resolution(4096, 4096) == "other"

    def test_rectangular_uses_max_dim(self):
        assert _classify_resolution(512, 768) == "768"


class TestClassifySampler:
    def test_euler(self):
        assert _classify_sampler("euler_ancestral") == "euler"

    def test_dpm(self):
        assert _classify_sampler("dpm_2m_karras") == "dpm"

    def test_lcm(self):
        assert _classify_sampler("lcm") == "lcm"

    def test_uni(self):
        assert _classify_sampler("uni_pc") == "uni"

    def test_unknown(self):
        assert _classify_sampler("totally_novel_sampler") == "other"

    def test_heun_maps_to_euler(self):
        assert _classify_sampler("heun") == "euler"

    def test_case_insensitive(self):
        assert _classify_sampler("DPM_2M_SDE") == "dpm"


class TestClassifyStyle:
    def test_realistic(self):
        assert _classify_style("realisticVisionV60.safetensors") == "photorealistic"

    def test_anime(self):
        assert _classify_style("animeMix_v3.safetensors") == "anime"

    def test_unknown(self):
        assert _classify_style("model_v1.safetensors") == "unknown"

    def test_case_insensitive(self):
        assert _classify_style("PhotoRealistic_v2.ckpt") == "photorealistic"


class TestClassifyModelFamily:
    def test_flux_by_node(self):
        assert _classify_model_family({"FluxGuidance"}, "") == "flux"

    def test_flux_by_name(self):
        assert _classify_model_family(set(), "flux_dev.safetensors") == "flux"

    def test_sd3_by_node(self):
        assert _classify_model_family({"ControlNetApplySD3"}, "") == "sd3"

    def test_sdxl_by_name(self):
        assert _classify_model_family(set(), "sdxl_base.safetensors") == "sdxl"

    def test_sd15_by_name(self):
        assert _classify_model_family(set(), "v1-5-pruned.safetensors") == "sd15"

    def test_unknown(self):
        assert _classify_model_family(set(), "mymodel.safetensors") == "unknown"

    def test_xl_in_name_is_sdxl(self):
        assert _classify_model_family(set(), "juggernautXL_v9.safetensors") == "sdxl"


# ---------------------------------------------------------------------------
# from_workflow_json
# ---------------------------------------------------------------------------

class TestFromWorkflowJson:
    def test_sd15_workflow(self, sd15_workflow):
        sig = from_workflow_json(sd15_workflow)
        assert sig.model_family == "sd15"
        assert sig.resolution_band == "512"
        assert sig.sampler_class == "euler"
        assert sig.controlnet is False
        assert sig.lora_count == 0

    def test_flux_workflow(self, flux_workflow):
        sig = from_workflow_json(flux_workflow)
        assert sig.model_family == "flux"
        assert sig.resolution_band == "1024"
        assert sig.sampler_class == "dpm"

    def test_controlnet_detected(self, controlnet_lora_workflow):
        sig = from_workflow_json(controlnet_lora_workflow)
        assert sig.controlnet is True

    def test_lora_count(self, controlnet_lora_workflow):
        sig = from_workflow_json(controlnet_lora_workflow)
        assert sig.lora_count == 2

    def test_style_detection(self, controlnet_lora_workflow):
        sig = from_workflow_json(controlnet_lora_workflow)
        assert sig.style_target == "photorealistic"

    def test_empty_workflow(self):
        sig = from_workflow_json({})
        assert sig.model_family == "unknown"
        assert sig.resolution_band == "other"

    def test_minimal_workflow(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        sig = from_workflow_json(wf)
        assert sig.model_family == "unknown"

    def test_missing_inputs_key(self):
        wf = {"1": {"class_type": "KSampler"}}
        sig = from_workflow_json(wf)
        assert sig.sampler_class == "other"

    def test_none_values_in_inputs(self):
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": None},
            }
        }
        sig = from_workflow_json(wf)
        assert sig.model_family == "unknown"

    def test_non_string_ckpt_name_ignored(self):
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": 12345},
            }
        }
        sig = from_workflow_json(wf)
        assert sig.model_family == "unknown"


# ---------------------------------------------------------------------------
# from_stage (requires usd-core)
# ---------------------------------------------------------------------------

class TestFromStage:
    def test_empty_stage_returns_defaults(self, usd_stage):
        sig = from_stage(usd_stage, "nonexistent")
        assert sig.model_family == "unknown"

    def test_sd15_from_stage(self, usd_stage, sd15_workflow):
        from agent.stage.workflow_mapper import workflow_json_to_prims
        workflow_json_to_prims(usd_stage, sd15_workflow, "test_wf")
        sig = from_stage(usd_stage, "test_wf")
        assert sig.model_family == "sd15"
        assert sig.resolution_band == "512"
        assert sig.sampler_class == "euler"

    def test_flux_from_stage(self, usd_stage, flux_workflow):
        from agent.stage.workflow_mapper import workflow_json_to_prims
        workflow_json_to_prims(usd_stage, flux_workflow, "flux_wf")
        sig = from_stage(usd_stage, "flux_wf")
        assert sig.model_family == "flux"
        assert sig.resolution_band == "1024"

    def test_controlnet_from_stage(self, usd_stage, controlnet_lora_workflow):
        from agent.stage.workflow_mapper import workflow_json_to_prims
        workflow_json_to_prims(usd_stage, controlnet_lora_workflow, "cn_wf")
        sig = from_stage(usd_stage, "cn_wf")
        assert sig.controlnet is True
        assert sig.lora_count == 2

    def test_round_trip_hash_matches(self, usd_stage, sd15_workflow):
        from agent.stage.workflow_mapper import workflow_json_to_prims
        json_sig = from_workflow_json(sd15_workflow)
        workflow_json_to_prims(usd_stage, sd15_workflow, "rt")
        stage_sig = from_stage(usd_stage, "rt")
        assert json_sig.signature_hash() == stage_sig.signature_hash()
