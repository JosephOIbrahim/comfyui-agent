"""Tests for anchor parameter immunity — constitutional enforcement."""

import pytest

from agent.stage.anchors import (
    ANCHOR_PARAMS,
    AnchorViolationError,
    check_anchor,
    is_anchor,
)


class TestIsAnchor:
    """Test anchor detection."""

    def test_checkpoint_ckpt_name_is_anchor(self):
        assert is_anchor("CheckpointLoaderSimple", "ckpt_name") is True

    def test_empty_latent_width_is_anchor(self):
        assert is_anchor("EmptyLatentImage", "width") is True

    def test_empty_latent_height_is_anchor(self):
        assert is_anchor("EmptyLatentImage", "height") is True

    def test_vae_loader_vae_name_is_anchor(self):
        assert is_anchor("VAELoader", "vae_name") is True

    def test_lora_loader_lora_name_is_anchor(self):
        assert is_anchor("LoraLoader", "lora_name") is True

    def test_unet_loader_is_anchor(self):
        assert is_anchor("UNETLoader", "unet_name") is True

    def test_dual_clip_both_anchored(self):
        assert is_anchor("DualCLIPLoader", "clip_name1") is True
        assert is_anchor("DualCLIPLoader", "clip_name2") is True

    def test_triple_clip_all_anchored(self):
        assert is_anchor("TripleCLIPLoader", "clip_name1") is True
        assert is_anchor("TripleCLIPLoader", "clip_name2") is True
        assert is_anchor("TripleCLIPLoader", "clip_name3") is True

    def test_ksampler_steps_is_not_anchor(self):
        assert is_anchor("KSampler", "steps") is False

    def test_ksampler_cfg_is_not_anchor(self):
        assert is_anchor("KSampler", "cfg") is False

    def test_ksampler_seed_is_not_anchor(self):
        assert is_anchor("KSampler", "seed") is False

    def test_clip_text_encode_text_is_not_anchor(self):
        assert is_anchor("CLIPTextEncode", "text") is False

    def test_unknown_node_type_is_not_anchor(self):
        assert is_anchor("TotallyMadeUpNode", "anything") is False

    def test_known_node_unprotected_param_is_not_anchor(self):
        # CheckpointLoaderSimple has ckpt_name anchored, but not other params
        assert is_anchor("CheckpointLoaderSimple", "some_other_param") is False

    def test_empty_strings(self):
        assert is_anchor("", "") is False

    def test_all_anchor_params_are_strings(self):
        for node_type, params in ANCHOR_PARAMS.items():
            assert isinstance(node_type, str)
            for p in params:
                assert isinstance(p, str)


class TestCheckAnchor:
    """Test check_anchor raises on protected params, passes on others."""

    def test_raises_on_anchor(self):
        with pytest.raises(AnchorViolationError) as exc_info:
            check_anchor("CheckpointLoaderSimple", "ckpt_name")
        assert "ckpt_name" in str(exc_info.value)
        assert "CheckpointLoaderSimple" in str(exc_info.value)
        assert "Constitutional violation" in str(exc_info.value)

    def test_raises_on_resolution_anchor(self):
        with pytest.raises(AnchorViolationError):
            check_anchor("EmptyLatentImage", "width")

    def test_passes_on_non_anchor(self):
        # Should not raise
        check_anchor("KSampler", "steps")
        check_anchor("KSampler", "cfg")
        check_anchor("CLIPTextEncode", "text")

    def test_passes_on_unknown_node(self):
        check_anchor("UnknownNode", "unknown_param")

    def test_error_attributes(self):
        with pytest.raises(AnchorViolationError) as exc_info:
            check_anchor("VAELoader", "vae_name")
        err = exc_info.value
        assert err.node_type == "VAELoader"
        assert err.param_name == "vae_name"

    def test_anchor_error_is_exception(self):
        """AnchorViolationError is a proper Exception subclass."""
        err = AnchorViolationError("Test", "param")
        assert isinstance(err, Exception)


class TestAnchorCompleteness:
    """Verify that key model-loading nodes are covered."""

    def test_all_loader_nodes_have_anchors(self):
        loader_nodes = [
            "CheckpointLoaderSimple",
            "UNETLoader",
            "CLIPLoader",
            "DualCLIPLoader",
            "TripleCLIPLoader",
            "VAELoader",
            "LoraLoader",
            "LoraLoaderModelOnly",
        ]
        for node in loader_nodes:
            assert node in ANCHOR_PARAMS, f"{node} missing from ANCHOR_PARAMS"

    def test_resolution_nodes_have_anchors(self):
        assert "EmptyLatentImage" in ANCHOR_PARAMS
        assert "EmptySD3LatentImage" in ANCHOR_PARAMS
