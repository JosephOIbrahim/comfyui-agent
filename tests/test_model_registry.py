"""Tests for the model registry — USD-backed model tracking."""

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core not installed")

from agent.stage.cognitive_stage import CognitiveWorkflowStage, StageError
from agent.stage.model_registry import (
    MODEL_TYPES,
    VALID_STATUSES,
    find_model,
    get_model,
    list_models_by_status,
    list_models_by_type,
    reconcile,
    register_model,
    update_status,
)


class TestRegisterModel:
    """Model registration in the USD stage."""

    def test_register_checkpoint(self):
        cws = CognitiveWorkflowStage()
        path = register_model(
            cws, "checkpoints", "v1-5-pruned-emaonly.safetensors",
            status="materialized",
            base_model="SD 1.5",
        )
        assert path.startswith("/models/checkpoints/")
        assert cws.read(path, "filename") == "v1-5-pruned-emaonly.safetensors"
        assert cws.read(path, "status") == "materialized"
        assert cws.read(path, "base_model") == "SD 1.5"

    def test_register_lora(self):
        cws = CognitiveWorkflowStage()
        path = register_model(
            cws, "loras", "detail_enhancer_v2.safetensors",
            status="available",
            source_url="https://civitai.com/models/12345",
            sha256="abc123",
        )
        assert cws.read(path, "source_url") == "https://civitai.com/models/12345"
        assert cws.read(path, "sha256") == "abc123"

    def test_register_with_all_fields(self):
        cws = CognitiveWorkflowStage()
        path = register_model(
            cws, "checkpoints", "model.safetensors",
            status="materialized",
            source_url="https://example.com/model",
            sha256="deadbeef",
            file_path="/path/to/model.safetensors",
            size_bytes=4_000_000_000,
            base_model="SDXL",
            description="A great model",
        )
        attrs = get_model(cws, path)
        assert attrs["filename"] == "model.safetensors"
        assert attrs["status"] == "materialized"
        assert attrs["source_url"] == "https://example.com/model"
        assert attrs["sha256"] == "deadbeef"
        assert attrs["file_path"] == "/path/to/model.safetensors"
        assert attrs["size_bytes"] == 4_000_000_000
        assert attrs["base_model"] == "SDXL"
        assert attrs["description"] == "A great model"

    def test_invalid_model_type_raises(self):
        cws = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="Unknown model type"):
            register_model(cws, "invalid_type", "model.safetensors")

    def test_invalid_status_raises(self):
        cws = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="Invalid status"):
            register_model(
                cws, "checkpoints", "model.safetensors", status="bogus"
            )

    def test_safe_name_handles_special_chars(self):
        cws = CognitiveWorkflowStage()
        path = register_model(
            cws, "checkpoints", "my-model (v2.1).safetensors"
        )
        # Prim name should be sanitized
        assert cws.prim_exists(path)
        assert cws.read(path, "filename") == "my-model (v2.1).safetensors"

    def test_safe_name_handles_numeric_start(self):
        cws = CognitiveWorkflowStage()
        path = register_model(cws, "loras", "3d_style_lora.safetensors")
        assert cws.prim_exists(path)
        assert cws.read(path, "filename") == "3d_style_lora.safetensors"

    def test_register_default_status_is_available(self):
        cws = CognitiveWorkflowStage()
        path = register_model(cws, "checkpoints", "model.safetensors")
        assert cws.read(path, "status") == "available"


class TestUpdateStatus:
    """Status transitions."""

    def test_update_to_downloading(self):
        cws = CognitiveWorkflowStage()
        path = register_model(cws, "checkpoints", "model.safetensors")
        update_status(cws, path, "downloading")
        assert cws.read(path, "status") == "downloading"

    def test_update_to_materialized_with_path(self):
        cws = CognitiveWorkflowStage()
        path = register_model(cws, "checkpoints", "model.safetensors")
        update_status(
            cws, path, "materialized",
            file_path="/models/checkpoints/model.safetensors",
            sha256="verified_hash",
        )
        assert cws.read(path, "status") == "materialized"
        assert cws.read(path, "file_path") == "/models/checkpoints/model.safetensors"
        assert cws.read(path, "sha256") == "verified_hash"

    def test_update_to_failed(self):
        cws = CognitiveWorkflowStage()
        path = register_model(cws, "checkpoints", "model.safetensors")
        update_status(cws, path, "failed")
        assert cws.read(path, "status") == "failed"

    def test_update_nonexistent_raises(self):
        cws = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="not registered"):
            update_status(cws, "/models/checkpoints/nonexistent", "materialized")

    def test_update_invalid_status_raises(self):
        cws = CognitiveWorkflowStage()
        path = register_model(cws, "checkpoints", "model.safetensors")
        with pytest.raises(StageError, match="Invalid status"):
            update_status(cws, path, "invalid")


class TestGetModel:
    """Model retrieval."""

    def test_get_registered_model(self):
        cws = CognitiveWorkflowStage()
        path = register_model(
            cws, "loras", "detail.safetensors",
            status="materialized",
            base_model="SDXL",
        )
        model = get_model(cws, path)
        assert model is not None
        assert model["filename"] == "detail.safetensors"
        assert model["status"] == "materialized"

    def test_get_nonexistent_returns_none(self):
        cws = CognitiveWorkflowStage()
        assert get_model(cws, "/models/checkpoints/nope") is None


class TestListModels:
    """Model listing and filtering."""

    def test_list_by_type(self):
        cws = CognitiveWorkflowStage()
        register_model(cws, "checkpoints", "model_a.safetensors", status="materialized")
        register_model(cws, "checkpoints", "model_b.safetensors", status="available")
        register_model(cws, "loras", "lora_a.safetensors", status="materialized")

        checkpoints = list_models_by_type(cws, "checkpoints")
        assert len(checkpoints) == 2

        loras = list_models_by_type(cws, "loras")
        assert len(loras) == 1

    def test_list_by_type_empty(self):
        cws = CognitiveWorkflowStage()
        assert list_models_by_type(cws, "vae") == []

    def test_list_by_status(self):
        cws = CognitiveWorkflowStage()
        register_model(cws, "checkpoints", "a.safetensors", status="materialized")
        register_model(cws, "checkpoints", "b.safetensors", status="available")
        register_model(cws, "loras", "c.safetensors", status="materialized")

        materialized = list_models_by_status(cws, "materialized")
        assert len(materialized) == 2

        available = list_models_by_status(cws, "available")
        assert len(available) == 1

    def test_list_includes_prim_path(self):
        cws = CognitiveWorkflowStage()
        register_model(cws, "checkpoints", "model.safetensors")
        models = list_models_by_type(cws, "checkpoints")
        assert "_prim_path" in models[0]


class TestFindModel:
    """Find model by filename across all types."""

    def test_find_existing(self):
        cws = CognitiveWorkflowStage()
        register_model(cws, "checkpoints", "target.safetensors", base_model="SD 1.5")
        register_model(cws, "loras", "other.safetensors")

        found = find_model(cws, "target.safetensors")
        assert found is not None
        assert found["base_model"] == "SD 1.5"
        assert found["model_type"] == "checkpoints"

    def test_find_nonexistent(self):
        cws = CognitiveWorkflowStage()
        register_model(cws, "checkpoints", "model.safetensors")
        assert find_model(cws, "not_here.safetensors") is None

    def test_find_in_different_type(self):
        cws = CognitiveWorkflowStage()
        register_model(cws, "vae", "vae_model.safetensors")
        found = find_model(cws, "vae_model.safetensors")
        assert found is not None
        assert found["model_type"] == "vae"


class TestReconcile:
    """Cross-reference registry with filesystem."""

    def test_materialized_but_missing_on_disk(self, tmp_path):
        cws = CognitiveWorkflowStage()
        register_model(
            cws, "checkpoints", "missing.safetensors",
            status="materialized",
        )
        # Create models dir structure but don't create the file
        (tmp_path / "checkpoints").mkdir()

        result = reconcile(cws, tmp_path)
        assert len(result["registered_missing"]) == 1

    def test_on_disk_but_not_registered(self, tmp_path):
        cws = CognitiveWorkflowStage()
        # Create a model file on disk
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "unregistered.safetensors").write_bytes(b"fake")

        result = reconcile(cws, tmp_path)
        assert len(result["unregistered_on_disk"]) == 1

    def test_available_but_actually_on_disk(self, tmp_path):
        cws = CognitiveWorkflowStage()
        register_model(
            cws, "checkpoints", "present.safetensors",
            status="available",
        )
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "present.safetensors").write_bytes(b"fake")

        result = reconcile(cws, tmp_path)
        assert len(result["status_mismatch"]) == 1

    def test_clean_reconcile(self, tmp_path):
        """Everything consistent — no discrepancies."""
        cws = CognitiveWorkflowStage()
        register_model(
            cws, "checkpoints", "good.safetensors",
            status="materialized",
        )
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "good.safetensors").write_bytes(b"model")

        result = reconcile(cws, tmp_path)
        assert result["registered_missing"] == []
        assert result["unregistered_on_disk"] == []
        assert result["status_mismatch"] == []

    def test_reconcile_empty(self, tmp_path):
        cws = CognitiveWorkflowStage()
        result = reconcile(cws, tmp_path)
        assert result["registered_missing"] == []
        assert result["unregistered_on_disk"] == []
        assert result["status_mismatch"] == []


class TestModelTypes:
    """Verify model type constants."""

    def test_common_types_present(self):
        assert "checkpoints" in MODEL_TYPES
        assert "loras" in MODEL_TYPES
        assert "controlnet" in MODEL_TYPES
        assert "vae" in MODEL_TYPES
        assert "embeddings" in MODEL_TYPES
        assert "upscale_models" in MODEL_TYPES
        assert "diffusion_models" in MODEL_TYPES

    def test_valid_statuses(self):
        assert "available" in VALID_STATUSES
        assert "downloading" in VALID_STATUSES
        assert "materialized" in VALID_STATUSES
        assert "failed" in VALID_STATUSES
