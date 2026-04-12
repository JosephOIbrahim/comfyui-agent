"""Tests for model provision checking in the autonomous pipeline.

Verifies that missing models produce warnings and existing models
don't, with graceful skip when models_dir is None.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from cognitive.pipeline.autonomous import (
    AutonomousPipeline,
    PipelineConfig,
    PipelineStage,
    _extract_model_names,
)
from cognitive.experience.accumulator import ExperienceAccumulator
from cognitive.prediction.cwm import CognitiveWorldModel
from cognitive.prediction.arbiter import SimulationArbiter
from cognitive.prediction.counterfactual import CounterfactualGenerator


# ---------------------------------------------------------------------------
# Known-good compose result (immune to template changes)
# ---------------------------------------------------------------------------

_KNOWN_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "2": {
        "class_type": "KSampler",
        "inputs": {"model": ["1", 0], "seed": 42, "steps": 20, "cfg": 7.0,
                   "sampler_name": "euler", "scheduler": "normal",
                   "denoise": 1.0},
    },
}


def _mock_compose(*args, **kwargs):
    """Return a known-good compose result regardless of templates."""
    return SimpleNamespace(
        success=True,
        workflow_data=dict(_KNOWN_WORKFLOW),
        plan=SimpleNamespace(model_family="SD1.5", parameters={"cfg": 7.0, "steps": 20}),
        error=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    cwm = CognitiveWorldModel()
    cwm.add_prior_rule("SD1.5", "cfg", (5.0, 12.0), 7.0)
    cwm.add_prior_rule("SD1.5", "steps", (10, 50), 20)
    return AutonomousPipeline(
        accumulator=ExperienceAccumulator(),
        cwm=cwm,
        arbiter=SimulationArbiter(),
        counterfactual_gen=CounterfactualGenerator(),
    )


def _ok_executor(workflow_data):
    result = MagicMock()
    result.success = True
    result.output_filenames = []
    return result


# ---------------------------------------------------------------------------
# _extract_model_names helper
# ---------------------------------------------------------------------------

class TestExtractModelNames:

    def test_extracts_ckpt_name(self):
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned.safetensors"},
            }
        }
        assert _extract_model_names(wf) == ["v1-5-pruned.safetensors"]

    def test_extracts_multiple_types(self):
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {"lora_name": "detail.safetensors"},
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "vae-ft.safetensors"},
            },
        }
        names = _extract_model_names(wf)
        assert "model.safetensors" in names
        assert "detail.safetensors" in names
        assert "vae-ft.safetensors" in names

    def test_deduplicates(self):
        wf = {
            "1": {"class_type": "A", "inputs": {"ckpt_name": "same.safetensors"}},
            "2": {"class_type": "B", "inputs": {"ckpt_name": "same.safetensors"}},
        }
        assert _extract_model_names(wf) == ["same.safetensors"]


# ---------------------------------------------------------------------------
# Pipeline provision check integration
# ---------------------------------------------------------------------------

class TestProvisionCheck:

    @pytest.fixture(autouse=True)
    def _patch_compose(self, monkeypatch):
        import cognitive.pipeline.autonomous as _mod
        monkeypatch.setattr(_mod, "compose_workflow", _mock_compose)

    def test_missing_model_warning(self, pipeline, tmp_path):
        """Workflow references a model that doesn't exist on disk →
        warning in result.warnings."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        result = pipeline.run(PipelineConfig(
            intent="test missing model",
            executor=_ok_executor,
            models_dir=models_dir,
        ))

        assert result.stage == PipelineStage.COMPLETE
        missing = [w for w in result.warnings if w.startswith("Missing model:")]
        assert len(missing) >= 1

    def test_existing_model_no_warning(self, pipeline, tmp_path):
        """Workflow references model that exists → no missing model warning."""
        models_dir = tmp_path / "models"
        ckpt_dir = models_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "sd_xl_base_1.0.safetensors").write_text("fake")

        result = pipeline.run(PipelineConfig(
            intent="test existing model",
            executor=_ok_executor,
            models_dir=models_dir,
        ))

        assert result.stage == PipelineStage.COMPLETE
        missing = [w for w in result.warnings if w.startswith("Missing model:")]
        assert len(missing) == 0

    def test_no_models_dir_skips_check(self, pipeline):
        """When models_dir is None, no provision check is performed."""
        result = pipeline.run(PipelineConfig(
            intent="test no models_dir",
            executor=_ok_executor,
            models_dir=None,
        ))

        assert result.stage == PipelineStage.COMPLETE
        missing = [w for w in result.warnings if w.startswith("Missing model:")]
        assert len(missing) == 0

    def test_multiple_missing_models(self, pipeline, tmp_path):
        """Workflow with multiple models, some missing → correct warning count."""
        models_dir = tmp_path / "models"
        ckpt_dir = models_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        # Create a workflow with 3 model references
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "exists.safetensors"},
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {"lora_name": "missing_lora.safetensors"},
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "missing_vae.safetensors"},
            },
        }

        # Only create the checkpoint that should exist
        (ckpt_dir / "exists.safetensors").write_text("fake")

        # We need to patch compose_workflow to return our custom workflow
        from cognitive.tools.compose import CompositionResult, CompositionPlan

        fake_result = CompositionResult(
            success=True,
            workflow_data=workflow,
            plan=CompositionPlan(
                intent="multi model test",
                model_family="SD1.5",
                base_template="test",
                parameters={},
                reasoning="test",
            ),
        )
        with patch(
            "cognitive.pipeline.autonomous.compose_workflow",
            return_value=fake_result,
        ):
            result = pipeline.run(PipelineConfig(
                intent="multi model test",
                executor=_ok_executor,
                models_dir=models_dir,
            ))

        missing = [w for w in result.warnings if w.startswith("Missing model:")]
        assert len(missing) == 2
        assert any("missing_lora" in w for w in missing)
        assert any("missing_vae" in w for w in missing)
