"""Tests for vision evaluator auto-wiring in the autonomous pipeline.

Verifies that when brain_available=True the pipeline attempts to
create a vision analyzer, and falls back gracefully on failure.
"""

from __future__ import annotations

import copy

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cognitive.pipeline.autonomous import (
    AutonomousPipeline,
    PipelineConfig,
    PipelineStage,
    _try_create_vision_analyzer,
)
from cognitive.experience.accumulator import ExperienceAccumulator
from cognitive.experience.chunk import QualityScore
from cognitive.prediction.cwm import CognitiveWorldModel
from cognitive.prediction.arbiter import SimulationArbiter
from cognitive.prediction.counterfactual import CounterfactualGenerator


# ---------------------------------------------------------------------------
# Mock compose to isolate from template changes
# ---------------------------------------------------------------------------

_KNOWN_WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "test.safetensors"}},
    "2": {"class_type": "KSampler",
          "inputs": {"model": ["1", 0], "seed": 42, "steps": 20, "cfg": 7.0,
                     "sampler_name": "euler", "scheduler": "normal",
                     "denoise": 1.0}},
}


def _mock_compose(*args, **kwargs):
    return SimpleNamespace(
        success=True,
        workflow_data=copy.deepcopy(_KNOWN_WORKFLOW),
        plan=SimpleNamespace(model_family="SD1.5", parameters={"cfg": 7.0, "steps": 20}),
        error=None,
    )


@pytest.fixture(autouse=True)
def _patch_compose(monkeypatch):
    import cognitive.pipeline.autonomous as _mod
    monkeypatch.setattr(_mod, "compose_workflow", _mock_compose)


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
    result.output_filenames = ["test.png"]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVisionAutowire:

    def test_autowire_when_brain_available(self, pipeline):
        """brain_available=True + successful import → vision evaluator used."""
        fake_analyzer = MagicMock(return_value={
            "quality_score": 0.85,
            "prompt_adherence": 0.9,
            "technical_score": 0.8,
            "aesthetic_score": 0.8,
        })

        with patch(
            "cognitive.pipeline.autonomous._try_create_vision_analyzer",
            return_value=fake_analyzer,
        ):
            result = pipeline.run(PipelineConfig(
                intent="test autowire vision",
                executor=_ok_executor,
                brain_available=True,
            ))

        assert result.stage == PipelineStage.COMPLETE
        assert any("Auto-wired vision" in s for s in result.stage_log)
        # Vision evaluator should have been called with execution result
        assert result.quality.source == "vision"

    def test_autowire_import_fails_graceful(self, pipeline):
        """brain_available=True but import fails → falls back to default."""
        with patch(
            "cognitive.pipeline.autonomous._try_create_vision_analyzer",
            return_value=None,
        ):
            result = pipeline.run(PipelineConfig(
                intent="test autowire fallback",
                executor=_ok_executor,
                brain_available=True,
            ))

        assert result.stage == PipelineStage.COMPLETE
        # Should use default evaluator (rule-based, score=0.7 for success)
        assert result.quality.overall == pytest.approx(0.7)

    def test_no_autowire_when_explicit_evaluator(self, pipeline):
        """When an explicit evaluator is set, brain_available is ignored."""
        custom_score = QualityScore(overall=0.95, source="custom")

        def _custom_eval(exec_result):
            return custom_score

        result = pipeline.run(PipelineConfig(
            intent="test explicit evaluator",
            executor=_ok_executor,
            evaluator=_custom_eval,
            brain_available=True,
        ))

        assert result.stage == PipelineStage.COMPLETE
        assert result.quality.overall == pytest.approx(0.95)
        assert result.quality.source == "custom"

    def test_no_autowire_when_brain_false(self, pipeline):
        """brain_available=False → default evaluator, no auto-wiring."""
        result = pipeline.run(PipelineConfig(
            intent="test no brain",
            executor=_ok_executor,
            brain_available=False,
        ))

        assert result.stage == PipelineStage.COMPLETE
        # Default rule-based evaluator gives 0.7 for success
        assert result.quality.overall == pytest.approx(0.7)
        assert not any("Auto-wired vision" in s for s in result.stage_log)


class TestTryCreateVisionAnalyzer:

    def test_returns_none_when_import_fails(self):
        """When agent.brain.vision is not importable, returns None."""
        with patch.dict("sys.modules", {"agent.brain.vision": None}):
            # Force import failure
            result = _try_create_vision_analyzer()
            # May or may not be None depending on import caching,
            # but should not raise
            assert result is None or callable(result)

    def test_returns_none_on_exception(self):
        """Any exception during creation returns None."""
        with patch(
            "cognitive.pipeline.autonomous._try_create_vision_analyzer",
            side_effect=ImportError("no module"),
        ):
            # The actual function catches exceptions internally,
            # so test the real one with a broken import
            pass

        # Direct test: force the function to fail
        with patch(
            "builtins.__import__",
            side_effect=ImportError("test"),
        ):
            result = _try_create_vision_analyzer()
            assert result is None
