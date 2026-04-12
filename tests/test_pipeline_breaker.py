"""Tests for circuit breaker integration in the autonomous pipeline.

Verifies that the pipeline consults the circuit breaker before execution
attempts and records success/failure outcomes appropriately.
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
    """Fake executor that always succeeds."""
    result = MagicMock()
    result.success = True
    result.output_filenames = []
    return result


def _fail_executor(workflow_data):
    """Fake executor that always raises."""
    raise RuntimeError("ComfyUI unreachable")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineBreakerIntegration:

    def test_breaker_blocks_retry(self, pipeline):
        """When breaker.allow_request() returns False, pipeline returns
        FAILED without executing."""
        mock_breaker = MagicMock()
        mock_breaker.allow_request.return_value = False

        with patch(
            "cognitive.pipeline.autonomous._get_breaker",
            return_value=mock_breaker,
        ):
            result = pipeline.run(PipelineConfig(
                intent="test breaker block",
                executor=_ok_executor,
            ))

        assert result.stage == PipelineStage.FAILED
        assert "Circuit breaker open" in result.error
        mock_breaker.allow_request.assert_called()

    def test_breaker_records_success(self, pipeline):
        """On successful execution, breaker.record_success() is called."""
        mock_breaker = MagicMock()
        mock_breaker.allow_request.return_value = True

        with patch(
            "cognitive.pipeline.autonomous._get_breaker",
            return_value=mock_breaker,
        ):
            result = pipeline.run(PipelineConfig(
                intent="test breaker success",
                executor=_ok_executor,
            ))

        assert result.stage == PipelineStage.COMPLETE
        mock_breaker.record_success.assert_called()

    def test_breaker_records_failure(self, pipeline):
        """On execution exception, breaker.record_failure() is called."""
        mock_breaker = MagicMock()
        mock_breaker.allow_request.return_value = True

        with patch(
            "cognitive.pipeline.autonomous._get_breaker",
            return_value=mock_breaker,
        ):
            result = pipeline.run(PipelineConfig(
                intent="test breaker failure",
                executor=_fail_executor,
            ))

        assert result.stage == PipelineStage.FAILED
        mock_breaker.record_failure.assert_called()

    def test_breaker_unavailable_graceful(self, pipeline):
        """When _get_breaker returns None, pipeline runs normally."""
        with patch(
            "cognitive.pipeline.autonomous._get_breaker",
            return_value=None,
        ):
            result = pipeline.run(PipelineConfig(
                intent="test no breaker",
                executor=_ok_executor,
            ))

        assert result.stage == PipelineStage.COMPLETE

    def test_breaker_open_mid_retry(self, pipeline):
        """First attempt succeeds but quality is low; breaker opens before
        second attempt, resulting in FAILED."""
        call_count = 0

        def _low_quality_executor(workflow_data):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.output_filenames = []
            return result

        def _low_evaluator(exec_result):
            return QualityScore(overall=0.2, source="test")

        mock_breaker = MagicMock()
        # First call: allow. Second call: deny (breaker opened).
        mock_breaker.allow_request.side_effect = [True, False]

        with patch(
            "cognitive.pipeline.autonomous._get_breaker",
            return_value=mock_breaker,
        ):
            result = pipeline.run(PipelineConfig(
                intent="test mid-retry breaker",
                executor=_low_quality_executor,
                evaluator=_low_evaluator,
                quality_threshold=0.5,
                max_retries=3,
            ))

        assert result.stage == PipelineStage.FAILED
        assert "Circuit breaker open" in result.error
        assert call_count == 1  # Only one execution attempt
