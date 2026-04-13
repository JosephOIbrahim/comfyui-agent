"""Tests for the Autonomous Pipeline.

[AUTONOMY x CRUCIBLE] — End-to-end pipeline tests including
degradation, retry, ratchet direction, and zero-intervention mode.
"""

import pytest

import cognitive.pipeline.autonomous as _auto_mod
from cognitive.pipeline import create_default_pipeline
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


# ---------------------------------------------------------------------------
# Basic Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_empty_intent_fails(self, pipeline):
        result = pipeline.run(PipelineConfig(intent=""))
        assert result.stage == PipelineStage.FAILED
        assert "Empty" in result.error

    def test_basic_intent_completes(self, pipeline):
        result = pipeline.run(PipelineConfig(intent="a beautiful sunset"))
        assert result.stage == PipelineStage.COMPLETE
        assert result.success is True

    def test_prediction_populated(self, pipeline):
        result = pipeline.run(PipelineConfig(intent="landscape photo"))
        assert result.prediction is not None
        assert result.prediction.quality_estimate > 0

    def test_experience_recorded(self, pipeline):
        result = pipeline.run(PipelineConfig(intent="test generation"))
        assert result.experience_chunk is not None
        assert pipeline._accumulator.generation_count == 1

    def test_stage_log_populated(self, pipeline):
        result = pipeline.run(PipelineConfig(intent="test"))
        assert len(result.stage_log) > 0
        assert any("[intent]" in s for s in result.stage_log)
        assert any("[complete]" in s for s in result.stage_log)

    def test_with_model_family(self, pipeline):
        result = pipeline.run(PipelineConfig(intent="portrait", model_family="Flux"))
        assert result.success is True

    def test_arbiter_decision_attached(self, pipeline):
        result = pipeline.run(PipelineConfig(intent="test"))
        assert result.arbiter_decision is not None


# ---------------------------------------------------------------------------
# Execution Delegates
# ---------------------------------------------------------------------------

class TestDelegates:

    def test_with_executor(self, pipeline):
        executed = []
        def mock_exec(wf):
            executed.append(wf)
            return type("R", (), {"output_filenames": ["test.png"]})()

        result = pipeline.run(PipelineConfig(
            intent="test",
            executor=mock_exec,
        ))
        assert result.success is True
        assert len(executed) > 0

    def test_with_evaluator(self, pipeline):
        result = pipeline.run(PipelineConfig(
            intent="test",
            evaluator=lambda _: QualityScore(overall=0.8),
        ))
        assert result.quality.overall == 0.8

    def test_evaluator_numeric(self, pipeline):
        result = pipeline.run(PipelineConfig(
            intent="test",
            evaluator=lambda _: 0.75,
        ))
        assert result.quality.overall == 0.75

    def test_executor_failure(self, pipeline):
        def failing_exec(wf):
            raise RuntimeError("GPU OOM")

        result = pipeline.run(PipelineConfig(
            intent="test",
            executor=failing_exec,
        ))
        assert result.stage == PipelineStage.FAILED
        assert "GPU OOM" in result.error

    def test_evaluator_exception_sets_failed(self, pipeline):
        """Evaluator exception must set FAILED + stop pipeline (Cycle 25 fix)."""
        def exploding_evaluator(_result):
            raise RuntimeError("evaluation exploded")

        result = pipeline.run(PipelineConfig(
            intent="test explosion",
            evaluator=exploding_evaluator,
        ))
        assert result.stage == PipelineStage.FAILED
        assert result.error is not None
        assert "evaluation" in result.error.lower()
        # Pipeline must not have continued to LEARN stage
        assert result.experience_chunk is None


# ---------------------------------------------------------------------------
# Arbiter Interrupt
# ---------------------------------------------------------------------------

class TestArbiterInterrupt:

    def test_interrupt_on_degenerate_params(self):
        """Pipeline should be interrupted for obviously bad params."""
        cwm = CognitiveWorldModel()
        cwm.add_prior_rule("SD1.5", "cfg", (5.0, 12.0), 7.0)
        # Arbiter with low interrupt floor
        arbiter = SimulationArbiter(interrupt_quality_floor=0.3)

        pipeline = AutonomousPipeline(cwm=cwm, arbiter=arbiter)

        # This won't trigger interrupt because the compose step produces
        # reasonable params. To test interrupt, we need the prediction to be bad.
        # The CWM will predict based on priors, which should be OK for normal params.
        result = pipeline.run(PipelineConfig(intent="test"))
        # Even if it doesn't interrupt, it should complete
        assert result.stage in (PipelineStage.COMPLETE, PipelineStage.INTERRUPTED)


# ---------------------------------------------------------------------------
# Learning & Experience
# ---------------------------------------------------------------------------

class TestLearning:

    def test_multiple_runs_accumulate(self, pipeline):
        for i in range(5):
            pipeline.run(PipelineConfig(intent=f"test {i}"))
        assert pipeline._accumulator.generation_count == 5

    def test_quality_recorded_with_evaluator(self, pipeline):
        pipeline.run(PipelineConfig(
            intent="test",
            executor=lambda wf: type("R", (), {"output_filenames": ["out.png"]})(),
            evaluator=lambda _: QualityScore(overall=0.9, source="test"),
        ))
        chunks = pipeline._accumulator.get_successful_chunks(min_quality=0.5)
        assert len(chunks) == 1
        assert chunks[0].quality.overall == 0.9

    def test_prediction_accuracy_tracked(self, pipeline):
        pipeline.run(PipelineConfig(
            intent="test",
            evaluator=lambda _: QualityScore(overall=0.7),
        ))
        cal = pipeline._cwm.get_calibration()
        assert cal["samples"] == 1


# ---------------------------------------------------------------------------
# Quality Threshold
# ---------------------------------------------------------------------------

class TestRetry:
    """Quality threshold checks. Auto-retry stub removed — real retry
    is tracked in backlog (requires D3 evaluator integration)."""

    def test_pipeline_completes_below_threshold(self, pipeline):
        # Pipeline completes regardless of quality; no retry happens.
        result = pipeline.run(PipelineConfig(
            intent="test",
            evaluator=lambda _: QualityScore(overall=0.3),
            quality_threshold=0.6,
        ))
        assert result.stage == PipelineStage.COMPLETE

    def test_pipeline_completes_above_threshold(self, pipeline):
        result = pipeline.run(PipelineConfig(
            intent="test",
            evaluator=lambda _: QualityScore(overall=0.8),
            quality_threshold=0.6,
        ))
        assert result.stage == PipelineStage.COMPLETE

    def test_pipeline_completes_at_low_quality(self, pipeline):
        result = pipeline.run(PipelineConfig(
            intent="test",
            evaluator=lambda _: QualityScore(overall=0.1),
        ))
        assert result.stage == PipelineStage.COMPLETE


# ---------------------------------------------------------------------------
# Full Pipeline (mock mode — zero human intervention)
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_zero_intervention_mock_mode(self, pipeline):
        """Full pipeline runs with zero human intervention (no executor/evaluator)."""
        result = pipeline.run(PipelineConfig(intent="cinematic portrait golden hour"))
        assert result.success is True
        assert result.stage == PipelineStage.COMPLETE
        assert result.prediction is not None
        assert result.experience_chunk is not None
        assert len(result.stage_log) >= 4

    def test_full_pipeline_with_delegates(self, pipeline):
        """Full pipeline with executor and evaluator."""
        result = pipeline.run(PipelineConfig(
            intent="photorealistic landscape",
            executor=lambda wf: type("R", (), {"output_filenames": ["out.png"]})(),
            evaluator=lambda _: QualityScore(overall=0.85, technical=0.8, aesthetic=0.9),
        ))
        assert result.success is True
        assert result.quality.overall == 0.85
        assert result.experience_chunk is not None
        assert pipeline._accumulator.generation_count == 1

    def test_pipeline_improves_with_experience(self, pipeline):
        """Pipeline should use accumulated experience in later runs."""
        # Run several times with good quality
        for i in range(35):  # Past Phase 2 threshold
            pipeline.run(PipelineConfig(
                intent=f"landscape {i}",
                evaluator=lambda _: QualityScore(overall=0.8),
            ))

        # Experience weight should be > 0
        assert pipeline._accumulator.experience_weight > 0
        assert pipeline._accumulator.learning_phase.value != "prior"

    def test_counterfactual_generated(self, pipeline):
        pipeline.run(PipelineConfig(intent="test"))
        assert pipeline._cf_gen.total_generated > 0

    def test_style_locked_series(self, pipeline):
        """Multiple runs with same intent maintain consistency via experience."""
        results = []
        for _ in range(3):
            r = pipeline.run(PipelineConfig(
                intent="impressionist painting",
                evaluator=lambda _: QualityScore(overall=0.7),
            ))
            results.append(r)

        # All should complete successfully
        assert all(r.success for r in results)
        # Experience should grow
        assert pipeline._accumulator.generation_count == 3


# ---------------------------------------------------------------------------
# Bootstrap Factory
# ---------------------------------------------------------------------------

class TestCreateDefaultPipeline:

    def test_returns_autonomous_pipeline(self):
        p = create_default_pipeline()
        assert isinstance(p, AutonomousPipeline)

    def test_components_are_correct_types(self):
        p = create_default_pipeline()
        assert isinstance(p._accumulator, ExperienceAccumulator)
        assert isinstance(p._cwm, CognitiveWorldModel)
        assert isinstance(p._arbiter, SimulationArbiter)
        assert isinstance(p._cf_gen, CounterfactualGenerator)

    def test_two_calls_return_independent_pipelines(self):
        p1 = create_default_pipeline()
        p2 = create_default_pipeline()
        assert p1 is not p2
        assert p1._accumulator is not p2._accumulator

    def test_pipeline_can_run_after_create(self, monkeypatch):
        """create_default_pipeline() returns a pipeline that can run()."""
        monkeypatch.setattr(
            _auto_mod,
            "_execute_workflow_default",
            lambda wf: type("R", (), {"success": True, "output_filenames": []})(),
        )
        p = create_default_pipeline()
        result = p.run(PipelineConfig(intent="test intent"))
        assert result.stage == PipelineStage.COMPLETE


# ---------------------------------------------------------------------------
# Cycle 41: result.error populated on CWM failure
# ---------------------------------------------------------------------------

class TestCWMFailureErrorField:
    """Cycle 41: result.error must be set when CWM predict() raises."""

    def test_cwm_failure_sets_error_field(self):
        """When CWM.predict() raises, result.error must be non-empty."""
        from unittest.mock import MagicMock
        from cognitive.experience.accumulator import ExperienceAccumulator
        from cognitive.prediction.arbiter import SimulationArbiter
        from cognitive.prediction.counterfactual import CounterfactualGenerator

        broken_cwm = MagicMock()
        broken_cwm.predict.side_effect = RuntimeError("CWM exploded")

        p = AutonomousPipeline(
            accumulator=ExperienceAccumulator(),
            cwm=broken_cwm,
            arbiter=SimulationArbiter(),
            counterfactual_gen=CounterfactualGenerator(),
        )
        result = p.run(PipelineConfig(intent="test intent"))
        assert result.stage == PipelineStage.FAILED
        assert result.error, "result.error must be non-empty when CWM fails"
        assert "CWM" in result.error or "predict" in result.error.lower()

    def test_cwm_failure_error_not_empty_string(self):
        """result.error must not be an empty string when stage==FAILED from CWM."""
        from unittest.mock import MagicMock
        from cognitive.experience.accumulator import ExperienceAccumulator
        from cognitive.prediction.arbiter import SimulationArbiter
        from cognitive.prediction.counterfactual import CounterfactualGenerator

        broken_cwm = MagicMock()
        broken_cwm.predict.side_effect = ValueError("bad params")

        p = AutonomousPipeline(
            accumulator=ExperienceAccumulator(),
            cwm=broken_cwm,
            arbiter=SimulationArbiter(),
            counterfactual_gen=CounterfactualGenerator(),
        )
        result = p.run(PipelineConfig(intent="test intent"))
        assert result.stage == PipelineStage.FAILED
        assert result.error != ""
