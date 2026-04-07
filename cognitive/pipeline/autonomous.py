"""Autonomous generation pipeline.

End-to-end: creative intent → composed workflow → predicted outcome
→ execution → evaluation → learning. Zero human intervention.

The pipeline orchestrates all cognitive components:
- Composer: intent → workflow
- CWM: predict quality before executing
- Arbiter: decide whether to surface predictions
- Engine: non-destructive mutations
- Experience: capture and learn from outcomes
- Ratchet: iterative optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..core.graph import CognitiveGraphEngine
from ..experience.chunk import ExperienceChunk, QualityScore
from ..experience.accumulator import ExperienceAccumulator
from ..experience.signature import GenerationContextSignature
from ..prediction.cwm import CognitiveWorldModel, Prediction
from ..prediction.arbiter import SimulationArbiter, DeliveryMode
from ..prediction.counterfactual import CounterfactualGenerator
from ..tools.compose import compose_workflow
from ..tools.analyze import analyze_workflow


class PipelineStage(Enum):
    """Stages of the autonomous pipeline."""

    INTENT = "intent"
    COMPOSE = "compose"
    PREDICT = "predict"
    GATE = "gate"  # Arbiter decides whether to proceed
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    LEARN = "learn"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class PipelineConfig:
    """Configuration for an autonomous pipeline run."""

    intent: str = ""
    model_family: str | None = None
    max_retries: int = 3
    quality_threshold: float = 0.6
    auto_retry: bool = True
    executor: Callable | None = None  # Actual execution delegate
    evaluator: Callable | None = None  # Quality evaluation delegate


@dataclass
class PipelineResult:
    """Result of an autonomous pipeline run."""

    stage: PipelineStage = PipelineStage.INTENT
    intent: str = ""
    workflow_data: dict[str, Any] = field(default_factory=dict)
    prediction: Prediction | None = None
    arbiter_decision: Any | None = None
    execution_result: Any | None = None
    quality: QualityScore = field(default_factory=QualityScore)
    experience_chunk: ExperienceChunk | None = None
    retries: int = 0
    error: str = ""
    stage_log: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.stage == PipelineStage.COMPLETE and not self.error

    def log(self, message: str) -> None:
        self.stage_log.append(f"[{self.stage.value}] {message}")


class AutonomousPipeline:
    """Orchestrates the full autonomous generation pipeline.

    Components are injected at construction time. Missing components
    degrade gracefully — the pipeline adapts to what's available.
    """

    def __init__(
        self,
        accumulator: ExperienceAccumulator | None = None,
        cwm: CognitiveWorldModel | None = None,
        arbiter: SimulationArbiter | None = None,
        counterfactual_gen: CounterfactualGenerator | None = None,
    ):
        self._accumulator = accumulator or ExperienceAccumulator()
        self._cwm = cwm or CognitiveWorldModel()
        self._arbiter = arbiter or SimulationArbiter()
        self._cf_gen = counterfactual_gen or CounterfactualGenerator()

    def run(self, config: PipelineConfig) -> PipelineResult:
        """Execute the full autonomous pipeline.

        Stages:
        1. INTENT: Parse the creative intent
        2. COMPOSE: Build a workflow from intent + experience
        3. PREDICT: Predict quality before executing
        4. GATE: Arbiter decides whether to proceed
        5. EXECUTE: Run the workflow (via delegate)
        6. EVALUATE: Assess output quality
        7. LEARN: Store experience + counterfactual

        Returns:
            PipelineResult with full execution details.
        """
        result = PipelineResult(intent=config.intent)

        # Stage 1: INTENT
        result.stage = PipelineStage.INTENT
        if not config.intent.strip():
            result.error = "Empty intent"
            result.stage = PipelineStage.FAILED
            return result
        result.log(f"Intent: {config.intent}")

        # Stage 2: COMPOSE
        result.stage = PipelineStage.COMPOSE
        experience_patterns = self._get_experience_patterns(config)
        composition = compose_workflow(
            config.intent,
            model_family=config.model_family,
            experience_patterns=experience_patterns,
        )
        if not composition.success:
            result.error = f"Composition failed: {composition.error}"
            result.stage = PipelineStage.FAILED
            return result

        result.workflow_data = composition.workflow_data
        model_family = composition.plan.model_family if composition.plan else ""
        params = composition.plan.parameters if composition.plan else {}
        result.log(f"Composed workflow: family={model_family}, params={len(params)}")

        # Stage 3: PREDICT
        result.stage = PipelineStage.PREDICT
        exp_weight = self._accumulator.experience_weight
        cf_adjustment = self._cf_gen.get_adjustment()

        prediction = self._cwm.predict(
            model_family=model_family,
            parameters=params,
            experience_quality=self._get_avg_experience_quality(config),
            experience_weight=exp_weight,
            counterfactual_adjustment=cf_adjustment,
        )
        result.prediction = prediction
        result.log(
            f"Predicted quality: {prediction.quality_estimate:.1%} "
            f"(confidence: {prediction.confidence:.1%})"
        )

        # Stage 4: GATE (Arbiter)
        result.stage = PipelineStage.GATE
        decision = self._arbiter.decide(
            prediction.quality_estimate,
            prediction.confidence,
            prediction.risk_factors,
        )
        result.arbiter_decision = decision

        if decision.should_interrupt:
            result.log(f"Arbiter interrupted: {decision.message}")
            result.error = f"Arbiter: {decision.message}"
            result.stage = PipelineStage.INTERRUPTED
            return result

        if decision.mode != DeliveryMode.SILENT:
            result.log(f"Arbiter ({decision.mode.value}): {decision.message}")

        # Stage 5: EXECUTE
        result.stage = PipelineStage.EXECUTE
        if config.executor is not None:
            try:
                exec_result = config.executor(result.workflow_data)
                result.execution_result = exec_result
                result.log("Execution complete")
            except Exception as e:
                result.error = f"Execution failed: {e}"
                result.stage = PipelineStage.FAILED
                return result
        else:
            result.log("No executor provided — execution skipped (mock mode)")

        # Stage 6: EVALUATE
        result.stage = PipelineStage.EVALUATE
        if config.evaluator is not None:
            try:
                quality = config.evaluator(result.execution_result)
                if isinstance(quality, QualityScore):
                    result.quality = quality
                elif isinstance(quality, (int, float)):
                    result.quality = QualityScore(overall=float(quality))
                result.log(f"Quality: {result.quality.overall:.1%}")
            except Exception as e:
                result.log(f"Evaluation failed: {e}")
        else:
            result.log("No evaluator — skipping quality assessment")

        # Stage 7: LEARN
        result.stage = PipelineStage.LEARN
        chunk = ExperienceChunk(
            model_family=model_family,
            prompt=config.intent,
            parameters={"composed": params},
            quality=result.quality,
            output_filenames=[],
        )
        if result.execution_result is not None:
            chunk.output_filenames = getattr(
                result.execution_result, "output_filenames", [],
            )

        self._accumulator.record(chunk)
        result.experience_chunk = chunk
        result.log(f"Recorded experience (total: {self._accumulator.generation_count})")

        # Record prediction accuracy
        if result.quality.is_scored and result.prediction is not None:
            self._cwm.record_accuracy(
                result.prediction.quality_estimate,
                result.quality.overall,
            )

        # Generate counterfactual
        cf = self._cf_gen.generate(params, prediction.quality_estimate)
        if cf is not None:
            result.log(f"Counterfactual: vary {cf.changed_parameter}")

        # Auto-retry if quality below threshold
        if (
            config.auto_retry
            and result.quality.is_scored
            and result.quality.overall < config.quality_threshold
            and result.retries < config.max_retries
        ):
            result.retries += 1
            result.log(f"Quality below threshold — retry {result.retries}/{config.max_retries}")
            # In a real implementation, we'd adjust params and re-run.
            # For now, just log the intent to retry.

        result.stage = PipelineStage.COMPLETE
        result.log("Pipeline complete")
        return result

    def _get_experience_patterns(self, config: PipelineConfig) -> list[dict]:
        """Get relevant experience patterns for composition."""
        if self._accumulator.generation_count == 0:
            return []

        # Build a rough signature from intent keywords
        sig = GenerationContextSignature()
        if config.model_family:
            sig.model_family = config.model_family

        retrieval = self._accumulator.retrieve(sig, top_k=5, min_similarity=0.0)
        return [
            {
                "confidence": chunk.quality.overall * chunk.decay_weight,
                "parameters": chunk.parameters,
            }
            for chunk in retrieval.matches
        ]

    def _get_avg_experience_quality(self, config: PipelineConfig) -> float | None:
        """Get average quality from relevant experience."""
        successful = self._accumulator.get_successful_chunks(min_quality=0.3)
        if not successful:
            return None
        return sum(c.quality.overall for c in successful) / len(successful)
