"""Autoresearch Runner — chains the full FORESIGHT pipeline.

Orchestrates: parse program.md → initialize ratchet with CWM+experience+arbiter
→ loop (propose → execute → score → keep/discard → record) → counterfactuals
→ morning report.

All ComfyUI execution is done through injected callbacks, making this
fully testable with mocks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for an autoresearch run."""

    budget_hours: float = 1.0
    experiment_seconds: float = 30.0
    max_experiments: int = 100
    program_path: str | None = None
    session_name: str = "autoresearch"
    resume: bool = False


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    delta_id: str
    axis_scores: dict[str, float]
    kept: bool
    composite: float
    change_context: dict[str, Any]
    elapsed_seconds: float = 0.0


@dataclass
class RunResult:
    """Result of a complete autoresearch run."""

    experiments: list[ExperimentResult] = field(default_factory=list)
    counterfactual_ids: list[str] = field(default_factory=list)
    report: str = ""
    total_seconds: float = 0.0
    stopped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_count": len(self.experiments),
            "kept_count": sum(1 for e in self.experiments if e.kept),
            "counterfactuals": len(self.counterfactual_ids),
            "total_seconds": self.total_seconds,
            "stopped_reason": self.stopped_reason,
            "has_report": len(self.report) > 0,
        }


class AutoresearchRunner:
    """Runs the full autoresearch pipeline.

    The runner is decoupled from ComfyUI — all execution happens through
    injected callbacks. This makes it fully testable.

    Args:
        config: RunnerConfig with budget, timing, and paths.
        execute_fn: Callback that takes a change_context dict and returns
            axis_scores dict. Simulates running a ComfyUI experiment.
        propose_fn: Callback that returns a proposed change_context dict.
            If None, uses CWM-driven proposals or random parameter changes.
        cws: CognitiveWorkflowStage instance (optional).
    """

    def __init__(
        self,
        config: RunnerConfig,
        *,
        execute_fn: Callable[[dict], dict[str, float]] | None = None,
        propose_fn: Callable[[], dict[str, Any]] | None = None,
        cws: Any | None = None,
    ):
        self._config = config
        self._execute_fn = execute_fn
        self._propose_fn = propose_fn
        self._cws = cws
        self._program = None
        self._ratchet = None
        self._result = RunResult()
        self._start_time = 0.0

    @property
    def config(self) -> RunnerConfig:
        return self._config

    @property
    def result(self) -> RunResult:
        return self._result

    def setup(self) -> None:
        """Initialize ratchet, CWM, and parse program if provided."""
        # Parse program
        if self._config.program_path:
            try:
                from .program_parser import parse_program
                self._program = parse_program(self._config.program_path)
            except Exception:
                log.warning("Failed to parse program", exc_info=True)

        # Initialize ratchet with FORESIGHT wiring
        from .ratchet import Ratchet
        ratchet_kwargs: dict[str, Any] = {}

        if self._cws is not None:
            ratchet_kwargs["cws"] = self._cws
            try:
                from .cwm import predict
                ratchet_kwargs["cwm"] = predict
            except ImportError:
                pass
            try:
                from .arbiter import Arbiter
                ratchet_kwargs["arbiter"] = Arbiter()
            except ImportError:
                pass

        self._ratchet = Ratchet(**ratchet_kwargs)

    def run(self) -> RunResult:
        """Execute the autoresearch loop.

        Returns:
            RunResult with all experiment data and report.
        """
        self._start_time = time.time()
        self._result = RunResult()

        self.setup()

        budget_seconds = self._config.budget_hours * 3600
        experiment_count = 0

        while True:
            # Check budget
            elapsed = time.time() - self._start_time
            if elapsed >= budget_seconds:
                self._result.stopped_reason = "budget_exhausted"
                break

            if experiment_count >= self._config.max_experiments:
                self._result.stopped_reason = "max_experiments"
                break

            # Propose
            change_context = self._propose()
            if change_context is None:
                self._result.stopped_reason = "no_more_proposals"
                break

            # Execute
            exp_start = time.time()
            try:
                scores = self._execute(change_context)
            except Exception:
                log.warning("Experiment execution failed", exc_info=True)
                experiment_count += 1
                continue
            exp_elapsed = time.time() - exp_start

            # Decide
            delta_id = f"exp_{experiment_count}"
            kept = self._ratchet.decide(
                delta_id, scores,
                change_context=change_context,
            )

            self._result.experiments.append(ExperimentResult(
                delta_id=delta_id,
                axis_scores=scores,
                kept=kept,
                composite=self._ratchet.history[-1].composite,
                change_context=change_context,
                elapsed_seconds=exp_elapsed,
            ))

            experiment_count += 1

        # Close session: counterfactuals
        self._result.counterfactual_ids = self._ratchet.close_session()

        # Total time
        self._result.total_seconds = time.time() - self._start_time

        # Generate report
        self._result.report = self._generate_report()

        return self._result

    def _propose(self) -> dict[str, Any] | None:
        """Propose the next experiment."""
        if self._propose_fn is not None:
            return self._propose_fn()

        # Default: cycle through known parameter changes
        changes = [
            {"param": "steps", "direction": "increase"},
            {"param": "cfg", "direction": "increase"},
            {"param": "steps", "direction": "decrease"},
            {"param": "cfg", "direction": "decrease"},
            {"action": "add_lora"},
            {"action": "add_controlnet"},
        ]
        idx = len(self._result.experiments) % len(changes)
        return changes[idx]

    def _execute(self, change_context: dict) -> dict[str, float]:
        """Execute an experiment and return scores."""
        if self._execute_fn is not None:
            return self._execute_fn(change_context)
        # Default: return neutral scores (no real ComfyUI)
        return {"aesthetic": 0.5, "lighting": 0.5}

    def _generate_report(self) -> str:
        """Generate the morning report."""
        try:
            from .morning_report import generate_report
            from .experience import get_statistics

            history = [
                {
                    "delta_id": d.delta_id,
                    "kept": d.kept,
                    "composite": d.composite,
                    "axis_scores": d.axis_scores,
                    "prediction_accuracy": d.prediction_accuracy,
                }
                for d in self._ratchet.history
            ]

            stats = {}
            if self._cws is not None:
                try:
                    stats = get_statistics(self._cws)
                except Exception as _e:  # Cycle 62: log instead of silently swallow
                    log.debug("Statistics unavailable for autoresearch report: %s", _e)

            objective = ""
            if self._program:
                objective = self._program.objective

            return generate_report(
                ratchet_history=history,
                experience_stats=stats,
                session_name=self._config.session_name,
                counterfactual_count=len(self._result.counterfactual_ids),
                program_objective=objective,
            )
        except Exception:
            log.warning("Report generation failed", exc_info=True)
            return "# Report generation failed"
