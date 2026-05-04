"""CozyLoop — long-running self-healing harness.

Wraps `AutoresearchRunner` (agent/stage/autoresearch_runner.py) with three
additions per the Cozy Constitution:

  Article III  (bounded-failure ladder)
    Errors are classified TRANSIENT / RECOVERABLE / TERMINAL via
    `constitution.self_healing_ladder`. The harness retries TRANSIENT with
    exponential backoff, escalates RECOVERABLE through the ratchet, and
    halts cleanly on TERMINAL after writing a BLOCKER.md.

  Article IV   (checkpoint integrity)
    On every iteration boundary, and after every TERMINAL classification,
    the harness flushes the stage to STAGE_DEFAULT_PATH (atomic via the
    existing `flush()` method) and saves the ratchet history alongside.
    Crash-resume is supported via the existing `RunnerConfig.resume` flag.

  Article VI   (ratchet sovereignty)
    MetaAgent improvement proposals from the RECOVERABLE branch MUST pass
    `Ratchet.decide()` before being applied. Tier-1 auto-fixes are limited
    to non-state-mutating dials (retry timeout, backoff factor).

Sync threading only — no asyncio. Health loop runs on a daemon Thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..stage.autoresearch_runner import (
    AutoresearchRunner,
    RunnerConfig,
    RunResult,
)
from ..stage.constitution import (
    RECOVERABLE,
    TERMINAL,
    TRANSIENT,
    self_healing_ladder,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CozyLoopConfig:
    """Configuration for the long-running harness."""

    # Budget — same semantics as RunnerConfig
    budget_hours: float = 24.0
    max_experiments: int = 1000
    experiment_seconds: float = 30.0

    # Checkpointing
    checkpoint_path: str | None = None  # default: STAGE_DEFAULT_PATH
    checkpoint_every_n: int = 1  # checkpoint every N iterations
    checkpoint_every_seconds: float = 300.0  # also checkpoint on time

    # Self-healing
    max_transient_retries: int = 3
    transient_backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)
    max_recoverable_per_signature: int = 3  # promote to TERMINAL after N

    # Health loop
    health_check_seconds: float = 30.0
    halt_on_terminal: bool = True

    # Session metadata
    session_name: str = "cozy_autonomous"
    program_path: str | None = None
    resume: bool = False

    def to_runner_config(self) -> RunnerConfig:
        """Project to the underlying AutoresearchRunner config."""
        return RunnerConfig(
            budget_hours=self.budget_hours,
            experiment_seconds=self.experiment_seconds,
            max_experiments=self.max_experiments,
            program_path=self.program_path,
            session_name=self.session_name,
            resume=self.resume,
        )


# ---------------------------------------------------------------------------
# Reporting types
# ---------------------------------------------------------------------------

@dataclass
class HealthSnapshot:
    """Periodic health report from the harness."""

    timestamp: float
    iterations_completed: int
    transient_retries_total: int
    recoverable_repairs_total: int
    terminal_count: int
    elapsed_seconds: float
    halt_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "halt_reason": self.halt_reason,
            "iterations_completed": self.iterations_completed,
            "recoverable_repairs_total": self.recoverable_repairs_total,
            "terminal_count": self.terminal_count,
            "timestamp": self.timestamp,
            "transient_retries_total": self.transient_retries_total,
        }


@dataclass
class CozyLoopResult:
    """Final result of a harness run."""

    run_result: RunResult | None = None
    health_snapshots: list[HealthSnapshot] = field(default_factory=list)
    halt_reason: str = ""
    total_seconds: float = 0.0
    blocker_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "blocker_path": self.blocker_path,
            "halt_reason": self.halt_reason,
            "health_snapshot_count": len(self.health_snapshots),
            "run_result": self.run_result.to_dict() if self.run_result else None,
            "total_seconds": self.total_seconds,
        }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class CozyLoop:
    """Long-running self-healing harness governed by the Cozy Constitution.

    Composes (does not subclass) AutoresearchRunner — keeps the inner loop
    simple while wrapping its `_execute` callback with the self-healing ladder.

    Args:
        config: CozyLoopConfig with budget, checkpoint, and healing knobs.
        execute_fn: ComfyUI execute callback (passed through to AutoresearchRunner).
        propose_fn: change-context proposer (passed through).
        cws: CognitiveWorkflowStage instance — used for checkpointing.
        ratchet: optional pre-built Ratchet (for tests). If None, the inner
            AutoresearchRunner builds one in setup().
        repair_fn: optional callback invoked on RECOVERABLE classification
            BEFORE the signature counter increments. Signature:
            `(error: BaseException, change_context: dict) -> dict | None`.
            If it returns a new change_context, the harness retries execute
            once with that context. If it returns None, the iteration is
            counted as a recoverable failure and the signature counter
            advances toward TERMINAL promotion. Per Article III, RECOVERABLE
            errors should route to a specialist for repair; this hook is
            the in-process embodiment of that route.
        meta_agent: optional MetaAgent instance for Tier-1 dial adjustments.
            When the harness sees the SAME RECOVERABLE signature twice in a
            row, it asks meta_agent for a proposal. Tier-1 (auto-apply)
            proposals are accepted and a single approved dial is mutated
            (e.g. `max_transient_retries`); Tier-2/3 are deferred to the
            Ratchet path per Article VI. None = no MetaAgent integration.
    """

    def __init__(
        self,
        config: CozyLoopConfig,
        *,
        execute_fn: Callable[[dict], dict[str, float]] | None = None,
        propose_fn: Callable[[], dict[str, Any]] | None = None,
        cws: Any | None = None,
        ratchet: Any | None = None,
        repair_fn: Callable[[BaseException, dict], dict | None] | None = None,
        meta_agent: Any | None = None,
    ):
        self._config = config
        self._execute_fn = execute_fn
        self._propose_fn = propose_fn
        self._cws = cws
        self._ratchet_override = ratchet
        self._repair_fn = repair_fn
        self._meta_agent = meta_agent
        self._meta_proposals_applied = 0
        self._last_recoverable_sig: str | None = None

        # Self-healing counters
        self._transient_retries_total = 0
        self._recoverable_repairs_total = 0
        self._terminal_count = 0
        self._recoverable_signature_counts: dict[str, int] = {}
        self._iterations_completed = 0

        # Lifecycle
        self._shutdown = threading.Event()
        self._health_thread: threading.Thread | None = None
        self._snapshots: list[HealthSnapshot] = []
        self._snapshots_lock = threading.Lock()
        self._start_time = 0.0

        # Resolve checkpoint path: explicit > env > None
        if config.checkpoint_path:
            self._checkpoint_path: Path | None = Path(config.checkpoint_path)
        else:
            try:
                from ..config import STAGE_DEFAULT_PATH
                self._checkpoint_path = (
                    Path(STAGE_DEFAULT_PATH) if STAGE_DEFAULT_PATH else None
                )
            except ImportError:
                self._checkpoint_path = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def snapshots(self) -> list[HealthSnapshot]:
        """Read-only view of health snapshots taken so far."""
        with self._snapshots_lock:
            return list(self._snapshots)

    def shutdown(self) -> None:
        """Request a clean shutdown. Safe to call from any thread."""
        self._shutdown.set()

    def run(self) -> CozyLoopResult:
        """Execute the harness loop. Blocks until budget exhausted or halt."""
        self._start_time = time.time()
        result = CozyLoopResult()

        # Spin up the health loop
        self._start_health_thread()

        # Build the inner runner with our wrapped execute callback
        runner_config = self._config.to_runner_config()
        wrapped_execute = self._wrap_execute(self._execute_fn) if self._execute_fn else None

        runner = AutoresearchRunner(
            runner_config,
            execute_fn=wrapped_execute,
            propose_fn=self._propose_fn,
            cws=self._cws,
        )

        # If we were given a ratchet, slot it in after setup. Otherwise the
        # runner builds its own in setup().
        runner.setup()
        if self._ratchet_override is not None:
            runner._ratchet = self._ratchet_override

        # Replace the runner's loop with our checkpointing variant. We can't
        # subclass cleanly because the loop and the loop body are entangled
        # in `run()`, so we re-implement just the iteration step here.
        try:
            self._driven_loop(runner, result)
        except _TerminalHalt as t:
            result.halt_reason = f"TERMINAL: {t.reason}"
            result.blocker_path = self._write_blocker(t.reason, t.error)
            self._final_checkpoint(runner)
        except KeyboardInterrupt:
            result.halt_reason = "interrupted"
            self._final_checkpoint(runner)
        else:
            # Normal exit — close the ratchet session and snapshot final state
            try:
                if runner._ratchet is not None:
                    runner._result.counterfactual_ids = runner._ratchet.close_session()
            except Exception as exc:
                log.warning("close_session failed: %s", exc)
            self._final_checkpoint(runner)
            result.halt_reason = result.halt_reason or runner._result.stopped_reason

        # Stop health loop and finalize
        self._shutdown.set()
        if self._health_thread is not None:
            self._health_thread.join(timeout=2.0)

        runner._result.total_seconds = time.time() - self._start_time
        result.run_result = runner._result
        result.total_seconds = time.time() - self._start_time
        result.health_snapshots = self.snapshots
        return result

    # ------------------------------------------------------------------
    # Inner loop — manually drives the runner so we can checkpoint
    # ------------------------------------------------------------------

    def _driven_loop(self, runner: AutoresearchRunner, result: CozyLoopResult) -> None:
        """Re-implementation of AutoresearchRunner.run()'s body.

        Identical semantics for budget / proposal / decide; adds checkpointing
        and self-healing classification on top.
        """
        budget_seconds = self._config.budget_hours * 3600
        last_checkpoint_time = time.time()

        while not self._shutdown.is_set():
            elapsed = time.time() - self._start_time
            if elapsed >= budget_seconds:
                runner._result.stopped_reason = "budget_exhausted"
                break
            if self._iterations_completed >= self._config.max_experiments:
                runner._result.stopped_reason = "max_experiments"
                break

            # Propose
            try:
                change_context = runner._propose()
            except Exception as exc:
                cls = self._classify_and_route(exc)
                if cls == TERMINAL:
                    raise _TerminalHalt(reason="proposal failed terminally", error=exc) from exc
                # Non-terminal proposal failures: skip to next iteration
                continue

            if change_context is None:
                runner._result.stopped_reason = "no_more_proposals"
                break

            # Execute (with self-healing wrapper if provided)
            exp_start = time.time()
            try:
                scores = runner._execute(change_context)
            except Exception as exc:
                # Pre-classification: try in-process repair if this looks
                # RECOVERABLE and a repair_fn is wired. The repair_fn embodies
                # the Article-III "route to a specialist" step inside a
                # single Python process; the Claude Code subagent layer is
                # the equivalent at the conversational layer.
                cls = self_healing_ladder(exc)
                if cls == RECOVERABLE and self._repair_fn is not None:
                    try:
                        new_ctx = self._repair_fn(exc, change_context)
                    except Exception as repair_exc:
                        log.warning(
                            "repair_fn raised %s — escalating original error",
                            type(repair_exc).__name__,
                        )
                        new_ctx = None
                    if new_ctx is not None:
                        # Retry with the repaired context. If THIS attempt
                        # raises, we fall through to the ladder below — no
                        # second repair attempt within one iteration to
                        # prevent infinite repair loops.
                        try:
                            scores = runner._execute(new_ctx)
                            change_context = new_ctx
                            exp_elapsed = time.time() - exp_start
                            self._recoverable_repairs_total += 1
                            log.info(
                                "RECOVERABLE %s repaired in-process",
                                type(exc).__name__,
                            )
                            # Skip the ladder and proceed to ratchet decide()
                            self._post_execute(
                                runner, scores, change_context, exp_elapsed,
                            )
                            self._maybe_checkpoint(runner, last_checkpoint_time)
                            last_checkpoint_time = time.time()
                            continue
                        except Exception as retry_exc:
                            exc = retry_exc  # re-classify with retry's error
                # Classification + counter routing
                cls = self._classify_and_route(exc)
                if cls == TERMINAL:
                    raise _TerminalHalt(reason="execution failed terminally", error=exc) from exc
                # TRANSIENT/RECOVERABLE: try a Tier-1 MetaAgent dial bump
                # before giving up on this iteration. Article VI says
                # Tier-1 fixes are auto-applied IFF they don't mutate state
                # — bumping `max_transient_retries` qualifies.
                if cls == RECOVERABLE:
                    self._maybe_apply_meta_agent_dial(exc)
                self._iterations_completed += 1
                continue
            exp_elapsed = time.time() - exp_start

            # Decide via the ratchet (Article VI: ratchet sovereignty)
            self._post_execute(runner, scores, change_context, exp_elapsed)
            last_checkpoint_time = self._maybe_checkpoint(
                runner, last_checkpoint_time,
            )

    def _post_execute(
        self,
        runner: AutoresearchRunner,
        scores: dict[str, float],
        change_context: dict,
        exp_elapsed: float,
    ) -> None:
        """Record the experiment via the ratchet. Extracted so the repair
        retry path and the happy path use the same record-keeping."""
        from ..stage.autoresearch_runner import ExperimentResult
        delta_id = f"exp_{self._iterations_completed}"
        kept = runner._ratchet.decide(
            delta_id, scores,
            change_context=change_context,
        )
        runner._result.experiments.append(ExperimentResult(
            delta_id=delta_id,
            axis_scores=scores,
            kept=kept,
            composite=runner._ratchet.history[-1].composite,
            change_context=change_context,
            elapsed_seconds=exp_elapsed,
        ))
        self._iterations_completed += 1

    def _maybe_checkpoint(
        self,
        runner: AutoresearchRunner,
        last_checkpoint_time: float,
    ) -> float:
        """Checkpoint if iteration-count or time threshold crossed.
        Returns the (possibly updated) last_checkpoint_time."""
        now = time.time()
        should_checkpoint = (
            self._iterations_completed % self._config.checkpoint_every_n == 0
            or (now - last_checkpoint_time) >= self._config.checkpoint_every_seconds
        )
        if should_checkpoint:
            self._checkpoint(runner)
            return now
        return last_checkpoint_time

    # ------------------------------------------------------------------
    # Self-healing ladder (Article III)
    # ------------------------------------------------------------------

    def _wrap_execute(
        self,
        inner: Callable[[dict], dict[str, float]],
    ) -> Callable[[dict], dict[str, float]]:
        """Wrap the execute callback with TRANSIENT-retry + classification."""

        def _wrapped(change_context: dict) -> dict[str, float]:
            last_exc: BaseException | None = None
            for attempt in range(self._config.max_transient_retries + 1):
                try:
                    return inner(change_context)
                except Exception as exc:
                    cls = self_healing_ladder(exc)
                    if cls == TRANSIENT and attempt < self._config.max_transient_retries:
                        # Backoff and retry
                        idx = min(attempt, len(self._config.transient_backoff_seconds) - 1)
                        wait = self._config.transient_backoff_seconds[idx]
                        log.info(
                            "TRANSIENT %s — backoff %.1fs (attempt %d/%d)",
                            type(exc).__name__, wait,
                            attempt + 1, self._config.max_transient_retries,
                        )
                        self._transient_retries_total += 1
                        time.sleep(wait)
                        last_exc = exc
                        continue
                    # Re-raise — classify_and_route in driver decides next action
                    raise
            # Exhausted retries → re-raise
            assert last_exc is not None
            raise last_exc

        return _wrapped

    def _classify_and_route(self, exc: BaseException) -> str:
        """Classify an error and update counters. Returns the classification."""
        cls = self_healing_ladder(exc)
        if cls == TRANSIENT:
            self._transient_retries_total += 1
            return TRANSIENT

        if cls == RECOVERABLE:
            sig = type(exc).__name__
            self._recoverable_signature_counts[sig] = (
                self._recoverable_signature_counts.get(sig, 0) + 1
            )
            self._recoverable_repairs_total += 1
            # Promote to TERMINAL if same recoverable signature exceeds limit
            if self._recoverable_signature_counts[sig] > self._config.max_recoverable_per_signature:
                log.warning(
                    "RECOVERABLE '%s' exceeded %d repeats — promoting to TERMINAL",
                    sig, self._config.max_recoverable_per_signature,
                )
                self._terminal_count += 1
                return TERMINAL
            return RECOVERABLE

        # TERMINAL
        self._terminal_count += 1
        return TERMINAL

    def _maybe_apply_meta_agent_dial(self, exc: BaseException) -> None:
        """Optional Article-VI escape: ask MetaAgent for a Tier-1 dial bump
        when the same RECOVERABLE signature repeats consecutively.

        Tier-1 means non-state-mutating — the only dials we mutate from
        here are `max_transient_retries` and `transient_backoff_seconds`.
        Tier-2/3 proposals are NOT applied because they require Ratchet
        scoring infrastructure for harness state, which we don't have.
        Such proposals are logged for human review.
        """
        if self._meta_agent is None:
            return
        sig = type(exc).__name__
        # Only act if this is the SAME signature as last time — avoids
        # racing toward dial bumps on every recoverable error.
        if self._last_recoverable_sig != sig:
            self._last_recoverable_sig = sig
            return
        try:
            improvement = self._meta_agent.propose_improvement(
                category="optimization_param",
                description=f"Repeated {sig} suggests retry budget too low",
                proposed_change={"dial": "max_transient_retries", "delta": +1},
                rationale=(
                    f"Observed {sig} twice consecutively in harness loop; "
                    f"current max_transient_retries="
                    f"{self._config.max_transient_retries}"
                ),
            )
        except Exception as meta_exc:
            log.warning("MetaAgent.propose_improvement raised: %s", meta_exc)
            return
        if not self._meta_agent.can_auto_apply(improvement):
            log.info(
                "MetaAgent proposal %s requires ratchet/human gate — deferred",
                getattr(improvement, "improvement_id", "?"),
            )
            return
        # Apply the dial bump
        self._config.max_transient_retries += 1
        self._meta_proposals_applied += 1
        log.info(
            "MetaAgent Tier-1 applied: max_transient_retries -> %d",
            self._config.max_transient_retries,
        )

    # ------------------------------------------------------------------
    # Checkpointing (Article IV)
    # ------------------------------------------------------------------

    def _checkpoint(self, runner: AutoresearchRunner) -> None:
        """Flush stage + ratchet history. No-op if no checkpoint path set."""
        if self._checkpoint_path is None or self._cws is None:
            return
        try:
            self._cws.flush(self._checkpoint_path)
        except Exception as exc:
            log.warning("Stage checkpoint failed: %s", exc)
            return

        # Ratchet history sidecar
        try:
            from ..memory.session import save_ratchet
            if runner._ratchet is not None:
                save_ratchet(self._config.session_name, runner._ratchet)
        except Exception as exc:
            log.warning("Ratchet checkpoint failed: %s", exc)

    def _final_checkpoint(self, runner: AutoresearchRunner) -> None:
        """Final flush at halt — Article IV mandates a checkpoint before exit."""
        self._checkpoint(runner)

    def _write_blocker(self, reason: str, error: BaseException | None) -> str | None:
        """Write a BLOCKER.md atomically per the repo's Git Authority C3 convention.

        MoE-R6: write to <path>.tmp first, then os.replace into place.
        Mirrors the atomic-flush pattern at agent/stage/cognitive_stage.py.
        A SIGKILL between Export and replace leaves the .tmp orphaned but
        the canonical BLOCKER.md is either intact (pre-write) or fully
        rewritten (post-write) — never partially written. Important: this
        file is the post-mortem; if it's corrupt, debugging the halt is
        much harder.
        """
        import os
        try:
            from ..config import PROJECT_DIR
            path = PROJECT_DIR / "BLOCKER.md"
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            content = (
                f"# Cozy Harness BLOCKER\n\n"
                f"Halted at iteration {self._iterations_completed} after "
                f"{time.time() - self._start_time:.1f}s.\n\n"
                f"## Reason\n\n{reason}\n\n"
            )
            if error is not None:
                content += f"## Error\n\n```\n{type(error).__name__}: {error}\n```\n"
            try:
                tmp_path.write_text(content, encoding="utf-8")
                os.replace(tmp_path, path)
            except Exception:
                # Best-effort cleanup of the orphan .tmp.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            return str(path)
        except Exception as exc:
            log.warning("Failed to write BLOCKER.md: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Health loop
    # ------------------------------------------------------------------

    def _start_health_thread(self) -> None:
        """Spawn the daemon health-snapshot thread."""

        def _tick():
            while not self._shutdown.is_set():
                snap = HealthSnapshot(
                    timestamp=time.time(),
                    iterations_completed=self._iterations_completed,
                    transient_retries_total=self._transient_retries_total,
                    recoverable_repairs_total=self._recoverable_repairs_total,
                    terminal_count=self._terminal_count,
                    elapsed_seconds=time.time() - self._start_time,
                )
                with self._snapshots_lock:
                    self._snapshots.append(snap)
                self._shutdown.wait(self._config.health_check_seconds)

        t = threading.Thread(target=_tick, daemon=True, name="cozy-health")
        t.start()
        self._health_thread = t


# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------

class _TerminalHalt(Exception):
    """Internal signal that the harness must halt cleanly per Article III."""

    def __init__(self, reason: str, error: BaseException | None = None):
        super().__init__(reason)
        self.reason = reason
        self.error = error
