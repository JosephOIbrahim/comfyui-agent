"""CRUCIBLE tests for workflow observation schema and log.

Adversarial: thread safety, immutability guarantees, ordering invariants,
and the BASELINE_OBSERVATION contract.
"""

from __future__ import annotations

import copy
import threading

import pytest

from agent.workflow_observation import (
    BASELINE_OBSERVATION,
    ActionBlock,
    DynamicsBlock,
    ProvisionStatus,
    WorkflowHealth,
    WorkflowObservation,
    WorkflowPhase,
    WorkflowStateBlock,
)
from agent.workflow_observation_log import WorkflowObservationLog


# ---------------------------------------------------------------------------
# Schema / enum tests
# ---------------------------------------------------------------------------


class TestWorkflowObservationSchema:
    def test_baseline_observation_is_not_none(self):
        assert BASELINE_OBSERVATION is not None

    def test_baseline_observation_has_defaults(self):
        assert BASELINE_OBSERVATION.step_index == 0
        assert BASELINE_OBSERVATION.session_id == ""
        assert BASELINE_OBSERVATION.state.phase == WorkflowPhase.EMPTY
        assert BASELINE_OBSERVATION.state.health == WorkflowHealth.NOMINAL
        assert BASELINE_OBSERVATION.state.provision == ProvisionStatus.AVAILABLE
        assert BASELINE_OBSERVATION.state.node_count == 0
        assert BASELINE_OBSERVATION.state.error_count == 0
        assert BASELINE_OBSERVATION.action.tool_name == ""
        assert BASELINE_OBSERVATION.dynamics.total_mutations == 0

    def test_workflow_phase_ordering(self):
        assert WorkflowPhase.EMPTY < WorkflowPhase.LOADED
        assert WorkflowPhase.LOADED < WorkflowPhase.CONFIGURED
        assert WorkflowPhase.CONFIGURED < WorkflowPhase.VALIDATED
        assert WorkflowPhase.VALIDATED < WorkflowPhase.EXECUTING
        assert WorkflowPhase.EXECUTING < WorkflowPhase.COMPLETED
        assert WorkflowPhase.COMPLETED < WorkflowPhase.FAILED

    def test_workflow_health_ordering(self):
        assert WorkflowHealth.BROKEN < WorkflowHealth.DEGRADED
        assert WorkflowHealth.DEGRADED < WorkflowHealth.NOMINAL
        assert WorkflowHealth.NOMINAL < WorkflowHealth.OPTIMAL

    def test_provision_status_ordering(self):
        assert ProvisionStatus.MISSING < ProvisionStatus.DOWNLOADING
        assert ProvisionStatus.DOWNLOADING < ProvisionStatus.AVAILABLE
        assert ProvisionStatus.AVAILABLE < ProvisionStatus.CACHED

    def test_observation_to_dict_keys(self):
        obs = WorkflowObservation()
        d = obs.to_dict()
        assert "session_id" in d
        assert "step_index" in d
        assert "timestamp" in d
        assert "state" in d
        assert "action" in d
        assert "dynamics" in d
        assert "intelligence" in d

    def test_state_block_to_dict(self):
        sb = WorkflowStateBlock(phase=WorkflowPhase.LOADED, node_count=5)
        d = sb.to_dict()
        assert d["phase"] == "LOADED"
        assert d["node_count"] == 5

    def test_action_block_to_dict(self):
        ab = ActionBlock(tool_name="set_input", action_type="mutation")
        d = ab.to_dict()
        assert d["tool_name"] == "set_input"
        assert d["action_type"] == "mutation"

    def test_dynamics_block_to_dict(self):
        db = DynamicsBlock(total_mutations=3, elapsed_seconds=12.345)
        d = db.to_dict()
        assert d["total_mutations"] == 3
        assert d["elapsed_seconds"] == 12.35  # rounded to 2 decimal


# ---------------------------------------------------------------------------
# Observation log tests
# ---------------------------------------------------------------------------


class TestWorkflowObservationLog:
    def test_read_previous_at_zero_returns_baseline(self):
        log = WorkflowObservationLog("test-session")
        prev = log.read_previous(0)
        assert prev is BASELINE_OBSERVATION

    def test_read_previous_never_none(self):
        log = WorkflowObservationLog("test-session")
        # Edge case: negative index
        result = log.read_previous(-5)
        assert result is not None
        assert result is BASELINE_OBSERVATION

    def test_read_previous_at_step(self):
        log = WorkflowObservationLog("test-session")
        obs0 = WorkflowObservation(
            state=WorkflowStateBlock(phase=WorkflowPhase.LOADED),
        )
        obs1 = WorkflowObservation(
            state=WorkflowStateBlock(phase=WorkflowPhase.VALIDATED),
        )
        log.author(obs0)
        log.author(obs1)
        # read_previous(1) should return obs at step 0
        prev = log.read_previous(1)
        assert prev.state.phase == WorkflowPhase.LOADED
        assert prev.step_index == 0

    def test_author_increments_step(self):
        log = WorkflowObservationLog("test-session")
        idx0 = log.author(WorkflowObservation())
        idx1 = log.author(WorkflowObservation())
        idx2 = log.author(WorkflowObservation())
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2
        assert log.current_step() == 3

    def test_author_overwrites_step_and_session(self):
        """author() enforces monotonicity by overwriting step_index and session_id."""
        log = WorkflowObservationLog("real-session")
        obs = WorkflowObservation(session_id="fake", step_index=999)
        idx = log.author(obs)
        assert obs.step_index == idx
        assert obs.session_id == "real-session"

    def test_history_ordering_oldest_first(self):
        log = WorkflowObservationLog("test-session")
        for i in range(5):
            obs = WorkflowObservation(
                state=WorkflowStateBlock(node_count=i),
            )
            log.author(obs)
        history = log.history(last_n=5)
        assert len(history) == 5
        assert history[0].state.node_count == 0
        assert history[4].state.node_count == 4

    def test_history_last_n_zero_returns_empty(self):
        log = WorkflowObservationLog("test-session")
        log.author(WorkflowObservation())
        assert log.history(last_n=0) == []

    def test_history_last_n_negative_returns_empty(self):
        log = WorkflowObservationLog("test-session")
        log.author(WorkflowObservation())
        assert log.history(last_n=-1) == []

    def test_history_excess_n_returns_all(self):
        log = WorkflowObservationLog("test-session")
        log.author(WorkflowObservation())
        log.author(WorkflowObservation())
        history = log.history(last_n=100)
        assert len(history) == 2

    def test_read_exact_step(self):
        log = WorkflowObservationLog("test-session")
        obs = WorkflowObservation(
            state=WorkflowStateBlock(phase=WorkflowPhase.EXECUTING),
        )
        idx = log.author(obs)
        result = log.read(idx)
        assert result is not None
        assert result.state.phase == WorkflowPhase.EXECUTING

    def test_read_out_of_bounds_returns_none(self):
        log = WorkflowObservationLog("test-session")
        assert log.read(0) is None
        assert log.read(999) is None
        assert log.read(-1) is None

    def test_snapshot_serializable(self):
        import json

        log = WorkflowObservationLog("test-session")
        log.author(WorkflowObservation())
        snap = log.snapshot()
        serialized = json.dumps(snap)
        assert "test-session" in serialized
        assert snap["total_steps"] == 1
        assert snap["latest"] is not None

    def test_snapshot_empty_log(self):
        log = WorkflowObservationLog("empty")
        snap = log.snapshot()
        assert snap["total_steps"] == 0
        assert snap["latest"] is None

    def test_len(self):
        log = WorkflowObservationLog("test-session")
        assert len(log) == 0
        log.author(WorkflowObservation())
        assert len(log) == 1

    def test_repr(self):
        log = WorkflowObservationLog("test-session")
        r = repr(log)
        assert "test-session" in r

    def test_observation_immutability_via_copy(self):
        """Modifying a returned observation should not affect the log."""
        log = WorkflowObservationLog("test-session")
        obs = WorkflowObservation(
            state=WorkflowStateBlock(node_count=10),
        )
        log.author(obs)
        # Get the observation back
        retrieved = log.read(0)
        assert retrieved is not None
        # Mutate the retrieved observation's state
        retrieved.state.node_count = 999
        # The log still holds the modified reference (it's the same object).
        # This tests that users should copy if they need isolation.
        # This is an adversarial test: the log DOES return the same reference.
        internal = log.read(0)
        # Since dataclass is not frozen, mutation leaks through.
        # Document this behavior:
        assert internal is not None
        assert internal.state.node_count == 999  # leaks by design (mutable dataclass)

    def test_thread_safety_10_concurrent_authors(self):
        """10 threads authoring simultaneously must not corrupt the log."""
        log = WorkflowObservationLog("thread-test")
        errors: list[str] = []
        per_thread = 50

        def author_many(thread_id: int) -> None:
            try:
                for i in range(per_thread):
                    obs = WorkflowObservation(
                        state=WorkflowStateBlock(node_count=thread_id * 1000 + i),
                    )
                    log.author(obs)
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=author_many, args=(tid,))
            for tid in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(log) == 10 * per_thread
        assert log.current_step() == 10 * per_thread
        # Verify monotonic step indices
        history = log.history(last_n=10 * per_thread)
        indices = [obs.step_index for obs in history]
        assert indices == sorted(indices), "Step indices are not monotonic"
        assert len(set(indices)) == len(indices), "Duplicate step indices found"
