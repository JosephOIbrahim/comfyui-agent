"""CRUCIBLE tests for LIVRPS Mutation Bridge (agent/stage/mutation_bridge.py).

Adversarial: tests degraded mode (no USD), multi-agent composition,
rollback, thread safety, and audit trail integrity.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent.stage.mutation_bridge import MutationBridge, MutationResolution


# ---------------------------------------------------------------------------
# Basics (degraded mode — no USD available in test env)
# ---------------------------------------------------------------------------


class TestMutationBridgeBasics:
    def test_mutate_creates_resolution(self):
        bridge = MutationBridge(stage=None)
        res = bridge.mutate("set_input", "agent_A", {"seed": 42})
        assert isinstance(res, MutationResolution)

    def test_mutate_records_agent_name(self):
        bridge = MutationBridge(stage=None)
        res = bridge.mutate("set_input", "agent_B", {"seed": 42})
        assert res.agent_name == "agent_B"

    def test_mutate_records_operation(self):
        bridge = MutationBridge(stage=None)
        res = bridge.mutate("apply_patch", "agent_A", {"cfg": 7.0})
        assert res.operation == "apply_patch"

    def test_mutate_has_timestamp(self):
        bridge = MutationBridge(stage=None)
        before = time.time()
        res = bridge.mutate("set_input", "agent_A", {"x": 1})
        after = time.time()
        assert before <= res.timestamp <= after

    def test_mutate_has_delta_layer_id(self):
        bridge = MutationBridge(stage=None)
        res = bridge.mutate("set_input", "agent_A", {"x": 1})
        assert res.delta_layer_id.startswith("degraded_")
        assert "agent_A" in res.delta_layer_id

    def test_mutate_resolved_values_prefixed(self):
        """Keys should be prefixed with workflow_path."""
        bridge = MutationBridge(stage=None)
        res = bridge.mutate("set_input", "agent_A", {"seed": 42})
        assert "/workflows/current/seed" in res.resolved_values

    def test_mutate_absolute_keys_not_double_prefixed(self):
        """Keys starting with / should not be double-prefixed."""
        bridge = MutationBridge(stage=None)
        res = bridge.mutate(
            "set_input", "agent_A",
            {"/custom/path:attr": "val"},
        )
        assert "/custom/path:attr" in res.resolved_values

    def test_mutate_custom_workflow_path(self):
        bridge = MutationBridge(stage=None)
        res = bridge.mutate(
            "set_input", "agent_A", {"seed": 42},
            workflow_path="/workflows/alt",
        )
        assert "/workflows/alt/seed" in res.resolved_values


# ---------------------------------------------------------------------------
# Resolution list
# ---------------------------------------------------------------------------


class TestResolutionList:
    def test_list_resolutions_empty_initially(self):
        bridge = MutationBridge(stage=None)
        assert bridge.list_resolutions() == []

    def test_list_resolutions_accumulates(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "A", {"x": 1})
        bridge.mutate("op2", "B", {"y": 2})
        bridge.mutate("op3", "A", {"z": 3})
        resolutions = bridge.list_resolutions()
        assert len(resolutions) == 3
        assert resolutions[0].operation == "op1"
        assert resolutions[2].operation == "op3"

    def test_list_resolutions_returns_copy(self):
        """Modifying the returned list should not affect the bridge."""
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "A", {"x": 1})
        resolutions = bridge.list_resolutions()
        resolutions.clear()
        assert len(bridge.list_resolutions()) == 1


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_agent_removes_deltas(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "agent_A", {"x": 1})
        bridge.mutate("op2", "agent_A", {"y": 2})
        bridge.mutate("op3", "agent_B", {"z": 3})
        removed = bridge.rollback_agent("agent_A")
        assert removed == 2
        remaining = bridge.list_resolutions()
        assert len(remaining) == 1
        assert remaining[0].agent_name == "agent_B"

    def test_rollback_unknown_agent_returns_zero(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "agent_A", {"x": 1})
        removed = bridge.rollback_agent("nonexistent")
        assert removed == 0

    def test_rollback_idempotent(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "agent_A", {"x": 1})
        bridge.rollback_agent("agent_A")
        removed = bridge.rollback_agent("agent_A")
        assert removed == 0

    def test_rollback_clears_degraded_deltas(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "agent_A", {"x": 1})
        bridge.rollback_agent("agent_A")
        state = bridge.get_composed_state()
        # Agent A's deltas should be gone from composed state
        assert "/workflows/current/x" not in state


# ---------------------------------------------------------------------------
# Composed state
# ---------------------------------------------------------------------------


class TestComposedState:
    def test_composed_state_empty(self):
        bridge = MutationBridge(stage=None)
        assert bridge.get_composed_state() == {}

    def test_composed_state_after_mutations(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "A", {"seed": 42})
        bridge.mutate("op2", "B", {"cfg": 7.0})
        state = bridge.get_composed_state()
        assert state["/workflows/current/seed"] == 42
        assert state["/workflows/current/cfg"] == 7.0

    def test_composed_state_last_write_wins_degraded(self):
        """In degraded mode, last write wins per key."""
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "A", {"seed": 1})
        bridge.mutate("op2", "B", {"seed": 999})
        state = bridge.get_composed_state()
        assert state["/workflows/current/seed"] == 999


# ---------------------------------------------------------------------------
# Degraded mode detection
# ---------------------------------------------------------------------------


class TestDegradedMode:
    def test_degraded_mode_without_usd(self):
        bridge = MutationBridge(stage=None)
        assert bridge.has_stage is False

    def test_degraded_mode_overridden_by_empty(self):
        """Degraded mode still returns [] for overridden_by."""
        bridge = MutationBridge(stage=None)
        res = bridge.mutate("op1", "A", {"x": 1})
        assert res.overridden_by == []


# ---------------------------------------------------------------------------
# Multiple agents
# ---------------------------------------------------------------------------


class TestMultipleAgents:
    def test_multiple_agents_compose(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "lighting_td", {"exposure": 2.0})
        bridge.mutate("op2", "compositor", {"grade": "warm"})
        state = bridge.get_composed_state()
        assert "/workflows/current/exposure" in state
        assert "/workflows/current/grade" in state

    def test_rollback_one_agent_preserves_other(self):
        bridge = MutationBridge(stage=None)
        bridge.mutate("op1", "A", {"x": 1})
        bridge.mutate("op2", "B", {"y": 2})
        bridge.rollback_agent("A")
        state = bridge.get_composed_state()
        assert "/workflows/current/y" in state
        assert "/workflows/current/x" not in state


# ---------------------------------------------------------------------------
# MutationResolution immutability
# ---------------------------------------------------------------------------


class TestMutationResolutionImmutability:
    def test_resolution_is_frozen(self):
        bridge = MutationBridge(stage=None)
        res = bridge.mutate("op1", "A", {"x": 1})
        with pytest.raises(AttributeError):
            res.operation = "hacked"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            res.agent_name = "evil"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestMutationBridgeThreadSafety:
    def test_concurrent_mutations(self):
        bridge = MutationBridge(stage=None)
        errors: list[str] = []
        per_thread = 50

        def mutate_many(agent_id: int) -> None:
            try:
                for i in range(per_thread):
                    bridge.mutate(
                        f"op_{agent_id}_{i}",
                        f"agent_{agent_id}",
                        {f"key_{agent_id}_{i}": i},
                    )
            except Exception as exc:
                errors.append(f"Agent {agent_id}: {exc}")

        threads = [
            threading.Thread(target=mutate_many, args=(tid,))
            for tid in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        resolutions = bridge.list_resolutions()
        assert len(resolutions) == 10 * per_thread

    def test_concurrent_mutate_and_rollback(self):
        bridge = MutationBridge(stage=None)
        errors: list[str] = []

        def mutate_agent_a() -> None:
            try:
                for i in range(30):
                    bridge.mutate("op", "agent_A", {f"a_{i}": i})
            except Exception as exc:
                errors.append(str(exc))

        def rollback_agent_a() -> None:
            try:
                for _ in range(10):
                    bridge.rollback_agent("agent_A")
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(str(exc))

        t1 = threading.Thread(target=mutate_agent_a)
        t2 = threading.Thread(target=rollback_agent_a)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # No assertion on count — race is expected. Just no crash.


# ---------------------------------------------------------------------------
# Cycle 62: pre-mutation stage read failure must log at DEBUG
# ---------------------------------------------------------------------------

class TestPreMutationReadLogging:
    """stage.read() failure in _mutate_usd → log.debug (Cycle 62)."""

    def test_stage_read_failure_logs_debug(self, caplog):
        """When stage.read() raises during pre-mutation snapshot, debug log appears."""
        import logging
        from unittest.mock import MagicMock, patch

        mock_stage = MagicMock()
        mock_stage.read.side_effect = RuntimeError("USD not writeable")
        mock_stage.add_agent_delta.return_value = "layer_001"

        bridge = MutationBridge(stage=mock_stage)

        # HAS_USD is False in test env — patch it so has_stage returns True
        with patch("agent.stage.mutation_bridge.HAS_USD", True), \
             caplog.at_level(logging.DEBUG, logger="agent.stage.mutation_bridge"):
            # Use "key:attr" format so colon parsing resolves prim_path + attr_name
            bridge.mutate("set_input", "test_agent", {"/workflows/current/seed:value": 42})

        assert any("pre-mutation" in r.message.lower() or "read" in r.message.lower()
                   for r in caplog.records), "Expected debug log on pre-mutation read failure"

    def test_stage_read_failure_continues_mutation(self):
        """Pre-mutation read failure must not abort the mutation."""
        from unittest.mock import MagicMock, patch

        mock_stage = MagicMock()
        mock_stage.read.side_effect = RuntimeError("USD error")
        mock_stage.add_agent_delta.return_value = "layer_x"

        bridge = MutationBridge(stage=mock_stage)
        with patch("agent.stage.mutation_bridge.HAS_USD", True):
            result = bridge.mutate("set_input", "agent_x", {"/w/current/cfg:value": 7.5})

        assert result is not None
        assert result.agent_name == "agent_x"
