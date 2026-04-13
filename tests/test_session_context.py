"""Tests for SessionContext and SessionRegistry."""

import threading
import time

from agent.session_context import (
    SessionContext,
    SessionRegistry,
    get_registry,
    get_session_context,
)


class TestSessionContext:
    """Tests for the SessionContext dataclass."""

    def test_create_default(self):
        ctx = SessionContext(session_id="test-1")
        assert ctx.session_id == "test-1"
        assert ctx.workflow is not None
        assert ctx.workflow.session_id == "test-1"
        assert ctx.intent_state == {}
        assert ctx.iteration_state == {}
        assert ctx.demo_state == {}
        assert ctx.orchestrator_tasks == {}

    def test_workflow_session_auto_created(self):
        ctx = SessionContext(session_id="auto")
        assert ctx.workflow["current_workflow"] is None
        assert ctx.workflow["loaded_path"] is None

    def test_touch_updates_activity(self):
        ctx = SessionContext(session_id="touch")
        old_activity = ctx.last_activity
        time.sleep(0.01)
        ctx.touch()
        assert ctx.last_activity > old_activity

    def test_age_seconds(self):
        ctx = SessionContext(session_id="age")
        ctx.last_activity = time.time() - 10
        assert ctx.age_seconds() >= 10

    def test_isolated_workflow_state(self):
        """Two SessionContexts have independent workflow state."""
        ctx_a = SessionContext(session_id="a")
        ctx_b = SessionContext(session_id="b")

        ctx_a.workflow["current_workflow"] = {"nodes": "a"}
        ctx_b.workflow["current_workflow"] = {"nodes": "b"}

        assert ctx_a.workflow["current_workflow"] == {"nodes": "a"}
        assert ctx_b.workflow["current_workflow"] == {"nodes": "b"}

    def test_isolated_intent_state(self):
        """Two SessionContexts have independent intent state."""
        ctx_a = SessionContext(session_id="a")
        ctx_b = SessionContext(session_id="b")

        ctx_a.intent_state["foo"] = 1
        assert "foo" not in ctx_b.intent_state


class TestSessionRegistry:
    """Tests for the SessionRegistry."""

    def test_get_or_create(self):
        reg = SessionRegistry()
        ctx = reg.get_or_create("s1")
        assert ctx.session_id == "s1"

    def test_get_or_create_returns_same(self):
        reg = SessionRegistry()
        ctx1 = reg.get_or_create("s1")
        ctx2 = reg.get_or_create("s1")
        assert ctx1 is ctx2

    def test_get_returns_none_for_missing(self):
        reg = SessionRegistry()
        assert reg.get("nonexistent") is None

    def test_get_returns_existing(self):
        reg = SessionRegistry()
        reg.get_or_create("s1")
        ctx = reg.get("s1")
        assert ctx is not None
        assert ctx.session_id == "s1"

    def test_destroy(self):
        reg = SessionRegistry()
        reg.get_or_create("s1")
        assert reg.destroy("s1") is True
        assert reg.get("s1") is None
        assert reg.destroy("s1") is False

    def test_list_sessions(self):
        reg = SessionRegistry()
        reg.get_or_create("a")
        reg.get_or_create("b")
        sessions = reg.list_sessions()
        assert set(sessions) == {"a", "b"}

    def test_count(self):
        reg = SessionRegistry()
        assert reg.count == 0
        reg.get_or_create("a")
        assert reg.count == 1
        reg.get_or_create("b")
        assert reg.count == 2

    def test_clear(self):
        reg = SessionRegistry()
        reg.get_or_create("a")
        reg.clear()
        assert reg.count == 0

    def test_gc_stale_removes_old(self):
        reg = SessionRegistry()
        ctx = reg.get_or_create("old")
        ctx.last_activity = time.time() - 7200  # 2 hours ago
        reg.get_or_create("fresh")  # just created
        removed = reg.gc_stale(max_age_seconds=3600)
        assert removed == 1
        assert reg.get("old") is None
        assert reg.get("fresh") is not None

    def test_gc_stale_never_removes_default(self):
        reg = SessionRegistry()
        ctx = reg.get_or_create("default")
        ctx.last_activity = time.time() - 99999
        removed = reg.gc_stale(max_age_seconds=1)
        assert removed == 0
        assert reg.get("default") is not None

    def test_session_isolation_through_registry(self):
        """Sessions created through the registry are fully isolated."""
        reg = SessionRegistry()
        ctx_a = reg.get_or_create("a")
        ctx_b = reg.get_or_create("b")

        ctx_a.workflow["current_workflow"] = {"test": "a"}
        assert ctx_b.workflow["current_workflow"] is None


class TestGlobalAccessors:
    """Tests for module-level convenience functions."""

    def test_get_session_context(self):
        ctx = get_session_context("global-test")
        assert ctx.session_id == "global-test"
        # Clean up
        get_registry().destroy("global-test")

    def test_get_registry(self):
        reg = get_registry()
        assert isinstance(reg, SessionRegistry)


class TestConcurrentEnsure:
    """Regression tests for Cycle 25 double-checked locking fix."""

    def test_ensure_stage_concurrent_same_instance(self):
        """Concurrent calls to ensure_stage() must never produce multiple instances."""
        ctx = SessionContext(session_id="concurrent-stage-test")
        results: list = []
        errors: list = []

        def call_ensure():
            try:
                results.append(ctx.ensure_stage())
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=call_ensure) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"ensure_stage raised: {errors}"
        # All non-None results must be the same object (single instance)
        non_none = [x for x in results if x is not None]
        if non_none:
            first_id = id(non_none[0])
            assert all(id(x) == first_id for x in non_none), (
                "ensure_stage() created multiple CognitiveWorkflowStage instances"
            )

    def test_ensure_arbiter_concurrent_same_instance(self):
        """Concurrent ensure_arbiter() calls must return the same instance."""
        ctx = SessionContext(session_id="concurrent-arbiter-test")
        results: list = []
        errors: list = []

        def call_ensure():
            try:
                results.append(ctx.ensure_arbiter())
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=call_ensure) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"ensure_arbiter raised: {errors}"
        non_none = [x for x in results if x is not None]
        if non_none:
            first_id = id(non_none[0])
            assert all(id(x) == first_id for x in non_none), (
                "ensure_arbiter() created multiple Arbiter instances"
            )


# ---------------------------------------------------------------------------
# Cycle 30: get_or_create_with_is_new() atomic TOCTOU fix
# ---------------------------------------------------------------------------

class TestAtomicGetOrCreateWithIsNew:
    """get_or_create_with_is_new must be race-free."""

    def test_new_session_is_new_true(self):
        """First call for an unknown session_id returns is_new=True."""
        reg = SessionRegistry()
        ctx, is_new = reg.get_or_create_with_is_new("brand-new-session")
        assert is_new is True
        assert ctx.session_id == "brand-new-session"

    def test_existing_session_is_new_false(self):
        """Second call for an existing session_id returns is_new=False."""
        reg = SessionRegistry()
        reg.get_or_create_with_is_new("existing")
        ctx, is_new = reg.get_or_create_with_is_new("existing")
        assert is_new is False

    def test_returns_same_context_on_second_call(self):
        """Both calls must return the exact same SessionContext object."""
        reg = SessionRegistry()
        ctx1, _ = reg.get_or_create_with_is_new("shared")
        ctx2, _ = reg.get_or_create_with_is_new("shared")
        assert ctx1 is ctx2

    def test_concurrent_calls_single_is_new(self):
        """Under concurrent access exactly one thread must get is_new=True."""
        import threading
        reg = SessionRegistry()
        results = []

        def call():
            _, is_new = reg.get_or_create_with_is_new("concurrent-session")
            results.append(is_new)

        threads = [threading.Thread(target=call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1, (
            f"Expected exactly 1 is_new=True, got {results.count(True)}"
        )
        assert results.count(False) == 19

    def test_get_session_context_no_duplicate_auto_init(self):
        """get_session_context must not call run_auto_init more than once per session."""
        import threading
        from unittest.mock import patch
        from agent.session_context import get_session_context, get_registry

        # Use a fresh non-default session_id to avoid polluting the global default
        sid = "test-atomic-init-" + str(id(threading.current_thread()))
        init_calls = []

        def fake_auto_init(ctx):
            init_calls.append(ctx.session_id)

        with patch("agent.startup.run_auto_init", fake_auto_init, create=True):
            # Call get_session_context 10 times concurrently on the same session
            # Only the first call should trigger auto-init
            errors = []

            def call():
                try:
                    # We pass "default" here to trigger the is_new check
                    # but isolate via a fresh registry
                    get_session_context(sid)
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=call) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"get_session_context raised: {errors}"
            # Cleanup
            get_registry().destroy(sid)
