"""Tests for WorkflowSession — dict-like access, registry isolation, thread safety."""

import copy
import threading

import pytest

from agent.workflow_session import WorkflowSession, get_session, clear_sessions, _MAX_SESSIONS


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clear session registry between tests."""
    clear_sessions()
    yield
    clear_sessions()


# ---------------------------------------------------------------------------
# WorkflowSession dict-like access
# ---------------------------------------------------------------------------

class TestWorkflowSessionAccess:
    def test_getitem(self):
        s = WorkflowSession("test")
        assert s["loaded_path"] is None

    def test_setitem(self):
        s = WorkflowSession("test")
        s["loaded_path"] = "/some/path.json"
        assert s["loaded_path"] == "/some/path.json"

    def test_get_with_default(self):
        s = WorkflowSession("test")
        assert s.get("loaded_path") is None
        assert s.get("nonexistent", "fallback") == "fallback"

    def test_contains(self):
        s = WorkflowSession("test")
        assert "loaded_path" in s
        assert "nonexistent" not in s

    def test_keys(self):
        s = WorkflowSession("test")
        assert set(s.keys()) == {"loaded_path", "base_workflow", "current_workflow", "history", "format", "_engine"}

    def test_values(self):
        s = WorkflowSession("test")
        vals = list(s.values())
        # Default: None, None, None, [], None, None (includes _engine)
        assert len(vals) == 6

    def test_items(self):
        s = WorkflowSession("test")
        items = dict(s.items())
        assert items["loaded_path"] is None
        assert items["history"] == []

    def test_getitem_keyerror(self):
        s = WorkflowSession("test")
        with pytest.raises(KeyError):
            _ = s["nonexistent"]

    def test_history_is_mutable_list(self):
        """History accessed via __getitem__ should be the same mutable list."""
        s = WorkflowSession("test")
        s["history"].append({"snapshot": 1})
        assert len(s["history"]) == 1
        assert s["history"][0] == {"snapshot": 1}

    def test_set_workflow(self):
        s = WorkflowSession("test")
        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        s["current_workflow"] = wf
        assert s["current_workflow"] is wf

    def test_update_from_dict(self):
        s = WorkflowSession("test")
        s.update({"loaded_path": "/updated.json", "format": "api"})
        assert s["loaded_path"] == "/updated.json"
        assert s["format"] == "api"

    def test_update_from_session(self):
        s1 = WorkflowSession("a")
        s2 = WorkflowSession("b")
        s1["loaded_path"] = "/a.json"
        s2.update(s1)
        assert s2["loaded_path"] == "/a.json"

    def test_repr(self):
        s = WorkflowSession("mytest")
        r = repr(s)
        assert "mytest" in r
        assert "loaded_path" in r


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

class TestSessionRegistry:
    def test_get_session_creates(self):
        s = get_session("alpha")
        assert isinstance(s, WorkflowSession)
        assert s.session_id == "alpha"

    def test_get_session_same_id_returns_same(self):
        s1 = get_session("beta")
        s2 = get_session("beta")
        assert s1 is s2

    def test_get_session_different_ids_are_isolated(self):
        sa = get_session("a")
        sb = get_session("b")
        sa["loaded_path"] = "/path/a.json"
        sb["loaded_path"] = "/path/b.json"
        assert sa["loaded_path"] == "/path/a.json"
        assert sb["loaded_path"] == "/path/b.json"

    def test_default_session(self):
        s = get_session()
        assert s.session_id == "default"

    def test_clear_sessions(self):
        get_session("x")
        get_session("y")
        clear_sessions()
        # New session should be a fresh instance
        s = get_session("x")
        assert s["loaded_path"] is None

    def test_workflow_isolation(self):
        """Modifying workflow in one session doesn't affect another."""
        sa = get_session("a")
        sb = get_session("b")

        wf = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
        sa["current_workflow"] = copy.deepcopy(wf)
        sb["current_workflow"] = copy.deepcopy(wf)

        # Modify session a
        sa["current_workflow"]["1"]["inputs"]["seed"] = 99

        # Session b should be unchanged
        assert sb["current_workflow"]["1"]["inputs"]["seed"] == 42


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_get_session(self):
        """Multiple threads requesting the same session get the same instance."""
        results = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            s = get_session("shared")
            results.append(id(s))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1  # All same object

    def test_session_has_lock(self):
        s = get_session("locktest")
        assert isinstance(s._lock, type(threading.RLock()))

    def test_lock_is_acquirable(self):
        s = get_session("locktest2")
        acquired = s._lock.acquire(timeout=1.0)
        assert acquired
        s._lock.release()


# ---------------------------------------------------------------------------
# Concurrent access (lock-protected dict methods)
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    def test_concurrent_read_write(self):
        """10 threads doing mixed reads/writes — no exceptions, consistent state."""
        s = WorkflowSession("concurrent")
        barrier = threading.Barrier(10)
        errors = []

        def writer(idx):
            try:
                barrier.wait()
                for i in range(50):
                    s[f"key_{idx}_{i}"] = f"val_{idx}_{i}"
            except Exception as exc:
                errors.append(exc)

        def reader(idx):
            try:
                barrier.wait()
                for _ in range(50):
                    _ = s.get(f"key_{idx}_0", "missing")
                    _ = s.keys()
                    _ = "loaded_path" in s
            except Exception as exc:
                errors.append(exc)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent access raised: {errors}"
        # All writer keys present
        for i in range(5):
            for j in range(50):
                assert s[f"key_{i}_{j}"] == f"val_{i}_{j}"

    def test_concurrent_update(self):
        """Multiple threads calling update() — all updates applied."""
        s = WorkflowSession("upd")
        barrier = threading.Barrier(10)
        errors = []

        def updater(idx):
            try:
                barrier.wait()
                s.update({f"u_{idx}": idx})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for i in range(10):
            assert s[f"u_{i}"] == i

    def test_keys_returns_snapshot(self):
        """keys() returns a list (snapshot), not a live dict_keys view."""
        s = WorkflowSession("snap")
        k = s.keys()
        assert isinstance(k, list)

    def test_values_returns_snapshot(self):
        """values() returns a list (snapshot), not a live dict_values view."""
        s = WorkflowSession("snap")
        v = s.values()
        assert isinstance(v, list)

    def test_items_returns_snapshot(self):
        """items() returns a list (snapshot), not a live dict_items view."""
        s = WorkflowSession("snap")
        it = s.items()
        assert isinstance(it, list)


# ---------------------------------------------------------------------------
# Deep copy
# ---------------------------------------------------------------------------

class TestDeepCopy:
    def test_deepcopy_session_data(self):
        """Session data should be deep-copyable for save_session."""
        s = get_session("copytest")
        s["current_workflow"] = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
        s["loaded_path"] = "/test.json"
        s["history"].append({"old": True})

        # This is what save_session does
        snapshot = copy.deepcopy(dict(s.items()))
        assert snapshot["loaded_path"] == "/test.json"
        assert snapshot["current_workflow"]["1"]["inputs"]["seed"] == 42

        # Mutating snapshot shouldn't affect session
        snapshot["loaded_path"] = "/other.json"
        assert s["loaded_path"] == "/test.json"

    def test_deepcopy_session_object(self):
        """copy.deepcopy on a WorkflowSession returns a new session with copied data."""
        s = WorkflowSession("original")
        s["loaded_path"] = "/test.json"
        s["current_workflow"] = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}

        s2 = copy.deepcopy(s)
        assert s2.session_id == "original"
        assert s2["loaded_path"] == "/test.json"
        assert s2 is not s
        assert s2._data is not s._data

        # Mutating copy doesn't affect original
        s2["loaded_path"] = "/other.json"
        assert s["loaded_path"] == "/test.json"

    def test_deepcopy_then_update_pattern(self):
        """The pattern used in test fixtures: deepcopy then update to restore."""
        s = WorkflowSession("fixture")
        s["loaded_path"] = "/original.json"
        s["format"] = "api"

        # Save state (like test fixture setup)
        original = copy.deepcopy(s)

        # Modify (like test body)
        s["loaded_path"] = "/modified.json"
        s["format"] = "ui_with_api"

        # Restore (like test fixture teardown)
        s.update(original)
        assert s["loaded_path"] == "/original.json"
        assert s["format"] == "api"


# ---------------------------------------------------------------------------
# Session registry eviction (G3 — unbounded growth fix)
# ---------------------------------------------------------------------------


class TestSessionRegistryEviction:
    """_MAX_SESSIONS cap: oldest non-default sessions are evicted when full."""

    def test_default_session_never_evicted(self):
        """The 'default' session must survive even when the cap is hit."""
        default = get_session("default")
        default["loaded_path"] = "/kept.json"

        # Fill to cap with non-default sessions
        for i in range(_MAX_SESSIONS):
            get_session(f"conn_{i:04d}")

        # default must still be the same object with the same data
        still_default = get_session("default")
        assert still_default is default
        assert still_default["loaded_path"] == "/kept.json"

    def test_oldest_non_default_evicted_at_cap(self):
        """When the cap is hit, the oldest non-default session is removed."""
        # Fill registry to exactly _MAX_SESSIONS non-default sessions
        for i in range(_MAX_SESSIONS):
            get_session(f"conn_{i:04d}")

        # Capture a reference to conn_0000 (oldest non-default)
        oldest = get_session("conn_0000")  # re-fetch, no new entry

        # Adding one more triggers eviction of conn_0000 (oldest non-default)
        get_session("conn_extra")

        # conn_0000 was evicted; fetching it again creates a brand new object
        new_obj = get_session("conn_0000")
        assert new_obj is not oldest

    def test_existing_session_not_evicted(self):
        """get_session() for an already-registered ID never triggers eviction."""
        from agent.workflow_session import _sessions

        s = get_session("conn_kept")
        s["loaded_path"] = "/data.json"
        initial_count = len(_sessions)

        # Re-fetching the same ID must not grow the registry or evict anything
        s2 = get_session("conn_kept")
        assert s2 is s
        assert len(_sessions) == initial_count

    def test_registry_stays_at_or_below_cap(self):
        """After N >> _MAX_SESSIONS insertions, registry size stays bounded."""
        for i in range(_MAX_SESSIONS * 3):
            get_session(f"conn_{i:06d}")

        from agent.workflow_session import _sessions
        assert len(_sessions) <= _MAX_SESSIONS
