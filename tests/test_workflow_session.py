"""Tests for WorkflowSession — dict-like access, registry isolation, thread safety."""

import copy
import threading

import pytest

from agent.workflow_session import WorkflowSession, get_session, clear_sessions


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
        assert set(s.keys()) == {"loaded_path", "base_workflow", "current_workflow", "history", "format"}

    def test_values(self):
        s = WorkflowSession("test")
        vals = list(s.values())
        # Default: None, None, None, [], None
        assert len(vals) == 5

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
        assert isinstance(s._lock, type(threading.Lock()))

    def test_lock_is_acquirable(self):
        s = get_session("locktest2")
        acquired = s._lock.acquire(timeout=1.0)
        assert acquired
        s._lock.release()


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
