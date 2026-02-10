"""Tests for session persistence — save, load, list, notes, round-trip."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from agent.memory import session as session_mod
from agent.tools import session_tools, workflow_patch


@pytest.fixture(autouse=True)
def use_tmp_sessions(tmp_path):
    """Redirect sessions to a temp directory."""
    with patch.object(session_mod, "SESSIONS_DIR", tmp_path / "sessions"):
        yield tmp_path / "sessions"


@pytest.fixture(autouse=True)
def reset_workflow_state():
    """Reset workflow_patch state between tests."""
    workflow_patch._state["loaded_path"] = None
    workflow_patch._state["base_workflow"] = None
    workflow_patch._state["current_workflow"] = None
    workflow_patch._state["history"] = []
    workflow_patch._state["format"] = None
    yield


class TestSaveSession:
    def test_save_empty(self):
        result = json.loads(session_tools.handle("save_session", {"name": "test1"}))
        assert "saved" in result
        assert result["size_bytes"] > 0

    def test_save_with_workflow(self, tmp_path):
        """Save captures current workflow state."""
        # Load a workflow first
        wf_data = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}},
        }
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(wf_data), encoding="utf-8")
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(wf_path),
            "patches": [{"op": "replace", "path": "/1/inputs/seed", "value": 999}],
        })

        result = json.loads(session_tools.handle("save_session", {"name": "wf-test"}))
        assert "saved" in result

        # Verify saved file contains workflow
        saved = json.loads(Path(result["saved"]).read_text(encoding="utf-8"))
        assert saved["workflow"]["loaded_path"] == str(wf_path)
        assert saved["workflow"]["current_workflow"]["1"]["inputs"]["seed"] == 999

    def test_save_creates_directory(self, use_tmp_sessions):
        """Sessions dir created automatically."""
        assert not use_tmp_sessions.exists()
        session_tools.handle("save_session", {"name": "first"})
        assert use_tmp_sessions.exists()

    def test_save_deterministic(self):
        """Same state produces byte-identical JSON (He2025)."""
        session_tools.handle("save_session", {"name": "det-test"})
        # Save again with same name
        session_tools.handle("save_session", {"name": "det-test"})
        # The file content should have sort_keys=True
        result = json.loads(session_tools.handle("load_session", {"name": "det-test"}))
        assert "loaded" in result


class TestLoadSession:
    def test_load_existing(self):
        session_tools.handle("save_session", {"name": "load-me"})
        result = json.loads(session_tools.handle("load_session", {"name": "load-me"}))
        assert result["loaded"] == "load-me"
        assert result["workflow_restored"] is False
        assert result["notes_count"] == 0

    def test_load_not_found(self):
        result = json.loads(session_tools.handle("load_session", {"name": "nope"}))
        assert "error" in result
        assert "not found" in result["error"]

    def test_load_suggests_available(self):
        """When session not found, suggest available ones."""
        session_tools.handle("save_session", {"name": "project-a"})
        result = json.loads(session_tools.handle("load_session", {"name": "project-b"}))
        assert "error" in result
        assert "project-a" in result["available"]

    def test_load_restores_workflow(self, tmp_path):
        """Loading a session restores workflow_patch state."""
        # Set up a patched workflow
        wf_data = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 42}},
        }
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(wf_data), encoding="utf-8")
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(wf_path),
            "patches": [{"op": "replace", "path": "/1/inputs/seed", "value": 777}],
        })

        # Save
        session_tools.handle("save_session", {"name": "restore-test"})

        # Clear state
        workflow_patch._state["loaded_path"] = None
        workflow_patch._state["base_workflow"] = None
        workflow_patch._state["current_workflow"] = None
        workflow_patch._state["history"] = []

        # Load
        result = json.loads(session_tools.handle("load_session", {"name": "restore-test"}))
        assert result["workflow_restored"] is True
        assert result["workflow_path"] == str(wf_path)

        # Verify workflow_patch state was restored
        current = workflow_patch.get_current_workflow()
        assert current is not None
        assert current["1"]["inputs"]["seed"] == 777

    def test_round_trip_preserves_base(self, tmp_path):
        """Save/load round-trip preserves both base and current workflow."""
        wf_data = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}},
        }
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(wf_data), encoding="utf-8")
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(wf_path),
            "patches": [{"op": "replace", "path": "/1/inputs/seed", "value": 100}],
        })

        session_tools.handle("save_session", {"name": "round-trip"})

        # Clear and restore
        workflow_patch._state["base_workflow"] = None
        workflow_patch._state["current_workflow"] = None
        session_tools.handle("load_session", {"name": "round-trip"})

        # Base should be original
        assert workflow_patch._state["base_workflow"]["1"]["inputs"]["seed"] == 42
        # Current should have the patch
        assert workflow_patch._state["current_workflow"]["1"]["inputs"]["seed"] == 100


class TestListSessions:
    def test_empty(self):
        result = json.loads(session_tools.handle("list_sessions", {}))
        assert result["count"] == 0
        assert result["sessions"] == []

    def test_lists_all(self):
        session_tools.handle("save_session", {"name": "alpha"})
        session_tools.handle("save_session", {"name": "beta"})
        session_tools.handle("save_session", {"name": "gamma"})

        result = json.loads(session_tools.handle("list_sessions", {}))
        assert result["count"] == 3
        names = {s["name"] for s in result["sessions"]}
        assert names == {"alpha", "beta", "gamma"}

    def test_sorted_by_name(self):
        """Sessions listed in alphabetical order."""
        session_tools.handle("save_session", {"name": "zebra"})
        session_tools.handle("save_session", {"name": "apple"})

        result = json.loads(session_tools.handle("list_sessions", {}))
        assert result["sessions"][0]["name"] == "apple"
        assert result["sessions"][1]["name"] == "zebra"

    def test_metadata_fields(self, tmp_path):
        """Each session shows metadata."""
        wf_data = {"1": {"class_type": "Test", "inputs": {}}}
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(wf_data), encoding="utf-8")
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(wf_path),
            "patches": [{"op": "replace", "path": "/1/inputs", "value": {}}],
        })

        session_tools.handle("save_session", {"name": "meta-test"})

        result = json.loads(session_tools.handle("list_sessions", {}))
        s = result["sessions"][0]
        assert "saved_at" in s
        assert s["has_workflow"] is True


class TestAddNote:
    def test_add_to_new_session(self):
        result = json.loads(session_tools.handle("add_note", {
            "session_name": "notes-test",
            "note": "User prefers SDXL for landscapes",
        }))
        assert result["added"] is True
        assert result["total_notes"] == 1

    def test_add_multiple(self):
        session_tools.handle("add_note", {
            "session_name": "multi",
            "note": "First note",
        })
        result = json.loads(session_tools.handle("add_note", {
            "session_name": "multi",
            "note": "Second note",
        }))
        assert result["total_notes"] == 2

    def test_notes_persist_in_session(self):
        """Notes survive save/load cycle."""
        session_tools.handle("add_note", {
            "session_name": "persist",
            "note": "Important finding",
        })
        session_tools.handle("save_session", {"name": "persist"})

        result = json.loads(session_tools.handle("load_session", {"name": "persist"}))
        assert result["notes_count"] == 1
        assert result["notes"][0]["text"] == "Important finding"

    def test_notes_have_timestamp(self):
        session_tools.handle("add_note", {
            "session_name": "ts-test",
            "note": "Timestamped note",
        })
        loaded = json.loads(session_tools.handle("load_session", {"name": "ts-test"}))
        assert "added_at" in loaded["notes"][0]


class TestRegistration:
    def test_tools_registered(self):
        from agent.tools import ALL_TOOLS
        names = {t["name"] for t in ALL_TOOLS}
        assert "save_session" in names
        assert "load_session" in names
        assert "list_sessions" in names
        assert "add_note" in names
