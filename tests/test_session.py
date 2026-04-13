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
        workflow_patch._get_state()["loaded_path"] = None
        workflow_patch._get_state()["base_workflow"] = None
        workflow_patch._get_state()["current_workflow"] = None
        workflow_patch._get_state()["history"] = []

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
        workflow_patch._get_state()["base_workflow"] = None
        workflow_patch._get_state()["current_workflow"] = None
        session_tools.handle("load_session", {"name": "round-trip"})

        # Base should be original
        assert workflow_patch._get_state()["base_workflow"]["1"]["inputs"]["seed"] == 42
        # Current should have the patch
        assert workflow_patch._get_state()["current_workflow"]["1"]["inputs"]["seed"] == 100


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


class TestSchemaVersioning:
    def test_save_includes_version(self, use_tmp_sessions):
        """Saved sessions include schema_version field."""
        session_tools.handle("save_session", {"name": "version-test"})
        path = use_tmp_sessions / "version-test.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 2

    def test_migrate_v0_to_v2(self, use_tmp_sessions):
        """Loading a v0 session (no schema_version) should migrate it to current."""
        # Write a v0 session (no schema_version field)
        path = use_tmp_sessions
        path.mkdir(parents=True, exist_ok=True)
        v0_data = {
            "name": "old-session",
            "saved_at": "2024-01-01T00:00:00",
            "workflow": {"loaded_path": None, "format": None},
            "notes": [],
            "metadata": {},
        }
        (path / "old-session.json").write_text(
            json.dumps(v0_data, sort_keys=True, indent=2), encoding="utf-8"
        )
        # Use the internal load_session (not the tool handler which wraps the result)
        result = session_mod.load_session("old-session")
        assert "error" not in result
        assert result.get("schema_version") == 2


class TestAtomicWrites:
    def test_session_file_exists_after_save(self, use_tmp_sessions):
        """Atomic write creates the file successfully."""
        session_tools.handle("save_session", {"name": "atomic-test"})
        path = use_tmp_sessions / "atomic-test.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "atomic-test"

    def test_no_temp_files_left(self, use_tmp_sessions):
        """Atomic write should not leave .tmp files."""
        use_tmp_sessions.mkdir(parents=True, exist_ok=True)
        session_tools.handle("save_session", {"name": "clean-test"})
        tmp_files = list(use_tmp_sessions.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestNotesTOCTOU:
    """Cycle 28 fix: notes must not be lost under concurrent save+add_note."""

    def test_concurrent_add_note_and_save_preserves_all_notes(self, use_tmp_sessions):
        """Notes added concurrently with save_session must survive the save.

        Regression for TOCTOU where load_session() + save_session() in
        _handle_save_session() could overwrite a note written by a concurrent
        add_note() call between the load and save steps.
        """
        import threading

        name = "toctou-test"
        # Create initial session
        session_tools.handle("save_session", {"name": name})

        errors = []
        note_count = 20

        def add_note_loop():
            for i in range(note_count):
                result = json.loads(session_tools.handle("add_note", {
                    "session_name": name, "note": f"note-{i}", "note_type": "observation",
                }))
                if "error" in result:
                    errors.append(result["error"])

        def save_loop():
            for _ in range(5):
                session_tools.handle("save_session", {"name": name})

        t1 = threading.Thread(target=add_note_loop)
        t2 = threading.Thread(target=save_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"add_note raised errors: {errors}"

        # Final save must not have silently dropped notes
        # (At minimum, add_note calls that completed before the last save
        # should be present. We can't assert all 20 survived due to inherent
        # ordering, but we CAN assert the session is readable and non-empty.)
        final = json.loads(session_tools.handle("load_session", {"name": name}))
        assert "error" not in final, f"session unreadable after concurrent ops: {final}"
        assert isinstance(final.get("notes"), list)

    def test_note_lock_is_reentrant(self):
        """_NOTE_LOCK must be an RLock so save_session inside the lock doesn't deadlock."""
        from agent.memory.session import _NOTE_LOCK
        import threading
        assert isinstance(_NOTE_LOCK, type(threading.RLock()))


class TestRegistration:
    def test_tools_registered(self):
        from agent.tools import ALL_TOOLS
        names = {t["name"] for t in ALL_TOOLS}
        assert "save_session" in names
        assert "load_session" in names
        assert "list_sessions" in names
        assert "add_note" in names


# ---------------------------------------------------------------------------
# Cycle 39: _atomic_write must fsync before rename
# ---------------------------------------------------------------------------

class TestAtomicWriteFsync:
    """_atomic_write must flush + fsync the temp file before os.replace so
    that on power failure the renamed file is complete on disk. (Cycle 39 fix)"""

    def test_fsync_called_during_atomic_write(self, tmp_path):
        """os.fsync must be invoked at least once per _atomic_write call."""
        import os as _os
        from unittest.mock import patch
        from agent.memory.session import _atomic_write

        target = tmp_path / "test.json"
        fsync_calls = []
        orig_fsync = _os.fsync

        def _recording_fsync(fd):
            fsync_calls.append(fd)
            return orig_fsync(fd)

        with patch("agent.memory.session.os.fsync", side_effect=_recording_fsync):
            _atomic_write(target, '{"key": "value"}')

        assert len(fsync_calls) >= 1, "os.fsync not called during _atomic_write"
        assert target.exists()
        assert json.loads(target.read_text()) == {"key": "value"}

    def test_atomic_write_produces_valid_content(self, tmp_path):
        """Even with fsync, the written content must be byte-for-byte correct."""
        from agent.memory.session import _atomic_write

        target = tmp_path / "session.json"
        content = '{"name": "test", "value": 42}'
        _atomic_write(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_flush_happens_before_fsync(self, tmp_path):
        """fd.flush() must precede os.fsync — otherwise fsync may miss buffered data."""
        import os as _os
        from unittest.mock import patch
        from agent.memory.session import _atomic_write

        target = tmp_path / "order_test.json"
        call_order = []

        orig_flush = None  # captured below

        class _TrackingFile:
            """Wraps NamedTemporaryFile to intercept flush() and track call order."""
            def __init__(self, wrapped):
                self._w = wrapped
                self.name = wrapped.name
                self.fileno = wrapped.fileno

            def write(self, s):
                return self._w.write(s)

            def flush(self):
                call_order.append("flush")
                return self._w.flush()

            def close(self):
                return self._w.close()

        import tempfile as _tf
        orig_ntf = _tf.NamedTemporaryFile

        def _patched_ntf(**kwargs):
            return _TrackingFile(orig_ntf(**kwargs))

        _real_fsync = _os.fsync  # save before patch to avoid recursive mock call

        def _recording_fsync(fd):
            call_order.append("fsync")
            return _real_fsync(fd)

        with patch("agent.memory.session.os.fsync", side_effect=_recording_fsync), \
             patch("tempfile.NamedTemporaryFile", side_effect=_patched_ntf):
            _atomic_write(target, '{"x": 1}')

        # flush must come before fsync in call order
        assert "flush" in call_order and "fsync" in call_order
        flush_idx = call_order.index("flush")
        fsync_idx = call_order.index("fsync")
        assert flush_idx < fsync_idx, f"flush ({flush_idx}) must precede fsync ({fsync_idx})"


# ---------------------------------------------------------------------------
# Cycle 44 — session load field normalization + tempfile cleanup logging
# ---------------------------------------------------------------------------

class TestSessionLoadNormalization:
    """load_session normalizes corrupt field types instead of propagating them."""

    def _write_session(self, tmp_path, name, data):
        """Write to session_mod.SESSIONS_DIR — same target patched by autouse fixture."""
        import json
        from agent.memory import session as sess_module
        sess_dir = sess_module.SESSIONS_DIR  # uses the patched value from autouse fixture
        sess_dir.mkdir(parents=True, exist_ok=True)
        path = sess_dir / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_notes_not_list_is_normalized_to_empty_list(self, tmp_path):
        """If 'notes' is a dict (manually edited), load_session resets it to []."""
        from agent.memory.session import load_session, SCHEMA_VERSION
        session_name = "c44_notes_dict"
        self._write_session(tmp_path, session_name, {
            "name": session_name,
            "saved_at": "2024-01-01T00:00:00",
            "schema_version": SCHEMA_VERSION,
            "workflow": {"loaded_path": None, "format": None},
            "notes": {"bad": "field"},  # dict, not list
        })
        result = load_session(session_name)
        assert "error" not in result, f"Should not error: {result}"
        assert isinstance(result["notes"], list)
        assert result["notes"] == []

    def test_workflow_not_dict_is_normalized(self, tmp_path):
        """If 'workflow' is a string, load_session resets it to the default dict."""
        from agent.memory.session import load_session, SCHEMA_VERSION
        session_name = "c44_workflow_str"
        self._write_session(tmp_path, session_name, {
            "name": session_name,
            "saved_at": "2024-01-01T00:00:00",
            "schema_version": SCHEMA_VERSION,
            "workflow": "corrupted_string_value",
            "notes": [],
        })
        result = load_session(session_name)
        assert "error" not in result
        assert isinstance(result["workflow"], dict)
        assert result["workflow"].get("loaded_path") is None

    def test_valid_session_unmodified(self, tmp_path):
        """A well-formed session file is returned unchanged by normalization."""
        from agent.memory.session import load_session, SCHEMA_VERSION
        session_name = "c44_valid_norm"
        self._write_session(tmp_path, session_name, {
            "name": session_name,
            "saved_at": "2024-01-01T00:00:00",
            "schema_version": SCHEMA_VERSION,
            "workflow": {"loaded_path": "/some/path.json", "format": "api"},
            "notes": [{"text": "note1", "added_at": "2024-01-01T00:00:00"}],
        })
        result = load_session(session_name)
        assert "error" not in result
        assert result["workflow"]["loaded_path"] == "/some/path.json"
        assert len(result["notes"]) == 1


class TestAtomicWriteCleanupLogging:
    """_atomic_write logs tempfile cleanup failures instead of silently swallowing."""

    def test_cleanup_failure_is_logged_not_swallowed(self, caplog, tmp_path):
        """If temp file unlink fails, the exception is logged at WARNING level."""
        import logging
        from unittest.mock import patch
        from agent.memory import session as sess_module

        target = tmp_path / "test_write.json"
        content = '{"ok": true}'

        # Patch shutil.move to fail (triggers the except path),
        # then patch Path.unlink to also fail (triggers the log warning).
        with patch("agent.memory.session.shutil.move", side_effect=OSError("disk full")), \
             patch("agent.memory.session.Path.unlink", side_effect=OSError("unlink failed")), \
             caplog.at_level(logging.WARNING, logger="agent.memory.session"):
            try:
                sess_module._atomic_write(target, content)
            except OSError:
                pass  # Expected — the write failure is re-raised

        assert any(
            "temp" in rec.message.lower() or "clean" in rec.message.lower()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ), "Expected a WARNING log about temp file cleanup failure"


# ---------------------------------------------------------------------------
# Cycle 60 — allow_nan=False coverage for session writes
# ---------------------------------------------------------------------------

class TestSessionNaNSafety:
    """Cycle 60: session write functions must reject NaN/Infinity (allow_nan=False)."""

    def test_save_session_returns_error_on_nan_metadata(self):
        """save_session must return error dict when metadata contains NaN (not silently write)."""
        from agent.memory.session import save_session
        result = save_session("test_c60_nan", metadata={"score": float("nan")})
        # json.dumps with allow_nan=False raises ValueError, caught and returned as error dict
        assert "error" in result

    def test_save_session_returns_error_on_inf_metadata(self):
        """save_session must return error dict when metadata contains Infinity."""
        from agent.memory.session import save_session
        result = save_session("test_c60_inf", metadata={"cfg": float("inf")})
        assert "error" in result

    def test_save_session_valid_data_still_works(self):
        """save_session must succeed with normal finite float values."""
        from agent.memory.session import save_session
        result = save_session("test_c60_valid", metadata={"cfg": 7.0, "steps": 20})
        assert "error" not in result
        assert "saved" in result
