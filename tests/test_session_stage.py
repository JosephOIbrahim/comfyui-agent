"""Tests for CognitiveWorkflowStage integration with the session system.

Verifies that the stage is:
- Accessible via SessionContext
- Lazily initialized via ensure_stage()
- Saved alongside JSON session files as .usda
- Restored on session load
- Non-breaking when USD is unavailable
"""

import json

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core not installed")

from agent.session_context import SessionContext, SessionRegistry, get_session_context
from agent.stage import CognitiveWorkflowStage


class TestSessionContextStage:
    """Stage property on SessionContext."""

    def test_stage_is_none_by_default(self):
        ctx = SessionContext(session_id="test")
        assert ctx.stage is None

    def test_stage_setter(self):
        ctx = SessionContext(session_id="test")
        stage = CognitiveWorkflowStage()
        ctx.stage = stage
        assert ctx.stage is stage

    def test_ensure_stage_creates_stage(self):
        ctx = SessionContext(session_id="test")
        stage = ctx.ensure_stage()
        assert stage is not None
        assert isinstance(stage, CognitiveWorkflowStage)

    def test_ensure_stage_is_idempotent(self):
        ctx = SessionContext(session_id="test")
        stage1 = ctx.ensure_stage()
        stage2 = ctx.ensure_stage()
        assert stage1 is stage2

    def test_ensure_stage_does_not_replace_existing(self):
        ctx = SessionContext(session_id="test")
        manual_stage = CognitiveWorkflowStage()
        ctx.stage = manual_stage
        result = ctx.ensure_stage()
        assert result is manual_stage

    def test_stage_survives_touch(self):
        ctx = SessionContext(session_id="test")
        ctx.ensure_stage()
        ctx.touch()
        assert ctx.stage is not None

    def test_stage_accessible_from_registry(self):
        registry = SessionRegistry()
        ctx = registry.get_or_create("test_reg")
        stage = ctx.ensure_stage()
        assert stage is not None

        # Same session returns same stage
        ctx2 = registry.get_or_create("test_reg")
        assert ctx2.stage is stage
        registry.clear()

    def test_stage_with_data(self):
        ctx = SessionContext(session_id="test")
        stage = ctx.ensure_stage()
        stage.write("/workflows/test", "name", "session_stage_test")
        assert stage.read("/workflows/test", "name") == "session_stage_test"

    def test_different_sessions_get_different_stages(self):
        registry = SessionRegistry()
        ctx1 = registry.get_or_create("session_a")
        ctx2 = registry.get_or_create("session_b")
        ctx1.ensure_stage()
        ctx2.ensure_stage()
        assert ctx1.stage is not ctx2.stage
        registry.clear()


class TestStagePersistence:
    """Save and load stage alongside session JSON."""

    def test_save_stage_creates_usda(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        stage = CognitiveWorkflowStage()
        stage.write("/workflows/w1", "name", "persist_test")
        stage.write("/workflows/w1", "steps", 42)

        result = session_mod.save_stage("test_session", stage)
        assert "saved_stage" in result

        usda_path = tmp_path / "test_session.usda"
        assert usda_path.exists()

    def test_load_stage_from_usda(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        # Save a stage
        stage = CognitiveWorkflowStage()
        stage.write("/workflows/w1", "name", "roundtrip")
        stage.write("/workflows/w1", "steps", 99)
        session_mod.save_stage("test_rt", stage)

        # Load it back
        loaded = session_mod.load_stage("test_rt")
        assert loaded is not None
        assert loaded.read("/workflows/w1", "name") == "roundtrip"
        assert loaded.read("/workflows/w1", "steps") == 99

    def test_load_stage_returns_none_when_no_file(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        result = session_mod.load_stage("nonexistent")
        assert result is None

    def test_save_stage_with_agent_deltas(self, tmp_path, monkeypatch):
        """Saved stage should include composed (flattened) values."""
        from agent.memory import session as session_mod

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        stage = CognitiveWorkflowStage()
        stage.write("/workflows/w1", "steps", 20)
        stage.add_agent_delta("forge", {"/workflows/w1:steps": 50})

        session_mod.save_stage("delta_test", stage)
        loaded = session_mod.load_stage("delta_test")

        # Flattened stage should have the composed value (50)
        assert loaded.read("/workflows/w1", "steps") == 50

    def test_stage_and_json_coexist(self, tmp_path, monkeypatch):
        """Both .json and .usda files saved for the same session name."""
        from agent.memory import session as session_mod

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        # Save JSON session
        session_mod.save_session("coexist", notes=[])

        # Save stage
        stage = CognitiveWorkflowStage()
        stage.write("/workflows/w1", "val", 42)
        session_mod.save_stage("coexist", stage)

        # Both files exist
        assert (tmp_path / "coexist.json").exists()
        assert (tmp_path / "coexist.usda").exists()

        # JSON still loads correctly
        loaded_json = session_mod.load_session("coexist")
        assert "error" not in loaded_json

        # Stage still loads correctly
        loaded_stage = session_mod.load_stage("coexist")
        assert loaded_stage.read("/workflows/w1", "val") == 42


class TestSessionToolsIntegration:
    """Test that save_session/load_session tools wire through to stage."""

    def test_save_session_includes_stage(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod
        from agent.tools import session_tools
        from agent.session_context import _registry

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        # Set up a stage on the default session
        ctx = _registry.get_or_create("default")
        stage = ctx.ensure_stage()
        stage.write("/workflows/w1", "tool_test", "saved_via_tool")

        # Save via tool handler
        result_str = session_tools.handle("save_session", {"name": "tool_test"})
        result = json.loads(result_str)
        assert "error" not in result
        assert result.get("stage_saved") is True

        # Verify .usda was created
        assert (tmp_path / "tool_test.usda").exists()

        # Clean up
        ctx.stage = None

    def test_load_session_restores_stage(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod
        from agent.tools import session_tools
        from agent.session_context import _registry

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        # Create a session with stage data on disk
        session_mod.save_session("load_test", notes=[])
        stage = CognitiveWorkflowStage()
        stage.write("/workflows/w1", "restored", "yes")
        session_mod.save_stage("load_test", stage)

        # Clear any existing stage on default context
        ctx = _registry.get_or_create("default")
        ctx.stage = None

        # Load via tool handler
        result_str = session_tools.handle("load_session", {"name": "load_test"})
        result = json.loads(result_str)
        assert result.get("stage_restored") is True

        # Verify stage data was restored
        ctx = _registry.get_or_create("default")
        assert ctx.stage is not None
        assert ctx.stage.read("/workflows/w1", "restored") == "yes"

        # Clean up
        ctx.stage = None

    def test_load_session_without_stage_reports_false(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod
        from agent.tools import session_tools
        from agent.session_context import _registry

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        # Save JSON only (no .usda)
        session_mod.save_session("no_stage", notes=[])

        ctx = _registry.get_or_create("default")
        ctx.stage = None

        result_str = session_tools.handle("load_session", {"name": "no_stage"})
        result = json.loads(result_str)
        assert result.get("stage_restored") is False

        # Stage should still be None
        ctx = _registry.get_or_create("default")
        assert ctx.stage is None

    def test_save_session_without_stage_skips_usda(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod
        from agent.tools import session_tools
        from agent.session_context import _registry

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        # Ensure no stage on default
        ctx = _registry.get_or_create("default")
        ctx.stage = None

        result_str = session_tools.handle("save_session", {"name": "no_stage"})
        result = json.loads(result_str)
        assert "error" not in result
        assert result.get("stage_saved") is None  # Key absent
        assert not (tmp_path / "no_stage.usda").exists()


class TestWorkflowMapperInSession:
    """End-to-end: map a workflow into the session stage, save, reload."""

    def test_full_roundtrip(self, tmp_path, monkeypatch):
        from agent.memory import session as session_mod
        from agent.stage import workflow_json_to_prims, prims_to_workflow_json

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)

        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.0,
                    "model": ["2", 0],
                },
            },
            "2": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
        }

        # Map into stage
        stage = CognitiveWorkflowStage()
        workflow_json_to_prims(stage, workflow, "test")

        # Save
        session_mod.save_stage("mapper_test", stage)

        # Load
        loaded = session_mod.load_stage("mapper_test")
        assert loaded is not None

        # Reconstruct
        result = prims_to_workflow_json(loaded, "test")
        assert set(result.keys()) == {"1", "2"}
        assert result["1"]["class_type"] == "KSampler"
        assert result["1"]["inputs"]["steps"] == 20
        assert result["1"]["inputs"]["model"] == ["2", 0]
        assert result["2"]["inputs"]["ckpt_name"] == "model.safetensors"
