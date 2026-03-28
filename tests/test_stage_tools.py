"""Tests for agent/stage/stage_tools.py — all mocked, no real I/O."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.stage.stage_tools import TOOLS, handle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: str) -> dict:
    return json.loads(result)


def _make_stage(
    read_return=None,
    delta_count: int = 0,
    list_deltas_return: list | None = None,
    reconstruct_return: dict | None = None,
):
    """Create a mock CognitiveWorkflowStage."""
    m = MagicMock()
    m.read.return_value = read_return
    m.delta_count = delta_count
    m.list_deltas.return_value = list_deltas_return or []
    m.reconstruct_clean.return_value = reconstruct_return or {}
    m.add_agent_delta.return_value = "anon:forge_delta.usda"
    m.rollback_to.return_value = 0
    return m


PATCH_GET_STAGE = "agent.stage.stage_tools._get_stage"


# ---------------------------------------------------------------------------
# TOOLS schema
# ---------------------------------------------------------------------------

class TestToolsSchema:
    EXPECTED_NAMES = {
        "stage_read",
        "stage_write",
        "stage_add_delta",
        "stage_rollback",
        "stage_reconstruct_clean",
        "stage_list_deltas",
    }

    def test_tools_is_list(self):
        assert isinstance(TOOLS, list)

    def test_tool_count(self):
        assert len(TOOLS) == 6

    def test_tool_names(self):
        assert {t["name"] for t in TOOLS} == self.EXPECTED_NAMES

    def test_each_tool_has_description(self):
        for t in TOOLS:
            assert isinstance(t["description"], str) and t["description"]

    def test_each_tool_has_input_schema(self):
        for t in TOOLS:
            assert "input_schema" in t
            assert t["input_schema"]["type"] == "object"

    def test_stage_read_required(self):
        schema = next(t for t in TOOLS if t["name"] == "stage_read")
        assert "prim_path" in schema["input_schema"]["required"]

    def test_stage_write_required(self):
        schema = next(t for t in TOOLS if t["name"] == "stage_write")
        required = set(schema["input_schema"]["required"])
        assert {"prim_path", "attr_name", "value"} <= required

    def test_stage_add_delta_required(self):
        schema = next(t for t in TOOLS if t["name"] == "stage_add_delta")
        required = set(schema["input_schema"]["required"])
        assert {"agent_name", "delta"} <= required

    def test_stage_rollback_required(self):
        schema = next(t for t in TOOLS if t["name"] == "stage_rollback")
        assert "n_deltas" in schema["input_schema"]["required"]

    def test_stage_reconstruct_required_empty(self):
        schema = next(t for t in TOOLS if t["name"] == "stage_reconstruct_clean")
        assert schema["input_schema"]["required"] == []

    def test_stage_list_deltas_required_empty(self):
        schema = next(t for t in TOOLS if t["name"] == "stage_list_deltas")
        assert schema["input_schema"]["required"] == []


# ---------------------------------------------------------------------------
# No stage available
# ---------------------------------------------------------------------------

class TestNoStage:
    """All tools return error JSON when the stage is unavailable."""

    @pytest.fixture(autouse=True)
    def no_stage(self):
        with patch(PATCH_GET_STAGE, return_value=None):
            yield

    def test_stage_read_no_stage(self):
        r = _parse(handle("stage_read", {"prim_path": "/workflows/w1"}))
        assert "error" in r

    def test_stage_write_no_stage(self):
        r = _parse(handle("stage_write", {"prim_path": "/a", "attr_name": "x", "value": 1}))
        assert "error" in r

    def test_stage_add_delta_no_stage(self):
        r = _parse(handle("stage_add_delta", {"agent_name": "forge", "delta": {}}))
        assert "error" in r

    def test_stage_rollback_no_stage(self):
        r = _parse(handle("stage_rollback", {"n_deltas": 1}))
        assert "error" in r

    def test_stage_reconstruct_clean_no_stage(self):
        r = _parse(handle("stage_reconstruct_clean", {}))
        assert "error" in r

    def test_stage_list_deltas_no_stage(self):
        r = _parse(handle("stage_list_deltas", {}))
        assert "error" in r


# ---------------------------------------------------------------------------
# stage_read
# ---------------------------------------------------------------------------

class TestStageRead:
    def test_read_existing_attr(self):
        stage = _make_stage(read_return=42)
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_read", {"prim_path": "/w/n", "attr_name": "steps"}))
        assert r["value"] == 42
        assert r["prim_path"] == "/w/n"
        assert r["attr_name"] == "steps"

    def test_read_prim_existence(self):
        stage = _make_stage(read_return=True)
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_read", {"prim_path": "/workflows/w1"}))
        assert r["value"] is True
        assert r["attr_name"] is None

    def test_read_missing_prim_returns_none(self):
        stage = _make_stage(read_return=None)
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_read", {"prim_path": "/missing"}))
        assert r["value"] is None

    def test_read_passes_prim_path_and_attr(self):
        stage = _make_stage(read_return="hello")
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_read", {"prim_path": "/p", "attr_name": "name"})
        stage.read.assert_called_once_with("/p", "name")

    def test_read_no_attr_passes_none(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_read", {"prim_path": "/p"})
        stage.read.assert_called_once_with("/p", None)

    def test_read_float_value(self):
        stage = _make_stage(read_return=3.14)
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_read", {"prim_path": "/p", "attr_name": "cfg"}))
        assert abs(r["value"] - 3.14) < 1e-9

    def test_read_string_value(self):
        stage = _make_stage(read_return="euler_a")
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_read", {"prim_path": "/p", "attr_name": "sampler"}))
        assert r["value"] == "euler_a"

    def test_read_bool_false(self):
        stage = _make_stage(read_return=False)
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_read", {"prim_path": "/p", "attr_name": "flag"}))
        assert r["value"] is False


# ---------------------------------------------------------------------------
# stage_write
# ---------------------------------------------------------------------------

class TestStageWrite:
    def test_write_success(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_write", {
                "prim_path": "/w/n", "attr_name": "steps", "value": 30
            }))
        assert r["written"] is True
        assert r["value"] == 30

    def test_write_calls_stage_write(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_write", {"prim_path": "/p", "attr_name": "x", "value": 5})
        stage.write.assert_called_once_with("/p", "x", 5, node_type=None)

    def test_write_passes_node_type(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_write", {
                "prim_path": "/p", "attr_name": "x", "value": 5, "node_type": "KSampler"
            })
        stage.write.assert_called_once_with("/p", "x", 5, node_type="KSampler")

    def test_write_error_from_stage(self):
        stage = _make_stage()
        stage.write.side_effect = Exception("anchor violation")
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_write", {
                "prim_path": "/p", "attr_name": "x", "value": 1
            }))
        assert "error" in r
        assert "anchor violation" in r["error"]

    def test_write_string_value(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_write", {
                "prim_path": "/p", "attr_name": "sampler", "value": "euler_a"
            }))
        assert r["written"] is True

    def test_write_bool_value(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_write", {
                "prim_path": "/p", "attr_name": "tiling", "value": True
            }))
        assert r["written"] is True

    def test_write_returns_prim_and_attr(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_write", {
                "prim_path": "/w/n", "attr_name": "steps", "value": 20
            }))
        assert r["prim_path"] == "/w/n"
        assert r["attr_name"] == "steps"


# ---------------------------------------------------------------------------
# stage_add_delta
# ---------------------------------------------------------------------------

class TestStageAddDelta:
    def test_add_delta_success(self):
        stage = _make_stage(delta_count=1)
        stage.add_agent_delta.return_value = "anon:forge_delta.usda"
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_add_delta", {
                "agent_name": "forge",
                "delta": {"/w/n:steps": 30},
            }))
        assert r["agent_name"] == "forge"
        assert r["layer_id"] == "anon:forge_delta.usda"
        assert r["keys_applied"] == 1

    def test_add_delta_calls_add_agent_delta(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_add_delta", {
                "agent_name": "scout",
                "delta": {"/p:x": 5, "/p:y": 10},
            })
        stage.add_agent_delta.assert_called_once_with("scout", {"/p:x": 5, "/p:y": 10})

    def test_add_delta_error_from_stage(self):
        stage = _make_stage()
        stage.add_agent_delta.side_effect = Exception("bad delta key")
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_add_delta", {
                "agent_name": "forge", "delta": {"bad_key": 1}
            }))
        assert "error" in r
        assert "bad delta key" in r["error"]

    def test_add_delta_empty_delta(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_add_delta", {"agent_name": "forge", "delta": {}}))
        assert r["keys_applied"] == 0

    def test_add_delta_multiple_keys(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_add_delta", {
                "agent_name": "forge",
                "delta": {"/a:x": 1, "/b:y": 2, "/c:z": 3},
            }))
        assert r["keys_applied"] == 3

    def test_add_delta_reports_delta_count(self):
        stage = _make_stage()
        stage.delta_count = 3
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_add_delta", {"agent_name": "f", "delta": {}}))
        assert r["delta_count"] == 3


# ---------------------------------------------------------------------------
# stage_rollback
# ---------------------------------------------------------------------------

class TestStageRollback:
    def test_rollback_success(self):
        stage = _make_stage(delta_count=1)
        stage.rollback_to.return_value = 2
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_rollback", {"n_deltas": 2}))
        assert r["removed"] == 2
        assert r["requested"] == 2

    def test_rollback_calls_rollback_to(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_rollback", {"n_deltas": 3})
        stage.rollback_to.assert_called_once_with(3)

    def test_rollback_fewer_than_requested(self):
        stage = _make_stage(delta_count=0)
        stage.rollback_to.return_value = 1
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_rollback", {"n_deltas": 5}))
        assert r["removed"] == 1
        assert r["requested"] == 5

    def test_rollback_zero_deltas(self):
        stage = _make_stage(delta_count=0)
        stage.rollback_to.return_value = 0
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_rollback", {"n_deltas": 1}))
        assert r["removed"] == 0
        assert r["remaining_deltas"] == 0

    def test_rollback_reports_remaining(self):
        stage = _make_stage()
        stage.delta_count = 2
        stage.rollback_to.return_value = 1
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_rollback", {"n_deltas": 1}))
        assert r["remaining_deltas"] == 2


# ---------------------------------------------------------------------------
# stage_reconstruct_clean
# ---------------------------------------------------------------------------

class TestStageReconstructClean:
    def test_empty_stage(self):
        stage = _make_stage(reconstruct_return={})
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_reconstruct_clean", {}))
        assert r["prim_count"] == 0
        assert r["prims"] == {}

    def test_with_prims(self):
        stage = _make_stage(reconstruct_return={
            "/workflows/w1": {"steps": 20, "cfg": 7.0},
            "/agents/forge": {"role": "optimizer"},
        })
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_reconstruct_clean", {}))
        assert r["prim_count"] == 2
        assert r["prims"]["/workflows/w1"]["steps"] == 20
        assert r["prims"]["/agents/forge"]["role"] == "optimizer"

    def test_calls_reconstruct_clean(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_reconstruct_clean", {})
        stage.reconstruct_clean.assert_called_once()

    def test_reconstruct_error(self):
        stage = _make_stage()
        stage.reconstruct_clean.side_effect = Exception("stage error")
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_reconstruct_clean", {}))
        assert "error" in r

    def test_string_values_preserved(self):
        stage = _make_stage(reconstruct_return={"/p": {"name": "test_workflow"}})
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_reconstruct_clean", {}))
        assert r["prims"]["/p"]["name"] == "test_workflow"

    def test_bool_values_preserved(self):
        stage = _make_stage(reconstruct_return={"/p": {"active": True}})
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_reconstruct_clean", {}))
        assert r["prims"]["/p"]["active"] is True

    def test_none_values_preserved(self):
        stage = _make_stage(reconstruct_return={"/p": {"x": None}})
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_reconstruct_clean", {}))
        assert r["prims"]["/p"]["x"] is None

    def test_float_values_preserved(self):
        stage = _make_stage(reconstruct_return={"/p": {"cfg": 7.5}})
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_reconstruct_clean", {}))
        assert abs(r["prims"]["/p"]["cfg"] - 7.5) < 1e-9


# ---------------------------------------------------------------------------
# stage_list_deltas
# ---------------------------------------------------------------------------

class TestStageListDeltas:
    def test_no_deltas(self):
        stage = _make_stage(list_deltas_return=[])
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_list_deltas", {}))
        assert r["delta_count"] == 0
        assert r["deltas"] == []

    def test_with_deltas(self):
        ids = ["anon:a.usda", "anon:b.usda", "anon:c.usda"]
        stage = _make_stage(list_deltas_return=ids)
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_list_deltas", {}))
        assert r["delta_count"] == 3
        assert r["deltas"] == ids

    def test_calls_list_deltas(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            handle("stage_list_deltas", {})
        stage.list_deltas.assert_called_once()

    def test_single_delta(self):
        stage = _make_stage(list_deltas_return=["anon:forge_delta.usda"])
        with patch(PATCH_GET_STAGE, return_value=stage):
            r = _parse(handle("stage_list_deltas", {}))
        assert r["delta_count"] == 1
        assert r["deltas"][0] == "anon:forge_delta.usda"


# ---------------------------------------------------------------------------
# Dispatch edge cases
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unknown_tool(self):
        r = _parse(handle("nonexistent_tool", {}))
        assert "error" in r
        assert "nonexistent_tool" in r["error"] or "Unknown" in r["error"]

    def test_handle_returns_string(self):
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            result = handle("stage_list_deltas", {})
        assert isinstance(result, str)

    def test_top_level_exception_caught(self):
        """handle() must never raise — exceptions become error JSON."""
        with patch(PATCH_GET_STAGE, side_effect=RuntimeError("boom")):
            r = _parse(handle("stage_read", {"prim_path": "/p"}))
        assert "error" in r

    def test_all_tool_names_dispatch(self):
        """Every tool name in TOOLS must be handled without 'Unknown tool' error."""
        stage = _make_stage()
        with patch(PATCH_GET_STAGE, return_value=stage):
            for tool in TOOLS:
                name = tool["name"]
                # Build minimal valid input from required fields
                inp: dict = {}
                for req in tool["input_schema"].get("required", []):
                    props = tool["input_schema"]["properties"]
                    if req == "n_deltas":
                        inp[req] = 1
                    elif req == "delta":
                        inp[req] = {}
                    elif req == "value":
                        inp[req] = 0
                    else:
                        inp[req] = "test"
                r = _parse(handle(name, inp))
                assert "Unknown tool" not in r.get("error", ""), (
                    f"Tool '{name}' was not dispatched"
                )
