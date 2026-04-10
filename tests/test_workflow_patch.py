"""Tests for workflow_patch tool — patching, undo, diff, save."""

import json
import pytest
from agent.tools import workflow_patch
from agent.workflow_session import clear_sessions


@pytest.fixture
def sample_workflow(tmp_path):
    """Create a simple API-format workflow file."""
    data = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd15.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a beautiful sunset", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "denoise": 1.0,
            },
        },
    }
    path = tmp_path / "test_wf.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module state between tests."""
    s = workflow_patch._get_state()
    s["loaded_path"] = None
    s["base_workflow"] = None
    s["current_workflow"] = None
    s["history"] = []
    s["format"] = None
    yield


class TestApplyPatch:
    def test_load_and_patch(self, sample_workflow):
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [
                {"op": "replace", "path": "/2/inputs/text", "value": "a cat on a roof"},
            ],
        }))
        assert result["applied"] == 1
        assert result["changes"][0]["before"] == "a beautiful sunset"
        assert result["changes"][0]["after"] == "a cat on a roof"

    def test_multiple_patches(self, sample_workflow):
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [
                {"op": "replace", "path": "/2/inputs/text", "value": "new prompt"},
                {"op": "replace", "path": "/3/inputs/seed", "value": 999},
                {"op": "replace", "path": "/3/inputs/steps", "value": 50},
            ],
        }))
        assert result["applied"] == 3
        assert result["total_changes_from_base"] == 3

    def test_patch_without_load_fails(self):
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "path": "/1/inputs/text", "value": "x"}],
        }))
        assert "error" in result

    def test_sequential_patches(self, sample_workflow):
        # First patch
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 100}],
        })
        # Second patch (no path needed, reuses loaded)
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "path": "/3/inputs/steps", "value": 50}],
        }))
        assert result["applied"] == 1
        assert result["total_changes_from_base"] == 2

    def test_invalid_patch_rolls_back(self, sample_workflow):
        # Load workflow
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 1}],
        })
        # Try invalid patch (nonexistent path with 'test' op)
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "test", "path": "/99/inputs/fake", "value": "nope"}],
        }))
        assert "error" in result
        # Seed should still be 1 (rolled back)
        current = workflow_patch.get_current_workflow()
        assert current["3"]["inputs"]["seed"] == 1

    def test_file_not_found(self):
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "path": "/nonexistent/wf.json",
            "patches": [{"op": "replace", "path": "/1/inputs/x", "value": 1}],
        }))
        assert "error" in result

    def test_ui_only_rejected(self, tmp_path):
        data = {
            "nodes": [{"id": 1, "type": "Test"}],
            "extra": {"ds": {"scale": 1.0}},
        }
        path = tmp_path / "ui_only.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "path": str(path),
            "patches": [{"op": "replace", "path": "/1/inputs/x", "value": 1}],
        }))
        assert "error" in result
        assert "without editable data" in result["error"]


class TestUndo:
    def test_undo_single(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 999}],
        })
        result = json.loads(workflow_patch.handle("undo_workflow_patch", {}))
        assert result["undone"] is True
        assert result["remaining_changes_from_base"] == 0

        # Verify the value reverted
        current = workflow_patch.get_current_workflow()
        assert current["3"]["inputs"]["seed"] == 42

    def test_undo_multiple(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 100}],
        })
        workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "path": "/3/inputs/steps", "value": 50}],
        })
        workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "path": "/2/inputs/text", "value": "new"}],
        })

        # Undo 3 times
        for i in range(3):
            result = json.loads(workflow_patch.handle("undo_workflow_patch", {}))
            assert result["undone"] is True

        # Should be back to original
        current = workflow_patch.get_current_workflow()
        assert current["3"]["inputs"]["seed"] == 42
        assert current["3"]["inputs"]["steps"] == 20
        assert current["2"]["inputs"]["text"] == "a beautiful sunset"

    def test_undo_nothing(self, sample_workflow):
        # Load but don't patch
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 42}],
        })
        workflow_patch.handle("undo_workflow_patch", {})
        # Now nothing to undo
        result = json.loads(workflow_patch.handle("undo_workflow_patch", {}))
        assert "error" in result

    def test_undo_not_loaded(self):
        result = json.loads(workflow_patch.handle("undo_workflow_patch", {}))
        assert "error" in result


class TestPreview:
    def test_preview_shows_changes(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 42}],
        })
        result = json.loads(workflow_patch.handle("preview_workflow_patch", {
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 999}],
        }))
        assert result["would_succeed"] is True
        assert result["preview"][0]["current_value"] == 42
        assert result["preview"][0]["new_value"] == 999

    def test_preview_doesnt_modify(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 42}],
        })
        workflow_patch.handle("preview_workflow_patch", {
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 999}],
        })
        # Value should NOT have changed
        current = workflow_patch.get_current_workflow()
        assert current["3"]["inputs"]["seed"] == 42


class TestDiff:
    def test_no_changes(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 42}],
        })
        workflow_patch.handle("undo_workflow_patch", {})
        result = json.loads(workflow_patch.handle("get_workflow_diff", {}))
        assert result["changes"] == 0

    def test_shows_diff(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [
                {"op": "replace", "path": "/3/inputs/seed", "value": 999},
                {"op": "replace", "path": "/2/inputs/text", "value": "new text"},
            ],
        })
        result = json.loads(workflow_patch.handle("get_workflow_diff", {}))
        assert result["changes"] == 2
        assert len(result["diff"]) == 2


class TestSave:
    def test_save_to_new_file(self, sample_workflow, tmp_path):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 999}],
        })
        output = tmp_path / "saved.json"
        result = json.loads(workflow_patch.handle("save_workflow", {
            "output_path": str(output),
        }))
        assert result["saved"] == str(output)
        assert result["changes_from_base"] == 1

        # Verify file content
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["3"]["inputs"]["seed"] == 999

    def test_save_not_loaded(self):
        result = json.loads(workflow_patch.handle("save_workflow", {}))
        assert "error" in result


class TestReset:
    def test_reset(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/3/inputs/seed", "value": 999}],
        })
        result = json.loads(workflow_patch.handle("reset_workflow", {}))
        assert result["reset"] is True

        current = workflow_patch.get_current_workflow()
        assert current["3"]["inputs"]["seed"] == 42


# ---------------------------------------------------------------------------
# COMFY_AUTOGROW_V3 support
# ---------------------------------------------------------------------------

class TestAutogrowSetInput:
    """Test set_input with AUTOGROW dotted names."""

    @pytest.fixture
    def autogrow_workflow(self, tmp_path):
        data = {
            "1": {
                "class_type": "ComfyMathExpression",
                "inputs": {
                    "expression": "a + b",
                    "values": {"a": 42, "b": 7},
                },
            },
        }
        path = tmp_path / "autogrow_wf.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_set_autogrow_sub_input(self, autogrow_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(autogrow_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "1",
            "input_name": "values.a",
            "value": 100,
        }))
        assert result["set"] is True
        assert result["old_value"] == 42
        assert result["new_value"] == 100
        # Verify the nested structure is correct in the workflow
        wf = workflow_patch.get_current_workflow()
        assert wf["1"]["inputs"]["values"]["a"] == 100
        assert wf["1"]["inputs"]["values"]["b"] == 7  # unchanged

    def test_set_autogrow_new_sub_input(self, autogrow_workflow):
        """Adding a new sub-input to an AUTOGROW group."""
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(autogrow_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "1",
            "input_name": "values.c",
            "value": 99,
        }))
        assert result["set"] is True
        assert result["old_value"] is None
        wf = workflow_patch.get_current_workflow()
        assert wf["1"]["inputs"]["values"]["c"] == 99

    def test_set_normal_input_unchanged(self, autogrow_workflow):
        """Normal (non-dotted) inputs still work as before."""
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(autogrow_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "1",
            "input_name": "expression",
            "value": "a * b",
        }))
        assert result["set"] is True
        assert result["old_value"] == "a + b"


class TestConnectNodesValidation:
    """Input validation for connect_nodes (Cycle 25 fix)."""

    def test_negative_from_output_rejected(self, sample_workflow):
        """Negative from_output must return an error, not pass a negative slot index."""
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "1",
            "from_output": -1,
            "to_node": "3",
            "to_input": "model",
        }))
        assert "error" in result
        assert "from_output" in result["error"]

    def test_zero_from_output_accepted(self, sample_workflow):
        """from_output=0 is a valid slot index and must succeed."""
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "1",
            "from_output": 0,
            "to_node": "3",
            "to_input": "model",
        }))
        # Should succeed or fail for a workflow reason, never from_output validation
        assert "from_output" not in result.get("error", "")


class TestAutogrowConnectNodes:
    """Test connect_nodes with AUTOGROW dotted names."""

    @pytest.fixture
    def autogrow_workflow(self, tmp_path):
        data = {
            "1": {
                "class_type": "SomeIntNode",
                "inputs": {"value": 10},
            },
            "2": {
                "class_type": "ComfyMathExpression",
                "inputs": {
                    "expression": "a + b",
                    "values": {"a": 42, "b": 7},
                },
            },
        }
        path = tmp_path / "autogrow_conn.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_connect_to_autogrow_sub_input(self, autogrow_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(autogrow_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "1",
            "from_output": 0,
            "to_node": "2",
            "to_input": "values.a",
        }))
        assert result["connected"] is True
        assert result["previous_value"] == 42
        # Verify the nested structure
        wf = workflow_patch.get_current_workflow()
        assert wf["2"]["inputs"]["values"]["a"] == ["1", 0]
        assert wf["2"]["inputs"]["values"]["b"] == 7  # unchanged


class TestStateRegistryIsolation:
    """_get_state() must survive clear_sessions() without stale references."""

    def test_get_state_returns_live_session_after_clear(self):
        """After clear_sessions(), _get_state() returns the new live session.

        This is the regression test for the H1 bug: previously _state was a
        module-level binding set at import time. After clear_sessions(), the
        binding pointed to the old (now-orphaned) WorkflowSession. With
        _get_state() as a function, clear_sessions() + get_session("default")
        creates a fresh session that _get_state() correctly returns.
        """
        # Write a sentinel value into the current default session
        workflow_patch._get_state()["loaded_path"] = "/sentinel.json"
        assert workflow_patch._get_state()["loaded_path"] == "/sentinel.json"

        # clear_sessions() evicts all sessions from the registry
        clear_sessions()

        # _get_state() must now return the NEW fresh default session
        s_after = workflow_patch._get_state()
        assert s_after["loaded_path"] is None  # Fresh session, not the old one

    def test_get_state_consistent_within_lock(self):
        """Two consecutive _get_state() calls without clear_sessions() return the same object."""
        s1 = workflow_patch._get_state()
        s2 = workflow_patch._get_state()
        assert s1 is s2


# ---------------------------------------------------------------------------
# Cycle 30: patch element validation tests
# ---------------------------------------------------------------------------

class TestPatchElementValidation:
    """apply_workflow_patch must validate each patch element type and required fields."""

    def _load_simple_workflow(self):
        """Load a minimal workflow into patch state."""
        import copy
        from agent.tools.workflow_patch import _get_state
        s = _get_state()
        with s._lock:
            s["loaded_path"] = "<test>"
            s["format"] = "api"
            s["base_workflow"] = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
            s["current_workflow"] = copy.deepcopy(s["base_workflow"])
            s["history"] = []
            s["_engine"] = None

    def _clear_workflow(self):
        from agent.tools.workflow_patch import _get_state
        s = _get_state()
        with s._lock:
            s["loaded_path"] = None
            s["base_workflow"] = None
            s["current_workflow"] = None
            s["history"] = []
            s["_engine"] = None

    def setup_method(self):
        self._load_simple_workflow()

    def teardown_method(self):
        self._clear_workflow()

    def test_non_dict_patch_element_returns_error(self):
        """A string element in patches must return a clear error."""
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": ["not_a_dict"],
        }))
        assert "error" in result
        assert "dict" in result["error"].lower() or "patches[0]" in result["error"]

    def test_patch_missing_op_returns_error(self):
        """A patch element missing 'op' must return a clear error."""
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"path": "/1/inputs/seed", "value": 99}],
        }))
        assert "error" in result
        assert "op" in result["error"] or "missing" in result["error"].lower()

    def test_patch_missing_path_returns_error(self):
        """A patch element missing 'path' must return a clear error."""
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "value": 99}],
        }))
        assert "error" in result
        assert "path" in result["error"] or "missing" in result["error"].lower()

    def test_valid_patch_succeeds(self):
        """A well-formed patch must apply without error."""
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "path": "/1/inputs/seed", "value": 99}],
        }))
        assert "error" not in result
        assert result.get("applied") == 1

    def test_null_element_in_patches_returns_error(self):
        """None element in patches list must return a clear error."""
        result = json.loads(workflow_patch.handle("apply_workflow_patch", {
            "patches": [None],
        }))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 33: non-dict node value guard for connect_nodes and set_input
# ---------------------------------------------------------------------------

class TestNonDictNodeGuard:
    """connect_nodes and set_input must return error when a workflow node is not a dict.

    We inject the malformed node directly into workflow state (rather than via
    load_workflow, which may filter such nodes at parse time).
    """

    def _inject_malformed_node(self, sample_workflow):
        """Load a valid workflow via apply_workflow_patch (which also loads), then inject a non-dict node."""
        # Trigger a no-op patch to load the workflow into state
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })
        # Directly inject a non-dict node — simulates corrupted in-memory state.
        state = workflow_patch._get_state()
        state["current_workflow"]["99"] = "not_a_dict"

    def test_connect_nodes_from_node_non_dict_returns_error(self, sample_workflow):
        """connect_nodes: from_node is non-dict — must return error, not AttributeError."""
        self._inject_malformed_node(sample_workflow)
        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "99",
            "from_output": 0,
            "to_node": "1",
            "to_input": "model",
        }))
        assert "error" in result
        assert "Malformed" in result["error"] or "not a dict" in result["error"]

    def test_connect_nodes_to_node_non_dict_returns_error(self, sample_workflow):
        """connect_nodes: to_node is non-dict — must return error, not AttributeError."""
        self._inject_malformed_node(sample_workflow)
        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "1",
            "from_output": 0,
            "to_node": "99",
            "to_input": "model",
        }))
        assert "error" in result
        assert "Malformed" in result["error"] or "not a dict" in result["error"]

    def test_set_input_non_dict_node_returns_error(self, sample_workflow):
        """set_input: node is non-dict — must return error, not AttributeError."""
        self._inject_malformed_node(sample_workflow)
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "99",
            "input_name": "ckpt_name",
            "value": "my_model.safetensors",
        }))
        assert "error" in result
        assert "Malformed" in result["error"] or "not a dict" in result["error"]

    def test_valid_nodes_still_work(self, sample_workflow):
        """Valid dict nodes must not be affected by the guard."""
        self._inject_malformed_node(sample_workflow)
        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "1",
            "input_name": "ckpt_name",
            "value": "sd15_base.safetensors",
        }))
        assert "error" not in result
        assert result.get("set") is True


# ---------------------------------------------------------------------------
# Cycle 39: preview_workflow_patch missing 'patches' guard
# ---------------------------------------------------------------------------

class TestPreviewPatchValidation:
    """Cycle 39: preview_workflow_patch must validate 'patches' before accessing."""

    def test_missing_patches_key_returns_error(self, sample_workflow):
        """preview_workflow_patch with no 'patches' key must return error JSON."""
        import json
        from agent.tools import workflow_patch
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("preview_workflow_patch", {
            "path": str(sample_workflow),
            # 'patches' intentionally omitted
        }))
        assert "error" in result

    def test_empty_patches_list_returns_error(self, sample_workflow):
        """preview_workflow_patch with patches=[] must return error JSON."""
        import json
        from agent.tools import workflow_patch
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("preview_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        }))
        assert "error" in result

    def test_valid_patches_still_work(self, sample_workflow):
        """Valid preview_workflow_patch calls must not be affected by the guard."""
        import json
        from agent.tools import workflow_patch
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })
        result = json.loads(workflow_patch.handle("preview_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [{"op": "replace", "path": "/1/inputs/ckpt_name", "value": "v2.safetensors"}],
        }))
        assert "error" not in result


# ---------------------------------------------------------------------------
# Cycle 45 — Engine mutation fallback guards
# ---------------------------------------------------------------------------

class TestAddNodeEngineFallback:
    """_handle_add_node: engine.mutate_workflow failure falls back to direct write.

    CRITICAL invariant: undo stack must have exactly 1 snapshot pushed before the
    engine call, and the node must appear in current_workflow regardless of engine
    outcome. A throwing engine must NOT leave a stale undo entry with the node absent.
    """

    def _load(self, sample_workflow):
        """Load via apply_workflow_patch (standard path — also creates engine)."""
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })

    def test_node_added_when_engine_raises(self, sample_workflow):
        """Node must appear in current_workflow even if engine.mutate_workflow raises."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = RuntimeError("engine exploded")
        workflow_patch._set_engine(bad_engine)

        result = json.loads(workflow_patch.handle("add_node", {
            "class_type": "VAEDecode",
            "inputs": {},
        }))

        assert result.get("added") is True
        nid = result["node_id"]
        wf = workflow_patch._get_state()["current_workflow"]
        assert nid in wf, "node must be in current_workflow after engine fallback"
        assert wf[nid]["class_type"] == "VAEDecode"

    def test_undo_stack_has_one_entry_after_engine_failure(self, sample_workflow):
        """Exactly one snapshot must be pushed to history even when engine raises."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)
        # Clear history so we start from 0
        workflow_patch._get_state()["history"].clear()

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = ValueError("boom")
        workflow_patch._set_engine(bad_engine)

        workflow_patch.handle("add_node", {"class_type": "KSampler", "inputs": {}})
        assert len(workflow_patch._get_state()["history"]) == 1

    def test_undo_after_engine_fallback_restores_state(self, sample_workflow):
        """Undo must restore workflow to the pre-add snapshot (engine was bypassed)."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)
        node_count_before = len(workflow_patch._get_state()["current_workflow"])

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = RuntimeError("fail")
        bad_engine.pop_delta.return_value = None
        workflow_patch._set_engine(bad_engine)

        workflow_patch.handle("add_node", {"class_type": "SaveImage", "inputs": {}})
        assert len(workflow_patch._get_state()["current_workflow"]) == node_count_before + 1

        # Restore engine to None so undo doesn't try the broken engine
        workflow_patch._set_engine(None)

        undo_result = json.loads(workflow_patch.handle("undo_workflow_patch", {}))
        assert undo_result.get("undone") is True
        assert len(workflow_patch._get_state()["current_workflow"]) == node_count_before


class TestConnectNodesEngineFallback:
    """_handle_connect_nodes: engine failure falls back to direct write (non-autogrow)
    or rebuilds engine (autogrow). Connection must always be applied.
    """

    def _load(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })

    def test_connection_made_when_engine_raises_non_autogrow(self, sample_workflow):
        """Non-dotted to_input: connection applied via direct write when engine fails."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = RuntimeError("oops")
        workflow_patch._set_engine(bad_engine)

        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "1",
            "from_output": 0,
            "to_node": "2",
            "to_input": "clip",
        }))

        assert result.get("connected") is True
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["2"]["inputs"]["clip"] == ["1", 0]

    def test_undo_stack_intact_after_connect_engine_failure(self, sample_workflow):
        """Exactly 1 snapshot pushed before engine call regardless of engine failure."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)
        workflow_patch._get_state()["history"].clear()

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = ValueError("kaboom")
        workflow_patch._set_engine(bad_engine)

        workflow_patch.handle("connect_nodes", {
            "from_node": "1",
            "from_output": 0,
            "to_node": "3",
            "to_input": "model",
        })
        assert len(workflow_patch._get_state()["history"]) == 1

    def test_autogrow_connection_made_when_engine_raises(self, sample_workflow):
        """Dotted to_input (autogrow): dict mutation already applied, engine rebuild attempted."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = RuntimeError("crash")
        workflow_patch._set_engine(bad_engine)

        result = json.loads(workflow_patch.handle("connect_nodes", {
            "from_node": "1",
            "from_output": 0,
            "to_node": "2",
            "to_input": "extra.my_slot",
        }))

        assert result.get("connected") is True
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["2"]["inputs"]["extra"]["my_slot"] == ["1", 0]


class TestSetInputEngineFallback:
    """_handle_set_input: engine failure falls back to direct write (non-autogrow)
    or rebuilds engine (autogrow). Input value must always be applied.
    """

    def _load(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })

    def test_value_set_when_engine_raises_non_autogrow(self, sample_workflow):
        """Non-dotted input_name: value applied via direct write when engine fails."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = RuntimeError("engine dead")
        workflow_patch._set_engine(bad_engine)

        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "3",
            "input_name": "steps",
            "value": 42,
        }))

        assert result.get("set") is True
        assert result["new_value"] == 42
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["3"]["inputs"]["steps"] == 42

    def test_undo_stack_intact_after_set_input_engine_failure(self, sample_workflow):
        """1 snapshot in history even when engine raises in set_input."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)
        workflow_patch._get_state()["history"].clear()

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = ValueError("nope")
        workflow_patch._set_engine(bad_engine)

        workflow_patch.handle("set_input", {
            "node_id": "1",
            "input_name": "ckpt_name",
            "value": "new_model.safetensors",
        })
        assert len(workflow_patch._get_state()["history"]) == 1

    def test_autogrow_value_set_when_engine_raises(self, sample_workflow):
        """Dotted input_name (autogrow): dict mutation applied, engine rebuild attempted."""
        from unittest.mock import MagicMock
        self._load(sample_workflow)

        bad_engine = MagicMock()
        bad_engine.mutate_workflow.side_effect = RuntimeError("oops")
        workflow_patch._set_engine(bad_engine)

        result = json.loads(workflow_patch.handle("set_input", {
            "node_id": "3",
            "input_name": "extra.strength",
            "value": 0.75,
        }))

        assert result.get("set") is True
        wf = workflow_patch._get_state()["current_workflow"]
        assert wf["3"]["inputs"]["extra"]["strength"] == 0.75


class TestUndoEngineRebuildFailure:
    """_handle_undo: if engine rebuild after a history-only undo raises,
    undo still succeeds and engine is set to None (disabled).
    """

    def _load(self, sample_workflow):
        workflow_patch.handle("apply_workflow_patch", {
            "path": str(sample_workflow),
            "patches": [],
        })

    def test_undo_succeeds_when_engine_rebuild_raises(self, sample_workflow):
        """Undo must restore workflow even if _create_engine raises during rebuild."""
        from unittest.mock import MagicMock, patch as mock_patch
        self._load(sample_workflow)

        # Make a real change to populate history
        workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "path": "/3/inputs/steps", "value": 99}],
        })
        assert workflow_patch._get_state()["current_workflow"]["3"]["inputs"]["steps"] == 99

        # Set an engine whose pop_delta() returns None → triggers rebuild path
        mock_engine = MagicMock()
        mock_engine.pop_delta.return_value = None
        workflow_patch._set_engine(mock_engine)

        # Make _create_engine raise so the rebuild fails
        with mock_patch.object(workflow_patch, "_create_engine", side_effect=RuntimeError("no engine")):
            undo_result = json.loads(workflow_patch.handle("undo_workflow_patch", {}))

        assert undo_result.get("undone") is True, "undo must still succeed when engine rebuild fails"
        # Workflow must be restored to pre-change state
        assert workflow_patch._get_state()["current_workflow"]["3"]["inputs"]["steps"] == 20

    def test_engine_disabled_when_rebuild_raises(self, sample_workflow):
        """Engine must be set to None (disabled) when rebuild after undo raises."""
        from unittest.mock import MagicMock, patch as mock_patch
        self._load(sample_workflow)

        # Make a change so history is non-empty
        workflow_patch.handle("apply_workflow_patch", {
            "patches": [{"op": "replace", "path": "/3/inputs/cfg", "value": 5.0}],
        })

        mock_engine = MagicMock()
        mock_engine.pop_delta.return_value = None
        workflow_patch._set_engine(mock_engine)

        with mock_patch.object(workflow_patch, "_create_engine", side_effect=RuntimeError("unavail")):
            workflow_patch.handle("undo_workflow_patch", {})

        assert workflow_patch._get_engine() is None, "engine must be disabled after rebuild failure"
