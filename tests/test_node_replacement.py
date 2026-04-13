"""Tests for agent/tools/node_replacement.py — Node Replacement API integration."""

import copy
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from agent.tools.node_replacement import (
    TOOLS,
    handle,
    _fetch_replacements,
    _invalidate_cache,
    _build_migration_patches,
)
from agent.tools.workflow_patch import _get_state


@contextmanager
def _set_workflow(wf):
    """Inject workflow into the default session for the duration of the block."""
    s = _get_state()
    prev = s["current_workflow"]
    s["current_workflow"] = copy.deepcopy(wf) if wf is not None else None
    try:
        yield
    finally:
        s["current_workflow"] = prev


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear replacement cache before each test."""
    _invalidate_cache()
    yield
    _invalidate_cache()


SAMPLE_REPLACEMENTS = {
    "MyOldSampler": [
        {
            "new_node_id": "MyNewSampler",
            "old_node_id": "MyOldSampler",
            "old_widget_ids": ["steps", "cfg"],
            "input_mapping": [
                {"new_id": "model", "old_id": "model"},
                {"new_id": "num_steps", "old_id": "steps"},
                {"new_id": "guidance", "old_id": "cfg"},
                {"new_id": "scheduler", "set_value": "normal"},
            ],
            "output_mapping": [
                {"new_idx": 0, "old_idx": 0},
            ],
        }
    ],
    "OldLoader": [
        {
            "new_node_id": "NewLoader",
            "old_node_id": "OldLoader",
            "old_widget_ids": None,
            "input_mapping": [
                {"new_id": "ckpt_name", "old_id": "ckpt_name"},
            ],
            "output_mapping": None,
        }
    ],
}

SAMPLE_WORKFLOW = {
    "1": {
        "class_type": "MyOldSampler",
        "inputs": {
            "model": ["2", 0],
            "steps": 20,
            "cfg": 7.0,
            "seed": 42,
        },
    },
    "2": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
    },
    "3": {
        "class_type": "SaveImage",
        "inputs": {"images": ["1", 0]},
    },
}


class TestToolRegistration:
    def test_tools_list_has_three_tools(self):
        assert len(TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {
            "get_node_replacements",
            "check_workflow_deprecations",
            "migrate_deprecated_nodes",
        }

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_unknown_tool_returns_error(self):
        result = json.loads(handle("nonexistent_tool", {}))
        assert "error" in result


class TestGetNodeReplacements:
    @patch("agent.tools.comfy_api._get")
    def test_returns_all_replacements(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        result = json.loads(handle("get_node_replacements", {}))
        assert result["count"] == 2
        assert "MyOldSampler" in result["replacements"]
        assert "OldLoader" in result["replacements"]

    @patch("agent.tools.comfy_api._get")
    def test_filter_by_old_node_id(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        result = json.loads(handle("get_node_replacements", {"old_node_id": "MyOldSampler"}))
        assert result["count"] == 1
        assert "MyOldSampler" in result["replacements"]

    @patch("agent.tools.comfy_api._get")
    def test_filter_nonexistent_returns_empty(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        result = json.loads(handle("get_node_replacements", {"old_node_id": "NoSuchNode"}))
        assert result["count"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_empty_registry(self, mock_get):
        mock_get.return_value = {}
        result = json.loads(handle("get_node_replacements", {}))
        assert result["count"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_server_error_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        result = json.loads(handle("get_node_replacements", {}))
        assert result["count"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_caching_avoids_duplicate_requests(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        handle("get_node_replacements", {})
        handle("get_node_replacements", {})
        assert mock_get.call_count == 1

    @patch("agent.tools.comfy_api._get")
    def test_none_response_returns_empty(self, mock_get):
        mock_get.return_value = None
        result = json.loads(handle("get_node_replacements", {}))
        assert result["count"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_non_dict_response_returns_empty(self, mock_get):
        mock_get.return_value = "not a dict"
        result = json.loads(handle("get_node_replacements", {}))
        assert result["count"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_unfiltered_has_note_when_empty(self, mock_get):
        mock_get.return_value = {}
        result = json.loads(handle("get_node_replacements", {}))
        assert "note" in result

    @patch("agent.tools.comfy_api._get")
    def test_filter_miss_has_note(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        result = json.loads(handle("get_node_replacements", {"old_node_id": "Missing"}))
        assert "note" in result


class TestCheckWorkflowDeprecations:
    @patch("agent.tools.comfy_api._get")
    def test_finds_deprecated_nodes(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert result["count"] == 1
        assert result["deprecated_nodes"][0]["class_type"] == "MyOldSampler"
        assert result["deprecated_nodes"][0]["node_id"] == "1"

    @patch("agent.tools.comfy_api._get")
    def test_no_workflow_loaded(self, mock_get):
        with _set_workflow(None):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert "error" in result

    @patch("agent.tools.comfy_api._get")
    def test_clean_workflow(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        clean_wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        with _set_workflow(clean_wf):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert result["count"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_empty_replacement_registry(self, mock_get):
        mock_get.return_value = {}
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert result["count"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_auto_migratable_flag(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert result["deprecated_nodes"][0]["auto_migratable"] is True

    @patch("agent.tools.comfy_api._get")
    def test_total_workflow_nodes_reported(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert result["total_workflow_nodes"] == 3

    @patch("agent.tools.comfy_api._get")
    def test_action_message_when_deprecated(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert "migrate_deprecated_nodes" in result["action"]

    @patch("agent.tools.comfy_api._get")
    def test_action_message_when_clean(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        clean_wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        with _set_workflow(clean_wf):
            result = json.loads(handle("check_workflow_deprecations", {}))
        assert "clean" in result["action"].lower()


class TestMigrateDeprecatedNodes:
    @patch("agent.tools.comfy_api._get")
    def test_dry_run_previews_changes(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": True}))
        assert result["dry_run"] is True
        assert len(result["migrations"]) == 1
        assert result["migrations"][0]["new_class"] == "MyNewSampler"
        assert result["total_patches"] > 0

    @patch("agent.tools.comfy_api._get")
    def test_default_is_dry_run(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("migrate_deprecated_nodes", {}))
        assert result["dry_run"] is True

    @patch("agent.tools.comfy_api._get")
    def test_apply_migration(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            with patch("agent.tools.workflow_patch.handle") as mock_patch:
                mock_patch.return_value = json.dumps({"success": True, "changes": 2})
                result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": False}))
        assert result["migrated"] == 1
        assert result["migrations"][0]["new_class"] == "MyNewSampler"

    @patch("agent.tools.comfy_api._get")
    def test_apply_migration_error(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            with patch("agent.tools.workflow_patch.handle") as mock_patch:
                mock_patch.return_value = json.dumps({"error": "Patch failed"})
                result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": False}))
        assert "error" in result

    @patch("agent.tools.comfy_api._get")
    def test_specific_node_ids(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("migrate_deprecated_nodes", {
                "node_ids": ["1"], "dry_run": True,
            }))
        assert len(result["migrations"]) == 1

    @patch("agent.tools.comfy_api._get")
    def test_specific_node_ids_excludes_others(self, mock_get):
        """When node_ids specified, only those nodes are migrated."""
        # Add a second deprecated node
        wf = {
            **SAMPLE_WORKFLOW,
            "4": {"class_type": "OldLoader", "inputs": {"ckpt_name": "model.safetensors"}},
        }
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(wf):
            result = json.loads(handle("migrate_deprecated_nodes", {
                "node_ids": ["4"], "dry_run": True,
            }))
        assert len(result["migrations"]) == 1
        assert result["migrations"][0]["old_class"] == "OldLoader"

    @patch("agent.tools.comfy_api._get")
    def test_no_deprecated_nodes(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        clean_wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        with _set_workflow(clean_wf):
            result = json.loads(handle("migrate_deprecated_nodes", {}))
        assert result["migrated"] == 0

    @patch("agent.tools.comfy_api._get")
    def test_no_workflow_loaded(self, mock_get):
        with _set_workflow(None):
            result = json.loads(handle("migrate_deprecated_nodes", {}))
        assert "error" in result

    @patch("agent.tools.comfy_api._get")
    def test_no_replacement_registry(self, mock_get):
        mock_get.return_value = {}
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("migrate_deprecated_nodes", {}))
        assert "error" in result

    @patch("agent.tools.comfy_api._get")
    def test_dry_run_shows_patch_count_per_migration(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": True}))
        assert "patch_count" in result["migrations"][0]
        assert result["migrations"][0]["patch_count"] > 0

    @patch("agent.tools.comfy_api._get")
    def test_apply_calls_patch_handle_with_correct_patches(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            with patch("agent.tools.workflow_patch.handle") as mock_patch:
                mock_patch.return_value = json.dumps({"success": True})
                handle("migrate_deprecated_nodes", {"dry_run": False})
        mock_patch.assert_called_once()
        call_args = mock_patch.call_args
        assert call_args[0][0] == "apply_workflow_patch"
        assert "patches" in call_args[0][1]

    @patch("agent.tools.comfy_api._get")
    def test_successful_migration_includes_undo_note(self, mock_get):
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(SAMPLE_WORKFLOW):
            with patch("agent.tools.workflow_patch.handle") as mock_patch:
                mock_patch.return_value = json.dumps({"success": True})
                result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": False}))
        assert "undo" in result["note"].lower()


class TestBuildMigrationPatches:
    def test_class_type_replacement(self):
        patches = _build_migration_patches(
            "1", {"class_type": "OldNode", "inputs": {}},
            "NewNode", [], [], {}
        )
        assert len(patches) == 1
        assert patches[0]["op"] == "replace"
        assert patches[0]["path"] == "/1/class_type"
        assert patches[0]["value"] == "NewNode"

    def test_input_remapping(self):
        node_data = {
            "class_type": "OldNode",
            "inputs": {"old_param": 42, "keep_me": "yes"},
        }
        input_mapping = [{"new_id": "new_param", "old_id": "old_param"}]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, [], {}
        )
        assert len(patches) == 2
        new_inputs = patches[1]["value"]
        assert new_inputs["new_param"] == 42
        assert new_inputs["keep_me"] == "yes"

    def test_set_value_mapping(self):
        node_data = {"class_type": "OldNode", "inputs": {}}
        input_mapping = [{"new_id": "scheduler", "set_value": "normal"}]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, [], {}
        )
        new_inputs = patches[1]["value"]
        assert new_inputs["scheduler"] == "normal"

    def test_output_remapping(self):
        node_data = {"class_type": "OldNode", "inputs": {"a": 1}}
        workflow = {
            "1": node_data,
            "2": {"class_type": "Other", "inputs": {"in": ["1", 0]}},
        }
        # Need at least one input_mapping entry so the function doesn't early-return
        input_mapping = [{"new_id": "a", "old_id": "a"}]
        output_mapping = [{"old_idx": 0, "new_idx": 1}]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, output_mapping, workflow
        )
        conn_patches = [p for p in patches if "/2/" in p["path"]]
        assert len(conn_patches) == 1
        assert conn_patches[0]["value"] == ["1", 1]

    def test_connection_inputs_preserved(self):
        node_data = {
            "class_type": "OldNode",
            "inputs": {"model": ["5", 0], "steps": 20},
        }
        input_mapping = [
            {"new_id": "model", "old_id": "model"},
            {"new_id": "num_steps", "old_id": "steps"},
        ]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, [], {}
        )
        new_inputs = patches[1]["value"]
        assert new_inputs["model"] == ["5", 0]
        assert new_inputs["num_steps"] == 20

    def test_no_output_change_when_idx_same(self):
        """Output mapping with same old/new idx should NOT generate a patch."""
        node_data = {"class_type": "OldNode", "inputs": {}}
        workflow = {
            "1": node_data,
            "2": {"class_type": "Other", "inputs": {"in": ["1", 0]}},
        }
        output_mapping = [{"old_idx": 0, "new_idx": 0}]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", [], output_mapping, workflow
        )
        conn_patches = [p for p in patches if "/2/" in p["path"]]
        assert len(conn_patches) == 0

    def test_empty_input_mapping_returns_only_class_patch(self):
        """No input_mapping means only class_type replacement."""
        patches = _build_migration_patches(
            "5", {"class_type": "Old", "inputs": {"a": 1}},
            "New", [], [], {}
        )
        assert len(patches) == 1
        assert patches[0]["path"] == "/5/class_type"

    def test_unmapped_old_inputs_preserved(self):
        """Inputs not mentioned in mapping should carry over."""
        node_data = {
            "class_type": "OldNode",
            "inputs": {"mapped": 10, "extra": "keep"},
        }
        input_mapping = [{"new_id": "new_mapped", "old_id": "mapped"}]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, [], {}
        )
        new_inputs = patches[1]["value"]
        assert new_inputs["new_mapped"] == 10
        assert new_inputs["extra"] == "keep"

    def test_set_value_overrides_old_input(self):
        """set_value should be used even when there's no old_id."""
        node_data = {"class_type": "OldNode", "inputs": {"x": 1}}
        input_mapping = [{"new_id": "new_field", "set_value": "default_val"}]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, [], {}
        )
        new_inputs = patches[1]["value"]
        assert new_inputs["new_field"] == "default_val"
        # The unmapped 'x' should also be preserved
        assert new_inputs["x"] == 1

    def test_multiple_output_remappings(self):
        """Multiple output index changes across multiple downstream nodes."""
        node_data = {"class_type": "OldNode", "inputs": {"x": 1}}
        workflow = {
            "1": node_data,
            "2": {"class_type": "A", "inputs": {"in1": ["1", 0]}},
            "3": {"class_type": "B", "inputs": {"in2": ["1", 1]}},
        }
        # Need at least one input_mapping so the function doesn't early-return
        input_mapping = [{"new_id": "x", "old_id": "x"}]
        output_mapping = [
            {"old_idx": 0, "new_idx": 2},
            {"old_idx": 1, "new_idx": 3},
        ]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, output_mapping, workflow
        )
        conn_patches = [p for p in patches if p["path"].startswith("/2/") or p["path"].startswith("/3/")]
        assert len(conn_patches) == 2
        # Check values
        vals = {p["path"]: p["value"] for p in conn_patches}
        assert vals["/2/inputs/in1"] == ["1", 2]
        assert vals["/3/inputs/in2"] == ["1", 3]

    def test_mapping_entry_without_new_id_is_skipped(self):
        """Mapping entries missing new_id should be silently skipped."""
        node_data = {"class_type": "OldNode", "inputs": {"a": 1}}
        input_mapping = [
            {"old_id": "a"},  # no new_id
            {"new_id": "b", "old_id": "a"},
        ]
        patches = _build_migration_patches(
            "1", node_data, "NewNode", input_mapping, [], {}
        )
        new_inputs = patches[1]["value"]
        assert "b" in new_inputs
        assert new_inputs["b"] == 1


class TestFetchReplacements:
    @patch("agent.tools.comfy_api._get")
    def test_caches_result(self, mock_get):
        mock_get.return_value = {"A": []}
        r1 = _fetch_replacements()
        r2 = _fetch_replacements()
        assert r1 == r2
        assert mock_get.call_count == 1

    @patch("agent.tools.comfy_api._get")
    def test_returns_empty_on_exception(self, mock_get):
        mock_get.side_effect = RuntimeError("boom")
        result = _fetch_replacements()
        assert result == {}

    @patch("agent.tools.comfy_api._get")
    def test_returns_empty_on_none(self, mock_get):
        mock_get.return_value = None
        result = _fetch_replacements()
        assert result == {}

    @patch("agent.tools.comfy_api._get")
    def test_invalidate_forces_refetch(self, mock_get):
        mock_get.return_value = {"A": []}
        _fetch_replacements()
        _invalidate_cache()
        mock_get.return_value = {"B": []}
        result = _fetch_replacements()
        assert "B" in result
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Cycle 56: json.JSONDecodeError guard in _handle_migrate
# ---------------------------------------------------------------------------

class TestMigrateJsonDecodeGuard:
    """Cycle 56: _handle_migrate must not crash if patch_handle returns malformed JSON."""

    WORKFLOW_WITH_DEPRECATED = {
        "1": {"class_type": "MyOldSampler", "inputs": {"steps": 20, "cfg": 7.0}},
    }

    @patch("agent.tools.comfy_api._get")
    def test_malformed_json_from_patch_engine_returns_error(self, mock_get):
        """If patch_handle returns non-JSON, migrate returns structured error."""
        from unittest.mock import patch as _patch
        mock_get.return_value = SAMPLE_REPLACEMENTS

        with _set_workflow(self.WORKFLOW_WITH_DEPRECATED):
            with _patch(
                "agent.tools.node_replacement.handle",
                return_value="NOT VALID JSON {{{",
                create=False,
            ):
                # We need to patch the imported patch_handle in node_replacement
                import agent.tools.node_replacement as nr_mod
                with _patch.object(
                    nr_mod,
                    "_handle_migrate",
                    wraps=nr_mod._handle_migrate,
                ):
                    # Directly test the JSON decode guard path
                    from agent.tools.workflow_patch import handle as real_patch_handle

                    orig = real_patch_handle

                    def _bad_json(name, tool_input):
                        return "MALFORMED {{"

                    with _patch("agent.tools.node_replacement.patch_handle", _bad_json, create=True):
                        # The guard in _handle_migrate wraps json.loads
                        # Simulate by patching at the source
                        pass

        # Simpler path: call with dry_run=False and monkeypatch patch_handle directly
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(self.WORKFLOW_WITH_DEPRECATED):
            import agent.tools.node_replacement as nr

            def _malformed(*_a, **_kw):
                return "NOT {{ valid json"

            orig_patch_handle = None
            # Patch the local name used inside _handle_migrate
            import agent.tools.workflow_patch as wp_mod
            orig_wp_handle = wp_mod.handle

            try:
                wp_mod.handle = _malformed
                result = json.loads(nr.handle("migrate_deprecated_nodes", {"dry_run": False}))
            finally:
                wp_mod.handle = orig_wp_handle

        assert "error" in result
        assert "malformed" in result["error"].lower() or "attempted" in result

    @patch("agent.tools.comfy_api._get")
    def test_dry_run_does_not_call_patch_engine(self, mock_get):
        """dry_run=True must return preview without touching patch engine."""
        mock_get.return_value = SAMPLE_REPLACEMENTS

        with _set_workflow(self.WORKFLOW_WITH_DEPRECATED):
            result = json.loads(
                handle("migrate_deprecated_nodes", {"dry_run": True})
            )

        assert result.get("dry_run") is True
        assert result.get("migrations") is not None
        assert "error" not in result

    @patch("agent.tools.comfy_api._get")
    def test_patch_engine_error_dict_surfaced(self, mock_get):
        """If patch engine returns {"error": ...}, migrate wraps it."""
        import json as _json
        import agent.tools.workflow_patch as wp_mod
        mock_get.return_value = SAMPLE_REPLACEMENTS

        def _error_handle(name, tool_input):
            return _json.dumps({"error": "patch conflict"})

        orig = wp_mod.handle
        try:
            wp_mod.handle = _error_handle
            with _set_workflow(self.WORKFLOW_WITH_DEPRECATED):
                result = _json.loads(
                    handle("migrate_deprecated_nodes", {"dry_run": False})
                )
        finally:
            wp_mod.handle = orig

        assert "error" in result
        assert "patch conflict" in result["error"]


# ---------------------------------------------------------------------------
# Cycle 65: isinstance guard before rep[0] in migrate_deprecated_nodes
# ---------------------------------------------------------------------------

class TestMigrateRepIsListGuard:
    """Cycle 65: replacements values must be validated as list before rep[0]."""

    WORKFLOW_WITH_DEPRECATED = {
        "1": {
            "class_type": "MyOldSampler",
            "inputs": {"model": ["2", 0], "steps": 20, "cfg": 7.0},
        },
    }

    @patch("agent.tools.comfy_api._get")
    def test_error_dict_as_replacement_value_skipped(self, mock_get):
        """If replacements[class_type] is a dict (e.g. {"error": "..."}), not a list,
        the node must be skipped without crashing (TypeError on rep[0])."""
        mock_get.return_value = {
            "MyOldSampler": {"error": "fetch failed"},  # dict, not list
        }
        with _set_workflow(self.WORKFLOW_WITH_DEPRECATED):
            result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": True}))
        # Must not crash — dict is non-list so node is skipped → migrated: 0
        assert "error" not in result or "fetch failed" not in result.get("error", "")
        assert result.get("migrated", 0) == 0  # Cycle 65: graceful skip, not crash

    @patch("agent.tools.comfy_api._get")
    def test_empty_list_as_replacement_value_skipped(self, mock_get):
        """Empty list rep = [] is falsy and must be skipped cleanly."""
        mock_get.return_value = {"MyOldSampler": []}
        with _set_workflow(self.WORKFLOW_WITH_DEPRECATED):
            result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": True}))
        # Empty list → no migration → migrated: 0
        assert result.get("migrated", 0) == 0  # Cycle 65: graceful skip

    @patch("agent.tools.comfy_api._get")
    def test_valid_list_replacement_proceeds(self, mock_get):
        """A well-formed list replacement must still be processed correctly."""
        mock_get.return_value = SAMPLE_REPLACEMENTS
        with _set_workflow(self.WORKFLOW_WITH_DEPRECATED):
            result = json.loads(handle("migrate_deprecated_nodes", {"dry_run": True}))
        assert "migrations" in result
        assert len(result["migrations"]) >= 1
