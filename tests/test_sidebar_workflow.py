"""Tests for sidebar workflow context injection.

Covers: load_workflow_from_data, summarize_workflow_data,
state population, and PILOT tools after data load.
"""

import copy

import pytest

from agent.tools import workflow_patch, workflow_parse


@pytest.fixture(autouse=True)
def reset_workflow_state():
    """Reset workflow_patch state between tests."""
    original = copy.deepcopy(dict(workflow_patch._state))
    yield
    workflow_patch._state.update(original)


# -- Sample workflows ---------------------------------------------------------

SAMPLE_API_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "model.safetensors"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a beautiful sunset", "clip": ["1", 1]},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "model": ["1", 0],
            "positive": ["2", 0],
        },
    },
    "4": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["1", 2]},
    },
    "5": {
        "class_type": "SaveImage",
        "inputs": {"images": ["4", 0], "filename_prefix": "output"},
    },
}

SAMPLE_UI_WITH_API = {
    "nodes": [{"id": 1, "type": "CheckpointLoaderSimple"}],
    "links": [],
    "extra": {
        "prompt": copy.deepcopy(SAMPLE_API_WORKFLOW),
    },
}

SAMPLE_UI_ONLY = {
    "nodes": [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["model.safetensors"]},
        {"id": 2, "type": "KSampler", "widgets_values": [42, 20, 7.0]},
    ],
    "links": [],
}


# -- load_workflow_from_data --------------------------------------------------

class TestLoadWorkflowFromData:
    def test_api_format(self):
        err = workflow_patch.load_workflow_from_data(SAMPLE_API_WORKFLOW)
        assert err is None
        assert workflow_patch._state["format"] == "api"
        assert workflow_patch._state["loaded_path"] == "<sidebar>"
        assert workflow_patch._state["current_workflow"] is not None
        assert len(workflow_patch._state["current_workflow"]) == 5

    def test_ui_with_api_format(self):
        err = workflow_patch.load_workflow_from_data(SAMPLE_UI_WITH_API)
        assert err is None
        assert workflow_patch._state["format"] == "ui_with_api"
        assert len(workflow_patch._state["current_workflow"]) == 5

    def test_ui_only_returns_error(self):
        err = workflow_patch.load_workflow_from_data(SAMPLE_UI_ONLY)
        assert err is not None
        assert "UI-only" in err

    def test_empty_dict_returns_error(self):
        err = workflow_patch.load_workflow_from_data({})
        assert err is not None
        assert "No nodes" in err

    def test_history_cleared_on_load(self):
        # Load once, make a change, load again — history should be empty
        workflow_patch.load_workflow_from_data(SAMPLE_API_WORKFLOW)
        workflow_patch._state["history"].append({"fake": "entry"})
        workflow_patch.load_workflow_from_data(SAMPLE_API_WORKFLOW)
        assert workflow_patch._state["history"] == []


# -- State populated correctly ------------------------------------------------

class TestStateAfterDataLoad:
    def test_pilot_tools_work_after_data_load(self):
        """add_node, set_input should work after load_workflow_from_data."""
        workflow_patch.load_workflow_from_data(SAMPLE_API_WORKFLOW)

        # add_node
        result = workflow_patch.handle("add_node", {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        })
        import json
        data = json.loads(result)
        assert data.get("added") is True
        new_id = data["node_id"]

        # set_input on existing node
        result2 = workflow_patch.handle("set_input", {
            "node_id": "3",
            "input_name": "steps",
            "value": 30,
        })
        data2 = json.loads(result2)
        assert data2.get("set") is True
        assert data2["new_value"] == 30


# -- summarize_workflow_data --------------------------------------------------

class TestSummarizeWorkflowData:
    def test_returns_correct_structure(self):
        summary = workflow_parse.summarize_workflow_data(SAMPLE_API_WORKFLOW)
        assert summary["format"] == "api"
        assert summary["node_count"] == 5
        assert summary["connection_count"] > 0
        assert "summary" in summary
        assert "nodes" in summary
        assert "editable_fields" in summary
        assert summary["editable_field_count"] > 0

    def test_summary_contains_pipeline_info(self):
        summary = workflow_parse.summarize_workflow_data(SAMPLE_API_WORKFLOW)
        text = summary["summary"]
        assert "5 nodes" in text
        # Should mention loaders and sampling
        assert "Loader" in text or "loader" in text.lower()


# -- Unchanged workflow skips reload ------------------------------------------

class TestWorkflowChangeDetection:
    def test_unchanged_workflow_preserves_history(self):
        """If the same workflow is injected twice, undo history survives."""
        from ui.server.routes import _inject_workflow, ConversationState

        conv = ConversationState()
        _inject_workflow(conv, SAMPLE_API_WORKFLOW)

        # Simulate a patch (add to undo history)
        workflow_patch._state["history"].append({"fake": "undo_entry"})

        # Inject same workflow again — should skip reload
        _inject_workflow(conv, SAMPLE_API_WORKFLOW)

        # History should still have our entry (wasn't cleared by reload)
        assert len(workflow_patch._state["history"]) >= 1
