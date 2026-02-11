"""Tests for workflow_patch tool — patching, undo, diff, save."""

import json
import pytest
from agent.tools import workflow_patch


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
    workflow_patch._state["loaded_path"] = None
    workflow_patch._state["base_workflow"] = None
    workflow_patch._state["current_workflow"] = None
    workflow_patch._state["history"] = []
    workflow_patch._state["format"] = None
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
        assert "UI-only" in result["error"]


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
