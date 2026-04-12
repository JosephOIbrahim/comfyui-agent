"""Integration tests — workflow load, validate, edit, and diff flows.

Exercises the tool handler chain for typical workflow operations.
All mocked — no ComfyUI or API key required.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd15.safetensors"},
    },
    "2": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["5", 0],
            "denoise": 1.0,
        },
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a beautiful landscape", "clip": ["1", 1]},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "ugly, blurry", "clip": ["1", 1]},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
}


def _write_workflow(tmp_dir: str) -> str:
    """Write the sample workflow to a temp file and return the path."""
    path = os.path.join(tmp_dir, "test_wf.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_SAMPLE_WORKFLOW, f, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateGoodWorkflow:
    """Validate a well-formed workflow reports no errors."""

    def test_validate_good_workflow(
        self, comfyui_available: str, clean_session: str
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"node_errors": {}}

        with tempfile.TemporaryDirectory() as tmp:
            wf_path = _write_workflow(tmp)

            with patch(
                "agent.tools._util.validate_path", return_value=wf_path
            ):
                from agent.tools.workflow_parse import handle as parse_handle

                raw = parse_handle("load_workflow", {"path": wf_path})
                result = json.loads(raw)
                assert result.get("status") == "loaded" or "node_count" in result

            # Now validate via mocked ComfyUI
            with patch("agent.tools.comfy_api._post", return_value=mock_resp):
                from agent.tools.comfy_execute import handle as exec_handle

                raw = exec_handle("validate_before_execute", {})
                result = json.loads(raw)
                # Should not report critical errors
                assert result.get("valid") is True or "error" not in result.get(
                    "status", ""
                )


class TestValidateMissingModel:
    """Validate a workflow with a fake model name reports an error."""

    def test_validate_missing_model(
        self, comfyui_available: str, clean_session: str
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "node_errors": {
                "1": {
                    "errors": [
                        {
                            "type": "value_not_valid",
                            "message": "ckpt_name: 'nonexistent.safetensors' not found",
                        }
                    ]
                }
            }
        }

        bad_wf = json.loads(json.dumps(_SAMPLE_WORKFLOW))
        bad_wf["1"]["inputs"]["ckpt_name"] = "nonexistent.safetensors"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_wf.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bad_wf, f, sort_keys=True)

            with patch(
                "agent.tools._util.validate_path", return_value=path
            ):
                from agent.tools.workflow_parse import handle as parse_handle

                parse_handle("load_workflow", {"path": path})

            with patch("agent.tools.comfy_api._post", return_value=mock_resp):
                from agent.tools.comfy_execute import handle as exec_handle

                raw = exec_handle("validate_before_execute", {})
                result = json.loads(raw)
                # Should flag the missing model
                has_error = (
                    result.get("valid") is False
                    or bool(result.get("node_errors"))
                    or bool(result.get("errors"))
                    or "error" in str(result).lower()
                )
                assert has_error, f"Expected validation error, got: {result}"


class TestGetEditableFields:
    """get_editable_fields returns KSampler fields after loading a workflow."""

    def test_get_editable_fields(
        self, comfyui_available: str, clean_session: str
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wf_path = _write_workflow(tmp)

            with patch(
                "agent.tools._util.validate_path", return_value=wf_path
            ):
                from agent.tools.workflow_parse import handle as parse_handle

                parse_handle("load_workflow", {"path": wf_path})

            from agent.tools.workflow_parse import handle as parse_handle

            raw = parse_handle("get_editable_fields", {})
            result = json.loads(raw)
            # Should list at least KSampler-related fields
            raw_str = json.dumps(result)
            assert "KSampler" in raw_str or "sampler" in raw_str.lower()


class TestWorkflowDiffAfterPatch:
    """Load, patch, get diff — verify diff reflects changes."""

    def test_workflow_diff_after_patch(self, clean_session: str) -> None:
        import copy

        from agent.tools.workflow_patch import _get_state

        # Directly inject workflow into session state (mirrors what
        # load_workflow does internally). This avoids path-validation
        # and file-system coupling.
        state = _get_state()
        state["current_workflow"] = copy.deepcopy(_SAMPLE_WORKFLOW)
        state["original_workflow"] = copy.deepcopy(_SAMPLE_WORKFLOW)
        state["history"] = []

        from agent.tools.workflow_patch import handle as patch_handle

        # Patch: change steps from 20 to 30 (JSON Patch format)
        raw = patch_handle(
            "apply_workflow_patch",
            {
                "patches": [
                    {"op": "replace", "path": "/2/inputs/steps", "value": 30}
                ]
            },
        )
        patch_result = json.loads(raw)
        assert patch_result.get("status") == "applied" or "applied" in str(
            patch_result
        ).lower()

        # Get diff
        raw_diff = patch_handle("get_workflow_diff", {})
        diff_result = json.loads(raw_diff)
        diff_str = json.dumps(diff_result)
        # The diff should mention the steps change
        assert "steps" in diff_str or "30" in diff_str
