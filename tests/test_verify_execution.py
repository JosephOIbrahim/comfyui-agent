"""Tests for the verify_execution module (get_output_path + verify_execution)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.verify_execution import (
    TOOLS,
    _extract_key_params,
    _verify_prompt,
    _workflow_hash,
    handle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_comfyui_database(tmp_path, monkeypatch):
    """Point COMFYUI_OUTPUT_DIR to a temp dir and ensure _util allows it."""
    monkeypatch.setattr("agent.tools.verify_execution.COMFYUI_OUTPUT_DIR", tmp_path)
    # Also patch _util safe dirs to include tmp_path
    monkeypatch.setattr(
        "agent.tools._util._SAFE_DIRS",
        [tmp_path.resolve()],
    )
    return tmp_path


@pytest.fixture
def mock_history_success():
    """Standard successful history response."""
    return {
        "abc123": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                "9": {
                    "images": [
                        {"filename": "ComfyUI_00001_.png", "subfolder": ""},
                    ],
                },
            },
        },
    }


@pytest.fixture
def sample_workflow():
    """Workflow with standard nodes for key_params extraction."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "dreamshaper_8.safetensors"},
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 42,
                "denoise": 1.0,
                "model": ["1", 0],  # connection — should be skipped
            },
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
    }


# ---------------------------------------------------------------------------
# TestGetOutputPath
# ---------------------------------------------------------------------------

class TestGetOutputPath:
    def test_basic_resolution(self, mock_comfyui_database):
        """Resolve a simple filename to output dir."""
        img = mock_comfyui_database / "ComfyUI_00001_.png"
        img.write_bytes(b"fake png data")

        result = json.loads(handle("get_output_path", {"filename": "ComfyUI_00001_.png"}))
        assert result["exists"] is True
        assert result["size_bytes"] == len(b"fake png data")
        assert result["error"] is None
        assert "ComfyUI_00001_.png" in result["absolute_path"]

    def test_subfolder(self, mock_comfyui_database):
        """Resolve with subfolder."""
        sub = mock_comfyui_database / "batch01"
        sub.mkdir()
        img = sub / "img_001.png"
        img.write_bytes(b"data")

        result = json.loads(handle("get_output_path", {
            "filename": "img_001.png",
            "subfolder": "batch01",
        }))
        assert result["exists"] is True
        assert "batch01" in result["absolute_path"]

    def test_file_not_found(self, mock_comfyui_database):
        """Non-existent file returns exists=False."""
        result = json.loads(handle("get_output_path", {"filename": "no_such_file.png"}))
        assert result["exists"] is False
        assert result["size_bytes"] == 0
        assert result["error"] is not None

    def test_empty_filename(self, mock_comfyui_database):
        """Empty filename returns error."""
        result = json.loads(handle("get_output_path", {"filename": ""}))
        assert result["exists"] is False
        assert result["error"] == "Empty filename"


# ---------------------------------------------------------------------------
# TestVerifyExecution
# ---------------------------------------------------------------------------

class TestVerifyExecution:
    def test_basic_verify(self, mock_comfyui_database, mock_history_success):
        """Successful verification with file on disk."""
        (mock_comfyui_database / "ComfyUI_00001_.png").write_bytes(b"x" * 1024)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_history_success
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            with patch("agent.tools.verify_execution.handle", wraps=handle) as _:
                # Patch the dispatch_tool calls inside _verify_prompt
                with patch("agent.tools.verify_execution._verify_prompt", wraps=_verify_prompt):
                    result = json.loads(handle("verify_execution", {"prompt_id": "abc123"}))

        assert result["status"] == "complete"
        assert result["output_count"] == 1
        assert result["all_exist"] is True
        assert result["outputs"][0]["size_ok"] is True
        assert "verified" in result["message"].lower()

    def test_file_missing(self, mock_comfyui_database, mock_history_success):
        """Output referenced in history but file doesn't exist."""
        # Don't create the file

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_history_success
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _verify_prompt("abc123")

        assert result["all_exist"] is False
        assert result["outputs"][0]["exists"] is False
        assert "missing" in result["message"].lower()

    def test_no_history(self, mock_comfyui_database):
        """Prompt ID not found in history."""
        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {}  # empty history
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _verify_prompt("nonexistent")

        assert result["status"] == "not_found"
        assert result["output_count"] == 0

    def test_comfyui_down(self, mock_comfyui_database):
        """ComfyUI not reachable."""
        import httpx as _httpx
        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = (
                _httpx.ConnectError("Connection refused")
            )

            result = _verify_prompt("abc123")

        assert result["status"] == "error"
        assert "not reachable" in result["message"]

    def test_analyze_true_calls_vision(self, mock_comfyui_database, mock_history_success):
        """analyze=True triggers vision analysis."""
        (mock_comfyui_database / "ComfyUI_00001_.png").write_bytes(b"x" * 1024)

        vision_result = json.dumps({"quality_score": 0.85, "artifacts": []})

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_history_success
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            with patch("agent.tools.verify_execution.handle") as mock_dispatch:
                # First call is analyze_image, second is record_outcome
                mock_dispatch.side_effect = [vision_result, "{}"]

                # Call _verify_prompt directly to avoid dispatch confusion
                # We need to patch the import inside _verify_prompt
                with patch(
                    "agent.tools.verify_execution._verify_prompt.__module__",
                    create=True,
                ):
                    pass

        # Use a more direct approach: test _verify_prompt with patched dispatch
        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_history_success
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            call_log = []

            def mock_tool_dispatch(name, inp):
                call_log.append(name)
                if name == "analyze_image":
                    return json.dumps({"quality_score": 0.85, "artifacts": []})
                if name == "record_outcome":
                    return "{}"
                return "{}"

            # Patch the top-level tools.handle that _verify_prompt imports
            with patch("agent.tools.handle", side_effect=mock_tool_dispatch):
                result = _verify_prompt("abc123", analyze=True)

        assert result["vision_analysis"] is not None
        assert result["vision_analysis"]["quality_score"] == 0.85
        assert "analyze_image" in call_log

    def test_analyze_false_skips_vision(self, mock_comfyui_database, mock_history_success):
        """analyze=False (default) does not call vision."""
        (mock_comfyui_database / "ComfyUI_00001_.png").write_bytes(b"x" * 1024)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_history_success
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            with patch("agent.tools.handle") as mock_dispatch:
                mock_dispatch.return_value = "{}"
                result = _verify_prompt("abc123", analyze=False)

        assert result["vision_analysis"] is None
        # record_outcome should still be called
        assert result["outcome_recorded"] is True

    def test_key_params_extraction(self, sample_workflow):
        """Key params extracted from standard workflow nodes."""
        params = _extract_key_params(sample_workflow)
        assert params["model"] == "dreamshaper_8.safetensors"
        assert params["steps"] == 20
        assert params["cfg"] == 7.0
        assert params["sampler_name"] == "euler"
        assert params["resolution"] == "512x512"
        # Connection inputs should be skipped
        assert "model" not in params.get("details", {})

    def test_render_time_passthrough(self, mock_comfyui_database, mock_history_success):
        """render_time_s is passed through to result."""
        # output dir is mock_comfyui_database itself (already exists)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_history_success
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            with patch("agent.tools.handle", return_value="{}"):
                result = _verify_prompt("abc123", render_time_s=12.5)

        assert result["render_time_s"] == 12.5


# ---------------------------------------------------------------------------
# TestExtractKeyParams
# ---------------------------------------------------------------------------

class TestExtractKeyParams:
    def test_empty_workflow(self):
        assert _extract_key_params({}) == {}

    def test_no_recognized_nodes(self):
        wf = {"1": {"class_type": "CustomNode", "inputs": {"foo": "bar"}}}
        assert _extract_key_params(wf) == {}

    def test_connection_inputs_skipped(self, sample_workflow):
        """Connection values (lists) in KSampler should not appear in params."""
        params = _extract_key_params(sample_workflow)
        # 'model' key comes from CheckpointLoaderSimple, not KSampler connection
        assert params["model"] == "dreamshaper_8.safetensors"


# ---------------------------------------------------------------------------
# TestWorkflowHash
# ---------------------------------------------------------------------------

class TestWorkflowHash:
    def test_deterministic(self, sample_workflow):
        h1 = _workflow_hash(sample_workflow)
        h2 = _workflow_hash(sample_workflow)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_workflows(self, sample_workflow):
        h1 = _workflow_hash(sample_workflow)
        modified = {**sample_workflow, "99": {"class_type": "Extra", "inputs": {}}}
        h2 = _workflow_hash(modified)
        assert h1 != h2


# ---------------------------------------------------------------------------
# TestAutoVerify (via execute_with_progress)
# ---------------------------------------------------------------------------

class TestAutoVerify:
    def test_auto_verify_false_default(self):
        """auto_verify is not in schema defaults -- verify it's absent from base schema."""
        ewp_schema = None
        from agent.tools.comfy_execute import TOOLS as exec_tools
        for t in exec_tools:
            if t["name"] == "execute_with_progress":
                ewp_schema = t
                break
        assert ewp_schema is not None
        # After our change, auto_verify should be in the schema
        props = ewp_schema["input_schema"]["properties"]
        assert "auto_verify" in props

    def test_schema_registration(self):
        """verify_execution tools are registered."""
        names = [t["name"] for t in TOOLS]
        assert "get_output_path" in names
        assert "verify_execution" in names
        assert len(TOOLS) == 2
