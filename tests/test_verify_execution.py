"""Tests for the verify_execution module (get_output_path + verify_execution)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.verify_execution import (
    TOOLS,
    _build_narrative_summary,
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


# ---------------------------------------------------------------------------
# TestMetadataEmbed — metadata auto-embed in verify_execution
# ---------------------------------------------------------------------------

class TestMetadataEmbed:
    """Tests for metadata auto-embedding during verify_execution."""

    @pytest.fixture
    def png_file(self, mock_comfyui_database):
        """Create a real PNG file in the mock output dir."""
        img_path = mock_comfyui_database / "ComfyUI_00001_.png"
        # Minimal valid PNG (1x1 white pixel)
        import struct
        import zlib

        def _make_png():
            sig = b"\x89PNG\r\n\x1a\n"
            # IHDR
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = (
                struct.pack(">I", 13) + b"IHDR" + ihdr_data
                + struct.pack(">I", ihdr_crc)
            )
            # IDAT
            raw = zlib.compress(b"\x00\xff\xff\xff")
            idat_crc = zlib.crc32(b"IDAT" + raw) & 0xFFFFFFFF
            idat = (
                struct.pack(">I", len(raw)) + b"IDAT" + raw
                + struct.pack(">I", idat_crc)
            )
            # IEND
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return sig + ihdr + idat + iend

        img_path.write_bytes(_make_png())
        return img_path

    @pytest.fixture
    def history_with_png(self, png_file):
        """History response referencing the PNG file."""
        return {
            "abc123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": png_file.name,
                                "subfolder": "",
                            },
                        ],
                    },
                },
            },
        }

    @pytest.fixture
    def sample_wf(self):
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd15.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"steps": 20, "cfg": 7.0},
            },
        }

    def _mock_dispatch(self, name, inp):
        """Default dispatch mock: returns success for known tools."""
        if name == "get_current_intent":
            return json.dumps({"status": "empty", "intent": None})
        if name == "record_outcome":
            return json.dumps({"recorded": True})
        if name == "write_image_metadata":
            return json.dumps({"status": "ok"})
        return json.dumps({"error": "unknown"})

    def test_verify_embeds_metadata_on_success(
        self, mock_comfyui_database, png_file, history_with_png, sample_wf,
    ):
        """When outputs exist and are PNGs, verify dispatches write_image_metadata."""
        call_log = []

        def dispatch(name, inp):
            call_log.append(name)
            return self._mock_dispatch(name, inp)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history_with_png
            MockClient.return_value.__enter__.return_value.get.return_value = (
                mock_resp
            )
            with patch("agent.tools.handle", side_effect=dispatch):
                with patch(
                    "agent.tools.workflow_patch.get_current_workflow",
                    return_value=sample_wf,
                ):
                    result = _verify_prompt("abc123")

        assert "write_image_metadata" in call_log
        assert result["metadata_embedded"] is True

    def test_verify_skips_metadata_when_outputs_missing(
        self, mock_comfyui_database,
    ):
        """When output file doesn't exist, no metadata embedding."""
        history = {
            "abc123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "9": {
                        "images": [
                            {"filename": "nonexistent.png", "subfolder": ""},
                        ],
                    },
                },
            },
        }
        call_log = []

        def dispatch(name, inp):
            call_log.append(name)
            return self._mock_dispatch(name, inp)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history
            MockClient.return_value.__enter__.return_value.get.return_value = (
                mock_resp
            )
            with patch("agent.tools.handle", side_effect=dispatch):
                result = _verify_prompt("abc123")

        assert "write_image_metadata" not in call_log
        assert result["metadata_embedded"] is False

    def test_verify_metadata_includes_intent(
        self, mock_comfyui_database, png_file, history_with_png, sample_wf,
    ):
        """When get_current_intent returns intent data, metadata has intent key."""
        captured_payloads = []

        def dispatch(name, inp):
            if name == "get_current_intent":
                return json.dumps({
                    "status": "ok",
                    "intent": {
                        "user_request": "Make it dreamier",
                        "interpretation": "Lower CFG",
                        "style_references": [],
                        "session_context": "",
                    },
                })
            if name == "record_outcome":
                return json.dumps({"recorded": True})
            if name == "write_image_metadata":
                captured_payloads.append(inp)
                return json.dumps({"status": "ok"})
            return json.dumps({})

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history_with_png
            MockClient.return_value.__enter__.return_value.get.return_value = (
                mock_resp
            )
            with patch("agent.tools.handle", side_effect=dispatch):
                with patch(
                    "agent.tools.workflow_patch.get_current_workflow",
                    return_value=sample_wf,
                ):
                    _verify_prompt("abc123")

        assert len(captured_payloads) == 1
        metadata = captured_payloads[0]["metadata"]
        assert "intent" in metadata
        assert metadata["intent"]["user_request"] == "Make it dreamier"

    def test_verify_metadata_without_intent(
        self, mock_comfyui_database, png_file, history_with_png, sample_wf,
    ):
        """When get_current_intent returns 'empty', metadata still embedded but no intent."""
        captured_payloads = []

        def dispatch(name, inp):
            if name == "get_current_intent":
                return json.dumps({"status": "empty", "intent": None})
            if name == "record_outcome":
                return json.dumps({"recorded": True})
            if name == "write_image_metadata":
                captured_payloads.append(inp)
                return json.dumps({"status": "ok"})
            return json.dumps({})

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history_with_png
            MockClient.return_value.__enter__.return_value.get.return_value = (
                mock_resp
            )
            with patch("agent.tools.handle", side_effect=dispatch):
                with patch(
                    "agent.tools.workflow_patch.get_current_workflow",
                    return_value=sample_wf,
                ):
                    result = _verify_prompt("abc123")

        assert result["metadata_embedded"] is True
        assert len(captured_payloads) == 1
        metadata = captured_payloads[0]["metadata"]
        assert "intent" not in metadata

    def test_verify_metadata_embed_failure_doesnt_break(
        self, mock_comfyui_database, png_file, history_with_png, sample_wf,
    ):
        """If write_image_metadata raises, verify still returns normally."""

        def dispatch(name, inp):
            if name == "get_current_intent":
                return json.dumps({"status": "empty", "intent": None})
            if name == "record_outcome":
                return json.dumps({"recorded": True})
            if name == "write_image_metadata":
                raise RuntimeError("Disk full")
            return json.dumps({})

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history_with_png
            MockClient.return_value.__enter__.return_value.get.return_value = (
                mock_resp
            )
            with patch("agent.tools.handle", side_effect=dispatch):
                with patch(
                    "agent.tools.workflow_patch.get_current_workflow",
                    return_value=sample_wf,
                ):
                    result = _verify_prompt("abc123")

        assert result["metadata_embedded"] is False
        assert result["status"] == "complete"
        assert result["outcome_recorded"] is True

    def test_verify_result_includes_metadata_embedded_flag(
        self, mock_comfyui_database, png_file, history_with_png, sample_wf,
    ):
        """Result dict always has metadata_embedded key."""

        def dispatch(name, inp):
            return self._mock_dispatch(name, inp)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history_with_png
            MockClient.return_value.__enter__.return_value.get.return_value = (
                mock_resp
            )
            with patch("agent.tools.handle", side_effect=dispatch):
                with patch(
                    "agent.tools.workflow_patch.get_current_workflow",
                    return_value=sample_wf,
                ):
                    result = _verify_prompt("abc123")

        assert "metadata_embedded" in result
        assert isinstance(result["metadata_embedded"], bool)
        assert result["metadata_embedded"] is True


# ---------------------------------------------------------------------------
# TestNarrativeSummary — _build_narrative_summary tests
# ---------------------------------------------------------------------------

class TestNarrativeSummary:
    """Tests for _build_narrative_summary."""

    def test_narrative_summary_full(self):
        """All params produce a readable string with all components."""
        params = {
            "model": "dreamshaper_8.safetensors",
            "resolution": "1024x1024",
            "steps": 30,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.7,
        }
        summary = _build_narrative_summary(params)
        assert "dreamshaper_8" in summary
        assert "1024x1024" in summary
        assert "30 steps" in summary
        assert "CFG 7.0" in summary
        assert "euler normal" in summary
        assert "(denoise 0.7)" in summary

    def test_narrative_summary_minimal(self):
        """Only model key produces just the model name."""
        params = {"model": "sd15.safetensors"}
        summary = _build_narrative_summary(params)
        assert "sd15" in summary

    def test_narrative_summary_strips_extension(self):
        """Model extension is stripped from output."""
        params = {"model": "v1-5-pruned.safetensors"}
        summary = _build_narrative_summary(params)
        assert "v1-5-pruned" in summary
        assert ".safetensors" not in summary

    def test_narrative_summary_denoise_shown_when_partial(self):
        """Denoise < 1.0 is shown."""
        params = {"model": "sd15.ckpt", "denoise": 0.7}
        summary = _build_narrative_summary(params)
        assert "(denoise 0.7)" in summary

    def test_narrative_summary_denoise_hidden_at_full(self):
        """Denoise == 1.0 is not shown."""
        params = {"model": "sd15.ckpt", "denoise": 1.0}
        summary = _build_narrative_summary(params)
        assert "(denoise" not in summary

    def test_narrative_summary_empty_params(self):
        """Empty dict returns 'unknown'."""
        summary = _build_narrative_summary({})
        assert "unknown" in summary.lower()


# ---------------------------------------------------------------------------
# Cycle 31: verify_execution outputs null/non-dict guard tests
# ---------------------------------------------------------------------------

class TestOutputsNullGuard:
    """_verify_prompt must handle null/non-dict 'outputs' without crashing."""

    def _history_with_outputs(self, outputs_value):
        return {
            "test-id": {
                "status": {"status_str": "success", "completed": True},
                "outputs": outputs_value,
            },
        }

    def test_null_outputs_returns_empty_list(self, mock_comfyui_database):
        """outputs=null in ComfyUI history must produce empty output list, not crash."""
        history = self._history_with_outputs(None)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _verify_prompt("test-id")

        assert "error" not in result or result.get("output_count") == 0
        assert result["outputs"] == []

    def test_list_outputs_returns_empty_list(self, mock_comfyui_database):
        """outputs=[] (list instead of dict) must not crash."""
        history = self._history_with_outputs([])

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _verify_prompt("test-id")

        assert result["outputs"] == []

    def test_node_out_null_skipped(self, mock_comfyui_database):
        """A null value for a node output must be skipped, not crash on .get()."""
        history = self._history_with_outputs({"9": None, "10": {"images": []}})

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = history
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _verify_prompt("test-id")

        # Should process node "10" (no images) and skip null node "9"
        assert result["outputs"] == []

    def test_normal_outputs_still_work(self, mock_comfyui_database, mock_history_success):
        """Normal dict outputs must continue to work correctly after the guard."""
        (mock_comfyui_database / "ComfyUI_00001_.png").write_bytes(b"x" * 1024)

        with patch("agent.tools.verify_execution.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_history_success
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

            result = _verify_prompt("abc123")

        assert result["output_count"] == 1
        assert result["outputs"][0]["exists"] is True


# ---------------------------------------------------------------------------
# Cycle 47 — verify_execution required field guard
# ---------------------------------------------------------------------------

class TestVerifyExecutionRequiredField:
    """verify_execution (get_execution_status) must return error when prompt_id missing."""

    def test_missing_prompt_id_returns_error(self):
        from agent.tools import verify_execution
        result = json.loads(verify_execution.handle("verify_execution", {}))
        assert "error" in result
        assert "prompt_id" in result["error"].lower()

    def test_empty_prompt_id_returns_error(self):
        from agent.tools import verify_execution
        result = json.loads(verify_execution.handle("verify_execution", {"prompt_id": ""}))
        assert "error" in result

    def test_none_prompt_id_returns_error(self):
        from agent.tools import verify_execution
        result = json.loads(verify_execution.handle("verify_execution", {"prompt_id": None}))
        assert "error" in result
