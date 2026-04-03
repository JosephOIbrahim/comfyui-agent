"""Tests for provision_pipeline — unified model provisioning pipeline."""

import json
from unittest.mock import patch

from agent.tools import provision_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DISCOVER_RESULTS = {
    "results": [
        {
            "name": "Flux.1 Dev",
            "filename": "flux1-dev-fp8.safetensors",
            "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors",
            "model_type": "checkpoints",
            "installed": False,
        },
        {
            "name": "Flux.1 Schnell",
            "filename": "flux1-schnell.safetensors",
            "url": "https://huggingface.co/example/flux1-schnell.safetensors",
            "model_type": "checkpoints",
            "installed": False,
        },
    ],
}

_DISCOVER_INSTALLED = {
    "results": [
        {
            "name": "Flux.1 Dev",
            "filename": "flux1-dev-fp8.safetensors",
            "url": "https://huggingface.co/example/flux1-dev-fp8.safetensors",
            "model_type": "checkpoints",
            "installed": True,
        },
    ],
}

_DISCOVER_EMPTY = {"results": []}

_DOWNLOAD_OK = {
    "downloaded": "flux1-dev-fp8.safetensors",
    "path": "G:/COMFYUI_Database/models/checkpoints/flux1-dev-fp8.safetensors",
    "model_type": "checkpoints",
    "size_gb": 11.9,
    "elapsed_seconds": 120.0,
    "speed_mbps": 101.5,
    "message": "Downloaded 'flux1-dev-fp8.safetensors' (11.9 GB) to checkpoints/.",
}

_DOWNLOAD_FAIL = {"error": "Download failed (server returned 404). Try again later."}

_FAMILY_FLUX = {
    "model": "flux1-dev-fp8.safetensors",
    "family": "flux",
    "label": "Flux",
    "resolution": "1024x1024",
    "lora_compatible": True,
    "incompatible_with": ["Stable Diffusion 1.5", "Stable Diffusion XL", "Stable Diffusion 3"],
}

_WIRE_OK = {
    "wired": True,
    "node_id": "1",
    "class_type": "CheckpointLoaderSimple",
    "input_field": "ckpt_name",
    "previous_value": "old_model.safetensors",
    "new_value": "flux1-dev-fp8.safetensors",
}

_SUGGEST_WIRING = {
    "loaders": [
        {
            "node_id": "1",
            "class_type": "CheckpointLoaderSimple",
            "model_type": "checkpoints",
            "input_field": "ckpt_name",
            "current_value": "sdxl_base.safetensors",
        },
    ],
    "loader_count": 1,
    "missing_core_loaders": [],
    "total_nodes": 5,
}

_MISSING_NODES_NONE = {"missing": []}

_MISSING_NODES_SOME = {
    "missing": [
        {"class_type": "IPAdapterApply", "pack_name": "ComfyUI_IPAdapter_plus"},
    ],
}

_COMPAT_OK = {
    "compatible": True,
    "family": "sdxl",
    "family_label": "Stable Diffusion XL",
    "resolution": "1024x1024",
    "models": {"sdxl_base.safetensors": "sdxl"},
}

_COMPAT_FAIL = {
    "compatible": False,
    "families_detected": ["sd15", "sdxl"],
    "models": {"sdxl_base.safetensors": "sdxl", "control_v11p_sd15_depth.pth": "sd15"},
    "conflicts": [{"reason": "mismatch"}],
}


# ---------------------------------------------------------------------------
# Helper to patch all downstream handlers
# ---------------------------------------------------------------------------

def _patch_handles(discover=None, provision=None, compat=None, wire=None):
    """Return a dict of mock patches for downstream handle() functions."""
    patches = {}
    if discover is not None:
        patches["agent.tools.comfy_discover.handle"] = lambda n, i: json.dumps(discover)
    if provision is not None:
        patches["agent.tools.comfy_provision.handle"] = lambda n, i: json.dumps(provision)
    if compat is not None:
        patches["agent.tools.model_compat.handle"] = lambda n, i: json.dumps(compat)
    if wire is not None:
        patches["agent.tools.auto_wire.handle"] = lambda n, i: json.dumps(wire)
    return patches


# ---------------------------------------------------------------------------
# provision_model tests
# ---------------------------------------------------------------------------

class TestProvisionModel:
    """Tests for the provision_model tool."""

    def test_full_pipeline_auto_download(self):
        """Full pipeline: discover -> download -> verify -> wire."""
        mocks = _patch_handles(
            discover=_DISCOVER_RESULTS,
            provision=_DOWNLOAD_OK,
            compat=_FAMILY_FLUX,
            wire=_WIRE_OK,
        )
        with patch.dict("agent.tools.provision_pipeline.__builtins__", {}):
            with patch("agent.tools.comfy_discover.handle", side_effect=mocks["agent.tools.comfy_discover.handle"]), \
                 patch("agent.tools.comfy_provision.handle", side_effect=mocks["agent.tools.comfy_provision.handle"]), \
                 patch("agent.tools.model_compat.handle", side_effect=mocks["agent.tools.model_compat.handle"]), \
                 patch("agent.tools.auto_wire.handle", side_effect=mocks["agent.tools.auto_wire.handle"]):

                result = json.loads(provision_pipeline.handle("provision_model", {
                    "query": "Flux dev",
                    "auto_download": True,
                    "auto_wire": True,
                }))

        assert result["step"] == "complete"
        assert result["downloaded"]["downloaded"] == "flux1-dev-fp8.safetensors"
        assert result["model_family"]["family"] == "flux"
        assert result["wired"]["wired"] is True

    def test_candidates_mode(self):
        """auto_download=false returns candidates for selection."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_RESULTS)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "Flux dev",
                "auto_download": False,
            }))

        assert result["step"] == "candidates"
        assert len(result["candidates"]) == 2
        assert result["candidates"][0]["name"] == "Flux.1 Dev"

    def test_already_installed_model(self):
        """If the best match is already installed, skip download and wire directly."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_INSTALLED)), \
             patch("agent.tools.auto_wire.handle", return_value=json.dumps(_WIRE_OK)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "Flux dev",
                "auto_download": True,
                "auto_wire": True,
            }))

        assert result["step"] == "already_installed"
        assert result["wired"]["wired"] is True

    def test_already_installed_no_wire(self):
        """Already installed, auto_wire=false: no wiring attempted."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_INSTALLED)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "Flux dev",
                "auto_wire": False,
            }))

        assert result["step"] == "already_installed"
        assert "wired" not in result

    def test_download_failure(self):
        """Download error is surfaced with step context."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_RESULTS)), \
             patch("agent.tools.comfy_provision.handle", return_value=json.dumps(_DOWNLOAD_FAIL)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "Flux dev",
                "auto_download": True,
            }))

        assert "error" in result
        assert result["step"] == "download"

    def test_no_results_found(self):
        """No models found returns error with step context."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_EMPTY)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "nonexistent_model_xyz",
            }))

        assert "error" in result
        assert result["step"] == "discover"

    def test_no_download_url(self):
        """Best match has no URL: error at download step."""
        no_url_results = {
            "results": [{
                "name": "Some Model",
                "filename": "some.safetensors",
                "url": "",
                "model_type": "checkpoints",
                "installed": False,
            }],
        }
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(no_url_results)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "some model",
                "auto_download": True,
            }))

        assert "error" in result
        assert result["step"] == "download"

    def test_auto_download_false_is_default(self):
        """Default behavior is to return candidates, not auto-download."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_RESULTS)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "Flux dev",
            }))

        assert result["step"] == "candidates"

    def test_source_filtering(self):
        """source='civitai' passes only civitai to discover."""
        calls = []

        def capture_discover(name, inp):
            calls.append(inp)
            return json.dumps(_DISCOVER_RESULTS)

        with patch("agent.tools.comfy_discover.handle", side_effect=capture_discover):
            provision_pipeline.handle("provision_model", {
                "query": "Flux dev",
                "source": "civitai",
            })

        assert calls[0]["sources"] == ["civitai"]

    def test_model_type_passed_to_discover(self):
        """model_type is forwarded to discover input."""
        calls = []

        def capture_discover(name, inp):
            calls.append(inp)
            return json.dumps(_DISCOVER_RESULTS)

        with patch("agent.tools.comfy_discover.handle", side_effect=capture_discover):
            provision_pipeline.handle("provision_model", {
                "query": "detail LoRA",
                "model_type": "loras",
            })

        assert calls[0]["model_type"] == "loras"

    def test_auto_wire_false_skips_wiring(self):
        """auto_wire=false skips the wiring step."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_RESULTS)), \
             patch("agent.tools.comfy_provision.handle", return_value=json.dumps(_DOWNLOAD_OK)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_FAMILY_FLUX)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "Flux dev",
                "auto_download": True,
                "auto_wire": False,
            }))

        assert result["step"] == "complete"
        assert "wired" not in result


# ---------------------------------------------------------------------------
# provision_pipeline_status tests
# ---------------------------------------------------------------------------

class TestProvisionPipelineStatus:
    """Tests for the provision_pipeline_status tool."""

    def test_status_ready(self):
        """All green: no missing nodes, compatible models."""
        with patch("agent.tools.auto_wire.handle", return_value=json.dumps(_SUGGEST_WIRING)), \
             patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_MISSING_NODES_NONE)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_COMPAT_OK)):
            result = json.loads(provision_pipeline.handle("provision_pipeline_status", {}))

        assert result["status"] == "ready"

    def test_status_missing_nodes(self):
        """Workflow has missing nodes."""
        with patch("agent.tools.auto_wire.handle", return_value=json.dumps(_SUGGEST_WIRING)), \
             patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_MISSING_NODES_SOME)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_COMPAT_OK)):
            result = json.loads(provision_pipeline.handle("provision_pipeline_status", {}))

        assert result["status"] == "missing_nodes"

    def test_status_incompatible_models(self):
        """Models are incompatible."""
        with patch("agent.tools.auto_wire.handle", return_value=json.dumps(_SUGGEST_WIRING)), \
             patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_MISSING_NODES_NONE)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_COMPAT_FAIL)):
            result = json.loads(provision_pipeline.handle("provision_pipeline_status", {}))

        assert result["status"] == "incompatible_models"

    def test_status_missing_and_incompatible(self):
        """Both missing nodes and incompatible models."""
        with patch("agent.tools.auto_wire.handle", return_value=json.dumps(_SUGGEST_WIRING)), \
             patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_MISSING_NODES_SOME)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_COMPAT_FAIL)):
            result = json.loads(provision_pipeline.handle("provision_pipeline_status", {}))

        assert result["status"] == "missing_nodes_and_incompatible"

    def test_status_no_workflow(self):
        """No workflow loaded: wiring returns error."""
        wiring_err = {"error": "No workflow is open. Load a workflow first with load_workflow."}
        with patch("agent.tools.auto_wire.handle", return_value=json.dumps(wiring_err)), \
             patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_MISSING_NODES_NONE)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_COMPAT_OK)):
            result = json.loads(provision_pipeline.handle("provision_pipeline_status", {}))

        assert result["status"] == "no_workflow"


# ---------------------------------------------------------------------------
# provision_pipeline_verify tests
# ---------------------------------------------------------------------------

class TestProvisionPipelineVerify:
    """Tests for the provision_pipeline_verify tool."""

    def test_verify_existing_model(self, tmp_path):
        """Verify a model that exists on disk."""
        model_dir = tmp_path / "checkpoints"
        model_dir.mkdir()
        model_file = model_dir / "flux1-dev-fp8.safetensors"
        model_file.write_bytes(b"\x00" * 1024)

        with patch("agent.tools.provision_pipeline.MODELS_DIR", tmp_path, create=True), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_FAMILY_FLUX)) as mock_compat:
            # Make compat handle return different things based on tool name
            def compat_dispatch(name, inp):
                if name == "identify_model_family":
                    return json.dumps(_FAMILY_FLUX)
                return json.dumps(_COMPAT_OK)

            mock_compat.side_effect = compat_dispatch

            # We need to patch MODELS_DIR inside the handler scope
            with patch("agent.tools.provision_pipeline.MODELS_DIR", tmp_path, create=True):
                result = json.loads(provision_pipeline._handle_provision_pipeline_verify({
                    "filename": "flux1-dev-fp8.safetensors",
                    "model_type": "checkpoints",
                }))

        assert result["exists"] is True
        assert result["size_bytes"] == 1024
        assert result["family"]["family"] == "flux"

    def test_verify_missing_model(self, tmp_path):
        """Verify a model that doesn't exist on disk."""
        model_dir = tmp_path / "checkpoints"
        model_dir.mkdir()

        with patch("agent.tools.model_compat.handle") as mock_compat:
            mock_compat.side_effect = lambda n, i: (
                json.dumps(_FAMILY_FLUX) if n == "identify_model_family"
                else json.dumps(_COMPAT_OK)
            )
            with patch("agent.tools.provision_pipeline.MODELS_DIR", tmp_path, create=True):
                result = json.loads(provision_pipeline._handle_provision_pipeline_verify({
                    "filename": "nonexistent.safetensors",
                    "model_type": "checkpoints",
                }))

        assert result["exists"] is False
        assert result["size_bytes"] == 0


# ---------------------------------------------------------------------------
# _filename_from_url tests
# ---------------------------------------------------------------------------

class TestFilenameFromUrl:
    """Tests for the _filename_from_url helper."""

    def test_simple_url(self):
        url = "https://huggingface.co/repo/resolve/main/model.safetensors"
        assert provision_pipeline._filename_from_url(url) == "model.safetensors"

    def test_encoded_url(self):
        url = "https://example.com/path/my%20model.safetensors"
        assert provision_pipeline._filename_from_url(url) == "my model.safetensors"

    def test_no_extension(self):
        url = "https://example.com/download"
        assert provision_pipeline._filename_from_url(url) == "downloaded_model.safetensors"

    def test_empty_path(self):
        url = "https://example.com/"
        assert provision_pipeline._filename_from_url(url) == "downloaded_model.safetensors"


# ---------------------------------------------------------------------------
# Dispatch / unknown tool
# ---------------------------------------------------------------------------

class TestDispatch:
    """Tests for the handle() dispatcher."""

    def test_unknown_tool(self):
        result = json.loads(provision_pipeline.handle("nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_dispatch_provision_model(self):
        """Dispatch routes provision_model correctly."""
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_DISCOVER_EMPTY)):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "test",
            }))
        assert "error" in result  # no results, but dispatched correctly

    def test_dispatch_provision_pipeline_status(self):
        """Dispatch routes provision_pipeline_status correctly."""
        with patch("agent.tools.auto_wire.handle", return_value=json.dumps(_SUGGEST_WIRING)), \
             patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_MISSING_NODES_NONE)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_COMPAT_OK)):
            result = json.loads(provision_pipeline.handle("provision_pipeline_status", {}))
        assert "status" in result
