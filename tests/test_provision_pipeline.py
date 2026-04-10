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


# ---------------------------------------------------------------------------
# Cycle 32: family identification and wire failure resilience
# ---------------------------------------------------------------------------

# Distinct names to avoid shadowing module-level _DOWNLOAD_OK fixture.
_C32_DOWNLOAD_OK = {
    "downloaded": "flux1-dev-fp8.safetensors",
    "path": "G:/COMFYUI_Database/models/checkpoints/flux1-dev-fp8.safetensors",
    "model_type": "checkpoints",
    "size_gb": 11.9,
    "elapsed_seconds": 120.0,
    "speed_mbps": 101.5,
    "message": "Downloaded 'flux1-dev-fp8.safetensors' (11.9 GB) to checkpoints/.",
}

_C32_FAMILY_OK = {
    "model": "flux1-dev-fp8.safetensors",
    "family": "flux",
    "label": "Flux",
    "resolution": "1024x1024",
    "lora_compatible": True,
    "incompatible_with": [],
}

_C32_DISCOVER = {
    "results": [{
        "name": "Flux.1 Dev",
        "filename": "flux1-dev-fp8.safetensors",
        "url": "https://example.com/flux1-dev-fp8.safetensors",
        "model_type": "checkpoints",
        "installed": False,
    }],
}


class TestProvisionFamilyAndWireResilience:
    """provision_model must not crash when family ID or wiring raises."""

    def test_family_id_exception_returns_partial_success(self):
        """If identify_model_family raises, provision_model must still return a result."""
        from unittest.mock import patch
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_C32_DISCOVER)), \
             patch("agent.tools.comfy_provision.handle", return_value=json.dumps(_C32_DOWNLOAD_OK)), \
             patch("agent.tools.model_compat.handle", side_effect=RuntimeError("compat boom")):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "flux dev",
                "auto_download": True,  # required to proceed past candidates step
                "auto_wire": False,
            }))
        # Should not propagate the exception; step should be "complete"
        assert "step" in result
        assert result["step"] == "complete"
        # Family should contain error info
        family = result.get("model_family", {})
        assert "error" in family or family.get("family") == "unknown"

    def test_wire_exception_returns_partial_success(self):
        """If auto_wire_handle raises, provision_model must still return a result with wired.error."""
        from unittest.mock import patch
        with patch("agent.tools.comfy_discover.handle", return_value=json.dumps(_C32_DISCOVER)), \
             patch("agent.tools.comfy_provision.handle", return_value=json.dumps(_C32_DOWNLOAD_OK)), \
             patch("agent.tools.model_compat.handle", return_value=json.dumps(_C32_FAMILY_OK)), \
             patch("agent.tools.auto_wire.handle", side_effect=RuntimeError("wire boom")):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "flux dev",
                "auto_download": True,  # required to proceed past candidates step
                "auto_wire": True,
            }))
        assert "wired" in result
        assert "error" in result["wired"]
        assert result["step"] == "complete"


# ---------------------------------------------------------------------------
# Cycle 41: json.loads guards on discover, wire, download
# ---------------------------------------------------------------------------

class TestProvisionJsonDecodeGuards:
    """Cycle 41: provision_pipeline must handle non-JSON from sub-handlers."""

    def test_discover_non_json_returns_error(self):
        """If discover returns non-JSON, provision_model must return error JSON."""
        from unittest.mock import patch

        # discover_handle is imported locally inside _handle_provision_model;
        # patch at the source module to intercept the local binding.
        with patch("agent.tools.comfy_discover.handle", return_value="NOT JSON"):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "sdxl base",
                "model_type": "checkpoints",
            }))
        assert "error" in result
        assert result.get("step") == "discover"

    def test_download_non_json_returns_error(self):
        """If download_model returns non-JSON, provision_model must return error JSON."""
        from unittest.mock import patch

        good_discover = json.dumps({
            "results": [{"filename": "model.safetensors", "url": "http://example.com/m.safetensors",
                         "installed": False, "model_type": "checkpoints"}]
        })

        with patch("agent.tools.comfy_discover.handle", return_value=good_discover), \
             patch("agent.tools.comfy_provision.handle", return_value="GARBAGE"):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "sdxl base",
                "model_type": "checkpoints",
                "auto_download": True,
            }))
        assert "error" in result
        assert result.get("step") == "download"

    def test_wire_non_json_falls_back_gracefully(self):
        """If auto_wire returns non-JSON for already-installed model, response still succeeds."""
        from unittest.mock import patch

        good_discover = json.dumps({
            "results": [{"filename": "model.safetensors", "installed": True, "model_type": "checkpoints"}]
        })

        with patch("agent.tools.comfy_discover.handle", return_value=good_discover), \
             patch("agent.tools.auto_wire.handle", return_value="BAD JSON"):
            result = json.loads(provision_pipeline.handle("provision_model", {
                "query": "sdxl base",
                "model_type": "checkpoints",
                "auto_wire": True,
            }))
        # The provision itself succeeds (model already installed)
        # wired field should be a dict, not a crash
        assert "error" not in result or result.get("step") == "already_installed"
        if "wired" in result:
            assert isinstance(result["wired"], dict)


# ---------------------------------------------------------------------------
# Cycle 42 — comfy_provision install loop non-JSON guard
# ---------------------------------------------------------------------------

class TestInstallLoopNonJsonGuard:
    """Guard against non-JSON returned by _handle_install_node_pack in repair loop."""

    def test_install_non_json_does_not_crash(self):
        """If _handle_install_node_pack returns non-JSON, the loop continues, not crashes."""
        from unittest.mock import patch
        import json

        from agent.tools import comfy_provision

        # repair_workflow calls comfy_discover.handle("find_missing_nodes") then
        # _handle_install_node_pack for each pack. We simulate a missing node that
        # has a known pack, then make the installer return garbage.
        missing_nodes_response = json.dumps({
            "missing": [
                {
                    "class_type": "TestNode",
                    "pack_url": "https://github.com/owner/test-pack",
                    "pack_name": "TestPack",
                }
            ]
        })

        with patch("agent.tools.comfy_discover.handle", return_value=missing_nodes_response), \
             patch(
                 "agent.tools.comfy_provision._handle_install_node_pack",
                 return_value="DEFINITELY NOT JSON",
             ):
            # repair_workflow should survive the non-JSON from the installer
            try:
                result = json.loads(comfy_provision.handle("repair_workflow", {}))
                # Must not crash — whatever the result, it must be JSON-parseable
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"repair_workflow crashed on non-JSON installer output: {e}")


# ---------------------------------------------------------------------------
# Cycle 48 — provision_model / provision_pipeline_verify required field guards
# ---------------------------------------------------------------------------

class TestProvisionModelRequiredField:
    """provision_model must return structured error when query is missing."""

    def test_missing_query_returns_error(self):
        import json
        from agent.tools import provision_pipeline
        result = json.loads(provision_pipeline.handle("provision_model", {}))
        assert "error" in result
        assert "query" in result["error"].lower()

    def test_empty_query_returns_error(self):
        import json
        from agent.tools import provision_pipeline
        result = json.loads(provision_pipeline.handle("provision_model", {"query": ""}))
        assert "error" in result

    def test_none_query_returns_error(self):
        import json
        from agent.tools import provision_pipeline
        result = json.loads(provision_pipeline.handle("provision_model", {"query": None}))
        assert "error" in result


class TestProvisionPipelineVerifyRequiredFields:
    """provision_pipeline_verify must return error when filename or model_type is missing."""

    def test_missing_filename_returns_error(self):
        import json
        from agent.tools import provision_pipeline
        result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
            "model_type": "checkpoints",
        }))
        assert "error" in result
        assert "filename" in result["error"].lower()

    def test_missing_model_type_returns_error(self):
        import json
        from agent.tools import provision_pipeline
        result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
            "filename": "model.safetensors",
        }))
        assert "error" in result
        assert "model_type" in result["error"].lower()

    def test_empty_filename_returns_error(self):
        import json
        from agent.tools import provision_pipeline
        result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
            "filename": "", "model_type": "checkpoints",
        }))
        assert "error" in result

    def test_none_model_type_returns_error(self):
        import json
        from agent.tools import provision_pipeline
        result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
            "filename": "model.safetensors", "model_type": None,
        }))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 65: json.loads guards on cross-tool calls in provision_pipeline_verify
# ---------------------------------------------------------------------------

class TestProvisionVerifyJsonLoadsGuard:
    """Cycle 65: provision_pipeline_verify must not crash when compat_handle
    returns malformed JSON from identify_model_family or check_model_compatibility."""

    def test_malformed_family_json_returns_partial_result(self):
        """identify_model_family returning malformed JSON must not crash verify."""
        import json
        from unittest.mock import patch
        from agent.tools import provision_pipeline

        # compat_handle is a local import inside the function — patch at source
        with patch("agent.tools.model_compat.handle") as mock_handle:
            mock_handle.return_value = "not valid json {{{"  # malformed
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))
        # Must not crash — returns result with empty family (both defaulted to {})
        assert "filename" in result or "error" in result
        assert "json" not in result.get("error", "").lower()

    def test_malformed_compat_json_returns_partial_result(self):
        """check_model_compatibility returning malformed JSON must not crash verify."""
        import json
        from unittest.mock import patch
        from agent.tools import provision_pipeline

        call_count = [0]

        def side_effect(name, tool_input):
            call_count[0] += 1
            if name == "identify_model_family":
                return json.dumps({"family": "SDXL"})
            else:
                return "malformed {{{json"

        # Patch at the source module (local import pattern)
        with patch("agent.tools.model_compat.handle", side_effect=side_effect):
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))
        # Must not crash; compat defaults to {} → workflow_compatible is None (unknown, Cycle 68)
        assert "filename" in result or "error" in result
        assert result.get("workflow_compatible") is None  # unknown when compat call fails

    def test_valid_json_from_compat_handle_processed_normally(self):
        """Well-formed JSON responses must be parsed and included in result."""
        import json
        from unittest.mock import patch
        from agent.tools import provision_pipeline

        family_response = json.dumps({"family": "SDXL", "base": "sdxl"})
        compat_response = json.dumps({"compatible": False, "reason": "family mismatch"})

        def side_effect(name, tool_input):
            if name == "identify_model_family":
                return family_response
            else:
                return compat_response

        # Patch at the source module (local import pattern)
        with patch("agent.tools.model_compat.handle", side_effect=side_effect):
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))
        assert result.get("family", {}).get("family") == "SDXL"
        assert result.get("workflow_compatible") is False


# ---------------------------------------------------------------------------
# Cycle 68: workflow_compatible must be None (unknown) when compat call fails
# ---------------------------------------------------------------------------

class TestWorkflowCompatibleErrorDefaultCycle68:
    """Cycle 68: workflow_compatible must not default to True when compat check fails."""

    def test_error_dict_from_compat_gives_none_not_true(self):
        """When check_model_compatibility returns error dict, workflow_compatible must be None."""
        import json
        from unittest.mock import patch
        from agent.tools import provision_pipeline

        def side_effect(name, tool_input):
            if name == "identify_model_family":
                return json.dumps({"family": "SDXL"})
            # check_model_compatibility returns an error (e.g., no workflow loaded)
            return json.dumps({"error": "No workflow loaded. Load a workflow first."})

        with patch("agent.tools.model_compat.handle", side_effect=side_effect):
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))

        # Must be None (unknown), NOT True (compatible) — True was the wrong default
        assert result.get("workflow_compatible") is None, \
            f"Expected None (unknown) when compat fails, got {result.get('workflow_compatible')!r}"

    def test_successful_compat_false_preserved(self):
        """check_model_compatibility returning compatible=False must be preserved."""
        import json
        from unittest.mock import patch
        from agent.tools import provision_pipeline

        def side_effect(name, tool_input):
            if name == "identify_model_family":
                return json.dumps({"family": "SD15"})
            return json.dumps({"compatible": False, "reason": "family mismatch"})

        with patch("agent.tools.model_compat.handle", side_effect=side_effect):
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))

        assert result.get("workflow_compatible") is False

    def test_successful_compat_true_preserved(self):
        """check_model_compatibility returning compatible=True must be preserved."""
        import json
        from unittest.mock import patch
        from agent.tools import provision_pipeline

        def side_effect(name, tool_input):
            if name == "identify_model_family":
                return json.dumps({"family": "SDXL"})
            return json.dumps({"compatible": True})

        with patch("agent.tools.model_compat.handle", side_effect=side_effect):
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))

        assert result.get("workflow_compatible") is True


# ---------------------------------------------------------------------------
# Cycle 69: provision_pipeline_verify stat() TOCTOU guard
# ---------------------------------------------------------------------------

class TestProvisionVerifyStatToctooCycle69:
    """Cycle 69: stat().st_size after exists() must be guarded for TOCTOU race."""

    def test_stat_oserror_does_not_propagate(self):
        """OSError on stat() (file deleted between exists and stat) must not crash."""
        import json
        from unittest.mock import patch, MagicMock
        from agent.tools import provision_pipeline
        from pathlib import Path

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.stat.side_effect = OSError("file not found (deleted between check and stat)")

        with patch("agent.tools.provision_pipeline.MODELS_DIR", MagicMock()) as mock_dir, \
             patch("agent.tools.model_compat.handle", return_value=json.dumps({})):
            mock_dir.__truediv__ = lambda self, other: MagicMock(
                __truediv__=lambda s, o: mock_path
            )
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))

        # Must not crash — stat() TOCTOU is handled
        assert "traceback" not in result.get("error", "").lower()
        assert "oserror" not in result.get("error", "").lower()

    def test_stat_success_still_works(self):
        """Normal stat() (file exists throughout) must return correct size."""
        import json
        from unittest.mock import patch, MagicMock
        from agent.tools import provision_pipeline

        with patch("agent.tools.model_compat.handle", return_value=json.dumps({})), \
             patch("agent.tools.provision_pipeline.MODELS_DIR") as mock_dir:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_stat = MagicMock()
            mock_stat.st_size = 4_000_000_000  # 4 GB
            mock_path.stat.return_value = mock_stat
            mock_dir.__truediv__ = lambda self, other: MagicMock(
                __truediv__=lambda s, o: mock_path
            )
            result = json.loads(provision_pipeline.handle("provision_pipeline_verify", {
                "filename": "model.safetensors",
                "model_type": "checkpoints",
            }))

        assert result.get("exists") is True
        assert result.get("size_bytes") == 4_000_000_000
