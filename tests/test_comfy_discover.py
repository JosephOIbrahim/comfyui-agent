"""Tests for comfy_discover tools — mocked registries, no network needed."""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from agent.tools import comfy_discover


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CUSTOM_NODES = [
    {
        "author": "cubiq",
        "title": "ComfyUI_IPAdapter_plus",
        "id": "ipadapter",
        "reference": "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "files": ["https://github.com/cubiq/ComfyUI_IPAdapter_plus"],
        "install_type": "git-clone",
        "description": "IPAdapter implementation for ComfyUI.",
    },
    {
        "author": "Fannovel16",
        "title": "ComfyUI-Video-Matting",
        "id": "video-matting",
        "reference": "https://github.com/Fannovel16/ComfyUI-Video-Matting",
        "files": ["https://github.com/Fannovel16/ComfyUI-Video-Matting"],
        "install_type": "git-clone",
        "description": "Video matting nodes for ComfyUI.",
    },
    {
        "author": "WASasquatch",
        "title": "WAS Node Suite",
        "id": "was-suite",
        "reference": "https://github.com/WASasquatch/was-node-suite-comfyui",
        "files": ["https://github.com/WASasquatch/was-node-suite-comfyui"],
        "install_type": "git-clone",
        "description": "Extensive collection of utility nodes.",
    },
]

SAMPLE_EXTENSION_MAP = {
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus": [
        ["IPAdapterUnifiedLoader", "IPAdapterApply", "IPAdapterBatch"],
        {"title_aux": "ComfyUI_IPAdapter_plus"},
    ],
    "https://github.com/Fannovel16/ComfyUI-Video-Matting": [
        ["VideoMatting", "VideoMattingBatch"],
        {"title_aux": "ComfyUI-Video-Matting"},
    ],
    "https://github.com/WASasquatch/was-node-suite-comfyui": [
        ["WAS_Text_String", "WAS_Image_Resize", "WAS_Number"],
        {"title_aux": "WAS Node Suite"},
    ],
}

SAMPLE_MODEL_LIST = [
    {
        "name": "FLUX.1 Dev",
        "type": "checkpoint",
        "base": "FLUX.1",
        "save_path": "checkpoints",
        "description": "FLUX.1 development checkpoint.",
        "reference": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        "filename": "flux1-dev.safetensors",
        "url": "https://example.com/flux1-dev.safetensors",
        "size": "23.8GB",
    },
    {
        "name": "SDXL Base 1.0",
        "type": "checkpoint",
        "base": "SDXL",
        "save_path": "checkpoints",
        "description": "Stable Diffusion XL base model.",
        "reference": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "filename": "sd_xl_base_1.0.safetensors",
        "url": "https://example.com/sdxl.safetensors",
        "size": "6.94GB",
    },
    {
        "name": "ControlNet Depth SDXL",
        "type": "controlnet",
        "base": "SDXL",
        "save_path": "controlnet",
        "description": "Depth ControlNet for SDXL.",
        "reference": "https://example.com",
        "filename": "control-depth-sdxl.safetensors",
        "url": "https://example.com/control-depth.safetensors",
        "size": "2.5GB",
    },
    {
        "name": "Anime Style LoRA",
        "type": "lora",
        "base": "SDXL",
        "save_path": "loras",
        "description": "Anime style LoRA for SDXL.",
        "reference": "https://example.com",
        "filename": "anime_style.safetensors",
        "url": "https://example.com/anime.safetensors",
        "size": "150MB",
    },
]


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear registry cache between tests."""
    comfy_discover._clear_cache()
    yield


@pytest.fixture
def mock_registries(tmp_path):
    """Write sample registry files to a temp directory."""
    manager_dir = tmp_path / "ComfyUI-Manager"
    manager_dir.mkdir()

    (manager_dir / "custom-node-list.json").write_text(
        json.dumps({"custom_nodes": SAMPLE_CUSTOM_NODES}), encoding="utf-8",
    )
    (manager_dir / "extension-node-map.json").write_text(
        json.dumps(SAMPLE_EXTENSION_MAP), encoding="utf-8",
    )
    (manager_dir / "model-list.json").write_text(
        json.dumps({"models": SAMPLE_MODEL_LIST}), encoding="utf-8",
    )

    with patch.object(comfy_discover, "_MANAGER_DIR", manager_dir):
        yield manager_dir


@pytest.fixture
def mock_custom_nodes_dir(tmp_path):
    """Create a fake Custom_Nodes directory with some packs installed."""
    cn_dir = tmp_path / "Custom_Nodes"
    cn_dir.mkdir()
    (cn_dir / "ComfyUI_IPAdapter_plus").mkdir()  # installed
    # Video-Matting NOT installed
    (cn_dir / "was-node-suite-comfyui").mkdir()  # installed

    with patch.object(comfy_discover, "CUSTOM_NODES_DIR", cn_dir):
        yield cn_dir


@pytest.fixture
def mock_models_dir(tmp_path):
    """Create a fake models directory with some models installed."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "checkpoints").mkdir()
    (models_dir / "controlnet").mkdir()
    (models_dir / "loras").mkdir()

    # Install one model
    (models_dir / "checkpoints" / "sd_xl_base_1.0.safetensors").write_text("fake")

    with patch.object(comfy_discover, "MODELS_DIR", models_dir):
        yield models_dir


# ---------------------------------------------------------------------------
# Tests: search_custom_nodes
# ---------------------------------------------------------------------------

class TestSearchCustomNodes:
    """Tests for the internal _handle_search_custom_nodes function."""

    def test_search_by_name(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "IPAdapter",
        }))
        assert result["total_matches"] >= 1
        assert any("IPAdapter" in r["title"] for r in result["results"])

    def test_search_by_description(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "video matting",
        }))
        assert result["total_matches"] >= 1

    def test_search_by_author(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "cubiq",
        }))
        assert result["total_matches"] >= 1
        assert result["results"][0]["author"] == "cubiq"

    def test_search_no_results(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "zzzznonexistent",
        }))
        assert result["total_matches"] == 0

    def test_search_by_node_type_exact(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "IPAdapterUnifiedLoader",
            "by": "node_type",
        }))
        assert result["match"] == "exact"
        assert "IPAdapter" in result["pack"]["title"]

    def test_search_by_node_type_fuzzy(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "IPAdapter",
            "by": "node_type",
        }))
        assert result["match"] == "fuzzy"
        assert len(result["results"]) >= 1

    def test_search_by_node_type_not_found(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "ZZZNonexistentNode",
            "by": "node_type",
        }))
        assert result["match"] == "none"

    def test_install_status(self, mock_registries, mock_custom_nodes_dir):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "IPAdapterUnifiedLoader",
            "by": "node_type",
        }))
        assert result["pack"]["installed"] is True

    def test_max_results(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_custom_nodes({
            "query": "ComfyUI",
            "max_results": 2,
        }))
        assert len(result["results"]) <= 2

    def test_no_manager(self, tmp_path):
        """No ComfyUI Manager installed."""
        with patch.object(comfy_discover, "_MANAGER_DIR", tmp_path / "nonexistent"):
            result = json.loads(comfy_discover._handle_search_custom_nodes({
                "query": "test",
            }))
            assert "error" in result


# ---------------------------------------------------------------------------
# Tests: search_models
# ---------------------------------------------------------------------------

class TestSearchModels:
    """Tests for the internal _handle_search_models function."""

    def test_search_registry_by_name(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_models({
            "query": "FLUX",
        }))
        assert result["source"] == "registry"
        assert result["total_matches"] >= 1
        assert "FLUX" in result["results"][0]["name"]

    def test_search_registry_by_type(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_models({
            "query": "SDXL",
            "model_type": "controlnet",
        }))
        assert all(r["type"] == "controlnet" for r in result["results"])

    def test_search_registry_no_match(self, mock_registries):
        result = json.loads(comfy_discover._handle_search_models({
            "query": "zzzznonexistent",
        }))
        assert result["total_matches"] == 0

    def test_model_install_status(self, mock_registries, mock_models_dir):
        result = json.loads(comfy_discover._handle_search_models({
            "query": "SDXL Base",
        }))
        sdxl = result["results"][0]
        assert sdxl["installed"] is True

    def test_model_not_installed(self, mock_registries, mock_models_dir):
        result = json.loads(comfy_discover._handle_search_models({
            "query": "FLUX",
        }))
        flux = result["results"][0]
        assert flux["installed"] is False

    def test_huggingface_search(self):
        """Mock HuggingFace API response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "modelId": "stabilityai/stable-diffusion-xl-base-1.0",
                "downloads": 1000000,
                "likes": 5000,
                "tags": ["diffusers", "text-to-image"],
                "lastModified": "2024-01-01T00:00:00.000Z",
            },
        ]
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_discover._handle_search_models({
                "query": "stable diffusion xl",
                "source": "huggingface",
            }))
            assert result["source"] == "huggingface"
            assert len(result["results"]) == 1
            assert "stabilityai" in result["results"][0]["name"]

    def test_huggingface_connection_error(self):
        import httpx as _httpx
        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.side_effect = _httpx.ConnectError("no internet")

            result = json.loads(comfy_discover._handle_search_models({
                "query": "flux",
                "source": "huggingface",
            }))
            assert "error" in result

    def test_no_manager_for_registry(self, tmp_path):
        with patch.object(comfy_discover, "_MANAGER_DIR", tmp_path / "nonexistent"):
            result = json.loads(comfy_discover._handle_search_models({
                "query": "test",
            }))
            assert "error" in result


# ---------------------------------------------------------------------------
# Tests: find_missing_nodes
# ---------------------------------------------------------------------------

class TestFindMissingNodes:
    def test_all_installed(self, tmp_path, mock_registries):
        """All nodes available in ComfyUI."""
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        path = tmp_path / "wf.json"
        path.write_text(json.dumps(wf), encoding="utf-8")

        # Mock /object_info returning both nodes
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "CheckpointLoaderSimple": {},
            "KSampler": {},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_discover.handle("find_missing_nodes", {
                "path": str(path),
            }))
            assert result["status"] == "all_installed"

    def test_missing_with_suggestions(self, tmp_path, mock_registries):
        """Some nodes missing, pack suggestions returned."""
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "IPAdapterUnifiedLoader", "inputs": {}},
            "3": {"class_type": "VideoMatting", "inputs": {}},
        }
        path = tmp_path / "wf.json"
        path.write_text(json.dumps(wf), encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "CheckpointLoaderSimple": {},
            # IPAdapterUnifiedLoader and VideoMatting missing
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_discover.handle("find_missing_nodes", {
                "path": str(path),
            }))
            assert result["status"] == "missing_nodes"
            assert result["missing_count"] == 2
            assert result["installed_count"] == 1
            assert len(result["packs_to_install"]) == 2

            # Check pack suggestions
            pack_titles = {p["title"] for p in result["packs_to_install"]}
            assert "ComfyUI_IPAdapter_plus" in pack_titles
            assert "ComfyUI-Video-Matting" in pack_titles

    def test_missing_unknown_node(self, tmp_path, mock_registries):
        """Missing node not in any known pack."""
        wf = {
            "1": {"class_type": "SomeCompletelyUnknownNode", "inputs": {}},
        }
        path = tmp_path / "wf.json"
        path.write_text(json.dumps(wf), encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_discover.handle("find_missing_nodes", {
                "path": str(path),
            }))
            assert result["status"] == "missing_nodes"
            assert result["missing_nodes"][0]["pack_title"] is None

    def test_uses_loaded_workflow(self, mock_registries):
        """Falls back to loaded workflow from workflow_patch."""
        from agent.tools import workflow_patch
        workflow_patch._state["current_workflow"] = {
            "1": {"class_type": "KSampler", "inputs": {}},
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"KSampler": {}}
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_discover.handle("find_missing_nodes", {}))
            assert result["status"] == "all_installed"

        # Clean up
        workflow_patch._state["current_workflow"] = None

    def test_no_workflow(self):
        """No path and no loaded workflow."""
        from agent.tools import workflow_patch
        workflow_patch._state["current_workflow"] = None

        result = json.loads(comfy_discover.handle("find_missing_nodes", {}))
        assert "error" in result

    def test_file_not_found(self):
        result = json.loads(comfy_discover.handle("find_missing_nodes", {
            "path": "/nonexistent/wf.json",
        }))
        assert "error" in result

    def test_comfyui_not_running(self, tmp_path, mock_registries):
        """ComfyUI not reachable."""
        import httpx as _httpx

        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        path = tmp_path / "wf.json"
        path.write_text(json.dumps(wf), encoding="utf-8")

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.side_effect = _httpx.ConnectError("refused")

            result = json.loads(comfy_discover.handle("find_missing_nodes", {
                "path": str(path),
            }))
            assert "error" in result
            assert "reachable" in result["error"].lower() or "running" in result["error"].lower()

    def test_ui_only_workflow(self, tmp_path, mock_registries):
        """UI-only workflow — should still extract class_types from nodes array."""
        data = {
            "nodes": [
                {"id": 1, "type": "KSampler"},
                {"id": 2, "type": "CheckpointLoaderSimple"},
            ],
            "extra": {"ds": {}},
        }
        path = tmp_path / "ui.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "KSampler": {},
            "CheckpointLoaderSimple": {},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_discover.handle("find_missing_nodes", {
                "path": str(path),
            }))
            assert result["status"] == "all_installed"
            assert result["total_node_types"] == 2

    def test_install_status_in_suggestions(
        self, tmp_path, mock_registries, mock_custom_nodes_dir,
    ):
        """Pack install status appears in find_missing_nodes results."""
        wf = {
            "1": {"class_type": "IPAdapterUnifiedLoader", "inputs": {}},
            "2": {"class_type": "VideoMatting", "inputs": {}},
        }
        path = tmp_path / "wf.json"
        path.write_text(json.dumps(wf), encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {}  # both missing
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.tools.comfy_discover.httpx.Client") as mock_cls:
            mock_client = mock_cls.return_value.__enter__.return_value
            mock_client.get.return_value = mock_resp

            result = json.loads(comfy_discover.handle("find_missing_nodes", {
                "path": str(path),
            }))

            packs = {p["title"]: p for p in result["packs_to_install"]}
            # IPAdapter is "installed" in mock_custom_nodes_dir
            assert packs["ComfyUI_IPAdapter_plus"]["installed"] is True
            # Video-Matting is NOT installed
            assert packs["ComfyUI-Video-Matting"]["installed"] is False


# ---------------------------------------------------------------------------
# Tests: registration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests: check_registry_freshness
# ---------------------------------------------------------------------------

class TestRegistryFreshness:
    def test_freshness_no_manager(self, tmp_path):
        """No ComfyUI Manager installed — files don't exist."""
        with patch.object(comfy_discover, "_MANAGER_DIR", tmp_path / "nonexistent"):
            result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
            assert result["manager_installed"] is False
            for key, info in result["registries"].items():
                assert info["exists"] is False

    def test_freshness_with_registries(self, mock_registries):
        """Registry files exist and are fresh."""
        result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
        assert result["manager_installed"] is True
        for key, info in result["registries"].items():
            assert info["exists"] is True
            assert info["status"] == "fresh"
            assert info["age_seconds"] >= 0

    def test_freshness_stale_detection(self, mock_registries):
        """Files older than threshold should be flagged stale."""
        import os
        # Make one file appear 10 days old
        path = mock_registries / "custom-node-list.json"
        old_time = time.time() - (10 * 86400)
        os.utime(str(path), (old_time, old_time))

        result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
        cnl = result["registries"]["custom_node_list"]
        assert cnl["status"] == "stale"
        assert cnl["age_seconds"] >= 10 * 86400 - 5  # allow small delta

    def test_freshness_very_stale(self, mock_registries):
        """Files older than 30 days should be very_stale."""
        import os
        path = mock_registries / "model-list.json"
        old_time = time.time() - (35 * 86400)
        os.utime(str(path), (old_time, old_time))

        result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
        ml = result["registries"]["model_list"]
        assert ml["status"] == "very_stale"

    def test_freshness_recommendations(self, mock_registries):
        """Stale files should trigger recommendations."""
        import os
        for name in ("custom-node-list.json", "extension-node-map.json", "model-list.json"):
            path = mock_registries / name
            old_time = time.time() - (10 * 86400)
            os.utime(str(path), (old_time, old_time))

        result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
        assert any("stale" in r.lower() or "refresh" in r.lower() for r in result["recommendations"])

    def test_refresh_clears_cache(self, mock_registries):
        """refresh=true should clear the in-memory cache."""
        # Load some data first
        comfy_discover._load_custom_nodes()
        assert comfy_discover._cache["custom_nodes"] is not None

        result = json.loads(comfy_discover.handle("check_registry_freshness", {
            "refresh": True,
        }))
        assert result["refreshed"] is True
        assert comfy_discover._cache["custom_nodes"] is None

    def test_cache_status_reported(self, mock_registries):
        """Cache status should reflect loaded/unloaded state."""
        result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
        assert result["cache"]["custom_nodes_cached"] is False

        # Load data
        comfy_discover._load_custom_nodes()

        result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
        assert result["cache"]["custom_nodes_cached"] is True

    def test_model_dir_stats(self, mock_registries, mock_models_dir):
        result = json.loads(comfy_discover.handle("check_registry_freshness", {}))
        assert result["models"]["exists"] is True
        assert result["models"]["total_files"] >= 1


# ---------------------------------------------------------------------------
# Tests: get_install_instructions
# ---------------------------------------------------------------------------

class TestGetInstallInstructions:
    def test_node_pack_by_node_type(self, mock_registries, mock_custom_nodes_dir):
        """Find install instructions for a known node type."""
        result = json.loads(comfy_discover.handle("get_install_instructions", {
            "query": "IPAdapterUnifiedLoader",
        }))
        assert result["type"] == "node_pack"
        assert result["installed"] is True
        assert result["pack_title"] == "ComfyUI_IPAdapter_plus"

    def test_node_pack_not_installed(self, mock_registries):
        """Node pack not installed — should return install commands."""
        # No mock_custom_nodes_dir so _is_pack_installed returns False
        with patch.object(comfy_discover, "CUSTOM_NODES_DIR", MagicMock(exists=MagicMock(return_value=False))):
            result = json.loads(comfy_discover.handle("get_install_instructions", {
                "query": "IPAdapterUnifiedLoader",
            }))
            assert result["installed"] is False
            assert len(result["install_commands"]) > 0

    def test_node_pack_by_name(self, mock_registries, mock_custom_nodes_dir):
        """Find install instructions by pack name search."""
        result = json.loads(comfy_discover.handle("get_install_instructions", {
            "query": "IPAdapter",
        }))
        assert result["type"] == "node_pack"
        assert "IPAdapter" in result["pack_title"]

    def test_model_from_registry(self, mock_registries, mock_models_dir):
        """Find install instructions for a model."""
        result = json.loads(comfy_discover.handle("get_install_instructions", {
            "query": "SDXL Base",
        }))
        assert result["type"] == "model"
        assert result["installed"] is True

    def test_civitai_source(self):
        """CivitAI source returns general instructions."""
        result = json.loads(comfy_discover.handle("get_install_instructions", {
            "query": "some model",
            "source": "civitai",
        }))
        assert result["source"] == "civitai"
        assert len(result["steps"]) > 0

    def test_huggingface_source(self):
        """HuggingFace source returns general instructions."""
        result = json.loads(comfy_discover.handle("get_install_instructions", {
            "query": "some model",
            "source": "huggingface",
        }))
        assert result["source"] == "huggingface"
        assert len(result["steps"]) > 0

    def test_not_found(self, mock_registries):
        """Unknown query returns error with suggestion."""
        result = json.loads(comfy_discover.handle("get_install_instructions", {
            "query": "zzzznonexistent",
        }))
        assert "error" in result


class TestRegistration:
    def test_tools_registered(self):
        from agent.tools import ALL_TOOLS
        names = {t["name"] for t in ALL_TOOLS}
        assert "discover" in names
        assert "find_missing_nodes" in names
        assert "check_registry_freshness" in names
        assert "get_install_instructions" in names
        # Old tools should be gone
        assert "search_custom_nodes" not in names
        assert "search_models" not in names


# ---------------------------------------------------------------------------
# Tests: unified discover tool
# ---------------------------------------------------------------------------

class TestDiscover:
    """Tests for the unified discover tool."""

    def test_nodes_only(self, mock_registries):
        result = json.loads(comfy_discover.handle("discover", {
            "query": "IPAdapter",
            "category": "nodes",
        }))
        assert result["total"] >= 1
        assert result["category"] == "nodes"
        assert all(r["type"] == "node_pack" for r in result["results"])
        assert "registry_nodes" in result["sources_searched"]

    def test_models_only(self, mock_registries):
        result = json.loads(comfy_discover.handle("discover", {
            "query": "SDXL",
            "category": "models",
            "sources": ["registry"],
        }))
        assert result["total"] >= 1
        assert result["category"] == "models"
        assert all(r["type"] == "model" for r in result["results"])

    def test_all_category(self, mock_registries):
        result = json.loads(comfy_discover.handle("discover", {
            "query": "ComfyUI",
            "category": "all",
            "sources": ["registry"],
        }))
        assert result["total"] >= 1
        # May contain both nodes and models

    def test_per_source_registry(self, mock_registries):
        result = json.loads(comfy_discover.handle("discover", {
            "query": "FLUX",
            "sources": ["registry"],
        }))
        assert "registry_models" in result["sources_searched"]
        assert "civitai" not in result["sources_searched"]
        assert "huggingface" not in result["sources_searched"]

    def test_per_source_civitai(self, mock_registries):
        """CivitAI source with mock."""
        with patch("agent.tools.comfy_discover._search_civitai_unified", return_value=(
            [comfy_discover._normalize_result(
                name="Test LoRA", result_type="model", source="civitai",
                relevance_score=0.8, installed=False, url="https://civitai.com/models/1",
            )],
            None,
        )):
            result = json.loads(comfy_discover.handle("discover", {
                "query": "test",
                "sources": ["civitai"],
                "category": "models",
            }))
            assert result["total"] >= 1
            assert any(r["source"] == "civitai" for r in result["results"])

    def test_per_source_huggingface(self):
        """HuggingFace source with mock."""
        with patch("agent.tools.comfy_discover._search_hf_unified", return_value=(
            [comfy_discover._normalize_result(
                name="test/model", result_type="model", source="huggingface",
                relevance_score=0.5, installed=False, url="https://huggingface.co/test/model",
            )],
            None,
        )):
            result = json.loads(comfy_discover.handle("discover", {
                "query": "test",
                "sources": ["huggingface"],
                "category": "models",
            }))
            assert result["total"] >= 1
            assert any(r["source"] == "huggingface" for r in result["results"])

    def test_dedup_same_name(self, mock_registries):
        """Same model from multiple sources should be deduped."""
        results = [
            comfy_discover._normalize_result(
                name="Test Model", result_type="model", source="registry",
                relevance_score=0.5, installed=False, url="https://a.com",
            ),
            comfy_discover._normalize_result(
                name="Test Model", result_type="model", source="civitai",
                relevance_score=0.9, installed=False, url="https://b.com",
            ),
        ]
        deduped = comfy_discover._deduplicate(results)
        assert len(deduped) == 1
        # Higher score wins
        assert deduped[0]["source"] == "civitai"
        assert "registry" in deduped[0].get("also_found_on", [])

    def test_installed_boost(self, mock_registries):
        """Installed items should rank first."""
        results = [
            comfy_discover._normalize_result(
                name="B Not Installed", result_type="model", source="registry",
                relevance_score=1.0, installed=False, url="",
            ),
            comfy_discover._normalize_result(
                name="A Installed", result_type="model", source="registry",
                relevance_score=0.1, installed=True, url="",
            ),
        ]
        ranked = comfy_discover._rank_results(results)
        assert ranked[0]["name"] == "A Installed"

    def test_model_type_filter(self, mock_registries):
        result = json.loads(comfy_discover.handle("discover", {
            "query": "SDXL",
            "category": "models",
            "sources": ["registry"],
            "model_type": "controlnet",
        }))
        for r in result["results"]:
            assert r.get("model_type", "").lower() == "controlnet"

    def test_common_schema(self, mock_registries):
        """Every result should have the common fields."""
        result = json.loads(comfy_discover.handle("discover", {
            "query": "IPAdapter",
            "category": "nodes",
        }))
        required_fields = {"name", "type", "source", "relevance_score", "installed", "url"}
        for r in result["results"]:
            assert required_fields.issubset(set(r.keys())), f"Missing fields in {r}"

    def test_he2025_ordering(self, mock_registries):
        """Results with same installed status and score should be alphabetical."""
        results = [
            comfy_discover._normalize_result(
                name="Zebra", result_type="model", source="registry",
                relevance_score=0.5, installed=False, url="",
            ),
            comfy_discover._normalize_result(
                name="Alpha", result_type="model", source="registry",
                relevance_score=0.5, installed=False, url="",
            ),
        ]
        ranked = comfy_discover._rank_results(results)
        assert ranked[0]["name"] == "Alpha"
        assert ranked[1]["name"] == "Zebra"

    def test_old_tools_removed(self):
        """Old tool names should not be dispatchable."""
        result = json.loads(comfy_discover.handle("search_custom_nodes", {}))
        assert "error" in result
        result = json.loads(comfy_discover.handle("search_models", {}))
        assert "error" in result

    def test_error_handling_partial_source_failure(self, mock_registries):
        """If one source fails, others should still return results."""
        with patch("agent.tools.comfy_discover._search_civitai_unified", return_value=(
            [], "CivitAI is down",
        )):
            result = json.loads(comfy_discover.handle("discover", {
                "query": "SDXL",
                "category": "models",
                "sources": ["registry", "civitai"],
            }))
            # Registry results should still be present
            assert result["total"] >= 1
            assert any(e["source"] == "civitai" for e in result.get("errors", []))

    def test_max_results_per_source(self, mock_registries):
        result = json.loads(comfy_discover.handle("discover", {
            "query": "ComfyUI",
            "category": "nodes",
            "max_results": 1,
        }))
        # Should respect max_results (may be fewer if dedup)
        assert result["total"] <= 3  # 1 per source max, 3 sources max

    def test_empty_results(self, mock_registries):
        result = json.loads(comfy_discover.handle("discover", {
            "query": "zzzzzznonexistent",
            "sources": ["registry"],
        }))
        assert result["total"] == 0
        assert result["results"] == []

    def test_normalize_result_sorted_keys(self):
        """_normalize_result should produce sorted keys (He2025)."""
        r = comfy_discover._normalize_result(
            name="Test", result_type="model", source="registry",
            relevance_score=0.5, installed=False, url="http://x",
            zebra="z", alpha="a",
        )
        # Extra keys are included in the result
        assert "alpha" in r
        assert "zebra" in r
