"""Tests for comfy_discover tools — mocked registries, no network needed."""

import json
import pytest
from pathlib import Path
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
    comfy_discover._cache["custom_nodes"] = None
    comfy_discover._cache["extension_map"] = None
    comfy_discover._cache["node_to_pack"] = None
    comfy_discover._cache["model_list"] = None
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
    def test_search_by_name(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "IPAdapter",
        }))
        assert result["total_matches"] >= 1
        assert any("IPAdapter" in r["title"] for r in result["results"])

    def test_search_by_description(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "video matting",
        }))
        assert result["total_matches"] >= 1

    def test_search_by_author(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "cubiq",
        }))
        assert result["total_matches"] >= 1
        assert result["results"][0]["author"] == "cubiq"

    def test_search_no_results(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "zzzznonexistent",
        }))
        assert result["total_matches"] == 0

    def test_search_by_node_type_exact(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "IPAdapterUnifiedLoader",
            "by": "node_type",
        }))
        assert result["match"] == "exact"
        assert "IPAdapter" in result["pack"]["title"]

    def test_search_by_node_type_fuzzy(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "IPAdapter",
            "by": "node_type",
        }))
        assert result["match"] == "fuzzy"
        assert len(result["results"]) >= 1

    def test_search_by_node_type_not_found(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "ZZZNonexistentNode",
            "by": "node_type",
        }))
        assert result["match"] == "none"

    def test_install_status(self, mock_registries, mock_custom_nodes_dir):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "IPAdapterUnifiedLoader",
            "by": "node_type",
        }))
        assert result["pack"]["installed"] is True

    def test_max_results(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_custom_nodes", {
            "query": "ComfyUI",
            "max_results": 2,
        }))
        assert len(result["results"]) <= 2

    def test_no_manager(self, tmp_path):
        """No ComfyUI Manager installed."""
        with patch.object(comfy_discover, "_MANAGER_DIR", tmp_path / "nonexistent"):
            result = json.loads(comfy_discover.handle("search_custom_nodes", {
                "query": "test",
            }))
            assert "error" in result


# ---------------------------------------------------------------------------
# Tests: search_models
# ---------------------------------------------------------------------------

class TestSearchModels:
    def test_search_registry_by_name(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_models", {
            "query": "FLUX",
        }))
        assert result["source"] == "registry"
        assert result["total_matches"] >= 1
        assert "FLUX" in result["results"][0]["name"]

    def test_search_registry_by_type(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_models", {
            "query": "SDXL",
            "model_type": "controlnet",
        }))
        assert all(r["type"] == "controlnet" for r in result["results"])

    def test_search_registry_no_match(self, mock_registries):
        result = json.loads(comfy_discover.handle("search_models", {
            "query": "zzzznonexistent",
        }))
        assert result["total_matches"] == 0

    def test_model_install_status(self, mock_registries, mock_models_dir):
        result = json.loads(comfy_discover.handle("search_models", {
            "query": "SDXL Base",
        }))
        sdxl = result["results"][0]
        assert sdxl["installed"] is True

    def test_model_not_installed(self, mock_registries, mock_models_dir):
        result = json.loads(comfy_discover.handle("search_models", {
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

            result = json.loads(comfy_discover.handle("search_models", {
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

            result = json.loads(comfy_discover.handle("search_models", {
                "query": "flux",
                "source": "huggingface",
            }))
            assert "error" in result

    def test_no_manager_for_registry(self, tmp_path):
        with patch.object(comfy_discover, "_MANAGER_DIR", tmp_path / "nonexistent"):
            result = json.loads(comfy_discover.handle("search_models", {
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

class TestRegistration:
    def test_tools_registered(self):
        from agent.tools import ALL_TOOLS
        names = {t["name"] for t in ALL_TOOLS}
        assert "search_custom_nodes" in names
        assert "search_models" in names
        assert "find_missing_nodes" in names
