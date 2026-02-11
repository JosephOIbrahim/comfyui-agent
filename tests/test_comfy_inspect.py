"""Tests for comfy_inspect tool — uses temp directories, no real ComfyUI needed."""

import json
import pytest
from unittest.mock import patch
from agent.tools import comfy_inspect


@pytest.fixture
def fake_custom_nodes(tmp_path):
    """Create a fake Custom_Nodes directory."""
    cn = tmp_path / "Custom_Nodes"
    cn.mkdir()

    # Real node pack
    pack1 = cn / "comfyui-impact-pack"
    pack1.mkdir()
    (pack1 / "__init__.py").write_text(
        "NODE_CLASS_MAPPINGS = {'ImpactNode': ImpactNode}\n"
        "NODE_DISPLAY_NAME_MAPPINGS = {'ImpactNode': 'Impact Node'}\n"
    )
    (pack1 / "README.md").write_text("# Impact Pack")
    (pack1 / "requirements.txt").write_text("numpy\n")

    # Another pack without NODE_CLASS_MAPPINGS
    pack2 = cn / "some-utils"
    pack2.mkdir()
    (pack2 / "__init__.py").write_text("# just a helper\nimport os\n")

    # Ignored: __pycache__, hidden, non-dir
    (cn / "__pycache__").mkdir()
    (cn / ".hidden").mkdir()
    (cn / ".hidden" / "x.py").write_text("")
    (cn / "__init__.py").write_text("")  # file, not dir

    return cn


@pytest.fixture
def fake_models(tmp_path):
    """Create a fake models directory."""
    models = tmp_path / "models"
    models.mkdir()

    ckpts = models / "checkpoints"
    ckpts.mkdir()
    (ckpts / "sd15.safetensors").write_bytes(b"\x00" * 1024)
    (ckpts / "sdxl.safetensors").write_bytes(b"\x00" * 2048)

    loras = models / "loras"
    loras.mkdir()
    (loras / "style.safetensors").write_bytes(b"\x00" * 512)

    # Empty dir
    (models / "controlnet").mkdir()

    return models


class TestListCustomNodes:
    def test_lists_packs(self, fake_custom_nodes):
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            result = json.loads(comfy_inspect.handle("list_custom_nodes", {}))
            assert result["count"] == 2
            names = [p["name"] for p in result["packs"]]
            assert "comfyui-impact-pack" in names
            assert "some-utils" in names

    def test_detects_node_registration(self, fake_custom_nodes):
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            result = json.loads(comfy_inspect.handle("list_custom_nodes", {}))
            impact = next(p for p in result["packs"] if p["name"] == "comfyui-impact-pack")
            assert impact["registers_nodes"] is True
            assert impact["has_readme"] is True
            assert impact["has_requirements"] is True

    def test_name_filter(self, fake_custom_nodes):
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            result = json.loads(
                comfy_inspect.handle("list_custom_nodes", {"name_filter": "impact"})
            )
            assert result["count"] == 1

    def test_missing_dir(self, tmp_path):
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", tmp_path / "nope"):
            result = json.loads(comfy_inspect.handle("list_custom_nodes", {}))
            assert "error" in result


class TestListModels:
    def test_lists_checkpoints(self, fake_models):
        with patch.object(comfy_inspect, "MODELS_DIR", fake_models):
            result = json.loads(
                comfy_inspect.handle("list_models", {"model_type": "checkpoints"})
            )
            assert result["count"] == 2
            names = [m["name"] for m in result["models"]]
            assert "sd15.safetensors" in names
            assert "sdxl.safetensors" in names

    def test_missing_type_suggests(self, fake_models):
        with patch.object(comfy_inspect, "MODELS_DIR", fake_models):
            result = json.loads(
                comfy_inspect.handle("list_models", {"model_type": "notreal"})
            )
            assert "error" in result
            assert "checkpoints" in result["available_types"]

    def test_empty_dir(self, fake_models):
        with patch.object(comfy_inspect, "MODELS_DIR", fake_models):
            result = json.loads(
                comfy_inspect.handle("list_models", {"model_type": "controlnet"})
            )
            assert result["count"] == 0


class TestGetModelsSummary:
    def test_summary(self, fake_models):
        with patch.object(comfy_inspect, "MODELS_DIR", fake_models):
            result = json.loads(comfy_inspect.handle("get_models_summary", {}))
            assert result["types"]["checkpoints"] == 2
            assert result["types"]["loras"] == 1
            # controlnet is empty, shouldn't appear
            assert "controlnet" not in result["types"]


class TestReadNodeSource:
    def test_reads_init(self, fake_custom_nodes):
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            result = json.loads(
                comfy_inspect.handle("read_node_source", {"node_pack": "comfyui-impact-pack"})
            )
            assert "NODE_CLASS_MAPPINGS" in result["source"]
            assert result["truncated"] is False

    def test_missing_pack_suggests(self, fake_custom_nodes):
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            result = json.loads(
                comfy_inspect.handle("read_node_source", {"node_pack": "impact"})
            )
            assert "error" in result
            assert "similar" in result
