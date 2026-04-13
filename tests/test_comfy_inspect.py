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

    def test_negative_max_lines_rejected(self, fake_custom_nodes):
        """Negative max_lines must return an error, not invert Python slice semantics (Cycle 25 fix)."""
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            result = json.loads(
                comfy_inspect.handle(
                    "read_node_source",
                    {"node_pack": "comfyui-impact-pack", "max_lines": -10},
                )
            )
            assert "error" in result
            assert "non-negative" in result["error"]

    def test_zero_max_lines(self, fake_custom_nodes):
        """max_lines=0 is a valid (though unusual) request — returns empty source."""
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            result = json.loads(
                comfy_inspect.handle(
                    "read_node_source",
                    {"node_pack": "comfyui-impact-pack", "max_lines": 0},
                )
            )
            assert "error" not in result
            assert result["source"] == ""


# ---------------------------------------------------------------------------
# Cycle 32: model_type required guard
# ---------------------------------------------------------------------------

class TestListModelsRequiredField:
    """list_models must validate model_type is present and non-empty."""

    def test_missing_model_type_returns_error(self, fake_custom_nodes, tmp_path):
        """Omitting model_type must return error, not KeyError."""
        from unittest.mock import patch
        with patch.object(comfy_inspect, "MODELS_DIR", tmp_path):
            result = json.loads(comfy_inspect.handle("list_models", {}))
        assert "error" in result
        assert "model_type" in result["error"]

    def test_none_model_type_returns_error(self, fake_custom_nodes, tmp_path):
        """Explicit None model_type must return error."""
        from unittest.mock import patch
        with patch.object(comfy_inspect, "MODELS_DIR", tmp_path):
            result = json.loads(comfy_inspect.handle("list_models", {"model_type": None}))
        assert "error" in result

    def test_empty_string_model_type_returns_error(self, fake_custom_nodes, tmp_path):
        """Empty string model_type must return error."""
        from unittest.mock import patch
        with patch.object(comfy_inspect, "MODELS_DIR", tmp_path):
            result = json.loads(comfy_inspect.handle("list_models", {"model_type": ""}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 56: MODELS_DIR existence guard when model_type not found
# ---------------------------------------------------------------------------

class TestListModelsModelsDirectoryGuard:
    """Cycle 56: _handle_list_models must not call iterdir() on a missing MODELS_DIR."""

    def test_missing_models_dir_returns_empty_available_types(self, tmp_path):
        """If MODELS_DIR itself doesn't exist, available_types is [] not a crash."""
        missing_dir = tmp_path / "does_not_exist"
        with patch.object(comfy_inspect, "MODELS_DIR", missing_dir):
            result = json.loads(
                comfy_inspect.handle("list_models", {"model_type": "checkpoints"})
            )
        assert "error" in result
        assert result.get("available_types") == []

    def test_existing_models_dir_lists_available_types(self, fake_models):
        """When MODELS_DIR exists but model_type is wrong, available_types lists real dirs."""
        with patch.object(comfy_inspect, "MODELS_DIR", fake_models):
            result = json.loads(
                comfy_inspect.handle("list_models", {"model_type": "notreal"})
            )
        assert "error" in result
        assert isinstance(result.get("available_types"), list)
        assert "checkpoints" in result["available_types"]

    def test_silent_exception_logs_debug_not_crashes(self, fake_custom_nodes):
        """Cycle 56: unreadable __init__.py must log.debug and continue, not crash."""
        with patch.object(comfy_inspect, "CUSTOM_NODES_DIR", fake_custom_nodes):
            with patch.object(comfy_inspect.log, "debug") as mock_debug:
                # Patch open to raise on the impact-pack __init__.py
                from pathlib import Path
                orig_read = Path.read_text

                def _raise_on_impact(self, *a, **kw):
                    if "comfyui-impact-pack" in str(self):
                        raise PermissionError("no read")
                    return orig_read(self, *a, **kw)

                with patch.object(Path, "read_text", _raise_on_impact):
                    result = json.loads(comfy_inspect.handle("list_custom_nodes", {}))

        # Should still return packs, just without registers_nodes for impact-pack
        assert result["count"] == 2
        impact = next(p for p in result["packs"] if p["name"] == "comfyui-impact-pack")
        assert "registers_nodes" not in impact
        # Debug was called with the error
        assert mock_debug.called
