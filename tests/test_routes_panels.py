"""Tests for panel builder functions in ui/server/routes.py.

Covers: _classify_slot, _extract_flow_chain, _panel_workflow_analysis,
_panel_discovery, _build_panel_for_tool.
"""

import json
import sys
from pathlib import Path

import pytest

# UI tests require aiohttp — skip gracefully when it's not installed
pytest.importorskip("aiohttp", reason="UI tests require aiohttp (pip install aiohttp)")

# Ensure ui package is importable
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.server.routes import (  # noqa: E402
    _classify_slot,
    _extract_flow_chain,
    _panel_workflow_analysis,
    _panel_discovery,
    _build_panel_for_tool,
)


# -- _classify_slot -----------------------------------------------------------

class TestClassifySlot:
    def test_model_keywords(self):
        assert _classify_slot("CheckpointLoaderSimple") == "model"
        assert _classify_slot("LoraLoader") == "model"

    def test_clip_keywords(self):
        assert _classify_slot("CLIPTextEncode") == "clip"

    def test_sampler_keywords(self):
        assert _classify_slot("KSampler") == "sampler"
        assert _classify_slot("KSamplerAdvanced") == "sampler"

    def test_vae_keyword(self):
        assert _classify_slot("VAEDecode") == "vae"
        assert _classify_slot("VAEEncode") == "vae"

    def test_image_keyword(self):
        assert _classify_slot("SaveImage") == "image"
        assert _classify_slot("PreviewImage") == "image"
        assert _classify_slot("LoadImage") == "image"

    def test_conditioning_keyword(self):
        assert _classify_slot("ConditioningCombine") == "conditioning"

    def test_controlnet_keyword(self):
        assert _classify_slot("ControlNetApply") == "controlnet"

    def test_latent_keyword(self):
        assert _classify_slot("EmptyLatentImage") == "latent"

    def test_fallback(self):
        assert _classify_slot("SomeUnknownNode") == "model"

    def test_case_insensitive(self):
        assert _classify_slot("cliploader") == "clip"
        assert _classify_slot("VAEDECODE") == "vae"


# -- _extract_flow_chain ------------------------------------------------------

SAMPLE_NODES = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "model.safetensors"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "hello", "clip": ["1", 1]},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {"model": ["1", 0], "positive": ["2", 0]},
    },
    "4": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["1", 2]},
    },
    "5": {
        "class_type": "SaveImage",
        "inputs": {"images": ["4", 0]},
    },
}


class TestExtractFlowChain:
    def test_basic_chain(self):
        chain = _extract_flow_chain(SAMPLE_NODES)
        assert len(chain) >= 3
        # First node should be the root (CheckpointLoaderSimple)
        assert chain[0]["label"] == "CheckpointLoaderSimple"
        assert chain[0]["slotType"] == "model"

    def test_max_nodes_limit(self):
        chain = _extract_flow_chain(SAMPLE_NODES, max_nodes=2)
        assert len(chain) <= 2

    def test_empty_nodes(self):
        chain = _extract_flow_chain({})
        assert chain == []

    def test_single_node(self):
        single = {"1": {"class_type": "KSampler", "inputs": {}}}
        chain = _extract_flow_chain(single)
        assert len(chain) == 1
        assert chain[0]["label"] == "KSampler"

    def test_all_nodes_have_slot_type(self):
        chain = _extract_flow_chain(SAMPLE_NODES)
        for node in chain:
            assert "label" in node
            assert "slotType" in node


# -- _panel_workflow_analysis -------------------------------------------------

class TestPanelWorkflowAnalysis:
    def test_full_workflow(self):
        panel = _panel_workflow_analysis({"nodes": SAMPLE_NODES})
        assert panel is not None
        assert panel["type"] == "workflow_analysis"
        assert "header" in panel
        assert panel["header"]["label"] == "workflow \u00b7 analysis"
        assert len(panel["sections"]) > 0

    def test_header_stats(self):
        panel = _panel_workflow_analysis({"nodes": SAMPLE_NODES})
        stats = panel["header"]["stats"]
        # Should have node count
        assert any(s["label"] == "nodes" for s in stats)
        node_stat = next(s for s in stats if s["label"] == "nodes")
        assert node_stat["value"] == "5"

    def test_signal_flow_section(self):
        panel = _panel_workflow_analysis({"nodes": SAMPLE_NODES})
        flow_sections = [s for s in panel["sections"] if s["title"] == "Signal Flow"]
        assert len(flow_sections) == 1
        assert flow_sections[0]["defaultOpen"] is True
        assert flow_sections[0]["type"] == "flow_chain"

    def test_node_group_sections_collapsed(self):
        panel = _panel_workflow_analysis({"nodes": SAMPLE_NODES})
        non_flow = [s for s in panel["sections"] if s["title"] != "Signal Flow"]
        for sec in non_flow:
            assert sec["defaultOpen"] is False

    def test_footer_actions(self):
        panel = _panel_workflow_analysis({"nodes": SAMPLE_NODES})
        assert panel["footer"]["actions"][0]["label"] == "Modify"
        assert panel["footer"]["actions"][1]["label"] == "Run"

    def test_minimal_result(self):
        """Test with node_count but no nodes dict (header-only panel)."""
        panel = _panel_workflow_analysis({
            "node_count": 10,
            "connection_count": 8,
            "format": "api",
            "loaded_path": "/path/to/workflow.json",
        })
        assert panel is not None
        assert panel["header"]["title"] == "workflow.json"
        assert "10" in panel["header"]["stats"][0]["value"]

    def test_none_on_string_nodes(self):
        result = _panel_workflow_analysis({"nodes": "not a dict"})
        assert result is None

    def test_workflow_type_detection(self):
        """Detect video workflow type."""
        video_nodes = {
            "1": {"class_type": "AnimateDiffLoader", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        }
        panel = _panel_workflow_analysis({"nodes": video_nodes})
        assert panel["header"]["badge"] == "video gen"


# -- _panel_discovery ---------------------------------------------------------

class TestPanelDiscovery:
    def test_basic_discovery(self):
        result = {
            "results": [
                {"name": "SDXL Lightning", "description": "Fast model", "source": "civitai"},
                {"name": "Flux.1", "description": "Diffusion", "source": "huggingface"},
            ],
        }
        panel = _panel_discovery(result, {"query": "fast models", "category": "models"})
        assert panel is not None
        assert panel["type"] == "discovery"
        assert panel["header"]["badge"] == "2"
        assert "fast models" in panel["header"]["title"]

    def test_max_five_results(self):
        results = [{"name": f"model_{i}", "description": "test"} for i in range(10)]
        panel = _panel_discovery({"results": results}, {"query": "test"})
        assert len(panel["sections"]) == 5

    def test_first_result_open(self):
        result = {
            "results": [
                {"name": "A", "description": "first"},
                {"name": "B", "description": "second"},
            ],
        }
        panel = _panel_discovery(result, {"query": "test"})
        assert panel["sections"][0]["defaultOpen"] is True
        assert panel["sections"][1]["defaultOpen"] is False

    def test_empty_results(self):
        panel = _panel_discovery({"results": []}, {"query": "nothing"})
        assert panel is None

    def test_installed_tag(self):
        result = {
            "results": [
                {"name": "MyNode", "description": "A node", "installed": True},
            ],
        }
        panel = _panel_discovery(result, {"query": "test"})
        # Installed items should have the installed tag
        sec = panel["sections"][0]
        if sec["type"] == "slot_tags":
            installed_tags = [t for t in sec["data"]["tags"] if t["label"] == "installed"]
            assert len(installed_tags) == 1


# -- _build_panel_for_tool ----------------------------------------------------

class TestBuildPanelForTool:
    def test_unsupported_tool(self):
        assert _build_panel_for_tool("get_system_stats", {}, '{"cpu": 50}') is None

    def test_load_workflow(self):
        result = json.dumps({"nodes": SAMPLE_NODES})
        panel = _build_panel_for_tool("load_workflow", {}, result)
        assert panel is not None
        assert panel["type"] == "workflow_analysis"

    def test_discover(self):
        result = json.dumps({
            "results": [{"name": "Test", "description": "test"}],
        })
        panel = _build_panel_for_tool("discover", {"query": "test"}, result)
        assert panel is not None
        assert panel["type"] == "discovery"

    def test_error_result(self):
        result = json.dumps({"error": "Something failed"})
        panel = _build_panel_for_tool("load_workflow", {}, result)
        assert panel is None

    def test_invalid_json(self):
        panel = _build_panel_for_tool("load_workflow", {}, "not json")
        assert panel is None

    def test_validate_workflow(self):
        result = json.dumps({"nodes": SAMPLE_NODES})
        panel = _build_panel_for_tool("validate_workflow", {}, result)
        assert panel is not None
