"""Graft E demo tests — verify 3D knowledge, discovery, and workflow parsing.

These tests validate the end-to-end integration of Grafts A-D:
- 3D data type recognition in workflow parsing (Graft A)
- Partner node discovery and ranking (Graft B)
- Splat-to-mesh knowledge loading (Graft C)
- Viewport tool discovery (Graft D)
- Camera pipeline knowledge (Phase 4)
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

from agent.system_prompt import _detect_relevant_knowledge
from agent.tools import workflow_parse, comfy_discover

# ---------------------------------------------------------------------------
# Paths to fixture workflows
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(name: str) -> dict:
    """Load a fixture JSON file."""
    path = FIXTURES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Graft C: Splat-to-mesh discovery
# ---------------------------------------------------------------------------

class TestSplatToMeshDiscovery:
    """Verify that splat-to-mesh queries trigger the right knowledge."""

    def test_splat_to_mesh_triggers_partner_knowledge(self):
        """'splat to mesh' should trigger 3d_partner_nodes knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "how do I convert a gaussian splat to mesh?"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_partner_nodes" in triggers

    def test_splat_to_mesh_triggers_workflow_knowledge(self):
        """'splat to mesh' should also trigger 3d_workflows knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "convert gaussian splat to exportable mesh"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers

    def test_marching_cubes_query_triggers_knowledge(self):
        """Marching cubes is the primary conversion method — should trigger knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "use marching cubes to get GLB from splat"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers


# ---------------------------------------------------------------------------
# Graft D: Viewport tool discovery
# ---------------------------------------------------------------------------

class TestControlNet3DToolDiscovery:
    """Verify that 3D viewport/control queries trigger knowledge."""

    def test_vnccs_triggers_3d_knowledge(self):
        """VNCCS query should trigger 3d_workflows knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "set up VNCCS for character posing"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers

    def test_action_director_triggers_3d_knowledge(self):
        """Action Director query should trigger 3d_workflows knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "use action director for camera angles"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers

    def test_depth_map_from_3d_triggers_knowledge(self):
        """'depth map from 3d' should trigger 3d_workflows."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "render depth map from 3d model for controlnet"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_workflows" in triggers


# ---------------------------------------------------------------------------
# Graft B: Partner node comparison
# ---------------------------------------------------------------------------

class TestPartnerNodeComparison:
    """Verify partner nodes are ranked above community in discover results."""

    def test_partner_ranked_above_community(self):
        """Partner nodes should sort before community nodes."""
        # Create mixed results with partner and community entries
        results = [
            {
                "name": "ComfyUI-3D-Pack",
                "source_tier": "community",
                "relevance_score": 0.9,
                "installed": False,
            },
            {
                "name": "Hunyuan3D",
                "source_tier": "partner",
                "relevance_score": 0.8,
                "installed": False,
            },
        ]
        ranked = comfy_discover._rank_results(results)
        assert ranked[0]["name"] == "Hunyuan3D"
        assert ranked[1]["name"] == "ComfyUI-3D-Pack"

    def test_get_source_tier_for_partner(self):
        """Known partner URLs should return 'partner' tier."""
        tier = comfy_discover._get_source_tier(
            "Hunyuan3D",
            "https://github.com/Tencent/Hunyuan3D-2",
        )
        assert tier == "partner"

    def test_get_source_tier_for_community(self):
        """Unknown URLs should return 'community' tier."""
        tier = comfy_discover._get_source_tier(
            "SomeRandomPack",
            "https://github.com/someone/SomeRandomPack",
        )
        assert tier == "community"

    def test_all_partners_registered(self):
        """All 4 partner nodes should be in the PARTNER_NODES registry."""
        assert "Hunyuan3D" in comfy_discover.PARTNER_NODES
        assert "Meshy" in comfy_discover.PARTNER_NODES
        assert "Tripo" in comfy_discover.PARTNER_NODES
        assert "Rodin" in comfy_discover.PARTNER_NODES


# ---------------------------------------------------------------------------
# Graft A: 3D workflow parsing (UNDERSTAND layer)
# ---------------------------------------------------------------------------

class TestSplatToMeshWorkflowParse:
    """Verify UNDERSTAND layer categorizes 3D nodes correctly."""

    def test_splat_workflow_categorizes_3d_nodes(self):
        """Load3DGaussian, MarchingCubes should be categorized as 3D Processing."""
        data = _load_fixture("workflow_splat_to_mesh.json")
        nodes, fmt = workflow_parse._extract_api_format(data)
        connections = workflow_parse._trace_connections(nodes)
        summary = workflow_parse._build_summary(nodes, connections, fmt)
        assert "3D Processing" in summary

    def test_splat_workflow_traces_connections(self):
        """Connections in the splat-to-mesh pipeline should be traced."""
        data = _load_fixture("workflow_splat_to_mesh.json")
        nodes, fmt = workflow_parse._extract_api_format(data)
        connections = workflow_parse._trace_connections(nodes)
        # Load3DGaussian -> MarchingCubes
        assert any(
            c["from_class"] == "Load3DGaussian" and c["to_class"] == "MarchingCubes"
            for c in connections
        )

    def test_controlnet_3d_workflow_mixed_categories(self):
        """ControlNet 3D workflow has both 3D and 2D pipeline stages."""
        data = _load_fixture("workflow_controlnet_3d.json")
        nodes, fmt = workflow_parse._extract_api_format(data)
        connections = workflow_parse._trace_connections(nodes)
        summary = workflow_parse._build_summary(nodes, connections, fmt)
        # Should have both 3D and standard pipeline stages
        assert "3D Processing" in summary or "Load3D" in summary
        assert "Sampling" in summary

    def test_partner_comparison_workflow_identifies_3d_generators(self):
        """Partner comparison workflow should identify all 3D generator nodes."""
        data = _load_fixture("workflow_partner_comparison.json")
        nodes, fmt = workflow_parse._extract_api_format(data)
        connections = workflow_parse._trace_connections(nodes)
        summary = workflow_parse._build_summary(nodes, connections, fmt)
        assert "3D Processing" in summary


# ---------------------------------------------------------------------------
# Phase 4: Camera pipeline knowledge
# ---------------------------------------------------------------------------

class TestCameraPipelineKnowledge:
    """Verify camera pipeline triggers are wired up."""

    def test_camera_control_triggers_knowledge(self):
        """'camera control' should trigger 3d_camera_pipeline knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "set up camera control for my 3D scene"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_camera_pipeline" in triggers

    def test_load3d_camera_triggers_knowledge(self):
        """'LOAD3D_CAMERA' should trigger camera pipeline knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "what does the LOAD3D_CAMERA output contain?"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_camera_pipeline" in triggers

    def test_focal_length_triggers_knowledge(self):
        """'focal length' should trigger camera pipeline knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "set focal length to 50mm for portrait framing"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_camera_pipeline" in triggers

    def test_cinematic_triggers_knowledge(self):
        """'cinematic' should trigger camera pipeline knowledge."""
        ctx = {
            "workflow": {},
            "notes": [{"text": "cinematic camera setup for my scene"}],
        }
        triggers = _detect_relevant_knowledge(ctx)
        assert "3d_camera_pipeline" in triggers

    def test_camera_pipeline_knowledge_file_exists(self):
        """The 3d_camera_pipeline.md knowledge file should exist."""
        knowledge_dir = Path(__file__).parent.parent / "agent" / "knowledge"
        assert (knowledge_dir / "3d_camera_pipeline.md").exists()

    def test_load3d_camera_in_type_system(self):
        """LOAD3D_CAMERA should be mentioned in comfyui_core.md."""
        knowledge_dir = Path(__file__).parent.parent / "agent" / "knowledge"
        core_content = (knowledge_dir / "comfyui_core.md").read_text(encoding="utf-8")
        assert "LOAD3D_CAMERA" in core_content
