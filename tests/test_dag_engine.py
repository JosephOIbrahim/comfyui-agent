"""CRUCIBLE tests for the Workflow Intelligence DAG (agent/stage/dag/).

Adversarial: tests edge cases, boundary conditions, determinism, and
graceful degradation when networkx is missing.
"""

from __future__ import annotations

import copy
import sys
from unittest.mock import patch

import pytest

from agent.stage.dag import (
    ComplexityLevel,
    ModelRequirements,
    OptimizationVector,
    ReadinessGrade,
    RiskLevel,
    WorkflowIntelligence,
    build_dag,
    evaluate_dag,
)
from agent.stage.dag.compute_complexity import compute_complexity
from agent.stage.dag.compute_model_reqs import compute_model_reqs
from agent.stage.dag.compute_risk import compute_risk
from agent.stage.dag.compute_readiness import compute_readiness
from agent.stage.dag.compute_tool_scope import compute_tool_scope

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMPTY_WORKFLOW: dict = {}

SIMPLE_WORKFLOW: dict = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a photo", "clip": ["1", 1]},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "bad", "clip": ["1", 1]},
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "test"},
    },
}

COMPLEX_WORKFLOW: dict = {
    **{str(i): {"class_type": f"Node_{i}", "inputs": {}} for i in range(1, 35)},
    "35": {
        "class_type": "LoraLoader",
        "inputs": {"model": ["1", 0], "lora_name": "detail.safetensors"},
    },
    "36": {
        "class_type": "LoraLoader",
        "inputs": {"model": ["35", 0], "lora_name": "style.safetensors"},
    },
    "37": {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": "control.safetensors"},
    },
}

# Flux workflow with FluxGuidance node
FLUX_WORKFLOW: dict = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "flux_dev.safetensors"},
    },
    "2": {"class_type": "FluxGuidance", "inputs": {"guidance": 1.0}},
    "3": {
        "class_type": "DualCLIPLoader",
        "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl.safetensors"},
    },
}

# Family-mismatch workflow: SD1.5 LoRA names but SDXL nodes
FAMILY_MISMATCH_WORKFLOW: dict = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "2": {"class_type": "FluxGuidance", "inputs": {}},
    "3": {"class_type": "EmptySD3LatentImage", "inputs": {}},
}

# High-res workflow that should trigger VRAM concerns
HIGHRES_WORKFLOW: dict = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "flux_dev.safetensors"},
    },
    "2": {"class_type": "FluxGuidance", "inputs": {}},
    "3": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 2048, "height": 2048, "batch_size": 1},
    },
}


# ---------------------------------------------------------------------------
# DAG construction tests
# ---------------------------------------------------------------------------


class TestBuildDag:
    def test_build_dag_returns_digraph(self):
        import networkx as nx

        dag = build_dag()
        assert isinstance(dag, nx.DiGraph)

    def test_build_dag_has_correct_nodes(self):
        dag = build_dag()
        expected = {"complexity", "model_reqs", "optimization", "risk", "readiness", "tool_scope"}
        assert set(dag.nodes) == expected

    def test_build_dag_has_correct_edges(self):
        dag = build_dag()
        edges = set(dag.edges)
        assert ("complexity", "optimization") in edges
        assert ("model_reqs", "optimization") in edges
        assert ("model_reqs", "risk") in edges
        assert ("risk", "readiness") in edges
        assert ("risk", "tool_scope") in edges
        assert ("readiness", "tool_scope") in edges

    def test_build_dag_is_acyclic(self):
        import networkx as nx

        dag = build_dag()
        assert nx.is_directed_acyclic_graph(dag)

    def test_build_dag_no_self_loops(self):
        dag = build_dag()
        for node in dag.nodes:
            assert not dag.has_edge(node, node), f"Self-loop on {node}"

    def test_build_dag_topological_sort_possible(self):
        import networkx as nx

        dag = build_dag()
        order = list(nx.topological_sort(dag))
        assert len(order) == 6


# ---------------------------------------------------------------------------
# DAG evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateDag:
    def test_evaluate_dag_empty_workflow(self):
        dag = build_dag()
        intel = evaluate_dag(dag, EMPTY_WORKFLOW)
        assert intel.complexity == ComplexityLevel.TRIVIAL
        assert intel.risk == RiskLevel.SAFE
        assert intel.readiness == ReadinessGrade.READY
        assert intel.evaluated is True

    def test_evaluate_dag_simple_workflow(self):
        dag = build_dag()
        intel = evaluate_dag(dag, SIMPLE_WORKFLOW)
        assert intel.complexity == ComplexityLevel.SIMPLE
        assert intel.evaluated is True
        assert isinstance(intel.model_requirements, ModelRequirements)

    def test_evaluate_dag_complex_workflow(self):
        dag = build_dag()
        intel = evaluate_dag(dag, COMPLEX_WORKFLOW)
        # 37 nodes should be COMPLEX (31-50)
        assert intel.complexity >= ComplexityLevel.COMPLEX
        assert intel.model_requirements.lora_count == 2
        assert intel.model_requirements.controlnet_present is True

    def test_evaluate_dag_returns_workflow_intelligence(self):
        dag = build_dag()
        intel = evaluate_dag(dag, SIMPLE_WORKFLOW)
        assert isinstance(intel, WorkflowIntelligence)
        assert hasattr(intel, "complexity")
        assert hasattr(intel, "model_requirements")
        assert hasattr(intel, "optimization")
        assert hasattr(intel, "risk")
        assert hasattr(intel, "readiness")
        assert hasattr(intel, "tool_scope")

    def test_evaluate_dag_is_pure(self):
        """Same inputs must produce identical outputs (determinism)."""
        dag = build_dag()
        wf = copy.deepcopy(SIMPLE_WORKFLOW)
        intel1 = evaluate_dag(dag, wf)
        intel2 = evaluate_dag(dag, wf)
        assert intel1.complexity == intel2.complexity
        assert intel1.risk == intel2.risk
        assert intel1.readiness == intel2.readiness
        assert intel1.model_requirements == intel2.model_requirements
        assert intel1.tool_scope == intel2.tool_scope

    def test_evaluate_dag_missing_nodes_risky(self):
        """Nodes not in registry should increase risk."""
        dag = build_dag()
        registry = {"CheckpointLoaderSimple": {}, "CLIPTextEncode": {}}
        intel = evaluate_dag(dag, SIMPLE_WORKFLOW, node_registry=registry)
        # KSampler, EmptyLatentImage, VAEDecode, SaveImage are missing
        assert intel.risk >= RiskLevel.RISKY

    def test_evaluate_dag_vram_overflow(self):
        """High VRAM requirement with tiny GPU should be BLOCKED."""
        dag = build_dag()
        stats = {"vram_total_gb": 2.0}
        intel = evaluate_dag(dag, HIGHRES_WORKFLOW, system_stats=stats)
        assert intel.risk == RiskLevel.BLOCKED

    def test_evaluate_dag_to_dict_serializable(self):
        """to_dict must return JSON-serializable data."""
        import json

        dag = build_dag()
        intel = evaluate_dag(dag, SIMPLE_WORKFLOW)
        d = intel.to_dict()
        serialized = json.dumps(d, sort_keys=True)
        assert isinstance(serialized, str)
        assert "SIMPLE" in serialized or "TRIVIAL" in serialized

    def test_evaluate_dag_workflow_intelligence_is_frozen(self):
        """WorkflowIntelligence is a frozen dataclass."""
        dag = build_dag()
        intel = evaluate_dag(dag, SIMPLE_WORKFLOW)
        with pytest.raises(AttributeError):
            intel.complexity = ComplexityLevel.EXTREME  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Compute complexity tests
# ---------------------------------------------------------------------------


class TestComputeComplexity:
    def test_trivial_threshold(self):
        """0-5 nodes -> TRIVIAL."""
        assert compute_complexity({}) == ComplexityLevel.TRIVIAL
        wf = {str(i): {"class_type": f"N{i}", "inputs": {}} for i in range(1, 4)}
        assert compute_complexity(wf) == ComplexityLevel.TRIVIAL

    def test_simple_threshold(self):
        """6-15 nodes -> SIMPLE."""
        wf = {str(i): {"class_type": f"N{i}", "inputs": {}} for i in range(1, 10)}
        assert compute_complexity(wf) == ComplexityLevel.SIMPLE

    def test_moderate_threshold(self):
        """16-30 nodes -> MODERATE."""
        wf = {str(i): {"class_type": f"N{i}", "inputs": {}} for i in range(1, 25)}
        assert compute_complexity(wf) == ComplexityLevel.MODERATE

    def test_complex_threshold(self):
        """31-50 nodes -> COMPLEX."""
        wf = {str(i): {"class_type": f"N{i}", "inputs": {}} for i in range(1, 40)}
        assert compute_complexity(wf) == ComplexityLevel.COMPLEX

    def test_extreme_threshold(self):
        """50+ nodes -> EXTREME."""
        wf = {str(i): {"class_type": f"N{i}", "inputs": {}} for i in range(1, 60)}
        assert compute_complexity(wf) == ComplexityLevel.EXTREME

    def test_boundary_5_nodes(self):
        """Exactly 5 nodes -> TRIVIAL."""
        wf = {str(i): {"class_type": f"N{i}", "inputs": {}} for i in range(1, 6)}
        assert compute_complexity(wf) == ComplexityLevel.TRIVIAL

    def test_boundary_6_nodes(self):
        """Exactly 6 nodes -> SIMPLE."""
        wf = {str(i): {"class_type": f"N{i}", "inputs": {}} for i in range(1, 7)}
        assert compute_complexity(wf) == ComplexityLevel.SIMPLE

    def test_high_branching_bumps_level(self):
        """High fan-out (>4) can push complexity up."""
        # 10 nodes (SIMPLE) but node 1 feeds 6 others -> should bump
        wf = {"1": {"class_type": "Source", "inputs": {}}}
        for i in range(2, 12):
            wf[str(i)] = {
                "class_type": f"N{i}",
                "inputs": {"data": ["1", 0]},
            }
        level = compute_complexity(wf)
        # With 11 nodes and branching > 4, should be MODERATE
        assert level >= ComplexityLevel.MODERATE


# ---------------------------------------------------------------------------
# Compute model requirements tests
# ---------------------------------------------------------------------------


class TestComputeModelReqs:
    def test_empty_workflow(self):
        reqs = compute_model_reqs({})
        assert reqs.checkpoint_family == "unknown"
        assert reqs.lora_count == 0
        assert reqs.controlnet_present is False

    def test_sdxl_detection(self):
        reqs = compute_model_reqs(SIMPLE_WORKFLOW)
        assert reqs.checkpoint_family == "sdxl"
        assert reqs.resolution_band == "1024"

    def test_flux_detection(self):
        reqs = compute_model_reqs(FLUX_WORKFLOW)
        assert reqs.checkpoint_family == "flux"

    def test_lora_count(self):
        reqs = compute_model_reqs(COMPLEX_WORKFLOW)
        assert reqs.lora_count == 2

    def test_controlnet_detection(self):
        reqs = compute_model_reqs(COMPLEX_WORKFLOW)
        assert reqs.controlnet_present is True

    def test_sampler_detection(self):
        reqs = compute_model_reqs(SIMPLE_WORKFLOW)
        assert reqs.sampler_class == "standard"

    def test_vram_estimate_positive(self):
        reqs = compute_model_reqs(SIMPLE_WORKFLOW)
        assert reqs.vram_estimate_gb > 0.0

    def test_model_requirements_frozen(self):
        reqs = compute_model_reqs(SIMPLE_WORKFLOW)
        with pytest.raises(AttributeError):
            reqs.checkpoint_family = "hacked"  # type: ignore[misc]

    def test_describe_string(self):
        reqs = compute_model_reqs(SIMPLE_WORKFLOW)
        desc = reqs.describe()
        assert "sdxl" in desc
        assert isinstance(desc, str)


# ---------------------------------------------------------------------------
# Compute risk tests
# ---------------------------------------------------------------------------


class TestComputeRisk:
    def test_empty_workflow_safe(self):
        assert compute_risk({}, ModelRequirements()) == RiskLevel.SAFE

    def test_family_mismatch_blocked(self):
        """Mixing flux and sd3 family indicators -> BLOCKED."""
        risk = compute_risk(FAMILY_MISMATCH_WORKFLOW, ModelRequirements())
        assert risk == RiskLevel.BLOCKED

    def test_missing_critical_node_blocked(self):
        registry = {"CLIPTextEncode": {}, "EmptyLatentImage": {}}
        # CheckpointLoaderSimple is critical prefix and missing
        risk = compute_risk(
            SIMPLE_WORKFLOW,
            ModelRequirements(),
            node_registry=registry,
        )
        assert risk == RiskLevel.BLOCKED

    def test_missing_noncritical_node_risky(self):
        # Registry has loaders/samplers but not SaveImage
        registry = {
            "CheckpointLoaderSimple": {},
            "CLIPTextEncode": {},
            "EmptyLatentImage": {},
            "KSampler": {},
            "VAEDecode": {},
        }
        risk = compute_risk(
            SIMPLE_WORKFLOW,
            ModelRequirements(),
            node_registry=registry,
        )
        assert risk >= RiskLevel.RISKY

    def test_vram_overflow_blocked(self):
        reqs = ModelRequirements(vram_estimate_gb=20.0)
        stats = {"vram_total_gb": 8.0}
        risk = compute_risk(SIMPLE_WORKFLOW, reqs, system_stats=stats)
        assert risk == RiskLevel.BLOCKED

    def test_high_lora_count_caution(self):
        reqs = ModelRequirements(lora_count=5)
        risk = compute_risk(SIMPLE_WORKFLOW, reqs)
        assert risk >= RiskLevel.CAUTION

    def test_no_registry_skips_node_check(self):
        risk = compute_risk(SIMPLE_WORKFLOW, ModelRequirements())
        # Without registry, no node missing check
        assert risk == RiskLevel.SAFE


# ---------------------------------------------------------------------------
# Compute readiness tests
# ---------------------------------------------------------------------------


class TestComputeReadiness:
    def test_blocked_risk_propagates(self):
        assert compute_readiness(RiskLevel.BLOCKED) == ReadinessGrade.BLOCKED

    def test_risky_needs_fix(self):
        assert compute_readiness(RiskLevel.RISKY) == ReadinessGrade.NEEDS_FIX

    def test_missing_nodes_needs_provision(self):
        result = compute_readiness(RiskLevel.SAFE, missing_nodes=["CustomNode"])
        assert result == ReadinessGrade.NEEDS_PROVISION

    def test_safe_and_complete_ready(self):
        assert compute_readiness(RiskLevel.SAFE) == ReadinessGrade.READY

    def test_caution_with_no_missing_ready(self):
        assert compute_readiness(RiskLevel.CAUTION) == ReadinessGrade.READY


# ---------------------------------------------------------------------------
# Compute tool scope tests
# ---------------------------------------------------------------------------


class TestComputeToolScope:
    def test_no_workflow_offers_load(self):
        scope = compute_tool_scope(workflow_loaded=False)
        assert "load" in scope
        assert "template" in scope
        assert "session" in scope
        assert "execute" not in scope

    def test_loaded_workflow_offers_edit(self):
        scope = compute_tool_scope(workflow_loaded=True)
        assert "edit" in scope
        assert "validate" in scope

    def test_validated_ready_offers_execute(self):
        scope = compute_tool_scope(
            workflow_loaded=True,
            workflow_validated=True,
            risk=RiskLevel.SAFE,
            readiness=ReadinessGrade.READY,
        )
        assert "execute" in scope
        assert "optimize" in scope

    def test_executed_offers_verify(self):
        scope = compute_tool_scope(
            workflow_loaded=True,
            workflow_validated=True,
            workflow_executed=True,
            risk=RiskLevel.SAFE,
            readiness=ReadinessGrade.READY,
        )
        assert "verify" in scope

    def test_risky_workflow_offers_discover(self):
        scope = compute_tool_scope(
            workflow_loaded=True,
            risk=RiskLevel.RISKY,
            readiness=ReadinessGrade.NEEDS_FIX,
        )
        assert "discover" in scope
        assert "provision" in scope

    def test_tool_scope_per_phase(self):
        """Verify tool recommendations change with workflow lifecycle phase."""
        scope_empty = compute_tool_scope(workflow_loaded=False)
        scope_loaded = compute_tool_scope(workflow_loaded=True)
        scope_validated = compute_tool_scope(
            workflow_loaded=True,
            workflow_validated=True,
            readiness=ReadinessGrade.READY,
        )
        # Each phase should add tools, not remove
        assert scope_empty != scope_loaded
        assert len(scope_validated) >= len(scope_loaded)


# ---------------------------------------------------------------------------
# Graceful degradation without networkx
# ---------------------------------------------------------------------------


class TestWithoutNetworkx:
    def test_build_dag_without_networkx_raises(self):
        """build_dag should raise RuntimeError when networkx is absent."""
        with patch.dict(sys.modules, {"networkx": None}):
            # Force reimport of the engine to pick up HAS_NX = False
            import importlib
            import agent.stage.dag.engine as engine_mod

            orig_has_nx = engine_mod.HAS_NX
            engine_mod.HAS_NX = False
            try:
                with pytest.raises(RuntimeError, match="networkx"):
                    engine_mod.build_dag()
            finally:
                engine_mod.HAS_NX = orig_has_nx

    def test_evaluate_dag_without_networkx_raises(self):
        import agent.stage.dag.engine as engine_mod

        orig_has_nx = engine_mod.HAS_NX
        engine_mod.HAS_NX = False
        try:
            with pytest.raises(RuntimeError, match="networkx"):
                engine_mod.evaluate_dag(None, {})
        finally:
            engine_mod.HAS_NX = orig_has_nx
