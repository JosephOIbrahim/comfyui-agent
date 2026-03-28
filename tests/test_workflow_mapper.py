"""Tests for workflow JSON <-> USD prim mapper.

Core test: round-trip fidelity. Load JSON -> prims -> JSON -> compare.
"""

import json

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core not installed")

from agent.stage.cognitive_stage import CognitiveWorkflowStage
from agent.stage.workflow_mapper import (
    prims_to_workflow_json,
    workflow_json_to_prims,
)

# Minimal txt2img workflow (from templates/txt2img_sd15.json structure)
MINIMAL_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a beautiful landscape",
            "clip": ["1", 1],
        },
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
        },
    },
}

# Full SD1.5 txt2img workflow matching the real template
FULL_SD15_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a beautiful landscape, mountains, sunset, detailed",
            "clip": ["1", 1],
        },
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "ugly, blurry, low quality",
            "clip": ["1", 1],
        },
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["6", 0]},
    },
}


def _compare_workflows(original: dict, reconstructed: dict) -> None:
    """Assert two workflow JSONs are functionally identical."""
    assert set(original.keys()) == set(reconstructed.keys()), (
        f"Node ID mismatch: {set(original.keys())} vs {set(reconstructed.keys())}"
    )

    for node_id in original:
        orig_node = original[node_id]
        recon_node = reconstructed[node_id]

        assert orig_node["class_type"] == recon_node["class_type"], (
            f"Node {node_id}: class_type mismatch"
        )

        orig_inputs = orig_node.get("inputs", {})
        recon_inputs = recon_node.get("inputs", {})

        assert set(orig_inputs.keys()) == set(recon_inputs.keys()), (
            f"Node {node_id}: input keys mismatch "
            f"{set(orig_inputs.keys())} vs {set(recon_inputs.keys())}"
        )

        for key in orig_inputs:
            orig_val = orig_inputs[key]
            recon_val = recon_inputs[key]

            if isinstance(orig_val, list):
                # Connection — compare as lists
                assert isinstance(recon_val, list), (
                    f"Node {node_id}.{key}: expected connection list, "
                    f"got {type(recon_val)}"
                )
                assert str(orig_val[0]) == str(recon_val[0]), (
                    f"Node {node_id}.{key}: source node mismatch"
                )
                assert int(orig_val[1]) == int(recon_val[1]), (
                    f"Node {node_id}.{key}: output slot mismatch"
                )
            elif isinstance(orig_val, float):
                assert abs(orig_val - recon_val) < 1e-10, (
                    f"Node {node_id}.{key}: float mismatch "
                    f"{orig_val} vs {recon_val}"
                )
            else:
                assert orig_val == recon_val, (
                    f"Node {node_id}.{key}: value mismatch "
                    f"{orig_val!r} vs {recon_val!r}"
                )


class TestWorkflowToprims:
    """JSON -> USD prims conversion."""

    def test_creates_workflow_prim(self):
        cws = CognitiveWorkflowStage()
        base = workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        assert base == "/workflows/test"
        assert cws.prim_exists("/workflows/test")

    def test_creates_node_prims(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        assert cws.prim_exists("/workflows/test/nodes/node_1")
        assert cws.prim_exists("/workflows/test/nodes/node_2")
        assert cws.prim_exists("/workflows/test/nodes/node_3")

    def test_stores_class_type(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        assert (
            cws.read("/workflows/test/nodes/node_1", "class_type")
            == "CheckpointLoaderSimple"
        )
        assert (
            cws.read("/workflows/test/nodes/node_3", "class_type")
            == "KSampler"
        )

    def test_stores_node_id(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        assert cws.read("/workflows/test/nodes/node_1", "node_id") == "1"
        assert cws.read("/workflows/test/nodes/node_3", "node_id") == "3"

    def test_stores_literal_inputs(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        assert (
            cws.read("/workflows/test/nodes/node_3", "input:steps") == 20
        )
        assert (
            abs(cws.read("/workflows/test/nodes/node_3", "input:cfg") - 7.0)
            < 1e-10
        )
        assert (
            cws.read("/workflows/test/nodes/node_3", "input:sampler_name")
            == "euler"
        )

    def test_stores_string_inputs(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        assert (
            cws.read("/workflows/test/nodes/node_2", "input:text")
            == "a beautiful landscape"
        )

    def test_stores_node_count(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        assert cws.read("/workflows/test", "node_count") == 3

    def test_stores_connections_as_relationships(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")

        # KSampler (node_3) has connection: model -> node_1
        node3 = cws.stage.GetPrimAtPath("/workflows/test/nodes/node_3")
        rel = node3.GetRelationship("conn:model")
        assert rel.IsValid()
        targets = rel.GetTargets()
        assert len(targets) == 1
        assert str(targets[0]) == "/workflows/test/nodes/node_1"

    def test_stores_connection_slot(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        # CLIPTextEncode (node_2): clip -> node_1, slot 1
        slot = cws.read("/workflows/test/nodes/node_2", "conn:clip:slot")
        assert slot == 1


class TestPrimsToWorkflow:
    """USD prims -> JSON conversion."""

    def test_reconstructs_minimal_workflow(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "test")
        result = prims_to_workflow_json(cws, "test")
        _compare_workflows(MINIMAL_WORKFLOW, result)

    def test_reconstructs_full_sd15_workflow(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, FULL_SD15_WORKFLOW, "sd15")
        result = prims_to_workflow_json(cws, "sd15")
        _compare_workflows(FULL_SD15_WORKFLOW, result)

    def test_empty_workflow(self):
        cws = CognitiveWorkflowStage()
        result = prims_to_workflow_json(cws, "nonexistent")
        assert result == {}

    def test_preserves_node_ids_exactly(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, FULL_SD15_WORKFLOW, "test")
        result = prims_to_workflow_json(cws, "test")
        assert set(result.keys()) == {"1", "2", "3", "4", "5", "6", "7"}

    def test_preserves_all_value_types(self):
        """int, float, str, bool all survive round-trip."""
        workflow = {
            "1": {
                "class_type": "TestNode",
                "inputs": {
                    "int_val": 42,
                    "float_val": 3.14,
                    "str_val": "hello world",
                    "bool_val": True,
                },
            },
        }
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, workflow, "types")
        result = prims_to_workflow_json(cws, "types")

        inputs = result["1"]["inputs"]
        assert inputs["int_val"] == 42
        assert isinstance(inputs["int_val"], int)
        assert abs(inputs["float_val"] - 3.14) < 1e-10
        assert inputs["str_val"] == "hello world"
        assert inputs["bool_val"] is True


class TestRoundTrip:
    """Full round-trip fidelity tests."""

    def test_minimal_roundtrip(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "rt")
        result = prims_to_workflow_json(cws, "rt")
        _compare_workflows(MINIMAL_WORKFLOW, result)

    def test_full_sd15_roundtrip(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, FULL_SD15_WORKFLOW, "rt")
        result = prims_to_workflow_json(cws, "rt")
        _compare_workflows(FULL_SD15_WORKFLOW, result)

    def test_roundtrip_with_real_template(self):
        """Load actual template file and round-trip it."""
        import pathlib

        template_path = (
            pathlib.Path(__file__).parent.parent
            / "agent"
            / "templates"
            / "txt2img_sd15.json"
        )
        if not template_path.exists():
            pytest.skip("Template file not found")

        with open(template_path) as f:
            workflow = json.load(f)

        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, workflow, "template")
        result = prims_to_workflow_json(cws, "template")
        _compare_workflows(workflow, result)

    def test_roundtrip_preserves_connections(self):
        """Every connection [node_id, slot] survives round-trip."""
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, FULL_SD15_WORKFLOW, "conn")
        result = prims_to_workflow_json(cws, "conn")

        # Verify all connections in KSampler node
        ks_inputs = result["5"]["inputs"]
        assert ks_inputs["model"] == ["1", 0]
        assert ks_inputs["positive"] == ["2", 0]
        assert ks_inputs["negative"] == ["3", 0]
        assert ks_inputs["latent_image"] == ["4", 0]

        # Verify VAEDecode connections
        vae_inputs = result["6"]["inputs"]
        assert vae_inputs["samples"] == ["5", 0]
        assert vae_inputs["vae"] == ["1", 2]

    def test_roundtrip_after_agent_delta(self):
        """Agent delta modifies a value; round-trip reflects it."""
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "delta")

        # Agent changes steps from 20 to 50
        cws.add_agent_delta("forge", {
            "/workflows/delta/nodes/node_3:input:steps": 50,
        })

        result = prims_to_workflow_json(cws, "delta")
        assert result["3"]["inputs"]["steps"] == 50

    def test_roundtrip_after_rollback(self):
        """After rollback, round-trip shows base values."""
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "rb")

        cws.add_agent_delta("forge", {
            "/workflows/rb/nodes/node_3:input:steps": 50,
        })
        cws.rollback_to(1)

        result = prims_to_workflow_json(cws, "rb")
        assert result["3"]["inputs"]["steps"] == 20  # Base value restored


class TestMultipleWorkflows:
    """Multiple workflows coexist in the same stage."""

    def test_two_workflows_independent(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "wf_a")
        workflow_json_to_prims(cws, FULL_SD15_WORKFLOW, "wf_b")

        result_a = prims_to_workflow_json(cws, "wf_a")
        result_b = prims_to_workflow_json(cws, "wf_b")

        _compare_workflows(MINIMAL_WORKFLOW, result_a)
        _compare_workflows(FULL_SD15_WORKFLOW, result_b)

    def test_delta_affects_only_target_workflow(self):
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "wf_a")
        workflow_json_to_prims(cws, MINIMAL_WORKFLOW, "wf_b")

        # Delta only changes wf_a
        cws.add_agent_delta("forge", {
            "/workflows/wf_a/nodes/node_3:input:steps": 99,
        })

        result_a = prims_to_workflow_json(cws, "wf_a")
        result_b = prims_to_workflow_json(cws, "wf_b")

        assert result_a["3"]["inputs"]["steps"] == 99
        assert result_b["3"]["inputs"]["steps"] == 20  # Unchanged


class TestEdgeCases:
    """Edge cases and adversarial inputs."""

    def test_single_node_workflow(self):
        workflow = {
            "1": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "test"},
            },
        }
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, workflow, "single")
        result = prims_to_workflow_json(cws, "single")
        _compare_workflows(workflow, result)

    def test_no_inputs_node(self):
        workflow = {
            "1": {
                "class_type": "EmptyNode",
                "inputs": {},
            },
        }
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, workflow, "empty")
        result = prims_to_workflow_json(cws, "empty")
        assert result["1"]["class_type"] == "EmptyNode"
        assert result["1"]["inputs"] == {}

    def test_large_seed(self):
        """Seeds can be very large integers."""
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": 2**62},
            },
        }
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, workflow, "seed")
        result = prims_to_workflow_json(cws, "seed")
        assert result["1"]["inputs"]["seed"] == 2**62

    def test_long_prompt_text(self):
        long_text = "a " * 5000  # 10000 chars
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": long_text},
            },
        }
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, workflow, "long")
        result = prims_to_workflow_json(cws, "long")
        assert result["1"]["inputs"]["text"] == long_text

    def test_special_chars_in_prompt(self):
        text = 'photo of a "cat" with <lora:detail:0.8>, (masterpiece:1.2)'
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": text},
            },
        }
        cws = CognitiveWorkflowStage()
        workflow_json_to_prims(cws, workflow, "special")
        result = prims_to_workflow_json(cws, "special")
        assert result["1"]["inputs"]["text"] == text
