"""Tests for adversarial workflow generators.

Verifies that each generator:
  1. Produces a valid dict in ComfyUI API format.
  2. Produces the specific adversarial condition it advertises.
  3. Can be serialized to JSON (round-trip safe).
"""

from __future__ import annotations

import json

import pytest

from agent.testing.adversarial import (
    ALL_GENERATORS,
    cycle_workflow,
    disconnected_nodes,
    empty_prompt,
    empty_workflow,
    invalid_cfg_range,
    missing_node_type,
    mixed_model_family,
    negative_dimensions,
    oversized_workflow,
    type_mismatch_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_connection(value) -> bool:
    """Return True if *value* looks like a ComfyUI connection [node_id, slot]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _get_all_connections(workflow: dict) -> list[tuple[str, str]]:
    """Return (source_id, target_id) for every connection in the workflow."""
    connections = []
    for node_id, node in workflow.items():
        for _input_name, value in node.get("inputs", {}).items():
            if _is_connection(value):
                connections.append((value[0], node_id))
    return connections


def _node_class_types(workflow: dict) -> list[str]:
    """Return all class_type values in the workflow."""
    return [node["class_type"] for node in workflow.values()]


# ---------------------------------------------------------------------------
# Structural validity — every generator must produce valid API format
# ---------------------------------------------------------------------------


class TestStructuralValidity:
    """All generators produce dicts conforming to ComfyUI API format."""

    @pytest.mark.parametrize("name", sorted(ALL_GENERATORS))
    def test_returns_dict(self, name):
        gen = ALL_GENERATORS[name]
        wf = gen() if name != "oversized_workflow" else gen(10)
        assert isinstance(wf, dict)

    @pytest.mark.parametrize("name", sorted(ALL_GENERATORS))
    def test_node_ids_are_strings(self, name):
        gen = ALL_GENERATORS[name]
        wf = gen() if name != "oversized_workflow" else gen(10)
        for node_id in wf:
            assert isinstance(node_id, str), f"Node id {node_id!r} is not a string"

    @pytest.mark.parametrize("name", sorted(ALL_GENERATORS))
    def test_nodes_have_class_type(self, name):
        gen = ALL_GENERATORS[name]
        wf = gen() if name != "oversized_workflow" else gen(10)
        for node_id, node in wf.items():
            assert "class_type" in node, f"Node {node_id} missing class_type"
            assert isinstance(node["class_type"], str)

    @pytest.mark.parametrize("name", sorted(ALL_GENERATORS))
    def test_nodes_have_inputs_dict(self, name):
        gen = ALL_GENERATORS[name]
        wf = gen() if name != "oversized_workflow" else gen(10)
        for node_id, node in wf.items():
            assert "inputs" in node, f"Node {node_id} missing inputs"
            assert isinstance(node["inputs"], dict)

    @pytest.mark.parametrize("name", sorted(ALL_GENERATORS))
    def test_json_round_trip(self, name):
        gen = ALL_GENERATORS[name]
        wf = gen() if name != "oversized_workflow" else gen(10)
        dumped = json.dumps(wf, sort_keys=True)
        loaded = json.loads(dumped)
        assert loaded == wf


# ---------------------------------------------------------------------------
# Adversarial condition tests — each generator's specific edge case
# ---------------------------------------------------------------------------


class TestEmptyWorkflow:
    def test_has_no_nodes(self):
        wf = empty_workflow()
        assert len(wf) == 0

    def test_is_empty_dict(self):
        assert empty_workflow() == {}


class TestDisconnectedNodes:
    def test_has_multiple_nodes(self):
        wf = disconnected_nodes()
        assert len(wf) >= 2

    def test_no_valid_connections(self):
        wf = disconnected_nodes()
        connections = _get_all_connections(wf)
        assert len(connections) == 0, (
            "Disconnected workflow should have no [node_id, slot] connections"
        )


class TestCycleWorkflow:
    def test_has_connections(self):
        wf = cycle_workflow()
        connections = _get_all_connections(wf)
        assert len(connections) >= 3

    def test_contains_cycle(self):
        """Walk the graph and verify a cycle exists."""
        wf = cycle_workflow()
        # Build adjacency: source -> [targets]
        adj: dict[str, list[str]] = {nid: [] for nid in wf}
        for src, tgt in _get_all_connections(wf):
            if src in adj:
                adj[src].append(tgt)

        # DFS cycle detection
        visited: set[str] = set()
        in_stack: set[str] = set()
        has_cycle = False

        def dfs(node: str) -> bool:
            nonlocal has_cycle
            visited.add(node)
            in_stack.add(node)
            for neighbor in adj.get(node, []):
                if neighbor in in_stack:
                    has_cycle = True
                    return True
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            in_stack.discard(node)
            return False

        for node in adj:
            if node not in visited:
                dfs(node)
        assert has_cycle, "cycle_workflow must contain a circular dependency"


class TestTypeMismatchWorkflow:
    def test_save_image_feeds_ksampler_model(self):
        wf = type_mismatch_workflow()
        ksampler = [
            n for n in wf.values() if n["class_type"] == "KSampler"
        ]
        assert len(ksampler) == 1
        model_input = ksampler[0]["inputs"]["model"]
        assert _is_connection(model_input)
        source_id = model_input[0]
        assert wf[source_id]["class_type"] == "SaveImage"


class TestMissingNodeType:
    def test_contains_unknown_class(self):
        wf = missing_node_type()
        classes = _node_class_types(wf)
        known = {
            "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler",
            "EmptyLatentImage", "VAEDecode", "SaveImage", "LoraLoader",
            "VAEEncode",
        }
        unknown = [c for c in classes if c not in known]
        assert len(unknown) >= 1, "Must contain at least one unknown class_type"

    def test_fake_class_name(self):
        wf = missing_node_type()
        classes = _node_class_types(wf)
        assert any("TotallyFake" in c for c in classes)


class TestOversizedWorkflow:
    def test_default_size(self):
        wf = oversized_workflow()
        assert len(wf) == 1000

    def test_custom_size(self):
        wf = oversized_workflow(50)
        assert len(wf) == 50

    def test_minimum_clamp(self):
        wf = oversized_workflow(0)
        assert len(wf) == 1

    def test_chain_connectivity(self):
        """Each node (after the first) references the previous node."""
        wf = oversized_workflow(10)
        for i in range(2, 11):
            node = wf[str(i)]
            model_link = node["inputs"]["model"]
            assert model_link == [str(i - 1), 0]

    def test_first_node_has_no_model_link(self):
        wf = oversized_workflow(5)
        assert "model" not in wf["1"]["inputs"]


class TestMixedModelFamily:
    def test_has_checkpoint_and_lora(self):
        wf = mixed_model_family()
        classes = _node_class_types(wf)
        assert "CheckpointLoaderSimple" in classes
        assert "LoraLoader" in classes

    def test_sd15_checkpoint_with_sdxl_lora(self):
        wf = mixed_model_family()
        checkpoint = [
            n for n in wf.values()
            if n["class_type"] == "CheckpointLoaderSimple"
        ][0]
        lora = [
            n for n in wf.values()
            if n["class_type"] == "LoraLoader"
        ][0]
        assert "v1-5" in checkpoint["inputs"]["ckpt_name"]
        assert "sdxl" in lora["inputs"]["lora_name"]


class TestInvalidCfgRange:
    def test_cfg_is_500(self):
        wf = invalid_cfg_range()
        ksampler = [
            n for n in wf.values() if n["class_type"] == "KSampler"
        ][0]
        assert ksampler["inputs"]["cfg"] == 500


class TestNegativeDimensions:
    def test_negative_width_and_height(self):
        wf = negative_dimensions()
        latent = [
            n for n in wf.values()
            if n["class_type"] == "EmptyLatentImage"
        ][0]
        assert latent["inputs"]["width"] < 0
        assert latent["inputs"]["height"] < 0

    def test_both_are_minus_512(self):
        wf = negative_dimensions()
        latent = [
            n for n in wf.values()
            if n["class_type"] == "EmptyLatentImage"
        ][0]
        assert latent["inputs"]["width"] == -512
        assert latent["inputs"]["height"] == -512


class TestEmptyPrompt:
    def test_clip_text_is_empty_string(self):
        wf = empty_prompt()
        clip_nodes = [
            n for n in wf.values()
            if n["class_type"] == "CLIPTextEncode"
        ]
        assert len(clip_nodes) >= 1
        assert clip_nodes[0]["inputs"]["text"] == ""

    def test_still_has_sampler(self):
        wf = empty_prompt()
        classes = _node_class_types(wf)
        assert "KSampler" in classes


# ---------------------------------------------------------------------------
# ALL_GENERATORS registry completeness
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_generators_present(self):
        expected = {
            "empty_workflow",
            "disconnected_nodes",
            "cycle_workflow",
            "type_mismatch_workflow",
            "missing_node_type",
            "oversized_workflow",
            "mixed_model_family",
            "invalid_cfg_range",
            "negative_dimensions",
            "empty_prompt",
        }
        assert set(ALL_GENERATORS.keys()) == expected

    def test_all_generators_callable(self):
        for name, gen in ALL_GENERATORS.items():
            assert callable(gen), f"{name} is not callable"
