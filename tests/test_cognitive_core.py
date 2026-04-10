"""Adversarial tests for the Cognitive Graph Engine.

[GRAPH x CRUCIBLE] — These tests actively try to break the
CognitiveGraphEngine, DeltaLayer, and WorkflowGraph implementations.
Every test category from TRACK_A_DESIGN.md is mandatory.
"""

import copy
import json

import pytest

from cognitive.core.models import ComfyNode, WorkflowGraph
from cognitive.core.delta import DeltaLayer, LIVRPS_PRIORITY, _compute_hash
from cognitive.core.graph import CognitiveGraphEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_workflow():
    """A representative ComfyUI workflow with links, multiple nodes, and varied inputs."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned.safetensors",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "a beautiful landscape",
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "ugly, blurry",
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0],
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["4", 0],
                "vae": ["1", 2],
            },
        },
    }


@pytest.fixture
def engine(sample_workflow):
    """A CognitiveGraphEngine initialized with the sample workflow."""
    return CognitiveGraphEngine(sample_workflow)


# ---------------------------------------------------------------------------
# 1. Link Preservation
# ---------------------------------------------------------------------------

class TestLinkPreservation:
    """Links (["node_id", output_index]) must survive all operations."""

    def test_links_survive_parsing(self, sample_workflow):
        """Link arrays survive WorkflowGraph.from_api_json round-trip."""
        graph = WorkflowGraph.from_api_json(sample_workflow)
        output = graph.to_api_json()
        assert output["4"]["inputs"]["model"] == ["1", 0]
        assert output["4"]["inputs"]["positive"] == ["2", 0]
        assert output["4"]["inputs"]["negative"] == ["3", 0]
        assert output["4"]["inputs"]["latent_image"] == ["5", 0]
        assert output["6"]["inputs"]["samples"] == ["4", 0]
        assert output["6"]["inputs"]["vae"] == ["1", 2]
        assert output["2"]["inputs"]["clip"] == ["1", 1]

    def test_links_survive_unrelated_mutation(self, engine):
        """Mutating a different input preserves link arrays on the same node."""
        engine.mutate_workflow({"4": {"cfg": 12.0}}, opinion="L")
        resolved = engine.to_api_json()
        # cfg changed
        assert resolved["4"]["inputs"]["cfg"] == 12.0
        # all links untouched
        assert resolved["4"]["inputs"]["model"] == ["1", 0]
        assert resolved["4"]["inputs"]["positive"] == ["2", 0]
        assert resolved["4"]["inputs"]["negative"] == ["3", 0]
        assert resolved["4"]["inputs"]["latent_image"] == ["5", 0]

    def test_links_survive_multiple_mutations(self, engine):
        """Multiple deltas don't corrupt links."""
        engine.mutate_workflow({"4": {"steps": 30}}, opinion="L")
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        engine.mutate_workflow({"2": {"text": "a mountain scene"}}, opinion="L")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["model"] == ["1", 0]
        assert resolved["2"]["inputs"]["clip"] == ["1", 1]
        assert resolved["6"]["inputs"]["samples"] == ["4", 0]

    def test_link_can_be_overwritten_by_mutation(self, engine):
        """A mutation CAN explicitly change a link input."""
        engine.mutate_workflow({"4": {"model": ["99", 0]}}, opinion="L")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["model"] == ["99", 0]
        # Other links unaffected
        assert resolved["4"]["inputs"]["positive"] == ["2", 0]


# ---------------------------------------------------------------------------
# 2. LIVRPS Strongest-Opinion-Wins
# ---------------------------------------------------------------------------

class TestLIVRPSPriority:
    """S overrides L, L overrides I, etc."""

    def test_safety_overrides_local(self, engine):
        """S-opinion wins over L-opinion on the same param."""
        engine.mutate_workflow({"4": {"cfg": 7.0}}, opinion="L")
        engine.mutate_workflow({"4": {"cfg": 1.0}}, opinion="S")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["cfg"] == 1.0

    def test_local_overrides_inherits(self, engine):
        """L-opinion wins over I-opinion."""
        engine.mutate_workflow({"4": {"steps": 10}}, opinion="I")
        engine.mutate_workflow({"4": {"steps": 50}}, opinion="L")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["steps"] == 50

    def test_full_priority_chain(self, engine):
        """Apply all 6 opinions on the same param. S should win."""
        for opinion, value in [("P", 1), ("R", 2), ("V", 3), ("I", 4), ("L", 5), ("S", 6)]:
            engine.mutate_workflow({"4": {"seed": value}}, opinion=opinion)
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["seed"] == 6  # S wins

    def test_priority_ordering_matches_table(self):
        """LIVRPS_PRIORITY values match the defined hierarchy."""
        assert LIVRPS_PRIORITY["P"] < LIVRPS_PRIORITY["R"]
        assert LIVRPS_PRIORITY["R"] < LIVRPS_PRIORITY["V"]
        assert LIVRPS_PRIORITY["V"] < LIVRPS_PRIORITY["I"]
        assert LIVRPS_PRIORITY["I"] < LIVRPS_PRIORITY["L"]
        assert LIVRPS_PRIORITY["L"] < LIVRPS_PRIORITY["S"]

    def test_weaker_opinion_applied_first_then_overridden(self, engine):
        """Even if S is pushed first and L second, S still wins."""
        engine.mutate_workflow({"4": {"cfg": 1.0}}, opinion="S")
        engine.mutate_workflow({"4": {"cfg": 99.0}}, opinion="L")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["cfg"] == 1.0  # S still wins


# ---------------------------------------------------------------------------
# 3. SHA-256 Tamper Detection
# ---------------------------------------------------------------------------

class TestTamperDetection:
    """Modifying mutations after creation must be detected."""

    def test_intact_layer(self):
        """Unmodified layer passes integrity check."""
        delta = DeltaLayer.create({"4": {"cfg": 5.0}}, opinion="L")
        assert delta.is_intact is True

    def test_tampered_mutations_detected(self):
        """Modifying mutations after creation changes layer_hash."""
        delta = DeltaLayer.create({"4": {"cfg": 5.0}}, opinion="L")
        original_hash = delta.creation_hash
        # Tamper with the mutations
        delta.mutations["4"]["cfg"] = 999.0
        assert delta.is_intact is False
        assert delta.layer_hash != original_hash
        assert delta.creation_hash == original_hash  # Creation hash unchanged

    def test_tampered_opinion_detected(self):
        """Changing opinion after creation is detected via hash mismatch."""
        delta = DeltaLayer.create({"4": {"cfg": 5.0}}, opinion="L")
        delta.opinion = "S"  # type: ignore[assignment]  — intentional tamper
        assert delta.is_intact is False

    def test_engine_verify_detects_tampered_layer(self, engine):
        """verify_stack_integrity catches a tampered delta."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        # Tamper
        engine._delta_stack[0].mutations["4"]["cfg"] = 999.0
        ok, errors = engine.verify_stack_integrity()
        assert ok is False
        assert len(errors) == 1
        assert "tampered" in errors[0]

    def test_engine_verify_passes_clean_stack(self, engine):
        """Clean stack passes integrity check."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        engine.mutate_workflow({"4": {"steps": 30}}, opinion="L")
        ok, errors = engine.verify_stack_integrity()
        assert ok is True
        assert errors == []

    def test_empty_stack_passes_integrity(self, engine):
        """No deltas = no errors."""
        ok, errors = engine.verify_stack_integrity()
        assert ok is True
        assert errors == []


# ---------------------------------------------------------------------------
# 4. Temporal Query Rollback
# ---------------------------------------------------------------------------

class TestTemporalQuery:
    """Verify correct historical state at any point."""

    def test_rollback_one_step(self, engine):
        """temporal_query(back_steps=1) excludes the last delta."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        engine.mutate_workflow({"4": {"steps": 50}}, opinion="L")

        rolled_back = engine.temporal_query(back_steps=1)
        result = rolled_back.to_api_json()
        assert result["4"]["inputs"]["cfg"] == 5.0  # First delta applied
        assert result["4"]["inputs"]["steps"] == 20  # Second delta NOT applied (original)

    def test_rollback_to_base(self, engine):
        """temporal_query(back_steps=N) where N >= stack size returns base."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        engine.mutate_workflow({"4": {"steps": 50}}, opinion="L")
        engine.mutate_workflow({"2": {"text": "changed"}}, opinion="L")

        rolled_back = engine.temporal_query(back_steps=3)
        result = rolled_back.to_api_json()
        assert result["4"]["inputs"]["cfg"] == 7.5  # Original
        assert result["4"]["inputs"]["steps"] == 20  # Original
        assert result["2"]["inputs"]["text"] == "a beautiful landscape"

    def test_rollback_more_than_stack_size(self, engine):
        """Requesting more rollback than stack size doesn't error."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        rolled_back = engine.temporal_query(back_steps=100)
        result = rolled_back.to_api_json()
        assert result["4"]["inputs"]["cfg"] == 7.5  # Base value

    def test_rollback_zero_returns_current(self, engine):
        """back_steps=0 returns current resolved graph."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        current = engine.temporal_query(back_steps=0)
        assert current.to_api_json()["4"]["inputs"]["cfg"] == 5.0

    def test_rollback_matches_up_to_index(self, engine):
        """temporal_query(1) == get_resolved_graph(up_to_index=N-1)."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        engine.mutate_workflow({"4": {"steps": 50}}, opinion="L")
        engine.mutate_workflow({"4": {"seed": 999}}, opinion="L")

        temporal = engine.temporal_query(back_steps=1).to_api_json()
        explicit = engine.get_resolved_graph(up_to_index=2).to_api_json()
        assert temporal == explicit


# ---------------------------------------------------------------------------
# 5. Multi-Node Atomic Mutations
# ---------------------------------------------------------------------------

class TestMultiNodeMutations:
    """Single delta layer modifying multiple nodes."""

    def test_three_node_mutation(self, engine):
        """One mutate_workflow call modifies 3 nodes."""
        engine.mutate_workflow(
            {
                "4": {"cfg": 12.0, "steps": 30},
                "2": {"text": "a new prompt"},
                "5": {"width": 1024, "height": 1024},
            },
            opinion="L",
            description="multi-node edit",
        )
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["cfg"] == 12.0
        assert resolved["4"]["inputs"]["steps"] == 30
        assert resolved["2"]["inputs"]["text"] == "a new prompt"
        assert resolved["5"]["inputs"]["width"] == 1024
        assert resolved["5"]["inputs"]["height"] == 1024

    def test_multi_node_is_single_delta(self, engine):
        """Multi-node mutation produces exactly one delta layer."""
        engine.mutate_workflow(
            {"4": {"cfg": 12.0}, "2": {"text": "new"}, "5": {"width": 768}},
            opinion="L",
        )
        assert len(engine.delta_stack) == 1

    def test_multi_node_preserves_untouched_nodes(self, engine):
        """Nodes not in the mutation are completely unchanged."""
        engine.mutate_workflow(
            {"4": {"cfg": 12.0}, "2": {"text": "new"}},
            opinion="L",
        )
        resolved = engine.to_api_json()
        # Node 1, 3, 5, 6 unchanged
        assert resolved["1"]["inputs"]["ckpt_name"] == "v1-5-pruned.safetensors"
        assert resolved["3"]["inputs"]["text"] == "ugly, blurry"
        assert resolved["5"]["inputs"]["width"] == 512
        assert resolved["6"]["inputs"]["samples"] == ["4", 0]


# ---------------------------------------------------------------------------
# 6. Node Injection
# ---------------------------------------------------------------------------

class TestNodeInjection:
    """Delta can add a node that doesn't exist in the base graph."""

    def test_inject_new_node(self, engine):
        """Delta with class_type for non-existent node creates it."""
        engine.mutate_workflow(
            {"99": {"class_type": "UpscaleModelLoader", "model_name": "RealESRGAN_x4.pth"}},
            opinion="L",
        )
        resolved = engine.to_api_json()
        assert "99" in resolved
        assert resolved["99"]["class_type"] == "UpscaleModelLoader"
        assert resolved["99"]["inputs"]["model_name"] == "RealESRGAN_x4.pth"

    def test_inject_preserves_existing_nodes(self, engine):
        """Injecting a new node doesn't affect existing nodes."""
        engine.mutate_workflow(
            {"99": {"class_type": "NewNode", "param": "value"}},
            opinion="L",
        )
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["model"] == ["1", 0]
        assert resolved["4"]["inputs"]["cfg"] == 7.5

    def test_inject_without_class_type_is_ignored(self, engine):
        """Node not in base AND no class_type = silently skipped."""
        engine.mutate_workflow(
            {"99": {"param": "value"}},
            opinion="L",
        )
        resolved = engine.to_api_json()
        assert "99" not in resolved

    def test_inject_with_link_inputs(self, engine):
        """Injected node can have link array inputs."""
        engine.mutate_workflow(
            {"99": {"class_type": "VAEDecode", "samples": ["4", 0], "vae": ["1", 2]}},
            opinion="L",
        )
        resolved = engine.to_api_json()
        assert resolved["99"]["inputs"]["samples"] == ["4", 0]
        assert resolved["99"]["inputs"]["vae"] == ["1", 2]


# ---------------------------------------------------------------------------
# 7. Empty Delta Stack
# ---------------------------------------------------------------------------

class TestEmptyDeltaStack:
    """Resolving with no deltas returns clean copy of base."""

    def test_no_deltas_returns_base_copy(self, engine, sample_workflow):
        """get_resolved_graph with empty stack equals base."""
        resolved = engine.get_resolved_graph()
        api = resolved.to_api_json()
        for node_id in sample_workflow:
            assert node_id in api
            assert api[node_id]["class_type"] == sample_workflow[node_id]["class_type"]
            assert api[node_id]["inputs"] == sample_workflow[node_id]["inputs"]

    def test_no_deltas_returns_deep_copy(self, engine):
        """Resolved graph with no deltas is a deep copy, not a reference."""
        resolved1 = engine.to_api_json()
        resolved2 = engine.to_api_json()
        # Mutate one, verify the other is unchanged
        resolved1["4"]["inputs"]["cfg"] = 999.0
        assert resolved2["4"]["inputs"]["cfg"] == 7.5

    def test_to_api_json_matches_get_resolved_graph(self, engine):
        """to_api_json() and get_resolved_graph().to_api_json() produce same result."""
        api = engine.to_api_json()
        graph_api = engine.get_resolved_graph().to_api_json()
        assert api == graph_api


# ---------------------------------------------------------------------------
# 8. Same-Opinion Chronological Ordering
# ---------------------------------------------------------------------------

class TestChronologicalOrdering:
    """Two same-opinion layers: later one wins (stable sort)."""

    def test_later_l_opinion_wins(self, engine):
        """Second L-opinion delta overrides first on same param."""
        engine.mutate_workflow({"4": {"steps": 10}}, opinion="L")
        engine.mutate_workflow({"4": {"steps": 50}}, opinion="L")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["steps"] == 50

    def test_later_s_opinion_wins(self, engine):
        """Second S-opinion delta overrides first on same param."""
        engine.mutate_workflow({"4": {"cfg": 1.0}}, opinion="S")
        engine.mutate_workflow({"4": {"cfg": 2.0}}, opinion="S")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["cfg"] == 2.0

    def test_chronological_across_different_params(self, engine):
        """Same-opinion deltas on different params both apply."""
        engine.mutate_workflow({"4": {"steps": 10}}, opinion="L")
        engine.mutate_workflow({"4": {"cfg": 3.0}}, opinion="L")
        resolved = engine.to_api_json()
        assert resolved["4"]["inputs"]["steps"] == 10
        assert resolved["4"]["inputs"]["cfg"] == 3.0

    def test_insertion_order_preserved_in_stack(self, engine):
        """Delta stack preserves insertion order."""
        engine.mutate_workflow({"4": {"steps": 10}}, opinion="L", layer_id="first")
        engine.mutate_workflow({"4": {"steps": 50}}, opinion="L", layer_id="second")
        stack = engine.delta_stack
        assert stack[0].layer_id == "first"
        assert stack[1].layer_id == "second"


# ---------------------------------------------------------------------------
# 9. Round-Trip Fidelity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Parse -> mutate -> serialize -> parse -> compare."""

    def test_parse_serialize_identity(self, sample_workflow):
        """from_api_json -> to_api_json produces equivalent structure."""
        graph = WorkflowGraph.from_api_json(sample_workflow)
        output = graph.to_api_json()
        for node_id in sample_workflow:
            assert output[node_id]["class_type"] == sample_workflow[node_id]["class_type"]
            assert output[node_id]["inputs"] == sample_workflow[node_id]["inputs"]

    def test_mutate_round_trip(self, engine, sample_workflow):
        """Mutate -> serialize -> parse -> compare with expected."""
        engine.mutate_workflow({"4": {"cfg": 12.0, "steps": 30}}, opinion="L")
        serialized = engine.to_api_json()

        # Parse the serialized output
        engine2 = CognitiveGraphEngine(serialized)
        reparsed = engine2.to_api_json()

        assert reparsed["4"]["inputs"]["cfg"] == 12.0
        assert reparsed["4"]["inputs"]["steps"] == 30
        assert reparsed["4"]["inputs"]["model"] == ["1", 0]  # Links preserved

    def test_full_round_trip_with_injection(self, engine):
        """Inject node, serialize, reparse, verify."""
        engine.mutate_workflow(
            {"99": {"class_type": "NewNode", "param": "val", "link": ["4", 0]}},
            opinion="L",
        )
        serialized = engine.to_api_json()
        engine2 = CognitiveGraphEngine(serialized)
        reparsed = engine2.to_api_json()

        assert reparsed["99"]["class_type"] == "NewNode"
        assert reparsed["99"]["inputs"]["param"] == "val"
        assert reparsed["99"]["inputs"]["link"] == ["4", 0]

    def test_json_serializable(self, engine):
        """to_api_json output is JSON-serializable."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        api = engine.to_api_json()
        serialized = json.dumps(api, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["4"]["inputs"]["cfg"] == 5.0


# ---------------------------------------------------------------------------
# 10. Deep Copy Isolation
# ---------------------------------------------------------------------------

class TestDeepCopyIsolation:
    """Mutations on resolved graph must not leak to base or delta stack."""

    def test_resolved_graph_is_independent(self, engine):
        """Mutating resolved graph doesn't affect engine state."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        resolved = engine.to_api_json()
        resolved["4"]["inputs"]["cfg"] = 999.0
        # Engine state unchanged
        fresh = engine.to_api_json()
        assert fresh["4"]["inputs"]["cfg"] == 5.0

    def test_base_is_frozen(self, engine):
        """Base workflow is never modified by mutations."""
        original_base = copy.deepcopy(engine.base.to_api_json())
        engine.mutate_workflow({"4": {"cfg": 999.0}}, opinion="L")
        engine.mutate_workflow({"4": {"steps": 999}}, opinion="L")
        current_base = engine.base.to_api_json()
        assert current_base == original_base

    def test_delta_stack_returns_defensive_copy(self, engine):
        """engine.delta_stack returns a copy, not the internal list."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        stack_copy = engine.delta_stack
        stack_copy.clear()
        assert len(engine.delta_stack) == 1  # Internal stack unchanged

    def test_workflow_graph_deep_copy(self, sample_workflow):
        """WorkflowGraph.deep_copy produces independent copy."""
        graph = WorkflowGraph.from_api_json(sample_workflow)
        copy_graph = graph.deep_copy()
        copy_graph.nodes["4"].inputs["cfg"] = 999.0
        assert graph.nodes["4"].inputs["cfg"] == 7.5


# ---------------------------------------------------------------------------
# Additional Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases that don't fit neatly into the 10 categories."""

    def test_pop_delta_returns_most_recent(self, engine):
        """pop_delta removes and returns the last pushed layer."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L", layer_id="first")
        engine.mutate_workflow({"4": {"steps": 30}}, opinion="L", layer_id="second")
        popped = engine.pop_delta()
        assert popped.layer_id == "second"
        assert len(engine.delta_stack) == 1
        # Resolved graph reflects only first delta
        assert engine.to_api_json()["4"]["inputs"]["steps"] == 20  # base value
        assert engine.to_api_json()["4"]["inputs"]["cfg"] == 5.0  # first delta

    def test_pop_empty_stack_returns_none(self, engine):
        """pop_delta on empty stack returns None."""
        assert engine.pop_delta() is None

    def test_comfy_node_from_api_dict(self):
        """ComfyNode.from_api_dict correctly parses node data."""
        data = {"class_type": "KSampler", "inputs": {"seed": 42, "model": ["1", 0]}}
        node = ComfyNode.from_api_dict("4", data)
        assert node.node_id == "4"
        assert node.class_type == "KSampler"
        assert node.inputs["seed"] == 42
        assert node.inputs["model"] == ["1", 0]

    def test_comfy_node_missing_inputs(self):
        """ComfyNode handles missing inputs key."""
        data = {"class_type": "SomeNode"}
        node = ComfyNode.from_api_dict("1", data)
        assert node.inputs == {}

    def test_workflow_graph_ignores_non_node_entries(self):
        """from_api_json ignores entries without class_type."""
        data = {
            "1": {"class_type": "Node", "inputs": {}},
            "extra_data": {"version": 1},
            "metadata": "not a node",
        }
        graph = WorkflowGraph.from_api_json(data)
        assert "1" in graph.nodes
        assert "extra_data" not in graph.nodes
        assert "metadata" not in graph.nodes

    def test_integer_node_ids_normalized_to_strings(self):
        """from_api_json must normalize integer keys to strings.

        ComfyUI sometimes emits JSON with integer keys (e.g. {1: {...}}).
        Python's json.loads() parses these as int. Mixed int/str keys cause
        sorted() to raise TypeError. All node IDs must be str. (Cycle 29 fix)
        """
        # Simulate what json.loads() produces for {1: ..., 2: ...}
        data = {
            1: {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
            2: {"class_type": "KSampler", "inputs": {"model": [1, 0]}},
        }
        graph = WorkflowGraph.from_api_json(data)
        # All keys must be strings
        for node_id in graph.nodes:
            assert isinstance(node_id, str), f"node_id {node_id!r} is not str"
        assert "1" in graph.nodes
        assert "2" in graph.nodes

    def test_integer_node_ids_sortable_in_to_api_json(self):
        """to_api_json must not raise TypeError when iterating normalized nodes."""
        data = {
            10: {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
            2: {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "c.safetensors"}},
        }
        graph = WorkflowGraph.from_api_json(data)
        # Must not raise — mixed int/str would fail sorted() here
        api_json = graph.to_api_json()
        assert set(api_json.keys()) == {"2", "10"}

    def test_delta_layer_create_auto_id(self):
        """DeltaLayer.create generates unique layer_id."""
        d1 = DeltaLayer.create({"4": {"cfg": 5.0}})
        d2 = DeltaLayer.create({"4": {"cfg": 5.0}})
        assert d1.layer_id != d2.layer_id

    def test_delta_layer_explicit_id(self):
        """DeltaLayer.create respects explicit layer_id."""
        d = DeltaLayer.create({"4": {"cfg": 5.0}}, layer_id="my-id")
        assert d.layer_id == "my-id"

    def test_empty_workflow(self):
        """Engine handles empty workflow gracefully."""
        engine = CognitiveGraphEngine({})
        resolved = engine.to_api_json()
        assert resolved == {}

    def test_single_node_workflow(self):
        """Engine works with a single-node workflow."""
        engine = CognitiveGraphEngine({
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "test.safetensors"}},
        })
        engine.mutate_workflow({"1": {"ckpt_name": "other.safetensors"}}, opinion="L")
        assert engine.to_api_json()["1"]["inputs"]["ckpt_name"] == "other.safetensors"

    def test_hash_deterministic(self):
        """Same opinion + mutations always produce the same hash."""
        mutations = {"4": {"cfg": 5.0, "steps": 20}}
        h1 = _compute_hash("L", mutations)
        h2 = _compute_hash("L", mutations)
        assert h1 == h2

    def test_hash_differs_for_different_opinions(self):
        """Different opinions produce different hashes."""
        mutations = {"4": {"cfg": 5.0}}
        h_l = _compute_hash("L", mutations)
        h_s = _compute_hash("S", mutations)
        assert h_l != h_s

    def test_complex_livrps_scenario(self, engine):
        """Complex scenario: multiple opinions, multiple nodes, multiple params."""
        # R sets base config
        engine.mutate_workflow(
            {"4": {"cfg": 7.0, "steps": 20, "denoise": 0.8}},
            opinion="R",
        )
        # I adjusts from experience
        engine.mutate_workflow(
            {"4": {"cfg": 8.5, "steps": 25}},
            opinion="I",
        )
        # L is user's session edit
        engine.mutate_workflow(
            {"4": {"steps": 40}},
            opinion="L",
        )
        # S enforces safety constraint
        engine.mutate_workflow(
            {"4": {"denoise": 0.5}},
            opinion="S",
        )

        resolved = engine.to_api_json()
        # cfg: R=7.0, I=8.5 -> I wins (higher priority)
        assert resolved["4"]["inputs"]["cfg"] == 8.5
        # steps: R=20, I=25, L=40 -> L wins
        assert resolved["4"]["inputs"]["steps"] == 40
        # denoise: R=0.8, S=0.5 -> S wins (safety)
        assert resolved["4"]["inputs"]["denoise"] == 0.5
        # Links still intact
        assert resolved["4"]["inputs"]["model"] == ["1", 0]


# ---------------------------------------------------------------------------
# Thread safety (Cycle 28 fix — _delta_stack_lock)
# ---------------------------------------------------------------------------

class TestDeltaStackThreadSafety:
    """Concurrent mutate/pop/resolve must not corrupt _delta_stack."""

    def test_concurrent_mutate_no_corruption(self, engine):
        """100 concurrent mutate_workflow() calls must all land in the stack."""
        import threading

        results = []
        errors = []

        def mutate(i):
            try:
                engine.mutate_workflow({"4": {"cfg": float(i)}}, opinion="L")
                results.append(i)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=mutate, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent mutate raised: {errors}"
        assert len(engine.delta_stack) == 100

    def test_concurrent_mutate_and_resolve(self, engine):
        """Concurrent mutate() + to_api_json() must never raise."""
        import threading

        errors = []

        def mutate():
            for i in range(10):
                try:
                    engine.mutate_workflow({"4": {"cfg": float(i)}}, opinion="L")
                except Exception as e:
                    errors.append(f"mutate: {e}")

        def resolve():
            for _ in range(10):
                try:
                    engine.to_api_json()
                except Exception as e:
                    errors.append(f"resolve: {e}")

        threads = (
            [threading.Thread(target=mutate) for _ in range(4)]
            + [threading.Thread(target=resolve) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent mutate/resolve raised: {errors}"

    def test_deepcopy_with_lock_succeeds(self, engine):
        """copy.deepcopy(engine) must not raise even after _delta_stack_lock added."""
        engine.mutate_workflow({"4": {"cfg": 5.0}}, opinion="L")
        clone = copy.deepcopy(engine)
        # Clone is independent
        clone.mutate_workflow({"4": {"cfg": 99.0}}, opinion="L")
        assert engine.to_api_json()["4"]["inputs"]["cfg"] == 5.0
        assert clone.to_api_json()["4"]["inputs"]["cfg"] == 99.0


# ---------------------------------------------------------------------------
# Cycle 39: _delta_stack FIFO cap
# ---------------------------------------------------------------------------

class TestDeltaStackCap:
    """Cycle 39: delta stack must not grow unbounded on long-running servers."""

    def test_stack_capped_at_max(self, engine):
        """Delta stack length must not exceed _max_delta_stack."""
        engine._max_delta_stack = 5
        for i in range(10):
            engine.mutate_workflow({"4": {"cfg": float(i)}}, opinion="L")
        with engine._delta_stack_lock:
            assert len(engine._delta_stack) <= 5

    def test_oldest_delta_evicted_first(self, engine):
        """Oldest delta is evicted when cap is hit (FIFO)."""
        engine._max_delta_stack = 3
        ids = []
        for i in range(5):
            d = engine.mutate_workflow({"4": {"cfg": float(i)}}, opinion="L")
            ids.append(d.layer_id)
        # Only last 3 deltas should remain
        with engine._delta_stack_lock:
            remaining_ids = [d.layer_id for d in engine._delta_stack]
        assert remaining_ids == ids[-3:]

    def test_resolved_graph_reflects_most_recent_deltas(self, engine):
        """After eviction, resolved graph uses the surviving (newest) deltas."""
        engine._max_delta_stack = 2
        engine.mutate_workflow({"4": {"cfg": 1.0}}, opinion="L")  # evicted
        engine.mutate_workflow({"4": {"cfg": 2.0}}, opinion="L")  # evicted
        engine.mutate_workflow({"4": {"cfg": 7.0}}, opinion="L")  # kept
        engine.mutate_workflow({"4": {"cfg": 9.0}}, opinion="L")  # kept — wins
        result = engine.to_api_json()
        assert result["4"]["inputs"]["cfg"] == 9.0

    def test_default_cap_constant_exists(self):
        """_MAX_DELTA_STACK module constant must exist and be reasonable."""
        from cognitive.core.graph import _MAX_DELTA_STACK
        assert _MAX_DELTA_STACK >= 100
