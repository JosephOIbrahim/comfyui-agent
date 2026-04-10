"""Tests for CognitiveWorkflowStage — USD-native composed stage.

Adversarial: tests try to BREAK what was built, not just confirm it works.
"""

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core not installed")

from agent.stage.anchors import AnchorViolationError  # noqa: E402
from agent.stage.cognitive_stage import (  # noqa: E402
    STAGE_HIERARCHY,
    CognitiveWorkflowStage,
    StageError,
)


class TestBootstrap:
    """Stage creation and hierarchy bootstrap."""

    def test_in_memory_creates_hierarchy(self):
        s = CognitiveWorkflowStage()
        for path in STAGE_HIERARCHY:
            assert s.prim_exists(path), f"Missing hierarchy prim: {path}"

    def test_hierarchy_prims_are_scope(self):
        s = CognitiveWorkflowStage()
        for path in STAGE_HIERARCHY:
            prim = s.stage.GetPrimAtPath(path)
            assert prim.GetTypeName() == "Scope"

    def test_in_memory_has_no_root_path(self):
        s = CognitiveWorkflowStage()
        assert s._root_path is None

    def test_file_backed_creates_and_reopens(self, tmp_path):
        path = tmp_path / "test_stage.usda"
        s = CognitiveWorkflowStage(path)
        s.write("/workflows/test", "name", "hello")
        s.flush()

        # Reopen
        s2 = CognitiveWorkflowStage(path)
        assert s2.read("/workflows/test", "name") == "hello"
        for h in STAGE_HIERARCHY:
            assert s2.prim_exists(h)

    def test_file_backed_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "stage.usda"
        s = CognitiveWorkflowStage(path)
        assert s.prim_exists("/workflows")
        s.flush()
        assert path.exists()


class TestReadWrite:
    """Attribute read/write operations."""

    def test_write_and_read_string(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "name", "my_workflow")
        assert s.read("/workflows/w1", "name") == "my_workflow"

    def test_write_and_read_int(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "steps", 20)
        assert s.read("/workflows/w1", "steps") == 20

    def test_write_and_read_float(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "cfg", 7.5)
        val = s.read("/workflows/w1", "cfg")
        assert abs(val - 7.5) < 1e-10

    def test_write_and_read_bool(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "enabled", True)
        assert s.read("/workflows/w1", "enabled") is True

    def test_write_and_read_large_int(self):
        """Seeds can be large — verify Int64 handles them."""
        s = CognitiveWorkflowStage()
        seed = 2**62  # Large but within Int64 range
        s.write("/workflows/w1", "seed", seed)
        assert s.read("/workflows/w1", "seed") == seed

    def test_overwrite_existing_attribute(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "steps", 20)
        s.write("/workflows/w1", "steps", 50)
        assert s.read("/workflows/w1", "steps") == 50

    def test_read_nonexistent_prim_returns_none(self):
        s = CognitiveWorkflowStage()
        assert s.read("/nonexistent") is None

    def test_read_nonexistent_attr_returns_none(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "name", "test")
        assert s.read("/workflows/w1", "missing_attr") is None

    def test_read_prim_existence(self):
        s = CognitiveWorkflowStage()
        assert s.read("/workflows") is True
        assert s.read("/nonexistent") is None

    def test_write_auto_creates_prim(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1/nodes/node_3", "class_type", "KSampler")
        assert s.prim_exists("/workflows/w1/nodes/node_3")

    def test_unsupported_type_raises(self):
        s = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="Unsupported type"):
            s.write("/workflows/w1", "data", [1, 2, 3])

    def test_nested_prim_creation(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1/nodes/node_5/meta", "version", 1)
        assert s.read("/workflows/w1/nodes/node_5/meta", "version") == 1


class TestAnchorImmunity:
    """Verify constitutional protection is enforced at write time."""

    def test_write_to_anchor_raises(self):
        s = CognitiveWorkflowStage()
        with pytest.raises(AnchorViolationError):
            s.write(
                "/workflows/w1/nodes/node_1",
                "ckpt_name",
                "evil_model.safetensors",
                node_type="CheckpointLoaderSimple",
            )

    def test_write_to_resolution_anchor_raises(self):
        s = CognitiveWorkflowStage()
        with pytest.raises(AnchorViolationError):
            s.write(
                "/workflows/w1/nodes/node_5",
                "width",
                256,
                node_type="EmptyLatentImage",
            )

    def test_write_non_anchor_succeeds(self):
        s = CognitiveWorkflowStage()
        s.write(
            "/workflows/w1/nodes/node_3",
            "steps",
            30,
            node_type="KSampler",
        )
        assert s.read("/workflows/w1/nodes/node_3", "steps") == 30

    def test_write_without_node_type_bypasses_check(self):
        """Without node_type, anchor check is skipped (raw write)."""
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1/nodes/node_1", "ckpt_name", "model.safetensors")
        assert (
            s.read("/workflows/w1/nodes/node_1", "ckpt_name")
            == "model.safetensors"
        )

    def test_anchor_doesnt_block_different_node_type(self):
        """ckpt_name is only anchored on CheckpointLoaderSimple, not others."""
        s = CognitiveWorkflowStage()
        s.write(
            "/workflows/w1/nodes/node_1",
            "ckpt_name",
            "value",
            node_type="CustomNode",
        )


class TestAgentDeltas:
    """Agent delta sublayers — LIVRPS Local opinion."""

    def test_delta_overrides_base_value(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1/nodes/node_3", "steps", 20)

        s.add_agent_delta("forge", {
            "/workflows/w1/nodes/node_3:steps": 50,
        })

        # Delta wins (strongest opinion)
        assert s.read("/workflows/w1/nodes/node_3", "steps") == 50

    def test_newest_delta_wins(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1/nodes/node_3", "steps", 20)

        s.add_agent_delta("forge", {
            "/workflows/w1/nodes/node_3:steps": 50,
        })
        s.add_agent_delta("crucible", {
            "/workflows/w1/nodes/node_3:steps": 99,
        })

        # Newest delta wins
        assert s.read("/workflows/w1/nodes/node_3", "steps") == 99

    def test_delta_count_tracks(self):
        s = CognitiveWorkflowStage()
        assert s.delta_count == 0
        s.add_agent_delta("a", {"/workflows/w1:x": 1})
        assert s.delta_count == 1
        s.add_agent_delta("b", {"/workflows/w1:y": 2})
        assert s.delta_count == 2

    def test_delta_returns_layer_id(self):
        s = CognitiveWorkflowStage()
        layer_id = s.add_agent_delta("forge", {"/workflows/w1:x": 1})
        assert isinstance(layer_id, str)
        assert "forge_delta" in layer_id

    def test_list_deltas(self):
        s = CognitiveWorkflowStage()
        id1 = s.add_agent_delta("a", {"/workflows/w1:x": 1})
        id2 = s.add_agent_delta("b", {"/workflows/w1:y": 2})
        deltas = s.list_deltas()
        assert deltas == [id1, id2]

    def test_multi_attr_delta(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1/nodes/node_3", "steps", 20)
        s.write("/workflows/w1/nodes/node_3", "cfg", 7.0)

        s.add_agent_delta("forge", {
            "/workflows/w1/nodes/node_3:steps": 30,
            "/workflows/w1/nodes/node_3:cfg": 5.0,
        })

        assert s.read("/workflows/w1/nodes/node_3", "steps") == 30
        assert abs(s.read("/workflows/w1/nodes/node_3", "cfg") - 5.0) < 1e-10

    def test_delta_invalid_key_raises(self):
        s = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="prim_path:attr_name"):
            s.add_agent_delta("bad", {"no_colon_here": 1})

    def test_delta_unsupported_type_raises(self):
        s = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="Unsupported type"):
            s.add_agent_delta("bad", {"/w:attr": [1, 2, 3]})


class TestRollback:
    """Delta rollback — undo agent modifications."""

    def test_rollback_restores_base_value(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "steps", 20)
        s.add_agent_delta("forge", {"/workflows/w1:steps": 50})
        assert s.read("/workflows/w1", "steps") == 50

        removed = s.rollback_to(1)
        assert removed == 1
        assert s.read("/workflows/w1", "steps") == 20
        assert s.delta_count == 0

    def test_rollback_partial(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "steps", 20)
        s.add_agent_delta("a", {"/workflows/w1:steps": 30})
        s.add_agent_delta("b", {"/workflows/w1:steps": 40})
        s.add_agent_delta("c", {"/workflows/w1:steps": 50})

        removed = s.rollback_to(2)
        assert removed == 2
        assert s.delta_count == 1
        # Only delta "a" remains (oldest), which set steps=30
        assert s.read("/workflows/w1", "steps") == 30

    def test_rollback_more_than_available(self):
        s = CognitiveWorkflowStage()
        s.add_agent_delta("a", {"/workflows/w1:x": 1})
        removed = s.rollback_to(100)
        assert removed == 1
        assert s.delta_count == 0

    def test_rollback_zero_is_noop(self):
        s = CognitiveWorkflowStage()
        s.add_agent_delta("a", {"/workflows/w1:x": 1})
        removed = s.rollback_to(0)
        assert removed == 0
        assert s.delta_count == 1

    def test_rollback_empty_is_noop(self):
        s = CognitiveWorkflowStage()
        removed = s.rollback_to(5)
        assert removed == 0


class TestReconstructClean:
    """Base layer reconstruction without agent deltas."""

    def test_clean_shows_base_values(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "steps", 20)
        s.add_agent_delta("forge", {"/workflows/w1:steps": 50})

        clean = s.reconstruct_clean()
        assert clean["/workflows/w1"]["steps"] == 20

    def test_clean_ignores_deltas(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "base_attr", "original")
        s.add_agent_delta("forge", {"/workflows/w1:delta_attr": "new"})

        clean = s.reconstruct_clean()
        assert "base_attr" in clean.get("/workflows/w1", {})
        # delta_attr should not appear in clean reconstruction
        assert "delta_attr" not in clean.get("/workflows/w1", {})

    def test_clean_preserves_stage_state(self):
        """reconstruct_clean must not corrupt the stage."""
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "steps", 20)
        s.add_agent_delta("forge", {"/workflows/w1:steps": 50})

        s.reconstruct_clean()

        # Stage should still show delta value
        assert s.read("/workflows/w1", "steps") == 50
        assert s.delta_count == 1


class TestVariantProfiles:
    """Variant set selection for creative profiles."""

    def test_create_and_select_variant(self):
        s = CognitiveWorkflowStage()
        prim = s.stage.DefinePrim("/workflows/w1", "Scope")
        vset = prim.GetVariantSets().AddVariantSet("style")
        vset.AddVariant("explore")
        vset.AddVariant("creative")

        with vset.GetVariantEditContext():
            pass  # Just need the variants to exist

        vset.SetVariantSelection("explore")
        with vset.GetVariantEditContext():
            prim.CreateAttribute("creativity", pxr.Sdf.ValueTypeNames.Double).Set(0.3)

        vset.SetVariantSelection("creative")
        with vset.GetVariantEditContext():
            prim.CreateAttribute("creativity", pxr.Sdf.ValueTypeNames.Double).Set(0.9)

        # Now test select_profile
        s.select_profile("/workflows/w1", "style", "explore")
        assert abs(s.read("/workflows/w1", "creativity") - 0.3) < 1e-6

        s.select_profile("/workflows/w1", "style", "creative")
        assert abs(s.read("/workflows/w1", "creativity") - 0.9) < 1e-6

    def test_select_profile_missing_prim_raises(self):
        s = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="Prim not found"):
            s.select_profile("/nonexistent", "style", "explore")

    def test_select_profile_missing_variant_set_raises(self):
        s = CognitiveWorkflowStage()
        s.stage.DefinePrim("/workflows/w1", "Scope")
        with pytest.raises(StageError, match="Variant set"):
            s.select_profile("/workflows/w1", "nonexistent", "explore")


class TestInspection:
    """Prim existence, children, attributes."""

    def test_prim_exists_true(self):
        s = CognitiveWorkflowStage()
        assert s.prim_exists("/workflows") is True

    def test_prim_exists_false(self):
        s = CognitiveWorkflowStage()
        assert s.prim_exists("/nonexistent") is False

    def test_list_children(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/a", "x", 1)
        s.write("/workflows/b", "x", 2)
        children = s.list_children("/workflows")
        assert "a" in children
        assert "b" in children

    def test_list_children_empty(self):
        s = CognitiveWorkflowStage()
        assert s.list_children("/recipes") == []

    def test_list_children_nonexistent(self):
        s = CognitiveWorkflowStage()
        assert s.list_children("/nonexistent") == []

    def test_get_prim_attrs(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "name", "test")
        s.write("/workflows/w1", "steps", 20)
        attrs = s.get_prim_attrs("/workflows/w1")
        assert attrs["name"] == "test"
        assert attrs["steps"] == 20

    def test_get_prim_attrs_nonexistent(self):
        s = CognitiveWorkflowStage()
        assert s.get_prim_attrs("/nonexistent") == {}


class TestPersistence:
    """Flush and export operations."""

    def test_flush_to_file(self, tmp_path):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "name", "persisted")
        out = tmp_path / "output.usda"
        result = s.flush(out)
        assert result == str(out)
        assert out.exists()

    def test_flush_in_memory_no_path_raises(self):
        s = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="No output path"):
            s.flush()

    def test_export_flat(self, tmp_path):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "steps", 20)
        s.add_agent_delta("forge", {"/workflows/w1:steps": 50})

        out = tmp_path / "flat.usda"
        s.export_flat(out)
        assert out.exists()

        # Flattened stage should have resolved value
        content = out.read_text()
        assert "50" in content

    def test_to_usda_returns_string(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "name", "test")
        usda = s.to_usda()
        assert isinstance(usda, str)
        # Root layer shows composition structure (sublayer refs), not data
        assert "subLayers" in usda
        assert "base.usda" in usda

    def test_roundtrip_file(self, tmp_path):
        """Write stage to file, reopen, verify all data intact."""
        path = tmp_path / "roundtrip.usda"
        s1 = CognitiveWorkflowStage(path)
        s1.write("/workflows/w1", "name", "roundtrip_test")
        s1.write("/workflows/w1", "steps", 42)
        s1.write("/workflows/w1", "cfg", 7.5)
        s1.write("/workflows/w1", "enabled", True)
        s1.flush()

        s2 = CognitiveWorkflowStage(path)
        assert s2.read("/workflows/w1", "name") == "roundtrip_test"
        assert s2.read("/workflows/w1", "steps") == 42
        assert abs(s2.read("/workflows/w1", "cfg") - 7.5) < 1e-10
        assert s2.read("/workflows/w1", "enabled") is True


class TestAdversarial:
    """Try to break the stage."""

    def test_deep_prim_path(self):
        s = CognitiveWorkflowStage()
        deep = "/a/b/c/d/e/f/g/h/i/j"
        s.write(deep, "val", 42)
        assert s.read(deep, "val") == 42

    def test_special_characters_in_attr_name(self):
        """USD supports namespaced attributes with colons."""
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "input:steps", 20)
        assert s.read("/workflows/w1", "input:steps") == 20

    def test_empty_delta_dict(self):
        s = CognitiveWorkflowStage()
        layer_id = s.add_agent_delta("empty", {})
        assert isinstance(layer_id, str)
        assert s.delta_count == 1

    def test_many_deltas(self):
        """Stack 20 deltas, verify newest wins, rollback all."""
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "val", 0)
        for i in range(1, 21):
            s.add_agent_delta(f"agent_{i}", {"/workflows/w1:val": i})
        assert s.read("/workflows/w1", "val") == 20
        assert s.delta_count == 20

        s.rollback_to(20)
        assert s.read("/workflows/w1", "val") == 0
        assert s.delta_count == 0

    def test_delta_does_not_corrupt_hierarchy(self):
        """Adding deltas should not remove bootstrap prims."""
        s = CognitiveWorkflowStage()
        s.add_agent_delta("test", {"/workflows/w1:x": 1})
        for path in STAGE_HIERARCHY:
            assert s.prim_exists(path)

    def test_reconstruct_clean_after_many_deltas(self):
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "base", 100)
        for i in range(10):
            s.add_agent_delta(f"a{i}", {"/workflows/w1:base": i})

        clean = s.reconstruct_clean()
        assert clean["/workflows/w1"]["base"] == 100
        # And stage still resolves to latest delta
        assert s.read("/workflows/w1", "base") == 9

    def test_bool_before_int_in_type_detection(self):
        """bool is subclass of int — must detect bool first."""
        s = CognitiveWorkflowStage()
        s.write("/workflows/w1", "flag", True)
        val = s.read("/workflows/w1", "flag")
        assert val is True
        assert isinstance(val, bool)


# ---------------------------------------------------------------------------
# Cycle 64: select_profile wraps pxr exceptions as StageError
# ---------------------------------------------------------------------------

class TestSelectProfileExceptionWrapping:
    """Cycle 64: pxr exceptions in select_profile must be wrapped as StageError."""

    def test_invalid_variant_name_raises_stage_error(self):
        """Selecting a non-existent variant must raise StageError, not bare pxr exception."""
        s = CognitiveWorkflowStage()
        # Create a prim with a known variant set
        from pxr import Usd
        prim = s._stage.DefinePrim("/test_prim", "Scope")
        vsets = prim.GetVariantSets()
        vset = vsets.AddVariantSet("style")
        vset.AddVariant("photorealistic")
        vset.AddVariant("painterly")

        # Valid variant works
        s.select_profile("/test_prim", "style", "photorealistic")

        # Non-existent variant must raise StageError (wrapped from pxr)
        with pytest.raises(StageError, match="Could not select variant"):
            s.select_profile("/test_prim", "style", "nonexistent_variant_xyz")

    def test_prim_not_found_raises_stage_error(self):
        """Missing prim must raise StageError (existing guard, not pxr exception)."""
        s = CognitiveWorkflowStage()
        with pytest.raises(StageError, match="Prim not found"):
            s.select_profile("/no_such_prim", "style", "any")

    def test_missing_variant_set_raises_stage_error(self):
        """Missing variant set must raise StageError (existing guard)."""
        s = CognitiveWorkflowStage()
        s._stage.DefinePrim("/test_prim2", "Scope")
        with pytest.raises(StageError, match="Variant set"):
            s.select_profile("/test_prim2", "nonexistent_set", "any")
