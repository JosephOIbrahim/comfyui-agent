"""Tests for agent/stage/creative_profiles.py — no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.creative_profiles import (
    CREATIVE, EXPLORE, INTEGRATION, PROFILES, RADICAL,
    CreativeProfile, ProfileError,
    get_profile, read_active_profile, select_profile, store_as_variant_set,
)


@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


class TestCreativeProfile:
    def test_defaults(self):
        p = CreativeProfile(name="test")
        assert p.sampler_gain == 1.0
        assert p.cfg_gain == 1.0

    def test_frozen(self):
        with pytest.raises(AttributeError):
            EXPLORE.name = "modified"

    def test_apply(self):
        p = CreativeProfile(name="x", cfg_gain=2.0)
        assert p.apply(5.0, "cfg") == 10.0

    def test_apply_unknown_group(self):
        p = CreativeProfile(name="x", cfg_gain=2.0)
        assert p.apply(5.0, "unknown") == 5.0

    def test_get_gain(self):
        assert RADICAL.get_gain("sampler") == 3.0
        assert INTEGRATION.get_gain("cfg") == 0.6

    def test_to_dict(self):
        d = EXPLORE.to_dict()
        assert d["name"] == "explore"
        assert "sampler_gain" in d
        assert len(d) == 7


class TestBuiltInProfiles:
    def test_four_profiles(self):
        assert len(PROFILES) == 4

    def test_explore_mild(self):
        assert 1.0 < EXPLORE.sampler_gain < 2.0

    def test_creative_moderate(self):
        assert 1.5 < CREATIVE.sampler_gain < 2.5

    def test_radical_max(self):
        assert RADICAL.sampler_gain >= 2.5

    def test_integration_focused(self):
        assert INTEGRATION.sampler_gain < 1.0

    def test_get_profile_valid(self):
        p = get_profile("radical")
        assert p is RADICAL

    def test_get_profile_invalid(self):
        with pytest.raises(ProfileError, match="Unknown profile"):
            get_profile("nonexistent")


class TestVariantSetStorage:
    def test_store_default_profiles(self, usd_stage):
        path = store_as_variant_set(usd_stage)
        assert path == "/agents/creative_profile"
        prim = usd_stage.stage.GetPrimAtPath(path)
        assert prim.IsValid()

    def test_store_creates_variants(self, usd_stage):
        store_as_variant_set(usd_stage)
        prim = usd_stage.stage.GetPrimAtPath("/agents/creative_profile")
        vset = prim.GetVariantSets().GetVariantSet("profile")
        names = vset.GetVariantNames()
        assert "explore" in names
        assert "radical" in names

    def test_store_default_selection(self, usd_stage):
        store_as_variant_set(usd_stage)
        active = read_active_profile(usd_stage)
        assert active == "explore"

    def test_store_custom_profiles(self, usd_stage):
        custom = [CreativeProfile(name="custom", sampler_gain=5.0)]
        store_as_variant_set(usd_stage, custom)
        prim = usd_stage.stage.GetPrimAtPath("/agents/creative_profile")
        vset = prim.GetVariantSets().GetVariantSet("profile")
        assert "custom" in vset.GetVariantNames()


class TestProfileSelection:
    def test_select_changes_variant(self, usd_stage):
        store_as_variant_set(usd_stage)
        select_profile(usd_stage, "radical")
        active = read_active_profile(usd_stage)
        assert active == "radical"

    def test_select_returns_profile(self, usd_stage):
        store_as_variant_set(usd_stage)
        p = select_profile(usd_stage, "creative")
        assert p is CREATIVE

    def test_select_unknown_raises(self, usd_stage):
        with pytest.raises(ProfileError):
            select_profile(usd_stage, "nonexistent")


class TestReadActiveProfile:
    def test_no_prim(self, usd_stage):
        assert read_active_profile(usd_stage) is None

    def test_after_store(self, usd_stage):
        store_as_variant_set(usd_stage)
        assert read_active_profile(usd_stage) == "explore"
