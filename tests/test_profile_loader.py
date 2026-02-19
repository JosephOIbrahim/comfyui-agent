"""Tests for the Model Profile Registry (agent.profiles).

Covers: load_profile resolution chain, caching, fallback behaviour,
convenience accessors, list_profiles, and YAML profile validation.
"""

import threading

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_profile_cache():
    """Reset the profile cache between tests."""
    from agent.profiles.loader import clear_cache
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaFile:
    """Verify the _schema.yaml reference file is well-formed."""

    def test_schema_file_exists(self):
        from agent.profiles.loader import PROFILES_DIR
        schema_path = PROFILES_DIR / "_schema.yaml"
        assert schema_path.is_file(), "_schema.yaml must exist"

    def test_schema_file_parses(self):
        from agent.profiles.loader import PROFILES_DIR
        schema_path = PROFILES_DIR / "_schema.yaml"
        with open(schema_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)
        assert "meta" in data
        assert "prompt_engineering" in data
        assert "parameter_space" in data
        assert "quality_signatures" in data


# ---------------------------------------------------------------------------
# Flux profile validation
# ---------------------------------------------------------------------------

class TestFluxProfile:
    """Verify the flux1-dev.yaml profile is complete and correct."""

    def test_flux_profile_exists(self):
        from agent.profiles.loader import PROFILES_DIR
        path = PROFILES_DIR / "flux1-dev.yaml"
        assert path.is_file(), "flux1-dev.yaml must exist"

    def test_flux_profile_loads(self):
        from agent.profiles import load_profile
        profile = load_profile("flux1-dev")
        assert profile["meta"]["model_id"] == "flux1-dev"
        assert profile["meta"]["model_class"] == "flux"
        assert profile["meta"]["base_arch"] == "dit"

    def test_flux_has_all_three_consumer_sections(self):
        from agent.profiles import load_profile
        profile = load_profile("flux1-dev")
        assert "prompt_engineering" in profile
        assert "parameter_space" in profile
        assert "quality_signatures" in profile

    def test_flux_parameter_space_has_required_fields(self):
        from agent.profiles import load_profile
        params = load_profile("flux1-dev")["parameter_space"]
        # steps
        assert "steps" in params
        assert "default" in params["steps"]
        assert "range" in params["steps"]
        assert "sweet_spot" in params["steps"]
        # cfg
        assert "cfg" in params
        assert "default" in params["cfg"]
        assert "range" in params["cfg"]
        assert "sweet_spot" in params["cfg"]
        # resolution
        assert "resolution" in params
        assert "native" in params["resolution"]

    def test_flux_prompt_engineering_style(self):
        from agent.profiles import load_profile
        pe = load_profile("flux1-dev")["prompt_engineering"]
        assert pe["style"] == "natural_language"

    def test_flux_is_not_fallback(self):
        from agent.profiles import is_fallback
        assert not is_fallback("flux1-dev")

    def test_flux_negative_prompt_effectiveness_low(self):
        """Flux ignores negative prompts — effectiveness should be low."""
        from agent.profiles import load_profile
        neg = load_profile("flux1-dev")["prompt_engineering"]["negative_prompt"]
        assert neg["effectiveness"] == "low"

    def test_flux_intent_translations_present(self):
        from agent.profiles import load_profile
        pe = load_profile("flux1-dev")["prompt_engineering"]
        translations = pe.get("intent_translations", {})
        assert len(translations) > 0, "Flux profile must have intent translations"


# ---------------------------------------------------------------------------
# load_profile resolution chain
# ---------------------------------------------------------------------------

class TestLoadProfile:
    """Test the 3-tier resolution: exact → fallback → minimal."""

    def test_exact_match(self):
        from agent.profiles import load_profile
        profile = load_profile("flux1-dev")
        meta = profile.get("meta", {})
        assert meta.get("model_id") == "flux1-dev"
        assert not meta.get("_is_fallback", False)
        assert not meta.get("_is_minimal", False)

    def test_unknown_model_gets_fallback(self):
        """A model with no exact YAML match gets a fallback profile."""
        from agent.profiles import load_profile
        profile = load_profile("totally_nonexistent_model_xyz")
        meta = profile["meta"]
        assert meta["_is_fallback"] is True
        assert meta["model_id"] == "totally_nonexistent_model_xyz"

    def test_fallback_has_required_sections(self):
        from agent.profiles import load_profile
        profile = load_profile("nonexistent_model_abc")
        assert "parameter_space" in profile
        assert "prompt_engineering" in profile
        assert "quality_signatures" in profile
        assert "steps" in profile["parameter_space"]
        assert "cfg" in profile["parameter_space"]
        assert "resolution" in profile["parameter_space"]

    def test_minimal_defaults_have_conservative_values(self):
        """Test _minimal_defaults directly (bypassing fallback YAMLs)."""
        from agent.profiles.loader import _minimal_defaults
        profile = _minimal_defaults("test_model")
        params = profile["parameter_space"]
        assert params["steps"]["default"] == 20
        assert params["cfg"]["default"] == 7.0
        assert params["resolution"]["native"] == [512, 512]
        assert isinstance(params["sampler"]["recommended"], list)
        assert len(params["sampler"]["recommended"]) > 0
        assert params["denoise"]["default"] == 1.0
        assert params["lora_behavior"]["max_simultaneous"] == 2

    def test_minimal_defaults_have_all_downstream_fields(self):
        """Every field path that downstream MoE agents access must exist
        in _minimal_defaults (the last-resort fallback)."""
        from agent.profiles.loader import _minimal_defaults
        profile = _minimal_defaults("test_moe_consumer")

        # prompt_engineering paths
        pe = profile["prompt_engineering"]
        assert isinstance(pe["style"], str)
        assert isinstance(pe["positive_prompt"]["structure"], str)
        assert isinstance(pe["positive_prompt"]["keyword_sensitivity"], (int, float))
        assert isinstance(pe["positive_prompt"]["token_weighting"], str)
        assert isinstance(pe["positive_prompt"]["max_effective_tokens"], int)
        assert isinstance(pe["positive_prompt"]["effective_patterns"], list)
        assert isinstance(pe["negative_prompt"]["required_base"], str)
        assert isinstance(pe["negative_prompt"]["style"], str)
        assert isinstance(pe["negative_prompt"]["effectiveness"], (int, float))

        # parameter_space paths
        ps = profile["parameter_space"]
        assert isinstance(ps["sampler"]["recommended"], list)
        assert isinstance(ps["sampler"]["avoid"], list)
        assert isinstance(ps["sampler"]["scheduler"], str)
        assert isinstance(ps["denoise"]["default"], (int, float))
        assert isinstance(ps["denoise"]["img2img_sweet_spot"], list)
        lora = ps["lora_behavior"]
        assert isinstance(lora["max_simultaneous"], int)
        assert isinstance(lora["strength_range"], list)
        assert isinstance(lora["default_strength"], (int, float))
        assert isinstance(lora["interaction_model"], str)
        assert isinstance(lora["known_conflicts"], list)

        # quality_signatures paths
        qs = profile["quality_signatures"]
        assert isinstance(qs["iteration_signals"], dict)
        assert "needs_more_steps" in qs["iteration_signals"]
        assert "needs_lower_cfg" in qs["iteration_signals"]
        assert "needs_higher_cfg" in qs["iteration_signals"]
        assert "needs_reprompt" in qs["iteration_signals"]
        assert "needs_inpaint" in qs["iteration_signals"]
        assert "model_limitation" in qs["iteration_signals"]
        assert isinstance(qs["quality_floor"]["description"], str)
        assert isinstance(qs["quality_floor"]["reference_score"], (int, float))

        # meta paths
        meta = profile["meta"]
        assert meta["model_class"] == "unknown"
        assert meta["base_arch"] == "unknown"
        assert meta["modality"] == "image"

    def test_returns_deep_copy(self):
        """Mutating the returned profile must not affect the cache."""
        from agent.profiles import load_profile
        p1 = load_profile("flux1-dev")
        p1["meta"]["model_id"] = "MUTATED"
        p2 = load_profile("flux1-dev")
        assert p2["meta"]["model_id"] == "flux1-dev"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    """Verify thread-safe caching behaviour."""

    def test_cache_returns_same_content(self):
        from agent.profiles import load_profile
        p1 = load_profile("flux1-dev")
        p2 = load_profile("flux1-dev")
        assert p1 == p2

    def test_cache_returns_different_objects(self):
        """Deep copy means different object identity."""
        from agent.profiles import load_profile
        p1 = load_profile("flux1-dev")
        p2 = load_profile("flux1-dev")
        assert p1 is not p2

    def test_clear_cache(self):
        from agent.profiles import load_profile
        from agent.profiles.loader import clear_cache, _cache
        load_profile("flux1-dev")
        assert len(_cache) > 0
        clear_cache()
        assert len(_cache) == 0

    def test_thread_safety(self):
        """Multiple threads loading the same profile should not corrupt."""
        from agent.profiles import load_profile
        results = []
        errors = []

        def load():
            try:
                p = load_profile("flux1-dev")
                results.append(p["meta"]["model_id"])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=load) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert all(r == "flux1-dev" for r in results)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

class TestAccessors:
    """Test the convenience section-access functions."""

    def test_get_intent_section(self):
        from agent.profiles import get_intent_section
        section = get_intent_section("flux1-dev")
        assert "style" in section
        assert section["style"] == "natural_language"

    def test_get_parameter_section(self):
        from agent.profiles import get_parameter_section
        section = get_parameter_section("flux1-dev")
        assert "steps" in section
        assert "cfg" in section

    def test_get_quality_section(self):
        from agent.profiles import get_quality_section
        section = get_quality_section("flux1-dev")
        assert "expected_characteristics" in section or "quality_floor" in section

    def test_is_fallback_for_known_model(self):
        from agent.profiles import is_fallback
        assert not is_fallback("flux1-dev")

    def test_is_fallback_for_unknown_model(self):
        from agent.profiles import is_fallback
        assert is_fallback("completely_unknown_model_qqq")

    def test_accessors_return_empty_for_unknown(self):
        """Accessors on minimal-default profiles should return dicts, not crash."""
        from agent.profiles import get_intent_section, get_parameter_section, get_quality_section
        assert isinstance(get_intent_section("unknown_xyz"), dict)
        assert isinstance(get_parameter_section("unknown_xyz"), dict)
        assert isinstance(get_quality_section("unknown_xyz"), dict)


# ---------------------------------------------------------------------------
# list_profiles
# ---------------------------------------------------------------------------

class TestListProfiles:
    """Test profile enumeration."""

    def test_list_profiles_returns_list(self):
        from agent.profiles import list_profiles
        profiles = list_profiles()
        assert isinstance(profiles, list)

    def test_list_profiles_includes_flux(self):
        from agent.profiles import list_profiles
        profiles = list_profiles()
        assert "flux1-dev" in profiles

    def test_list_profiles_excludes_schema(self):
        from agent.profiles import list_profiles
        profiles = list_profiles()
        assert "_schema" not in profiles

    def test_list_profiles_excludes_defaults(self):
        from agent.profiles import list_profiles
        profiles = list_profiles()
        for name in profiles:
            assert not name.startswith("default_"), f"default_ file leaked: {name}"

    def test_list_profiles_is_sorted(self):
        from agent.profiles import list_profiles
        profiles = list_profiles()
        assert profiles == sorted(profiles)


# ---------------------------------------------------------------------------
# Fallback YAML profiles
# ---------------------------------------------------------------------------

class TestFallbackYAMLProfiles:
    """Test architecture-level fallback YAML files (default_dit, default_unet).

    These tests verify that when fallback YAML files exist on disk, they
    load correctly and contain all sections that downstream MoE agents need.
    If the YAML files have not been created yet, the tests are skipped.
    """

    def test_default_dit_fallback_loads(self):
        from agent.profiles.loader import PROFILES_DIR
        from agent.profiles import load_profile
        path = PROFILES_DIR / "default_dit.yaml"
        if not path.is_file():
            pytest.skip("default_dit.yaml not yet created")
        profile = load_profile("some_unknown_dit_model")
        assert profile["meta"]["_is_fallback"] is True
        assert profile["meta"]["model_id"] == "some_unknown_dit_model"

    def test_default_unet_fallback_loads(self):
        from agent.profiles.loader import PROFILES_DIR
        from agent.profiles import load_profile
        path = PROFILES_DIR / "default_unet.yaml"
        if not path.is_file():
            pytest.skip("default_unet.yaml not yet created")
        profile = load_profile("some_unknown_unet_model")
        assert profile["meta"]["_is_fallback"] is True
        assert profile["meta"]["model_id"] == "some_unknown_unet_model"

    def test_fallback_profiles_have_all_consumer_sections(self):
        """Any fallback YAML that exists must have all three consumer sections."""
        from agent.profiles.loader import PROFILES_DIR, _load_yaml
        found_any = False
        for name in ("default_dit", "default_unet", "default_video"):
            path = PROFILES_DIR / f"{name}.yaml"
            if not path.is_file():
                continue
            found_any = True
            data = _load_yaml(path)
            assert "prompt_engineering" in data, f"{name}.yaml missing prompt_engineering"
            assert "parameter_space" in data, f"{name}.yaml missing parameter_space"
            assert "quality_signatures" in data, f"{name}.yaml missing quality_signatures"
        if not found_any:
            pytest.skip("No fallback YAML files exist yet")
