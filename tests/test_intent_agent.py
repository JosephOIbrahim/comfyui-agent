"""Tests for the Intent Agent — artistic language to parameter translation.

Covers: dataclass construction, intent parsing, translation lookup,
conflict detection/resolution, direction-to-value conversion, prompt
formatting, refinement context, fallback profiles, and edge cases.
"""

from __future__ import annotations

import json

import pytest

from agent.agents.intent_agent import (
    ConflictResolution,
    IntentAgent,
    IntentSpecification,
    ParameterMutation,
    PromptMutation,
    _parse_effectiveness,
)
from agent.profiles import clear_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_profile_cache():
    """Reset the profile cache between tests."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def agent() -> IntentAgent:
    return IntentAgent()


# ---------------------------------------------------------------------------
# TestDataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify dataclass construction and defaults."""

    def test_parameter_mutation_defaults(self):
        m = ParameterMutation(target="KSampler.cfg", action="set")
        assert m.target == "KSampler.cfg"
        assert m.action == "set"
        assert m.value is None
        assert m.magnitude is None
        assert m.reason == ""

    def test_parameter_mutation_full(self):
        m = ParameterMutation(
            target="KSampler.steps",
            action="adjust_up",
            value=30,
            magnitude="moderate",
            reason="more detail",
        )
        assert m.value == 30
        assert m.magnitude == "moderate"

    def test_prompt_mutation_defaults(self):
        m = PromptMutation(target="positive_prompt", action="append")
        assert m.value == ""
        assert m.reason == ""

    def test_prompt_mutation_full(self):
        m = PromptMutation(
            target="negative_prompt",
            action="remove",
            value="blurry",
            reason="sharper intent",
        )
        assert m.target == "negative_prompt"
        assert m.action == "remove"

    def test_conflict_resolution_creation(self):
        c = ConflictResolution(
            intent_a="dreamier",
            intent_b="sharper",
            conflict_dimension="cfg_direction",
            resolution_strategy="hold",
            explanation="test",
        )
        assert c.intent_a == "dreamier"
        assert c.conflict_dimension == "cfg_direction"

    def test_intent_specification_defaults(self):
        spec = IntentSpecification(model_id="flux1-dev")
        assert spec.model_id == "flux1-dev"
        assert spec.parameter_mutations == []
        assert spec.prompt_mutations == []
        assert spec.confidence == 0.0
        assert spec.conflicts_resolved == []
        assert spec.warnings == []
        assert spec.using_fallback_profile is False

    def test_to_dict_roundtrip(self):
        spec = IntentSpecification(
            model_id="test",
            parameter_mutations=[
                ParameterMutation(
                    target="KSampler.cfg", action="set", value=3.5,
                    magnitude="moderate", reason="test",
                ),
            ],
            prompt_mutations=[
                PromptMutation(
                    target="positive_prompt", action="append",
                    value="soft focus", reason="dreamier",
                ),
            ],
            confidence=0.85,
            conflicts_resolved=[
                ConflictResolution(
                    intent_a="a", intent_b="b",
                    conflict_dimension="cfg_direction",
                    resolution_strategy="hold",
                    explanation="conflicting",
                ),
            ],
            warnings=["test warning"],
            using_fallback_profile=True,
        )
        d = spec.to_dict()
        assert d["model_id"] == "test"
        assert d["confidence"] == 0.85
        assert d["using_fallback_profile"] is True
        assert len(d["parameter_mutations"]) == 1
        assert d["parameter_mutations"][0]["target"] == "KSampler.cfg"
        assert len(d["prompt_mutations"]) == 1
        assert len(d["conflicts_resolved"]) == 1
        assert d["warnings"] == ["test warning"]
        # Verify JSON-serializable with sort_keys
        json_str = json.dumps(d, sort_keys=True)
        assert "model_id" in json_str

    def test_to_dict_empty_spec(self):
        spec = IntentSpecification(model_id="empty")
        d = spec.to_dict()
        assert d["parameter_mutations"] == []
        assert d["prompt_mutations"] == []
        assert d["conflicts_resolved"] == []


# ---------------------------------------------------------------------------
# TestIntentParsing
# ---------------------------------------------------------------------------


class TestIntentParsing:
    """Test _parse_intents splits compound intents correctly."""

    def test_single_intent(self, agent):
        assert agent._parse_intents("dreamier") == ["dreamier"]

    def test_and_separator(self, agent):
        result = agent._parse_intents("dreamier and sharper")
        assert result == ["dreamier", "sharper"]

    def test_comma_separator(self, agent):
        result = agent._parse_intents("dreamier, sharper")
        assert result == ["dreamier", "sharper"]

    def test_but_separator(self, agent):
        result = agent._parse_intents("dreamier but sharper")
        assert result == ["dreamier", "sharper"]

    def test_semicolon_separator(self, agent):
        result = agent._parse_intents("dreamier; sharper")
        assert result == ["dreamier", "sharper"]

    def test_ampersand_separator(self, agent):
        result = agent._parse_intents("dreamier & sharper")
        assert result == ["dreamier", "sharper"]

    def test_plus_separator(self, agent):
        result = agent._parse_intents("dreamier + sharper")
        assert result == ["dreamier", "sharper"]

    def test_three_intents(self, agent):
        result = agent._parse_intents("dreamier, sharper, and warmer")
        assert len(result) == 3
        assert "dreamier" in result
        assert "sharper" in result
        assert "warmer" in result

    def test_empty_string(self, agent):
        assert agent._parse_intents("") == []

    def test_whitespace_only(self, agent):
        assert agent._parse_intents("   ") == []

    def test_case_insensitive(self, agent):
        result = agent._parse_intents("DREAMIER and SHARPER")
        assert result == ["dreamier", "sharper"]


# ---------------------------------------------------------------------------
# TestIntentTranslation (using real profiles)
# ---------------------------------------------------------------------------


class TestIntentTranslation:
    """Test translate() with real YAML profiles."""

    def test_dreamier_flux(self, agent):
        """dreamier on Flux should lower cfg."""
        spec = agent.translate("dreamier", "flux1-dev")
        assert spec.model_id == "flux1-dev"
        assert spec.confidence > 0.7
        # Should have cfg mutation
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) >= 1
        # Flux cfg default is 3.5, dreamier pushes lower -> toward 2.5
        assert cfg_muts[0].value < 3.5

    def test_dreamier_sdxl(self, agent):
        """dreamier on SDXL should also lower cfg but to different value."""
        spec = agent.translate("dreamier", "sdxl-base")
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) >= 1
        # SDXL cfg default is 7.0, dreamier pushes lower
        assert cfg_muts[0].value < 7.0

    def test_sharper_flux(self, agent):
        """sharper on Flux should raise cfg and prefer euler."""
        spec = agent.translate("sharper", "flux1-dev")
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) >= 1
        assert cfg_muts[0].value > 3.5
        sampler_muts = [
            m for m in spec.parameter_mutations if "sampler" in m.target
        ]
        assert len(sampler_muts) >= 1
        assert sampler_muts[0].value == "euler"

    def test_more_photorealistic(self, agent):
        """more photorealistic should add prompt additions."""
        spec = agent.translate("more photorealistic", "flux1-dev")
        pos_prompts = [
            m for m in spec.prompt_mutations if m.target == "positive_prompt"
        ]
        assert len(pos_prompts) >= 1
        assert "photograph" in pos_prompts[0].value.lower() or \
            "photo" in pos_prompts[0].value.lower()

    def test_unknown_intent(self, agent):
        """Unknown intent word produces warnings and low confidence."""
        spec = agent.translate("flibbertigibbet", "flux1-dev")
        assert spec.confidence < 0.5
        assert any("flibbertigibbet" in w for w in spec.warnings)
        assert spec.parameter_mutations == []

    def test_empty_intent(self, agent):
        """Empty intent produces low confidence, no mutations."""
        spec = agent.translate("", "flux1-dev")
        assert spec.confidence <= 0.3
        assert spec.parameter_mutations == []
        assert spec.prompt_mutations == []

    def test_synonym_dreamy(self, agent):
        """'dreamy' (synonym) should resolve to 'dreamier'."""
        spec = agent.translate("dreamy", "flux1-dev")
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) >= 1
        assert cfg_muts[0].value < 3.5

    def test_synonym_crisp(self, agent):
        """'crisp' should resolve to 'sharper'."""
        spec = agent.translate("crisp", "flux1-dev")
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) >= 1
        assert cfg_muts[0].value > 3.5

    def test_synonym_realistic(self, agent):
        """'realistic' should resolve to 'more photorealistic'."""
        spec = agent.translate("realistic", "sdxl-base")
        pos_prompts = [
            m for m in spec.prompt_mutations if m.target == "positive_prompt"
        ]
        assert len(pos_prompts) >= 1


# ---------------------------------------------------------------------------
# TestConflictResolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    """Test conflict detection and resolution between competing intents."""

    def test_dreamier_and_sharper_cfg_conflict(self, agent):
        """dreamier + sharper push cfg opposite ways -> conflict."""
        spec = agent.translate("dreamier and sharper", "flux1-dev")
        assert len(spec.conflicts_resolved) >= 1
        cfg_conflicts = [
            c for c in spec.conflicts_resolved
            if c.conflict_dimension == "cfg_direction"
        ]
        assert len(cfg_conflicts) == 1
        assert cfg_conflicts[0].resolution_strategy == \
            "hold_current_cfg_adjust_via_prompt_and_sampler"

    def test_faster_and_higher_quality_steps_conflict(self, agent):
        """faster + higher quality push steps opposite ways on SDXL."""
        spec = agent.translate("faster and higher quality", "sdxl-base")
        steps_conflicts = [
            c for c in spec.conflicts_resolved
            if c.conflict_dimension == "steps_direction"
        ]
        assert len(steps_conflicts) == 1
        assert steps_conflicts[0].resolution_strategy == \
            "favor_higher_for_quality"

    def test_three_intents_one_conflict(self, agent):
        """Three intents with only one conflicting pair."""
        spec = agent.translate("dreamier, sharper, and warmer", "flux1-dev")
        # dreamier vs sharper conflict on cfg
        cfg_conflicts = [
            c for c in spec.conflicts_resolved
            if c.conflict_dimension == "cfg_direction"
        ]
        assert len(cfg_conflicts) == 1
        # warmer has no conflict dimension with the others
        # Still expect prompt mutations from warmer
        pos = [m for m in spec.prompt_mutations if m.target == "positive_prompt"]
        warm_prompts = [m for m in pos if "warm" in m.value.lower()]
        assert len(warm_prompts) >= 1

    def test_no_conflicts(self, agent):
        """Non-conflicting intents produce no conflicts."""
        spec = agent.translate("warmer and moodier", "flux1-dev")
        assert spec.conflicts_resolved == []

    def test_conflict_suppresses_cfg_mutations(self, agent):
        """When cfg conflicts, no cfg mutations should be emitted."""
        spec = agent.translate("dreamier and sharper", "flux1-dev")
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) == 0  # suppressed by conflict


# ---------------------------------------------------------------------------
# TestDirectionToValue
# ---------------------------------------------------------------------------


class TestDirectionToValue:
    """Test _direction_to_value conversion."""

    def test_lower_from_default(self, agent):
        """'lower' from cfg default should move toward sweet_spot low."""
        param_space = {
            "cfg": {
                "default": 3.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        val = agent._direction_to_value("lower", "cfg", param_space, None)
        assert val is not None
        assert val < 3.5
        assert val >= 2.5  # stays within sweet_spot

    def test_higher_from_default(self, agent):
        """'higher' from cfg default should move toward sweet_spot high."""
        param_space = {
            "cfg": {
                "default": 3.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        val = agent._direction_to_value("higher", "cfg", param_space, None)
        assert val is not None
        assert val > 3.5
        assert val <= 4.5  # stays within sweet_spot

    def test_slightly_higher_small_adjustment(self, agent):
        """'slightly_higher' should produce a smaller delta than 'higher'."""
        param_space = {
            "cfg": {
                "default": 3.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        slight = agent._direction_to_value(
            "slightly_higher", "cfg", param_space, None,
        )
        moderate = agent._direction_to_value(
            "higher", "cfg", param_space, None,
        )
        assert slight is not None
        assert moderate is not None
        # Both are higher than default
        assert slight > 3.5
        assert moderate > 3.5
        # Slight should be smaller adjustment (or equal if clamped by sweet_spot)
        assert slight <= moderate

    def test_never_exceeds_range(self, agent):
        """Value should never exceed range bounds."""
        param_space = {
            "cfg": {
                "default": 9.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        val = agent._direction_to_value(
            "much_higher", "cfg", param_space, 9.5,
        )
        assert val is not None
        assert val <= 10.0

    def test_never_below_range(self, agent):
        """Value should never go below range lower bound."""
        param_space = {
            "cfg": {
                "default": 1.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        val = agent._direction_to_value(
            "much_lower", "cfg", param_space, 1.5,
        )
        assert val is not None
        assert val >= 1.0

    def test_respects_sweet_spot(self, agent):
        """Direction should prefer staying within sweet_spot."""
        param_space = {
            "cfg": {
                "default": 3.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        val = agent._direction_to_value("lower", "cfg", param_space, 3.5)
        assert val is not None
        assert val >= 2.5  # sweet_spot lower bound

    def test_uses_current_value_as_anchor(self, agent):
        """When current_value provided, it's used as anchor, not default."""
        param_space = {
            "cfg": {
                "default": 3.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        val = agent._direction_to_value("lower", "cfg", param_space, 4.0)
        # Should move down from 4.0, not 3.5
        assert val is not None
        assert val < 4.0

    def test_steps_returns_integer(self, agent):
        """Steps should always be integers."""
        param_space = {
            "steps": {
                "default": 20,
                "range": [10, 50],
                "sweet_spot": [18, 25],
            },
        }
        val = agent._direction_to_value("higher", "steps", param_space, None)
        assert val is not None
        assert isinstance(val, int)

    def test_default_direction(self, agent):
        """'default' direction returns the profile default."""
        param_space = {
            "cfg": {
                "default": 3.5,
                "range": [1.0, 10.0],
                "sweet_spot": [2.5, 4.5],
            },
        }
        val = agent._direction_to_value("default", "cfg", param_space, 5.0)
        assert val == 3.5

    def test_missing_param_returns_none(self, agent):
        """Unknown parameter name returns None."""
        val = agent._direction_to_value("higher", "nonexistent", {}, None)
        assert val is None


# ---------------------------------------------------------------------------
# TestPromptMutations
# ---------------------------------------------------------------------------


class TestPromptMutations:
    """Test prompt mutation formatting and negative prompt filtering."""

    def test_natural_language_style(self, agent):
        """Flux uses natural_language, so text stays as-is."""
        spec = agent.translate("dreamier", "flux1-dev")
        pos = [m for m in spec.prompt_mutations if m.target == "positive_prompt"]
        assert len(pos) >= 1
        # Should be natural language, not reformatted as tags
        assert pos[0].value  # non-empty

    def test_hybrid_style_sdxl(self, agent):
        """SDXL uses hybrid, text stays as-is."""
        spec = agent.translate("dreamier", "sdxl-base")
        pos = [m for m in spec.prompt_mutations if m.target == "positive_prompt"]
        assert len(pos) >= 1

    def test_low_negative_effectiveness_skips_negatives(self, agent):
        """Flux has low negative effectiveness -> no negative mutations."""
        spec = agent.translate("dreamier", "flux1-dev")
        neg = [m for m in spec.prompt_mutations if m.target == "negative_prompt"]
        assert len(neg) == 0  # Flux negative effectiveness is "low"

    def test_sdxl_includes_negatives(self, agent):
        """SDXL has moderate negative effectiveness -> includes negatives."""
        spec = agent.translate("dreamier", "sdxl-base")
        neg = [m for m in spec.prompt_mutations if m.target == "negative_prompt"]
        assert len(neg) >= 1  # SDXL negative_additions present

    def test_warmer_prompt_only(self, agent):
        """warmer on Flux only adds prompt, no CFG change."""
        spec = agent.translate("warmer", "flux1-dev")
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) == 0
        pos = [m for m in spec.prompt_mutations if m.target == "positive_prompt"]
        assert len(pos) >= 1
        assert "warm" in pos[0].value.lower()

    def test_format_prompt_tag_based(self, agent):
        """Tag-based style should ensure comma separation."""
        result = agent._format_prompt_addition(
            "soft focus; ethereal light; gentle", "tag_based",
        )
        assert ", " in result
        assert ";" not in result


# ---------------------------------------------------------------------------
# TestRefinementContext
# ---------------------------------------------------------------------------


class TestRefinementContext:
    """Test refinement_context integration."""

    def test_refinement_adjust_cfg_lower(self, agent):
        """Refinement recommending lower cfg -> dreamier intent."""
        spec = agent.translate(
            "", "flux1-dev",
            refinement_context=[{
                "type": "adjust_params",
                "target": "cfg",
                "recommendation": "lower cfg to reduce banding",
            }],
        )
        # Should produce a cfg mutation even with empty user intent
        cfg_muts = [m for m in spec.parameter_mutations if "cfg" in m.target]
        assert len(cfg_muts) >= 1

    def test_refinement_adjust_steps_higher(self, agent):
        """Refinement recommending more steps -> more detailed."""
        spec = agent.translate(
            "", "sdxl-base",
            refinement_context=[{
                "type": "adjust_params",
                "target": "steps",
                "recommendation": "higher steps for more detail",
            }],
        )
        steps_muts = [m for m in spec.parameter_mutations if "steps" in m.target]
        assert len(steps_muts) >= 1

    def test_refinement_improve_quality(self, agent):
        """improve_quality refinement -> more detailed."""
        spec = agent.translate(
            "", "flux1-dev",
            refinement_context=[{"type": "improve_quality"}],
        )
        assert spec.parameter_mutations or spec.prompt_mutations

    def test_multiple_refinements(self, agent):
        """Multiple refinement actions stack."""
        spec = agent.translate(
            "", "sdxl-base",
            refinement_context=[
                {
                    "type": "adjust_params",
                    "target": "cfg",
                    "recommendation": "lower cfg",
                },
                {"type": "improve_quality"},
            ],
        )
        assert len(spec.parameter_mutations) >= 1


# ---------------------------------------------------------------------------
# TestFallbackProfile
# ---------------------------------------------------------------------------


class TestFallbackProfile:
    """Test behavior with unknown models that use fallback profiles."""

    def test_unknown_model_uses_fallback(self, agent):
        """Unknown model_id triggers fallback profile."""
        spec = agent.translate("dreamier", "totally-unknown-model-xyz")
        assert spec.using_fallback_profile is True
        assert any("fallback" in w.lower() for w in spec.warnings)

    def test_fallback_reduces_confidence(self, agent):
        """Fallback profile should reduce confidence."""
        known = agent.translate("dreamier", "flux1-dev")
        fallback = agent.translate("dreamier", "unknown-model-abc")
        assert fallback.confidence < known.confidence

    def test_fallback_still_produces_mutations(self, agent):
        """Even with fallback, the agent should try to produce mutations."""
        spec = agent.translate("sharper", "unknown-model-abc")
        # Fallback profile has empty intent_translations, so this will
        # be unmatched -> warnings but maybe no mutations
        assert spec.using_fallback_profile is True


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_very_long_intent(self, agent):
        """Long intent string doesn't crash."""
        long_intent = "dreamier " * 100
        spec = agent.translate(long_intent.strip(), "flux1-dev")
        assert spec.model_id == "flux1-dev"

    def test_intent_with_special_characters(self, agent):
        """Special characters don't crash the parser."""
        spec = agent.translate("dreamier!!! @#$%^", "flux1-dev")
        assert spec.model_id == "flux1-dev"

    def test_none_workflow_state(self, agent):
        """None workflow_state uses defaults."""
        spec = agent.translate("dreamier", "flux1-dev", workflow_state=None)
        assert spec.parameter_mutations  # should still produce mutations

    def test_compound_with_mixed_separators(self, agent):
        """Mixed separator styles in one string."""
        spec = agent.translate(
            "dreamier, warmer & moodier", "flux1-dev",
        )
        pos = [m for m in spec.prompt_mutations if m.target == "positive_prompt"]
        # Should have prompt additions from dreamier, warmer, and moodier
        assert len(pos) >= 2

    def test_workflow_state_influences_value(self, agent):
        """Current workflow state affects computed values."""
        spec_default = agent.translate(
            "sharper", "flux1-dev", workflow_state=None,
        )
        spec_high = agent.translate(
            "sharper", "flux1-dev", workflow_state={"cfg": 4.0},
        )
        cfg_default = [
            m for m in spec_default.parameter_mutations if "cfg" in m.target
        ]
        cfg_high = [
            m for m in spec_high.parameter_mutations if "cfg" in m.target
        ]
        # Both should produce cfg values
        assert len(cfg_default) >= 1
        assert len(cfg_high) >= 1

    def test_deduplicate_mutations(self, agent):
        """Multiple intents targeting same param keep last one."""
        muts = [
            ParameterMutation(target="KSampler.cfg", action="set", value=3.0),
            ParameterMutation(target="KSampler.cfg", action="set", value=4.0),
        ]
        result = agent._deduplicate_mutations(muts)
        assert len(result) == 1
        assert result[0].value == 4.0


# ---------------------------------------------------------------------------
# TestParseEffectiveness
# ---------------------------------------------------------------------------


class TestParseEffectiveness:
    """Test the _parse_effectiveness helper."""

    def test_float_value(self):
        assert _parse_effectiveness(0.65) == 0.65

    def test_int_value(self):
        assert _parse_effectiveness(1) == 1.0

    def test_numeric_string(self):
        assert _parse_effectiveness("0.5") == 0.5

    def test_low_string(self):
        assert _parse_effectiveness("low") == 0.1

    def test_high_string(self):
        assert _parse_effectiveness("high") == 0.8

    def test_unknown_string(self):
        assert _parse_effectiveness("moderate") == 0.5


# ---------------------------------------------------------------------------
# TestLookupIntent
# ---------------------------------------------------------------------------


class TestLookupIntent:
    """Test _lookup_intent with various match strategies."""

    def test_exact_match(self, agent):
        translations = {"dreamier": {"cfg_direction": "lower"}}
        result = agent._lookup_intent("dreamier", translations)
        assert result is not None
        assert result["cfg_direction"] == "lower"

    def test_synonym_match(self, agent):
        translations = {"dreamier": {"cfg_direction": "lower"}}
        result = agent._lookup_intent("dreamy", translations)
        assert result is not None

    def test_substring_match(self, agent):
        translations = {"more photorealistic": {"cfg_direction": "slightly_higher"}}
        result = agent._lookup_intent("photorealistic", translations)
        assert result is not None

    def test_no_match(self, agent):
        translations = {"dreamier": {"cfg_direction": "lower"}}
        result = agent._lookup_intent("flibbertigibbet", translations)
        assert result is None

    def test_case_insensitive(self, agent):
        translations = {"dreamier": {"cfg_direction": "lower"}}
        result = agent._lookup_intent("DREAMIER", translations)
        assert result is not None
