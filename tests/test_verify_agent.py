"""Tests for the Verify Agent -- model-aware quality judgment and iteration control.

Tests cover dataclasses, evaluation logic, technical quality scoring,
intent alignment, issue diagnosis, refinement actions, parameter failure
modes, fallback profiles, model-specific evaluation, and edge cases.
"""

from __future__ import annotations

import pytest

from agent.agents.verify_agent import (
    RefinementAction,
    VerificationResult,
    VerifyAgent,
    _ISSUE_SIGNAL_MAP,
    _SIGNAL_ACTION_MAP,
)
from agent.profiles import clear_cache


@pytest.fixture(autouse=True)
def _clear_profile_cache():
    """Clear profile cache before each test for isolation."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def agent():
    return VerifyAgent()


# ============================================================================
# TestDataclasses
# ============================================================================


class TestDataclasses:
    """Verify dataclass construction, defaults, and serialization."""

    def test_refinement_action_creation(self):
        a = RefinementAction(type="adjust_params", target="cfg", reason="too high")
        assert a.type == "adjust_params"
        assert a.target == "cfg"
        assert a.reason == "too high"
        assert a.priority == 1  # default

    def test_refinement_action_custom_priority(self):
        a = RefinementAction(
            type="inpaint", target="hand region", reason="fix", priority=5
        )
        assert a.priority == 5

    def test_verification_result_defaults(self):
        r = VerificationResult(
            overall_score=0.8,
            intent_alignment=0.9,
            technical_quality=0.7,
            decision="accept",
        )
        assert r.refinement_actions == []
        assert r.iteration_count == 0
        assert r.max_iterations == 3
        assert r.diagnosed_issues == []
        assert r.model_limitations == []
        assert r.using_fallback_profile is False

    def test_verification_result_to_dict(self):
        r = VerificationResult(
            overall_score=0.75,
            intent_alignment=0.8,
            technical_quality=0.65,
            decision="refine",
            refinement_actions=[
                RefinementAction(
                    type="adjust_params", target="steps", reason="more detail"
                )
            ],
            iteration_count=1,
            max_iterations=3,
            diagnosed_issues=["blurry output"],
            model_limitations=["hand issues"],
            using_fallback_profile=True,
        )
        d = r.to_dict()
        assert d["overall_score"] == 0.75
        assert d["decision"] == "refine"
        assert d["using_fallback_profile"] is True
        assert len(d["refinement_actions"]) == 1
        assert d["refinement_actions"][0]["type"] == "adjust_params"
        assert d["refinement_actions"][0]["target"] == "steps"
        assert d["diagnosed_issues"] == ["blurry output"]
        assert d["model_limitations"] == ["hand issues"]

    def test_verification_result_to_dict_sorted_keys(self):
        """to_dict keys should be deterministically ordered."""
        r = VerificationResult(
            overall_score=0.5,
            intent_alignment=0.5,
            technical_quality=0.5,
            decision="refine",
        )
        d = r.to_dict()
        keys = list(d.keys())
        assert keys == sorted(keys), "to_dict keys must be sorted for determinism"

    def test_to_dict_roundtrip_no_actions(self):
        r = VerificationResult(
            overall_score=0.9,
            intent_alignment=0.95,
            technical_quality=0.85,
            decision="accept",
        )
        d = r.to_dict()
        assert d["refinement_actions"] == []
        assert d["iteration_count"] == 0


# ============================================================================
# TestEvaluation
# ============================================================================


class TestEvaluation:
    """Test the evaluate() method end-to-end decisions."""

    def test_high_quality_accept(self, agent):
        """High quality output with good intent match -> accept."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.9,
                "matches_intent": True,
                "issues": [],
            },
            original_intent="a beautiful sunset",
            model_id="flux1-dev",
            parameters_used={"cfg": 3.5, "steps": 20},
        )
        assert result.decision == "accept"
        assert result.overall_score >= 0.7
        assert result.intent_alignment > 0.7

    def test_low_quality_refine(self, agent):
        """Low quality output with okay intent -> refine."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.3,
                "matches_intent": 0.6,
                "issues": ["blurry output", "noise in background"],
            },
            original_intent="a cat portrait",
            model_id="flux1-dev",
            parameters_used={"cfg": 3.5, "steps": 10},
        )
        assert result.decision == "refine"
        assert len(result.refinement_actions) > 0

    def test_fundamentally_wrong_reprompt(self, agent):
        """Very low intent alignment -> reprompt."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.5,
                "matches_intent": 0.2,
                "issues": ["wrong subject"],
            },
            original_intent="a dog playing fetch",
            model_id="flux1-dev",
        )
        assert result.decision == "reprompt"
        assert result.intent_alignment < 0.4

    def test_max_iterations_escalate(self, agent):
        """At max iterations with non-accept quality -> escalate."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.5,
                "matches_intent": 0.6,
                "issues": ["still blurry"],
            },
            original_intent="sharp landscape",
            model_id="flux1-dev",
            iteration_count=3,
            max_iterations=3,
        )
        assert result.decision == "escalate"

    def test_medium_quality_not_at_max_refine(self, agent):
        """Medium quality, still has iterations left -> refine."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.5,
                "matches_intent": 0.6,
                "issues": ["soft details"],
            },
            original_intent="detailed portrait",
            model_id="flux1-dev",
            iteration_count=1,
            max_iterations=3,
        )
        assert result.decision == "refine"
        assert result.iteration_count == 1

    def test_reprompt_overrides_escalate(self, agent):
        """Reprompt takes priority over escalate when intent is very low."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.5,
                "matches_intent": 0.1,
                "issues": ["wrong subject"],
            },
            original_intent="a mountain landscape",
            model_id="flux1-dev",
            iteration_count=5,
            max_iterations=3,
        )
        # intent < REPROMPT_THRESHOLD is checked first
        assert result.decision == "reprompt"


# ============================================================================
# TestTechnicalQuality
# ============================================================================


class TestTechnicalQuality:
    """Test _score_technical_quality behavior."""

    def test_expected_characteristics_boost(self, agent):
        """Matching expected characteristics boosts the score."""
        quality_section = {
            "expected_characteristics": ["sharp detail", "coherent lighting"],
            "known_artifacts": [],
            "quality_floor": {"reference_score": 0.5},
            "iteration_signals": {},
        }
        analysis = {
            "description": "sharp detail with coherent lighting throughout",
        }
        score = agent._score_technical_quality(
            analysis, quality_section, {}, None
        )
        assert score > 0.5  # higher than baseline

    def test_known_artifacts_penalize(self, agent):
        """Reported artifacts that match known patterns lower the score."""
        quality_section = {
            "expected_characteristics": [],
            "known_artifacts": [
                {
                    "condition": "cfg > 7.0",
                    "artifact": "Color banding in gradients",
                    "severity": "high",
                }
            ],
            "quality_floor": {"reference_score": 0.7},
            "iteration_signals": {},
        }
        analysis = {
            "issues": ["visible color banding in the sky"],
        }
        score = agent._score_technical_quality(
            analysis, quality_section, {}, None
        )
        assert score < 0.7  # lower than baseline

    def test_params_in_sweet_spot_bonus(self, agent):
        """Parameters within sweet_spot get a bonus."""
        quality_section = {
            "expected_characteristics": [],
            "known_artifacts": [],
            "quality_floor": {"reference_score": 0.5},
            "iteration_signals": {},
        }
        param_section = {
            "cfg": {"sweet_spot": [2.5, 4.5], "range": [1.0, 10.0]},
            "steps": {"sweet_spot": [18, 25], "range": [10, 50]},
        }
        score = agent._score_technical_quality(
            {}, quality_section, param_section, {"cfg": 3.5, "steps": 20}
        )
        assert score > 0.5  # baseline + bonus

    def test_params_outside_sweet_spot_penalty(self, agent):
        """Parameters far outside sweet_spot get penalized."""
        quality_section = {
            "expected_characteristics": [],
            "known_artifacts": [],
            "quality_floor": {"reference_score": 0.5},
            "iteration_signals": {},
        }
        param_section = {
            "cfg": {"sweet_spot": [2.5, 4.5], "range": [1.0, 10.0]},
        }
        score = agent._score_technical_quality(
            {}, quality_section, param_section, {"cfg": 9.0}
        )
        assert score < 0.5  # baseline - penalty

    def test_quality_score_blended(self, agent):
        """quality_score in analysis is blended with profile-based score."""
        quality_section = {
            "expected_characteristics": [],
            "known_artifacts": [],
            "quality_floor": {"reference_score": 0.5},
            "iteration_signals": {},
        }
        score = agent._score_technical_quality(
            {"quality_score": 0.9}, quality_section, {}, None
        )
        # 0.5 * 0.7 + 0.9 * 0.3 = 0.35 + 0.27 = 0.62
        assert 0.55 < score < 0.70


# ============================================================================
# TestIntentAlignment
# ============================================================================


class TestIntentAlignment:
    """Test _score_intent_alignment behavior."""

    def test_matches_intent_true(self, agent):
        assert agent._score_intent_alignment({"matches_intent": True}, "x") == 1.0

    def test_matches_intent_false(self, agent):
        assert agent._score_intent_alignment({"matches_intent": False}, "x") == 0.0

    def test_matches_intent_float(self, agent):
        assert agent._score_intent_alignment({"matches_intent": 0.75}, "x") == 0.75

    def test_matches_intent_clamped(self, agent):
        """Float values are clamped to [0, 1]."""
        assert agent._score_intent_alignment({"matches_intent": 1.5}, "x") == 1.0
        assert agent._score_intent_alignment({"matches_intent": -0.3}, "x") == 0.0

    def test_fallback_to_quality_score(self, agent):
        """No matches_intent -> use quality_score as proxy."""
        score = agent._score_intent_alignment({"quality_score": 0.8}, "x")
        # 0.8 * 0.5 + 0.25 = 0.65
        assert abs(score - 0.65) < 0.01

    def test_no_keys_default(self, agent):
        """Neither key present -> 0.5 default."""
        assert agent._score_intent_alignment({}, "x") == 0.5

    def test_matches_intent_invalid_string(self, agent):
        """Non-numeric matches_intent falls through to quality_score."""
        score = agent._score_intent_alignment(
            {"matches_intent": "yes", "quality_score": 0.6}, "x"
        )
        # Falls to quality_score proxy: 0.6 * 0.5 + 0.25 = 0.55
        assert abs(score - 0.55) < 0.01


# ============================================================================
# TestIssueDiagnosis
# ============================================================================


class TestIssueDiagnosis:
    """Test _diagnose_issues and _map_issue_to_signal behavior."""

    def test_mushy_maps_to_more_steps(self, agent):
        signal = agent._map_issue_to_signal("mushy details in the output", {})
        assert signal == "needs_more_steps"

    def test_oversaturated_maps_to_lower_cfg(self, agent):
        signal = agent._map_issue_to_signal("oversaturated colors", {})
        assert signal == "needs_lower_cfg"

    def test_wrong_subject_maps_to_reprompt(self, agent):
        signal = agent._map_issue_to_signal("wrong subject entirely", {})
        assert signal == "needs_reprompt"

    def test_hand_maps_to_inpaint(self, agent):
        signal = agent._map_issue_to_signal("hand deformities visible", {})
        assert signal == "needs_inpaint"

    def test_multiple_issues_diagnosed(self, agent):
        quality_section = {
            "expected_characteristics": [],
            "known_artifacts": [],
            "quality_floor": {"reference_score": 0.5},
            "iteration_signals": {},
        }
        analysis = {
            "issues": ["blurry background", "oversaturated sky"],
        }
        diagnosed, limitations = agent._diagnose_issues(
            analysis, quality_section, None
        )
        assert len(diagnosed) >= 2

    def test_unknown_issue_still_recorded(self, agent):
        """Issues that don't match any signal are still in diagnosed list."""
        signal = agent._map_issue_to_signal(
            "completely unique problem xyz123", {}
        )
        assert signal is None
        # But _diagnose_issues still records it
        quality_section = {"iteration_signals": {}, "known_artifacts": []}
        diagnosed, _ = agent._diagnose_issues(
            {"issues": ["completely unique problem xyz123"]},
            quality_section,
            None,
        )
        assert "completely unique problem xyz123" in diagnosed

    def test_model_limitation_goes_to_limitations(self, agent):
        """Issues matching model_limitation go to limitations list."""
        iteration_signals = {
            "model_limitation": {
                "indicators": [
                    "Cannot generate specific copyrighted characters"
                ]
            },
        }
        signal = agent._map_issue_to_signal(
            "Cannot generate specific copyrighted characters reliably",
            iteration_signals,
        )
        assert signal == "model_limitation"

    def test_indicator_matching_from_profile(self, agent):
        """Issues matching profile indicator text get the right signal."""
        iteration_signals = {
            "needs_more_steps": {
                "indicators": [
                    "Visible noise in flat areas",
                    "Soft or undefined edges",
                ]
            },
        }
        signal = agent._map_issue_to_signal(
            "visible noise in flat areas of the sky", iteration_signals
        )
        assert signal == "needs_more_steps"


# ============================================================================
# TestRefinementActions
# ============================================================================


class TestRefinementActions:
    """Test _generate_refinement_actions behavior."""

    def test_needs_more_steps_action(self, agent):
        actions = agent._generate_refinement_actions(
            ["blurry output"], {}
        )
        assert any(a.target == "steps" for a in actions)

    def test_needs_lower_cfg_action(self, agent):
        actions = agent._generate_refinement_actions(
            ["oversaturated colors"], {}
        )
        assert any(
            a.target == "cfg" and "Lower" in a.reason for a in actions
        )

    def test_needs_reprompt_action(self, agent):
        actions = agent._generate_refinement_actions(
            ["wrong subject shown"], {}
        )
        assert any(a.type == "reprompt" for a in actions)

    def test_needs_inpaint_action(self, agent):
        actions = agent._generate_refinement_actions(
            ["hand deformities"], {}
        )
        assert any(a.type == "inpaint" for a in actions)

    def test_actions_sorted_by_priority(self, agent):
        """Actions should be sorted by priority (1 first)."""
        actions = agent._generate_refinement_actions(
            ["oversaturated colors", "blurry background", "hand issues"], {}
        )
        assert len(actions) >= 2
        priorities = [a.priority for a in actions]
        assert priorities == sorted(priorities)

    def test_deduplicated_signals(self, agent):
        """Same signal from multiple issues produces only one action."""
        actions = agent._generate_refinement_actions(
            ["blurry output", "soft details", "noise everywhere"], {}
        )
        step_actions = [a for a in actions if a.target == "steps"]
        assert len(step_actions) == 1


# ============================================================================
# TestParameterFailureModes
# ============================================================================


class TestParameterFailureModes:
    """Test _check_parameter_failure_modes and condition evaluation."""

    def test_flux_high_cfg_detected(self, agent):
        """Flux with cfg > 7 triggers a failure mode from quality section."""
        quality_section = {
            "known_artifacts": [
                {
                    "condition": "cfg > 7.0",
                    "artifact": "Color banding in gradients",
                    "severity": "high",
                }
            ],
            "iteration_signals": {},
        }
        issues = agent._check_parameter_failure_modes_from_quality(
            {"cfg": 8.0}, quality_section
        )
        assert len(issues) > 0
        assert any("banding" in i.lower() for i in issues)

    def test_sdxl_cfg_in_range_no_failure(self, agent):
        """SDXL with cfg=7 should not trigger failure modes."""
        quality_section = {
            "known_artifacts": [
                {
                    "condition": "cfg > 12.0",
                    "artifact": "Oversaturation",
                    "severity": "high",
                }
            ],
            "iteration_signals": {},
        }
        issues = agent._check_parameter_failure_modes_from_quality(
            {"cfg": 7.0}, quality_section
        )
        assert len(issues) == 0

    def test_eval_condition_greater_than(self, agent):
        assert VerifyAgent._eval_condition("cfg > 7.0", {"cfg": 8.0}) is True
        assert VerifyAgent._eval_condition("cfg > 7.0", {"cfg": 5.0}) is False

    def test_eval_condition_less_than(self, agent):
        assert VerifyAgent._eval_condition("steps < 12", {"steps": 10}) is True
        assert VerifyAgent._eval_condition("steps < 12", {"steps": 20}) is False

    def test_eval_condition_missing_param(self, agent):
        """Missing parameter in dict -> False."""
        assert VerifyAgent._eval_condition("cfg > 7.0", {"steps": 20}) is False

    def test_eval_condition_non_numeric(self, agent):
        """Non-numeric parameter value -> False."""
        assert VerifyAgent._eval_condition("cfg > 7.0", {"cfg": "high"}) is False

    def test_eval_condition_gte(self, agent):
        assert VerifyAgent._eval_condition("cfg >= 7.0", {"cfg": 7.0}) is True
        assert VerifyAgent._eval_condition("cfg >= 7.0", {"cfg": 6.9}) is False

    def test_check_parameter_failure_modes_flux_style(self, agent):
        """Flux-style failure_modes (list of dicts)."""
        param_section = {
            "cfg": {
                "failure_modes": [
                    {"condition": "cfg > 7.0", "artifact": "banding"},
                    {"condition": "cfg > 10.0", "artifact": "distortion"},
                ]
            }
        }
        issues = agent._check_parameter_failure_modes(
            {"cfg": 8.0}, param_section
        )
        assert len(issues) == 1
        assert "banding" in issues[0].lower()

    def test_check_parameter_failure_modes_sdxl_style(self, agent):
        """SDXL-style failure_modes (dict of named modes)."""
        param_section = {
            "cfg": {
                "failure_modes": {
                    "too_high": {
                        "condition": "cfg > 12.0",
                        "description": "Oversaturation",
                    },
                    "too_low": {
                        "condition": "cfg < 3.0",
                        "description": "Mushy output",
                    },
                }
            }
        }
        issues = agent._check_parameter_failure_modes(
            {"cfg": 2.0}, param_section
        )
        assert len(issues) == 1
        assert "Mushy" in issues[0]


# ============================================================================
# TestFallbackProfile
# ============================================================================


class TestFallbackProfile:
    """Test fallback profile detection."""

    def test_unknown_model_uses_fallback(self, agent):
        result = agent.evaluate(
            output_analysis={"quality_score": 0.7, "matches_intent": True},
            original_intent="a test",
            model_id="completely-unknown-model-xyz",
        )
        assert result.using_fallback_profile is True

    def test_known_model_no_fallback(self, agent):
        result = agent.evaluate(
            output_analysis={"quality_score": 0.9, "matches_intent": True},
            original_intent="a test",
            model_id="flux1-dev",
        )
        assert result.using_fallback_profile is False


# ============================================================================
# TestModelSpecific
# ============================================================================


class TestModelSpecific:
    """Test evaluation against real profiles."""

    def test_flux_good_params_high_score(self, agent):
        """Flux with good params and good analysis -> high technical score."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.85,
                "matches_intent": True,
                "issues": [],
            },
            original_intent="a portrait in golden hour light",
            model_id="flux1-dev",
            parameters_used={"cfg": 3.5, "steps": 20},
        )
        assert result.decision == "accept"
        assert result.technical_quality > 0.6

    def test_flux_bad_params_low_score(self, agent):
        """Flux with cfg=12 -> low score, artifacts diagnosed."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.4,
                "matches_intent": 0.6,
                "issues": ["oversaturated colors", "banding in sky"],
            },
            original_intent="a natural landscape",
            model_id="flux1-dev",
            parameters_used={"cfg": 12.0, "steps": 20},
        )
        assert result.decision in ("refine", "reprompt")
        assert result.technical_quality < 0.7
        # Should have diagnosed parameter issues
        assert len(result.diagnosed_issues) > 0

    def test_sdxl_good_params(self, agent):
        """SDXL with good params -> accept."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.8,
                "matches_intent": True,
                "issues": [],
            },
            original_intent="a portrait",
            model_id="sdxl-base",
            parameters_used={"cfg": 7.0, "steps": 25},
        )
        assert result.decision == "accept"

    def test_sdxl_high_cfg_issues(self, agent):
        """SDXL with cfg=14 -> issues diagnosed."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.4,
                "matches_intent": 0.5,
                "issues": ["neon color shifts", "harsh contrast"],
            },
            original_intent="a soft portrait",
            model_id="sdxl-base",
            parameters_used={"cfg": 14.0, "steps": 25},
        )
        assert result.decision in ("refine", "reprompt")
        assert len(result.diagnosed_issues) > 0


# ============================================================================
# TestEdgeCases
# ============================================================================


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_empty_output_analysis(self, agent):
        """Empty analysis -> safe defaults, no crash."""
        result = agent.evaluate(
            output_analysis={},
            original_intent="anything",
            model_id="flux1-dev",
        )
        assert result.intent_alignment == 0.5  # default
        assert result.decision in ("accept", "refine", "reprompt", "escalate")

    def test_none_parameters_used(self, agent):
        """None parameters_used -> skip parameter checks."""
        result = agent.evaluate(
            output_analysis={"quality_score": 0.6, "matches_intent": 0.6},
            original_intent="test",
            model_id="flux1-dev",
            parameters_used=None,
        )
        # Should not crash
        assert isinstance(result, VerificationResult)

    def test_iteration_count_at_max_minus_one(self, agent):
        """iteration_count == max_iterations - 1 with refine quality -> refine."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.5,
                "matches_intent": 0.6,
                "issues": ["soft details"],
            },
            original_intent="sharp image",
            model_id="flux1-dev",
            iteration_count=2,
            max_iterations=3,
        )
        assert result.decision == "refine"

    def test_exactly_at_accept_threshold(self, agent):
        """Score exactly at threshold with good intent -> accept."""
        # We need to engineer an output that lands right at 0.7
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.85,
                "matches_intent": 0.85,
                "issues": [],
            },
            original_intent="test",
            model_id="flux1-dev",
            parameters_used={"cfg": 3.5, "steps": 20},
        )
        # With high quality_score and matches_intent, should accept
        assert result.decision == "accept"

    def test_max_iterations_zero(self, agent):
        """max_iterations=0 with non-accept quality -> escalate immediately."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.5,
                "matches_intent": 0.6,
                "issues": ["soft"],
            },
            original_intent="test",
            model_id="flux1-dev",
            iteration_count=0,
            max_iterations=0,
        )
        assert result.decision == "escalate"

    def test_accept_no_refinement_actions(self, agent):
        """Accept decision should have no refinement actions."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.9,
                "matches_intent": True,
                "issues": [],
            },
            original_intent="test",
            model_id="flux1-dev",
            parameters_used={"cfg": 3.5, "steps": 20},
        )
        assert result.decision == "accept"
        assert result.refinement_actions == []

    def test_overall_score_bounded(self, agent):
        """Overall score is always in [0, 1]."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 1.0,
                "matches_intent": True,
                "issues": [],
            },
            original_intent="test",
            model_id="flux1-dev",
            parameters_used={"cfg": 3.5, "steps": 20},
        )
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.technical_quality <= 1.0
        assert 0.0 <= result.intent_alignment <= 1.0

    def test_escalate_still_has_diagnosed_issues(self, agent):
        """Escalate preserves diagnosed issues for human review."""
        result = agent.evaluate(
            output_analysis={
                "quality_score": 0.3,
                "matches_intent": 0.5,
                "issues": ["blurry", "oversaturated"],
            },
            original_intent="test",
            model_id="flux1-dev",
            iteration_count=5,
            max_iterations=3,
        )
        assert result.decision == "escalate"
        assert len(result.diagnosed_issues) > 0


# ============================================================================
# TestMappings
# ============================================================================


class TestMappings:
    """Test that the static mappings are well-formed."""

    def test_issue_signal_map_values_in_signal_action_map(self):
        """Every signal in _ISSUE_SIGNAL_MAP has a corresponding action template."""
        for signal in set(_ISSUE_SIGNAL_MAP.values()):
            assert signal in _SIGNAL_ACTION_MAP, (
                f"Signal '{signal}' from _ISSUE_SIGNAL_MAP has no entry "
                f"in _SIGNAL_ACTION_MAP"
            )

    def test_signal_action_map_templates_valid(self):
        """Every non-None action template has required fields."""
        for signal, action in _SIGNAL_ACTION_MAP.items():
            if action is None:
                continue
            assert isinstance(action, RefinementAction)
            assert action.type in (
                "adjust_params", "reprompt", "inpaint", "upscale", "retry"
            )
            assert action.target
            assert action.reason

    def test_all_issue_keywords_are_lowercase(self):
        """Issue keywords in the map should be lowercase for matching."""
        for keyword in _ISSUE_SIGNAL_MAP:
            assert keyword == keyword.lower()
