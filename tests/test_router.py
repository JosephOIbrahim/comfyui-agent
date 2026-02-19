"""Tests for the Router — authority delegation and loop control.

The Router is a lightweight sequencer. These tests verify:
- Intent classification
- Delegation sequencing
- Precondition checking
- Confidence gating
- Loop control (should_continue)
- Authority validation
- Exception handling
- Full pipeline execution
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.agents.intent_agent import IntentAgent, IntentSpecification, ParameterMutation
from agent.agents.router import (
    AUTHORITY_RULES,
    DELEGATION_SEQUENCES,
    ROUTER_EXCEPTIONS,
    Router,
    RouterContext,
)
from agent.agents.verify_agent import RefinementAction, VerificationResult, VerifyAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router():
    """A Router with default agents."""
    return Router()


@pytest.fixture
def router_custom_max():
    """A Router with max_iterations=5."""
    return Router(max_iterations=5)


@pytest.fixture
def mock_intent_agent():
    """A mocked IntentAgent."""
    return MagicMock(spec=IntentAgent)


@pytest.fixture
def mock_verify_agent():
    """A mocked VerifyAgent."""
    return MagicMock(spec=VerifyAgent)


def _make_intent_spec(
    model_id: str = "sdxl-base",
    confidence: float = 0.9,
    fallback: bool = False,
) -> IntentSpecification:
    """Helper to create an IntentSpecification."""
    return IntentSpecification(
        model_id=model_id,
        parameter_mutations=[
            ParameterMutation(
                target="KSampler.cfg",
                action="set",
                value=5.0,
                reason="test",
            ),
        ],
        confidence=confidence,
        using_fallback_profile=fallback,
    )


def _make_verification(
    decision: str = "accept",
    overall: float = 0.85,
    intent: float = 0.9,
    technical: float = 0.8,
    refinement_actions: list | None = None,
) -> VerificationResult:
    """Helper to create a VerificationResult."""
    return VerificationResult(
        overall_score=overall,
        intent_alignment=intent,
        technical_quality=technical,
        decision=decision,
        refinement_actions=refinement_actions or [],
    )


# ===========================================================================
# TestRouterContext
# ===========================================================================


class TestRouterContext:
    """RouterContext creation and serialization."""

    def test_creation_with_defaults(self):
        ctx = RouterContext(
            user_intent="make it dreamier",
            intent_type="modification",
            active_model="sdxl-base",
            workflow_state="configured",
        )
        assert ctx.iteration_count == 0
        assert ctx.max_iterations == 3
        assert ctx.history == []
        assert ctx.schemas == {
            "intent": "default",
            "execution": "default",
            "verify": "default",
        }

    def test_creation_with_custom_schemas(self):
        ctx = RouterContext(
            user_intent="test",
            intent_type="generation",
            active_model="flux1-dev",
            workflow_state="empty",
            schemas={"intent": "flux_optimized", "verify": "strict"},
        )
        assert ctx.schemas["intent"] == "flux_optimized"
        assert ctx.schemas["verify"] == "strict"

    def test_to_dict_roundtrip(self):
        ctx = RouterContext(
            user_intent="dreamier",
            intent_type="modification",
            active_model="sdxl-base",
            workflow_state="configured",
            iteration_count=2,
            max_iterations=5,
            history=[{"iteration": 1}],
            schemas={"intent": "custom"},
        )
        d = ctx.to_dict()
        assert d["user_intent"] == "dreamier"
        assert d["intent_type"] == "modification"
        assert d["active_model"] == "sdxl-base"
        assert d["workflow_state"] == "configured"
        assert d["iteration_count"] == 2
        assert d["max_iterations"] == 5
        assert d["history"] == [{"iteration": 1}]
        assert d["schemas"] == {"intent": "custom"}

    def test_to_dict_keys_are_sorted(self):
        """Verify dict keys come out in alphabetical order (He2025)."""
        ctx = RouterContext(
            user_intent="test",
            intent_type="modification",
            active_model="sdxl-base",
            workflow_state="configured",
        )
        d = ctx.to_dict()
        keys = list(d.keys())
        assert keys == sorted(keys)


# ===========================================================================
# TestIntentClassification
# ===========================================================================


class TestIntentClassification:
    """Router.classify_intent keyword matching."""

    def test_generation_create_with_empty_state(self, router):
        assert router.classify_intent("create a new workflow", "empty") == "generation"

    def test_generation_from_scratch(self, router):
        assert router.classify_intent("make from scratch", "configured") == "generation"

    def test_generation_generate_with_empty(self, router):
        assert router.classify_intent("generate an image", "empty") == "generation"

    def test_modification_default(self, router):
        assert router.classify_intent("make it dreamier", "configured") == "modification"

    def test_modification_is_default_fallback(self, router):
        assert router.classify_intent("adjust the colors", "configured") == "modification"

    def test_evaluation_with_has_output(self, router):
        assert router.classify_intent("how good is this", "has_output") == "evaluation"

    def test_evaluation_evaluate_keyword(self, router):
        assert router.classify_intent("evaluate quality", "has_output") == "evaluation"

    def test_evaluation_without_output_falls_to_modification(self, router):
        """Evaluation intent without output -> modification (no output to judge)."""
        assert router.classify_intent("evaluate quality", "configured") == "modification"

    def test_exploration_what_would(self, router):
        assert router.classify_intent("what would happen if I used flux", "configured") == "exploration"

    def test_exploration_translate(self, router):
        assert router.classify_intent("translate this intent", "configured") == "exploration"

    def test_exploration_plan(self, router):
        assert router.classify_intent("plan a workflow for portraits", "configured") == "exploration"

    def test_unknown_intent_defaults_to_modification(self, router):
        assert router.classify_intent("xyzzy", "configured") == "modification"

    def test_empty_intent_defaults_to_modification(self, router):
        assert router.classify_intent("", "configured") == "modification"

    def test_none_safe_empty_intent(self, router):
        """Empty string should not crash."""
        result = router.classify_intent("", "empty")
        assert result == "modification"

    def test_generation_new_workflow(self, router):
        result = router.classify_intent("start a new workflow for textures", "empty")
        assert result == "generation"

    def test_start_fresh_forces_generation(self, router):
        """'start fresh' triggers generation even when not empty."""
        result = router.classify_intent("start fresh with a new approach", "configured")
        assert result == "generation"


# ===========================================================================
# TestDelegationSequence
# ===========================================================================


class TestDelegationSequence:
    """Router.get_delegation_sequence."""

    def test_generation_sequence(self, router):
        assert router.get_delegation_sequence("generation") == ["intent", "execution", "verify"]

    def test_modification_sequence(self, router):
        assert router.get_delegation_sequence("modification") == ["intent", "execution", "verify"]

    def test_evaluation_sequence(self, router):
        assert router.get_delegation_sequence("evaluation") == ["verify"]

    def test_exploration_sequence(self, router):
        assert router.get_delegation_sequence("exploration") == ["intent"]

    def test_unknown_type_falls_back_to_modification(self, router):
        assert router.get_delegation_sequence("unknown_type") == ["intent", "execution", "verify"]

    def test_delegation_sequences_constant_integrity(self):
        """Verify the constant has all four intent types."""
        assert set(DELEGATION_SEQUENCES.keys()) == {
            "generation", "modification", "evaluation", "exploration",
        }


# ===========================================================================
# TestPreconditions
# ===========================================================================


class TestPreconditions:
    """Router.check_preconditions."""

    @patch("agent.agents.router.is_fallback", return_value=False)
    @patch("agent.agents.router.list_schemas", return_value=["default"])
    def test_no_warnings_for_known_model(self, _mock_ls, _mock_fb, router):
        ctx = RouterContext(
            user_intent="test",
            intent_type="modification",
            active_model="sdxl-base",
            workflow_state="configured",
        )
        warnings = router.check_preconditions(ctx)
        assert warnings == []

    @patch("agent.agents.router.is_fallback", return_value=True)
    @patch("agent.agents.router.list_schemas", return_value=["default"])
    def test_fallback_model_warning(self, _mock_ls, _mock_fb, router):
        ctx = RouterContext(
            user_intent="test",
            intent_type="modification",
            active_model="unknown-model",
            workflow_state="configured",
        )
        warnings = router.check_preconditions(ctx)
        assert any("fallback" in w.lower() for w in warnings)

    @patch("agent.agents.router.is_fallback", return_value=False)
    @patch("agent.agents.router.list_schemas", return_value=["default"])
    def test_evaluation_without_output_warning(self, _mock_ls, _mock_fb, router):
        ctx = RouterContext(
            user_intent="evaluate",
            intent_type="evaluation",
            active_model="sdxl-base",
            workflow_state="empty",
        )
        warnings = router.check_preconditions(ctx)
        assert any("has_output" in w for w in warnings)

    @patch("agent.agents.router.is_fallback", return_value=False)
    @patch("agent.agents.router.list_schemas", return_value=["default"])
    def test_invalid_schema_warning(self, mock_ls, _mock_fb, router):
        ctx = RouterContext(
            user_intent="test",
            intent_type="modification",
            active_model="sdxl-base",
            workflow_state="configured",
            schemas={"intent": "nonexistent_schema", "execution": "default", "verify": "default"},
        )
        warnings = router.check_preconditions(ctx)
        assert any("nonexistent_schema" in w for w in warnings)

    @patch("agent.agents.router.is_fallback", return_value=False)
    @patch("agent.agents.router.list_schemas", return_value=["default"])
    def test_valid_schemas_no_warning(self, _mock_ls, _mock_fb, router):
        ctx = RouterContext(
            user_intent="test",
            intent_type="modification",
            active_model="sdxl-base",
            workflow_state="configured",
        )
        warnings = router.check_preconditions(ctx)
        assert not any("schema" in w.lower() for w in warnings)

    @patch("agent.agents.router.is_fallback", return_value=False)
    @patch("agent.agents.router.list_schemas", return_value=["default"])
    def test_generation_while_executing_warning(self, _mock_ls, _mock_fb, router):
        ctx = RouterContext(
            user_intent="create new",
            intent_type="generation",
            active_model="sdxl-base",
            workflow_state="executing",
        )
        warnings = router.check_preconditions(ctx)
        assert any("executing" in w.lower() for w in warnings)


# ===========================================================================
# TestConfidenceCheck
# ===========================================================================


class TestConfidenceCheck:
    """Router.check_confidence."""

    def test_high_confidence_proceeds(self, router):
        spec = _make_intent_spec(confidence=0.9)
        result = router.check_confidence(spec)
        assert result["proceed"] is True
        assert result["confidence"] == 0.9

    def test_low_confidence_blocks(self, router):
        spec = _make_intent_spec(confidence=0.3)
        result = router.check_confidence(spec)
        assert result["proceed"] is False
        assert "clarification" in result["reason"].lower()

    def test_threshold_boundary_proceeds(self, router):
        """Exactly 0.5 should proceed (>=)."""
        spec = _make_intent_spec(confidence=0.5)
        result = router.check_confidence(spec)
        assert result["proceed"] is True

    def test_just_below_threshold_blocks(self, router):
        spec = _make_intent_spec(confidence=0.49)
        result = router.check_confidence(spec)
        assert result["proceed"] is False

    def test_custom_threshold(self, router):
        spec = _make_intent_spec(confidence=0.7)
        result = router.check_confidence(spec, threshold=0.8)
        assert result["proceed"] is False

    def test_custom_threshold_met(self, router):
        spec = _make_intent_spec(confidence=0.85)
        result = router.check_confidence(spec, threshold=0.8)
        assert result["proceed"] is True


# ===========================================================================
# TestShouldContinue
# ===========================================================================


class TestShouldContinue:
    """Router.should_continue loop control."""

    def test_accept_stops_loop(self, router):
        v = _make_verification(decision="accept")
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
        )
        result = router.should_continue(v, ctx)
        assert result["continue"] is False
        assert result["action"] == "accept"
        assert result["refinement_actions"] is None

    def test_escalate_stops_loop(self, router):
        v = _make_verification(decision="escalate")
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
        )
        result = router.should_continue(v, ctx)
        assert result["continue"] is False
        assert result["action"] == "escalate"

    def test_refine_at_iteration_0_continues(self, router):
        actions = [RefinementAction(type="adjust_params", target="cfg", reason="lower cfg")]
        v = _make_verification(decision="refine", refinement_actions=actions)
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
            iteration_count=0, max_iterations=3,
        )
        result = router.should_continue(v, ctx)
        assert result["continue"] is True
        assert result["action"] == "refine"
        assert result["refinement_actions"] is not None
        assert len(result["refinement_actions"]) == 1

    def test_reprompt_at_iteration_0_continues(self, router):
        v = _make_verification(decision="reprompt")
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
            iteration_count=0, max_iterations=3,
        )
        result = router.should_continue(v, ctx)
        assert result["continue"] is True
        assert result["action"] == "reprompt"

    def test_refine_at_max_iterations_escalates(self, router):
        v = _make_verification(decision="refine")
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
            iteration_count=3, max_iterations=3,
        )
        result = router.should_continue(v, ctx)
        assert result["continue"] is False
        assert result["action"] == "escalate"
        assert "max iterations" in result["reason"].lower()

    def test_reprompt_at_max_iterations_escalates(self, router):
        v = _make_verification(decision="reprompt")
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
            iteration_count=3, max_iterations=3,
        )
        result = router.should_continue(v, ctx)
        assert result["continue"] is False
        assert result["action"] == "escalate"

    def test_refine_no_actions_still_continues(self, router):
        """Refine with empty actions list still continues."""
        v = _make_verification(decision="refine", refinement_actions=[])
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
            iteration_count=1, max_iterations=3,
        )
        result = router.should_continue(v, ctx)
        assert result["continue"] is True
        assert result["action"] == "refine"


# ===========================================================================
# TestRecordIteration
# ===========================================================================


class TestRecordIteration:
    """Router.record_iteration state tracking."""

    def test_records_result_in_history(self, router):
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
        )
        router.record_iteration(ctx, {"score": 0.7})
        assert len(ctx.history) == 1
        assert ctx.history[0] == {"score": 0.7}

    def test_increments_iteration_count(self, router):
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
        )
        assert ctx.iteration_count == 0
        router.record_iteration(ctx, {"iter": 1})
        assert ctx.iteration_count == 1

    def test_multiple_iterations_accumulate(self, router):
        ctx = RouterContext(
            user_intent="test", intent_type="modification",
            active_model="sdxl-base", workflow_state="configured",
        )
        router.record_iteration(ctx, {"iter": 1})
        router.record_iteration(ctx, {"iter": 2})
        router.record_iteration(ctx, {"iter": 3})
        assert ctx.iteration_count == 3
        assert len(ctx.history) == 3


# ===========================================================================
# TestAuthorityValidation
# ===========================================================================


class TestAuthorityValidation:
    """Router.validate_authority enforcement."""

    def test_intent_owns_parameter_decisions(self, router):
        assert router.validate_authority("intent", "parameter_decisions") is True

    def test_intent_cannot_execute_workflows(self, router):
        assert router.validate_authority("intent", "execute_workflows") is False

    def test_intent_cannot_call_comfyui_api(self, router):
        assert router.validate_authority("intent", "call_comfyui_api") is False

    def test_execution_owns_workflow_mutation(self, router):
        assert router.validate_authority("execution", "workflow_mutation") is True

    def test_execution_cannot_judge_output_quality(self, router):
        assert router.validate_authority("execution", "judge_output_quality") is False

    def test_verify_owns_quality_judgment(self, router):
        assert router.validate_authority("verify", "quality_judgment") is True

    def test_verify_cannot_modify_prompts(self, router):
        assert router.validate_authority("verify", "modify_prompts") is False

    def test_verify_cannot_execute_workflows(self, router):
        assert router.validate_authority("verify", "execute_workflows") is False

    def test_verify_can_recommend_reprompt(self, router):
        assert router.validate_authority("verify", "reprompt") is True

    def test_verify_can_recommend_param_adjustments(self, router):
        assert router.validate_authority("verify", "param_adjustments") is True

    def test_unknown_agent_denied(self, router):
        assert router.validate_authority("nonexistent_agent", "anything") is False

    def test_unknown_action_denied(self, router):
        assert router.validate_authority("intent", "nonexistent_action") is False

    def test_authority_rules_constant_integrity(self):
        """Verify AUTHORITY_RULES has all three agents."""
        assert set(AUTHORITY_RULES.keys()) == {"intent", "execution", "verify"}
        for agent_rules in AUTHORITY_RULES.values():
            assert "owns" in agent_rules
            assert "cannot" in agent_rules


# ===========================================================================
# TestExceptionHandling
# ===========================================================================


class TestExceptionHandling:
    """Router.get_exception_action."""

    def test_timeout_action(self, router):
        assert router.get_exception_action("timeout") == "abort_and_return_partial"

    def test_loop_limit_action(self, router):
        assert router.get_exception_action("loop_limit") == "escalate_with_best_attempt"

    def test_model_not_found_action(self, router):
        assert router.get_exception_action("model_not_found") == "ask_user_to_select_model"

    def test_profile_fallback_action(self, router):
        assert router.get_exception_action("profile_fallback") == "warn_and_proceed"

    def test_low_confidence_action(self, router):
        assert router.get_exception_action("low_confidence") == "ask_user_for_clarification"

    def test_unknown_exception_defaults(self, router):
        assert router.get_exception_action("totally_unknown") == "escalate_with_best_attempt"

    def test_router_exceptions_constant_integrity(self):
        """All exception types have condition and action."""
        for exc_type, exc_data in ROUTER_EXCEPTIONS.items():
            assert "condition" in exc_data, f"{exc_type} missing condition"
            assert "action" in exc_data, f"{exc_type} missing action"


# ===========================================================================
# TestRunPipeline
# ===========================================================================


class TestRunPipeline:
    """Router.run_pipeline integration."""

    def test_modification_returns_intent_spec(self, mock_intent_agent, mock_verify_agent):
        spec = _make_intent_spec(confidence=0.9)
        mock_intent_agent.translate.return_value = spec

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="make it dreamier",
            model_id="sdxl-base",
        )
        assert result["status"] == "planned"
        assert result["intent_spec"] is not None
        assert result["verification"] is None
        mock_intent_agent.translate.assert_called_once()

    def test_evaluation_returns_verification(self, mock_intent_agent, mock_verify_agent):
        v = _make_verification(decision="accept")
        mock_verify_agent.evaluate.return_value = v

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="evaluate quality",
            model_id="sdxl-base",
            workflow_state={"state": "has_output"},
            output_analysis={"quality_score": 0.8, "matches_intent": True},
        )
        assert result["status"] == "evaluated"
        assert result["verification"] is not None
        assert result["intent_spec"] is None
        mock_verify_agent.evaluate.assert_called_once()

    def test_exploration_returns_intent_spec(self, mock_intent_agent, mock_verify_agent):
        spec = _make_intent_spec(confidence=0.8)
        mock_intent_agent.translate.return_value = spec

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="what would happen if I used flux",
            model_id="flux1-dev",
        )
        assert result["status"] == "planned"
        assert result["intent_spec"] is not None

    def test_low_confidence_needs_clarification(self, mock_intent_agent, mock_verify_agent):
        spec = _make_intent_spec(confidence=0.3)
        mock_intent_agent.translate.return_value = spec

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="xyzzy",
            model_id="sdxl-base",
        )
        assert result["status"] == "needs_clarification"

    def test_evaluation_without_analysis_errors(self, mock_intent_agent, mock_verify_agent):
        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="evaluate quality",
            model_id="sdxl-base",
            workflow_state={"state": "has_output"},
            output_analysis=None,
        )
        assert result["status"] == "error"
        assert any("output_analysis" in w for w in result["precondition_warnings"])

    @patch("agent.agents.router.is_fallback", return_value=True)
    @patch("agent.agents.router.list_schemas", return_value=["default"])
    def test_unknown_model_has_precondition_warnings(
        self, _mock_ls, _mock_fb, mock_intent_agent, mock_verify_agent,
    ):
        spec = _make_intent_spec(confidence=0.7, fallback=True)
        mock_intent_agent.translate.return_value = spec

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="make it sharper",
            model_id="totally-unknown-model",
        )
        assert any("fallback" in w.lower() for w in result["precondition_warnings"])

    def test_pipeline_delegation_sequence_included(self, mock_intent_agent, mock_verify_agent):
        spec = _make_intent_spec(confidence=0.9)
        mock_intent_agent.translate.return_value = spec

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="make it dreamier",
            model_id="sdxl-base",
        )
        assert "delegation_sequence" in result
        assert result["delegation_sequence"] == ["intent", "execution", "verify"]

    def test_pipeline_context_included(self, mock_intent_agent, mock_verify_agent):
        spec = _make_intent_spec(confidence=0.9)
        mock_intent_agent.translate.return_value = spec

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="dreamier",
            model_id="sdxl-base",
        )
        assert "context" in result
        assert result["context"]["active_model"] == "sdxl-base"

    def test_pipeline_with_custom_schemas(self, mock_intent_agent, mock_verify_agent):
        spec = _make_intent_spec(confidence=0.9)
        mock_intent_agent.translate.return_value = spec

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="dreamier",
            model_id="sdxl-base",
            schemas={"intent": "custom", "execution": "default", "verify": "default"},
        )
        assert result["context"]["schemas"]["intent"] == "custom"

    def test_pipeline_agent_error_caught(self, mock_intent_agent, mock_verify_agent):
        mock_intent_agent.translate.side_effect = RuntimeError("Agent exploded")

        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
        )
        result = r.run_pipeline(
            user_intent="dreamier",
            model_id="sdxl-base",
        )
        assert result["status"] == "error"
        assert any("Agent exploded" in w for w in result["precondition_warnings"])


# ===========================================================================
# TestSchemaAwareness
# ===========================================================================


class TestSchemaAwareness:
    """Schema propagation through the router."""

    def test_custom_schemas_propagate_to_context(self, router):
        ctx = router.create_context(
            user_intent="test",
            active_model="sdxl-base",
            schemas={"intent": "flux", "execution": "fast", "verify": "strict"},
        )
        assert ctx.schemas == {"intent": "flux", "execution": "fast", "verify": "strict"}

    def test_default_schemas_when_none(self, router):
        ctx = router.create_context(
            user_intent="test",
            active_model="sdxl-base",
        )
        assert ctx.schemas == {
            "intent": "default",
            "execution": "default",
            "verify": "default",
        }


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_user_intent(self, router):
        ctx = router.create_context(
            user_intent="",
            active_model="sdxl-base",
        )
        assert ctx.intent_type == "modification"  # default

    def test_very_long_intent_string(self, router):
        long_intent = "make it dreamier " * 500
        ctx = router.create_context(
            user_intent=long_intent,
            active_model="sdxl-base",
        )
        assert ctx.intent_type == "modification"
        assert ctx.user_intent == long_intent

    def test_router_with_custom_agents(self, mock_intent_agent, mock_verify_agent):
        r = Router(
            intent_agent=mock_intent_agent,
            verify_agent=mock_verify_agent,
            max_iterations=10,
        )
        assert r.intent_agent is mock_intent_agent
        assert r.verify_agent is mock_verify_agent
        assert r.max_iterations == 10

    def test_router_default_agents(self, router):
        assert isinstance(router.intent_agent, IntentAgent)
        assert isinstance(router.verify_agent, VerifyAgent)
        assert router.max_iterations == 3

    def test_custom_max_iterations(self, router_custom_max):
        assert router_custom_max.max_iterations == 5
        ctx = router_custom_max.create_context(
            user_intent="test",
            active_model="sdxl-base",
        )
        assert ctx.max_iterations == 5

    def test_create_context_classifies_intent(self, router):
        ctx = router.create_context(
            user_intent="create a new image",
            active_model="sdxl-base",
            workflow_state="empty",
        )
        assert ctx.intent_type == "generation"

    def test_workflow_state_defaults_to_configured(self, router):
        ctx = router.create_context(
            user_intent="test",
            active_model="sdxl-base",
        )
        assert ctx.workflow_state == "configured"


# ===========================================================================
# TestImports
# ===========================================================================


class TestImports:
    """Verify the public API is importable from agent.agents."""

    def test_router_importable(self):
        from agent.agents import Router
        assert Router is not None

    def test_router_context_importable(self):
        from agent.agents import RouterContext
        assert RouterContext is not None

    def test_constants_importable(self):
        from agent.agents import AUTHORITY_RULES, DELEGATION_SEQUENCES
        assert isinstance(AUTHORITY_RULES, dict)
        assert isinstance(DELEGATION_SEQUENCES, dict)
