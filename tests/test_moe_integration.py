"""Tests for MoE pipeline integration — iterative_refine brain tool.

Tests the capstone brain tool that orchestrates:
  - Intent classification via Router
  - Intent translation via Intent Agent
  - Quality verification via Verify Agent
  - Iterative refinement loops (verify -> refine -> re-translate)
  - Schema propagation
  - Edge cases and error handling
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agent.agents.intent_agent import (
    IntentAgent,
    IntentSpecification,
    ParameterMutation,
    PromptMutation,
)
from agent.agents.router import Router
from agent.agents.verify_agent import (
    RefinementAction,
    VerificationResult,
    VerifyAgent,
)
from agent.brain.iterative_refine import (
    TOOLS,
    IterativeRefineAgent,
    handle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(result: str) -> dict:
    """Parse a tool result JSON string."""
    return json.loads(result)


def _make_intent_spec(
    model_id: str = "sdxl-base",
    confidence: float = 0.85,
    *,
    mutations: list[ParameterMutation] | None = None,
    prompt_mutations: list[PromptMutation] | None = None,
    warnings: list[str] | None = None,
    fallback: bool = False,
) -> IntentSpecification:
    """Build a test IntentSpecification."""
    return IntentSpecification(
        model_id=model_id,
        parameter_mutations=mutations or [],
        prompt_mutations=prompt_mutations or [],
        confidence=confidence,
        warnings=warnings or [],
        using_fallback_profile=fallback,
    )


def _make_verification(
    decision: str = "accept",
    overall_score: float = 0.85,
    intent_alignment: float = 0.9,
    technical_quality: float = 0.8,
    *,
    refinement_actions: list[RefinementAction] | None = None,
    iteration_count: int = 0,
    max_iterations: int = 3,
    diagnosed_issues: list[str] | None = None,
) -> VerificationResult:
    """Build a test VerificationResult."""
    return VerificationResult(
        overall_score=overall_score,
        intent_alignment=intent_alignment,
        technical_quality=technical_quality,
        decision=decision,
        refinement_actions=refinement_actions or [],
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        diagnosed_issues=diagnosed_issues or [],
    )


# ---------------------------------------------------------------------------
# TestIterativeRefineBasic
# ---------------------------------------------------------------------------


class TestIterativeRefineBasic:
    """Basic pipeline dispatch without mocking agents."""

    def test_modification_intent_returns_planned(self):
        """Simple modification intent without output_analysis -> planned."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "make it dreamier",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "planned"
        assert result["intent_type"] == "modification"
        assert result["intent_spec"] is not None

    def test_generation_intent_correct_delegation(self):
        """Generation intent gets generation delegation sequence."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "create a new image from scratch",
            "model_id": "flux1-dev",
            "workflow_state": {"state": "empty"},
        }))
        assert result["delegation_sequence"] == ["intent", "execution", "verify"]

    def test_exploration_intent_no_verification(self):
        """Exploration intent: translation only, no verification."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "what would happen if I made it sharper",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "planned"
        assert result["intent_type"] == "exploration"
        assert result["intent_spec"] is not None
        assert result["verification"] is None

    def test_evaluation_with_output_analysis(self):
        """Evaluation intent with output_analysis -> verification result."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "evaluate quality of this output",
            "model_id": "sdxl-base",
            "workflow_state": {"state": "has_output"},
            "output_analysis": {
                "quality_score": 0.8,
                "matches_intent": True,
                "artifacts": [],
                "issues": [],
            },
        }))
        assert result["status"] == "evaluated"
        assert result["intent_type"] == "evaluation"
        assert result["verification"] is not None
        assert result["intent_spec"] is None

    def test_evaluation_without_output_analysis_errors(self):
        """Evaluation without output_analysis -> error."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "evaluate quality of this output",
            "model_id": "sdxl-base",
            "workflow_state": {"state": "has_output"},
        }))
        assert result["status"] == "error"
        assert any("output_analysis" in w for w in result["precondition_warnings"])

    def test_empty_user_intent_errors(self):
        """Empty user_intent -> error."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "error"

    def test_empty_model_id_errors(self):
        """Empty model_id -> error."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "",
        }))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TestIterativeRefinePipeline (with mocked agents)
# ---------------------------------------------------------------------------


class TestIterativeRefinePipeline:
    """Tests with mocked Intent/Verify agents for controlled pipeline flow."""

    def test_modification_with_good_quality_accepted(self):
        """Modification with output that passes verification -> accepted."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.return_value = _make_verification(
            decision="accept", overall_score=0.9
        )

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "make it dreamier",
                "model_id": "sdxl-base",
                "output_analysis": {
                    "quality_score": 0.9,
                    "matches_intent": True,
                    "artifacts": [],
                    "issues": [],
                },
            }))

        assert result["status"] == "accepted"
        assert result["iterations"] == 1
        assert len(result["history"]) == 1

    def test_modification_refine_loop(self):
        """Modification with bad quality -> refine loop -> accept."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        # First verify says refine, second says accept
        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.side_effect = [
            _make_verification(
                decision="refine",
                overall_score=0.5,
                intent_alignment=0.6,
                refinement_actions=[
                    RefinementAction(
                        type="adjust_params",
                        target="cfg",
                        reason="Lower CFG",
                        priority=1,
                    )
                ],
            ),
            _make_verification(
                decision="accept",
                overall_score=0.85,
                intent_alignment=0.9,
            ),
        ]

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "make it dreamier",
                "model_id": "sdxl-base",
                "output_analysis": {
                    "quality_score": 0.5,
                    "matches_intent": 0.6,
                    "artifacts": ["oversaturated"],
                    "issues": [],
                },
            }))

        assert result["status"] == "accepted"
        assert result["iterations"] == 2
        assert len(result["history"]) == 2
        # Intent agent should have been called twice (initial + refinement)
        assert mock_intent.translate.call_count == 2

    def test_modification_reprompt_loop(self):
        """Modification with very bad quality -> reprompt loop."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        # First verify says reprompt, second says accept
        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.side_effect = [
            _make_verification(
                decision="reprompt",
                overall_score=0.3,
                intent_alignment=0.2,
                refinement_actions=[
                    RefinementAction(
                        type="reprompt",
                        target="prompt",
                        reason="Significant prompt revision needed",
                        priority=1,
                    )
                ],
            ),
            _make_verification(
                decision="accept",
                overall_score=0.8,
                intent_alignment=0.85,
            ),
        ]

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "make it dreamier",
                "model_id": "sdxl-base",
                "output_analysis": {
                    "quality_score": 0.3,
                    "matches_intent": 0.2,
                    "artifacts": [],
                    "issues": ["wrong subject"],
                },
            }))

        assert result["status"] == "accepted"
        assert result["iterations"] == 2

    def test_max_iterations_escalates(self):
        """Refinement loop hits max_iterations -> escalated."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        # All verifications say refine
        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.return_value = _make_verification(
            decision="refine",
            overall_score=0.5,
            intent_alignment=0.6,
            refinement_actions=[
                RefinementAction(
                    type="adjust_params",
                    target="steps",
                    reason="More steps",
                    priority=2,
                )
            ],
        )

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
                max_iterations=2,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "sharper",
                "model_id": "sdxl-base",
                "max_iterations": 2,
                "output_analysis": {
                    "quality_score": 0.5,
                    "matches_intent": 0.6,
                    "artifacts": ["blurry"],
                    "issues": [],
                },
            }))

        assert result["status"] == "escalated"
        assert result["iterations"] >= 1

    def test_low_confidence_needs_clarification(self):
        """Low confidence spec -> needs_clarification."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.3)

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(intent_agent=mock_intent),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "xyzzy",
                "model_id": "sdxl-base",
            }))

        assert result["status"] == "needs_clarification"
        assert result["intent_spec"] is not None
        assert result["intent_spec"]["confidence"] == 0.3

    def test_verify_escalate_stops_loop(self):
        """Verify agent says escalate -> loop stops immediately."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.return_value = _make_verification(
            decision="escalate",
            overall_score=0.2,
            intent_alignment=0.1,
        )

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "make it better",
                "model_id": "sdxl-base",
                "output_analysis": {
                    "quality_score": 0.2,
                    "matches_intent": 0.1,
                },
            }))

        assert result["status"] == "escalated"
        assert result["iterations"] == 1
        # Only one verify call
        assert mock_verify.evaluate.call_count == 1


# ---------------------------------------------------------------------------
# TestRefinementLoop
# ---------------------------------------------------------------------------


class TestRefinementLoop:
    """Tests that refinement context flows correctly through iterations."""

    def test_refinement_actions_flow_to_intent_agent(self):
        """Verify's refinement actions are passed as refinement_context."""
        translate_calls = []

        class CapturingIntentAgent(IntentAgent):
            def translate(self, **kwargs):
                translate_calls.append(kwargs)
                return _make_intent_spec(confidence=0.9)

        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.side_effect = [
            _make_verification(
                decision="refine",
                overall_score=0.5,
                intent_alignment=0.6,
                refinement_actions=[
                    RefinementAction(
                        type="adjust_params",
                        target="cfg",
                        reason="Lower CFG to reduce artifacts",
                        priority=1,
                    ),
                ],
            ),
            _make_verification(decision="accept", overall_score=0.85),
        ]

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=CapturingIntentAgent(),
                verify_agent=mock_verify,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "dreamier",
                "model_id": "sdxl-base",
                "output_analysis": {
                    "quality_score": 0.5,
                    "matches_intent": 0.6,
                },
            }))

        assert result["status"] == "accepted"
        # First call: no refinement_context
        assert translate_calls[0].get("refinement_context") is None
        # Second call: has refinement_context from verify
        second_ctx = translate_calls[1].get("refinement_context")
        assert second_ctx is not None
        assert len(second_ctx) >= 1

    def test_history_accumulates_across_iterations(self):
        """History list grows with each iteration."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        call_count = [0]

        def side_effect_verify(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return _make_verification(
                    decision="refine",
                    overall_score=0.5,
                    refinement_actions=[
                        RefinementAction(
                            type="adjust_params", target="steps",
                            reason="More steps", priority=2,
                        )
                    ],
                )
            return _make_verification(decision="accept", overall_score=0.85)

        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.side_effect = side_effect_verify

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
                max_iterations=5,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "sharper",
                "model_id": "sdxl-base",
                "max_iterations": 5,
                "output_analysis": {
                    "quality_score": 0.5,
                    "matches_intent": 0.6,
                },
            }))

        assert result["status"] == "accepted"
        assert result["iterations"] == 3
        assert len(result["history"]) == 3
        # Each history entry has iteration number
        for i, entry in enumerate(result["history"], 1):
            assert entry["iteration"] == i

    def test_loop_exits_on_accept(self):
        """Loop exits immediately when verify says accept."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.return_value = _make_verification(
            decision="accept", overall_score=0.9
        )

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
                max_iterations=5,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "dreamier",
                "model_id": "sdxl-base",
                "max_iterations": 5,
                "output_analysis": {
                    "quality_score": 0.9,
                    "matches_intent": True,
                },
            }))

        assert result["status"] == "accepted"
        assert result["iterations"] == 1
        # Only one translate call (no refinement needed)
        assert mock_intent.translate.call_count == 1

    def test_loop_exits_on_max_iterations_with_escalation(self):
        """Loop terminates and escalates when max_iterations exhausted."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.return_value = _make_verification(
            decision="refine",
            overall_score=0.5,
            refinement_actions=[
                RefinementAction(
                    type="adjust_params", target="cfg",
                    reason="Lower CFG", priority=1,
                )
            ],
        )

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
                max_iterations=2,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "better quality",
                "model_id": "sdxl-base",
                "max_iterations": 2,
                "output_analysis": {
                    "quality_score": 0.5,
                    "matches_intent": 0.6,
                },
            }))

        assert result["status"] == "escalated"
        assert result["verification"] is not None


# ---------------------------------------------------------------------------
# TestClassifyIntentTool
# ---------------------------------------------------------------------------


class TestClassifyIntentTool:
    """Tests for the classify_intent standalone tool."""

    def test_basic_classification(self):
        result = _parse(handle("classify_intent", {
            "user_intent": "make it dreamier",
        }))
        assert result["intent_type"] == "modification"
        assert "delegation_sequence" in result

    def test_exploration_classification(self):
        result = _parse(handle("classify_intent", {
            "user_intent": "what would happen if I increased CFG",
        }))
        assert result["intent_type"] == "exploration"

    def test_evaluation_classification(self):
        result = _parse(handle("classify_intent", {
            "user_intent": "evaluate the quality of this",
            "workflow_state": "has_output",
        }))
        assert result["intent_type"] == "evaluation"

    def test_generation_classification(self):
        result = _parse(handle("classify_intent", {
            "user_intent": "create a new portrait from scratch",
            "workflow_state": "empty",
        }))
        assert result["intent_type"] == "generation"

    def test_default_workflow_state(self):
        """Default workflow_state is 'configured'."""
        result = _parse(handle("classify_intent", {
            "user_intent": "sharper",
        }))
        assert result["workflow_state"] == "configured"

    def test_returns_delegation_sequence(self):
        result = _parse(handle("classify_intent", {
            "user_intent": "create new image",
            "workflow_state": "empty",
        }))
        assert result["delegation_sequence"] == ["intent", "execution", "verify"]


# ---------------------------------------------------------------------------
# TestSchemaIntegration
# ---------------------------------------------------------------------------


class TestSchemaIntegration:
    """Tests for schema propagation through the pipeline."""

    def test_custom_schemas_in_response(self):
        """Custom schemas appear in schemas_used."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "make it dreamier",
            "model_id": "sdxl-base",
            "schemas": {
                "intent": "lora_workflow",
                "verify": "cinematographer",
            },
        }))
        assert result["schemas_used"]["intent"] == "lora_workflow"
        assert result["schemas_used"]["verify"] == "cinematographer"

    def test_default_schemas_when_none_specified(self):
        """Default schemas used when none specified."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "make it dreamier",
            "model_id": "sdxl-base",
        }))
        assert result["schemas_used"]["intent"] == "default"
        assert result["schemas_used"]["verify"] == "default"

    def test_schemas_used_present_on_error(self):
        """schemas_used is present even on error results."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "evaluate this",
            "model_id": "sdxl-base",
            "workflow_state": {"state": "has_output"},
            "schemas": {"intent": "custom"},
        }))
        # This is evaluation without output_analysis -> error
        assert "schemas_used" in result


# ---------------------------------------------------------------------------
# TestToolInterface
# ---------------------------------------------------------------------------


class TestToolInterface:
    """Tests for the tool dispatch and schema correctness."""

    def test_handle_dispatches_iterative_refine(self):
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        assert "status" in result

    def test_handle_dispatches_classify_intent(self):
        result = _parse(handle("classify_intent", {
            "user_intent": "dreamier",
        }))
        assert "intent_type" in result

    def test_unknown_tool_name_error(self):
        result = _parse(handle("nonexistent_tool", {}))
        assert "error" in result

    def test_tool_schemas_valid(self):
        """All tool schemas have required fields."""
        assert len(TOOLS) == 2
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_iterative_refine_schema_has_required_fields(self):
        schema = TOOLS[0]
        assert schema["name"] == "iterative_refine"
        assert "user_intent" in schema["input_schema"]["required"]
        assert "model_id" in schema["input_schema"]["required"]

    def test_classify_intent_schema_has_required_fields(self):
        schema = TOOLS[1]
        assert schema["name"] == "classify_intent"
        assert "user_intent" in schema["input_schema"]["required"]

    def test_missing_required_user_intent(self):
        """Missing user_intent -> graceful error, not crash."""
        result = _parse(handle("iterative_refine", {
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "error"

    def test_missing_required_model_id(self):
        """Missing model_id -> graceful error."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
        }))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TestWithRealProfiles
# ---------------------------------------------------------------------------


class TestWithRealProfiles:
    """Tests using real model profiles (no mocking of agents)."""

    def test_dreamier_with_sdxl(self):
        """'make it dreamier' with sdxl-base -> SDXL-appropriate mutations."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "make it dreamier",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "planned"
        spec = result["intent_spec"]
        assert spec is not None
        assert spec["model_id"] == "sdxl-base"
        # Should have parameter mutations for dreamier
        if spec["parameter_mutations"]:
            targets = [m["target"] for m in spec["parameter_mutations"]]
            # Dreamier typically adjusts CFG or sampler
            assert any("KSampler" in t for t in targets)

    def test_sharper_with_sdxl(self):
        """'make it sharper' with sdxl-base -> SDXL-appropriate mutations."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "sharper",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "planned"
        spec = result["intent_spec"]
        assert spec is not None

    def test_dreamier_with_flux(self):
        """'make it dreamier' with flux1-dev -> Flux-appropriate mutations."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "make it dreamier",
            "model_id": "flux1-dev",
        }))
        assert result["status"] == "planned"
        spec = result["intent_spec"]
        assert spec is not None
        assert spec["model_id"] == "flux1-dev"

    def test_unknown_model_fallback_warning(self):
        """Unknown model -> fallback warning in preconditions."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "totally-unknown-model-xyz",
        }))
        # Low confidence from fallback may trigger needs_clarification
        assert result["status"] in ("planned", "needs_clarification")
        # Should have a fallback warning
        assert any(
            "fallback" in w.lower()
            for w in result["precondition_warnings"]
        )

    def test_compound_intent_dreamier_and_sharper(self):
        """Compound intent 'dreamier and sharper' works."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier and sharper",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "planned"
        spec = result["intent_spec"]
        assert spec is not None
        # May have conflict resolutions
        # At minimum should have some mutations


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling."""

    def test_none_workflow_state(self):
        """None workflow_state handled gracefully."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
            "workflow_state": None,
        }))
        assert result["status"] == "planned"

    def test_empty_output_analysis(self):
        """Empty dict output_analysis handled gracefully."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
            "output_analysis": {},
        }))
        # Should still work (empty analysis)
        assert result["status"] in ("planned", "accepted", "refined", "escalated")

    def test_large_max_iterations_capped(self):
        """Very large max_iterations is capped to 10."""
        mock_intent = MagicMock(spec=IntentAgent)
        mock_intent.translate.return_value = _make_intent_spec(confidence=0.9)

        mock_verify = MagicMock(spec=VerifyAgent)
        mock_verify.evaluate.return_value = _make_verification(
            decision="accept", overall_score=0.9,
        )

        with patch(
            "agent.brain.iterative_refine.Router",
            return_value=Router(
                intent_agent=mock_intent,
                verify_agent=mock_verify,
            ),
        ):
            result = _parse(handle("iterative_refine", {
                "user_intent": "dreamier",
                "model_id": "sdxl-base",
                "max_iterations": 999,
                "output_analysis": {
                    "quality_score": 0.9,
                    "matches_intent": True,
                },
            }))
        # Should still work, not loop 999 times
        assert result["status"] == "accepted"
        assert result["iterations"] <= 10

    def test_negative_max_iterations_floored(self):
        """Negative max_iterations is floored to 1."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
            "max_iterations": -5,
        }))
        assert result["status"] == "planned"

    def test_workflow_state_with_params(self):
        """Workflow state with parameters is passed through."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
            "workflow_state": {
                "state": "configured",
                "cfg": 7.0,
                "steps": 30,
            },
        }))
        assert result["status"] == "planned"

    def test_result_is_valid_json(self):
        """Result is always valid JSON."""
        raw = handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        })
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_classify_intent_empty_intent(self):
        """Empty intent string for classify_intent."""
        result = _parse(handle("classify_intent", {
            "user_intent": "",
        }))
        # Should default to modification
        assert result["intent_type"] == "modification"

    def test_response_fields_always_present(self):
        """All expected fields are present in every response."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        expected_fields = [
            "status", "intent_type", "delegation_sequence",
            "precondition_warnings", "intent_spec", "verification",
            "iterations", "history", "schemas_used",
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_error_response_fields_always_present(self):
        """All expected fields present even on error."""
        result = _parse(handle("iterative_refine", {
            "user_intent": "",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "error"
        expected_fields = [
            "status", "intent_type", "delegation_sequence",
            "precondition_warnings", "intent_spec", "verification",
            "iterations", "history", "schemas_used",
        ]
        for field in expected_fields:
            assert field in result, f"Missing field on error: {field}"


# ---------------------------------------------------------------------------
# TestSDKClass
# ---------------------------------------------------------------------------


class TestSDKClass:
    """Tests for the IterativeRefineAgent SDK class."""

    def test_instantiation(self):
        agent = IterativeRefineAgent()
        assert agent is not None

    def test_tools_attribute(self):
        assert IterativeRefineAgent.TOOLS == TOOLS
        assert len(IterativeRefineAgent.TOOLS) == 2

    def test_handle_dispatches(self):
        agent = IterativeRefineAgent()
        result = _parse(agent.handle("classify_intent", {
            "user_intent": "dreamier",
        }))
        assert "intent_type" in result

    def test_handle_iterative_refine(self):
        agent = IterativeRefineAgent()
        result = _parse(agent.handle("iterative_refine", {
            "user_intent": "dreamier",
            "model_id": "sdxl-base",
        }))
        assert result["status"] == "planned"

    def test_handle_unknown_tool(self):
        agent = IterativeRefineAgent()
        result = _parse(agent.handle("bogus", {}))
        assert "error" in result

    def test_custom_config(self):
        from agent.brain._sdk import BrainConfig
        cfg = BrainConfig()
        agent = IterativeRefineAgent(cfg=cfg)
        assert agent.cfg is cfg


# ---------------------------------------------------------------------------
# TestBrainRegistration
# ---------------------------------------------------------------------------


class TestBrainRegistration:
    """Verify iterative_refine is correctly registered in brain/__init__."""

    def test_tools_in_all_brain_tools(self):
        from agent.brain import ALL_BRAIN_TOOLS
        tool_names = [t["name"] for t in ALL_BRAIN_TOOLS]
        assert "iterative_refine" in tool_names
        assert "classify_intent" in tool_names

    def test_handle_dispatch_from_brain(self):
        from agent.brain import handle as brain_handle
        result = _parse(brain_handle("classify_intent", {
            "user_intent": "dreamier",
        }))
        assert "intent_type" in result

    def test_iterative_refine_agent_exported(self):
        from agent.brain import IterativeRefineAgent as Exported
        assert Exported is IterativeRefineAgent
