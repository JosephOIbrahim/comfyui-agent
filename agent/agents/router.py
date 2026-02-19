"""Router -- authority delegation and loop control for the MoE pipeline.

The Router is a LIGHTWEIGHT SEQUENCER. It does NOT understand models,
judge quality, or translate intent. It understands:

- **SEQUENCING**: which agent runs in which order
- **AUTHORITY**: what each agent is allowed to decide
- **LOOP CONTROL**: when to iterate, when to stop, when to escalate

The Router delegates authority, it doesn't exercise it.
Specialists have domain authority. The Router has control flow authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent.agents.intent_agent import IntentAgent, IntentSpecification
from agent.agents.verify_agent import VerifyAgent, VerificationResult
from agent.profiles import is_fallback
from agent.schemas import list_schemas

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RouterContext:
    """State the Router maintains across the agent pipeline."""

    user_intent: str
    intent_type: Literal["generation", "modification", "evaluation", "exploration"]
    active_model: str
    workflow_state: Literal["empty", "configured", "executing", "has_output"]
    iteration_count: int = 0
    max_iterations: int = 3
    history: list[dict] = field(default_factory=list)
    schemas: dict[str, str] = field(default_factory=lambda: {
        "intent": "default",
        "execution": "default",
        "verify": "default",
    })

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON / schema validation."""
        return {
            "active_model": self.active_model,
            "history": self.history,
            "intent_type": self.intent_type,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "schemas": self.schemas,
            "user_intent": self.user_intent,
            "workflow_state": self.workflow_state,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Delegation sequences by intent type
DELEGATION_SEQUENCES: dict[str, list[str]] = {
    "generation": ["intent", "execution", "verify"],
    "modification": ["intent", "execution", "verify"],
    "evaluation": ["verify"],
    "exploration": ["intent"],
}

# Authority boundaries -- HARD rules
AUTHORITY_RULES: dict[str, dict[str, list[str]]] = {
    "intent": {
        "owns": [
            "parameter_decisions",
            "prompt_modifications",
            "intent_translation",
        ],
        "cannot": [
            "execute_workflows",
            "call_comfyui_api",
            "modify_workflow_json",
        ],
    },
    "execution": {
        "owns": [
            "workflow_mutation",
            "comfyui_communication",
            "rfc6902_patching",
        ],
        "cannot": [
            "change_parameters_without_intent_spec",
            "judge_output_quality",
        ],
    },
    "verify": {
        "owns": [
            "quality_judgment",
            "iteration_decisions",
            "issue_diagnosis",
        ],
        "cannot": [
            "modify_prompts",
            "change_parameters",
            "execute_workflows",
        ],
        "can_recommend": [
            "reprompt",
            "param_adjustments",
            "inpaint",
        ],
    },
}

# Router exception handling
ROUTER_EXCEPTIONS: dict[str, dict[str, str]] = {
    "timeout": {
        "condition": "execution exceeds timeout threshold",
        "action": "abort_and_return_partial",
    },
    "loop_limit": {
        "condition": "verify keeps saying refine past max_iterations",
        "action": "escalate_with_best_attempt",
    },
    "model_not_found": {
        "condition": "intent references model that doesn't exist",
        "action": "ask_user_to_select_model",
    },
    "profile_fallback": {
        "condition": "model profile is a fallback",
        "action": "warn_and_proceed",
    },
    "low_confidence": {
        "condition": "IntentSpecification.confidence < 0.5",
        "action": "ask_user_for_clarification",
    },
}

# ---------------------------------------------------------------------------
# Intent classification keywords
# ---------------------------------------------------------------------------

_GENERATION_KEYWORDS: list[str] = [
    "create",
    "generate",
    "make",
    "new",
    "from scratch",
    "build",
    "start fresh",
]

_EVALUATION_KEYWORDS: list[str] = [
    "evaluate",
    "judge",
    "rate",
    "score",
    "how good",
    "check quality",
    "assess",
    "review quality",
]

_EXPLORATION_KEYWORDS: list[str] = [
    "what would",
    "how would",
    "translate",
    "plan",
    "explore",
    "what if",
    "hypothetical",
]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Authority delegation and loop control for the MoE pipeline.

    The Router is NOT a god-agent. It delegates authority, doesn't exercise it.
    Specialists have domain authority. The Router has control flow authority.
    """

    def __init__(
        self,
        intent_agent: IntentAgent | None = None,
        verify_agent: VerifyAgent | None = None,
        max_iterations: int = 3,
    ):
        self.intent_agent = intent_agent or IntentAgent()
        self.verify_agent = verify_agent or VerifyAgent()
        self.max_iterations = max_iterations

    def classify_intent(self, user_intent: str, workflow_state: str) -> str:
        """Classify user intent into one of the four types.

        Rules:
        - If workflow_state is "empty" and intent suggests creation -> "generation"
        - If workflow_state is "has_output" and intent is about judging -> "evaluation"
        - If intent is asking to understand/translate without executing -> "exploration"
        - Default: "modification" (most common)
        """
        intent_lower = user_intent.lower().strip() if user_intent else ""

        # Check exploration first (hypothetical language)
        for keyword in _EXPLORATION_KEYWORDS:
            if keyword in intent_lower:
                return "exploration"

        # Check evaluation (judging existing output)
        for keyword in _EVALUATION_KEYWORDS:
            if keyword in intent_lower:
                if workflow_state == "has_output":
                    return "evaluation"
                # Evaluation intent without output falls through

        # Check generation (creating from nothing)
        if workflow_state == "empty":
            for keyword in _GENERATION_KEYWORDS:
                if keyword in intent_lower:
                    return "generation"

        # Generation keywords even when not empty, if strongly signaled
        for keyword in ("from scratch", "start fresh", "new workflow"):
            if keyword in intent_lower:
                return "generation"

        # Default: modification
        return "modification"

    def get_delegation_sequence(self, intent_type: str) -> list[str]:
        """Return the delegation sequence for a given intent type."""
        return DELEGATION_SEQUENCES.get(
            intent_type, DELEGATION_SEQUENCES["modification"]
        )

    def create_context(
        self,
        user_intent: str,
        active_model: str,
        workflow_state: str = "configured",
        schemas: dict[str, str] | None = None,
    ) -> RouterContext:
        """Create a new RouterContext for a pipeline run."""
        intent_type = self.classify_intent(user_intent, workflow_state)
        return RouterContext(
            user_intent=user_intent,
            intent_type=intent_type,
            active_model=active_model,
            workflow_state=workflow_state,
            max_iterations=self.max_iterations,
            schemas=schemas or {
                "intent": "default",
                "execution": "default",
                "verify": "default",
            },
        )

    def check_preconditions(self, context: RouterContext) -> list[str]:
        """Check preconditions before pipeline execution.

        Returns list of warnings/errors. Empty = all clear.

        Checks:
        - Is model profile a fallback? -> warning
        - Is workflow_state compatible with intent_type?
          (e.g., "evaluation" requires "has_output")
        - Do requested schemas exist? (validate via list_schemas)
        """
        warnings: list[str] = []

        # Check fallback profile
        if is_fallback(context.active_model):
            warnings.append(
                f"Model '{context.active_model}' is using a fallback profile. "
                f"Results may be less precise."
            )

        # Check workflow_state compatibility
        if (
            context.intent_type == "evaluation"
            and context.workflow_state != "has_output"
        ):
            warnings.append(
                f"Evaluation requires workflow_state='has_output', "
                f"but current state is '{context.workflow_state}'."
            )

        if (
            context.intent_type == "generation"
            and context.workflow_state == "executing"
        ):
            warnings.append(
                "Cannot start generation while workflow is executing."
            )

        # Validate schemas exist
        for agent_name, schema_name in sorted(context.schemas.items()):
            try:
                available = list_schemas(agent_name)
                if available and schema_name not in available:
                    warnings.append(
                        f"Schema '{schema_name}' not found for agent "
                        f"'{agent_name}'. Available: {available}"
                    )
            except Exception:
                # If the agent directory doesn't exist, that's fine for
                # agents that don't have schemas yet (e.g., execution)
                pass

        return warnings

    def check_confidence(
        self, intent_spec: IntentSpecification, threshold: float = 0.5,
    ) -> dict:
        """Check if intent spec confidence is above threshold.

        Returns:
            {"proceed": True/False, "reason": str, "confidence": float}
        """
        proceed = intent_spec.confidence >= threshold
        if proceed:
            reason = (
                f"Confidence {intent_spec.confidence:.2f} meets "
                f"threshold {threshold:.2f}."
            )
        else:
            reason = (
                f"Confidence {intent_spec.confidence:.2f} is below "
                f"threshold {threshold:.2f}. Consider asking the user "
                f"for clarification."
            )
        return {
            "confidence": intent_spec.confidence,
            "proceed": proceed,
            "reason": reason,
        }

    def should_continue(
        self,
        verification: VerificationResult,
        context: RouterContext,
    ) -> dict:
        """Determine if the loop should continue based on verification result.

        Returns:
            {
                "continue": True/False,
                "reason": str,
                "action": "proceed"|"refine"|"reprompt"|"escalate"|"accept",
                "refinement_actions": list[dict] | None,
            }
        """
        decision = verification.decision

        if decision == "accept":
            return {
                "action": "accept",
                "continue": False,
                "reason": "Output accepted by Verify Agent.",
                "refinement_actions": None,
            }

        if decision == "escalate":
            return {
                "action": "escalate",
                "continue": False,
                "reason": "Verify Agent escalated — needs human intervention.",
                "refinement_actions": None,
            }

        # refine or reprompt — check iteration limit
        if context.iteration_count >= context.max_iterations:
            return {
                "action": "escalate",
                "continue": False,
                "reason": (
                    f"Reached max iterations ({context.max_iterations}). "
                    f"Escalating with best attempt."
                ),
                "refinement_actions": None,
            }

        # Can continue
        actions = [
            {
                "priority": a.priority,
                "reason": a.reason,
                "target": a.target,
                "type": a.type,
            }
            for a in verification.refinement_actions
        ] if verification.refinement_actions else None

        return {
            "action": decision,  # "refine" or "reprompt"
            "continue": True,
            "reason": (
                f"Verify Agent recommends '{decision}' "
                f"(iteration {context.iteration_count + 1}"
                f"/{context.max_iterations})."
            ),
            "refinement_actions": actions,
        }

    def record_iteration(self, context: RouterContext, result: dict) -> None:
        """Record an iteration result in context history.

        Appends to context.history and increments context.iteration_count.
        """
        context.history.append(result)
        context.iteration_count += 1

    def validate_authority(self, agent: str, action: str) -> bool:
        """Check if an agent is authorized to perform an action.

        Uses AUTHORITY_RULES. Returns True if allowed, False if violated.
        """
        rules = AUTHORITY_RULES.get(agent)
        if rules is None:
            return False

        # Check if action is in the "owns" list
        if action in rules.get("owns", []):
            return True

        # Check if action is in "can_recommend" (still allowed)
        if action in rules.get("can_recommend", []):
            return True

        # If action is in "cannot", explicitly denied
        if action in rules.get("cannot", []):
            return False

        # Unknown action for this agent — deny by default
        return False

    def get_exception_action(self, exception_type: str) -> str:
        """Get the recommended action for a router exception."""
        return ROUTER_EXCEPTIONS.get(exception_type, {}).get(
            "action", "escalate_with_best_attempt"
        )

    def run_pipeline(
        self,
        user_intent: str,
        model_id: str,
        workflow_state: dict | None = None,
        schemas: dict[str, str] | None = None,
        output_analysis: dict | None = None,
    ) -> dict:
        """Execute the full MoE pipeline (synchronous, non-executing).

        This runs Intent and Verify agents but does NOT execute workflows
        (that requires ComfyUI). It returns the plan:

        Returns:
            {
                "context": RouterContext.to_dict(),
                "intent_spec": IntentSpecification.to_dict() | None,
                "verification": VerificationResult.to_dict() | None,
                "delegation_sequence": list[str],
                "precondition_warnings": list[str],
                "status": "planned"|"evaluated"|"needs_clarification"|"error",
            }
        """
        # Determine workflow state string
        ws_str: str = "configured"
        if isinstance(workflow_state, dict):
            ws_str = str(workflow_state.get("state", "configured"))
        elif workflow_state is None:
            # Infer from output_analysis presence
            ws_str = "has_output" if output_analysis else "configured"

        context = self.create_context(
            user_intent=user_intent,
            active_model=model_id,
            workflow_state=ws_str,
            schemas=schemas,
        )

        precondition_warnings = self.check_preconditions(context)
        delegation_sequence = self.get_delegation_sequence(context.intent_type)

        result: dict[str, Any] = {
            "context": context.to_dict(),
            "delegation_sequence": delegation_sequence,
            "intent_spec": None,
            "precondition_warnings": precondition_warnings,
            "status": "planned",
            "verification": None,
        }

        try:
            if context.intent_type == "evaluation":
                # Run Verify Agent on output_analysis
                if output_analysis is None:
                    result["status"] = "error"
                    result["precondition_warnings"].append(
                        "Evaluation requires output_analysis but none provided."
                    )
                    return result

                verification = self.verify_agent.evaluate(
                    output_analysis=output_analysis,
                    original_intent=user_intent,
                    model_id=model_id,
                    iteration_count=context.iteration_count,
                    max_iterations=context.max_iterations,
                )
                result["verification"] = verification.to_dict()
                result["status"] = "evaluated"

            elif context.intent_type in ("generation", "modification", "exploration"):
                # Run Intent Agent
                wf_state_params = None
                if isinstance(workflow_state, dict):
                    wf_state_params = {
                        k: v for k, v in workflow_state.items()
                        if k != "state"
                    } or None

                intent_spec = self.intent_agent.translate(
                    user_intent=user_intent,
                    model_id=model_id,
                    workflow_state=wf_state_params,
                )
                result["intent_spec"] = intent_spec.to_dict()

                # Check confidence
                confidence_check = self.check_confidence(intent_spec)
                if not confidence_check["proceed"]:
                    result["status"] = "needs_clarification"
                else:
                    result["status"] = "planned"

        except Exception as exc:
            result["status"] = "error"
            result["precondition_warnings"].append(
                f"Pipeline error: {exc}"
            )

        return result
