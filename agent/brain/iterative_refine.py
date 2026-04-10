"""Iterative refinement — MoE pipeline orchestrator brain tool.

Wires the Router, Intent Agent, and Verify Agent into a single brain tool
that translates artistic intent into parameter changes, optionally verifies
output quality, and supports iterative refinement loops where the Verify
Agent's feedback feeds back into the Intent Agent.

Integration points:
  - Router (agent.agents.router) — sequencing, authority, loop control
  - IntentAgent (agent.agents.intent_agent) — intent translation
  - VerifyAgent (agent.agents.verify_agent) — quality judgment
  - Profiles (agent.profiles) — model-specific parameters
  - Schemas (agent.schemas) — output validation
  - Memory (agent.brain.memory) — outcome recording from verification results
"""

import json as _json
import logging

from ._sdk import BrainAgent
from ..agents.router import Router
from .. import tools as _tools_mod
from .._conn_ctx import current_conn_session

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "name": "iterative_refine",
        "description": (
            "Execute the MoE pipeline: translate artistic intent into parameter "
            "changes via the Intent Agent, then optionally verify output quality "
            "via the Verify Agent. Supports iterative refinement loops where the "
            "Verify Agent's feedback feeds back into the Intent Agent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_intent": {
                    "type": "string",
                    "description": "Natural language artistic intent from the user",
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "Active model identifier (e.g., 'flux1-dev', 'sdxl-base')"
                    ),
                },
                "workflow_state": {
                    "type": "object",
                    "description": "Current workflow JSON state (optional)",
                },
                "output_analysis": {
                    "type": "object",
                    "description": (
                        "Analysis of the current output (from vision module or manual). "
                        "Required for evaluation intents. Should contain keys like "
                        "'quality_score', 'artifacts', 'issues', 'matches_intent'."
                    ),
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum refinement iterations (default: 3)",
                    "default": 3,
                },
                "schemas": {
                    "type": "object",
                    "description": (
                        "Per-agent schema overrides, e.g. "
                        "{'intent': 'lora_workflow', 'verify': 'cinematographer'}"
                    ),
                },
            },
            "required": ["user_intent", "model_id"],
        },
    },
    {
        "name": "classify_intent",
        "description": (
            "Classify a user's artistic intent into one of four types: "
            "generation, modification, evaluation, or exploration. "
            "Useful for understanding what pipeline path will be taken."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_intent": {
                    "type": "string",
                    "description": "The user's natural language intent",
                },
                "workflow_state": {
                    "type": "string",
                    "description": (
                        "Current state: empty, configured, executing, has_output"
                    ),
                    "default": "configured",
                },
            },
            "required": ["user_intent"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_to_json(obj: dict) -> str:
    """Deterministic JSON serialization, importing to_json lazily."""
    try:
        from ..tools._util import to_json
        return to_json(obj)
    except Exception:
        import json
        return json.dumps(obj, sort_keys=True, allow_nan=False)  # Cycle 57: NaN-safe fallback


def _record_to_memory(
    *,
    status: str,
    user_intent: str,
    model_id: str,
    verification: dict | None,
    intent_spec: dict | None,
    iterations: int,
) -> None:
    """Record MoE pipeline outcome to memory (fire-and-forget).

    Bridges verification results into memory's record_outcome so the
    learning loop can track quality scores, diagnosed issues, and
    refinement patterns across sessions.
    """
    if verification is None:
        return

    try:
        quality_score = verification.get("overall_score")
        diagnosed = verification.get("diagnosed_issues", [])
        decision = verification.get("decision", "")

        # Build vision_notes from verification diagnostics
        vision_notes = []
        if diagnosed:
            vision_notes.extend(diagnosed)
        if model_lims := verification.get("model_limitations"):  # Cycle 56: walrus avoids double lookup
            vision_notes.extend(f"model_limitation: {lim}" for lim in model_lims)
        if decision:
            vision_notes.append(f"verify_decision: {decision}")

        # Extract key_params from intent_spec mutations
        key_params: dict = {"model": model_id}
        if intent_spec and (param_muts := intent_spec.get("parameter_mutations")):  # Cycle 56: walrus avoids double lookup
            for mut in param_muts:
                target = mut.get("target", "")
                value = mut.get("value")
                if target and value is not None:
                    # Use the leaf parameter name (e.g. "cfg" from "KSampler.cfg")
                    param = target.rsplit(".", 1)[-1] if "." in target else target
                    key_params[param] = value

        outcome_input = {
            "session": current_conn_session(),
            "workflow_summary": (
                f"MoE {status}: \"{user_intent}\" on {model_id} "
                f"({iterations} iteration{'s' if iterations != 1 else ''})"
            ),
            "key_params": key_params,
            "model_combo": [model_id],
            "quality_score": quality_score,
            "vision_notes": vision_notes,
        }

        BrainAgent.dispatch("record_outcome", outcome_input)
        log.debug(
            "Recorded MoE outcome to memory: status=%s model=%s score=%s",
            status, model_id, quality_score,
        )
    except Exception as exc:
        log.warning("Failed to record MoE outcome to memory (non-fatal): %s", exc)


def _handle_classify_intent(tool_input: dict) -> str:
    """Classify user intent via the Router."""
    user_intent = tool_input.get("user_intent", "")
    workflow_state = tool_input.get("workflow_state", "configured")

    router = Router()
    intent_type = router.classify_intent(user_intent, workflow_state)
    delegation_sequence = router.get_delegation_sequence(intent_type)

    return _safe_to_json({
        "delegation_sequence": delegation_sequence,
        "intent_type": intent_type,
        "workflow_state": workflow_state,
    })


def _handle_iterative_refine(tool_input: dict) -> str:
    """Execute the full MoE pipeline with optional iterative refinement."""
    # Extract inputs
    user_intent = tool_input.get("user_intent", "")
    model_id = tool_input.get("model_id", "")
    workflow_state = tool_input.get("workflow_state")
    output_analysis = tool_input.get("output_analysis")
    max_iterations = tool_input.get("max_iterations", 3)
    schemas = tool_input.get("schemas")

    # Clamp max_iterations to a sane range
    max_iterations = max(1, min(max_iterations, 10))

    # Validate required fields
    if not user_intent:
        return _safe_to_json({
            "status": "error",
            "intent_type": "unknown",
            "delegation_sequence": [],
            "precondition_warnings": ["user_intent is required"],
            "intent_spec": None,
            "verification": None,
            "validation": [],
            "iterations": 0,
            "history": [],
            "schemas_used": {},
        })

    if not model_id:
        return _safe_to_json({
            "status": "error",
            "intent_type": "unknown",
            "delegation_sequence": [],
            "precondition_warnings": ["model_id is required"],
            "intent_spec": None,
            "verification": None,
            "validation": [],
            "iterations": 0,
            "history": [],
            "schemas_used": {},
        })

    # Create Router and context
    router = Router(max_iterations=max_iterations)

    # Determine workflow_state string for classification
    ws_str = "configured"
    if isinstance(workflow_state, dict):
        ws_str = str(workflow_state.get("state", "configured"))
    elif workflow_state is None:
        ws_str = "has_output" if output_analysis else "configured"

    context = router.create_context(
        user_intent=user_intent,
        active_model=model_id,
        workflow_state=ws_str,
        schemas=schemas,
    )

    precondition_warnings = router.check_preconditions(context)
    delegation_sequence = router.get_delegation_sequence(context.intent_type)

    # Determine schemas used
    schemas_used = dict(context.schemas)

    try:
        return _dispatch_by_intent_type(
            router=router,
            context=context,
            user_intent=user_intent,
            model_id=model_id,
            workflow_state=workflow_state,
            output_analysis=output_analysis,
            max_iterations=max_iterations,
            delegation_sequence=delegation_sequence,
            precondition_warnings=precondition_warnings,
            schemas_used=schemas_used,
        )
    except Exception as exc:
        log.error("MoE pipeline error: %s", exc, exc_info=True)
        return _safe_to_json({
            "status": "error",
            "intent_type": context.intent_type,
            "delegation_sequence": delegation_sequence,
            "precondition_warnings": precondition_warnings + [str(exc)],
            "intent_spec": None,
            "verification": None,
            "validation": [],
            "iterations": 0,
            "history": [],
            "schemas_used": schemas_used,
        })


def _dispatch_by_intent_type(
    *,
    router,
    context,
    user_intent: str,
    model_id: str,
    workflow_state,
    output_analysis,
    max_iterations: int,
    delegation_sequence: list[str],
    precondition_warnings: list[str],
    schemas_used: dict[str, str],
) -> str:
    """Route to the correct pipeline path based on intent type."""
    intent_type = context.intent_type

    if intent_type == "evaluation":
        return _handle_evaluation(
            router=router,
            context=context,
            user_intent=user_intent,
            model_id=model_id,
            workflow_state=workflow_state,
            output_analysis=output_analysis,
            delegation_sequence=delegation_sequence,
            precondition_warnings=precondition_warnings,
            schemas_used=schemas_used,
        )

    if intent_type == "exploration":
        return _handle_exploration(
            router=router,
            context=context,
            user_intent=user_intent,
            model_id=model_id,
            workflow_state=workflow_state,
            delegation_sequence=delegation_sequence,
            precondition_warnings=precondition_warnings,
            schemas_used=schemas_used,
        )

    # generation or modification
    return _handle_generation_or_modification(
        router=router,
        context=context,
        user_intent=user_intent,
        model_id=model_id,
        workflow_state=workflow_state,
        output_analysis=output_analysis,
        max_iterations=max_iterations,
        delegation_sequence=delegation_sequence,
        precondition_warnings=precondition_warnings,
        schemas_used=schemas_used,
    )


def _handle_evaluation(
    *,
    router,
    context,
    user_intent: str,
    model_id: str,
    workflow_state=None,
    output_analysis,
    delegation_sequence: list[str],
    precondition_warnings: list[str],
    schemas_used: dict[str, str],
) -> str:
    """Handle evaluation intent — run Verify Agent only."""
    if output_analysis is None:
        return _safe_to_json({
            "status": "error",
            "intent_type": "evaluation",
            "delegation_sequence": delegation_sequence,
            "precondition_warnings": precondition_warnings + [
                "Evaluation requires output_analysis but none provided."
            ],
            "intent_spec": None,
            "verification": None,
            "validation": [],
            "iterations": 0,
            "history": [],
            "schemas_used": schemas_used,
        })

    # Extract parameters from workflow for Verify Agent enrichment
    parameters_used = _extract_parameters_from_workflow(workflow_state)

    verification = router.verify_agent.evaluate(
        output_analysis=output_analysis,
        original_intent=user_intent,
        model_id=model_id,
        parameters_used=parameters_used,
        iteration_count=context.iteration_count,
        max_iterations=context.max_iterations,
    )

    v_dict = verification.to_dict()
    _record_to_memory(
        status="evaluated",
        user_intent=user_intent,
        model_id=model_id,
        verification=v_dict,
        intent_spec=None,
        iterations=0,
    )

    return _safe_to_json({
        "status": "evaluated",
        "intent_type": "evaluation",
        "delegation_sequence": delegation_sequence,
        "precondition_warnings": precondition_warnings,
        "intent_spec": None,
        "verification": v_dict,
        "validation": [],
        "iterations": 0,
        "history": [],
        "schemas_used": schemas_used,
    })


def _handle_exploration(
    *,
    router,
    context,
    user_intent: str,
    model_id: str,
    workflow_state,
    delegation_sequence: list[str],
    precondition_warnings: list[str],
    schemas_used: dict[str, str],
) -> str:
    """Handle exploration intent — run Intent Agent only, no verification."""
    wf_params = _extract_workflow_params(workflow_state)

    intent_spec = router.intent_agent.translate(
        user_intent=user_intent,
        model_id=model_id,
        workflow_state=wf_params,
    )

    # Validate mutations against live ComfyUI (non-blocking)
    validation_results: list[dict] = []
    if _is_comfyui_available():
        validation_results = _validate_intent_mutations(intent_spec)
        for vr in validation_results:
            if vr.get("status") == "warning":
                precondition_warnings.append(
                    f"Validation: {vr.get('message', '')}"
                )

    return _safe_to_json({
        "status": "planned",
        "intent_type": "exploration",
        "delegation_sequence": delegation_sequence,
        "precondition_warnings": precondition_warnings,
        "intent_spec": intent_spec.to_dict(),
        "verification": None,
        "validation": validation_results,
        "iterations": 0,
        "history": [],
        "schemas_used": schemas_used,
    })


def _handle_generation_or_modification(
    *,
    router,
    context,
    user_intent: str,
    model_id: str,
    workflow_state,
    output_analysis,
    max_iterations: int,
    delegation_sequence: list[str],
    precondition_warnings: list[str],
    schemas_used: dict[str, str],
) -> str:
    """Handle generation/modification — Intent Agent + optional Verify loop."""
    wf_params = _extract_workflow_params(workflow_state)
    history: list[dict] = []
    iterations = 0

    # Query memory for learned patterns to inform Intent Agent
    learned_patterns = None
    try:
        raw = BrainAgent.dispatch("get_learned_patterns", {"session": current_conn_session(), "model_filter": model_id})
        parsed = _json.loads(raw) if raw else {}
        if parsed and not parsed.get("error"):
            learned_patterns = parsed
            log.debug("Loaded learned patterns for model %s: %d entries", model_id,
                      len(parsed.get("best_models", [])))
    except Exception as exc:
        log.debug("Could not load learned patterns for MoE: %s", exc)

    # Build refinement context from learned patterns (negative patterns become hints)
    initial_refinement = None
    if learned_patterns and learned_patterns.get("negative_patterns"):
        initial_refinement = [
            {"type": "avoid_pattern", "target": p.get("param", ""),
             "recommendation": p.get("issue", "")}
            for p in learned_patterns["negative_patterns"][:2]
        ]

    # First pass: run Intent Agent
    intent_spec = router.intent_agent.translate(
        user_intent=user_intent,
        model_id=model_id,
        workflow_state=wf_params,
        refinement_context=initial_refinement,
    )

    # Validate mutations against live ComfyUI (non-blocking)
    validation_results: list[dict] = []
    if _is_comfyui_available():
        validation_results = _validate_intent_mutations(intent_spec)
        for vr in validation_results:
            if vr.get("status") == "warning":
                precondition_warnings.append(
                    f"Validation: {vr.get('message', '')}"
                )

    # Check confidence
    confidence_check = router.check_confidence(intent_spec)
    if not confidence_check.get("proceed", True):  # Cycle 56: guard bare bracket access
        return _safe_to_json({
            "status": "needs_clarification",
            "intent_type": context.intent_type,
            "delegation_sequence": delegation_sequence,
            "precondition_warnings": precondition_warnings,
            "intent_spec": intent_spec.to_dict(),
            "verification": None,
            "validation": validation_results,
            "iterations": 0,
            "history": [],
            "schemas_used": schemas_used,
        })

    # If no output_analysis, just return the plan
    if output_analysis is None:
        return _safe_to_json({
            "status": "planned",
            "intent_type": context.intent_type,
            "delegation_sequence": delegation_sequence,
            "precondition_warnings": precondition_warnings,
            "intent_spec": intent_spec.to_dict(),
            "verification": None,
            "validation": validation_results,
            "iterations": 0,
            "history": [],
            "schemas_used": schemas_used,
        })

    # Extract parameters from workflow for Verify Agent enrichment
    parameters_used = _extract_parameters_from_workflow(workflow_state)

    # Output analysis provided — run verify + refinement loop
    refinement_context = None
    last_verification = None

    for i in range(max_iterations):
        iterations = i + 1

        # Run Verify Agent
        verification = router.verify_agent.evaluate(
            output_analysis=output_analysis,
            original_intent=user_intent,
            model_id=model_id,
            parameters_used=parameters_used,
            iteration_count=context.iteration_count,
            max_iterations=context.max_iterations,
        )
        last_verification = verification

        # Record in context
        iteration_record = {
            "iteration": iterations,
            "intent_spec": intent_spec.to_dict(),
            "verification": verification.to_dict(),
        }
        router.record_iteration(context, iteration_record)
        history.append(iteration_record)

        # Check loop control
        loop_decision = router.should_continue(verification, context)

        if loop_decision.get("action") == "accept":  # Cycle 56: guard bare bracket access
            i_dict = intent_spec.to_dict()
            v_dict = verification.to_dict()
            _record_to_memory(
                status="accepted",
                user_intent=user_intent,
                model_id=model_id,
                verification=v_dict,
                intent_spec=i_dict,
                iterations=iterations,
            )
            return _safe_to_json({
                "status": "accepted",
                "intent_type": context.intent_type,
                "delegation_sequence": delegation_sequence,
                "precondition_warnings": precondition_warnings,
                "intent_spec": i_dict,
                "verification": v_dict,
                "validation": validation_results,
                "iterations": iterations,
                "history": history,
                "schemas_used": schemas_used,
            })

        if not loop_decision.get("continue", False):  # Cycle 56: guard bare bracket access
            # Escalated or max iterations
            i_dict = intent_spec.to_dict()
            v_dict = verification.to_dict()
            _record_to_memory(
                status="escalated",
                user_intent=user_intent,
                model_id=model_id,
                verification=v_dict,
                intent_spec=i_dict,
                iterations=iterations,
            )
            return _safe_to_json({
                "status": "escalated",
                "intent_type": context.intent_type,
                "delegation_sequence": delegation_sequence,
                "precondition_warnings": precondition_warnings,
                "intent_spec": i_dict,
                "verification": v_dict,
                "validation": validation_results,
                "iterations": iterations,
                "history": history,
                "schemas_used": schemas_used,
            })

        # Continue: feed refinement actions back to Intent Agent
        refinement_context = loop_decision.get("refinement_actions")

        # Re-run Intent Agent with refinement context
        intent_spec = router.intent_agent.translate(
            user_intent=user_intent,
            model_id=model_id,
            workflow_state=wf_params,
            refinement_context=refinement_context,
        )

    # If we get here, we exhausted max_iterations via the for loop
    i_dict = intent_spec.to_dict()
    v_dict = last_verification.to_dict() if last_verification else None
    _record_to_memory(
        status="refined",
        user_intent=user_intent,
        model_id=model_id,
        verification=v_dict,
        intent_spec=i_dict,
        iterations=iterations,
    )
    return _safe_to_json({
        "status": "refined",
        "intent_type": context.intent_type,
        "delegation_sequence": delegation_sequence,
        "precondition_warnings": precondition_warnings,
        "intent_spec": i_dict,
        "verification": v_dict,
        "validation": validation_results,
        "iterations": iterations,
        "history": history,
        "schemas_used": schemas_used,
    })


def _extract_workflow_params(workflow_state) -> dict | None:
    """Extract workflow params dict from workflow_state input."""
    if isinstance(workflow_state, dict):
        params = {
            k: v for k, v in workflow_state.items()
            if k != "state"
        }
        return params or None
    return None


# ---------------------------------------------------------------------------
# Validation helpers (P4: MoE → Tools grounding)
# ---------------------------------------------------------------------------

_EXTRACTABLE_PARAMS = frozenset({
    "cfg", "steps", "denoise", "sampler_name", "scheduler",
})


def _is_comfyui_available() -> bool:
    """Check whether ComfyUI is reachable (fire-and-forget)."""
    try:
        raw = _tools_mod.handle("is_comfyui_running", {})
        parsed = _json.loads(raw) if isinstance(raw, str) else raw
        return bool(parsed.get("running"))
    except Exception:
        return False


def _validate_intent_mutations(intent_spec) -> list[dict]:
    """Validate Intent Agent mutations against real ComfyUI node info.

    For each ParameterMutation.target (format ``NodeClass.input_name``),
    checks that the node class exists and the input name is registered.
    Caches ``get_node_info`` results per node class within a single call.

    Returns a list of validation result dicts, one per mutation.
    On any exception the function returns an empty list (non-blocking).
    """
    try:
        mutations = intent_spec.parameter_mutations
        if not mutations:
            return []

        node_cache: dict[str, dict | None] = {}
        results: list[dict] = []

        for m in mutations:
            target = m.target
            if "." not in target:
                results.append({
                    "target": target,
                    "node_class": target,
                    "input_name": "",
                    "node_exists": False,
                    "input_exists": False,
                    "status": "warning",
                    "message": f"Invalid target format (expected NodeClass.input): {target}",
                })
                continue

            node_class, input_name = target.rsplit(".", 1)

            # Fetch node info (cached per node_class)
            if node_class not in node_cache:
                try:
                    raw = _tools_mod.handle("get_node_info", {"node_type": node_class})
                    parsed = _json.loads(raw) if isinstance(raw, str) else raw
                    node_cache[node_class] = parsed if not parsed.get("error") else None
                except Exception as _e:  # Cycle 62: log instead of silently swallow
                    log.debug("Node info cache miss for %r: %s", node_class, _e)
                    node_cache[node_class] = None

            info = node_cache[node_class]
            node_exists = info is not None

            # Check input exists in required or optional
            input_exists = False
            if node_exists:
                required = info.get("input", {}).get("required", {})
                optional = info.get("input", {}).get("optional", {})
                input_exists = input_name in required or input_name in optional

            if not node_exists:
                results.append({
                    "target": target,
                    "node_class": node_class,
                    "input_name": input_name,
                    "node_exists": False,
                    "input_exists": False,
                    "status": "warning",
                    "message": f"Node class '{node_class}' not found in ComfyUI",
                })
            elif not input_exists:
                results.append({
                    "target": target,
                    "node_class": node_class,
                    "input_name": input_name,
                    "node_exists": True,
                    "input_exists": False,
                    "status": "warning",
                    "message": f"Input '{input_name}' not found on node '{node_class}'",
                })
            else:
                results.append({
                    "target": target,
                    "node_class": node_class,
                    "input_name": input_name,
                    "node_exists": True,
                    "input_exists": True,
                    "status": "ok",
                })

        return results
    except Exception as exc:
        log.debug("Validation of intent mutations failed (non-fatal): %s", exc)
        return []


def _extract_parameters_from_workflow(workflow_state) -> dict | None:
    """Walk workflow nodes and extract literal parameter values.

    Looks for ``cfg``, ``steps``, ``denoise``, ``sampler_name``, and
    ``scheduler`` inputs across all nodes.  Skips connection inputs
    (``[node_id, idx]`` lists).  Returns a flat dict or ``None``.
    """
    if not isinstance(workflow_state, dict):
        return None

    params: dict = {}
    for _node_id, node in sorted(workflow_state.items()):
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, val in inputs.items():
            if key in _EXTRACTABLE_PARAMS and not isinstance(val, list):
                params[key] = val

    return params or None


# ---------------------------------------------------------------------------
# IterativeRefineAgent — SDK class
# ---------------------------------------------------------------------------

class IterativeRefineAgent(BrainAgent):
    """MoE pipeline orchestrator -- brain agent wrapper."""

    TOOLS = _TOOLS

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def handle(self, name: str, tool_input: dict) -> str:
        if name == "iterative_refine":
            return _handle_iterative_refine(tool_input)
        elif name == "classify_intent":
            return _handle_classify_intent(tool_input)
        return _safe_to_json({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Backward-compat re-exports (consumed by tests outside test_brain_*.py)
# ---------------------------------------------------------------------------

TOOLS = _TOOLS


def handle(name: str, tool_input: dict) -> str:
    """Module-level dispatch shim — routes to the registered instance."""
    BrainAgent._register_all()
    agent = BrainAgent._registry.get(name)
    if agent is not None:
        return agent.handle(name, tool_input)
    import json as _json  # Cycle 43: return error directly; avoid circular import via brain package
    return _json.dumps({"error": f"Unknown tool: {name}"}, sort_keys=True, allow_nan=False)  # Cycle 59
