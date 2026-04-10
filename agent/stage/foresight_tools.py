"""FORESIGHT tools — expose CWM, Experience, and Counterfactual ops as MCP tools.

Five tools for the FORESIGHT prediction/learning pipeline:
  predict_experiment     — CWM prediction for a proposed change
  record_experience      — manually record a workflow experience
  get_experience_stats   — aggregate statistics over stored experiences
  list_counterfactuals   — list pending + validated counterfactuals
  get_prediction_accuracy — prediction accuracy stats from ratchet history

Tool pattern: TOOLS list[dict] + handle(name, tool_input) -> str.
Registered in agent/tools/__init__.py via the stage module import.
"""

from __future__ import annotations

import logging

from ..tools._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "predict_experiment",
        "description": (
            "Predict the outcome of a proposed workflow change using the "
            "Cognitive World Model (CWM). Returns predicted axis scores, "
            "confidence, learning phase, and the arbiter's surfacing mode."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "proposed_change": {
                    "type": "object",
                    "description": (
                        "Description of the proposed change. Common keys: "
                        "'param' (steps/cfg/resolution), 'direction' "
                        "(increase/decrease), 'action' (add_controlnet/add_lora)."
                    ),
                },
            },
            "required": ["proposed_change"],
        },
    },
    {
        "name": "record_experience",
        "description": (
            "Manually record a workflow experience with outcome scores. "
            "Captures initial state, decisions made, and 6-axis outcome "
            "scores (aesthetic, depth, normals, camera, segmentation, lighting)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "initial_state": {
                    "type": "object",
                    "description": "Snapshot of workflow state before change.",
                },
                "decisions": {
                    "type": "array",
                    "description": "List of decisions made (dicts).",
                    "items": {"type": "object"},
                },
                "outcome": {
                    "type": "object",
                    "description": (
                        "6-axis outcome scores. Keys: aesthetic, depth, normals, "
                        "camera, segmentation, lighting. Values: float [0, 1]."
                    ),
                },
                "context_signature_hash": {
                    "type": "string",
                    "description": "WorkflowSignature hash for context matching.",
                },
            },
            "required": ["initial_state", "decisions", "outcome"],
        },
    },
    {
        "name": "get_experience_stats",
        "description": (
            "Get aggregate statistics over stored experiences. Returns total "
            "count, average outcome per axis, average prediction accuracy, "
            "and unique signature count. Optionally filter by signature hash."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context_signature_hash": {
                    "type": "string",
                    "description": (
                        "Filter by signature hash. Omit for all experiences."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_counterfactuals",
        "description": (
            "List pending and validated counterfactual simulations. "
            "Counterfactuals are 'what if' alternatives generated at session "
            "close. Pending ones can be validated; validated ones have been "
            "promoted based on observed accuracy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "validated", "all"],
                    "description": "Filter by status. Default: all.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_prediction_accuracy",
        "description": (
            "Get prediction accuracy statistics from the current ratchet "
            "session. Shows how well the CWM predicted actual outcomes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def _get_stage(session_id: str = "default"):
    """Return the CognitiveWorkflowStage for this session, or None."""
    from ..session_context import get_session_context
    ctx = get_session_context(session_id)
    return ctx.ensure_stage()


def _get_ratchet(session_id: str = "default"):
    """Return the Ratchet for this session, or None."""
    from ..session_context import get_session_context
    ctx = get_session_context(session_id)
    return ctx.ensure_ratchet()


def _get_ctx(session_id: str = "default"):
    """Return the SessionContext."""
    from ..session_context import get_session_context
    return get_session_context(session_id)


_NO_STAGE = to_json({
    "error": (
        "CognitiveWorkflowStage is not available. "
        "usd-core may not be installed in this environment."
    )
})


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_predict_experiment(tool_input: dict) -> str:
    proposed_change = tool_input.get("proposed_change")  # Cycle 55: guard before stage check
    if proposed_change is None:
        return to_json({"error": "proposed_change is required."})

    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    try:
        from .cwm import predict
        ctx = _get_ctx()
        prediction = predict(
            stage, proposed_change,
            current_signature=ctx.workflow_signature,
        )

        result = prediction.to_dict()

        # Optionally consult arbiter
        if ctx.arbiter is not None:
            ratchet = ctx.ratchet
            current_composite = 0.5
            if ratchet and ratchet.best():
                current_composite = ratchet.best().composite
            try:
                ad = ctx.arbiter.prioritize_experiment(
                    prediction, current_composite,
                )
                result["arbiter_mode"] = ad.mode
                result["arbiter_reasoning"] = ad.reasoning
            except Exception as _arb_exc:  # noqa: BLE001
                log.warning("Arbiter prioritize_experiment failed: %s", _arb_exc)

        return to_json(result)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


def _handle_record_experience(tool_input: dict) -> str:
    # Cycle 55: guard required fields before stage/infrastructure checks
    initial_state = tool_input.get("initial_state")
    decisions = tool_input.get("decisions")
    outcome = tool_input.get("outcome")
    if initial_state is None:
        return to_json({"error": "initial_state is required."})
    if decisions is None:
        return to_json({"error": "decisions is required."})
    if outcome is None:
        return to_json({"error": "outcome is required."})

    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    try:
        from .experience import record_experience
        ctx = _get_ctx()
        sig_hash = tool_input.get("context_signature_hash", "")
        if not sig_hash and ctx.workflow_signature:
            sig_hash = ctx.workflow_signature.signature_hash()
        chunk = record_experience(
            stage,
            initial_state=initial_state,
            decisions=decisions,
            outcome=outcome,
            context_signature_hash=sig_hash,
        )
        return to_json({
            "recorded": True,
            "chunk_id": chunk.chunk_id,
            "prediction_accuracy": chunk.prediction_accuracy,
        })
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


def _handle_get_experience_stats(tool_input: dict) -> str:
    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    try:
        from .experience import get_statistics
        sig_hash = tool_input.get("context_signature_hash")
        stats = get_statistics(stage, context_signature_hash=sig_hash)
        return to_json(stats)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


def _handle_list_counterfactuals(tool_input: dict) -> str:
    stage = _get_stage()
    if stage is None:
        return _NO_STAGE

    try:
        from .counterfactuals import list_pending, list_validated
        status = tool_input.get("status", "all")

        result: dict = {}

        if status in ("pending", "all"):
            pending = list_pending(stage)
            result["pending"] = [cf.to_dict() for cf in pending]
            result["pending_count"] = len(pending)

        if status in ("validated", "all"):
            validated = list_validated(stage)
            result["validated"] = [cf.to_dict() for cf in validated]
            result["validated_count"] = len(validated)

        return to_json(result)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})


def _handle_get_prediction_accuracy(tool_input: dict) -> str:  # noqa: ARG001
    ratchet = _get_ratchet()
    if ratchet is None:
        return to_json({
            "error": "Ratchet not available. Stage may not be initialized."
        })

    predicted = [
        d for d in ratchet.history if d.prediction_accuracy is not None
    ]

    if not predicted:
        return to_json({
            "predictions_made": 0,
            "foresight_enabled": ratchet.has_foresight,
            "message": "No predictions recorded yet.",
        })

    accuracies = [d.prediction_accuracy for d in predicted]
    return to_json({
        "foresight_enabled": ratchet.has_foresight,
        "predictions_made": len(predicted),
        "avg_accuracy": sum(accuracies) / len(accuracies),
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "total_decisions": len(ratchet.history),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "predict_experiment": _handle_predict_experiment,
    "record_experience": _handle_record_experience,
    "get_experience_stats": _handle_get_experience_stats,
    "list_counterfactuals": _handle_list_counterfactuals,
    "get_prediction_accuracy": _handle_get_prediction_accuracy,
}


def handle(name: str, tool_input: dict) -> str:
    """Execute a FORESIGHT tool call. Returns JSON string."""
    try:
        handler = _DISPATCH.get(name)
        if handler is None:
            return to_json({"error": f"Unknown tool: {name}"})
        return handler(tool_input)
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})
