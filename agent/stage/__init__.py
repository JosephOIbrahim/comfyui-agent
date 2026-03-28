"""USD-native cognitive stage for the ComfyUI agent ecosystem.

This package provides the CognitiveWorkflowStage (a pxr.Usd.Stage wrapper)
with LIVRPS composition, anchor parameter immunity, and bidirectional
workflow JSON <-> USD prim mapping.

Requires usd-core: pip install usd-core

Usage:
    from agent.stage import CognitiveWorkflowStage, workflow_json_to_prims

    cws = CognitiveWorkflowStage()
    workflow_json_to_prims(cws, workflow_json, "my_workflow")
    cws.add_agent_delta("forge", {"/workflows/my_workflow/nodes/node_3:input:steps": 50})
"""

from .anchors import (
    ANCHOR_PARAMS,
    AnchorViolationError,
    check_anchor,
    is_anchor,
)
from .cognitive_stage import (
    HAS_USD,
    STAGE_HIERARCHY,
    CognitiveWorkflowStage,
    StageError,
)
from .model_registry import (
    MODEL_TYPES,
    VALID_STATUSES,
    find_model,
    get_model,
    list_models_by_status,
    list_models_by_type,
    reconcile,
    register_model,
    update_status,
)
from .workflow_mapper import prims_to_workflow_json, workflow_json_to_prims

__all__ = [
    "ANCHOR_PARAMS",
    "AnchorViolationError",
    "CognitiveWorkflowStage",
    "HAS_USD",
    "STAGE_HIERARCHY",
    "StageError",
    "check_anchor",
    "is_anchor",
    "MODEL_TYPES",
    "VALID_STATUSES",
    "find_model",
    "get_model",
    "list_models_by_status",
    "list_models_by_type",
    "prims_to_workflow_json",
    "reconcile",
    "register_model",
    "update_status",
    "workflow_json_to_prims",
]
