"""Compute recommended tool scope for the current workflow state.

Pure function — examines workflow presence, validation status, risk,
and readiness to recommend which tool categories are relevant.
Returns a ``frozenset[str]`` of tool-category tags, not individual
tool names.
"""

from __future__ import annotations

from .schemas import ReadinessGrade, RiskLevel


# ---------------------------------------------------------------------------
# Tool category constants
# ---------------------------------------------------------------------------

# Each constant names a logical group — the consumer maps these to
# actual tool names via the ToolScope system in ``agent/tool_scope.py``.

CAT_LOAD = "load"                   # load_workflow, list_workflow_templates
CAT_TEMPLATE = "template"           # get_workflow_template
CAT_VALIDATE = "validate"           # validate_workflow, validate_before_execute
CAT_EDIT = "edit"                   # add_node, connect_nodes, set_input, patch
CAT_EXECUTE = "execute"             # execute_workflow, execute_with_progress
CAT_VERIFY = "verify"              # analyze_image, compare_outputs
CAT_DISCOVER = "discover"           # discover, find_missing_nodes
CAT_PROVISION = "provision"         # install_node_pack, download_model
CAT_OPTIMIZE = "optimize"           # profile_workflow, suggest_optimizations
CAT_SESSION = "session"             # save_session, load_session, add_note


def compute_tool_scope(
    *,
    workflow_loaded: bool = False,
    workflow_validated: bool = False,
    workflow_executed: bool = False,
    risk: RiskLevel = RiskLevel.SAFE,
    readiness: ReadinessGrade = ReadinessGrade.READY,
) -> frozenset[str]:
    """Recommend relevant tool categories for current state.

    The returned set grows as the workflow progresses through its
    lifecycle.  Higher risk / lower readiness adds discovery and
    provisioning tools.

    Args:
        workflow_loaded: True if a workflow JSON is in memory.
        workflow_validated: True if validation passed.
        workflow_executed: True if at least one execution completed.
        risk: Current risk level.
        readiness: Current readiness grade.

    Returns:
        Frozenset of tool category tags.
    """
    scope: set[str] = set()

    # Always available
    scope.add(CAT_SESSION)

    if not workflow_loaded:
        # Nothing loaded — offer load and template tools
        scope.add(CAT_LOAD)
        scope.add(CAT_TEMPLATE)
        return frozenset(scope)

    # Workflow is loaded
    scope.add(CAT_VALIDATE)
    scope.add(CAT_EDIT)

    # Risk / readiness gates
    if risk > RiskLevel.CAUTION or readiness >= ReadinessGrade.NEEDS_PROVISION:
        scope.add(CAT_DISCOVER)
        scope.add(CAT_PROVISION)

    if workflow_validated and readiness == ReadinessGrade.READY:
        scope.add(CAT_EXECUTE)
        scope.add(CAT_OPTIMIZE)

    if workflow_executed:
        scope.add(CAT_VERIFY)

    return frozenset(scope)
