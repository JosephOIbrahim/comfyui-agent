"""Tool authority scoping for MoE agent role isolation.

Enforces Commandment 5 (Role Isolation) at runtime: each agent gets a
ToolScope defining which tools it can call. The ScopedDispatcher wraps
the central handle() function and rejects unauthorized calls.

Evolves from:
  - agents/router.py AUTHORITY_RULES (advisory, not enforced)
  - brain/orchestrator.py _TOOL_PROFILES (per-subtask, not per-agent)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolScope:
    """Immutable set of tools an agent is authorized to call.

    Authority is checked via allowed_tools (whitelist). Denied_tools
    is an explicit blocklist that overrides allowed_tools for safety.
    """

    name: str
    allowed_tools: frozenset[str]
    denied_tools: frozenset[str] = field(default_factory=frozenset)

    def check(self, tool_name: str) -> bool:
        """Return True if the tool is authorized in this scope."""
        if tool_name in self.denied_tools:
            return False
        return tool_name in self.allowed_tools

    def describe(self) -> dict:
        """Return a serializable description of this scope."""
        return {
            "name": self.name,
            "allowed_count": len(self.allowed_tools),
            "denied_count": len(self.denied_tools),
        }


class ScopedDispatcher:
    """Wraps tool dispatch with authority enforcement.

    When an agent tries to call a tool outside its scope, returns
    a clear error explaining the restriction and what to do instead.
    """

    def __init__(self, dispatch_fn, scope: ToolScope, ctx=None):
        """Initialize with a dispatch function, scope, and optional context.

        Args:
            dispatch_fn: The underlying handle(name, tool_input, ...) function.
            scope: ToolScope defining what this dispatcher can call.
            ctx: Optional SessionContext for session-scoped state.
        """
        self._dispatch = dispatch_fn
        self.scope = scope
        self.ctx = ctx

    def __call__(self, name: str, tool_input: dict, **kwargs) -> str:
        """Dispatch a tool call, enforcing scope."""
        if not self.scope.check(name):
            return json.dumps({
                "error": (
                    f"Tool '{name}' is not authorized for the "
                    f"'{self.scope.name}' scope."
                ),
                "hint": (
                    f"The {self.scope.name} agent cannot call {name}. "
                    "This tool belongs to a different agent's authority."
                ),
                "scope": self.scope.name,
            }, sort_keys=True, allow_nan=False)  # Cycle 61: NaN-safe
        if self.ctx is not None:
            kwargs["ctx"] = self.ctx
        return self._dispatch(name, tool_input, **kwargs)


# ---------------------------------------------------------------------------
# Predefined scopes for the MoE pipeline agents
# ---------------------------------------------------------------------------

# Read-only tools available to all agents
_READ_TOOLS = frozenset({
    "is_comfyui_running",
    "get_all_nodes",
    "get_node_info",
    "get_system_stats",
    "get_queue_status",
    "get_history",
    "list_custom_nodes",
    "list_models",
    "get_models_summary",
    "read_node_source",
    "load_workflow",
    "validate_workflow",
    "get_editable_fields",
    "list_workflow_templates",
    "get_workflow_template",
    "discover",
    "find_missing_nodes",
    "check_registry_freshness",
    "get_install_instructions",
    "identify_model_family",
    "check_model_compatibility",
    "get_civitai_model",
    "get_trending_models",
    "check_node_updates",
    "get_repo_releases",
    "get_output_path",
    "classify_workflow",
    "classify_intent",
})

SCOPE_INTENT = ToolScope(
    name="intent",
    allowed_tools=_READ_TOOLS | frozenset({
        # Intent capture
        "capture_intent",
        "get_current_intent",
        # Memory (read)
        "get_learned_patterns",
        "get_recommendations",
        "detect_implicit_feedback",
        # Planning (read)
        "get_plan",
    }),
    denied_tools=frozenset({
        # Cannot execute or mutate workflows
        "execute_workflow",
        "execute_with_progress",
        "apply_workflow_patch",
        "set_input",
        "add_node",
        "connect_nodes",
        "save_workflow",
        "reset_workflow",
        # Cannot judge quality
        "analyze_image",
        "compare_outputs",
    }),
)

SCOPE_EXECUTION = ToolScope(
    name="execution",
    allowed_tools=_READ_TOOLS | frozenset({
        # Workflow mutation
        "add_node",
        "connect_nodes",
        "set_input",
        "apply_workflow_patch",
        "preview_workflow_patch",
        "undo_workflow_patch",
        "get_workflow_diff",
        "save_workflow",
        "reset_workflow",
        # Node migration
        "get_node_replacements",
        "check_workflow_deprecations",
        "migrate_deprecated_nodes",
        # Execution
        "validate_before_execute",
        "execute_workflow",
        "execute_with_progress",
        "get_execution_status",
        # Pipeline
        "create_pipeline",
        "run_pipeline",
        "get_pipeline_status",
        # Session
        "save_session",
        "load_session",
        "list_sessions",
        "add_note",
        # Optimization
        "profile_workflow",
        "suggest_optimizations",
        "check_tensorrt_status",
        "apply_optimization",
    }),
    denied_tools=frozenset({
        # Cannot judge quality (verify agent's authority)
        "analyze_image",
        "compare_outputs",
        "suggest_improvements",
    }),
)

SCOPE_VERIFY = ToolScope(
    name="verify",
    allowed_tools=_READ_TOOLS | frozenset({
        # Vision + analysis
        "analyze_image",
        "compare_outputs",
        "suggest_improvements",
        "hash_compare_images",
        # Read-only workflow inspection
        "get_workflow_diff",
        "verify_execution",
        # Memory (read + write outcomes)
        "record_outcome",
        "get_learned_patterns",
        "get_recommendations",
        "detect_implicit_feedback",
        # Iteration tracking
        "start_iteration_tracking",
        "record_iteration_step",
        "finalize_iterations",
        # Metadata
        "read_image_metadata",
        "write_image_metadata",
        "reconstruct_context",
    }),
    denied_tools=frozenset({
        # Cannot mutate workflows (execution agent's authority)
        "apply_workflow_patch",
        "set_input",
        "add_node",
        "connect_nodes",
        "save_workflow",
        "reset_workflow",
        # Cannot execute (must recommend, not act)
        "execute_workflow",
        "execute_with_progress",
    }),
)

# Cycle 64: SCOPE_FULL is intentionally NOT in SCOPES — FullScopeDispatcher (below)
# is the production path for unrestricted access. A ToolScope with empty
# allowed_tools makes check() always return False, which is the opposite of
# "full access". Using FullScopeDispatcher avoids this semantic inversion.

# All predefined scopes by name
SCOPES: dict[str, ToolScope] = {
    "intent": SCOPE_INTENT,
    "execution": SCOPE_EXECUTION,
    "verify": SCOPE_VERIFY,
}


class FullScopeDispatcher:
    """Dispatcher with no restrictions. Used for MCP-facing tool calls.

    Unlike ScopedDispatcher, this does not check any scope — all tools
    are allowed. This is the dispatcher used by the MCP server for
    external clients (Claude Code, Claude Desktop).
    """

    def __init__(self, dispatch_fn, ctx=None):
        self._dispatch = dispatch_fn
        self.ctx = ctx

    def __call__(self, name: str, tool_input: dict, **kwargs) -> str:
        if self.ctx is not None:
            kwargs["ctx"] = self.ctx
        return self._dispatch(name, tool_input, **kwargs)
