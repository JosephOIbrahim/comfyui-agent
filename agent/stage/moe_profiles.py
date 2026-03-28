"""MoE Agent Profiles — 6 specialist roles for the agentic ecosystem.

Each profile defines:
  - name: Role identifier
  - system_prompt_fragment: Injected into the agent's system prompt
  - allowed_tools: Tool names this agent may invoke (from the 99 available)
  - authority_rules: What the agent owns and what it cannot do
  - handoff_artifact_type: Typed artifact produced for the next agent

Based on AUTHORITY_RULES from agent/agents/router.py, expanded to 6 roles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    """Immutable profile for a specialist agent."""

    name: str
    system_prompt_fragment: str
    allowed_tools: tuple[str, ...]
    authority_rules: dict[str, list[str]]
    handoff_artifact_type: str

    def to_dict(self) -> dict:
        return {
            "allowed_tools": list(self.allowed_tools),
            "authority_rules": self.authority_rules,
            "handoff_artifact_type": self.handoff_artifact_type,
            "name": self.name,
            "system_prompt_fragment": self.system_prompt_fragment,
        }

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if this profile is allowed to use a given tool."""
        return tool_name in self.allowed_tools

    def owns(self, action: str) -> bool:
        """Check if this profile owns the given action."""
        return action in self.authority_rules.get("owns", [])

    def is_forbidden(self, action: str) -> bool:
        """Check if this profile is explicitly forbidden from the action."""
        return action in self.authority_rules.get("cannot", [])


# ---------------------------------------------------------------------------
# The 6 MoE Agent Profiles
# ---------------------------------------------------------------------------

SCOUT = AgentProfile(
    name="scout",
    system_prompt_fragment=(
        "You are the Scout. Your job is RECONNAISSANCE ONLY. "
        "Discover what nodes, models, and custom nodes are available. "
        "Map the environment. Report findings. Never modify anything."
    ),
    allowed_tools=(
        "is_comfyui_running", "get_all_nodes", "get_node_info",
        "list_custom_nodes", "list_models", "get_models_summary",
        "read_node_source", "discover", "find_missing_nodes",
        "check_registry_freshness", "get_install_instructions",
        "get_civitai_model", "get_trending_models",
        "identify_model_family", "check_model_compatibility",
        "check_node_updates", "get_repo_releases",
        "list_workflow_templates", "get_workflow_template",
        "get_system_stats", "get_queue_status",
        "stage_read", "stage_list_deltas",
        "get_experience_stats", "get_prediction_accuracy",
        "list_counterfactuals",
    ),
    authority_rules={
        "owns": [
            "environment_discovery",
            "model_identification",
            "node_enumeration",
            "compatibility_checks",
        ],
        "cannot": [
            "modify_workflow",
            "execute_workflow",
            "write_stage",
            "provision_models",
            "judge_quality",
        ],
    },
    handoff_artifact_type="recon_report",
)

ARCHITECT = AgentProfile(
    name="architect",
    system_prompt_fragment=(
        "You are the Architect. Design workflow modifications and plans. "
        "Translate user intent into actionable specifications. "
        "You plan but never execute or modify workflows directly."
    ),
    allowed_tools=(
        "plan_goal", "get_plan", "complete_step", "replan",
        "capture_intent", "get_current_intent", "classify_intent",
        "classify_workflow", "predict_experiment",
        "stage_read", "stage_write", "stage_add_delta",
        "get_editable_fields", "load_workflow", "validate_workflow",
        "list_workflow_templates", "get_workflow_template",
    ),
    authority_rules={
        "owns": [
            "intent_translation",
            "parameter_decisions",
            "workflow_planning",
            "experiment_prediction",
        ],
        "cannot": [
            "execute_workflow",
            "apply_patches",
            "provision_models",
            "judge_quality",
        ],
    },
    handoff_artifact_type="design_spec",
)

PROVISIONER = AgentProfile(
    name="provisioner",
    system_prompt_fragment=(
        "You are the Provisioner. Download, verify, and manage models. "
        "Ensure all required assets are present before build begins. "
        "You provision but never modify workflows or execute them."
    ),
    allowed_tools=(
        "provision_download", "provision_verify", "provision_status",
        "discover", "list_models", "get_models_summary",
        "get_civitai_model", "get_trending_models",
        "identify_model_family", "check_model_compatibility",
    ),
    authority_rules={
        "owns": [
            "model_provisioning",
            "asset_verification",
            "download_management",
        ],
        "cannot": [
            "modify_workflow",
            "execute_workflow",
            "judge_quality",
            "translate_intent",
        ],
    },
    handoff_artifact_type="provision_manifest",
)

FORGE = AgentProfile(
    name="forge",
    system_prompt_fragment=(
        "You are the Forge. Build and modify workflows surgically. "
        "Apply validated patches, add nodes, wire connections. "
        "You build but never judge quality or provision models."
    ),
    allowed_tools=(
        "load_workflow", "validate_workflow", "get_editable_fields",
        "apply_workflow_patch", "preview_workflow_patch",
        "undo_workflow_patch", "get_workflow_diff",
        "save_workflow", "reset_workflow",
        "add_node", "connect_nodes", "set_input",
        "get_node_info", "get_all_nodes",
        "stage_write", "stage_add_delta",
        "get_node_replacements", "check_workflow_deprecations",
        "migrate_deprecated_nodes",
    ),
    authority_rules={
        "owns": [
            "workflow_mutation",
            "rfc6902_patching",
            "node_wiring",
            "deprecation_migration",
        ],
        "cannot": [
            "execute_workflow",
            "judge_quality",
            "provision_models",
            "translate_intent",
        ],
    },
    handoff_artifact_type="build_artifact",
)

CRUCIBLE = AgentProfile(
    name="crucible",
    system_prompt_fragment=(
        "You are the Crucible. Execute workflows and verify results. "
        "Run validation, execute, check outputs. "
        "You test but never modify workflows or translate intent."
    ),
    allowed_tools=(
        "validate_before_execute", "execute_workflow",
        "get_execution_status", "execute_with_progress",
        "verify_execution", "get_output_path",
        "hash_compare_images",
        "get_queue_status", "get_history", "get_system_stats",
        "validate_workflow",
        "check_workflow_deprecations",
    ),
    authority_rules={
        "owns": [
            "workflow_execution",
            "execution_verification",
            "output_validation",
        ],
        "cannot": [
            "modify_workflow",
            "translate_intent",
            "provision_models",
            "judge_aesthetic_quality",
        ],
    },
    handoff_artifact_type="execution_result",
)

VISION = AgentProfile(
    name="vision",
    system_prompt_fragment=(
        "You are the Vision agent. Analyze outputs, judge quality, "
        "record experiences, and recommend improvements. "
        "You judge but never modify or execute workflows."
    ),
    allowed_tools=(
        "analyze_image", "compare_outputs", "suggest_improvements",
        "record_outcome", "get_learned_patterns", "get_recommendations",
        "detect_implicit_feedback", "record_experience",
        "iterative_refine",
        "start_iteration_tracking", "record_iteration_step",
        "finalize_iterations",
        "write_image_metadata", "read_image_metadata",
        "reconstruct_context",
    ),
    authority_rules={
        "owns": [
            "quality_judgment",
            "iteration_decisions",
            "experience_recording",
            "improvement_recommendations",
        ],
        "cannot": [
            "modify_workflow",
            "execute_workflow",
            "provision_models",
            "translate_intent",
        ],
    },
    handoff_artifact_type="quality_report",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_PROFILES: dict[str, AgentProfile] = {
    "scout": SCOUT,
    "architect": ARCHITECT,
    "provisioner": PROVISIONER,
    "forge": FORGE,
    "crucible": CRUCIBLE,
    "vision": VISION,
}

PROFILE_NAMES: tuple[str, ...] = tuple(ALL_PROFILES.keys())

# Default chain order for full pipeline
DEFAULT_CHAIN: tuple[str, ...] = (
    "scout", "architect", "provisioner", "forge", "crucible", "vision",
)


def get_profile(name: str) -> AgentProfile | None:
    """Look up a profile by name. Returns None if not found."""
    return ALL_PROFILES.get(name)


def get_allowed_tools(name: str) -> tuple[str, ...]:
    """Return allowed tools for a profile. Empty tuple if not found."""
    profile = ALL_PROFILES.get(name)
    return profile.allowed_tools if profile else ()


def filter_tools(profile_name: str, tool_names: list[str]) -> list[str]:
    """Filter a list of tool names to only those allowed by the profile."""
    profile = ALL_PROFILES.get(profile_name)
    if profile is None:
        return []
    allowed = set(profile.allowed_tools)
    return [t for t in tool_names if t in allowed]
