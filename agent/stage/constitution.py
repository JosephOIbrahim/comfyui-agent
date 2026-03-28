"""Constitutional Enforcer — 8 commandments as executable pre/post checks.

Each commandment is a callable check that returns (passed: bool, reason: str).
Checks can be used as pre-conditions (before action) or post-conditions (after).

Commandments:
  1. scout_before_act      — first action must be read/discover
  2. verify_after_mutation  — test after every change
  3. bounded_failure        — 3 retries max
  4. complete_output        — no stubs/TODOs in output
  5. role_isolation          — agent can only use allowed tools
  6. explicit_handoffs       — typed artifacts required between agents
  7. adversarial_verification — builder != breaker
  8. human_gates             — pause at irreversible transitions
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    """Result of a constitutional check."""

    passed: bool
    commandment: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "commandment": self.commandment,
            "passed": self.passed,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Read-only / recon tool names (allowed as first actions)
# ---------------------------------------------------------------------------

_RECON_TOOLS: frozenset[str] = frozenset({
    "is_comfyui_running", "get_all_nodes", "get_node_info",
    "list_custom_nodes", "list_models", "get_models_summary",
    "read_node_source", "discover", "find_missing_nodes",
    "check_registry_freshness", "get_install_instructions",
    "load_workflow", "validate_workflow", "get_editable_fields",
    "classify_workflow", "stage_read", "stage_list_deltas",
    "stage_reconstruct_clean", "get_system_stats", "get_queue_status",
    "get_history", "get_plan",
    "list_workflow_templates", "get_workflow_template",
    "list_sessions", "get_experience_stats", "list_counterfactuals",
    "get_prediction_accuracy", "get_learned_patterns",
    "get_recommendations", "get_current_intent",
    "get_execution_status", "get_output_path",
    "get_civitai_model", "get_trending_models",
    "identify_model_family", "check_model_compatibility",
    "check_node_updates", "get_repo_releases",
    "get_node_replacements", "check_workflow_deprecations",
    "read_image_metadata", "reconstruct_context",
    "get_models_summary", "provision_status",
    "get_workflow_diff",
})

_MUTATION_TOOLS: frozenset[str] = frozenset({
    "apply_workflow_patch", "add_node", "connect_nodes", "set_input",
    "save_workflow", "reset_workflow", "undo_workflow_patch",
    "stage_write", "stage_add_delta", "stage_rollback",
    "migrate_deprecated_nodes",
    "provision_download",
})

_VERIFY_TOOLS: frozenset[str] = frozenset({
    "validate_before_execute", "validate_workflow",
    "execute_workflow", "execute_with_progress",
    "verify_execution", "hash_compare_images",
    "analyze_image", "compare_outputs",
})

# Stub/TODO patterns
_STUB_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"\bXXX\b"),
    re.compile(r"\bstub\b", re.IGNORECASE),
    re.compile(r"pass\s*#\s*(todo|fixme|stub)", re.IGNORECASE),
    re.compile(r"raise\s+NotImplementedError"),
)

# Irreversible actions that require human gate
_IRREVERSIBLE_ACTIONS: frozenset[str] = frozenset({
    "reset_workflow",
    "provision_download",
    "stage_rollback",
    "migrate_deprecated_nodes",
})


# ---------------------------------------------------------------------------
# Individual Commandment Checks
# ---------------------------------------------------------------------------

def scout_before_act(
    action_history: list[str],
    proposed_tool: str,
) -> CheckResult:
    """Commandment 1: First action in a chain must be a recon/read tool.

    If no actions have been taken yet and the proposed tool is not a recon
    tool, this check fails.
    """
    if not action_history and proposed_tool not in _RECON_TOOLS:
        return CheckResult(
            passed=False,
            commandment="scout_before_act",
            reason=(
                f"First action must be a recon tool, not '{proposed_tool}'. "
                f"Scout before you act."
            ),
        )
    return CheckResult(
        passed=True,
        commandment="scout_before_act",
        reason="Recon requirement satisfied.",
    )


def verify_after_mutation(
    last_action: str | None,
    proposed_tool: str,
    verified_since_mutation: bool,
) -> CheckResult:
    """Commandment 2: A verification tool must follow any mutation tool.

    If the last action was a mutation and the proposed action is not
    verification, this check fails.
    """
    if (
        last_action in _MUTATION_TOOLS
        and not verified_since_mutation
        and proposed_tool not in _VERIFY_TOOLS
        and proposed_tool not in _RECON_TOOLS
    ):
        return CheckResult(
            passed=False,
            commandment="verify_after_mutation",
            reason=(
                f"Must verify after mutation '{last_action}' before "
                f"proceeding with '{proposed_tool}'."
            ),
        )
    return CheckResult(
        passed=True,
        commandment="verify_after_mutation",
        reason="Verify-after-mutation satisfied.",
    )


def bounded_failure(
    retry_count: int,
    max_retries: int = 3,
) -> CheckResult:
    """Commandment 3: Maximum retry limit before marking as BLOCKER."""
    if retry_count >= max_retries:
        return CheckResult(
            passed=False,
            commandment="bounded_failure",
            reason=(
                f"Exceeded {max_retries} retries. Marking as BLOCKER."
            ),
        )
    return CheckResult(
        passed=True,
        commandment="bounded_failure",
        reason=f"Retry {retry_count}/{max_retries} — within bounds.",
    )


def complete_output(output: str) -> CheckResult:
    """Commandment 4: Output must not contain stubs or TODOs."""
    for pattern in _STUB_PATTERNS:
        match = pattern.search(output)
        if match:
            return CheckResult(
                passed=False,
                commandment="complete_output",
                reason=(
                    f"Output contains stub/TODO: '{match.group()}'. "
                    f"Complete the implementation."
                ),
            )
    return CheckResult(
        passed=True,
        commandment="complete_output",
        reason="Output is complete — no stubs or TODOs found.",
    )


def role_isolation(
    agent_name: str,
    tool_name: str,
    allowed_tools: frozenset[str] | set[str] | tuple[str, ...] | list[str],
) -> CheckResult:
    """Commandment 5: Agent can only use tools in its allowed set."""
    allowed = set(allowed_tools)
    if tool_name not in allowed:
        return CheckResult(
            passed=False,
            commandment="role_isolation",
            reason=(
                f"Agent '{agent_name}' is not allowed to use "
                f"tool '{tool_name}'."
            ),
        )
    return CheckResult(
        passed=True,
        commandment="role_isolation",
        reason=f"Tool '{tool_name}' is allowed for '{agent_name}'.",
    )


def explicit_handoffs(
    artifact: dict | None,
    expected_type: str,
) -> CheckResult:
    """Commandment 6: Typed artifacts are required between agents."""
    if artifact is None:
        return CheckResult(
            passed=False,
            commandment="explicit_handoffs",
            reason=(
                f"No handoff artifact provided. Expected type: "
                f"'{expected_type}'."
            ),
        )
    actual_type = artifact.get("artifact_type", "")
    if actual_type != expected_type:
        return CheckResult(
            passed=False,
            commandment="explicit_handoffs",
            reason=(
                f"Artifact type mismatch: got '{actual_type}', "
                f"expected '{expected_type}'."
            ),
        )
    return CheckResult(
        passed=True,
        commandment="explicit_handoffs",
        reason=f"Handoff artifact type '{expected_type}' matches.",
    )


def adversarial_verification(
    builder: str,
    verifier: str,
) -> CheckResult:
    """Commandment 7: Builder and verifier must be different agents."""
    if builder == verifier:
        return CheckResult(
            passed=False,
            commandment="adversarial_verification",
            reason=(
                f"Builder and verifier are the same agent: '{builder}'. "
                f"Adversarial verification requires different agents."
            ),
        )
    return CheckResult(
        passed=True,
        commandment="adversarial_verification",
        reason=f"Builder '{builder}' != verifier '{verifier}'. OK.",
    )


def human_gates(
    proposed_tool: str,
    human_approved: bool = False,
) -> CheckResult:
    """Commandment 8: Pause at irreversible transitions unless approved."""
    if proposed_tool in _IRREVERSIBLE_ACTIONS and not human_approved:
        return CheckResult(
            passed=False,
            commandment="human_gates",
            reason=(
                f"Tool '{proposed_tool}' is irreversible and requires "
                f"human approval before proceeding."
            ),
        )
    return CheckResult(
        passed=True,
        commandment="human_gates",
        reason="No human gate required, or approval granted.",
    )


# ---------------------------------------------------------------------------
# Aggregate checks
# ---------------------------------------------------------------------------

ALL_COMMANDMENTS: tuple[str, ...] = (
    "scout_before_act",
    "verify_after_mutation",
    "bounded_failure",
    "complete_output",
    "role_isolation",
    "explicit_handoffs",
    "adversarial_verification",
    "human_gates",
)


def run_pre_checks(
    *,
    agent_name: str,
    proposed_tool: str,
    allowed_tools: frozenset[str] | set[str] | tuple[str, ...] | list[str],
    action_history: list[str],
    last_action: str | None = None,
    verified_since_mutation: bool = True,
    retry_count: int = 0,
    max_retries: int = 3,
    human_approved: bool = False,
) -> list[CheckResult]:
    """Run all pre-action constitutional checks.

    Returns list of CheckResults (both passed and failed).
    """
    results: list[CheckResult] = []

    results.append(scout_before_act(action_history, proposed_tool))
    results.append(verify_after_mutation(
        last_action, proposed_tool, verified_since_mutation,
    ))
    results.append(bounded_failure(retry_count, max_retries))
    results.append(role_isolation(agent_name, proposed_tool, allowed_tools))
    results.append(human_gates(proposed_tool, human_approved))

    return results


def run_post_checks(
    *,
    output: str,
    builder: str | None = None,
    verifier: str | None = None,
    handoff_artifact: dict | None = None,
    expected_artifact_type: str = "",
) -> list[CheckResult]:
    """Run all post-action constitutional checks.

    Returns list of CheckResults (both passed and failed).
    """
    results: list[CheckResult] = []

    results.append(complete_output(output))

    if builder is not None and verifier is not None:
        results.append(adversarial_verification(builder, verifier))

    if expected_artifact_type:
        results.append(explicit_handoffs(
            handoff_artifact, expected_artifact_type,
        ))

    return results


def all_passed(results: list[CheckResult]) -> bool:
    """Check if all results passed."""
    return all(r.passed for r in results)


def failed_checks(results: list[CheckResult]) -> list[CheckResult]:
    """Return only the failed checks."""
    return [r for r in results if not r.passed]
