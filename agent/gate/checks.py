"""Five pre-dispatch checks, each returning (passed: bool, reason: str).

Check order:
  1. System health   — circuit breaker state
  2. Consent         — session active, validation done, escalation needed
  3. Constitution    — wraps run_pre_checks() from constitution.py
  4. Reversibility   — undo availability for mutation tools
  5. Scope           — path sanitization for filesystem-touching inputs
"""

from __future__ import annotations

from pathlib import Path

from .risk_levels import RiskLevel
from ..stage.constitution import run_pre_checks, all_passed, failed_checks


# ---------------------------------------------------------------------------
# Blocked path prefixes (mirrors _util.py but avoids circular import)
# ---------------------------------------------------------------------------

_BLOCKED_PREFIXES: tuple[str, ...] = (
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\ProgramData",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/var",
    "/root",
)

_PATH_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "file_path",
        "output_path",
        "input_path",
        "workflow_path",
        "model_path",
        "image_path",
        "save_path",
        "directory",
        "dir",
        "source",
        "destination",
        "target",
    }
)


# ---------------------------------------------------------------------------
# Check 1: System Health
# ---------------------------------------------------------------------------


def check_system_health(
    tool_name: str,
    risk: RiskLevel,
    breaker_state: str,
) -> tuple[bool, str]:
    """Level 0 always passes.  Level 1+ fails if circuit breaker OPEN."""
    if risk == RiskLevel.READ_ONLY:
        return True, "READ_ONLY tool — system health check bypassed."

    if breaker_state == "open":
        return (
            False,
            f"Circuit breaker is OPEN. Tool '{tool_name}' (risk={risk.name}) "
            f"blocked until service recovers.",
        )

    if breaker_state == "half_open":
        return (
            True,
            f"Circuit breaker is HALF_OPEN. Tool '{tool_name}' allowed as test request.",
        )

    return True, "Circuit breaker CLOSED. System healthy."


# ---------------------------------------------------------------------------
# Check 2: Consent
# ---------------------------------------------------------------------------


def check_consent(
    tool_name: str,
    risk: RiskLevel,
    session_active: bool,
    validated: bool,
) -> tuple[bool, str]:
    """Level 0-1 pass with active session.
    Level 2 requires prior validation.
    Level 3 escalates (returned as fail with escalation hint).
    Level 4 always locked.
    """
    if risk == RiskLevel.READ_ONLY:
        return True, "READ_ONLY tool — consent check bypassed."

    if risk == RiskLevel.DESTRUCTIVE:
        return (
            False,
            f"Tool '{tool_name}' is DESTRUCTIVE (risk level 4). "
            f"Requires explicit human approval — gate returns LOCKED.",
        )

    if risk == RiskLevel.PROVISION:
        return (
            False,
            f"Tool '{tool_name}' is PROVISION (risk level 3). "
            f"Requires user confirmation — gate returns ESCALATE.",
        )

    if not session_active:
        return (
            False,
            f"No active session. Tool '{tool_name}' requires an active session.",
        )

    if risk == RiskLevel.EXECUTION and not validated:
        return (
            False,
            f"Tool '{tool_name}' is EXECUTION (risk level 2) but workflow "
            f"has not been validated. Call validate_before_execute first.",
        )

    return True, "Consent satisfied."


# ---------------------------------------------------------------------------
# Check 3: Constitution
# ---------------------------------------------------------------------------


def check_constitution(
    action_history: list[str],
    tool_name: str,
    **kwargs: object,
) -> tuple[bool, str]:
    """Wraps constitution.run_pre_checks() and returns aggregate pass/fail.

    Keyword args are forwarded to run_pre_checks(). At minimum we need
    agent_name and allowed_tools; others fall back to safe defaults.
    """
    agent_name: str = str(kwargs.get("agent_name", "default"))
    allowed_tools = kwargs.get("allowed_tools", None)
    # If no allowed_tools given, pass all tools (don't block on role check)
    if allowed_tools is None:
        allowed_tools = frozenset({tool_name})

    results = run_pre_checks(
        agent_name=agent_name,
        proposed_tool=tool_name,
        allowed_tools=allowed_tools,
        action_history=action_history,
        last_action=kwargs.get("last_action", action_history[-1] if action_history else None),
        verified_since_mutation=bool(kwargs.get("verified_since_mutation", True)),
        retry_count=int(kwargs.get("retry_count", 0)),
        max_retries=int(kwargs.get("max_retries", 3)),
        human_approved=bool(kwargs.get("human_approved", False)),
    )

    if all_passed(results):
        return True, "All constitutional checks passed."

    failures = failed_checks(results)
    reasons = "; ".join(f"[{f.commandment}] {f.reason}" for f in failures)
    return False, f"Constitutional check(s) failed: {reasons}"


# ---------------------------------------------------------------------------
# Check 4: Reversibility
# ---------------------------------------------------------------------------


def check_reversibility(
    tool_name: str,
    risk: RiskLevel,
    has_undo: bool,
) -> tuple[bool, str]:
    """Level 0 always passes.
    Level 1 requires undo capability.
    Level 2 is acceptable (outputs are additive, not destructive).
    Level 3+ warns.
    """
    if risk == RiskLevel.READ_ONLY:
        return True, "READ_ONLY — reversibility not applicable."

    if risk == RiskLevel.REVERSIBLE and not has_undo:
        return (
            False,
            f"Tool '{tool_name}' is REVERSIBLE but no undo capability "
            f"is available. Load or save the workflow first to enable undo.",
        )

    if risk == RiskLevel.EXECUTION:
        return (
            True,
            f"Tool '{tool_name}' is EXECUTION — outputs are additive. Reversibility acceptable.",
        )

    if risk >= RiskLevel.PROVISION:
        return (
            True,
            f"Tool '{tool_name}' (risk={risk.name}) — reversibility cannot "
            f"be guaranteed. Proceed with caution.",
        )

    return True, "Reversibility check passed."


# ---------------------------------------------------------------------------
# Check 5: Scope
# ---------------------------------------------------------------------------


def check_scope(
    tool_name: str,
    tool_input: dict,
) -> tuple[bool, str]:
    """Validates paths in tool_input against allowed directories.

    Scans all string values in tool_input whose keys look like path
    parameters.  Rejects paths that resolve into blocked system directories
    or contain traversal patterns.
    """
    for key, value in tool_input.items():
        if key not in _PATH_KEYS:
            continue
        if not isinstance(value, str):
            continue
        if not value:
            continue

        # Check for traversal patterns
        if ".." in value:
            return (
                False,
                f"Path traversal detected in '{key}': '{value}'. Blocked.",
            )

        # Resolve and check against blocked prefixes
        try:
            resolved = str(Path(value).resolve())
        except (OSError, ValueError) as exc:
            return False, f"Invalid path in '{key}': {exc}"

        for prefix in _BLOCKED_PREFIXES:
            if resolved.startswith(prefix):
                return (
                    False,
                    f"Path '{key}' resolves to protected system directory: "
                    f"'{resolved}'. Access denied.",
                )

    return True, "Scope check passed — all paths within allowed boundaries."
