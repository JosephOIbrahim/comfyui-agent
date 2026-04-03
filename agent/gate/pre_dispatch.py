"""Pre-dispatch gate — default-deny check pipeline for all tool calls.

All five checks must pass for ALLOW.  Special handling:
  - READ_ONLY (level 0): bypasses all checks — zero latency overhead.
  - DESTRUCTIVE (level 4): always returns LOCKED.
  - PROVISION (level 3): returns ESCALATE (needs user confirmation).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .risk_levels import RiskLevel, get_risk_level
from .checks import (
    check_system_health,
    check_consent,
    check_constitution,
    check_reversibility,
    check_scope,
)


class GateDecision(Enum):
    """Possible gate outcomes."""

    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"
    LOCKED = "locked"


@dataclass(frozen=True)
class GateResult:
    """Outcome of a pre-dispatch gate evaluation."""

    decision: GateDecision
    checks: dict[str, bool]
    risk_level: RiskLevel
    reason: str = ""


def pre_dispatch_check(
    tool_name: str,
    tool_input: dict,
    *,
    breaker_state: str = "closed",
    session_active: bool = True,
    validated: bool = False,
    has_undo: bool = False,
    action_history: list[str] | None = None,
    **constitution_kwargs: object,
) -> GateResult:
    """Default-deny gate.  All 5 checks must pass for ALLOW.

    Args:
        tool_name: Name of the tool being dispatched.
        tool_input: Dict of arguments the tool will receive.
        breaker_state: Current circuit breaker state ("closed"/"open"/"half_open").
        session_active: Whether a workflow session is active.
        validated: Whether validate_before_execute has been called since
                   the last mutation.
        has_undo: Whether undo capability is available (workflow loaded).
        action_history: List of previously executed tool names this session.
        **constitution_kwargs: Forwarded to check_constitution (agent_name,
                               allowed_tools, verified_since_mutation, etc.)

    Returns:
        GateResult with decision, per-check outcomes, and reason.
    """
    history = action_history if action_history is not None else []
    risk = get_risk_level(tool_name)

    # --- Fast paths for extreme risk levels ---

    if risk == RiskLevel.READ_ONLY:
        return GateResult(
            decision=GateDecision.ALLOW,
            checks={
                "system_health": True,
                "consent": True,
                "constitution": True,
                "reversibility": True,
                "scope": True,
            },
            risk_level=risk,
            reason="READ_ONLY tool — all checks bypassed.",
        )

    if risk == RiskLevel.DESTRUCTIVE:
        return GateResult(
            decision=GateDecision.LOCKED,
            checks={
                "system_health": False,
                "consent": False,
                "constitution": False,
                "reversibility": False,
                "scope": False,
            },
            risk_level=risk,
            reason=(
                f"Tool '{tool_name}' is DESTRUCTIVE (level 4). "
                f"Requires explicit human approval. Gate returns LOCKED."
            ),
        )

    # --- Run all 5 checks ---

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    # 1. System health
    passed, reason = check_system_health(tool_name, risk, breaker_state)
    checks["system_health"] = passed
    if not passed:
        reasons.append(reason)

    # 2. Consent
    passed, reason = check_consent(tool_name, risk, session_active, validated)
    checks["consent"] = passed
    if not passed:
        reasons.append(reason)
        # Detect ESCALATE vs DENY for provision tools
        if risk == RiskLevel.PROVISION:
            # Run remaining checks for completeness but return ESCALATE
            checks.setdefault("constitution", True)
            checks.setdefault("reversibility", True)
            # Still run scope check
            scope_passed, scope_reason = check_scope(tool_name, tool_input)
            checks["scope"] = scope_passed
            if not scope_passed:
                reasons.append(scope_reason)
            return GateResult(
                decision=GateDecision.ESCALATE,
                checks=checks,
                risk_level=risk,
                reason="; ".join(reasons) if reasons else "Escalation required.",
            )

    # 3. Constitution
    passed, reason = check_constitution(
        history,
        tool_name,
        **constitution_kwargs,
    )
    checks["constitution"] = passed
    if not passed:
        reasons.append(reason)

    # 4. Reversibility
    passed, reason = check_reversibility(tool_name, risk, has_undo)
    checks["reversibility"] = passed
    if not passed:
        reasons.append(reason)

    # 5. Scope
    passed, reason = check_scope(tool_name, tool_input)
    checks["scope"] = passed
    if not passed:
        reasons.append(reason)

    # --- Decide ---

    all_ok = all(checks.values())

    if all_ok:
        return GateResult(
            decision=GateDecision.ALLOW,
            checks=checks,
            risk_level=risk,
            reason="All checks passed.",
        )

    return GateResult(
        decision=GateDecision.DENY,
        checks=checks,
        risk_level=risk,
        reason="; ".join(reasons),
    )
