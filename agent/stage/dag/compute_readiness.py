"""Compute readiness grade — the final go/no-go gate.

Pure function combining all upstream DAG signals into a single
``ReadinessGrade`` value.
"""

from __future__ import annotations

from .schemas import ReadinessGrade, RiskLevel


def compute_readiness(
    risk: RiskLevel,
    missing_nodes: list[str] | None = None,
) -> ReadinessGrade:
    """Derive ``ReadinessGrade`` from upstream risk and provision state.

    Args:
        risk: Already-computed risk level.
        missing_nodes: List of class_types not found in the node registry.
            ``None`` means node availability was not checked.

    Returns:
        Readiness grade enum value.
    """
    # Hard block
    if risk >= RiskLevel.BLOCKED:
        return ReadinessGrade.BLOCKED

    # Workflow errors that need manual repair
    if risk >= RiskLevel.RISKY:
        return ReadinessGrade.NEEDS_FIX

    # Missing nodes that could be installed
    if missing_nodes:
        return ReadinessGrade.NEEDS_PROVISION

    return ReadinessGrade.READY
