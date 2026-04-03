"""Pre-dispatch gate — Subsystem 4.

Evaluates every tool call against five checks before dispatch.
"""

from __future__ import annotations

from .pre_dispatch import GateDecision, GateResult, pre_dispatch_check
from .risk_levels import RiskLevel, get_risk_level

__all__ = [
    "GateDecision",
    "GateResult",
    "RiskLevel",
    "get_risk_level",
    "pre_dispatch_check",
]
