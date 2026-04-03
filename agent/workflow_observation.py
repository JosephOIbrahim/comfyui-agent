"""Time-sampled workflow state — Subsystem 3 data types.

Every tool call or state transition produces a ``WorkflowObservation``
snapshot keyed by a monotonic ``step_index``.  The observation captures
*what the workflow looks like right now* (``WorkflowStateBlock``),
*what just happened* (``ActionBlock``), *cumulative session dynamics*
(``DynamicsBlock``), and an optional DAG intelligence snapshot.

Design invariant: ``BASELINE_OBSERVATION`` is the zero-state.  The
observation log guarantees that ``read_previous(0)`` returns this
baseline — never ``None``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkflowPhase(IntEnum):
    """Lifecycle phase of the active workflow."""

    EMPTY = 0
    LOADED = 1
    CONFIGURED = 2
    VALIDATED = 3
    EXECUTING = 4
    COMPLETED = 5
    FAILED = 6


class WorkflowHealth(IntEnum):
    """Aggregate health of the workflow graph."""

    BROKEN = 0
    DEGRADED = 1
    NOMINAL = 2
    OPTIMAL = 3


class ProvisionStatus(IntEnum):
    """Model / node provisioning state."""

    MISSING = 0
    DOWNLOADING = 1
    AVAILABLE = 2
    CACHED = 3


# ---------------------------------------------------------------------------
# Sub-blocks
# ---------------------------------------------------------------------------


@dataclass
class WorkflowStateBlock:
    """Point-in-time workflow state."""

    phase: WorkflowPhase = WorkflowPhase.EMPTY
    health: WorkflowHealth = WorkflowHealth.NOMINAL
    provision: ProvisionStatus = ProvisionStatus.AVAILABLE
    node_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict:
        """Serializable snapshot."""
        return {
            "phase": self.phase.name,
            "health": self.health.name,
            "provision": self.provision.name,
            "node_count": self.node_count,
            "error_count": self.error_count,
        }


@dataclass
class ActionBlock:
    """What just happened — the tool call that produced this observation."""

    tool_name: str = ""
    tool_input_hash: str = ""
    action_type: str = ""

    def to_dict(self) -> dict:
        """Serializable snapshot."""
        return {
            "tool_name": self.tool_name,
            "tool_input_hash": self.tool_input_hash,
            "action_type": self.action_type,
        }


@dataclass
class DynamicsBlock:
    """Cumulative session dynamics — running counters."""

    total_mutations: int = 0
    undo_count: int = 0
    execution_count: int = 0
    validation_failures: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serializable snapshot."""
        return {
            "total_mutations": self.total_mutations,
            "undo_count": self.undo_count,
            "execution_count": self.execution_count,
            "validation_failures": self.validation_failures,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass
class WorkflowObservation:
    """Single time-sampled observation of the workflow + session state.

    ``step_index`` is monotonically increasing and is the sole temporal
    key.  ``timestamp`` is wall-clock for display only — never use it
    for ordering.
    """

    session_id: str = ""
    step_index: int = 0
    timestamp: float = field(default_factory=time.time)
    state: WorkflowStateBlock = field(default_factory=WorkflowStateBlock)
    action: ActionBlock = field(default_factory=ActionBlock)
    dynamics: DynamicsBlock = field(default_factory=DynamicsBlock)
    intelligence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializable snapshot for session persistence."""
        return {
            "session_id": self.session_id,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "intelligence": self.intelligence,
        }


# The zero-state observation. Never None. Always available.
BASELINE_OBSERVATION = WorkflowObservation()
