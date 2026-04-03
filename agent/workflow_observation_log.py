"""Thread-safe observation log for time-sampled workflow state.

Stores ``WorkflowObservation`` instances keyed by monotonic
``step_index``.  Provides authoring, reading, and history access
with a hard invariant: ``read_previous(0)`` always returns
``BASELINE_OBSERVATION``, never ``None``.

Thread safety is provided by ``threading.RLock`` — the log may be
authored from tool handlers running on different threads (e.g. the
MCP server's asyncio executor or the orchestrator's subtask threads).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .workflow_observation import (
    BASELINE_OBSERVATION,
    WorkflowObservation,
)


class WorkflowObservationLog:
    """Thread-safe observation log keyed by step_index.

    Invariant: ``read_previous(0)`` returns ``BASELINE_OBSERVATION``.
    The log never returns ``None`` from ``read_previous``.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._lock = threading.RLock()
        self._log: list[WorkflowObservation] = []
        self._step: int = 0
        self._created_at: float = time.time()

    # -----------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------

    def author(self, observation: WorkflowObservation) -> int:
        """Append an observation and return its assigned step_index.

        The observation's ``step_index`` and ``session_id`` are
        overwritten to ensure monotonicity and ownership.

        Args:
            observation: The observation to record.

        Returns:
            The step_index assigned to this observation.
        """
        with self._lock:
            idx = self._step
            self._step += 1
            # Enforce ownership and ordering
            observation.step_index = idx
            observation.session_id = self._session_id
            self._log.append(observation)
            return idx

    # -----------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------

    def read(self, step_index: int) -> WorkflowObservation | None:
        """Read the observation at exactly ``step_index``.

        Returns ``None`` if no observation exists at that index.
        """
        with self._lock:
            if 0 <= step_index < len(self._log):
                return self._log[step_index]
            return None

    def read_previous(self, step_index: int) -> WorkflowObservation:
        """Read the observation just before ``step_index``.

        If ``step_index`` is 0 or no prior observation exists, returns
        ``BASELINE_OBSERVATION``.  **Never returns None.**

        Args:
            step_index: The current step — returns the one before it.

        Returns:
            The previous observation, or ``BASELINE_OBSERVATION``.
        """
        with self._lock:
            prev = step_index - 1
            if prev >= 0 and prev < len(self._log):
                return self._log[prev]
            return BASELINE_OBSERVATION

    def current_step(self) -> int:
        """Return the next step_index that will be assigned."""
        with self._lock:
            return self._step

    # -----------------------------------------------------------------
    # History
    # -----------------------------------------------------------------

    def history(self, last_n: int = 10) -> list[WorkflowObservation]:
        """Return the most recent *last_n* observations (oldest first).

        Args:
            last_n: Maximum number of observations to return.

        Returns:
            List of observations, oldest first.
        """
        with self._lock:
            if last_n <= 0:
                return []
            start = max(0, len(self._log) - last_n)
            return list(self._log[start:])

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable summary of the log.

        Includes the latest observation, total steps, and session timing.
        Does NOT serialize the entire history — use ``history()`` for
        bulk export.
        """
        with self._lock:
            latest = self._log[-1].to_dict() if self._log else None
            elapsed = time.time() - self._created_at
            return {
                "session_id": self._session_id,
                "total_steps": self._step,
                "elapsed_seconds": round(elapsed, 2),
                "latest": latest,
            }

    def __len__(self) -> int:
        """Number of observations in the log."""
        with self._lock:
            return len(self._log)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"WorkflowObservationLog("
                f"session={self._session_id!r}, "
                f"steps={self._step})"
            )
