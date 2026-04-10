"""Iteration Accumulator — tracks the refinement journey.

Records each iteration step (initial, refinement, variation, rollback)
with its trigger, patches applied, parameter snapshot, user feedback,
and agent observations. The accumulated history is embedded into the
final output image's metadata.

Thread-safe module-level state (matches orchestrator/demo pattern).
"""

import logging
import threading
import time

from ._sdk import BrainAgent, BrainConfig

log = logging.getLogger(__name__)

_MAX_STEPS = 200

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "name": "start_iteration_tracking",
        "description": (
            "Begin tracking iterations for a new generation session. "
            "Call this when starting a new workflow execution cycle. "
            "Clears any previous iteration state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_summary": {
                    "type": "string",
                    "description": "Brief summary of the artistic goal for this cycle.",
                },
            },
            "required": ["intent_summary"],
        },
    },
    {
        "name": "record_iteration_step",
        "description": (
            "Record a single iteration step in the refinement journey. "
            "Call after each workflow modification or re-execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iteration": {
                    "type": "integer",
                    "description": "Iteration number (1-based).",
                },
                "type": {
                    "type": "string",
                    "enum": ["initial", "refinement", "variation", "rollback"],
                    "description": "Type of iteration step.",
                },
                "trigger": {
                    "type": "string",
                    "description": (
                        "What triggered this iteration (e.g. 'user asked for warmer tones', "
                        "'agent suggested CFG reduction')."
                    ),
                },
                "patches": {
                    "type": "array",
                    "description": "RFC6902 patches applied in this step (can be empty for initial).",
                },
                "params": {
                    "type": "object",
                    "description": "Key parameter snapshot after this step.",
                },
                "feedback": {
                    "type": "string",
                    "description": "User feedback on this iteration's result.",
                },
                "observation": {
                    "type": "string",
                    "description": "Agent's observation about the result.",
                },
            },
            "required": ["iteration", "type", "trigger"],
        },
    },
    {
        "name": "finalize_iterations",
        "description": (
            "Mark the accepted iteration and return the full history. "
            "Call when the artist approves a result. The returned history "
            "is ready for embedding into image metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "accepted_iteration": {
                    "type": "integer",
                    "description": "The iteration number the artist accepted.",
                },
            },
            "required": ["accepted_iteration"],
        },
    },
]


# ---------------------------------------------------------------------------
# SDK Agent class
# ---------------------------------------------------------------------------

_MAX_SESSIONS = 100  # Cycle 52: FIFO cap on per-session state dicts


class IterationAccumulatorAgent(BrainAgent):
    """Tracks iteration steps for metadata embedding — per-session isolated.

    Cycle 52: Refactored from singleton-instance state to per-session dicts.
    Previously, a single shared instance caused state collisions between
    concurrent MCP connections. Now each session_id key has independent state.
    """

    TOOLS = _TOOLS

    def __init__(self, config: BrainConfig | None = None):
        super().__init__(config)
        # Per-session state: session_id → {intent_summary, steps, accepted, started_at}
        self._sessions: dict[str, dict] = {}
        self._sessions_mutex = threading.Lock()  # guards _sessions and _session_locks dicts
        self._session_locks: dict[str, threading.Lock] = {}

    def _get_session_lock(self, session: str) -> threading.Lock:
        """Return (creating if needed) a per-session lock. FIFO eviction at cap."""
        with self._sessions_mutex:
            if session not in self._session_locks:
                if len(self._session_locks) >= _MAX_SESSIONS:
                    oldest = next(iter(self._session_locks))
                    del self._session_locks[oldest]
                    self._sessions.pop(oldest, None)
                self._session_locks[session] = threading.Lock()
            return self._session_locks[session]

    def _get_state(self, session: str) -> dict:
        """Return (creating if needed) the mutable state dict for session.

        MUST be called while holding the session lock.
        """
        if session not in self._sessions:
            self._sessions[session] = {
                "intent_summary": "",
                "steps": [],
                "accepted": None,
                "started_at": None,
            }
        return self._sessions[session]

    def start(self, intent_summary: str, session: str = "default") -> dict:
        """Begin a new iteration tracking cycle for session."""
        lock = self._get_session_lock(session)
        with lock:
            state = self._get_state(session)
            state["intent_summary"] = intent_summary
            state["steps"] = []
            state["accepted"] = None
            state["started_at"] = time.time()
            started_at = state["started_at"]
        return {
            "status": "tracking",
            "intent_summary": intent_summary,
            "started_at": started_at,
            "session": session,
        }

    def record_step(
        self,
        iteration: int,
        step_type: str,
        trigger: str,
        patches: list | None = None,
        params: dict | None = None,
        feedback: str = "",
        observation: str = "",
        session: str = "default",
    ) -> dict:
        """Record a single iteration step for session."""
        lock = self._get_session_lock(session)
        with lock:
            state = self._get_state(session)
            if state["started_at"] is None:
                return {"error": "Call start_iteration_tracking before record_iteration_step."}
            step = {
                "iteration": iteration,
                "type": step_type,
                "trigger": trigger,
                "patches": patches or [],
                "params": params or {},
                "feedback": feedback,
                "observation": observation,
                "recorded_at": time.time(),
            }
            state["steps"].append(step)
            if len(state["steps"]) > _MAX_STEPS:
                log.warning(
                    "Iteration steps exceeded %d limit for session %r, trimming oldest",
                    _MAX_STEPS, session,
                )
                state["steps"] = state["steps"][-_MAX_STEPS:]
            step_count = len(state["steps"])
        return {
            "status": "recorded",
            "iteration": iteration,
            "total_steps": step_count,
            "session": session,
        }

    def finalize(self, accepted_iteration: int, session: str = "default") -> dict:
        """Mark the accepted iteration and return full history for session."""
        lock = self._get_session_lock(session)
        with lock:
            state = self._get_state(session)
            if state["started_at"] is None:
                return {"error": "Call start_iteration_tracking before finalize_iterations."}
            if not state["steps"]:
                return {"error": "No iteration steps recorded. Call record_iteration_step first."}
            state["accepted"] = accepted_iteration
            history = {
                "intent_summary": state["intent_summary"],
                "iterations": list(state["steps"]),
                "accepted_iteration": accepted_iteration,
                "started_at": state["started_at"],
                "finalized_at": time.time(),
                "total_steps": len(state["steps"]),
                "session": session,
            }
        return history

    def get_steps(self, session: str = "default") -> list[dict]:
        """Return current steps for session (for inspection)."""
        lock = self._get_session_lock(session)
        with lock:
            return list(self._get_state(session)["steps"])

    def is_tracking(self, session: str = "default") -> bool:
        """Check if tracking is active for session."""
        lock = self._get_session_lock(session)
        with lock:
            state = self._get_state(session)
            return state["started_at"] is not None and state["accepted"] is None

    def handle(self, name: str, tool_input: dict) -> str:
        session = tool_input.get("session", "default")  # Cycle 52: per-session isolation
        if not isinstance(session, str) or not session:
            session = "default"

        if name == "start_iteration_tracking":
            intent_summary = tool_input.get("intent_summary")  # Cycle 47: guard required field
            if not intent_summary or not isinstance(intent_summary, str):
                return self.to_json({"error": "intent_summary is required and must be a non-empty string."})
            result = self.start(intent_summary=intent_summary, session=session)
            return self.to_json(result)
        elif name == "record_iteration_step":
            iteration = tool_input.get("iteration")  # Cycle 47: guard required fields
            step_type = tool_input.get("type")
            trigger = tool_input.get("trigger")
            if iteration is None:
                return self.to_json({"error": "iteration is required."})
            if not step_type or not isinstance(step_type, str):
                return self.to_json({"error": "type is required and must be a non-empty string."})
            if not trigger or not isinstance(trigger, str):
                return self.to_json({"error": "trigger is required and must be a non-empty string."})
            result = self.record_step(
                iteration=iteration,
                step_type=step_type,
                trigger=trigger,
                patches=tool_input.get("patches", []),
                params=tool_input.get("params", {}),
                feedback=tool_input.get("feedback", ""),
                observation=tool_input.get("observation", ""),
                session=session,
            )
            return self.to_json(result)
        elif name == "finalize_iterations":
            accepted_iteration = tool_input.get("accepted_iteration")  # Cycle 47: guard required field
            if accepted_iteration is None:
                return self.to_json({"error": "accepted_iteration is required."})
            result = self.finalize(accepted_iteration=accepted_iteration, session=session)
            return self.to_json(result)
        else:
            return self.to_json({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Backward-compat re-exports (consumed by tests outside test_brain_*.py)
# ---------------------------------------------------------------------------

TOOLS = _TOOLS


def handle(name: str, tool_input: dict) -> str:
    """Module-level dispatch shim — routes to the registered instance."""
    BrainAgent._register_all()
    agent = BrainAgent._registry.get(name)
    if agent is not None:
        return agent.handle(name, tool_input)
    import json as _json  # Cycle 43: return error directly; avoid circular import via brain package
    return _json.dumps({"error": f"Unknown tool: {name}"}, sort_keys=True)
