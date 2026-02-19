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

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
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

class IterationAccumulatorAgent(BrainAgent):
    """Tracks iteration steps for metadata embedding."""

    TOOLS = TOOLS

    def __init__(self, config: BrainConfig | None = None):
        super().__init__(config)
        self._intent_summary: str = ""
        self._steps: list[dict] = []
        self._accepted: int | None = None
        self._started_at: float | None = None
        self._lock = threading.Lock()

    def start(self, intent_summary: str) -> dict:
        """Begin a new iteration tracking cycle."""
        with self._lock:
            self._intent_summary = intent_summary
            self._steps = []
            self._accepted = None
            self._started_at = time.time()
        return {
            "status": "tracking",
            "intent_summary": intent_summary,
            "started_at": self._started_at,
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
    ) -> dict:
        """Record a single iteration step."""
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
        with self._lock:
            self._steps.append(step)
            step_count = len(self._steps)
        return {
            "status": "recorded",
            "iteration": iteration,
            "total_steps": step_count,
        }

    def finalize(self, accepted_iteration: int) -> dict:
        """Mark the accepted iteration and return full history."""
        with self._lock:
            self._accepted = accepted_iteration
            history = {
                "intent_summary": self._intent_summary,
                "iterations": list(self._steps),
                "accepted_iteration": accepted_iteration,
                "started_at": self._started_at,
                "finalized_at": time.time(),
                "total_steps": len(self._steps),
            }
        return history

    def get_steps(self) -> list[dict]:
        """Return current steps (for inspection)."""
        with self._lock:
            return list(self._steps)

    def is_tracking(self) -> bool:
        """Check if tracking is active."""
        with self._lock:
            return self._started_at is not None and self._accepted is None

    def handle(self, name: str, tool_input: dict) -> str:
        if name == "start_iteration_tracking":
            result = self.start(
                intent_summary=tool_input["intent_summary"],
            )
            return self.to_json(result)
        elif name == "record_iteration_step":
            result = self.record_step(
                iteration=tool_input["iteration"],
                step_type=tool_input["type"],
                trigger=tool_input["trigger"],
                patches=tool_input.get("patches", []),
                params=tool_input.get("params", {}),
                feedback=tool_input.get("feedback", ""),
                observation=tool_input.get("observation", ""),
            )
            return self.to_json(result)
        elif name == "finalize_iterations":
            result = self.finalize(
                accepted_iteration=tool_input["accepted_iteration"],
            )
            return self.to_json(result)
        else:
            return self.to_json({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Module-level singleton (lazy, for backward compat with tools registry)
# ---------------------------------------------------------------------------

_singleton: IterationAccumulatorAgent | None = None
_singleton_lock = threading.Lock()


def _get_agent() -> IterationAccumulatorAgent:
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = IterationAccumulatorAgent()
        return _singleton


def handle(name: str, tool_input: dict) -> str:
    """Module-level dispatch."""
    return _get_agent().handle(name, tool_input)
