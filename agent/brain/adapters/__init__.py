"""Inter-module adapter registry for brain-to-brain data translation.

Adapters are pure functions that convert one module's output format into
another module's expected input format.  No side effects, no imports of
brain agent classes — just dict-in, dict-out transformations.

Usage:
    from agent.brain.adapters import adapt
    outcome = adapt("vision", "memory", vision_result)
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

# (source_module, target_module) -> adapter function
ADAPTER_REGISTRY: dict[tuple[str, str], Callable[[dict], dict]] = {}


def register_adapter(
    source: str,
    target: str,
    adapter_fn: Callable[[dict], dict],
) -> None:
    """Register an adapter for a given source -> target pair."""
    key = (source, target)
    if key in ADAPTER_REGISTRY:
        log.warning(
            "Overwriting adapter %s->%s (was %s, now %s)",
            source, target,
            ADAPTER_REGISTRY[key].__name__,
            adapter_fn.__name__,
        )
    ADAPTER_REGISTRY[key] = adapter_fn


def get_adapter(
    source: str, target: str,
) -> Callable[[dict], dict] | None:
    """Look up an adapter for a source -> target pair.  Returns None if absent."""
    return ADAPTER_REGISTRY.get((source, target))


def adapt(source: str, target: str, data: dict) -> dict:
    """Transform *data* from *source* format to *target* format.

    Raises ``KeyError`` if no adapter is registered for the pair.
    """
    key = (source, target)
    fn = ADAPTER_REGISTRY.get(key)
    if fn is None:
        raise KeyError(
            f"No adapter registered for {source}->{target}. "
            f"Available: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    return fn(data)


# ------------------------------------------------------------------
# Auto-register built-in adapters on import
# ------------------------------------------------------------------
def _register_builtins() -> None:
    from .vision_memory import (
        vision_to_outcome,
        patterns_to_vision_context,
    )
    from .planner_orchestrator import (
        plan_step_to_subtask,
        subtask_result_to_completion,
    )
    from .intent_verify import (
        intent_to_criteria,
        verify_against_intent,
    )

    register_adapter("vision", "memory", vision_to_outcome)
    register_adapter("memory", "vision", patterns_to_vision_context)
    register_adapter("planner", "orchestrator", plan_step_to_subtask)
    register_adapter("orchestrator", "planner", subtask_result_to_completion)
    register_adapter("intent", "verify", intent_to_criteria)
    def _verify_intent_wrapper(data: dict) -> dict:
        """Wrap 2-arg verify_against_intent for single-dict adapt() API.

        Expects ``{"verify_result": dict, "criteria": dict}``.
        """
        return verify_against_intent(
            data.get("verify_result", {}),
            data.get("criteria", {}),
        )

    register_adapter("verify", "intent", _verify_intent_wrapper)


_register_builtins()
