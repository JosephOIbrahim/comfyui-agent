"""Brain layer for the ComfyUI Comfy Cozy Agent.

Adds higher-order capabilities on top of the intelligence layers:
  Vision    — see and critique generated images
  Planner   — decompose goals into tracked sub-tasks
  Memory    — learn from outcomes, recommend what works
  Orchestrator — coordinate parallel sub-tasks
  Optimizer — GPU-aware performance engineering
  Demo      — guided walkthroughs for streams/podcasts
  IterativeRefine — autonomous quality iteration loop
  IntentCollector — capture artistic intent for metadata embedding
  IterationAccumulator — track refinement journey across iterations

SDK classes (for standalone/testing use):
  BrainConfig  — dependency injection container
  BrainAgent   — base class for all brain agents
  VisionAgent, PlannerAgent, MemoryAgent, OrchestratorAgent, OptimizerAgent,
  DemoAgent, IterativeRefineAgent, IntentCollectorAgent, IterationAccumulatorAgent
"""

import importlib
import logging

from ._sdk import BrainAgent, BrainConfig  # noqa: F401
from ..errors import error_json

log = logging.getLogger(__name__)

# Register brain submodules individually — each is isolated so a single broken
# module (missing pip dep, syntax error) does NOT crash the entire brain layer.
_BRAIN_SUBMODULES = [
    "vision", "planner", "memory", "orchestrator", "optimizer",
    "demo", "iterative_refine", "intent_collector", "iteration_accumulator",
]
for _bmod in _BRAIN_SUBMODULES:
    try:
        importlib.import_module(f".{_bmod}", package=__name__)
    except Exception as _e:
        log.warning("Brain submodule %r failed to register: %s", _bmod, _e)

# Re-export agent classes (guarded — a submodule may have failed above)
try:
    from .demo import DemoAgent  # noqa: F401
except ImportError:
    DemoAgent = None  # type: ignore[assignment,misc]
try:
    from .intent_collector import IntentCollectorAgent  # noqa: F401
except ImportError:
    IntentCollectorAgent = None  # type: ignore[assignment,misc]
try:
    from .iterative_refine import IterativeRefineAgent  # noqa: F401
except ImportError:
    IterativeRefineAgent = None  # type: ignore[assignment,misc]
try:
    from .iteration_accumulator import IterationAccumulatorAgent  # noqa: F401
except ImportError:
    IterationAccumulatorAgent = None  # type: ignore[assignment,misc]
try:
    from .memory import MemoryAgent  # noqa: F401
except ImportError:
    MemoryAgent = None  # type: ignore[assignment,misc]
try:
    from .optimizer import OptimizerAgent  # noqa: F401
except ImportError:
    OptimizerAgent = None  # type: ignore[assignment,misc]
try:
    from .orchestrator import OrchestratorAgent  # noqa: F401
except ImportError:
    OrchestratorAgent = None  # type: ignore[assignment,misc]
try:
    from .planner import PlannerAgent  # noqa: F401
except ImportError:
    PlannerAgent = None  # type: ignore[assignment,misc]
try:
    from .vision import VisionAgent  # noqa: F401
except ImportError:
    VisionAgent = None  # type: ignore[assignment,misc]

ALL_BRAIN_TOOLS = BrainAgent.get_all_tools()


def handle(name: str, tool_input: dict, *, progress=None) -> str:  # noqa: ARG001
    """Dispatch a brain tool call to the right handler.

    The progress= kwarg is accepted for API compatibility with the tool
    dispatch layer (agent/tools/__init__.py) but is not currently threaded
    into BrainAgent.dispatch() — brain sub-tools report progress via their
    own mechanisms. Accepting it here prevents the TypeError fallback path
    in the outer dispatcher.
    """
    try:
        return BrainAgent.dispatch(name, tool_input)
    except Exception as e:
        log.warning("Brain tool %s failed: %s", name, e, exc_info=True)
        return error_json(
            f"Internal error in {name}: {e}",
            hint="Try again, or check the ComfyUI console for details.",
        )
