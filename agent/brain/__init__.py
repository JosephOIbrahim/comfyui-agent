"""Brain layer for the ComfyUI SUPER DUPER Agent.

Adds higher-order capabilities on top of the intelligence layers:
  Vision    — see and critique generated images
  Planner   — decompose goals into tracked sub-tasks
  Memory    — learn from outcomes, recommend what works
  Orchestrator — coordinate parallel sub-tasks
  Optimizer — GPU-aware performance engineering
  Demo      — guided walkthroughs for streams/podcasts
  IterativeRefine — autonomous quality iteration loop

Each module exports:
  TOOLS: list[dict]    -- Anthropic tool schemas
  handle(name, input)  -- Execute a tool call, return result string

SDK classes (for standalone/testing use):
  BrainConfig  — dependency injection container
  BrainAgent   — base class for all brain agents
  VisionAgent, PlannerAgent, MemoryAgent, OrchestratorAgent, OptimizerAgent,
  DemoAgent, IterativeRefineAgent
"""

import logging

from . import vision, planner, memory, orchestrator, optimizer, demo, iterative_refine
from ._sdk import BrainAgent as BrainAgent, BrainConfig as BrainConfig
from .demo import DemoAgent as DemoAgent
from .iterative_refine import IterativeRefineAgent as IterativeRefineAgent
from .memory import MemoryAgent as MemoryAgent
from .optimizer import OptimizerAgent as OptimizerAgent
from .orchestrator import OrchestratorAgent as OrchestratorAgent
from .planner import PlannerAgent as PlannerAgent
from .vision import VisionAgent as VisionAgent

log = logging.getLogger(__name__)

_MODULES = (vision, planner, memory, orchestrator, optimizer, demo, iterative_refine)

# Collect all brain tool schemas
ALL_BRAIN_TOOLS: list[dict] = []
for _mod in _MODULES:
    ALL_BRAIN_TOOLS.extend(_mod.TOOLS)

# Map tool name -> handler module
_HANDLERS = {}
for _mod in _MODULES:
    for _tool in _mod.TOOLS:
        _HANDLERS[_tool["name"]] = _mod


def handle(name: str, tool_input: dict) -> str:
    """Dispatch a brain tool call to the right handler."""
    mod = _HANDLERS.get(name)
    if mod is None:
        log.warning("Unknown brain tool called: %s", name)
        return f"Unknown brain tool: {name}"
    try:
        return mod.handle(name, tool_input)
    except Exception as e:
        log.error("Unhandled error in brain tool %s", name, exc_info=True)
        from ..tools._util import to_json
        return to_json({"error": f"Internal error in {name}: {type(e).__name__}: {e}"})
