"""Planner module — goal decomposition and progress tracking.

Turns high-level goals ("build me a Flux portrait pipeline with ControlNet")
into tracked sequences of sub-tasks. Uses template-based decomposition for
speed and determinism. State persists to disk for cross-session continuity.
"""

import json
import logging
import time
from pathlib import Path

from ._protocol import make_id
from ._sdk import BrainAgent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Goal decomposition templates
# ---------------------------------------------------------------------------

_GOAL_PATTERNS: dict[str, dict] = {
    # --- Specific patterns first (checked before generic "build"/"create") ---
    "chain_workflows": {
        "triggers": [
            "chain", "multi-stage", "multistage", "then feed",
            "pass output", "connect workflows", "stage", "sequence",
            "image to 3d", "txt2img then", "generate then",
            "3d pipeline", "audio pipeline", "multi-modal",
        ],
        "steps": [
            {"id": "decompose", "action": "Decompose goal into pipeline stages",
             "tools": ["list_workflow_templates", "discover"]},
            {"id": "define_pipeline",
             "action": "Define the pipeline: stages, input/output mappings",
             "tools": ["create_pipeline"]},
            {"id": "verify_stages",
             "action": "Verify each stage's workflow is valid",
             "tools": ["validate_before_execute"]},
            {"id": "run_pipeline",
             "action": "Execute the full pipeline",
             "tools": ["run_pipeline"]},
            {"id": "verify_outputs",
             "action": "Verify pipeline outputs and record outcomes",
             "tools": ["verify_execution", "analyze_image"]},
            {"id": "iterate",
             "action": "Review results and iterate if needed",
             "tools": ["get_pipeline_status", "compare_outputs"]},
        ],
    },
    "generate_3d": {
        "triggers": [
            "3d model", "3d mesh", "generate 3d", "hunyuan3d", "hunyuan 3d",
            "text to 3d", "image to 3d", "3d asset", "3d object",
            "gaussian splat", "3dgs", "point cloud",
        ],
        "steps": [
            {"id": "find_3d_nodes",
             "action": "Find available 3D generation nodes and models",
             "tools": ["discover", "list_custom_nodes", "list_models"]},
            {"id": "plan_stages",
             "action": "Plan the 3D generation pipeline stages",
             "tools": ["create_pipeline"]},
            {"id": "configure",
             "action": "Configure each stage (conditioning, resolution, steps)",
             "tools": ["set_input"]},
            {"id": "run_pipeline",
             "action": "Execute the 3D generation pipeline",
             "tools": ["run_pipeline"]},
            {"id": "verify_output",
             "action": "Verify 3D output files exist and are valid",
             "tools": ["get_pipeline_status", "get_output_path"]},
        ],
    },
    "generate_audio": {
        "triggers": [
            "tts", "text to speech", "narration", "voice",
            "generate audio", "audio generation", "speech",
            "cosyvoice", "bark", "read aloud", "narrator",
        ],
        "steps": [
            {"id": "find_audio_nodes",
             "action": "Find available TTS/audio nodes and models",
             "tools": ["discover", "list_custom_nodes", "list_models"]},
            {"id": "select_template",
             "action": "Select or build an audio generation workflow",
             "tools": ["list_workflow_templates", "get_workflow_template"]},
            {"id": "configure",
             "action": "Configure TTS parameters (text, voice, speed)",
             "tools": ["set_input"]},
            {"id": "execute",
             "action": "Execute the audio generation workflow",
             "tools": ["execute_with_progress"]},
            {"id": "verify_output",
             "action": "Verify audio output exists",
             "tools": ["verify_execution", "get_output_path"]},
        ],
    },
    # --- Generic patterns (checked after specific ones) ---
    "build_workflow": {
        "triggers": ["build", "create", "make", "set up", "new workflow", "from scratch"],
        "steps": [
            {"id": "identify_model", "action": "Identify base model and checkpoint",
             "tools": ["list_models", "discover", "get_models_summary"]},
            {"id": "select_template", "action": "Select or load a starter template",
             "tools": ["list_workflow_templates", "get_workflow_template"]},
            {"id": "configure_base", "action": "Configure base parameters (resolution, steps, CFG, sampler)",
             "tools": ["set_input"]},
            {"id": "add_specializations", "action": "Add specialized nodes (ControlNet, LoRA, upscale, etc.)",
             "tools": ["add_node", "connect_nodes", "discover"]},
            {"id": "validate", "action": "Pre-flight validation",
             "tools": ["validate_before_execute"]},
            {"id": "test_execute", "action": "Execute with test seed",
             "tools": ["execute_workflow"]},
            {"id": "evaluate", "action": "Evaluate output quality",
             "tools": ["analyze_image"]},
            {"id": "iterate", "action": "Iterate or finalize",
             "tools": ["set_input", "compare_outputs"]},
        ],
    },
    "optimize_workflow": {
        "triggers": ["optimize", "speed up", "faster", "performance", "tensorrt", "trt"],
        "steps": [
            {"id": "profile", "action": "Profile current workflow performance",
             "tools": ["profile_workflow", "get_system_stats"]},
            {"id": "identify_bottlenecks", "action": "Identify bottlenecks and optimization targets",
             "tools": ["suggest_optimizations"]},
            {"id": "apply_optimization", "action": "Apply top-priority optimization",
             "tools": ["apply_optimization", "apply_workflow_patch"]},
            {"id": "benchmark", "action": "Benchmark: execute and compare render times",
             "tools": ["execute_workflow", "compare_outputs"]},
            {"id": "iterate", "action": "Apply next optimization or finalize",
             "tools": ["suggest_optimizations", "apply_optimization"]},
        ],
    },
    "debug_workflow": {
        "triggers": ["debug", "fix", "broken", "error", "not working", "failed"],
        "steps": [
            {"id": "reproduce", "action": "Reproduce the issue",
             "tools": ["validate_before_execute", "execute_workflow"]},
            {"id": "isolate", "action": "Isolate the failing node or connection",
             "tools": ["validate_workflow", "get_node_info"]},
            {"id": "diagnose", "action": "Check inputs, outputs, and connections",
             "tools": ["get_editable_fields", "load_workflow"]},
            {"id": "fix", "action": "Apply the fix",
             "tools": ["set_input", "connect_nodes", "apply_workflow_patch"]},
            {"id": "verify", "action": "Verify the fix works",
             "tools": ["validate_before_execute", "execute_workflow"]},
        ],
    },
    "swap_model": {
        "triggers": ["swap", "switch", "replace", "upgrade", "change model", "try another"],
        "steps": [
            {"id": "identify_current", "action": "Identify the current model in the workflow",
             "tools": ["load_workflow", "get_editable_fields"]},
            {"id": "find_alternatives", "action": "Search for compatible alternatives",
             "tools": ["list_models", "discover"]},
            {"id": "check_compatibility", "action": "Verify the new model is compatible",
             "tools": ["get_node_info", "validate_before_execute"]},
            {"id": "apply_swap", "action": "Swap the model in the workflow",
             "tools": ["set_input"]},
            {"id": "compare", "action": "Same-seed comparison with old vs new model",
             "tools": ["execute_workflow", "compare_outputs"]},
        ],
    },
    "add_controlnet": {
        "triggers": ["controlnet", "control net", "depth", "canny", "openpose", "guided"],
        "steps": [
            {"id": "find_nodes", "action": "Find required ControlNet nodes",
             "tools": ["discover", "find_missing_nodes"]},
            {"id": "find_model", "action": "Find appropriate ControlNet model",
             "tools": ["list_models", "discover"]},
            {"id": "add_nodes", "action": "Add ControlNet loader and apply nodes",
             "tools": ["add_node", "connect_nodes"]},
            {"id": "configure", "action": "Set ControlNet strength and parameters",
             "tools": ["set_input"]},
            {"id": "validate_run", "action": "Validate and test run",
             "tools": ["validate_before_execute", "execute_workflow"]},
            {"id": "evaluate", "action": "Evaluate ControlNet effect",
             "tools": ["analyze_image"]},
        ],
    },
    "explore_ecosystem": {
        "triggers": ["what's new", "anything new", "recommend", "discover", "trending"],
        "steps": [
            {"id": "scan_installed", "action": "Scan current installation",
             "tools": ["list_models", "list_custom_nodes", "get_models_summary"]},
            {"id": "search_new", "action": "Search for new/relevant options",
             "tools": ["discover"]},
            {"id": "evaluate_fit", "action": "Evaluate relevance to current workflow",
             "tools": ["get_learned_patterns", "get_recommendations"]},
            {"id": "present", "action": "Present findings with recommendations",
             "tools": []},
        ],
    },
}

# Generic fallback
_GENERIC_STEPS = [
    {"id": "understand", "action": "Understand the current state", "tools": ["load_workflow"]},
    {"id": "modify", "action": "Make the requested changes", "tools": []},
    {"id": "validate", "action": "Validate the changes", "tools": ["validate_before_execute"]},
    {"id": "execute", "action": "Execute and verify", "tools": ["execute_workflow"]},
]


# ---------------------------------------------------------------------------
# PlannerAgent class
# ---------------------------------------------------------------------------

class PlannerAgent(BrainAgent):
    """Goal decomposition and progress tracking."""

    GOAL_PATTERNS = _GOAL_PATTERNS
    GENERIC_STEPS = _GENERIC_STEPS

    TOOLS: list[dict] = [
        {
            "name": "plan_goal",
            "description": (
                "Decompose a high-level goal into a tracked sequence of sub-tasks. "
                "Uses pattern matching to select the right plan template. "
                "Persists the plan for cross-session continuity."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The high-level goal to decompose (e.g., 'Build a Flux portrait pipeline with ControlNet depth').",
                    },
                    "session": {
                        "type": "string",
                        "description": "Session name for persistence. Defaults to 'default'.",
                    },
                },
                "required": ["goal"],
            },
        },
        {
            "name": "get_plan",
            "description": (
                "Get the current plan with status of each step. "
                "Shows which steps are pending, active, done, or failed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Session name. Defaults to 'default'.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "complete_step",
            "description": (
                "Mark a plan step as completed and record what was accomplished. "
                "Automatically advances to the next step."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "string",
                        "description": "The ID of the step to mark complete.",
                    },
                    "result": {
                        "type": "string",
                        "description": "What was accomplished in this step.",
                    },
                    "session": {
                        "type": "string",
                        "description": "Session name. Defaults to 'default'.",
                    },
                },
                "required": ["step_id", "result"],
            },
        },
        {
            "name": "replan",
            "description": (
                "Revise the remaining steps of a plan without losing completed progress. "
                "Use when context changes or a step fails and needs a different approach."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why we're replanning (e.g., 'ControlNet node not available, switching to IP-Adapter').",
                    },
                    "new_remaining_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "action": {"type": "string"},
                            },
                            "required": ["id", "action"],
                        },
                        "description": "New steps to replace remaining incomplete steps.",
                    },
                    "session": {
                        "type": "string",
                        "description": "Session name. Defaults to 'default'.",
                    },
                },
                "required": ["reason"],
            },
        },
    ]

    # --- State management ---

    def _goals_path(self, session: str) -> Path:
        """Path to the goals file for a session."""
        self.cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
        return self.cfg.sessions_dir / f"{session}_goals.json"

    def _load_plan(self, session: str) -> dict | None:
        """Load a plan from disk."""
        path = self._goals_path(session)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("Failed to load plan for session %s: %s", session, e)
            return None

    def _save_plan(self, session: str, plan: dict) -> None:
        """Save a plan to disk."""
        path = self._goals_path(session)
        path.write_text(
            json.dumps(plan, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # --- Decomposition ---

    def _match_pattern(self, goal: str) -> tuple[str, dict]:
        """Match a goal string to a plan pattern."""
        goal_lower = goal.lower()
        for name, pattern in self.GOAL_PATTERNS.items():
            for trigger in pattern["triggers"]:
                if trigger in goal_lower:
                    return name, pattern
        return "generic", {"triggers": [], "steps": self.GENERIC_STEPS}

    # --- Handlers ---

    def handle(self, name: str, tool_input: dict) -> str:
        """Execute a planner tool call."""
        if name == "plan_goal":
            return self._handle_plan_goal(tool_input)
        elif name == "get_plan":
            return self._handle_get_plan(tool_input)
        elif name == "complete_step":
            return self._handle_complete_step(tool_input)
        elif name == "replan":
            return self._handle_replan(tool_input)
        else:
            return self.to_json({"error": f"Unknown planner tool: {name}"})

    def _handle_plan_goal(self, tool_input: dict) -> str:
        goal = tool_input["goal"]
        session = tool_input.get("session", "default")

        pattern_name, pattern = self._match_pattern(goal)

        steps = []
        for step_def in pattern["steps"]:
            steps.append({
                "id": step_def["id"],
                "action": step_def["action"],
                "tools": step_def.get("tools", []),
                "status": "pending",
                "result": None,
            })

        # Mark first step as active
        if steps:
            steps[0]["status"] = "active"

        goal_id = make_id()

        plan = {
            "goal_id": goal_id,
            "goal": goal,
            "pattern": pattern_name,
            "session": session,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "in_progress",
            "steps": steps,
            "replan_history": [],
        }

        self._save_plan(session, plan)

        return self.to_json({
            "planned": True,
            "goal_id": goal_id,
            "goal": goal,
            "pattern": pattern_name,
            "total_steps": len(steps),
            "current_step": steps[0]["id"] if steps else None,
            "steps": [{"id": s["id"], "action": s["action"], "status": s["status"]} for s in steps],
            "message": f"Plan created with {len(steps)} steps using '{pattern_name}' pattern. First step: {steps[0]['action'] if steps else 'none'}",
        })

    def _handle_get_plan(self, tool_input: dict) -> str:
        session = tool_input.get("session", "default")
        plan = self._load_plan(session)

        if plan is None:
            return self.to_json({"error": "No active plan.", "hint": "Use plan_goal to create one."})

        completed = sum(1 for s in plan["steps"] if s["status"] == "done")
        total = len(plan["steps"])
        active = next((s for s in plan["steps"] if s["status"] == "active"), None)

        return self.to_json({
            "goal_id": plan.get("goal_id"),
            "goal": plan["goal"],
            "pattern": plan["pattern"],
            "status": plan["status"],
            "progress": f"{completed}/{total}",
            "current_step": active["id"] if active else None,
            "current_action": active["action"] if active else None,
            "steps": [
                {
                    "id": s["id"],
                    "action": s["action"],
                    "status": s["status"],
                    "result": s["result"],
                }
                for s in plan["steps"]
            ],
        })

    def _handle_complete_step(self, tool_input: dict) -> str:
        step_id = tool_input["step_id"]
        result = tool_input["result"]
        session = tool_input.get("session", "default")

        plan = self._load_plan(session)
        if plan is None:
            return self.to_json({"error": "No active plan."})

        # Find and complete the step
        step = next((s for s in plan["steps"] if s["id"] == step_id), None)
        if step is None:
            return self.to_json({"error": f"Step '{step_id}' not found in plan."})

        step["status"] = "done"
        step["result"] = result

        # Advance to next pending step
        next_step = next((s for s in plan["steps"] if s["status"] == "pending"), None)
        if next_step:
            next_step["status"] = "active"

        # Check if plan is complete
        all_done = all(s["status"] == "done" for s in plan["steps"])
        if all_done:
            plan["status"] = "completed"

        plan["updated_at"] = time.time()
        self._save_plan(session, plan)

        completed = sum(1 for s in plan["steps"] if s["status"] == "done")
        total = len(plan["steps"])

        response = {
            "completed": step_id,
            "progress": f"{completed}/{total}",
            "plan_status": plan["status"],
        }

        if next_step and not all_done:
            response["next_step"] = next_step["id"]
            response["next_action"] = next_step["action"]
            response["next_tools"] = next_step.get("tools", [])
            response["message"] = f"Step '{step_id}' done. Next: {next_step['action']}"
        elif all_done:
            response["message"] = f"All {total} steps completed! Goal achieved: {plan['goal']}"
        else:
            response["message"] = f"Step '{step_id}' done."

        return self.to_json(response)

    def _handle_replan(self, tool_input: dict) -> str:
        reason = tool_input["reason"]
        new_steps_raw = tool_input.get("new_remaining_steps", [])
        session = tool_input.get("session", "default")

        plan = self._load_plan(session)
        if plan is None:
            return self.to_json({"error": "No active plan."})

        # Save replan history
        plan["replan_history"].append({
            "reason": reason,
            "timestamp": time.time(),
            "old_remaining": [
                s for s in plan["steps"] if s["status"] in ("pending", "active")
            ],
        })

        # Keep completed steps, replace remaining
        completed_steps = [s for s in plan["steps"] if s["status"] == "done"]

        if new_steps_raw:
            new_steps = []
            for i, step_def in enumerate(new_steps_raw):
                new_steps.append({
                    "id": step_def["id"],
                    "action": step_def["action"],
                    "tools": step_def.get("tools", []),
                    "status": "active" if i == 0 else "pending",
                    "result": None,
                })
            plan["steps"] = completed_steps + new_steps
        else:
            # If no new steps provided, just mark remaining as cancelled
            for s in plan["steps"]:
                if s["status"] in ("pending", "active"):
                    s["status"] = "cancelled"
            plan["status"] = "replanned"

        plan["updated_at"] = time.time()
        self._save_plan(session, plan)

        active = next((s for s in plan["steps"] if s["status"] == "active"), None)

        return self.to_json({
            "replanned": True,
            "reason": reason,
            "completed_preserved": len(completed_steps),
            "new_steps": len(plan["steps"]) - len(completed_steps),
            "current_step": active["id"] if active else None,
            "current_action": active["action"] if active else None,
            "message": f"Plan revised: {reason}. {len(completed_steps)} completed steps preserved.",
        })


# ---------------------------------------------------------------------------
# Backward compatibility — lazy singleton
# ---------------------------------------------------------------------------

_instance: PlannerAgent | None = None


def _get_instance() -> PlannerAgent:
    global _instance
    if _instance is None:
        _instance = PlannerAgent()
    return _instance


TOOLS = PlannerAgent.TOOLS
GOAL_PATTERNS = _GOAL_PATTERNS


def handle(name: str, tool_input: dict) -> str:
    """Execute a planner tool call."""
    return _get_instance().handle(name, tool_input)


def _load_plan(session: str) -> dict | None:
    """Module-level proxy for backward compatibility."""
    return _get_instance()._load_plan(session)


def __getattr__(name: str):
    """Proxy module-level state access to singleton instance."""
    if name == "_GENERIC_STEPS":
        return _GENERIC_STEPS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
