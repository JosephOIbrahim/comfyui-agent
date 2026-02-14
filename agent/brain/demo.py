"""Demo module — guided walkthroughs for streams and podcasts.

Provides scripted-but-adaptive demo scenarios that make the agent
look like magic. The agent narrates what it's doing, explains tool calls
in artist terms, and paces for an audience.
"""

import logging
import threading
import time

from ..tools._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

DEMO_SCENARIOS: dict[str, dict] = {
    "model_swap": {
        "title": "Upgrading Your Pipeline",
        "description": (
            "Take an existing workflow and upgrade the model. "
            "Shows: workflow analysis, model search, safe swap, "
            "same-seed comparison."
        ),
        "audience": "Artists curious about trying new models",
        "duration_estimate": "3-5 minutes",
        "steps": [
            {
                "id": "analyze",
                "label": "Analyze Current Workflow",
                "narration": (
                    "First, let me understand what we're working with. "
                    "I'll look at the workflow structure, what model it's using, "
                    "and what parameters are set."
                ),
                "suggested_tools": ["load_workflow", "get_editable_fields"],
            },
            {
                "id": "find_upgrade",
                "label": "Find a Better Model",
                "narration": (
                    "Now I'll search for alternatives. I'm looking for models "
                    "that are compatible with this workflow but offer better "
                    "quality, speed, or both."
                ),
                "suggested_tools": ["list_models", "discover", "get_recommendations"],
            },
            {
                "id": "apply_swap",
                "label": "Swap the Model",
                "narration": (
                    "Here's the key part. I'm making a single, targeted change "
                    "to swap the model. Everything else stays exactly the same "
                    "so we can do a fair comparison."
                ),
                "suggested_tools": ["set_input", "validate_before_execute"],
            },
            {
                "id": "compare",
                "label": "Compare Results",
                "narration": (
                    "Let me run both versions with the same seed and compare. "
                    "This gives us an apples-to-apples comparison of the two models."
                ),
                "suggested_tools": ["execute_workflow", "compare_outputs", "record_outcome"],
            },
        ],
    },
    "speed_run": {
        "title": "Making It Fast",
        "description": (
            "Profile a workflow and optimize it step by step. "
            "Shows: profiling, optimization suggestions, TensorRT check, "
            "benchmark comparison."
        ),
        "audience": "Artists dealing with slow renders",
        "duration_estimate": "5-8 minutes",
        "steps": [
            {
                "id": "profile",
                "label": "Profile the Workflow",
                "narration": (
                    "Let me analyze where this workflow spends its time. "
                    "I'll check which nodes are GPU-intensive, how much VRAM "
                    "it needs, and whether there are obvious bottlenecks."
                ),
                "suggested_tools": ["profile_workflow", "get_system_stats"],
            },
            {
                "id": "suggest",
                "label": "Identify Optimizations",
                "narration": (
                    "Based on the profile and your GPU, here are the optimizations "
                    "ranked by how much they'll help versus how hard they are to set up. "
                    "I always start with the free wins."
                ),
                "suggested_tools": ["suggest_optimizations", "check_tensorrt_status"],
            },
            {
                "id": "apply",
                "label": "Apply Top Optimization",
                "narration": (
                    "Let me apply the highest-impact optimization. "
                    "I'll make the change and then we'll benchmark to see "
                    "exactly how much faster it got."
                ),
                "suggested_tools": ["apply_optimization", "validate_before_execute"],
            },
            {
                "id": "benchmark",
                "label": "Benchmark the Result",
                "narration": (
                    "Time for the proof. Same workflow, same seed, "
                    "before and after. Let's see the numbers."
                ),
                "suggested_tools": ["execute_workflow", "compare_outputs", "record_outcome"],
            },
        ],
    },
    "controlnet_add": {
        "title": "Adding ControlNet Guidance",
        "description": (
            "Add ControlNet depth guidance to an existing workflow. "
            "Shows: node search, workflow modification, testing."
        ),
        "audience": "Artists who want more control over generation",
        "duration_estimate": "5-7 minutes",
        "steps": [
            {
                "id": "explain",
                "label": "What is ControlNet?",
                "narration": (
                    "ControlNet lets you guide the generation using a reference image. "
                    "Think of it as giving the AI a rough sketch or depth map to follow. "
                    "The result matches the structure of your guide while still following "
                    "the text prompt."
                ),
                "suggested_tools": [],
            },
            {
                "id": "find_nodes",
                "label": "Find Required Nodes",
                "narration": (
                    "We need two things: a ControlNet preprocessor (to create the guide) "
                    "and a ControlNet apply node (to use it during generation). "
                    "Let me check what's installed."
                ),
                "suggested_tools": ["discover", "find_missing_nodes"],
            },
            {
                "id": "wire_up",
                "label": "Add and Connect Nodes",
                "narration": (
                    "I'm adding the ControlNet nodes and wiring them into the pipeline. "
                    "The depth map goes into the ControlNet apply node, which feeds into "
                    "the sampler alongside the text prompt."
                ),
                "suggested_tools": ["add_node", "connect_nodes", "set_input"],
            },
            {
                "id": "test",
                "label": "Test Run",
                "narration": (
                    "Let's run it and see the result. The generation should follow "
                    "the depth structure of the reference while matching our prompt."
                ),
                "suggested_tools": [
                    "validate_before_execute", "execute_workflow", "analyze_image",
                ],
            },
        ],
    },
    "full_pipeline": {
        "title": "From Zero to Rendered",
        "description": (
            "Build a complete pipeline from a template. "
            "Shows: template selection, customization, optimization, execution."
        ),
        "audience": "New ComfyUI users or demo audiences",
        "duration_estimate": "8-12 minutes",
        "steps": [
            {
                "id": "choose_template",
                "label": "Choose a Starting Point",
                "narration": (
                    "Instead of building from scratch, I'll start with a proven template. "
                    "These are minimal, clean workflows that cover the most common patterns."
                ),
                "suggested_tools": ["list_workflow_templates"],
            },
            {
                "id": "customize",
                "label": "Customize for Your Needs",
                "narration": (
                    "Now I'll customize the template: set the prompt, choose a model, "
                    "adjust resolution and quality settings. Each change is small, "
                    "validated, and reversible."
                ),
                "suggested_tools": [
                    "get_workflow_template", "set_input", "list_models",
                ],
            },
            {
                "id": "optimize",
                "label": "Optimize for Your Hardware",
                "narration": (
                    "Before running, let me check if there are any quick optimizations "
                    "for your GPU. Free speed is always worth grabbing."
                ),
                "suggested_tools": ["profile_workflow", "suggest_optimizations"],
            },
            {
                "id": "execute",
                "label": "Execute and Evaluate",
                "narration": (
                    "Everything's set. Let me validate the pipeline, execute it, "
                    "and then analyze the output."
                ),
                "suggested_tools": [
                    "validate_before_execute", "execute_workflow", "analyze_image",
                ],
            },
            {
                "id": "iterate",
                "label": "Iterate and Improve",
                "narration": (
                    "Based on the output analysis, I can suggest tweaks. "
                    "Each iteration gets us closer to what you're looking for."
                ),
                "suggested_tools": ["suggest_improvements", "set_input", "compare_outputs"],
            },
        ],
    },
}

# Module-level state for active demo
_demo_state: dict = {
    "active": False,
    "scenario": None,
    "current_step_idx": 0,
    "started_at": None,
    "checkpoints": [],
}
_demo_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "start_demo",
        "description": (
            "Start a guided demo scenario. Activates narration mode where "
            "the agent explains every action in artist-friendly terms. "
            "Available scenarios: model_swap, speed_run, controlnet_add, full_pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "description": "Demo scenario name. Use 'list' to see all available scenarios.",
                },
            },
            "required": ["scenario"],
        },
    },
    {
        "name": "demo_checkpoint",
        "description": (
            "Mark a demo milestone. Summarizes what just happened and "
            "previews what comes next. Use for pacing during live demos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step_completed": {
                    "type": "string",
                    "description": "ID of the step that was just completed.",
                },
                "notes": {
                    "type": "string",
                    "description": "Any additional notes about what happened in this step.",
                },
            },
            "required": ["step_completed"],
        },
    },
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_start_demo(tool_input: dict) -> str:
    scenario_name = tool_input["scenario"]

    if scenario_name == "list":
        scenarios = []
        for name, sc in sorted(DEMO_SCENARIOS.items()):
            scenarios.append({
                "name": name,
                "title": sc["title"],
                "description": sc["description"],
                "audience": sc["audience"],
                "duration": sc["duration_estimate"],
                "steps": len(sc["steps"]),
            })
        return to_json({
            "available_scenarios": scenarios,
            "count": len(scenarios),
        })

    scenario = DEMO_SCENARIOS.get(scenario_name)
    if not scenario:
        return to_json({
            "error": f"Unknown scenario: {scenario_name}",
            "available": sorted(DEMO_SCENARIOS.keys()),
        })

    _demo_state["active"] = True
    _demo_state["scenario"] = scenario_name
    _demo_state["current_step_idx"] = 0
    _demo_state["started_at"] = time.time()
    _demo_state["checkpoints"] = []

    first_step = scenario["steps"][0]

    return to_json({
        "demo_started": True,
        "scenario": scenario_name,
        "title": scenario["title"],
        "description": scenario["description"],
        "total_steps": len(scenario["steps"]),
        "first_step": {
            "id": first_step["id"],
            "label": first_step["label"],
            "narration": first_step["narration"],
            "suggested_tools": first_step["suggested_tools"],
        },
        "message": (
            f"Demo '{scenario['title']}' started! "
            f"{len(scenario['steps'])} steps. "
            f"Estimated duration: {scenario['duration_estimate']}. "
            f"First up: {first_step['label']}."
        ),
    })


def _handle_demo_checkpoint(tool_input: dict) -> str:
    step_completed = tool_input["step_completed"]
    notes = tool_input.get("notes", "")

    if not _demo_state["active"]:
        return to_json({"error": "No active demo. Use start_demo first."})

    scenario = DEMO_SCENARIOS.get(_demo_state["scenario"])
    if not scenario:
        return to_json({"error": "Demo scenario not found."})

    # Record checkpoint
    _demo_state["checkpoints"].append({
        "step": step_completed,
        "notes": notes,
        "timestamp": time.time(),
    })

    # Advance to next step
    _demo_state["current_step_idx"] += 1
    step_idx = _demo_state["current_step_idx"]
    total = len(scenario["steps"])

    if step_idx >= total:
        # Demo complete
        elapsed = time.time() - _demo_state["started_at"]
        _demo_state["active"] = False

        return to_json({
            "demo_complete": True,
            "scenario": _demo_state["scenario"],
            "title": scenario["title"],
            "steps_completed": total,
            "elapsed_s": round(elapsed, 1),
            "elapsed_human": f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
            "checkpoints": _demo_state["checkpoints"],
            "message": (
                f"Demo complete! '{scenario['title']}' finished in "
                f"{int(elapsed // 60)}m {int(elapsed % 60)}s."
            ),
        })

    # Show next step
    next_step = scenario["steps"][step_idx]
    progress = f"{step_idx}/{total}"

    return to_json({
        "checkpoint": step_completed,
        "progress": progress,
        "next_step": {
            "id": next_step["id"],
            "label": next_step["label"],
            "narration": next_step["narration"],
            "suggested_tools": next_step["suggested_tools"],
        },
        "elapsed_s": round(time.time() - _demo_state["started_at"], 1),
        "message": (
            f"[{progress}] {step_completed} complete. "
            f"Next: {next_step['label']}."
        ),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a demo tool call."""
    with _demo_lock:
        if name == "start_demo":
            return _handle_start_demo(tool_input)
        elif name == "demo_checkpoint":
            return _handle_demo_checkpoint(tool_input)
        else:
            return to_json({"error": f"Unknown demo tool: {name}"})
