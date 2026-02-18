"""Post-execution verification tools.

Closes the execute -> verify -> learn loop by:
1. Resolving ComfyUI output filenames to absolute paths
2. Validating outputs exist and have non-zero size
3. Optionally running vision analysis
4. Recording outcomes to memory for cross-session learning
"""

import hashlib
import json
import logging

import httpx

from ..config import COMFYUI_OUTPUT_DIR, COMFYUI_URL
from ._util import to_json, validate_path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "get_output_path",
        "description": (
            "Resolve a ComfyUI output filename to its absolute path on disk. "
            "Use after execute_workflow or get_execution_status to find where "
            "ComfyUI saved the output file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Output filename from ComfyUI (e.g. 'ComfyUI_00001_.png').",
                },
                "subfolder": {
                    "type": "string",
                    "description": "Subfolder within output directory (default: empty).",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "verify_execution",
        "description": (
            "Post-execution verification: checks that outputs exist on disk, "
            "optionally analyzes image quality via Vision, and records the "
            "outcome to memory for learning. Call after execute_workflow or "
            "execute_with_progress to close the verify loop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt_id": {
                    "type": "string",
                    "description": "The prompt_id returned from execute_workflow.",
                },
                "analyze": {
                    "type": "boolean",
                    "description": (
                        "Run Claude Vision analysis on image outputs (costs tokens, "
                        "~120s). Default false -- set true for quality review."
                    ),
                },
                "session": {
                    "type": "string",
                    "description": "Session name for memory recording. Default 'default'.",
                },
                "render_time_s": {
                    "type": "number",
                    "description": "Total render time in seconds (from execute_with_progress).",
                },
                "goal_id": {
                    "type": "string",
                    "description": "Goal ID from planner, for linking outcome to a goal.",
                },
            },
            "required": ["prompt_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_output_path(filename: str, subfolder: str = "") -> dict:
    """Resolve a ComfyUI output filename to absolute path + stats.

    Returns a dict (not JSON string) for internal use.
    """
    if not filename:
        return {
            "filename": filename,
            "subfolder": subfolder,
            "absolute_path": None,
            "exists": False,
            "size_bytes": 0,
            "error": "Empty filename",
        }

    output_dir = COMFYUI_OUTPUT_DIR
    if subfolder:
        abs_path = output_dir / subfolder / filename
    else:
        abs_path = output_dir / filename

    abs_path = abs_path.resolve()

    # Validate path is within safe directories
    path_err = validate_path(str(abs_path))
    if path_err:
        return {
            "filename": filename,
            "subfolder": subfolder,
            "absolute_path": str(abs_path),
            "exists": False,
            "size_bytes": 0,
            "error": path_err,
        }

    exists = abs_path.exists()
    size_bytes = abs_path.stat().st_size if exists else 0

    return {
        "filename": filename,
        "subfolder": subfolder,
        "absolute_path": str(abs_path),
        "exists": exists,
        "size_bytes": size_bytes,
        "error": None if exists else f"File not found: {abs_path}",
    }


def _extract_key_params(workflow: dict) -> dict:
    """Extract key generation parameters from a workflow dict.

    Scans for common node types: CheckpointLoaderSimple, KSampler,
    EmptyLatentImage. Returns a flat dict of key params.
    """
    params: dict = {}

    # He2025: sort for deterministic iteration
    for _nid, node in sorted(workflow.items()):
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type == "CheckpointLoaderSimple":
            ckpt = inputs.get("ckpt_name")
            if isinstance(ckpt, str):
                params["model"] = ckpt

        elif class_type == "KSampler":
            for key in ("steps", "cfg", "sampler_name", "scheduler", "seed", "denoise"):
                val = inputs.get(key)
                if val is not None and not isinstance(val, list):
                    params[key] = val

        elif class_type == "EmptyLatentImage":
            w = inputs.get("width")
            h = inputs.get("height")
            if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                params["resolution"] = f"{int(w)}x{int(h)}"

    return params


def _workflow_hash(workflow: dict) -> str:
    """Deterministic hash of workflow for fingerprinting."""
    raw = json.dumps(workflow, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _verify_prompt(
    prompt_id: str,
    *,
    analyze: bool = False,
    session: str = "default",
    render_time_s: float | None = None,
    goal_id: str | None = None,
) -> dict:
    """Internal verification logic. Returns a dict (not JSON string).

    Called by verify_execution tool and by execute_with_progress auto_verify.
    """
    # 1. Fetch history for this prompt
    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{COMFYUI_URL}/history/{prompt_id}",
                timeout=10.0,
            )
            resp.raise_for_status()
            history = resp.json()
    except httpx.ConnectError:
        return {
            "prompt_id": prompt_id,
            "status": "error",
            "message": f"ComfyUI not reachable at {COMFYUI_URL}. Is it running?",
            "outputs": [],
            "output_count": 0,
            "all_exist": False,
            "vision_analysis": None,
            "outcome_recorded": False,
        }
    except Exception as e:
        return {
            "prompt_id": prompt_id,
            "status": "error",
            "message": str(e),
            "outputs": [],
            "output_count": 0,
            "all_exist": False,
            "vision_analysis": None,
            "outcome_recorded": False,
        }

    if prompt_id not in history:
        return {
            "prompt_id": prompt_id,
            "status": "not_found",
            "message": "Prompt not found in history. It may still be running.",
            "outputs": [],
            "output_count": 0,
            "all_exist": False,
            "vision_analysis": None,
            "outcome_recorded": False,
        }

    entry = history[prompt_id]
    status_info = entry.get("status", {})
    status_str = status_info.get("status_str", "unknown")

    # 2. Resolve each output to absolute path
    outputs = []
    # He2025: sort for deterministic output order
    for node_id, node_out in sorted(entry.get("outputs", {}).items()):
        for img in node_out.get("images", []):
            resolved = _resolve_output_path(
                img.get("filename", ""),
                img.get("subfolder", ""),
            )
            resolved["type"] = "image"
            resolved["size_ok"] = resolved["size_bytes"] > 0
            outputs.append(resolved)
        for vid in node_out.get("gifs", []):
            resolved = _resolve_output_path(
                vid.get("filename", ""),
                vid.get("subfolder", ""),
            )
            resolved["type"] = "video"
            resolved["size_ok"] = resolved["size_bytes"] > 0
            outputs.append(resolved)

    all_exist = all(o["exists"] for o in outputs) if outputs else False

    # 3. Extract key params from loaded workflow
    key_params: dict = {}
    workflow_hash = ""
    try:
        from .workflow_patch import get_current_workflow
        wf = get_current_workflow()
        if wf:
            key_params = _extract_key_params(wf)
            workflow_hash = _workflow_hash(wf)
    except Exception:
        pass

    # 4. Optional vision analysis
    vision_analysis = None
    if analyze and outputs:
        # Find first existing image output
        image_outputs = [o for o in outputs if o["type"] == "image" and o["exists"]]
        if image_outputs:
            try:
                from . import handle as dispatch_tool
                raw_result = dispatch_tool(
                    "analyze_image",
                    {"image_path": image_outputs[0]["absolute_path"]},
                )
                vision_analysis = json.loads(raw_result)
            except Exception as e:
                log.warning("Vision analysis failed: %s", e)
                vision_analysis = {"error": str(e)}

    # 5. Record outcome to memory
    # record_outcome expects top-level fields: key_params, render_time_s,
    # quality_score, workflow_hash, goal_id, model_combo, etc.
    outcome_recorded = False
    try:
        from . import handle as dispatch_tool

        # Build model_combo from key_params
        model_combo = []
        if key_params.get("model"):
            model_combo.append(key_params["model"])

        outcome_input: dict = {
            "session": session,
            "key_params": key_params,
            "workflow_hash": workflow_hash,
            "workflow_summary": f"{key_params.get('model', 'unknown')} "
                               f"{key_params.get('steps', '?')} steps "
                               f"CFG {key_params.get('cfg', '?')}",
            "model_combo": model_combo,
        }
        if render_time_s is not None:
            outcome_input["render_time_s"] = render_time_s
        if goal_id:
            outcome_input["goal_id"] = goal_id
        if vision_analysis and "quality_score" in vision_analysis:
            outcome_input["quality_score"] = vision_analysis["quality_score"]

        dispatch_tool("record_outcome", outcome_input)
        outcome_recorded = True
    except Exception as e:
        log.warning("Failed to record outcome: %s", e)

    # 6. Build result
    mapped_status = "complete" if status_str == "success" else status_str
    message_parts = []
    if all_exist and outputs:
        message_parts.append(f"{len(outputs)} output(s) verified on disk.")
    elif outputs:
        missing = sum(1 for o in outputs if not o["exists"])
        message_parts.append(f"{missing} of {len(outputs)} output(s) missing.")
    else:
        message_parts.append("No outputs found in history.")
    if vision_analysis and "quality_score" in vision_analysis:
        message_parts.append(f"Quality score: {vision_analysis['quality_score']}")
    if outcome_recorded:
        message_parts.append("Outcome recorded to memory.")

    return {
        "prompt_id": prompt_id,
        "status": mapped_status,
        "outputs": outputs,
        "output_count": len(outputs),
        "all_exist": all_exist,
        "vision_analysis": vision_analysis,
        "outcome_recorded": outcome_recorded,
        "workflow_hash": workflow_hash,
        "key_params": key_params,
        "render_time_s": render_time_s,
        "message": " ".join(message_parts),
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_get_output_path(tool_input: dict) -> str:
    filename = tool_input.get("filename", "")
    subfolder = tool_input.get("subfolder", "")
    result = _resolve_output_path(filename, subfolder)
    return to_json(result)


def _handle_verify_execution(tool_input: dict) -> str:
    prompt_id = tool_input["prompt_id"]
    analyze = tool_input.get("analyze", False)
    session = tool_input.get("session", "default")
    render_time_s = tool_input.get("render_time_s")
    goal_id = tool_input.get("goal_id")

    result = _verify_prompt(
        prompt_id,
        analyze=analyze,
        session=session,
        render_time_s=render_time_s,
        goal_id=goal_id,
    )
    return to_json(result)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a verify_execution tool call."""
    try:
        if name == "get_output_path":
            return _handle_get_output_path(tool_input)
        elif name == "verify_execution":
            return _handle_verify_execution(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
