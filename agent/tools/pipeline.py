"""Multi-workflow pipeline engine.

Chains multiple workflow executions into a single pipeline.
Each stage's output can be wired as input to subsequent stages,
enabling multi-step generation (e.g., txt2img -> upscale,
image -> 3D mesh, image + TTS -> video composite).

State is module-level: create a pipeline, run it, check status.
"""

import copy
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_pipeline_state: dict[str, Any] = {
    "current_pipeline": None,
    "execution_results": [],
    "status": "idle",
    "error": None,
}
_pipeline_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "create_pipeline",
        "description": (
            "Create a multi-stage pipeline that chains workflow executions. "
            "Each stage loads a workflow (from file or template), applies "
            "parameter overrides, and can receive outputs from previous "
            "stages as inputs. Use 'template:name' for built-in templates "
            "(e.g. 'template:txt2img_sdxl') or a file path for custom "
            "workflows. Returns the pipeline definition for run_pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stages": {
                    "type": "array",
                    "description": (
                        "Ordered list of pipeline stages. Each stage has: "
                        "stage_id (unique string), workflow_source (file "
                        "path or 'template:name'), input_mappings (list "
                        "of mappings from previous stage outputs to this "
                        "stage's node inputs), output_key (type to "
                        "capture: 'image' or 'video'), and "
                        "param_overrides (dict of "
                        "'node_id.input_name' -> value)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage_id": {
                                "type": "string",
                                "description": (
                                    "Unique identifier for this stage."
                                ),
                            },
                            "workflow_source": {
                                "type": "string",
                                "description": (
                                    "File path to workflow JSON, or "
                                    "'template:name' for a built-in "
                                    "template (e.g. "
                                    "'template:txt2img_sdxl')."
                                ),
                            },
                            "input_mappings": {
                                "type": "array",
                                "description": (
                                    "How this stage receives outputs "
                                    "from previous stages."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "node_id": {
                                            "type": "string",
                                            "description": (
                                                "Target node ID in "
                                                "this stage's workflow."
                                            ),
                                        },
                                        "input_name": {
                                            "type": "string",
                                            "description": (
                                                "Input name on the "
                                                "target node."
                                            ),
                                        },
                                        "from_stage": {
                                            "type": "string",
                                            "description": (
                                                "Stage ID to pull "
                                                "output from."
                                            ),
                                        },
                                        "output_index": {
                                            "type": "integer",
                                            "description": (
                                                "Index into the source "
                                                "stage's outputs "
                                                "(default 0)."
                                            ),
                                        },
                                    },
                                    "required": [
                                        "node_id",
                                        "input_name",
                                        "from_stage",
                                    ],
                                },
                            },
                            "output_key": {
                                "type": "string",
                                "description": (
                                    "Output type to capture: 'image' "
                                    "or 'video'. Default 'image'."
                                ),
                            },
                            "param_overrides": {
                                "type": "object",
                                "description": (
                                    "Parameter overrides as "
                                    "'node_id.input_name' -> value. "
                                    "Example: {'3.seed': 42, "
                                    "'6.text': 'a cat'}."
                                ),
                            },
                        },
                        "required": ["stage_id", "workflow_source"],
                    },
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Optional pipeline name for identification."
                    ),
                },
            },
            "required": ["stages"],
        },
    },
    {
        "name": "run_pipeline",
        "description": (
            "Execute a pipeline created by create_pipeline. Runs each "
            "stage sequentially: loads the workflow, applies parameter "
            "overrides, wires in outputs from previous stages, executes, "
            "and captures outputs. Returns a full execution report with "
            "per-stage results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pipeline": {
                    "type": "object",
                    "description": (
                        "Pipeline definition from create_pipeline, or "
                        "an inline pipeline definition."
                    ),
                },
                "timeout_per_stage": {
                    "type": "integer",
                    "description": (
                        "Max seconds per stage (default 300)."
                    ),
                },
                "stop_on_error": {
                    "type": "boolean",
                    "description": (
                        "Halt on first stage failure (default true)."
                    ),
                },
                "session": {
                    "type": "string",
                    "description": (
                        "Session name for memory recording. "
                        "Default 'default'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_pipeline_status",
        "description": (
            "Get the status of the most recently created or executed "
            "pipeline. Shows per-stage results, outputs, and errors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_pipeline_definition(pipeline: dict) -> str | None:
    """Validate a pipeline definition. Returns error message or None."""
    stages = pipeline.get("stages")
    if not stages:
        return "Pipeline must have at least one stage."

    if not isinstance(stages, list):
        return "Pipeline 'stages' must be an array."

    seen_ids: set[str] = set()
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            return f"Stage {i} must be an object."

        stage_id = stage.get("stage_id")
        if not stage_id or not isinstance(stage_id, str):
            return f"Stage {i} is missing a valid 'stage_id'."

        if stage_id in seen_ids:
            return f"Duplicate stage_id: '{stage_id}'."
        seen_ids.add(stage_id)

        source = stage.get("workflow_source")
        if not source or not isinstance(source, str):
            return f"Stage '{stage_id}' is missing 'workflow_source'."

        # Validate input mappings reference earlier stages
        for mapping in stage.get("input_mappings", []):
            from_stage = mapping.get("from_stage")
            if from_stage not in seen_ids:
                return (
                    f"Stage '{stage_id}' input_mapping references "
                    f"'{from_stage}' which hasn't been defined yet. "
                    f"Stages must reference earlier stages only."
                )

    return None


def _load_workflow_for_stage(
    source: str,
) -> tuple[dict | None, str | None]:
    """Load a workflow from file path or template name.

    Returns (workflow_dict, error_message).
    """
    if source.startswith("template:"):
        template_name = source[len("template:"):]
        return _load_template(template_name)
    return _load_from_file(source)


def _load_template(
    template_name: str,
) -> tuple[dict | None, str | None]:
    """Load a workflow from the templates directory."""
    templates_dir = Path(__file__).parent.parent / "templates"
    path = templates_dir / f"{template_name}.json"

    if not path.exists():
        return None, f"Template '{template_name}' not found at {path}."

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in template '{template_name}': {e}"

    return copy.deepcopy(data), None


def _load_from_file(
    path_str: str,
) -> tuple[dict | None, str | None]:
    """Load and extract API-format workflow from a file."""
    from ._util import validate_path

    path_err = validate_path(path_str, must_exist=True)
    if path_err:
        return None, path_err

    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    # Extract API format (same logic as workflow_patch._load_workflow)
    if "nodes" in data and isinstance(data["nodes"], list):
        prompt_data = data.get("extra", {}).get("prompt")
        if prompt_data and isinstance(prompt_data, dict):
            api_nodes = {
                k: v for k, v in prompt_data.items()
                if isinstance(v, dict) and "class_type" in v
            }
        else:
            return None, (
                "UI-only workflow format -- can't execute without "
                "API data. Re-export using 'Save (API Format)' "
                "in ComfyUI."
            )
    else:
        api_nodes = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and "class_type" in v
        }

    if not api_nodes:
        return None, "No nodes found in workflow."

    return copy.deepcopy(api_nodes), None


def _apply_param_overrides(
    workflow: dict, overrides: dict[str, Any],
) -> list[str]:
    """Apply param_overrides to a workflow in-place.

    Override keys use 'node_id.input_name' format.
    Returns a list of error messages (empty if all succeeded).
    """
    errors: list[str] = []
    # He2025: sorted for deterministic override order
    for key, value in sorted(overrides.items()):
        parts = key.split(".", 1)
        if len(parts) != 2:
            errors.append(
                f"Invalid override key '{key}': "
                f"expected 'node_id.input_name' format."
            )
            continue

        node_id, input_name = parts
        if node_id not in workflow:
            errors.append(
                f"Override target node '{node_id}' not found."
            )
            continue

        workflow[node_id].setdefault("inputs", {})[input_name] = value

    return errors


def _apply_input_mappings(
    workflow: dict,
    mappings: list[dict],
    stage_outputs: dict[str, list[dict]],
) -> list[str]:
    """Wire previous stage outputs into this workflow's inputs.

    Returns a list of error messages (empty if all succeeded).
    """
    errors: list[str] = []
    for mapping in mappings:
        node_id = mapping.get("node_id", "")
        input_name = mapping.get("input_name", "")
        from_stage = mapping.get("from_stage", "")
        output_index = mapping.get("output_index", 0)

        source_outputs = stage_outputs.get(from_stage, [])
        if not source_outputs:
            errors.append(
                f"Stage '{from_stage}' has no captured outputs "
                f"(needed by mapping to {node_id}.{input_name})."
            )
            continue

        if output_index < 0 or output_index >= len(source_outputs):
            errors.append(
                f"Output index {output_index} out of range for "
                f"stage '{from_stage}' "
                f"(has {len(source_outputs)} outputs)."
            )
            continue

        output_entry = source_outputs[output_index]
        filename = output_entry.get("filename", "")

        if not filename:
            errors.append(
                f"Stage '{from_stage}' output[{output_index}] "
                f"has no filename."
            )
            continue

        if node_id not in workflow:
            errors.append(
                f"Mapping target node '{node_id}' not found "
                f"in workflow."
            )
            continue

        # ComfyUI LoadImage accepts filenames from output/ dir
        workflow[node_id].setdefault(
            "inputs", {},
        )[input_name] = filename
        log.info(
            "Pipeline: wired %s.%s = '%s' (from stage '%s'[%d])",
            node_id, input_name, filename, from_stage, output_index,
        )

    return errors


def _execute_stage_workflow(
    workflow: dict, timeout: float,
) -> dict:
    """Execute a workflow and return the result dict."""
    from .comfy_execute import _execute_with_websocket

    return _execute_with_websocket(workflow, timeout=timeout)


def _resolve_stage_outputs(
    exec_result: dict, output_key: str,
) -> list[dict]:
    """Extract and resolve output entries from an execution result.

    Filters by output_key ('image' or 'video').
    """
    from .verify_execution import _resolve_output_path

    raw_outputs = exec_result.get("outputs", [])
    resolved: list[dict] = []

    for out in raw_outputs:
        out_type = out.get("type", "image")
        if out_type != output_key:
            continue

        filename = out.get("filename", "")
        subfolder = out.get("subfolder", "")

        path_info = _resolve_output_path(filename, subfolder)
        resolved.append({
            "filename": filename,
            "subfolder": subfolder,
            "absolute_path": path_info.get("absolute_path"),
            "exists": path_info.get("exists", False),
            "size_bytes": path_info.get("size_bytes", 0),
            "type": out_type,
        })

    return resolved


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_create_pipeline(tool_input: dict) -> str:
    """Create and validate a pipeline definition."""
    stages = tool_input.get("stages", [])
    name = tool_input.get("name", "unnamed_pipeline")

    pipeline = {
        "name": name,
        "stages": stages,
        "stage_count": len(stages),
        "created_at": time.time(),
    }

    err = _validate_pipeline_definition(pipeline)
    if err:
        return to_json({"error": err})

    # Build summary
    stage_summaries = []
    for stage in stages:
        summary: dict[str, Any] = {
            "stage_id": stage["stage_id"],
            "workflow_source": stage["workflow_source"],
        }
        mappings = stage.get("input_mappings", [])
        if mappings:
            summary["receives_from"] = sorted(set(
                m["from_stage"] for m in mappings
            ))
        overrides = stage.get("param_overrides", {})
        if overrides:
            summary["override_count"] = len(overrides)
        stage_summaries.append(summary)

    with _pipeline_lock:
        _pipeline_state["current_pipeline"] = pipeline
        _pipeline_state["execution_results"] = []
        _pipeline_state["status"] = "idle"
        _pipeline_state["error"] = None

    return to_json({
        "created": True,
        "name": name,
        "stage_count": len(stages),
        "stages": stage_summaries,
        "message": (
            f"Pipeline '{name}' created with {len(stages)} stage(s). "
            "Use run_pipeline to execute it."
        ),
    })


def _handle_run_pipeline(tool_input: dict) -> str:
    """Execute a multi-stage pipeline."""
    pipeline = tool_input.get("pipeline")
    if pipeline is None:
        with _pipeline_lock:
            pipeline = _pipeline_state.get("current_pipeline")
        if pipeline is None:
            return to_json({
                "error": (
                    "No pipeline to run. Provide a 'pipeline' "
                    "definition or create one with create_pipeline."
                ),
            })

    timeout_per_stage = tool_input.get("timeout_per_stage", 300)
    stop_on_error = tool_input.get("stop_on_error", True)
    session = tool_input.get("session", "default")

    err = _validate_pipeline_definition(pipeline)
    if err:
        return to_json({"error": err})

    stages = pipeline["stages"]
    pipeline_name = pipeline.get("name", "unnamed_pipeline")

    with _pipeline_lock:
        _pipeline_state["current_pipeline"] = pipeline
        _pipeline_state["execution_results"] = []
        _pipeline_state["status"] = "running"
        _pipeline_state["error"] = None

    stage_outputs: dict[str, list[dict]] = {}
    stage_results: list[dict] = []
    pipeline_start = time.monotonic()
    completed_count = 0
    failed_count = 0

    for stage in stages:
        stage_id = stage["stage_id"]
        source = stage["workflow_source"]
        output_key = stage.get("output_key", "image")
        overrides = stage.get("param_overrides", {})
        mappings = stage.get("input_mappings", [])

        stage_start = time.monotonic()
        stage_result: dict[str, Any] = {
            "stage_id": stage_id,
            "workflow_source": source,
            "status": "pending",
            "outputs": [],
            "errors": [],
        }

        log.info(
            "Pipeline '%s': starting stage '%s'",
            pipeline_name, stage_id,
        )

        # 1. Load workflow
        workflow, load_err = _load_workflow_for_stage(source)
        if load_err:
            stage_result["status"] = "error"
            stage_result["errors"].append(
                f"Load failed: {load_err}",
            )
            stage_result["duration_s"] = round(
                time.monotonic() - stage_start, 2,
            )
            stage_results.append(stage_result)
            failed_count += 1

            with _pipeline_lock:
                _pipeline_state["execution_results"] = list(
                    stage_results,
                )

            if stop_on_error:
                log.error(
                    "Pipeline '%s': stage '%s' load failed.",
                    pipeline_name, stage_id,
                )
                break
            continue

        # 2. Apply parameter overrides
        if overrides:
            override_errors = _apply_param_overrides(
                workflow, overrides,
            )
            if override_errors:
                stage_result["errors"].extend(override_errors)

        # 3. Apply input mappings from previous stages
        if mappings:
            mapping_errors = _apply_input_mappings(
                workflow, mappings, stage_outputs,
            )
            if mapping_errors:
                stage_result["status"] = "error"
                stage_result["errors"].extend(mapping_errors)
                stage_result["duration_s"] = round(
                    time.monotonic() - stage_start, 2,
                )
                stage_results.append(stage_result)
                failed_count += 1

                with _pipeline_lock:
                    _pipeline_state["execution_results"] = list(
                        stage_results,
                    )

                if stop_on_error:
                    log.error(
                        "Pipeline '%s': stage '%s' mapping failed.",
                        pipeline_name, stage_id,
                    )
                    break
                continue

        # 4. Execute
        log.info(
            "Pipeline '%s': executing stage '%s' (timeout=%ds)",
            pipeline_name, stage_id, timeout_per_stage,
        )
        exec_result = _execute_stage_workflow(
            workflow, timeout_per_stage,
        )

        if exec_result.get("error"):
            stage_result["status"] = "error"
            stage_result["errors"].append(exec_result["error"])
            stage_result["duration_s"] = round(
                time.monotonic() - stage_start, 2,
            )
            stage_results.append(stage_result)
            failed_count += 1

            with _pipeline_lock:
                _pipeline_state["execution_results"] = list(
                    stage_results,
                )

            if stop_on_error:
                log.error(
                    "Pipeline '%s': stage '%s' exec error: %s",
                    pipeline_name, stage_id,
                    exec_result.get("error"),
                )
                break
            continue

        exec_status = exec_result.get("status", "unknown")
        if exec_status == "timeout":
            stage_result["status"] = "timeout"
            stage_result["errors"].append(
                f"Stage timed out after {timeout_per_stage}s.",
            )
            stage_result["prompt_id"] = exec_result.get("prompt_id")
            stage_result["duration_s"] = round(
                time.monotonic() - stage_start, 2,
            )
            stage_results.append(stage_result)
            failed_count += 1

            with _pipeline_lock:
                _pipeline_state["execution_results"] = list(
                    stage_results,
                )

            if stop_on_error:
                break
            continue

        # 5. Capture and resolve outputs
        resolved_outputs = _resolve_stage_outputs(
            exec_result, output_key,
        )
        stage_outputs[stage_id] = resolved_outputs

        stage_result["status"] = "complete"
        stage_result["prompt_id"] = exec_result.get("prompt_id")
        stage_result["outputs"] = resolved_outputs
        stage_result["output_count"] = len(resolved_outputs)
        stage_result["duration_s"] = round(
            time.monotonic() - stage_start, 2,
        )
        if exec_result.get("total_time_s"):
            stage_result["render_time_s"] = exec_result[
                "total_time_s"
            ]

        stage_results.append(stage_result)
        completed_count += 1

        with _pipeline_lock:
            _pipeline_state["execution_results"] = list(
                stage_results,
            )

        log.info(
            "Pipeline '%s': stage '%s' complete "
            "(%d outputs, %.1fs)",
            pipeline_name, stage_id, len(resolved_outputs),
            stage_result["duration_s"],
        )

    # Pipeline finished
    total_time = round(time.monotonic() - pipeline_start, 2)
    pipeline_status = (
        "complete" if failed_count == 0 else "error"
    )

    with _pipeline_lock:
        _pipeline_state["execution_results"] = list(stage_results)
        _pipeline_state["status"] = pipeline_status
        if failed_count > 0:
            _pipeline_state["error"] = (
                f"{failed_count} stage(s) failed "
                f"out of {len(stages)}."
            )

    # Final outputs from last successful stage
    final_outputs: list[dict] = []
    for result in reversed(stage_results):
        if result["status"] == "complete" and result.get("outputs"):
            final_outputs = result["outputs"]
            break

    # Record pipeline outcome to memory
    outcome_recorded = False
    try:
        from . import handle as dispatch_tool
        dispatch_tool("record_outcome", {
            "session": session,
            "key_params": {
                "pipeline": pipeline_name,
                "stages": completed_count,
                "total_stages": len(stages),
            },
            "workflow_summary": (
                f"Pipeline '{pipeline_name}': "
                f"{completed_count}/{len(stages)} stages complete"
            ),
            "render_time_s": total_time,
        })
        outcome_recorded = True
    except Exception as e:
        log.warning("Failed to record pipeline outcome: %s", e)

    return to_json({
        "pipeline": pipeline_name,
        "status": pipeline_status,
        "stages_total": len(stages),
        "stages_completed": completed_count,
        "stages_failed": failed_count,
        "total_time_s": total_time,
        "stage_results": stage_results,
        "final_outputs": final_outputs,
        "outcome_recorded": outcome_recorded,
        "message": (
            f"Pipeline '{pipeline_name}' {pipeline_status}: "
            f"{completed_count}/{len(stages)} stages completed "
            f"in {total_time}s."
        ),
    })


def _handle_get_pipeline_status(tool_input: dict) -> str:
    """Return the current pipeline state."""
    with _pipeline_lock:
        pipeline = _pipeline_state["current_pipeline"]
        results = list(_pipeline_state["execution_results"])
        status = _pipeline_state["status"]
        error = _pipeline_state["error"]

    if pipeline is None:
        return to_json({
            "status": "no_pipeline",
            "message": (
                "No pipeline has been created yet. "
                "Use create_pipeline first."
            ),
        })

    completed = sum(
        1 for r in results if r.get("status") == "complete"
    )
    failed = sum(
        1 for r in results
        if r.get("status") in ("error", "timeout")
    )
    total = pipeline.get(
        "stage_count", len(pipeline.get("stages", [])),
    )

    all_outputs: list[dict] = []
    for result in results:
        for out in result.get("outputs", []):
            all_outputs.append({
                "stage_id": result.get("stage_id"),
                "filename": out.get("filename"),
                "absolute_path": out.get("absolute_path"),
                "exists": out.get("exists", False),
            })

    response: dict[str, Any] = {
        "pipeline_name": pipeline.get("name", "unnamed"),
        "status": status,
        "stages_total": total,
        "stages_completed": completed,
        "stages_failed": failed,
        "stage_results": results,
        "all_outputs": all_outputs,
    }

    if error:
        response["error"] = error

    return to_json(response)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def handle(name: str, tool_input: dict) -> str:
    """Execute a pipeline tool call."""
    try:
        if name == "create_pipeline":
            return _handle_create_pipeline(tool_input)
        elif name == "run_pipeline":
            return _handle_run_pipeline(tool_input)
        elif name == "get_pipeline_status":
            return _handle_get_pipeline_status(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        log.error(
            "Unhandled error in pipeline tool %s",
            name, exc_info=True,
        )
        return to_json({"error": str(e)})
