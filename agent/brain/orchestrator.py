"""Orchestrator module — parallel sub-task coordination.

Decides when a task benefits from parallel work and coordinates results.
Today: function calls in threads with filtered tool access.
Tomorrow: Agent SDK sub-agents with isolated context windows.
"""

import concurrent.futures
import logging
import threading
import time
import uuid

from ..tools._util import to_json

log = logging.getLogger(__name__)

# Max concurrent sub-tasks (matches global constraint)
_MAX_CONCURRENT = 3
_SUBTASK_TIMEOUT_S = 60
_COMPLETED_TTL_S = 600  # Evict completed tasks after 10 minutes

# Active sub-tasks: {task_id: {"status": ..., "result": ..., ...}}
_active_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

# Tool access levels for sub-tasks
_TOOL_PROFILES = {
    "researcher": {
        "description": "Read-only research: inspect, discover, search",
        "allowed_tools": {
            "is_comfyui_running", "get_all_nodes", "get_node_info",
            "get_system_stats", "get_queue_status",
            "list_custom_nodes", "list_models", "get_models_summary", "read_node_source",
            "search_custom_nodes", "search_models", "find_missing_nodes",
            "load_workflow", "validate_workflow", "get_editable_fields",
            "list_workflow_templates",
            "get_learned_patterns", "get_recommendations",
        },
    },
    "builder": {
        "description": "Full PILOT access: can modify workflows",
        "allowed_tools": {
            "apply_workflow_patch", "preview_workflow_patch", "undo_workflow_patch",
            "get_workflow_diff", "save_workflow", "reset_workflow",
            "add_node", "connect_nodes", "set_input",
            "get_workflow_template",
            "validate_before_execute",
        },
    },
    "validator": {
        "description": "Validation and execution",
        "allowed_tools": {
            "validate_before_execute", "execute_workflow", "get_execution_status",
            "analyze_image", "compare_outputs",
        },
    },
}


TOOLS: list[dict] = [
    {
        "name": "spawn_subtask",
        "description": (
            "Launch a focused sub-task in a background thread with filtered "
            "tool access. Use when the planner identifies independent work "
            "that can run in parallel (e.g., research multiple options simultaneously). "
            "Max 3 concurrent sub-tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "What this sub-task should accomplish.",
                },
                "profile": {
                    "type": "string",
                    "enum": ["researcher", "builder", "validator"],
                    "description": (
                        "Tool access profile. 'researcher' = read-only, "
                        "'builder' = can modify workflows, "
                        "'validator' = can execute and analyze."
                    ),
                },
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "input": {"type": "object"},
                        },
                        "required": ["tool", "input"],
                    },
                    "description": "List of tool calls to execute in sequence.",
                },
            },
            "required": ["task_description", "profile", "tool_calls"],
        },
    },
    {
        "name": "check_subtasks",
        "description": (
            "Check the status of all active sub-tasks. Returns completed "
            "results and progress updates for running tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific task IDs to check. If empty, checks all active tasks.",
                },
            },
            "required": [],
        },
    },
]


def _evict_stale_tasks() -> None:
    """Remove completed/errored tasks older than TTL. Must be called with _tasks_lock held."""
    now = time.time()
    stale = [
        tid for tid, task in _active_tasks.items()
        if task["status"] in ("completed", "error", "timeout")
        and now - task.get("completed_at", task.get("started_at", now)) > _COMPLETED_TTL_S
    ]
    for tid in stale:
        del _active_tasks[tid]


def _run_subtask(task_id: str, profile: str, tool_calls: list[dict]) -> dict:
    """Execute a sequence of tool calls with filtered access."""
    from ..tools import handle as handle_tool

    allowed = _TOOL_PROFILES.get(profile, {}).get("allowed_tools", set())
    results = []

    for call in tool_calls:
        tool_name = call["tool"]
        tool_input = call.get("input", {})

        if tool_name not in allowed:
            results.append({
                "tool": tool_name,
                "error": f"Tool '{tool_name}' not allowed in '{profile}' profile.",
            })
            continue

        try:
            t0 = time.monotonic()
            result = handle_tool(tool_name, tool_input)
            elapsed = time.monotonic() - t0
            results.append({
                "tool": tool_name,
                "result": result,
                "elapsed_s": round(elapsed, 2),
            })
        except Exception as e:
            results.append({
                "tool": tool_name,
                "error": str(e),
            })

    return {
        "task_id": task_id,
        "status": "completed",
        "results": results,
        "completed_at": time.time(),
    }


def _handle_spawn_subtask(tool_input: dict) -> str:
    task_description = tool_input["task_description"]
    profile = tool_input["profile"]
    tool_calls = tool_input["tool_calls"]

    with _tasks_lock:
        _evict_stale_tasks()
        # Check concurrency limit
        active_count = sum(
            1 for t in _active_tasks.values() if t["status"] == "running"
        )
    if active_count >= _MAX_CONCURRENT:
        return to_json({
            "error": f"Max {_MAX_CONCURRENT} concurrent sub-tasks. Wait for one to finish.",
            "active_count": active_count,
        })

    # Validate profile
    if profile not in _TOOL_PROFILES:
        return to_json({
            "error": f"Unknown profile: {profile}",
            "available": list(_TOOL_PROFILES.keys()),
        })

    # Validate tool access
    allowed = _TOOL_PROFILES[profile]["allowed_tools"]
    for call in tool_calls:
        if call["tool"] not in allowed:
            return to_json({
                "error": f"Tool '{call['tool']}' not allowed in '{profile}' profile.",
                "allowed_tools": sorted(allowed),
            })

    task_id = uuid.uuid4().hex[:8]

    with _tasks_lock:
        _active_tasks[task_id] = {
            "status": "running",
            "description": task_description,
            "profile": profile,
            "started_at": time.time(),
            "tool_count": len(tool_calls),
        }

    # Run in background thread
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_subtask, task_id, profile, tool_calls)

    def _on_done(fut):
        try:
            result = fut.result(timeout=_SUBTASK_TIMEOUT_S)
            with _tasks_lock:
                _active_tasks[task_id].update(result)
        except concurrent.futures.TimeoutError:
            with _tasks_lock:
                _active_tasks[task_id]["status"] = "timeout"
                _active_tasks[task_id]["error"] = f"Timed out after {_SUBTASK_TIMEOUT_S}s"
        except Exception as e:
            with _tasks_lock:
                _active_tasks[task_id]["status"] = "error"
                _active_tasks[task_id]["error"] = str(e)
        finally:
            executor.shutdown(wait=False)

    future.add_done_callback(_on_done)

    return to_json({
        "spawned": True,
        "task_id": task_id,
        "description": task_description,
        "profile": profile,
        "tool_count": len(tool_calls),
        "message": f"Sub-task '{task_id}' spawned with {len(tool_calls)} tool calls ({profile} profile).",
    })


def _handle_check_subtasks(tool_input: dict) -> str:
    task_ids = tool_input.get("task_ids", [])

    with _tasks_lock:
        _evict_stale_tasks()
        if task_ids:
            tasks = {tid: _active_tasks[tid] for tid in task_ids if tid in _active_tasks}
        else:
            tasks = dict(_active_tasks)

    if not tasks:
        return to_json({"tasks": [], "message": "No active sub-tasks."})

    summaries = []
    for tid, task in sorted(tasks.items()):
        summary = {
            "task_id": tid,
            "status": task["status"],
            "description": task.get("description", ""),
            "profile": task.get("profile", ""),
        }

        if task["status"] == "completed":
            results = task.get("results", [])
            summary["tool_results"] = len(results)
            summary["errors"] = sum(1 for r in results if "error" in r)
            summary["results"] = results
        elif task["status"] == "running":
            elapsed = time.time() - task.get("started_at", time.time())
            summary["elapsed_s"] = round(elapsed, 1)
            summary["tool_count"] = task.get("tool_count", 0)
        elif task["status"] in ("error", "timeout"):
            summary["error"] = task.get("error", "Unknown error")

        summaries.append(summary)

    completed = sum(1 for s in summaries if s["status"] == "completed")
    running = sum(1 for s in summaries if s["status"] == "running")

    return to_json({
        "tasks": summaries,
        "summary": {
            "total": len(summaries),
            "completed": completed,
            "running": running,
            "errors": sum(1 for s in summaries if s["status"] in ("error", "timeout")),
        },
    })


def handle(name: str, tool_input: dict) -> str:
    """Execute an orchestrator tool call."""
    if name == "spawn_subtask":
        return _handle_spawn_subtask(tool_input)
    elif name == "check_subtasks":
        return _handle_check_subtasks(tool_input)
    else:
        return to_json({"error": f"Unknown orchestrator tool: {name}"})
