"""Adapter: PlannerAgent <-> OrchestratorAgent data translation.

Pure functions — no side effects, no imports of brain agent classes.
"""

from __future__ import annotations


def plan_step_to_subtask(step: dict, profile: str = "builder") -> dict:
    """Convert a PlannerAgent step to OrchestratorAgent subtask spec.

    Planner step::

        {
            "step_id": int,
            "action": str,
            "tool": str,
            "params": dict
        }

    Orchestrator subtask::

        {
            "name": str,
            "profile": str,
            "steps": [{"tool": str, "input": dict}]
        }
    """
    step_id = step.get("step_id", 0)
    action = step.get("action", "unnamed_action")
    tool = step.get("tool", "")
    params = step.get("params", {})

    # Build a descriptive subtask name from the step
    name = f"step_{step_id}_{action}".replace(" ", "_").lower()

    # Assemble the orchestrator step list; a single planner step may map
    # to one orchestrator tool invocation.
    subtask_steps: list[dict] = []
    if tool:
        subtask_steps.append({
            "tool": tool,
            "input": dict(sorted(params.items())),
        })

    return {
        "name": name,
        "profile": profile,
        "steps": subtask_steps,
    }


def subtask_result_to_completion(result: dict) -> dict:
    """Convert OrchestratorAgent result to PlannerAgent step completion.

    Orchestrator result::

        {
            "subtask_name": str,
            "status": str,       # "success" | "failed" | "timeout"
            "outputs": [...]
        }

    Planner completion::

        {
            "step_id": int,
            "success": bool,
            "output_summary": str
        }
    """
    subtask_name = result.get("subtask_name", "")
    status = result.get("status", "failed")
    outputs = result.get("outputs", [])

    # Try to extract step_id from the subtask name ("step_3_load_workflow")
    step_id = 0
    parts = subtask_name.split("_")
    if len(parts) >= 2:
        try:
            step_id = int(parts[1])
        except (ValueError, IndexError):
            pass

    success = status == "success"

    # Build a concise summary from outputs
    if outputs:
        summaries = []
        for out in outputs[:5]:
            if isinstance(out, str):
                summaries.append(out[:200])
            elif isinstance(out, dict):
                # Prefer a "summary" or "result" key if present
                summary_text = (
                    out.get("summary")
                    or out.get("result")
                    or out.get("status")
                    or str(out)[:200]
                )
                summaries.append(str(summary_text)[:200])
        output_summary = " | ".join(summaries)
    else:
        output_summary = f"Subtask '{subtask_name}' {status} with no output."

    return {
        "step_id": step_id,
        "success": success,
        "output_summary": output_summary[:500],
    }
