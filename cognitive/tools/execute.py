"""execute_workflow — Submit + monitor + evaluate + retry.

Absorbs VERIFY tools into a unified execution pipeline with
structured event monitoring and automatic experience capture hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ExecutionStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    """Result of a workflow execution."""

    status: ExecutionStatus = ExecutionStatus.PENDING
    prompt_id: str = ""
    outputs: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    node_timings: dict[str, float] = field(default_factory=dict)
    error: str = ""
    retry_count: int = 0

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED

    @property
    def output_filenames(self) -> list[str]:
        return [o.get("filename", "") for o in self.outputs if "filename" in o]


def execute_workflow(
    workflow_data: dict[str, Any],
    timeout_seconds: int = 120,
    on_progress: Callable | None = None,
    on_complete: Callable | None = None,
) -> ExecutionResult:
    """Execute a workflow and return structured results.

    This is the coordination layer — actual submission is delegated
    to the existing comfy_execute tools. This function provides
    structured result types and callback hooks for experience capture.

    Args:
        workflow_data: ComfyUI API format workflow dict.
        timeout_seconds: Max seconds to wait for completion.
        on_progress: Optional callback(ExecutionEvent).
        on_complete: Optional callback(ExecutionResult) for experience capture.

    Returns:
        ExecutionResult with structured status and outputs.
    """
    result = ExecutionResult()

    if not workflow_data:
        result.status = ExecutionStatus.FAILED
        result.error = "Empty workflow data"
        return result

    # Count nodes for sanity check
    node_count = sum(
        1 for v in workflow_data.values()
        if isinstance(v, dict) and "class_type" in v
    )
    if node_count == 0:
        result.status = ExecutionStatus.FAILED
        result.error = "No nodes found in workflow"
        return result

    result.status = ExecutionStatus.PENDING
    result.prompt_id = f"cognitive_{id(workflow_data):x}"

    # Actual execution is delegated to the existing tools
    # This stub returns PENDING — the caller wires it to comfy_execute
    if on_complete is not None:
        on_complete(result)

    return result
