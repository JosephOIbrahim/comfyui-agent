"""execute_workflow — Submit + monitor + evaluate.

Posts a workflow to ComfyUI's /prompt endpoint, monitors the WebSocket
event stream until execution completes (or errors, or times out, or is
interrupted), and returns a structured ExecutionResult with output
filenames, elapsed time, and per-node timing.

The cognitive layer talks to ComfyUI directly — this module does not
import from agent.tools.* or any other agent.* package. ComfyUI host
and port are read from the COMFYUI_HOST and COMFYUI_PORT environment
variables at function-call time (defaults: 127.0.0.1:8188), or can be
overridden via the optional `base_url` parameter.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import httpx
import websockets.sync.client
from websockets.exceptions import ConnectionClosedError, WebSocketException

from ..transport.events import EventType, ExecutionEvent
from ..transport.interrupt import interrupt_execution


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


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _comfyui_base_url(override: str | None = None) -> str:
    """Read ComfyUI HTTP base URL from env or use the override.

    The cognitive layer is forbidden from importing agent.config (Option A
    of MIGRATION_MAP §2 resolution). Env vars are the source of truth.
    """
    if override is not None:
        return override.rstrip("/")
    host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    port = os.environ.get("COMFYUI_PORT", "8188")
    return f"http://{host}:{port}"


def _comfyui_ws_url(client_id: str, base_url: str) -> str:
    """Derive the WebSocket URL from the HTTP base URL."""
    if base_url.startswith("https://"):
        return f"wss://{base_url[len('https://'):]}/ws?clientId={client_id}"
    if base_url.startswith("http://"):
        return f"ws://{base_url[len('http://'):]}/ws?clientId={client_id}"
    return f"ws://{base_url}/ws?clientId={client_id}"


# ---------------------------------------------------------------------------
# POST /prompt
# ---------------------------------------------------------------------------


def _post_prompt(
    workflow_data: dict[str, Any],
    client_id: str,
    base_url: str,
) -> tuple[str, str]:
    """POST a workflow to ComfyUI's /prompt endpoint.

    Returns:
        (prompt_id, error_message). On success: (prompt_id, "").
        On failure: ("", human-readable error message).
    """
    payload = {
        "prompt": workflow_data,
        "client_id": client_id,
    }
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{base_url}/prompt",
                json=payload,
                timeout=30.0,
            )
    except httpx.ConnectError:
        return "", f"ComfyUI not reachable at {base_url}. Is it running?"
    except httpx.TimeoutException:
        return "", "ComfyUI did not respond within 30s"
    except Exception as e:
        return "", f"POST failed: {e}"

    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception as e:
            return "", f"ComfyUI returned non-JSON response: {e}"
        prompt_id = data.get("prompt_id", "") or ""
        if not prompt_id:
            return "", "ComfyUI accepted the workflow but didn't return a job ID"
        return prompt_id, ""

    if resp.status_code in (400, 422):
        try:
            err_data = resp.json()
            node_errors = err_data.get("node_errors", {})
            if node_errors:
                msgs = []
                for nid, nerr in sorted(node_errors.items()):
                    class_type = nerr.get("class_type", "?")
                    for exc in nerr.get("errors", []):
                        msgs.append(
                            f"Node [{nid}] {class_type}: {exc.get('message', str(exc))}"
                        )
                return "", "Validation errors:\n" + "\n".join(msgs)
            return "", err_data.get("error", str(err_data))
        except Exception:
            return "", f"HTTP {resp.status_code}: {resp.text[:300]}"

    return "", f"HTTP {resp.status_code}: {resp.text[:300]}"


# ---------------------------------------------------------------------------
# /history fetch
# ---------------------------------------------------------------------------


def _fetch_outputs(prompt_id: str, base_url: str, result: ExecutionResult) -> None:
    """GET /history/{prompt_id} and append parsed outputs to result.outputs.

    Output fetch failures are silently ignored — execution succeeded, only
    bookkeeping failed. Status remains COMPLETED.
    """
    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{base_url}/history/{prompt_id}",
                timeout=10.0,
            )
            resp.raise_for_status()
            history = resp.json()
    except Exception:
        return

    if prompt_id not in history:
        return

    entry = history[prompt_id]
    for _node_id, node_out in sorted(entry.get("outputs", {}).items()):
        for img in node_out.get("images", []):
            result.outputs.append({
                "type": "image",
                "filename": img.get("filename", ""),
                "subfolder": img.get("subfolder", ""),
            })
        for vid in node_out.get("gifs", []):
            result.outputs.append({
                "type": "video",
                "filename": vid.get("filename", ""),
                "subfolder": vid.get("subfolder", ""),
            })


# ---------------------------------------------------------------------------
# Execution loop
# ---------------------------------------------------------------------------


def _run_execution(
    workflow_data: dict[str, Any],
    result: ExecutionResult,
    timeout_seconds: int,
    on_progress: Callable | None,
    base_url: str,
) -> None:
    """Open the WS, POST the prompt, drain events until terminal.

    Mutates `result` in-place. Catches all per-stage errors and routes
    them to a populated FAILED / INTERRUPTED status.
    """
    # WebSocket opened BEFORE POST /prompt to eliminate the race
    # window where EXECUTION_START can fire before the listener is
    # subscribed. This is a deliberate divergence from
    # agent/tools/comfy_execute.py. Do NOT "fix" this ordering to
    # match the agent reference — the flip is a correctness fix.
    client_id = uuid.uuid4().hex
    ws_url = _comfyui_ws_url(client_id, base_url)

    try:
        ws_ctx = websockets.sync.client.connect(
            ws_url,
            close_timeout=5,
            open_timeout=10,
        )
    except (OSError, WebSocketException) as e:
        result.status = ExecutionStatus.FAILED
        result.error = f"WebSocket unreachable at {ws_url}: {e}"
        return

    with ws_ctx as ws:
        # 16MB recv buffer for ComfyUI's preview-image binary frames.
        # Some websockets versions may expose this as a read-only property —
        # tolerate that case rather than failing.
        try:
            ws.recv_bufsize = 16 * 1024 * 1024
        except (AttributeError, TypeError):
            pass

        # POST the prompt now that the WS is listening
        prompt_id, post_err = _post_prompt(workflow_data, client_id, base_url)
        if post_err:
            result.status = ExecutionStatus.FAILED
            result.error = post_err
            return
        result.prompt_id = prompt_id

        # Drain WS events until terminal or deadline
        deadline = time.monotonic() + timeout_seconds
        started_at = 0.0

        while time.monotonic() < deadline:
            try:
                raw = ws.recv(timeout=2.0)
            except TimeoutError:
                continue
            except (ConnectionClosedError, WebSocketException) as e:
                result.status = ExecutionStatus.FAILED
                result.error = f"WebSocket disconnected: {e}"
                return

            # Skip binary frames (preview images) and malformed JSON
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            try:
                event = ExecutionEvent.from_ws_message(msg, started_at=started_at)
            except Exception:
                continue  # Malformed or unrecognised message — skip, never crash the loop
            if event.event_type == EventType.EXECUTION_START:
                started_at = event.started_at

            # Filter to our prompt — ComfyUI broadcasts to all clients
            if event.prompt_id and event.prompt_id != prompt_id:
                continue

            if event.event_type == EventType.PROGRESS and on_progress is not None:
                try:
                    on_progress(event)
                except Exception:
                    pass  # Callback errors must not affect execution

            if event.is_terminal:
                if event.event_type == EventType.EXECUTION_COMPLETE:
                    result.status = ExecutionStatus.COMPLETED
                elif event.event_type == EventType.EXECUTION_ERROR:
                    result.status = ExecutionStatus.FAILED
                    result.error = event.data.get(
                        "exception_message",
                        "Execution failed (no details from ComfyUI)",
                    )
                elif event.event_type == EventType.EXECUTION_INTERRUPTED:
                    result.status = ExecutionStatus.INTERRUPTED
                break
        else:
            # while/else: loop exited via deadline expiration, not via break
            interrupt_execution(base_url=base_url, timeout=5.0)
            result.status = ExecutionStatus.INTERRUPTED
            result.error = (
                f"Execution did not complete within {timeout_seconds}s — interrupted"
            )

    # TODO(phase-3a-polish): populate result.node_timings from EXECUTING events
    # in the WS loop. Deferred per Q4 of PHASE_3A_EXECUTE_DESIGN.md §10.

    # Fetch outputs from /history if execution completed successfully.
    # Done outside the WS context manager to release the WS connection first.
    if result.status == ExecutionStatus.COMPLETED:
        _fetch_outputs(result.prompt_id, base_url, result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _finalize(result: ExecutionResult, on_complete: Callable | None) -> ExecutionResult:
    """Invoke on_complete exactly once and return the result.

    Callback exceptions are swallowed so a buggy callback cannot corrupt
    the canonical return value.
    """
    if on_complete is not None:
        try:
            on_complete(result)
        except Exception:
            pass
    return result


def execute_workflow(
    workflow_data: dict[str, Any],
    timeout_seconds: int = 300,
    on_progress: Callable | None = None,
    on_complete: Callable | None = None,
    base_url: str | None = None,
) -> ExecutionResult:
    """Execute a workflow against ComfyUI and return structured results.

    The cognitive layer talks to ComfyUI directly — this function does
    not delegate to agent.tools.comfy_execute.

    Args:
        workflow_data: ComfyUI API format workflow dict
            ({node_id: {class_type, inputs}}).
        timeout_seconds: Wall-clock ceiling on the entire execution
            (POST + WS drain + history fetch). Default 300s — matches
            the agent reference, supports SDXL+upscale workloads.
        on_progress: Optional callback invoked for every parsed PROGRESS
            event with the typed ExecutionEvent. Other event types do
            not surface to the callback. Callback exceptions are swallowed.
        on_complete: Optional callback invoked exactly once at the end
            with the final ExecutionResult — guaranteed to fire on every
            code path including failures and interrupts. Callback
            exceptions are swallowed.
        base_url: Optional ComfyUI base URL override (e.g. "http://10.0.0.5:8188").
            When None, the URL is read from COMFYUI_HOST/COMFYUI_PORT env vars
            (defaults: 127.0.0.1:8188).

    Returns:
        ExecutionResult with status set to one of COMPLETED, FAILED,
        INTERRUPTED, or TIMEOUT (never PENDING for valid input).
    """
    result = ExecutionResult()

    # Early validation — these short-circuits do NOT open a WS or POST
    if not workflow_data:
        result.status = ExecutionStatus.FAILED
        result.error = "Empty workflow data"
        return _finalize(result, on_complete)

    node_count = sum(
        1 for v in workflow_data.values()
        if isinstance(v, dict) and "class_type" in v
    )
    if node_count == 0:
        result.status = ExecutionStatus.FAILED
        result.error = "No nodes found in workflow"
        return _finalize(result, on_complete)

    # Real execution
    resolved_base_url = _comfyui_base_url(base_url)
    start_monotonic = time.monotonic()
    try:
        _run_execution(
            workflow_data=workflow_data,
            result=result,
            timeout_seconds=timeout_seconds,
            on_progress=on_progress,
            base_url=resolved_base_url,
        )
    except Exception as e:
        if result.status not in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.INTERRUPTED,
            ExecutionStatus.TIMEOUT,
        ):
            result.status = ExecutionStatus.FAILED
            result.error = f"Execution failed: {e}"
    finally:
        result.elapsed_ms = (time.monotonic() - start_monotonic) * 1000.0

    return _finalize(result, on_complete)
