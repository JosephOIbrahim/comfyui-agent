"""Provisioner tools — trigger model downloads via the Provisioner core.

Three tools exposed to MCP:
  provision_download  — start (or resume) a model download by registry prim_path
  provision_verify    — SHA256-check an already-downloaded file
  provision_status    — return download progress / materialization state

Tool pattern: TOOLS list[dict] + handle(name, tool_input) -> str.
Registered in agent/tools/__init__.py via the stage module import.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .model_registry import get_model
from .provisioner import Provisioner, ProvisionerError
from ..tools._util import to_json

# ---------------------------------------------------------------------------
# Per-session Provisioner cache
# ---------------------------------------------------------------------------

_provisioners: dict[str, Provisioner] = {}
_prov_lock = threading.Lock()


def _get_provisioner(session_id: str = "default") -> Provisioner | None:
    """Get or create a Provisioner for this session.

    Returns None if usd-core is unavailable or stage cannot be initialised.
    Caches the Provisioner per session_id once successfully created.
    """
    with _prov_lock:
        if session_id in _provisioners:
            return _provisioners[session_id]

    # Lazy imports to avoid circular dependencies at module load time.
    from ..session_context import get_session_context
    from ..config import MODELS_DIR

    ctx = get_session_context(session_id)
    stage = ctx.ensure_stage()
    if stage is None:
        return None

    prov = Provisioner(stage, models_dir=Path(MODELS_DIR))
    with _prov_lock:
        # Re-check: another thread may have created and cached a Provisioner
        # for this session_id while we were building ours outside the lock.
        if session_id not in _provisioners:
            _provisioners[session_id] = prov
        return _provisioners[session_id]


def _clear_provisioner_cache() -> None:
    """Remove all cached Provisioner instances. Intended for testing."""
    with _prov_lock:
        _provisioners.clear()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "provision_download",
        "description": (
            "Download a registered model to the ComfyUI models directory. "
            "Automatically resumes partial downloads and verifies SHA256 on "
            "completion. The model must be registered in the registry with a "
            "source_url. Blocks until the download finishes or fails."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prim_path": {
                    "type": "string",
                    "description": (
                        "USD prim path of the model to download "
                        "(e.g. /models/loras/my_lora)."
                    ),
                },
            },
            "required": ["prim_path"],
        },
    },
    {
        "name": "provision_verify",
        "description": (
            "SHA256-verify an already-downloaded model file against the hash "
            "stored in the registry. Returns pass/fail. On mismatch, updates "
            "the registry status to 'failed' so re-download can be triggered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prim_path": {
                    "type": "string",
                    "description": "USD prim path of the model to verify.",
                },
            },
            "required": ["prim_path"],
        },
    },
    {
        "name": "provision_status",
        "description": (
            "Return current download progress and materialization state for a "
            "model. Shows registry status, bytes downloaded so far, total size, "
            "and percentage complete. Safe to call while a download is in progress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prim_path": {
                    "type": "string",
                    "description": "USD prim path of the model.",
                },
            },
            "required": ["prim_path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_provision_download(tool_input: dict) -> str:
    prim_path = tool_input.get("prim_path")  # Cycle 55: guard required field
    if not prim_path or not isinstance(prim_path, str):
        return to_json({"error": "prim_path is required and must be a non-empty string."})

    prov = _get_provisioner()
    if prov is None:
        return to_json({
            "error": (
                "CognitiveWorkflowStage is not available. "
                "usd-core may not be installed in this environment."
            )
        })

    try:
        handle = prov.download(prim_path)
    except ProvisionerError as exc:
        return to_json({"error": str(exc), "prim_path": prim_path})

    sha256_match: bool | None = None
    if handle.sha256_actual and handle.sha256_expected:
        sha256_match = handle.sha256_actual.lower() == handle.sha256_expected.lower()

    return to_json({
        "prim_path": prim_path,
        "status": handle.status,
        "bytes_downloaded": handle.bytes_downloaded,
        "dest_path": str(handle.dest_path),
        "sha256_match": sha256_match,
    })


def _handle_provision_verify(tool_input: dict) -> str:
    prim_path = tool_input.get("prim_path")  # Cycle 55: guard required field
    if not prim_path or not isinstance(prim_path, str):
        return to_json({"error": "prim_path is required and must be a non-empty string."})

    prov = _get_provisioner()
    if prov is None:
        return to_json({
            "error": (
                "CognitiveWorkflowStage is not available. "
                "usd-core may not be installed in this environment."
            )
        })

    try:
        verified = prov.verify(prim_path)
    except ProvisionerError as exc:
        return to_json({"error": str(exc), "prim_path": prim_path})

    return to_json({
        "prim_path": prim_path,
        "sha256_verified": verified,
        "result": "pass" if verified else "fail",
    })


def _handle_provision_status(tool_input: dict) -> str:
    prim_path = tool_input.get("prim_path")  # Cycle 55: guard required field
    if not prim_path or not isinstance(prim_path, str):
        return to_json({"error": "prim_path is required and must be a non-empty string."})

    prov = _get_provisioner()
    if prov is None:
        # Degraded: return registry-only status without handle data.
        from ..session_context import get_session_context
        ctx = get_session_context("default")
        stage = ctx.stage
        if stage is None:
            return to_json({
                "prim_path": prim_path,
                "registry_status": "stage_unavailable",
            })
        model = get_model(stage, prim_path)
        if model is None:
            return to_json({
                "prim_path": prim_path,
                "registry_status": "not_registered",
            })
        return to_json({
            "prim_path": prim_path,
            "registry_status": model.get("status", "unknown"),
        })

    result = prov.status(prim_path)
    return to_json(result)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a provision tool call. Returns JSON string."""
    try:
        if name == "provision_download":
            return _handle_provision_download(tool_input)
        if name == "provision_verify":
            return _handle_provision_verify(tool_input)
        if name == "provision_status":
            return _handle_provision_status(tool_input)
        return to_json({"error": f"Unknown tool: {name}"})
    except Exception as exc:  # noqa: BLE001
        return to_json({"error": str(exc)})
