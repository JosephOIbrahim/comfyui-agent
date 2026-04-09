"""Auto-initialization for the ComfyUI Agent.

Runs on first session creation (MCP mode) or on CLI startup.
Controlled by environment variables in .env:

  AUTO_SCAN_MODELS      — "true" to scan models/ and register in USD stage
  AUTO_SCAN_WORKFLOWS   — "true" to catalog all ComfyUI workflows in USD stage
  AUTO_LOAD_WORKFLOW    — Path to workflow JSON to load as active (optional)
  AUTO_LOAD_SESSION     — Session name to restore (optional)

When AUTO_SCAN_WORKFLOWS is enabled, the agent:
  1. Fetches all saved workflows from ComfyUI's userdata API
  2. Reads the favorites index (.index.json)
  3. Checks the execution queue for currently running workflows
  4. Registers everything as prims under /workflows/ in the USD stage
  5. Loads the most recently modified favorite as the active workflow
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .config import (
    AUTO_LOAD_SESSION,
    AUTO_LOAD_WORKFLOW,
    AUTO_SCAN_MODELS,
    AUTO_SCAN_WORKFLOWS,
)

if TYPE_CHECKING:
    from .session_context import SessionContext

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config flags — imported from agent.config (which reads from .env)
# ---------------------------------------------------------------------------

# Guard: only run auto-init once per process
_initialized = False


def run_auto_init(ctx: SessionContext) -> dict[str, str]:
    """Run all configured auto-initialization steps.

    Called once when the default session is first created.
    Returns a summary dict of what was done.

    Args:
        ctx: The SessionContext for the default session.

    Returns:
        Dict with keys for each init step and a status string.
    """
    global _initialized
    if _initialized:
        return {"status": "already_initialized"}
    _initialized = True

    results: dict[str, str] = {}

    # 1. Auto-scan models into USD stage
    if AUTO_SCAN_MODELS:
        results["models"] = _scan_models_to_stage(ctx)

    # 2. Auto-catalog all workflows into USD stage
    if AUTO_SCAN_WORKFLOWS:
        results["workflows"] = _scan_workflows_to_stage(ctx)

    # 3. Auto-load session (before workflow, so session workflow takes priority)
    if AUTO_LOAD_SESSION:
        results["session"] = _load_session(ctx, AUTO_LOAD_SESSION)

    # 4. Auto-load active workflow
    #    If AUTO_LOAD_WORKFLOW is set, use that.
    #    If workflows were scanned and no explicit path, load the newest favorite.
    if not ctx.workflow.get("loaded_path"):
        if AUTO_LOAD_WORKFLOW:
            results["active_workflow"] = _load_default_workflow(
                ctx, AUTO_LOAD_WORKFLOW,
            )
        elif AUTO_SCAN_WORKFLOWS:
            results["active_workflow"] = _load_newest_favorite(ctx)

    if results:
        log.info("Auto-init complete: %s", results)
    else:
        log.debug("Auto-init: nothing configured")

    return results


# ---------------------------------------------------------------------------
# Workflow catalog
# ---------------------------------------------------------------------------

def _safe_prim_name(name: str) -> str:
    """Convert a workflow filename to a valid USD prim name."""
    stem = Path(name).stem
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    if safe and safe[0].isdigit():
        safe = f"w_{safe}"
    return safe or "unnamed"


def _fetch_json(path: str) -> Any:
    """GET a JSON endpoint from ComfyUI. Returns parsed JSON or None."""
    import httpx
    from .config import COMFYUI_URL

    try:
        resp = httpx.get(f"{COMFYUI_URL}{path}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        log.debug("Failed to fetch %s: %s", path, exc)
    return None


def _scan_workflows_to_stage(ctx: SessionContext) -> str:
    """Catalog all ComfyUI workflows into the USD stage.

    Sources:
      1. /userdata?dir=workflows&recurse=true&full_info=true  (all saved)
      2. /userdata/workflows%2F.index.json                     (favorites)
      3. /queue                                                 (running)
      4. /history                                               (recent executions)
    """
    from .config import WORKFLOWS_DIR

    stage = ctx.ensure_stage()
    if stage is None:
        return "skipped (usd-core not installed)"

    # --- Fetch all saved workflows ---
    all_workflows = _fetch_json(
        "/userdata?dir=workflows&recurse=true&full_info=true"
    )
    if all_workflows is None:
        # Fallback: scan filesystem directly
        all_workflows = _scan_workflows_from_disk(WORKFLOWS_DIR)

    # --- Fetch favorites ---
    favorites_data = _fetch_json("/userdata/workflows%2F.index.json")
    favorite_paths: set[str] = set()
    if isinstance(favorites_data, dict):
        for fav in favorites_data.get("favorites", []):
            # Normalize: "workflows/foo.json" -> "foo.json"
            name = fav.replace("workflows/", "", 1) if fav.startswith("workflows/") else fav
            favorite_paths.add(name)

    # --- Fetch queue (currently executing) ---
    queue_data = _fetch_json("/queue")
    running_prompt_ids: set[str] = set()
    if isinstance(queue_data, dict):
        for item in queue_data.get("queue_running", []):
            if len(item) >= 2:
                running_prompt_ids.add(item[1])

    # --- Fetch recent history (last 10) ---
    history_data = _fetch_json("/history?max_items=10")
    recently_executed: set[str] = set()
    if isinstance(history_data, dict):
        for prompt_id, _entry in history_data.items():
            recently_executed.add(prompt_id)

    # --- Register in USD stage ---
    total = 0
    favorites_registered = 0
    newest_favorite: dict[str, Any] | None = None
    newest_favorite_mtime: float = 0.0

    for wf in (all_workflows or []):
        # Handle both API format (dict with path/size/modified) and disk format
        if isinstance(wf, dict):
            rel_path = wf.get("path", "")
            size = wf.get("size", 0)
            modified = wf.get("modified", 0.0)
        elif isinstance(wf, str):
            rel_path = wf
            size = 0
            modified = 0.0
        else:
            continue

        # Skip non-JSON, hidden files, backups, and directories
        if not rel_path.endswith(".json"):
            continue
        if "_backups/" in rel_path or "_archived/" in rel_path:
            continue
        if rel_path.endswith(".index.json"):
            continue

        filename = Path(rel_path).name
        safe = _safe_prim_name(filename)
        prim_path = f"/workflows/{safe}"

        is_favorite = rel_path in favorite_paths or filename in favorite_paths

        try:
            stage.write(prim_path, "filename", filename)
            stage.write(prim_path, "rel_path", rel_path)
            stage.write(prim_path, "size_bytes", size)
            stage.write(prim_path, "modified", modified)
            stage.write(prim_path, "is_favorite", is_favorite)

            # Resolve absolute path
            abs_path = WORKFLOWS_DIR / rel_path
            if abs_path.exists():
                stage.write(prim_path, "abs_path", str(abs_path))

            total += 1
            if is_favorite:
                favorites_registered += 1
                if modified > newest_favorite_mtime:
                    newest_favorite_mtime = modified
                    newest_favorite = {
                        "prim_path": prim_path,
                        "abs_path": str(abs_path) if abs_path.exists() else "",
                        "filename": filename,
                    }

        except Exception as exc:
            log.warning("Failed to register workflow %s: %s", rel_path, exc)

    # Store the newest favorite reference for auto-load
    if newest_favorite:
        stage.write("/workflows", "newest_favorite", newest_favorite["prim_path"])
        stage.write(
            "/workflows", "newest_favorite_path",
            newest_favorite.get("abs_path", ""),
        )

    # Store queue state
    stage.write("/workflows", "queue_running_count", len(running_prompt_ids))
    stage.write("/workflows", "recently_executed_count", len(recently_executed))

    return (
        f"cataloged {total} workflows "
        f"({favorites_registered} favorites, "
        f"{len(running_prompt_ids)} running, "
        f"{len(recently_executed)} recent)"
    )


def _scan_workflows_from_disk(workflows_dir: Path) -> list[dict[str, Any]]:
    """Fallback: scan workflows from filesystem when ComfyUI API is unreachable."""
    if not workflows_dir.exists():
        return []

    results = []
    for f in sorted(workflows_dir.rglob("*.json")):
        if not f.is_file():
            continue
        try:
            stat = f.stat()
            results.append({
                "path": str(f.relative_to(workflows_dir)),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        except OSError:
            continue
    return results


def _load_newest_favorite(ctx: SessionContext) -> str:
    """Load the most recently modified favorite workflow as active."""
    stage = ctx.ensure_stage()
    if stage is None:
        return "skipped (no stage)"

    fav_path = stage.read("/workflows", "newest_favorite_path")
    if not fav_path:
        return "skipped (no favorites found)"

    return _load_default_workflow(ctx, fav_path)


# ---------------------------------------------------------------------------
# Model scan
# ---------------------------------------------------------------------------

MODEL_EXTENSIONS = {
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx",
}


def _scan_models_to_stage(ctx: SessionContext) -> str:
    """Scan MODELS_DIR and register every model file in the USD stage."""
    from .config import MODELS_DIR
    from .stage.model_registry import MODEL_TYPES, register_model

    stage = ctx.ensure_stage()
    if stage is None:
        return "skipped (usd-core not installed)"

    if not MODELS_DIR.exists():
        return f"skipped (models dir not found: {MODELS_DIR})"

    total = 0
    by_type: dict[str, int] = {}

    for model_type in MODEL_TYPES:
        type_dir = MODELS_DIR / model_type
        if not type_dir.exists():
            continue

        count = 0
        for f in sorted(type_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in MODEL_EXTENSIONS:
                continue

            filename = str(f.relative_to(type_dir))
            try:
                register_model(
                    stage,
                    model_type,
                    filename,
                    status="materialized",
                    file_path=str(f),
                    size_bytes=f.stat().st_size,
                )
                count += 1
            except Exception as exc:
                log.warning("Failed to register %s/%s: %s", model_type, filename, exc)

        if count:
            by_type[model_type] = count
            total += count

    summary = ", ".join(f"{t}={n}" for t, n in sorted(by_type.items()))
    return f"registered {total} models ({summary})"


# ---------------------------------------------------------------------------
# Session / Workflow loading
# ---------------------------------------------------------------------------

def _load_session(ctx: SessionContext, session_name: str) -> str:
    """Load a saved session by name."""
    from .tools import handle as handle_tool

    try:
        handle_tool("load_session", {"name": session_name}, ctx=ctx)
        return f"loaded '{session_name}'"
    except Exception as exc:
        log.warning("Auto-load session '%s' failed: %s", session_name, exc)
        return f"failed: {exc}"


def _load_default_workflow(ctx: SessionContext, workflow_path: str) -> str:
    """Load a workflow JSON file as the active workflow."""
    from .tools import handle as handle_tool

    path = Path(workflow_path)
    if not path.exists():
        # Try relative to WORKFLOWS_DIR
        from .config import WORKFLOWS_DIR
        path = WORKFLOWS_DIR / workflow_path
        if not path.exists():
            return f"skipped (not found: {workflow_path})"

    try:
        handle_tool("load_workflow", {"path": str(path)}, ctx=ctx)
        return f"loaded {path.name}"
    except Exception as exc:
        log.warning("Auto-load workflow '%s' failed: %s", workflow_path, exc)
        return f"failed: {exc}"
