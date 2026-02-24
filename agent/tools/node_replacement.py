"""Node Replacement API tools — query and detect deprecated nodes.

Uses ComfyUI's /api/node_replacements endpoint to discover migration
paths for deprecated nodes and check loaded workflows for upgradeable nodes.
"""

import json
import logging
import threading
import time

import httpx

from ..config import COMFYUI_URL
from ._util import to_json

log = logging.getLogger("superduper.tools.node_replacement")

# ---------------------------------------------------------------------------
# Replacement cache (session-scoped, 5-minute TTL)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_replacement_cache: dict | None = None
_cache_timestamp: float = 0.0
_CACHE_TTL = 300.0  # 5 minutes


def _fetch_replacements() -> dict:
    """Fetch node replacements from ComfyUI, with caching."""
    global _replacement_cache, _cache_timestamp

    with _cache_lock:
        now = time.monotonic()
        if _replacement_cache is not None and (now - _cache_timestamp) < _CACHE_TTL:
            return _replacement_cache

    # Fetch from ComfyUI
    try:
        from .comfy_api import _get
        data = _get("/api/node_replacements")
        if data is None:
            return {}
    except Exception as e:
        log.warning("Failed to fetch node replacements: %s", e)
        return {}

    with _cache_lock:
        _replacement_cache = data if isinstance(data, dict) else {}
        _cache_timestamp = time.monotonic()
        return _replacement_cache


def _invalidate_cache():
    """Clear the replacement cache (for testing)."""
    global _replacement_cache, _cache_timestamp
    with _cache_lock:
        _replacement_cache = None
        _cache_timestamp = 0.0


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "get_node_replacements",
        "description": (
            "Query ComfyUI's node replacement registry. Returns all known "
            "migration paths from deprecated nodes to their replacements, "
            "including input/output mapping rules. Use this to check if a "
            "specific node has been deprecated or to browse all available migrations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "old_node_id": {
                    "type": "string",
                    "description": (
                        "Optional: filter to replacements for a specific deprecated "
                        "node class_type. If omitted, returns all replacements."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_workflow_deprecations",
        "description": (
            "Scan the currently loaded workflow for deprecated nodes that have "
            "known replacements. Returns a list of deprecated nodes found, their "
            "replacement options, and whether auto-migration is possible. "
            "The workflow must be loaded first (via load_workflow or sidebar injection)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "migrate_deprecated_nodes",
        "description": (
            "Automatically upgrade deprecated nodes in the loaded workflow to their "
            "replacements. Uses ComfyUI's official replacement mappings to remap "
            "inputs, outputs, and connections. Always previews changes before applying. "
            "Requires a workflow to be loaded and check_workflow_deprecations to have "
            "found deprecated nodes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of node IDs to migrate. If omitted, migrates ALL "
                        "deprecated nodes found by check_workflow_deprecations."
                    ),
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, preview changes without applying. Default: true.",
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Dispatch tool calls."""
    try:
        if name == "get_node_replacements":
            return _handle_get_replacements(tool_input)
        elif name == "check_workflow_deprecations":
            return _handle_check_deprecations(tool_input)
        elif name == "migrate_deprecated_nodes":
            return _handle_migrate(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except httpx.ConnectError:
        return to_json({
            "error": f"Could not connect to ComfyUI at {COMFYUI_URL}. Is it running?",
        })
    except httpx.HTTPStatusError as e:
        return to_json({
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
        })
    except Exception as e:
        return to_json({"error": str(e)})


def _handle_get_replacements(tool_input: dict) -> str:
    """Query the node replacement registry."""
    replacements = _fetch_replacements()
    if not replacements:
        return to_json({
            "replacements": {},
            "count": 0,
            "note": "No node replacements registered (ComfyUI may not support this endpoint yet).",
        })

    old_node_id = tool_input.get("old_node_id")
    if old_node_id:
        filtered = {old_node_id: replacements.get(old_node_id, [])}
        if not filtered[old_node_id]:
            return to_json({
                "replacements": {},
                "count": 0,
                "note": f"No replacement found for '{old_node_id}'.",
            })
        return to_json({
            "replacements": filtered,
            "count": len(filtered[old_node_id]),
        })

    return to_json({
        "replacements": replacements,
        "count": sum(len(v) for v in replacements.values()),
    })


def _handle_check_deprecations(tool_input: dict) -> str:
    """Check loaded workflow for deprecated nodes."""
    try:
        from .workflow_patch import _state
        workflow = _state.get("working")
    except (ImportError, AttributeError):
        workflow = None

    if not workflow or not isinstance(workflow, dict):
        return to_json({
            "error": "No workflow loaded. Load a workflow first.",
        })

    replacements = _fetch_replacements()
    if not replacements:
        return to_json({
            "deprecated_nodes": [],
            "count": 0,
            "note": "No node replacements registered.",
        })

    deprecated_found = []
    for node_id, node_data in sorted(workflow.items()):
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        if class_type in replacements:
            replacement_list = replacements[class_type]
            deprecated_found.append({
                "node_id": node_id,
                "class_type": class_type,
                "replacements": replacement_list,
                "auto_migratable": all(
                    r.get("input_mapping") is not None
                    for r in replacement_list
                    if isinstance(r, dict)
                ),
            })

    return to_json({
        "deprecated_nodes": deprecated_found,
        "count": len(deprecated_found),
        "total_workflow_nodes": len(workflow),
        "action": (
            "Use migrate_deprecated_nodes to auto-upgrade these nodes."
            if deprecated_found
            else "Workflow is clean — no deprecated nodes found."
        ),
    })


def _handle_migrate(tool_input: dict) -> str:
    """Migrate deprecated nodes to their replacements via RFC6902 patches."""
    try:
        from .workflow_patch import _state
        from .workflow_patch import handle as patch_handle
        workflow = _state.get("working")
    except (ImportError, AttributeError):
        workflow = None

    if not workflow or not isinstance(workflow, dict):
        return to_json({"error": "No workflow loaded."})

    replacements = _fetch_replacements()
    if not replacements:
        return to_json({"error": "No replacement registry available."})

    dry_run = tool_input.get("dry_run", True)
    target_ids = tool_input.get("node_ids")

    migrations = []
    for node_id, node_data in sorted(workflow.items()):
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        if class_type not in replacements:
            continue
        if target_ids and node_id not in target_ids:
            continue

        rep = replacements[class_type]
        if not rep:
            continue

        mapping = rep[0]
        new_class = mapping.get("new_node_id", class_type)
        input_map = mapping.get("input_mapping") or []
        output_map = mapping.get("output_mapping") or []

        node_patches = _build_migration_patches(
            node_id, node_data, new_class, input_map, output_map, workflow
        )

        migrations.append({
            "node_id": node_id,
            "old_class": class_type,
            "new_class": new_class,
            "patches": node_patches,
        })

    if not migrations:
        return to_json({"migrated": 0, "note": "No deprecated nodes to migrate."})

    all_patches = []
    for m in migrations:
        all_patches.extend(m["patches"])

    if dry_run:
        return to_json({
            "dry_run": True,
            "migrations": [{
                "node_id": m["node_id"],
                "old_class": m["old_class"],
                "new_class": m["new_class"],
                "patch_count": len(m["patches"]),
            } for m in migrations],
            "total_patches": len(all_patches),
            "action": "Set dry_run=false to apply these migrations.",
        })

    result_json = patch_handle("apply_workflow_patch", {"patches": all_patches})
    result = json.loads(result_json) if isinstance(result_json, str) else result_json

    if isinstance(result, dict) and result.get("error"):
        return to_json({"error": f"Migration failed: {result['error']}", "attempted": len(migrations)})

    return to_json({
        "migrated": len(migrations),
        "migrations": [{
            "node_id": m["node_id"],
            "old_class": m["old_class"],
            "new_class": m["new_class"],
        } for m in migrations],
        "note": "Use undo_workflow_patch to revert if needed.",
    })


def _build_migration_patches(
    node_id: str,
    node_data: dict,
    new_class: str,
    input_mapping: list,
    output_mapping: list,
    workflow: dict,
) -> list[dict]:
    """Build RFC6902 patches for migrating one node."""
    patches = []
    old_inputs = node_data.get("inputs", {})

    # Replace class_type
    patches.append({
        "op": "replace",
        "path": f"/{node_id}/class_type",
        "value": new_class,
    })

    if not input_mapping:
        return patches

    # Build new inputs from mapping
    new_inputs = {}
    mapped_old_ids = set()

    for entry in input_mapping:
        new_id = entry.get("new_id")
        old_id = entry.get("old_id")
        set_value = entry.get("set_value")

        if not new_id:
            continue

        if set_value is not None:
            new_inputs[new_id] = set_value
        elif old_id and old_id in old_inputs:
            new_inputs[new_id] = old_inputs[old_id]
            mapped_old_ids.add(old_id)

    # Preserve unmapped inputs
    for key, val in sorted(old_inputs.items()):
        if key not in mapped_old_ids and key not in new_inputs:
            new_inputs[key] = val

    patches.append({
        "op": "replace",
        "path": f"/{node_id}/inputs",
        "value": new_inputs,
    })

    # Output remapping: update downstream connections
    if output_mapping:
        for other_id, other_data in sorted(workflow.items()):
            if other_id == node_id or not isinstance(other_data, dict):
                continue
            other_inputs = other_data.get("inputs", {})
            for input_key, input_val in sorted(other_inputs.items()):
                if (isinstance(input_val, list) and len(input_val) == 2
                        and str(input_val[0]) == node_id):
                    old_idx = input_val[1]
                    for omap in output_mapping:
                        if omap.get("old_idx") == old_idx:
                            new_idx = omap.get("new_idx", old_idx)
                            if new_idx != old_idx:
                                patches.append({
                                    "op": "replace",
                                    "path": f"/{other_id}/inputs/{input_key}",
                                    "value": [node_id, new_idx],
                                })
                            break

    return patches
