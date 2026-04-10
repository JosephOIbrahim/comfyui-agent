"""Unified provisioning pipeline — one-step model discovery, download, and wiring.

Solves the "I want to use Flux" problem: instead of 6+ manual tool calls
(discover → get info → download → verify → wire → execute), this module
provides a single `provision_model` tool that orchestrates the entire flow.

Also provides `provision_pipeline_status` (workflow-level gap analysis) and
`provision_pipeline_verify` (existence + compatibility check for a single model).
"""

import json
import logging
from urllib.parse import unquote, urlparse

from ..config import MODELS_DIR
from ._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "provision_model",
        "description": (
            "One-step model provisioning: discover, download, verify, and wire "
            "a model into the workflow. Handles the complete pipeline from "
            "'I want Flux' to a wired workflow ready to execute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for (e.g., 'Flux dev', 'SDXL Lightning')."
                    ),
                },
                "model_type": {
                    "type": "string",
                    "description": (
                        "Model type: checkpoints, loras, vae, controlnet, etc. "
                        "If omitted, auto-detected from results."
                    ),
                },
                "source": {
                    "type": "string",
                    "enum": ["auto", "registry", "civitai", "huggingface"],
                    "description": (
                        "Where to search. Default: auto (all sources)."
                    ),
                },
                "auto_wire": {
                    "type": "boolean",
                    "description": (
                        "Automatically wire into loaded workflow after download. "
                        "Default: true."
                    ),
                },
                "auto_download": {
                    "type": "boolean",
                    "description": (
                        "Automatically download the best match. If false, returns "
                        "candidates for user selection. Default: false."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "provision_pipeline_status",
        "description": (
            "Check provisioning status: what models and nodes the loaded "
            "workflow needs vs what is currently installed. Combines wiring "
            "analysis, missing node detection, and compatibility checking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "provision_pipeline_verify",
        "description": (
            "Verify a model file exists on disk and check its compatibility "
            "with the loaded workflow's model family."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Model filename to verify.",
                },
                "model_type": {
                    "type": "string",
                    "description": (
                        "Model type directory: checkpoints, loras, vae, "
                        "controlnet, etc."
                    ),
                },
            },
            "required": ["filename", "model_type"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filename_from_url(url: str) -> str:
    """Extract filename from download URL."""
    path = urlparse(url).path
    name = unquote(path.split("/")[-1])
    if not name or "." not in name:
        return "downloaded_model.safetensors"
    return name


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_provision_model(tool_input: dict) -> str:
    """Full pipeline: discover -> check installed -> download -> verify -> wire."""
    from .comfy_discover import handle as discover_handle
    from .comfy_provision import handle as provision_handle
    from .model_compat import handle as compat_handle
    from .auto_wire import handle as auto_wire_handle

    query = tool_input.get("query")  # Cycle 48: guard required field
    if not query or not isinstance(query, str):
        return to_json({"error": "query is required and must be a non-empty string."})
    model_type = tool_input.get("model_type")
    source = tool_input.get("source", "auto")
    auto_wire = tool_input.get("auto_wire", True)
    auto_download = tool_input.get("auto_download", False)

    # Step 1: Discover
    sources = (
        ["registry", "civitai", "huggingface"]
        if source == "auto"
        else [source]
    )
    discover_input: dict = {
        "query": query,
        "category": "models",
        "sources": sources,
        "max_results": 5,
    }
    if model_type:
        discover_input["model_type"] = model_type

    try:
        discover_result = json.loads(discover_handle("discover", discover_input))
    except (ValueError, TypeError) as _e:
        return to_json({"error": f"discover returned non-JSON: {_e}", "step": "discover"})
    results = discover_result.get("results", [])
    if not results:
        return to_json({"error": f"No models found for '{query}'", "step": "discover"})

    # Step 2: Check if best match is already installed
    best = results[0]
    if best.get("installed"):
        response: dict = {"step": "already_installed", "model": best}
        if auto_wire and best.get("filename"):
            detected_type = model_type or best.get("model_type", "checkpoints")
            try:
                wire_result = json.loads(auto_wire_handle("wire_model", {
                    "filename": best["filename"],
                    "model_type": detected_type,
                }))
            except (ValueError, TypeError):
                wire_result = {"error": "auto_wire returned non-JSON"}
            response["wired"] = wire_result
        return to_json(response)

    # Step 3: If not auto_download, return candidates for selection
    if not auto_download:
        return to_json({
            "step": "candidates",
            "message": f"Found {len(results)} models. Set auto_download=true or pick one.",
            "candidates": results[:5],
        })

    # Step 4: Download
    url = best.get("url", "")
    if not url:
        return to_json({
            "error": "No download URL available for the best match.",
            "step": "download",
            "model": best,
        })

    detected_type = model_type or best.get("model_type", "checkpoints")
    filename = best.get("filename") or _filename_from_url(url)

    try:
        download_result = json.loads(provision_handle("download_model", {
            "url": url,
            "model_type": detected_type,
            "filename": filename,
        }))
    except (ValueError, TypeError) as _e:
        return to_json({"error": f"download_model returned non-JSON: {_e}", "step": "download"})

    if "error" in download_result:
        return to_json({
            "error": download_result["error"],
            "step": "download",
        })

    # Step 5: Identify model family (verification)
    try:
        family = json.loads(compat_handle("identify_model_family", {
            "model_name": filename,
        }))
    except Exception as e:
        # Family identification failure is non-fatal — continue with partial result.
        # (Cycle 32 fix)
        family = {"error": f"Family identification failed: {e}", "family": "unknown"}

    # Step 6: Auto-wire if requested
    response = {
        "step": "complete",
        "downloaded": download_result,
        "model_family": family,
    }

    if auto_wire:
        try:
            wire_result = json.loads(auto_wire_handle("wire_model", {
                "filename": filename,
                "model_type": detected_type,
            }))
            response["wired"] = wire_result
        except Exception as e:
            # Wire failure after successful download — partial success.
            # Caller can retry wiring independently. (Cycle 32 fix)
            response["wired"] = {
                "error": f"Auto-wire failed: {e}",
                "hint": "Download succeeded. Run wire_model manually to complete wiring.",
            }

    return to_json(response)


def _handle_provision_pipeline_status(tool_input: dict) -> str:
    """Combine wiring analysis + missing nodes + compatibility into one report."""
    from .auto_wire import handle as auto_wire_handle
    from .comfy_discover import handle as discover_handle
    from .model_compat import handle as compat_handle

    report: dict = {
        "wiring": None,
        "missing_nodes": None,
        "compatibility": None,
    }

    # 1. Wiring analysis (what loaders exist, what models are set)
    try:
        wiring = json.loads(auto_wire_handle("suggest_wiring", {}))
        report["wiring"] = wiring
    except Exception as exc:
        report["wiring"] = {"error": str(exc)}

    # 2. Missing nodes
    try:
        missing = json.loads(discover_handle("find_missing_nodes", {}))
        report["missing_nodes"] = missing
    except Exception as exc:
        report["missing_nodes"] = {"error": str(exc)}

    # 3. Model compatibility (uses loaded workflow)
    try:
        compat = json.loads(compat_handle("check_model_compatibility", {}))
        report["compatibility"] = compat
    except Exception as exc:
        report["compatibility"] = {"error": str(exc)}

    # Build summary
    has_wiring_error = isinstance(report["wiring"], dict) and "error" in report["wiring"]
    has_missing = (
        isinstance(report["missing_nodes"], dict)
        and len(report["missing_nodes"].get("missing", [])) > 0
    )
    is_compatible = (
        isinstance(report["compatibility"], dict)
        and report["compatibility"].get("compatible", True)
    )

    if has_wiring_error:
        status = "no_workflow"
    elif has_missing and not is_compatible:
        status = "missing_nodes_and_incompatible"
    elif has_missing:
        status = "missing_nodes"
    elif not is_compatible:
        status = "incompatible_models"
    else:
        status = "ready"

    report["status"] = status
    return to_json(report)


def _handle_provision_pipeline_verify(tool_input: dict) -> str:
    """Check a model exists on disk and is compatible with the workflow."""
    from .model_compat import handle as compat_handle

    filename = tool_input.get("filename")  # Cycle 48: guard required fields
    model_type = tool_input.get("model_type")
    if not filename or not isinstance(filename, str):
        return to_json({"error": "filename is required and must be a non-empty string."})
    if not model_type or not isinstance(model_type, str):
        return to_json({"error": "model_type is required and must be a non-empty string."})

    # Check file existence
    model_path = MODELS_DIR / model_type / filename
    exists = model_path.exists()
    try:  # Cycle 69: file can be deleted between exists() and stat() (TOCTOU)
        size_bytes = model_path.stat().st_size if exists else 0
    except OSError:
        size_bytes = 0
        exists = False  # Treat deleted-between-check-and-stat as non-existent

    # Identify family
    try:  # Cycle 65: guard against malformed JSON from cross-tool call
        family = json.loads(compat_handle("identify_model_family", {
            "model_name": filename,
        }))
    except (ValueError, TypeError):
        family = {}

    # Compatibility with loaded workflow
    try:  # Cycle 65: guard against malformed JSON from cross-tool call
        compat = json.loads(compat_handle("check_model_compatibility", {
            "models": [filename],
        }))
    except (ValueError, TypeError):
        compat = {}

    return to_json({
        "filename": filename,
        "model_type": model_type,
        "exists": exists,
        "size_bytes": size_bytes,
        "family": family,
        # Cycle 68: default True was wrong when compat is {"error": "..."} — None = unknown
        "workflow_compatible": None if compat.get("error") else compat.get("compatible"),
        "compatibility_details": compat,
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a provision_pipeline tool call."""
    try:
        if name == "provision_model":
            return _handle_provision_model(tool_input)
        elif name == "provision_pipeline_status":
            return _handle_provision_pipeline_status(tool_input)
        elif name == "provision_pipeline_verify":
            return _handle_provision_pipeline_verify(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        log.error("Provision pipeline tool %s failed: %s", name, e, exc_info=True)
        return to_json({"error": str(e)})
