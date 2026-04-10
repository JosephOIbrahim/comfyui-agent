"""Workflow template tools.

Provides starter workflows that the agent can load and customize,
instead of building from scratch with raw JSON patches.

Sources (in priority order):
  1. Built-in templates  (agent/templates/)
  2. User workflows      (COMFYUI_DATABASE/Workflows/)
  3. ComfyUI blueprints  (COMFYUI_INSTALL_DIR/blueprints/)
"""

import copy
import json
from pathlib import Path

from ._util import to_json
from ..config import WORKFLOWS_DIR, COMFYUI_BLUEPRINTS_DIR

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Template metadata — keeps descriptions out of the JSON files
_TEMPLATE_INFO = {
    "txt2img_sd15": {
        "description": "Basic text-to-image with SD 1.5. 7 nodes: checkpoint, CLIP encode (pos/neg), empty latent, KSampler, VAE decode, save.",
        "base_model": "SD 1.5",
        "resolution": "512x512",
    },
    "txt2img_sdxl": {
        "description": "Text-to-image with SDXL. 7 nodes: checkpoint, CLIP encode (pos/neg), empty latent, KSampler, VAE decode, save.",
        "base_model": "SDXL",
        "resolution": "1024x1024",
    },
    "img2img": {
        "description": "Image-to-image transformation. Loads an image, encodes to latent, denoises with prompt guidance. Adjust denoise (0.3-0.8) to control similarity to original.",
        "base_model": "SDXL",
        "resolution": "from input image",
    },
    "txt2img_lora": {
        "description": "Text-to-image with LoRA. Adds a LoRA loader between checkpoint and the rest of the pipeline. Set lora_name and strength.",
        "base_model": "SDXL",
        "resolution": "1024x1024",
    },
}

# ---------------------------------------------------------------------------
# External workflow scanning
# ---------------------------------------------------------------------------

_EXT_WORKFLOW_EXTENSIONS = {".json"}
# Skip files that aren't real workflows
_EXT_SKIP_PREFIXES = (".", "_", "comfyui_bookmarks")
_EXT_SKIP_NAMES = {".index.json", "rtx4090_optimization_summary.json",
                   "performance_comparison_rtx4090.json"}


def _scan_external_workflows(directory: Path, source: str) -> list[dict]:
    """Scan a directory for workflow JSON files. Returns list of metadata dicts."""
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        name = path.stem
        # Skip non-workflow files
        if path.name in _EXT_SKIP_NAMES:
            continue
        if any(name.startswith(p) for p in _EXT_SKIP_PREFIXES):
            continue
        # Quick validation: must be a JSON object with class_type nodes or "nodes" array
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            # Check for API format (nodes with class_type) or UI format (nodes array)
            has_api_nodes = any(
                isinstance(v, dict) and "class_type" in v
                for v in data.values()
                if isinstance(v, dict)
            )
            has_ui_nodes = isinstance(data.get("nodes"), list)
            if not has_api_nodes and not has_ui_nodes:
                continue
            # Count nodes
            if has_api_nodes:
                node_count = sum(
                    1 for v in data.values()
                    if isinstance(v, dict) and "class_type" in v
                )
            else:
                node_count = len(data["nodes"])
        except (json.JSONDecodeError, OSError):
            continue

        results.append({
            "name": name,
            "source": source,
            "path": str(path),
            "node_count": node_count,
        })
    return results


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "list_workflow_templates",
        "description": (
            "List available workflow templates from all sources: built-in templates, "
            "user workflows (COMFYUI_Database/Workflows/), and ComfyUI blueprints. "
            "Use format='names_only' for a compact list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["all", "builtin", "workflows", "blueprints"],
                    "description": "Filter by source. Default: all.",
                },
                "format": {
                    "type": "string",
                    "enum": ["summary", "names_only"],
                    "description": "Output format. 'names_only' for compact listing.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_workflow_template",
        "description": (
            "Load a workflow template and make it the active workflow for editing. "
            "Works with built-in templates, user workflows, and ComfyUI blueprints. "
            "After loading, use set_input/connect_nodes/apply_workflow_patch "
            "to customize it, then execute_workflow to run it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": (
                        "Template name (e.g. 'txt2img_sdxl', 'video_ltx2_3_t2v', "
                        "'Canny to Image (Z-Image-Turbo)'). "
                        "Use list_workflow_templates to see available options."
                    ),
                },
            },
            "required": ["template"],
        },
    },
]

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_list_templates(tool_input: dict) -> str:
    source_filter = tool_input.get("source", "all")
    fmt = tool_input.get("format", "summary")
    templates = []

    # 1. Built-in templates
    if source_filter in ("all", "builtin"):
        for name in sorted(_TEMPLATE_INFO.keys()):
            info = _TEMPLATE_INFO[name]
            path = _TEMPLATES_DIR / f"{name}.json"
            entry = {
                "name": name,
                "source": "builtin",
                "available": path.exists(),
            }
            if fmt != "names_only":
                entry["description"] = info["description"]
                entry["base_model"] = info.get("base_model", "")
                entry["resolution"] = info.get("resolution", "")
            templates.append(entry)

    # 2. User workflows (COMFYUI_Database/Workflows/)
    if source_filter in ("all", "workflows"):
        for wf in _scan_external_workflows(WORKFLOWS_DIR, "workflows"):
            entry = {
                "name": wf["name"],
                "source": "workflows",
                "available": True,
            }
            if fmt != "names_only":
                entry["node_count"] = wf["node_count"]
                entry["path"] = wf["path"]
            templates.append(entry)

    # 3. ComfyUI blueprints
    if source_filter in ("all", "blueprints"):
        for bp in _scan_external_workflows(COMFYUI_BLUEPRINTS_DIR, "blueprints"):
            entry = {
                "name": bp["name"],
                "source": "blueprints",
                "available": True,
            }
            if fmt != "names_only":
                entry["node_count"] = bp["node_count"]
                entry["path"] = bp["path"]
            templates.append(entry)

    return to_json({
        "templates": templates,
        "count": len(templates),
        "sources": {
            "builtin": str(_TEMPLATES_DIR),
            "workflows": str(WORKFLOWS_DIR),
            "blueprints": str(COMFYUI_BLUEPRINTS_DIR),
        },
    })


def _resolve_template_path(template_name: str) -> Path | None:
    """Resolve a template name to a file path, checking all sources.

    template_name must be a simple filename stem — no path separators, no
    traversal sequences. Rejects anything that would escape the template dirs.
    (Cycle 29 path-traversal fix)
    """
    # Reject names that contain path separators or traversal sequences
    if any(c in template_name for c in ("/", "\\", "\x00")) or ".." in template_name:
        return None  # Caller returns a "not found" error

    # 1. Built-in templates (exact match)
    builtin = _TEMPLATES_DIR / f"{template_name}.json"
    if builtin.exists():
        return builtin

    # 2. User workflows
    if WORKFLOWS_DIR.exists():
        candidate = WORKFLOWS_DIR / f"{template_name}.json"
        if candidate.exists():
            return candidate

    # 3. ComfyUI blueprints
    if COMFYUI_BLUEPRINTS_DIR.exists():
        candidate = COMFYUI_BLUEPRINTS_DIR / f"{template_name}.json"
        if candidate.exists():
            return candidate

    # 4. Fuzzy match across all external dirs (case-insensitive)
    name_lower = template_name.lower()
    for directory in (WORKFLOWS_DIR, COMFYUI_BLUEPRINTS_DIR):
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            if path.stem.lower() == name_lower:
                return path

    return None


def _handle_get_template(tool_input: dict) -> str:
    template_name = tool_input.get("template")  # Cycle 45: guard required field
    if not template_name or not isinstance(template_name, str):
        return to_json({"error": "template is required and must be a non-empty string."})

    path = _resolve_template_path(template_name)

    if path is None:
        # Build hint of available templates
        available = sorted(_TEMPLATE_INFO.keys())
        if WORKFLOWS_DIR.exists():
            available.extend(
                p.stem for p in sorted(WORKFLOWS_DIR.glob("*.json"))
                if p.name not in _EXT_SKIP_NAMES
            )
        return to_json({
            "error": f"Template '{template_name}' not found.",
            "hint": "Use list_workflow_templates to see all available options.",
            "sample_builtin": available[:10],
        })

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return to_json({"error": f"Failed to read template: {e}"})

    # Determine source
    source = "builtin"
    if str(path).startswith(str(WORKFLOWS_DIR)):
        source = "workflows"
    elif str(path).startswith(str(COMFYUI_BLUEPRINTS_DIR)):
        source = "blueprints"

    # Handle UI format: extract API format from extra.prompt if available
    workflow_data = data
    fmt = "api"
    if "nodes" in data and isinstance(data["nodes"], list):
        api_data = data.get("extra", {}).get("prompt")
        if api_data and isinstance(api_data, dict):
            workflow_data = api_data
            fmt = "api (extracted from UI format)"
        else:
            # UI-only format — load as-is, note the limitation
            fmt = "ui_only"

    # Load into workflow_patch state so the agent can edit it
    from .workflow_patch import _get_state
    _s = _get_state()
    _s["loaded_path"] = str(path)
    _s["base_workflow"] = copy.deepcopy(workflow_data)
    _s["current_workflow"] = copy.deepcopy(workflow_data)
    _s["history"] = []
    _s["format"] = fmt.split()[0]  # "api" or "ui_only"

    info = _TEMPLATE_INFO.get(template_name, {})

    # Build node summary
    nodes = {}
    if fmt != "ui_only":
        for nid, node in sorted(workflow_data.items()):
            if not isinstance(node, dict) or "class_type" not in node:
                continue
            nodes[nid] = {
                "class_type": node.get("class_type", "?"),
                "editable_inputs": [
                    k for k, v in node.get("inputs", {}).items()
                    if not (isinstance(v, list) and len(v) == 2)
                ],
            }

    result = {
        "loaded": template_name,
        "source": source,
        "path": str(path),
        "format": fmt,
        "node_count": len(nodes) if nodes else len(data.get("nodes", [])),
        "nodes": nodes,
        "message": (
            "Template loaded as active workflow. Use set_input to customize values, "
            "connect_nodes to rewire, or apply_workflow_patch for advanced changes. "
            "Then execute_workflow to run it."
        ),
    }
    if info.get("description"):
        result["description"] = info["description"]
    if info.get("base_model"):
        result["base_model"] = info["base_model"]

    return to_json(result)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def handle(name: str, tool_input: dict) -> str:
    """Execute a workflow_templates tool call."""
    try:
        if name == "list_workflow_templates":
            return _handle_list_templates(tool_input)
        elif name == "get_workflow_template":
            return _handle_get_template(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
