"""Workflow template tools.

Provides starter workflows that the agent can load and customize,
instead of building from scratch with raw JSON patches.
"""

import copy
import json
from pathlib import Path

from ._util import to_json

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
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "list_workflow_templates",
        "description": (
            "List available workflow templates with descriptions. "
            "Templates are pre-built starter workflows for common patterns "
            "(txt2img, img2img, LoRA, etc.) that can be loaded and customized."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_workflow_template",
        "description": (
            "Load a workflow template and make it the active workflow for editing. "
            "After loading, use set_input/connect_nodes/apply_workflow_patch "
            "to customize it, then execute_workflow to run it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": (
                        "Template name (e.g. 'txt2img_sdxl', 'img2img', 'txt2img_lora'). "
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


def _handle_list_templates() -> str:
    templates = []
    for name in sorted(_TEMPLATE_INFO.keys()):
        info = _TEMPLATE_INFO[name]
        path = _TEMPLATES_DIR / f"{name}.json"
        templates.append({
            "name": name,
            "description": info["description"],
            "base_model": info.get("base_model", ""),
            "resolution": info.get("resolution", ""),
            "available": path.exists(),
        })

    return to_json({
        "templates": templates,
        "count": len(templates),
    })


def _handle_get_template(tool_input: dict) -> str:
    template_name = tool_input["template"]
    path = _TEMPLATES_DIR / f"{template_name}.json"

    if not path.exists():
        available = sorted(_TEMPLATE_INFO.keys())
        return to_json({
            "error": f"Template '{template_name}' not found.",
            "available": available,
        })

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return to_json({"error": f"Failed to read template: {e}"})

    # Load into workflow_patch state so the agent can edit it
    from .workflow_patch import _state
    _state["loaded_path"] = f"template:{template_name}"
    _state["base_workflow"] = copy.deepcopy(data)
    _state["current_workflow"] = copy.deepcopy(data)
    _state["history"] = []
    _state["format"] = "api"

    info = _TEMPLATE_INFO.get(template_name, {})

    # Build node summary
    nodes = {}
    for nid, node in sorted(data.items()):
        nodes[nid] = {
            "class_type": node.get("class_type", "?"),
            "editable_inputs": [
                k for k, v in node.get("inputs", {}).items()
                if not (isinstance(v, list) and len(v) == 2)
            ],
        }

    return to_json({
        "loaded": template_name,
        "description": info.get("description", ""),
        "base_model": info.get("base_model", ""),
        "node_count": len(data),
        "nodes": nodes,
        "message": (
            "Template loaded as active workflow. Use set_input to customize values, "
            "connect_nodes to rewire, or apply_workflow_patch for advanced changes. "
            "Then execute_workflow to run it."
        ),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def handle(name: str, tool_input: dict) -> str:
    """Execute a workflow_templates tool call."""
    try:
        if name == "list_workflow_templates":
            return _handle_list_templates()
        elif name == "get_workflow_template":
            return _handle_get_template(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
