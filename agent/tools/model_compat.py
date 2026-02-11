"""Model compatibility matrix — prevents mismatched model combos.

Knowledge base of SD 1.5 / SDXL / Flux / SD3 model families. Cross-references
checkpoint, VAE, ControlNet, and LoRA compatibility based on filename patterns
and base model families. Prevents the "swapped a model and now it crashes" scenario.
"""

import logging
import re
from pathlib import Path

from ._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility knowledge base
# ---------------------------------------------------------------------------

MODEL_FAMILIES = {
    "sd15": {
        "label": "Stable Diffusion 1.5",
        "resolution": "512x512",
        "checkpoint_patterns": [
            r"(?i)sd[-_]?v?1[-_.]?5", r"(?i)sd[-_]?1[-_.]?5",
            r"(?i)realisticVision", r"(?i)dreamshaper",
            r"(?i)deliberate", r"(?i)anything[-_]?v[345]",
            r"(?i)revAnimated", r"(?i)majicmix",
        ],
        "vae_patterns": [
            r"(?i)vae[-_]ft[-_]mse", r"(?i)sd[-_]?v?1.*vae",
            r"(?i)kl[-_]f8[-_]anime",
        ],
        "controlnet_patterns": [
            r"(?i)control_v11p_sd15", r"(?i)control_sd15",
            r"(?i)t2iadapter_.*sd15",
        ],
        "lora_compatible": True,
        "incompatible_families": ["sdxl", "flux", "sd3"],
    },
    "sdxl": {
        "label": "Stable Diffusion XL",
        "resolution": "1024x1024",
        "checkpoint_patterns": [
            r"(?i)sdxl", r"(?i)sd[-_]?xl",
            r"(?i)juggernaut[-_]?xl", r"(?i)dreamshaperXL",
            r"(?i)realvis[-_]?xl", r"(?i)pony",
            r"(?i)colossus", r"(?i)proteus",
        ],
        "vae_patterns": [
            r"(?i)sdxl.*vae", r"(?i)sdxl[-_]vae",
        ],
        "controlnet_patterns": [
            r"(?i)sdxl[-_]controlnet", r"(?i)control.*sdxl",
            r"(?i)diffusers_xl_.*controlnet",
        ],
        "lora_compatible": True,
        "incompatible_families": ["sd15", "flux", "sd3"],
    },
    "flux": {
        "label": "Flux",
        "resolution": "1024x1024",
        "checkpoint_patterns": [
            r"(?i)flux[-_.]?1", r"(?i)flux[-_]dev",
            r"(?i)flux[-_]schnell", r"(?i)flux[-_]fp",
        ],
        "vae_patterns": [
            r"(?i)ae[-_]?t5", r"(?i)flux.*vae",
        ],
        "controlnet_patterns": [
            r"(?i)flux.*controlnet", r"(?i)controlnet.*flux",
            r"(?i)union[-_]pro.*flux",
        ],
        "lora_compatible": True,
        "incompatible_families": ["sd15", "sdxl", "sd3"],
    },
    "sd3": {
        "label": "Stable Diffusion 3",
        "resolution": "1024x1024",
        "checkpoint_patterns": [
            r"(?i)sd[-_]?3", r"(?i)stable[-_]diffusion[-_]?3",
        ],
        "vae_patterns": [
            r"(?i)sd3.*vae",
        ],
        "controlnet_patterns": [
            r"(?i)sd3.*controlnet",
        ],
        "lora_compatible": True,
        "incompatible_families": ["sd15", "sdxl", "flux"],
    },
}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "check_model_compatibility",
        "description": (
            "Check if models are compatible with each other and the current workflow. "
            "Cross-references checkpoint, VAE, ControlNet, and LoRA by base model "
            "family (SD 1.5 / SDXL / Flux / SD3). Prevents mismatched model combos "
            "that cause silent failures or crashes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of model filenames to check compatibility between. "
                        "Example: ['sdxl_base.safetensors', 'control_v11p_sd15_depth.pth']"
                    ),
                },
                "workflow_path": {
                    "type": "string",
                    "description": (
                        "Optional path to workflow JSON. If provided, extracts model "
                        "references from the workflow and checks compatibility."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "identify_model_family",
        "description": (
            "Identify which base model family a model file belongs to. "
            "Returns: sd15, sdxl, flux, sd3, or unknown. Use to verify "
            "compatibility before swapping models."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model filename (e.g., 'realisticVisionV60B1.safetensors').",
                },
            },
            "required": ["model_name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

def _identify_family(model_name: str) -> str:
    """Identify model family from filename using pattern matching."""
    for family_id, family in sorted(MODEL_FAMILIES.items()):
        for pattern_list in [
            family["checkpoint_patterns"],
            family["vae_patterns"],
            family["controlnet_patterns"],
        ]:
            for pattern in pattern_list:
                if re.search(pattern, model_name):
                    return family_id
    return "unknown"


def _extract_models_from_workflow(workflow: dict) -> list[str]:
    """Extract model filenames referenced in a workflow."""
    models = []
    model_input_names = {
        "ckpt_name", "vae_name", "lora_name", "control_net_name",
        "model_name", "clip_name", "unet_name",
    }
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if key in model_input_names and isinstance(value, str):
                models.append(value)
    return sorted(set(models))


def _check_compatibility(models: list[str]) -> dict:
    """Check compatibility between a set of models."""
    identified = {}
    for m in models:
        identified[m] = _identify_family(m)

    # Find unique families (excluding unknown)
    families = set(f for f in identified.values() if f != "unknown")

    if len(families) <= 1:
        family = families.pop() if families else "unknown"
        return {
            "compatible": True,
            "family": family,
            "family_label": MODEL_FAMILIES.get(family, {}).get("label", "Unknown"),
            "resolution": MODEL_FAMILIES.get(family, {}).get("resolution", "unknown"),
            "models": identified,
            "message": f"All models belong to {MODEL_FAMILIES.get(family, {}).get('label', 'unknown')} family.",
        }

    # Multiple families detected — incompatible
    conflicts = []
    family_list = sorted(families)
    for i, f1 in enumerate(family_list):
        for f2 in family_list[i + 1:]:
            f1_models = [m for m, f in identified.items() if f == f1]
            f2_models = [m for m, f in identified.items() if f == f2]
            conflicts.append({
                "family_a": f1,
                "family_b": f2,
                "models_a": f1_models,
                "models_b": f2_models,
                "reason": f"{MODEL_FAMILIES[f1]['label']} models are incompatible with {MODEL_FAMILIES[f2]['label']}",
            })

    return {
        "compatible": False,
        "families_detected": family_list,
        "models": identified,
        "conflicts": conflicts,
        "message": f"Incompatible model families detected: {', '.join(MODEL_FAMILIES[f]['label'] for f in family_list)}",
        "suggestion": f"Use models from the same family. Detected: {', '.join(family_list)}",
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_check_compatibility(tool_input: dict) -> str:
    models = tool_input.get("models", [])
    workflow_path = tool_input.get("workflow_path")

    if workflow_path:
        import json
        path = Path(workflow_path)
        if not path.exists():
            return to_json({"error": f"File not found: {workflow_path}"})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return to_json({"error": f"Failed to parse workflow: {e}"})

        # Extract API-format nodes
        if "nodes" in data and isinstance(data["nodes"], list):
            prompt_data = data.get("extra", {}).get("prompt")
            if prompt_data:
                wf_models = _extract_models_from_workflow(prompt_data)
            else:
                wf_models = []
        else:
            wf_models = _extract_models_from_workflow(data)

        models = sorted(set(models + wf_models))

    if not models:
        # Try loaded workflow
        from .workflow_patch import get_current_workflow
        wf = get_current_workflow()
        if wf:
            models = _extract_models_from_workflow(wf)

    if not models:
        return to_json({
            "error": "No models to check. Provide model names, a workflow path, or load a workflow.",
        })

    result = _check_compatibility(models)
    return to_json(result)


def _handle_identify_family(tool_input: dict) -> str:
    model_name = tool_input["model_name"]
    family = _identify_family(model_name)
    info = MODEL_FAMILIES.get(family, {})

    return to_json({
        "model": model_name,
        "family": family,
        "label": info.get("label", "Unknown"),
        "resolution": info.get("resolution", "unknown"),
        "lora_compatible": info.get("lora_compatible", False),
        "incompatible_with": [
            MODEL_FAMILIES[f]["label"]
            for f in info.get("incompatible_families", [])
        ],
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a model compatibility tool call."""
    try:
        if name == "check_model_compatibility":
            return _handle_check_compatibility(tool_input)
        elif name == "identify_model_family":
            return _handle_identify_family(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
