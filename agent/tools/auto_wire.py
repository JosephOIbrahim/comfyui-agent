"""Auto-wire intelligence — connect downloaded models to workflows automatically.

Maps model types to loader node class_types, scans the loaded workflow for
matching loaders, and wires the model filename into the correct input field.
Includes compatibility checks to warn about family mismatches.
"""

import json
import logging

from ._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loader class_type -> input field mappings per model_type
# ---------------------------------------------------------------------------

_LOADER_MAP: dict[str, list[tuple[str, str]]] = {
    "checkpoints": [
        ("CheckpointLoaderSimple", "ckpt_name"),
        ("CheckpointLoader", "ckpt_name"),
        ("UNETLoader", "unet_name"),
    ],
    "loras": [
        ("LoraLoader", "lora_name"),
        ("LoraLoaderModelOnly", "lora_name"),
    ],
    "vae": [
        ("VAELoader", "vae_name"),
    ],
    "controlnet": [
        ("ControlNetLoader", "control_net_name"),
        ("DiffControlNetLoader", "control_net_name"),
    ],
    "clip": [
        ("CLIPLoader", "clip_name"),
        ("DualCLIPLoader", "clip_name1"),
    ],
    "upscale_models": [
        ("UpscaleModelLoader", "model_name"),
    ],
    "embeddings": [],  # Embeddings are referenced inline in prompts, not via loader nodes
    "text_encoders": [
        ("CLIPLoader", "clip_name"),
    ],
}

# Reverse index: class_type -> (model_type, input_field)
_CLASS_TO_MODEL_TYPE: dict[str, tuple[str, str]] = {}
for _mtype, _loaders in _LOADER_MAP.items():
    for _cls, _field in _loaders:
        if _cls not in _CLASS_TO_MODEL_TYPE:
            _CLASS_TO_MODEL_TYPE[_cls] = (_mtype, _field)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "wire_model",
        "description": (
            "Automatically wire a model file into the loaded workflow. "
            "Finds the appropriate loader node, sets the model filename, "
            "and validates compatibility. Use after downloading a model "
            "to connect it to the workflow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Model filename (e.g., 'flux-1-dev.safetensors').",
                },
                "model_type": {
                    "type": "string",
                    "description": (
                        "Model type: checkpoints, loras, vae, controlnet, "
                        "clip, upscale_models, embeddings, text_encoders."
                    ),
                },
            },
            "required": ["filename", "model_type"],
        },
    },
    {
        "name": "suggest_wiring",
        "description": (
            "Analyze the loaded workflow and suggest what models/nodes are "
            "needed. Returns a list of loader nodes with their current model "
            "settings and what could be changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_workflow() -> dict | None:
    """Get the current loaded workflow."""
    from .workflow_patch import get_current_workflow
    return get_current_workflow()


def _find_loader_nodes(workflow: dict, model_type: str) -> list[dict]:
    """Find all loader nodes in the workflow matching the given model_type.

    Returns list of dicts with node_id, class_type, input_field, current_value.
    """
    loaders = _LOADER_MAP.get(model_type, [])
    if not loaders:
        return []

    loader_class_types = {cls: field for cls, field in loaders}
    found = []

    for node_id, node_data in sorted(workflow.items()):
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        if class_type in loader_class_types:
            input_field = loader_class_types[class_type]
            inputs = node_data.get("inputs", {})
            current_value = inputs.get(input_field)
            found.append({
                "node_id": node_id,
                "class_type": class_type,
                "input_field": input_field,
                "current_value": current_value,
            })

    return found


def _check_family_compat(filename: str, workflow: dict) -> dict | None:
    """Check if the model family is compatible with existing workflow models.

    Returns a warning dict if mismatch detected, None if compatible or unknown.
    """
    from .model_compat import _identify_family, _extract_models_from_workflow

    new_family = _identify_family(filename)
    if new_family == "unknown":
        return None

    existing_models = _extract_models_from_workflow(workflow)
    if not existing_models:
        return None

    existing_families = set()
    for model in existing_models:
        fam = _identify_family(model)
        if fam != "unknown":
            existing_families.add(fam)

    if not existing_families:
        return None

    if new_family in existing_families:
        return None

    from .model_compat import MODEL_FAMILIES
    new_label = MODEL_FAMILIES.get(new_family, {}).get("label", new_family)
    existing_labels = [
        MODEL_FAMILIES.get(f, {}).get("label", f) for f in sorted(existing_families)
    ]

    return {
        "warning": "Family mismatch detected",
        "new_model_family": new_label,
        "workflow_families": existing_labels,
        "message": (
            f"'{filename}' is {new_label}, but the workflow uses "
            f"{', '.join(existing_labels)}. This may cause errors or "
            f"silent quality degradation."
        ),
    }


def _scan_all_loaders(workflow: dict) -> list[dict]:
    """Scan workflow for ALL loader nodes across every model type."""
    found = []
    seen_node_ids = set()

    for node_id, node_data in sorted(workflow.items()):
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        if class_type in _CLASS_TO_MODEL_TYPE and node_id not in seen_node_ids:
            model_type, input_field = _CLASS_TO_MODEL_TYPE[class_type]
            inputs = node_data.get("inputs", {})
            current_value = inputs.get(input_field)
            found.append({
                "node_id": node_id,
                "class_type": class_type,
                "model_type": model_type,
                "input_field": input_field,
                "current_value": current_value,
            })
            seen_node_ids.add(node_id)

    return found


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_wire_model(tool_input: dict) -> str:
    filename = tool_input["filename"]
    model_type = tool_input["model_type"]

    # Validate model_type
    if model_type not in _LOADER_MAP:
        valid_types = sorted(_LOADER_MAP.keys())
        return to_json({
            "error": f"Unknown model_type '{model_type}'.",
            "valid_types": valid_types,
        })

    # Embeddings are special — referenced in prompt text, not via loaders
    if model_type == "embeddings":
        return to_json({
            "error": (
                "Embeddings are referenced inline in prompt text "
                "(e.g., 'embedding:filename'), not wired via loader nodes. "
                "Edit the positive or negative prompt text to include "
                f"'embedding:{filename.rsplit('.', 1)[0]}'."
            ),
        })

    # Check workflow is loaded
    workflow = _get_workflow()
    if workflow is None:
        return to_json({
            "error": (
                "No workflow is open. Load a workflow first with "
                "load_workflow, then wire the model."
            ),
        })

    # Find matching loader nodes
    loaders = _find_loader_nodes(workflow, model_type)

    if not loaders:
        # Suggest which node to add
        recommended = _LOADER_MAP[model_type]
        if recommended:
            rec_class = recommended[0][0]
            return to_json({
                "error": (
                    f"No {model_type} loader node found in the workflow. "
                    f"Add a '{rec_class}' node first with add_node, "
                    f"then wire the model."
                ),
                "recommended_node": rec_class,
                "model_type": model_type,
            })
        return to_json({
            "error": f"No loader nodes available for model_type '{model_type}'.",
        })

    # Use the first matching loader
    target = loaders[0]

    # Compatibility check BEFORE wiring (warn, don't block)
    compat_warning = _check_family_compat(filename, workflow)

    # Wire it via set_input
    from .workflow_patch import handle as patch_handle
    result_str = patch_handle("set_input", {
        "node_id": target["node_id"],
        "input_name": target["input_field"],
        "value": filename,
    })
    result = json.loads(result_str)

    if "error" in result:
        return to_json(result)

    # Build response
    response = {
        "wired": True,
        "node_id": target["node_id"],
        "class_type": target["class_type"],
        "input_field": target["input_field"],
        "previous_value": target["current_value"],
        "new_value": filename,
    }

    # If multiple loaders exist, note it
    if len(loaders) > 1:
        response["other_loaders"] = [
            {
                "node_id": ldr["node_id"],
                "class_type": ldr["class_type"],
                "current_value": ldr["current_value"],
            }
            for ldr in loaders[1:]
        ]
        response["note"] = (
            f"Wired to the first {target['class_type']} node. "
            f"{len(loaders) - 1} other loader(s) of the same type exist."
        )

    # Attach compatibility warning if detected
    if compat_warning:
        response["compatibility_warning"] = compat_warning

    return to_json(response)


def _handle_suggest_wiring(tool_input: dict) -> str:
    workflow = _get_workflow()
    if workflow is None:
        return to_json({
            "error": (
                "No workflow is open. Load a workflow first with "
                "load_workflow."
            ),
        })

    loaders = _scan_all_loaders(workflow)

    # Identify model types that have no loader in the workflow
    present_types = {ldr["model_type"] for ldr in loaders}
    # Core types that most workflows should have
    core_types = {"checkpoints", "vae"}
    missing_core = []
    for mt in sorted(core_types - present_types):
        recommended = _LOADER_MAP.get(mt, [])
        if recommended:
            missing_core.append({
                "model_type": mt,
                "recommended_node": recommended[0][0],
                "note": f"No {mt} loader found — using ComfyUI default.",
            })

    return to_json({
        "loaders": loaders,
        "loader_count": len(loaders),
        "missing_core_loaders": missing_core,
        "total_nodes": len(workflow),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute an auto_wire tool call."""
    try:
        if name == "wire_model":
            return _handle_wire_model(tool_input)
        elif name == "suggest_wiring":
            return _handle_suggest_wiring(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        log.error("Unhandled error in auto_wire tool %s: %s", name, e, exc_info=True)
        return to_json({"error": str(e)})
