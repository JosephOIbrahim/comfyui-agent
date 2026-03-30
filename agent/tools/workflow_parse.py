"""Workflow analysis tools.

Load, analyze, summarize, and validate ComfyUI workflow JSON.
Handles both UI format (from ComfyUI "Save") and API format
(from "Save API Format" or programmatic use).

These tools give the agent deep understanding of what a workflow does,
what's connected to what, and what can be safely modified.
"""

import json
import re
from pathlib import Path

import httpx

from ._util import to_json
from ..config import COMFYUI_URL

# Regex to detect UUID-style component type strings.
# Component instance nodes use a UUID as their "type" instead of a class name,
# e.g. "b94257db-cdc1-45d3-8913-ca61e782d9c1".
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "load_workflow",
        "description": (
            "Load and analyze a ComfyUI workflow JSON file. "
            "Returns: format type, all nodes with class_type, "
            "connections between nodes (the graph), editable fields "
            "with current values, and a human-readable summary of the pipeline. "
            "This is the main tool for understanding what a workflow does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the workflow JSON file.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "validate_workflow",
        "description": (
            "Validate a workflow against the running ComfyUI instance. "
            "Checks that all node types exist (are installed), that required "
            "inputs are provided, and that connections have compatible types. "
            "Requires ComfyUI to be running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the workflow JSON file.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_editable_fields",
        "description": (
            "Get a focused list of all editable fields in a workflow — "
            "the parameters that can be changed without rewiring the graph. "
            "Groups fields by node class_type for easy scanning. "
            "This is useful before modifying a workflow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the workflow JSON file.",
                },
                "class_filter": {
                    "type": "string",
                    "description": (
                        "Optional: only show fields from nodes matching this "
                        "class_type substring (case-insensitive)."
                    ),
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "classify_workflow",
        "description": (
            "Classify a workflow into known pipeline patterns. "
            "Returns the base pattern (txt2img, img2img, inpaint, etc.), "
            "any modifiers (controlnet, lora, upscale, etc.), and a "
            "human-readable description. Use this to understand what "
            "kind of pipeline a workflow implements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the workflow JSON file.",
                },
            },
            "required": ["path"],
        },
    },
]

# ---------------------------------------------------------------------------
# Core parsing logic
# ---------------------------------------------------------------------------


def _load_json(path_str: str) -> tuple[dict, str | None]:
    """Load JSON from file. Returns (data, error_or_none)."""
    path = Path(path_str)
    if not path.exists():
        return {}, f"File not found: {path_str}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, None
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON: {e}"


# ---------------------------------------------------------------------------
# COMFY_AUTOGROW_V3 helpers (ComfyUI 0.16.3+)
# ---------------------------------------------------------------------------
# AUTOGROW inputs use dotted names in the UI ("values.a", "values.b") but
# nested dicts in the API format ({"values": {"a": ..., "b": ...}}).
# These helpers translate between the two representations.


def _is_autogrow_dotted_name(name: str) -> bool:
    """Check if an input name uses COMFY_AUTOGROW_V3 dotted notation."""
    return "." in name and not name.startswith(".")


def _group_autogrow_inputs(inputs: dict) -> dict:
    """Group dotted AUTOGROW inputs into nested dicts for API format.

    UI format: {"values.a": 42, "values.b": 7}
    API format: {"values": {"a": 42, "b": 7}}
    Non-dotted inputs pass through unchanged.
    """
    result = {}
    for name, value in inputs.items():
        if _is_autogrow_dotted_name(name):
            group, _, sub = name.partition(".")
            result.setdefault(group, {})[sub] = value
        else:
            result[name] = value
    return result


def _flatten_autogrow_inputs(inputs: dict) -> dict:
    """Flatten nested AUTOGROW dicts into dotted names for display.

    API format: {"values": {"a": 42, "b": 7}}
    Flat format: {"values.a": 42, "values.b": 7}
    Connection lists and dicts with class_type are left as-is.
    """
    result = {}
    for name, value in inputs.items():
        if (
            isinstance(value, dict)
            and "class_type" not in value
            and value  # skip empty dicts
        ):
            # Nested dict — likely AUTOGROW group. Flatten sub-keys.
            for sub, sub_val in value.items():
                result[f"{name}.{sub}"] = sub_val
        else:
            result[name] = value
    return result


def _extract_api_format(data: dict) -> tuple[dict, str]:
    """
    Extract API-format node dict from any workflow format.

    Returns (api_nodes, format_name).
    api_nodes: {node_id_str: {"class_type": ..., "inputs": {...}}}

    COMFY_AUTOGROW_V3 note: In API format, AUTOGROW inputs are stored
    as nested dicts (e.g. {"values": {"a": 42}}). The flatten/group
    helpers handle translation to/from dotted names when needed.
    """
    # Check for UI format (has "nodes" array)
    if "nodes" in data and isinstance(data["nodes"], list):
        # Prefer embedded API format in extra.prompt
        prompt_data = data.get("extra", {}).get("prompt")
        if prompt_data and isinstance(prompt_data, dict):
            nodes = {
                k: v for k, v in prompt_data.items()
                if isinstance(v, dict) and "class_type" in v
            }
            return nodes, "ui_with_api"

        # UI-only: convert nodes array
        nodes = {}
        for node in data["nodes"]:
            nid = str(node.get("id", ""))
            class_type = node.get("type", "Unknown")
            widgets = node.get("widgets_values", [])
            entry = {
                "class_type": class_type,
                "inputs": {},
                "_widgets_values": widgets,
                "_ui_node": True,
            }
            # Mark component instance nodes (type is a UUID)
            if _is_component_node(class_type):
                entry["_is_component"] = True
            nodes[nid] = entry
        return nodes, "ui_only"

    # API format: top-level keys are node IDs
    nodes = {
        k: v for k, v in data.items()
        if isinstance(v, dict) and "class_type" in v
    }
    return nodes, "api"


def _trace_connections(nodes: dict) -> list[dict]:
    """
    Trace connections between nodes.

    In API format, a connection is an input value that's a list:
    [source_node_id_str, output_index_int]

    Returns list of connection dicts.
    """
    connections = []
    for target_id, node in sorted(nodes.items()):
        if node.get("_ui_node"):
            continue  # UI-only nodes don't have connection info
        # Flatten AUTOGROW nested dicts so connections like
        # {"values": {"a": ["10", 0]}} are traced as "values.a"
        flat_inputs = _flatten_autogrow_inputs(node.get("inputs", {}))
        for input_name, value in sorted(flat_inputs.items()):
            if isinstance(value, list) and len(value) == 2:
                source_id = str(value[0])
                output_idx = value[1]
                source_class = nodes.get(source_id, {}).get("class_type", "?")
                target_class = node.get("class_type", "?")
                connections.append({
                    "from_node": source_id,
                    "from_class": source_class,
                    "from_output": output_idx,
                    "to_node": target_id,
                    "to_class": target_class,
                    "to_input": input_name,
                })
    return connections


def _find_editable_fields(nodes: dict, class_filter: str = "") -> list[dict]:
    """
    Find all non-connection input fields (the editable parameters).
    """
    fields = []
    filter_lower = class_filter.lower()

    for nid, node in sorted(nodes.items()):
        if node.get("_ui_node"):
            continue
        class_type = node.get("class_type", "")
        if filter_lower and filter_lower not in class_type.lower():
            continue

        # Flatten AUTOGROW nested dicts so sub-inputs like "values.a"
        # appear as individual editable fields.
        flat_inputs = _flatten_autogrow_inputs(node.get("inputs", {}))

        # He2025: sort for deterministic field order per node
        for field_name, value in sorted(flat_inputs.items()):
            # Skip connections (2-element list [node_id, output_index])
            if isinstance(value, list) and len(value) == 2:
                continue

            fields.append({
                "node_id": nid,
                "class_type": class_type,
                "field": field_name,
                "value": value,
                "type": type(value).__name__,
            })

    return fields


def _is_component_node(class_type: str) -> bool:
    """Return True if a node's class_type is a UUID (component instance)."""
    return bool(_UUID_RE.match(class_type))


def _extract_subgraph_nodes(workflow_data: dict) -> dict:
    """Extract internal node information from component/subgraph definitions.

    ComfyUI workflows can contain *component nodes* (subgraphs).  A component
    instance appears in the top-level ``nodes`` list with a UUID as its
    ``type``.  The matching subgraph definition lives under
    ``definitions.subgraphs`` and contains its own ``nodes`` and ``links``
    arrays describing the internal graph.

    Returns a dict keyed by component UUID::

        {
            "<uuid>": {
                "node_count": int,
                "link_count": int,
                "class_types": sorted list of unique class_type strings,
            },
            ...
        }

    Returns an empty dict when no subgraph definitions are present.
    """
    definitions = workflow_data.get("definitions")
    if not definitions or not isinstance(definitions, dict):
        return {}

    subgraphs_list = definitions.get("subgraphs")
    if not subgraphs_list or not isinstance(subgraphs_list, list):
        return {}

    result: dict[str, dict] = {}
    for subgraph in subgraphs_list:
        if not isinstance(subgraph, dict):
            continue

        # The subgraph's UUID may be stored as an "id" or "uuid" field,
        # or matched from a top-level node whose type is a UUID.
        sg_id = subgraph.get("id") or subgraph.get("uuid") or ""

        internal_nodes = subgraph.get("nodes", [])
        internal_links = subgraph.get("links", [])

        class_types: set[str] = set()
        for node in internal_nodes:
            if not isinstance(node, dict):
                continue
            ntype = node.get("type", "")
            if ntype:
                class_types.add(ntype)

        result[str(sg_id)] = {
            "node_count": len(internal_nodes),
            "link_count": len(internal_links) if isinstance(internal_links, list) else 0,
            "class_types": sorted(class_types),
        }

    return result


def _all_subgraph_class_types(subgraph_info: dict) -> set[str]:
    """Collect every class_type across all subgraph definitions."""
    types: set[str] = set()
    for sg in subgraph_info.values():
        types.update(sg.get("class_types", []))
    return types


def _build_summary(
    nodes: dict,
    connections: list[dict],
    fmt: str,
    classification: dict | None = None,
    subgraph_info: dict | None = None,
) -> str:
    """Build a human-readable summary of the workflow pipeline."""
    if not nodes:
        return "Empty workflow (no nodes found)."

    lines = []

    if classification:
        lines.append(
            f"Pipeline: {classification['description']}"
        )
        if classification.get("modifiers"):
            lines.append(
                f"Modifiers: {', '.join(classification['modifiers'])}"
            )
        lines.append("")

    # Class type counts
    class_counts: dict[str, int] = {}
    for node in nodes.values():
        ct = node.get("class_type", "Unknown")
        class_counts[ct] = class_counts.get(ct, 0) + 1

    lines.append(f"{len(nodes)} nodes, {len(connections)} connections")
    lines.append("")

    # Identify pipeline stages by looking at node classes
    loaders = []
    encoders = []
    samplers = []
    threed = []
    outputs = []
    other = []

    for ct in sorted(class_counts.keys()):
        ct_lower = ct.lower()
        if any(k in ct_lower for k in ("load", "checkpoint", "unet", "model")):
            loaders.append(ct)
        elif any(k in ct_lower for k in ("encode", "clip", "conditioning")):
            encoders.append(ct)
        elif any(k in ct_lower for k in ("sample", "sampler", "ksampler", "denoise")):
            samplers.append(ct)
        elif any(k in ct_lower for k in (
            "mesh", "voxel", "3d", "triplane", "gaussian", "splat",
            "point_cloud", "glb", "ply", "obj", "nerf",
            "pose", "camera", "hunyuan3d",
        )):
            threed.append(ct)
        elif any(k in ct_lower for k in ("save", "preview", "output", "combine", "video")):
            outputs.append(ct)
        else:
            other.append(ct)

    if loaders:
        lines.append(f"Loaders: {', '.join(loaders)}")
    if encoders:
        lines.append(f"Encoding: {', '.join(encoders)}")
    if samplers:
        lines.append(f"Sampling: {', '.join(samplers)}")
    if threed:
        lines.append(f"3D Processing: {', '.join(threed)}")
    if outputs:
        lines.append(f"Output: {', '.join(outputs)}")
    if other:
        lines.append(f"Other: {', '.join(other)}")

    # Describe data flow from connections
    if connections:
        lines.append("")
        lines.append("Data flow:")
        # Group by target node to show input sources
        by_target: dict[str, list[dict]] = {}
        for conn in connections:
            key = f"{conn['to_node']} ({conn['to_class']})"
            by_target.setdefault(key, []).append(conn)

        for target, conns in sorted(by_target.items()):
            sources = [f"{c['from_class']}.out[{c['from_output']}]" for c in conns]
            lines.append(f"  {target} <- {', '.join(sources)}")

    # Component/subgraph summary
    if subgraph_info:
        lines.append("")
        lines.append(
            f"Components: {len(subgraph_info)} subgraph definition(s)"
        )
        for sg_id, sg in sorted(subgraph_info.items()):
            lines.append(
                f"  Component {sg_id[:12]}...: "
                f"{sg['node_count']} nodes, {sg['link_count']} links — "
                f"{', '.join(sg['class_types'][:8])}"
                f"{'...' if len(sg['class_types']) > 8 else ''}"
            )

    if fmt == "ui_only":
        lines.append("")
        lines.append(
            "Note: UI-only format — editable field values not available. "
            "Use an API-format workflow for full analysis."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

_PATTERN_SIGNATURES: dict[str, dict] = {
    "txt2img": {
        "required": ["EmptyLatentImage"],
        "sampler": True,
        "loader": True,
        "description": "Text-to-image generation",
    },
    "img2img": {
        "required_any": [["LoadImage"]],
        "extra_required_any": [["VAEEncode", "VAEEncodeTiled"]],
        "sampler": True,
        "loader": True,
        "description": "Image-to-image transformation",
    },
    "inpaint": {
        "required_any": [
            ["SetLatentNoiseMask"],
            ["InpaintModelConditioning"],
        ],
        "sampler": True,
        "description": "Inpainting or outpainting",
    },
    "controlnet": {
        "required_any": [
            ["ControlNetApply"],
            ["ControlNetApplyAdvanced"],
            ["Apply ControlNet"],
        ],
        "sampler": True,
        "description": "ControlNet-guided generation",
    },
    "upscale": {
        "required_any": [
            ["UpscaleModelLoader"],
            ["ImageUpscaleWithModel"],
            ["LatentUpscale"],
            ["LatentUpscaleBy"],
        ],
        "description": "Image or latent upscaling",
    },
    "lora": {
        "required_any": [
            ["LoraLoader"],
            ["LoraLoaderModelOnly"],
        ],
        "description": "LoRA model adaptation",
    },
    "ip_adapter": {
        "required_any": [
            ["IPAdapterApply"],
            ["IPAdapter"],
            ["IPAdapterAdvanced"],
        ],
        "description": "IP-Adapter image-guided generation",
    },
    "video": {
        "required_any": [
            ["VHS_VideoCombine"],
            ["AnimateDiff"],
            ["SVD_img2vid_Conditioning"],
        ],
        "description": "Video generation or animation",
    },
}


def _classify_pattern(nodes: dict) -> dict:
    """Classify workflow into known pipeline patterns.

    Analyzes node class_types to identify what kind of pipeline this is.
    Returns a dict with pattern name, description, modifiers, and
    class_types.
    """
    # Collect all class_types (He2025: sorted for determinism)
    class_types = sorted({
        n.get("class_type", "")
        for n in nodes.values()
        if isinstance(n, dict) and not n.get("_ui_node")
    })
    class_set = set(class_types)

    has_sampler = any(
        "sampler" in ct.lower() or "ksampler" in ct.lower()
        for ct in class_types
    )
    has_loader = any(
        "checkpoint" in ct.lower() or "unetloader" in ct.lower()
        for ct in class_types
    )

    matched_patterns = []
    for pattern_name, sig in sorted(_PATTERN_SIGNATURES.items()):
        if "required" in sig:
            if not all(req in class_set for req in sig["required"]):
                continue

        if "required_any" in sig:
            found_any = False
            for group in sig["required_any"]:
                if any(
                    node_type in class_set for node_type in group
                ):
                    found_any = True
                    break
            if not found_any:
                continue

        if "extra_required_any" in sig:
            found_extra = False
            for group in sig["extra_required_any"]:
                if any(
                    node_type in class_set for node_type in group
                ):
                    found_extra = True
                    break
            if not found_extra:
                continue

        if sig.get("sampler") and not has_sampler:
            continue
        if sig.get("loader") and not has_loader:
            continue

        matched_patterns.append(pattern_name)

    # Determine base pattern
    base_pattern = "unknown"
    if not matched_patterns:
        if has_sampler and has_loader:
            base_pattern = "custom"
        elif has_sampler:
            base_pattern = "custom_sampler"
    else:
        base_priority = ["txt2img", "img2img", "inpaint"]
        for bp in base_priority:
            if bp in matched_patterns:
                base_pattern = bp
                break
        if base_pattern == "unknown":
            base_pattern = matched_patterns[0]

    modifiers = sorted(
        [p for p in matched_patterns if p != base_pattern]
    )

    desc_parts = [
        _PATTERN_SIGNATURES.get(base_pattern, {}).get(
            "description", base_pattern
        )
    ]
    for mod in modifiers:
        mod_desc = _PATTERN_SIGNATURES.get(mod, {}).get(
            "description", mod
        )
        desc_parts.append(f"with {mod_desc.lower()}")

    return {
        "base_pattern": base_pattern,
        "modifiers": modifiers,
        "all_patterns": sorted(matched_patterns),
        "description": " ".join(desc_parts),
        "node_count": len(nodes),
        "class_types": class_types,
    }


# ---------------------------------------------------------------------------
# Validation (requires live ComfyUI)
# ---------------------------------------------------------------------------

def _validate_against_comfyui(nodes: dict, connections: list[dict]) -> dict:
    """Validate workflow against running ComfyUI instance."""
    errors = []
    warnings = []

    # Fetch all node info
    try:
        with httpx.Client() as client:
            resp = client.get(f"{COMFYUI_URL}/object_info", timeout=30.0)
            resp.raise_for_status()
            all_info = resp.json()
    except httpx.ConnectError:
        return {"error": f"ComfyUI not reachable at {COMFYUI_URL}. Is it running?"}
    except Exception as e:
        return {"error": str(e)}

    # Check each node
    # He2025: sort for deterministic error order
    for nid, node in sorted(nodes.items()):
        if node.get("_ui_node"):
            continue
        class_type = node.get("class_type", "")

        # Check node type exists
        if class_type not in all_info:
            errors.append(f"Node [{nid}]: class_type '{class_type}' not installed.")
            continue

        node_schema = all_info[class_type]
        required_inputs = node_schema.get("input", {}).get("required", {})
        optional_inputs = node_schema.get("input", {}).get("optional", {})
        all_known_inputs = {**required_inputs, **optional_inputs}

        # Build set of AUTOGROW group names so we can recognize their
        # sub-inputs (e.g. "values.a" belongs to the "values" AUTOGROW input)
        autogrow_groups = {
            name
            for name, spec in all_known_inputs.items()
            if (
                isinstance(spec, (list, tuple))
                and len(spec) > 0
                and spec[0] == "COMFY_AUTOGROW_V3"
            )
        }

        # Check required inputs are provided
        node_inputs = node.get("inputs", {})
        for req_name in required_inputs:
            if req_name not in node_inputs:
                # AUTOGROW inputs may be present as a nested dict or
                # not yet populated (min=0 is valid). Only flag as
                # missing if it's not an AUTOGROW type.
                if req_name not in autogrow_groups:
                    errors.append(
                        f"Node [{nid}] {class_type}: "
                        f"missing required input '{req_name}'."
                    )

        # Check for unknown inputs
        for inp_name in node_inputs:
            if inp_name in all_known_inputs:
                continue
            # AUTOGROW sub-inputs: "values.a" -> group "values"
            if _is_autogrow_dotted_name(inp_name):
                group = inp_name.split(".")[0]
                if group in autogrow_groups:
                    continue
            # Nested dict keys (API-format AUTOGROW) are already matched
            # by the top-level name, so this only fires for truly unknown
            warnings.append(
                f"Node [{nid}] {class_type}: unknown input '{inp_name}' "
                f"(may be from a different version)."
            )

    # Check connection type compatibility
    for conn in connections:
        source_class = conn["from_class"]
        output_idx = conn["from_output"]
        target_class = conn["to_class"]
        target_input = conn["to_input"]

        # Get source output types
        source_info = all_info.get(source_class, {})
        source_outputs = source_info.get("output", [])
        if output_idx < len(source_outputs):
            source_type = source_outputs[output_idx]
        else:
            errors.append(
                f"Connection {source_class}[{output_idx}] -> "
                f"{target_class}.{target_input}: output index out of range."
            )
            continue

        # Get target input type
        target_info = all_info.get(target_class, {})
        all_target_inputs = {
            **target_info.get("input", {}).get("required", {}),
            **target_info.get("input", {}).get("optional", {}),
        }
        target_spec = all_target_inputs.get(target_input)

        # AUTOGROW dotted name: "values.a" -> look up "values" group
        # and use the template's sub-input type for validation
        if target_spec is None and _is_autogrow_dotted_name(target_input):
            group = target_input.split(".")[0]
            group_spec = all_target_inputs.get(group)
            if (
                group_spec
                and isinstance(group_spec, (list, tuple))
                and len(group_spec) > 0
                and group_spec[0] == "COMFY_AUTOGROW_V3"
            ):
                # Extract the template's sub-input type.
                # Schema: ['COMFY_AUTOGROW_V3', {'template': {'input':
                #   {'required': {'value': ['FLOAT,INT', {}]}}}, ...}]
                tmpl = group_spec[1].get("template", {}) if len(group_spec) > 1 else {}
                tmpl_inputs = tmpl.get("input", {}).get("required", {})
                # Use first template input's type spec
                if tmpl_inputs:
                    first_tmpl = next(iter(tmpl_inputs.values()))
                    target_spec = first_tmpl
        if target_spec and isinstance(target_spec, (list, tuple)) and len(target_spec) > 0:
            target_type = target_spec[0]
            # Check type compatibility (exact match or wildcard "*")
            if (
                isinstance(target_type, str)
                and isinstance(source_type, str)
                and target_type != "*"
                and source_type != "*"
                and source_type != target_type
            ):
                errors.append(
                    f"Type mismatch: {source_class}.out[{output_idx}] "
                    f"({source_type}) -> {target_class}.{target_input} "
                    f"(expects {target_type})."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_load_workflow(tool_input: dict) -> str:
    path_str = tool_input["path"]
    data, err = _load_json(path_str)
    if err:
        return to_json({"error": err})

    nodes, fmt = _extract_api_format(data)
    connections = _trace_connections(nodes)
    editable = _find_editable_fields(nodes)
    classification = _classify_pattern(nodes)

    # Extract component/subgraph definitions if present
    subgraph_info = _extract_subgraph_nodes(data)
    if subgraph_info:
        classification["has_components"] = True
        classification["component_count"] = len(subgraph_info)

    summary = _build_summary(
        nodes, connections, fmt, classification, subgraph_info
    )

    # Build node list for output
    node_list = {}
    for nid, node in nodes.items():
        entry = {
            "class_type": node.get("class_type", ""),
        }
        if _is_component_node(entry["class_type"]):
            entry["is_component"] = True
        if not node.get("_ui_node"):
            entry["input_count"] = len(node.get("inputs", {}))
        node_list[nid] = entry

    result = {
        "file": path_str,
        "format": fmt,
        "node_count": len(nodes),
        "connection_count": len(connections),
        "editable_field_count": len(editable),
        "classification": classification,
        "summary": summary,
        "nodes": node_list,
        "connections": connections,
        "editable_fields": editable,
    }
    if subgraph_info:
        result["components"] = subgraph_info

    return to_json(result)


def _handle_validate_workflow(tool_input: dict) -> str:
    path_str = tool_input["path"]
    data, err = _load_json(path_str)
    if err:
        return to_json({"error": err})

    nodes, fmt = _extract_api_format(data)
    if fmt == "ui_only":
        return to_json({
            "error": (
                "UI-only workflow format — can't validate without API format data. "
                "Re-export the workflow using 'Save (API Format)' in ComfyUI, "
                "or use a workflow that has extra.prompt embedded."
            ),
        })

    connections = _trace_connections(nodes)
    result = _validate_against_comfyui(nodes, connections)
    return to_json(result)


def _handle_classify_workflow(tool_input: dict) -> str:
    path_str = tool_input["path"]
    data, err = _load_json(path_str)
    if err:
        return to_json({"error": err})

    nodes, fmt = _extract_api_format(data)
    classification = _classify_pattern(nodes)
    classification["file"] = path_str
    classification["format"] = fmt

    # Flag component-based workflows
    subgraph_info = _extract_subgraph_nodes(data)
    if subgraph_info:
        classification["has_components"] = True
        classification["component_count"] = len(subgraph_info)
        # Include subgraph class_types so the caller knows what's inside
        classification["component_class_types"] = sorted(
            _all_subgraph_class_types(subgraph_info)
        )

    return to_json(classification)


def _handle_get_editable_fields(tool_input: dict) -> str:
    path_str = tool_input["path"]
    class_filter = tool_input.get("class_filter", "")

    data, err = _load_json(path_str)
    if err:
        return to_json({"error": err})

    nodes, fmt = _extract_api_format(data)
    if fmt == "ui_only":
        return to_json({
            "error": (
                "UI-only format — field values aren't available. "
                "Use an API-format workflow or one with extra.prompt embedded."
            ),
        })

    fields = _find_editable_fields(nodes, class_filter)

    # Group by class_type for readability
    by_class: dict[str, list[dict]] = {}
    for f in fields:
        by_class.setdefault(f["class_type"], []).append(f)

    return to_json({
        "file": path_str,
        "total_fields": len(fields),
        "fields_by_class": by_class,
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def summarize_workflow_data(data: dict) -> dict:
    """Summarize a raw workflow dict without filesystem I/O.

    Called by the sidebar backend to build context for the system prompt.
    Returns a structured summary dict.
    """
    nodes, fmt = _extract_api_format(data)
    connections = _trace_connections(nodes)
    editable = _find_editable_fields(nodes)
    classification = _classify_pattern(nodes)

    # Extract component/subgraph definitions if present
    subgraph_info = _extract_subgraph_nodes(data)
    if subgraph_info:
        classification["has_components"] = True
        classification["component_count"] = len(subgraph_info)

    summary = _build_summary(
        nodes, connections, fmt, classification, subgraph_info
    )

    node_list = {}
    for nid, node in sorted(nodes.items()):
        entry = {"class_type": node.get("class_type", "")}
        if _is_component_node(entry["class_type"]):
            entry["is_component"] = True
        if not node.get("_ui_node"):
            entry["input_count"] = len(node.get("inputs", {}))
        node_list[nid] = entry

    result = {
        "format": fmt,
        "node_count": len(nodes),
        "connection_count": len(connections),
        "editable_field_count": len(editable),
        "classification": classification,
        "summary": summary,
        "nodes": node_list,
        "editable_fields": editable,
    }
    if subgraph_info:
        result["components"] = subgraph_info

    return result


def handle(name: str, tool_input: dict) -> str:
    """Execute a workflow_parse tool call."""
    try:
        if name == "load_workflow":
            return _handle_load_workflow(tool_input)
        elif name == "validate_workflow":
            return _handle_validate_workflow(tool_input)
        elif name == "classify_workflow":
            return _handle_classify_workflow(tool_input)
        elif name == "get_editable_fields":
            return _handle_get_editable_fields(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
