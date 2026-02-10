"""Workflow analysis tools.

Load, analyze, summarize, and validate ComfyUI workflow JSON.
Handles both UI format (from ComfyUI "Save") and API format
(from "Save API Format" or programmatic use).

These tools give the agent deep understanding of what a workflow does,
what's connected to what, and what can be safely modified.
"""

import json
from pathlib import Path

import httpx

from ._util import to_json
from ..config import COMFYUI_URL

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


def _extract_api_format(data: dict) -> tuple[dict, str]:
    """
    Extract API-format node dict from any workflow format.

    Returns (api_nodes, format_name).
    api_nodes: {node_id_str: {"class_type": ..., "inputs": {...}}}
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
            nodes[nid] = {
                "class_type": class_type,
                "inputs": {},
                "_widgets_values": widgets,
                "_ui_node": True,
            }
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
        for input_name, value in sorted(node.get("inputs", {}).items()):
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

        for field_name, value in node.get("inputs", {}).items():
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


def _build_summary(nodes: dict, connections: list[dict], fmt: str) -> str:
    """Build a human-readable summary of the workflow pipeline."""
    if not nodes:
        return "Empty workflow (no nodes found)."

    lines = []

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
            inputs = [c['to_input'] for c in conns]
            lines.append(f"  {target} <- {', '.join(sources)}")

    if fmt == "ui_only":
        lines.append("")
        lines.append(
            "Note: UI-only format — editable field values not available. "
            "Use an API-format workflow for full analysis."
        )

    return "\n".join(lines)


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
    for nid, node in nodes.items():
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

        # Check required inputs are provided
        node_inputs = node.get("inputs", {})
        for req_name in required_inputs:
            if req_name not in node_inputs:
                errors.append(
                    f"Node [{nid}] {class_type}: missing required input '{req_name}'."
                )

        # Check for unknown inputs
        for inp_name in node_inputs:
            if inp_name not in all_known_inputs:
                warnings.append(
                    f"Node [{nid}] {class_type}: unknown input '{inp_name}' "
                    f"(may be from a different version)."
                )

    # Check connection type compatibility
    for conn in connections:
        source_id = conn["from_node"]
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
                f"Connection {source_class}[{output_idx}] → "
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
                    f"({source_type}) → {target_class}.{target_input} "
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
    summary = _build_summary(nodes, connections, fmt)

    # Build node list for output
    node_list = {}
    for nid, node in nodes.items():
        entry = {
            "class_type": node.get("class_type", ""),
        }
        if not node.get("_ui_node"):
            entry["input_count"] = len(node.get("inputs", {}))
        node_list[nid] = entry

    return to_json({
        "file": path_str,
        "format": fmt,
        "node_count": len(nodes),
        "connection_count": len(connections),
        "editable_field_count": len(editable),
        "summary": summary,
        "nodes": node_list,
        "connections": connections,
        "editable_fields": editable,
    })


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

def handle(name: str, tool_input: dict) -> str:
    """Execute a workflow_parse tool call."""
    try:
        if name == "load_workflow":
            return _handle_load_workflow(tool_input)
        elif name == "validate_workflow":
            return _handle_validate_workflow(tool_input)
        elif name == "get_editable_fields":
            return _handle_get_editable_fields(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
