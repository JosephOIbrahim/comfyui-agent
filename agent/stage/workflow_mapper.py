"""Bidirectional workflow JSON <-> USD prim mapper.

Translates between ComfyUI API-format workflow JSON and USD prim hierarchy.
Round-trip fidelity: JSON -> prims -> JSON produces functionally identical output.

USD prim layout:
  /workflows/{name}/
    /nodes/
      /node_{id}    (class_type, node_id attributes)
        input:{name} = literal_value
        rel conn:{name} -> /workflows/{name}/nodes/node_{source_id}
        int conn:{name}:slot = output_index
"""

from __future__ import annotations

from typing import Any

try:
    from pxr import Sdf
    HAS_USD = True
except ImportError:
    HAS_USD = False

from .cognitive_stage import CognitiveWorkflowStage, StageError


def _is_connection(value: Any) -> bool:
    """Check if a value is a ComfyUI connection reference [node_id, slot]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _safe_prim_name(node_id: str) -> str:
    """Convert a node ID to a valid USD prim name (can't start with digit)."""
    return f"node_{node_id}"


def _node_id_from_prim(name: str) -> str:
    """Extract original node ID from a prim name."""
    if name.startswith("node_"):
        return name[5:]
    return name


def workflow_json_to_prims(
    cws: CognitiveWorkflowStage,
    workflow_json: dict[str, dict],
    workflow_name: str,
) -> str:
    """Convert ComfyUI API-format workflow JSON to USD prims.

    Each node becomes a prim. Literal inputs become attributes.
    Connection inputs become USD relationships + slot attributes.

    Args:
        cws: The CognitiveWorkflowStage to write into.
        workflow_json: ComfyUI API-format JSON (node_id -> node_data).
        workflow_name: Name for this workflow (used in prim path).

    Returns:
        Base prim path of the created workflow (e.g. "/workflows/my_wf").
    """
    if not HAS_USD:
        raise StageError("USD not available")

    base = f"/workflows/{workflow_name}"
    nodes_base = f"{base}/nodes"

    # Write workflow metadata to base layer
    cws.write(base, "node_count", len(workflow_json))

    # Create all node prims with class_type and node_id
    for node_id, node_data in workflow_json.items():
        safe_name = _safe_prim_name(node_id)
        prim_path = f"{nodes_base}/{safe_name}"

        class_type = node_data.get("class_type", "")
        cws.write(prim_path, "class_type", class_type)
        cws.write(prim_path, "node_id", str(node_id))

        # Process inputs
        inputs = node_data.get("inputs", {})
        for input_name, value in inputs.items():
            if value is None:
                continue

            if _is_connection(value):
                # Connection: [source_node_id, output_index]
                source_id, output_idx = value
                source_path = f"{nodes_base}/{_safe_prim_name(source_id)}"

                # Use Sdf API via base layer for relationships
                _write_connection(
                    cws, prim_path, input_name, source_path, output_idx
                )
            else:
                # Literal value
                _write_literal_input(cws, prim_path, input_name, value)

    return base


def _write_literal_input(
    cws: CognitiveWorkflowStage,
    prim_path: str,
    input_name: str,
    value: Any,
) -> None:
    """Write a literal input value as a namespaced attribute."""
    attr_name = f"input:{input_name}"

    # Handle types that CognitiveWorkflowStage.write() supports
    if isinstance(value, (str, int, float, bool)):
        cws.write(prim_path, attr_name, value)
    else:
        # Fallback: serialize as string for unsupported types
        cws.write(prim_path, attr_name, str(value))


def _write_connection(
    cws: CognitiveWorkflowStage,
    prim_path: str,
    input_name: str,
    source_path: str,
    output_idx: int,
) -> None:
    """Write a connection as a USD relationship + slot attribute."""
    base_layer = cws.base_layer

    # Ensure prim spec exists in base layer
    prim_spec = Sdf.CreatePrimInLayer(base_layer, prim_path)
    if prim_spec.specifier == Sdf.SpecifierOver:
        prim_spec.specifier = Sdf.SpecifierDef

    # Create relationship: conn:{input_name} -> source prim
    rel_name = f"conn:{input_name}"
    rel_spec = Sdf.RelationshipSpec(prim_spec, rel_name)
    rel_spec.targetPathList.explicitItems = [Sdf.Path(source_path)]

    # Store output slot index as attribute
    slot_name = f"conn:{input_name}:slot"
    slot_spec = Sdf.AttributeSpec(
        prim_spec, slot_name, Sdf.ValueTypeNames.Int
    )
    slot_spec.default = output_idx


def prims_to_workflow_json(
    cws: CognitiveWorkflowStage,
    workflow_name: str,
) -> dict[str, dict]:
    """Convert USD prims back to ComfyUI API-format workflow JSON.

    Reads the composed stage (all LIVRPS resolved) and reconstructs
    the workflow JSON ready for POST to ComfyUI.

    Args:
        cws: The CognitiveWorkflowStage to read from.
        workflow_name: Name of the workflow to extract.

    Returns:
        ComfyUI API-format JSON dict (node_id -> node_data).
    """
    if not HAS_USD:
        raise StageError("USD not available")

    nodes_base = f"/workflows/{workflow_name}/nodes"
    nodes_prim = cws.stage.GetPrimAtPath(nodes_base)
    if not nodes_prim.IsValid():
        return {}

    result = {}

    for child in nodes_prim.GetChildren():
        # Read node metadata
        node_id_attr = child.GetAttribute("node_id")
        class_type_attr = child.GetAttribute("class_type")

        if not node_id_attr.IsValid() or not class_type_attr.IsValid():
            continue

        node_id = str(node_id_attr.Get())
        class_type = str(class_type_attr.Get())

        inputs: dict[str, Any] = {}

        # Collect literal inputs (input:* namespace)
        for attr in child.GetAttributes():
            name = attr.GetName()
            if name.startswith("input:"):
                input_name = name[6:]  # strip "input:"
                val = attr.Get()
                if val is not None:
                    inputs[input_name] = _usd_to_python(val)

        # Collect connections (conn:* relationships)
        for rel in child.GetRelationships():
            name = rel.GetName()
            if not name.startswith("conn:"):
                continue

            input_name = name[5:]  # strip "conn:"
            targets = rel.GetTargets()
            if not targets:
                continue

            # Resolve source node ID from target prim
            target_prim = cws.stage.GetPrimAtPath(targets[0])
            if not target_prim.IsValid():
                continue

            source_id_attr = target_prim.GetAttribute("node_id")
            if not source_id_attr.IsValid():
                continue

            source_id = str(source_id_attr.Get())

            # Get output slot index
            slot_attr = child.GetAttribute(f"conn:{input_name}:slot")
            slot = int(slot_attr.Get()) if slot_attr.IsValid() else 0

            inputs[input_name] = [source_id, slot]

        result[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }

    return result


def _usd_to_python(value: Any) -> Any:
    """Convert USD attribute value to JSON-compatible Python type."""
    # USD returns Python-compatible types for scalars, but some may need
    # explicit conversion (e.g., int64 -> int for JSON serialization)
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value
