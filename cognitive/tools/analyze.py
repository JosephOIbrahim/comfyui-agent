"""analyze_workflow — Semantic analysis + validation + resource estimate.

Absorbs UNDERSTAND tools into a single high-level analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowAnalysis:
    """Result of analyzing a workflow."""

    node_count: int = 0
    node_types: list[str] = field(default_factory=list)
    connections: list[dict[str, str]] = field(default_factory=list)
    editable_fields: list[dict[str, Any]] = field(default_factory=list)
    model_family: str = ""
    classification: str = ""  # e.g. "txt2img", "img2img", "inpainting"
    validation_errors: list[str] = field(default_factory=list)
    is_valid: bool = True
    estimated_vram_mb: int = 0
    summary: str = ""


def analyze_workflow(
    workflow_data: dict[str, Any],
    schema_cache: Any | None = None,
) -> WorkflowAnalysis:
    """Analyze a ComfyUI workflow for structure, validity, and resource needs.

    Args:
        workflow_data: ComfyUI API format workflow dict.
        schema_cache: Optional SchemaCache for enhanced validation.

    Returns:
        WorkflowAnalysis with all discovered information.
    """
    analysis = WorkflowAnalysis()

    nodes = {
        k: v for k, v in workflow_data.items()
        if isinstance(v, dict) and "class_type" in v
    }
    analysis.node_count = len(nodes)
    analysis.node_types = sorted(set(v["class_type"] for v in nodes.values()))

    # Extract connections
    for node_id, node_data in sorted(nodes.items()):
        for inp_name, inp_val in node_data.get("inputs", {}).items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_id, src_idx = inp_val
                if isinstance(src_id, str) and isinstance(src_idx, int):
                    analysis.connections.append({
                        "from_node": src_id,
                        "from_output": str(src_idx),
                        "to_node": node_id,
                        "to_input": inp_name,
                    })

    # Extract editable fields (non-connection inputs)
    for node_id, node_data in sorted(nodes.items()):
        ct = node_data["class_type"]
        for inp_name, inp_val in sorted(node_data.get("inputs", {}).items()):
            if isinstance(inp_val, list) and len(inp_val) == 2:
                if isinstance(inp_val[0], str) and isinstance(inp_val[1], int):
                    continue  # Skip connections
            analysis.editable_fields.append({
                "node_id": node_id,
                "class_type": ct,
                "input": inp_name,
                "value": inp_val,
            })

    # Classify workflow
    analysis.classification = _classify_workflow(nodes)

    # Detect model family
    analysis.model_family = _detect_model_family(nodes)

    # Schema validation if available
    if schema_cache is not None and hasattr(schema_cache, "validate_mutation"):
        for node_id, node_data in nodes.items():
            ct = node_data["class_type"]
            for inp_name, inp_val in node_data.get("inputs", {}).items():
                if isinstance(inp_val, list):
                    continue  # Skip connections
                valid, reason = schema_cache.validate_mutation(ct, inp_name, inp_val)
                if not valid:
                    analysis.validation_errors.append(
                        f"Node {node_id} ({ct}).{inp_name}: {reason}"
                    )
        if analysis.validation_errors:
            analysis.is_valid = False

    # Summary
    analysis.summary = (
        f"{analysis.classification} workflow with {analysis.node_count} nodes "
        f"({analysis.model_family or 'unknown family'}). "
        f"{len(analysis.connections)} connections, "
        f"{len(analysis.editable_fields)} editable fields."
    )

    return analysis


def _classify_workflow(nodes: dict[str, dict]) -> str:
    """Classify the workflow type based on node presence."""
    class_types = {v["class_type"] for v in nodes.values()}

    has_empty_latent = any("EmptyLatent" in ct for ct in class_types)
    has_load_image = any("LoadImage" in ct for ct in class_types)
    has_inpaint = any("inpaint" in ct.lower() for ct in class_types)
    has_controlnet = any("controlnet" in ct.lower() for ct in class_types)
    has_video = any(
        kw in ct.lower()
        for ct in class_types
        for kw in ("video", "animatediff", "svd", "ltx", "wan")
    )

    if has_video:
        return "video_generation"
    if has_inpaint:
        return "inpainting"
    if has_load_image and has_controlnet:
        return "controlnet_img2img"
    if has_load_image:
        return "img2img"
    if has_empty_latent:
        return "txt2img"
    return "unknown"


def _detect_model_family(nodes: dict[str, dict]) -> str:
    """Detect the model family from node types and input values."""
    class_types = {v["class_type"] for v in nodes.values()}

    if any("flux" in ct.lower() for ct in class_types):
        return "Flux"
    if any("sd3" in ct.lower() for ct in class_types):
        return "SD3"

    # Check checkpoint names for family hints
    for node_data in nodes.values():
        ckpt = node_data.get("inputs", {}).get("ckpt_name", "")
        if isinstance(ckpt, str):
            lower = ckpt.lower()
            if "flux" in lower:
                return "Flux"
            if "sdxl" in lower or "sd_xl" in lower:
                return "SDXL"
            if "sd3" in lower:
                return "SD3"

    # Check resolution for hints
    for node_data in nodes.values():
        inputs = node_data.get("inputs", {})
        w = inputs.get("width", 0)
        h = inputs.get("height", 0)
        if isinstance(w, int) and isinstance(h, int):
            if w >= 1024 or h >= 1024:
                return "SDXL"
            if w == 512 and h == 512:
                return "SD1.5"

    return ""
