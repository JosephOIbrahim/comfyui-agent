"""ComfyNode and WorkflowGraph models.

Typed representations of ComfyUI workflow nodes and graphs.
Link arrays (["node_id", output_index]) are preserved through
all parsing, mutation, and serialization operations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComfyNode:
    """Single node in a ComfyUI workflow."""

    node_id: str
    class_type: str
    inputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_dict(cls, node_id: str, data: dict[str, Any]) -> ComfyNode:
        """Parse from ComfyUI API format: {"class_type": ..., "inputs": {...}}."""
        return cls(
            node_id=node_id,
            class_type=data["class_type"],
            inputs=copy.deepcopy(data.get("inputs", {})),
        )

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to ComfyUI API format."""
        return {
            "class_type": self.class_type,
            "inputs": copy.deepcopy(self.inputs),
        }


@dataclass
class WorkflowGraph:
    """Complete workflow as a dict of nodes keyed by node_id."""

    nodes: dict[str, ComfyNode] = field(default_factory=dict)

    @classmethod
    def from_api_json(cls, data: dict[str, Any]) -> WorkflowGraph:
        """Parse from ComfyUI API JSON (top-level dict of node_id -> node_data).

        Only entries with a "class_type" key are treated as nodes.
        Link arrays in inputs are preserved via deepcopy.
        """
        nodes = {}
        for node_id, node_data in data.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                nodes[node_id] = ComfyNode.from_api_dict(node_id, node_data)
        return cls(nodes=nodes)

    def to_api_json(self) -> dict[str, Any]:
        """Serialize to ComfyUI API JSON format.

        CRITICAL: Link arrays (["node_id", output_index]) are preserved
        exactly as they appear in the inputs via deepcopy in ComfyNode.
        """
        return {
            node_id: node.to_api_dict()
            for node_id, node in sorted(self.nodes.items(), key=lambda x: x[0])
        }

    def deep_copy(self) -> WorkflowGraph:
        """Return a fully independent copy."""
        return WorkflowGraph(
            nodes={
                nid: ComfyNode(
                    node_id=node.node_id,
                    class_type=node.class_type,
                    inputs=copy.deepcopy(node.inputs),
                )
                for nid, node in self.nodes.items()
            }
        )
