"""CognitiveGraphEngine — LIVRPS composition engine.

Manages a base workflow and a stack of delta layers. Resolution
applies deltas weakest-to-strongest (last write wins = strongest
opinion wins). Link arrays are never modified unless explicitly
targeted by a mutation.
"""

from __future__ import annotations

import copy
from typing import Any

from .delta import DeltaLayer, LIVRPS_PRIORITY, Opinion
from .models import WorkflowGraph


class CognitiveGraphEngine:
    """Non-destructive workflow mutation engine with LIVRPS composition.

    The engine holds a frozen base workflow and a stack of delta layers.
    Resolution produces a new WorkflowGraph by applying deltas in
    priority order (weakest first, strongest last = strongest wins).
    """

    def __init__(self, base_workflow_data: dict[str, Any]):
        """Initialize with raw ComfyUI API JSON.

        Args:
            base_workflow_data: ComfyUI API format dict
                {node_id: {"class_type": ..., "inputs": {...}}, ...}
        """
        self._base = WorkflowGraph.from_api_json(base_workflow_data)
        self._base_raw = copy.deepcopy(base_workflow_data)
        self._delta_stack: list[DeltaLayer] = []

    @property
    def base(self) -> WorkflowGraph:
        """The frozen base workflow (never mutated)."""
        return self._base

    @property
    def delta_stack(self) -> list[DeltaLayer]:
        """Defensive copy of the delta stack."""
        return list(self._delta_stack)

    def mutate_workflow(
        self,
        mutations: dict[str, dict[str, Any]],
        opinion: Opinion = "L",
        layer_id: str | None = None,
        description: str = "",
    ) -> DeltaLayer:
        """Create and push a new delta layer.

        Args:
            mutations: {node_id: {param: value, ...}, ...}
                If a node_id is not in the base graph and "class_type"
                is present in the mutation dict, the node is injected.
            opinion: LIVRPS tier for this mutation.
            layer_id: Optional explicit layer ID.
            description: Human-readable description.

        Returns:
            The created DeltaLayer.
        """
        delta = DeltaLayer.create(
            mutations=mutations,
            opinion=opinion,
            layer_id=layer_id,
            description=description,
        )
        self._delta_stack.append(delta)
        return delta

    def get_resolved_graph(self, up_to_index: int | None = None) -> WorkflowGraph:
        """Resolve base + deltas into a single WorkflowGraph.

        Resolution order:
        1. Deep copy base workflow (raw dict for maximum fidelity)
        2. Sort deltas by LIVRPS priority (stable sort preserves
           chronological order for same-priority layers)
        3. Apply mutations weakest-to-strongest (strongest writes last = wins)
        4. For each mutation: update only specified keys in node inputs,
           preserving all other inputs and link arrays
        5. If mutation references a node not in base AND includes class_type:
           inject as new node

        Args:
            up_to_index: If provided, only consider deltas[0:up_to_index]
                in the original insertion order.
        """
        resolved = self._resolve_from_raw(up_to_index)
        return WorkflowGraph.from_api_json(resolved)

    def _resolve_from_raw(self, up_to_index: int | None = None) -> dict[str, Any]:
        """Internal resolution on raw dicts for maximum link fidelity."""
        result = copy.deepcopy(self._base_raw)

        deltas = self._delta_stack[:up_to_index]

        # Stable sort by LIVRPS priority: same priority preserves insertion order
        sorted_deltas = sorted(deltas, key=lambda d: d.priority)

        for delta in sorted_deltas:
            for node_id, params in delta.mutations.items():
                if node_id in result:
                    # Existing node: update only specified input keys
                    node = result[node_id]
                    inputs = node.setdefault("inputs", {})
                    for param_name, param_value in params.items():
                        if param_name == "class_type":
                            continue
                        inputs[param_name] = copy.deepcopy(param_value)
                else:
                    # New node injection: requires class_type
                    if "class_type" in params:
                        new_inputs = {
                            k: copy.deepcopy(v)
                            for k, v in params.items()
                            if k != "class_type"
                        }
                        result[node_id] = {
                            "class_type": params["class_type"],
                            "inputs": new_inputs,
                        }

        return result

    def verify_stack_integrity(self) -> tuple[bool, list[str]]:
        """Check all delta layers for tampering.

        Returns:
            (all_intact, list_of_error_messages)
            Empty error list when all layers are intact.
        """
        errors = []
        for delta in self._delta_stack:
            if not delta.is_intact:
                errors.append(
                    f"Layer {delta.layer_id!r} (opinion={delta.opinion}) "
                    f"has been tampered with: creation_hash != current hash"
                )
        return (len(errors) == 0, errors)

    def temporal_query(self, back_steps: int = 1) -> WorkflowGraph:
        """Get the resolved graph at a previous point in time.

        Args:
            back_steps: Number of delta layers to exclude from the top.
                1 = exclude last delta, 2 = exclude last two, etc.
                0 or negative = return current resolved graph.

        Returns:
            WorkflowGraph resolved with only the older deltas.
        """
        if back_steps <= 0:
            return self.get_resolved_graph()
        idx = max(0, len(self._delta_stack) - back_steps)
        return self.get_resolved_graph(up_to_index=idx)

    def pop_delta(self) -> DeltaLayer | None:
        """Remove and return the most recent delta layer.

        Returns None if the stack is empty.
        """
        if self._delta_stack:
            return self._delta_stack.pop()
        return None

    def to_api_json(self) -> dict[str, Any]:
        """Get the fully resolved workflow as ComfyUI API JSON.

        This is the primary output method — returns a dict ready to
        submit to ComfyUI's /prompt endpoint.
        """
        return self._resolve_from_raw()
