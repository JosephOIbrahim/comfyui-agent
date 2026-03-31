# TRACK A DESIGN — State Spine

**Agent:** `[GRAPH x ARCHITECT]`
**Date:** 2026-03-31

---

## Directory Structure

```
src/
└── cognitive/
    ├── __init__.py          # Package marker, exports CognitiveGraphEngine
    └── core/
        ├── __init__.py      # Exports models, delta, graph
        ├── models.py        # ComfyNode, WorkflowGraph
        ├── delta.py         # DeltaLayer with SHA-256 integrity
        └── graph.py         # CognitiveGraphEngine with LIVRPS resolver
```

---

## LIVRPS Priority (Inverted S — Safety Strongest)

```python
from typing import Literal

Opinion = Literal["P", "R", "V", "I", "L", "S"]

LIVRPS_PRIORITY: dict[str, int] = {
    "P": 1,  # Payloads — deep archive, loaded on demand
    "R": 2,  # References — base templates, prior rules
    "V": 3,  # VariantSets — context-dependent alternatives
    "I": 4,  # Inherits — experience-derived patterns
    "L": 5,  # Local — current session edits (strongest creative opinion)
    "S": 6,  # Safety — structural constraints (INVERTED: always wins)
}
```

---

## models.py — ComfyNode & WorkflowGraph

```python
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
        """Parse from ComfyUI API format: {"class_type": ..., "inputs": {...}}"""
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
        """Parse from ComfyUI API JSON (top-level dict of node_id -> node_data)."""
        nodes = {}
        for node_id, node_data in data.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                nodes[node_id] = ComfyNode.from_api_dict(node_id, node_data)
        return cls(nodes=nodes)

    def to_api_json(self) -> dict[str, Any]:
        """Serialize to ComfyUI API JSON format.

        CRITICAL: Link arrays (["node_id", output_index]) must be preserved
        exactly as they appear in the inputs. The deepcopy in ComfyNode handles this.
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
```

---

## delta.py — DeltaLayer

```python
from __future__ import annotations
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Opinion = Literal["P", "R", "V", "I", "L", "S"]

LIVRPS_PRIORITY: dict[str, int] = {
    "P": 1, "R": 2, "V": 3, "I": 4, "L": 5, "S": 6,
}


def _compute_hash(opinion: str, mutations: dict[str, dict[str, Any]]) -> str:
    """SHA-256 of opinion + deterministic JSON of mutations."""
    payload = json.dumps(
        {"opinion": opinion, "mutations": mutations},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class DeltaLayer:
    """Non-destructive mutation layer with LIVRPS opinion and integrity check."""
    layer_id: str
    opinion: Opinion
    timestamp: float
    description: str
    mutations: dict[str, dict[str, Any]]
    # mutations format: {node_id: {param_name: value, ...}, ...}
    # A mutation can also include "class_type" to inject a new node.
    creation_hash: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.creation_hash:
            self.creation_hash = _compute_hash(self.opinion, self.mutations)

    @property
    def layer_hash(self) -> str:
        """Recompute hash from current state. Compare with creation_hash for tamper detection."""
        return _compute_hash(self.opinion, self.mutations)

    @property
    def is_intact(self) -> bool:
        """True if layer has not been tampered with since creation."""
        return self.creation_hash == self.layer_hash

    @property
    def priority(self) -> int:
        """Numeric priority from LIVRPS_PRIORITY."""
        return LIVRPS_PRIORITY[self.opinion]

    @classmethod
    def create(
        cls,
        mutations: dict[str, dict[str, Any]],
        opinion: Opinion = "L",
        description: str = "",
        layer_id: str | None = None,
    ) -> DeltaLayer:
        """Factory method with auto-generated layer_id and timestamp."""
        return cls(
            layer_id=layer_id or uuid.uuid4().hex[:12],
            opinion=opinion,
            timestamp=time.time(),
            description=description,
            mutations=mutations,
        )
```

---

## graph.py — CognitiveGraphEngine

```python
from __future__ import annotations
import copy
from typing import Any, Literal

from .models import WorkflowGraph, ComfyNode
from .delta import DeltaLayer, LIVRPS_PRIORITY, Opinion


class CognitiveGraphEngine:
    """LIVRPS composition engine for non-destructive workflow mutation.

    Manages a base workflow and a stack of delta layers. The resolved
    graph applies deltas weakest-to-strongest (last write wins = strongest wins).
    """

    def __init__(self, base_workflow_data: dict[str, Any]):
        """Initialize with raw ComfyUI API JSON.

        Args:
            base_workflow_data: ComfyUI API format dict {node_id: {class_type, inputs}}
        """
        self._base = WorkflowGraph.from_api_json(base_workflow_data)
        self._base_raw = copy.deepcopy(base_workflow_data)  # Keep raw for fidelity
        self._delta_stack: list[DeltaLayer] = []

    @property
    def base(self) -> WorkflowGraph:
        return self._base

    @property
    def delta_stack(self) -> list[DeltaLayer]:
        return list(self._delta_stack)  # Defensive copy

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
                       If a node_id is not in base and "class_type" is in the
                       mutation dict, the node is injected as new.
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
        1. Deep copy base workflow
        2. Sort delta stack by LIVRPS priority (stable sort preserves
           chronological order for same-priority layers)
        3. Apply mutations weakest-to-strongest (strongest writes last = wins)
        4. For each mutation: update only specified keys in node inputs,
           preserving all other inputs and link arrays
        5. If mutation references a node not in base AND includes class_type:
           inject as new node

        Args:
            up_to_index: If provided, only consider deltas up to this index
                         (exclusive) in the original insertion order.
        """
        resolved = self._resolve_from_raw(up_to_index)
        return WorkflowGraph.from_api_json(resolved)

    def _resolve_from_raw(self, up_to_index: int | None = None) -> dict[str, Any]:
        """Internal resolution that works on raw dicts for maximum fidelity.

        This preserves link arrays exactly as they appear in the original JSON.
        """
        # Deep copy base (raw dict, preserves all original structure)
        result = copy.deepcopy(self._base_raw)

        # Select deltas
        deltas = self._delta_stack[:up_to_index]

        # Sort by LIVRPS priority (stable sort: same priority preserves insertion order)
        sorted_deltas = sorted(deltas, key=lambda d: d.priority)

        # Apply weakest-to-strongest
        for delta in sorted_deltas:
            for node_id, params in delta.mutations.items():
                if node_id in result:
                    # Existing node: update only specified input keys
                    node = result[node_id]
                    inputs = node.setdefault("inputs", {})
                    for param_name, param_value in params.items():
                        if param_name == "class_type":
                            continue  # class_type is not an input
                        inputs[param_name] = copy.deepcopy(param_value)
                else:
                    # New node injection: requires class_type in mutations
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
                    # If no class_type, silently skip (can't inject without type)

        return result

    def verify_stack_integrity(self) -> tuple[bool, list[str]]:
        """Check all delta layers for tampering.

        Returns:
            (all_intact, list_of_error_messages)
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
        """Get the fully resolved workflow as ComfyUI API JSON."""
        return self._resolve_from_raw()
```

---

## Wrapper Contracts (workflow_patch.py modifications)

### Engine Persistence
```python
# In SessionContext (session_context.py), add:
self._graph_engine: CognitiveGraphEngine | None = None

def ensure_graph_engine(self, workflow_data: dict) -> CognitiveGraphEngine:
    """Get or create engine. Resets if base workflow changes."""
    if self._graph_engine is None:
        from src.cognitive.core.graph import CognitiveGraphEngine
        self._graph_engine = CognitiveGraphEngine(workflow_data)
    return self._graph_engine

@property
def graph_engine(self) -> CognitiveGraphEngine | None:
    return self._graph_engine

@graph_engine.setter
def graph_engine(self, value):
    self._graph_engine = value
```

### Tool Wrapper Mapping

| Tool | Current Implementation | Wrapped Implementation |
|------|----------------------|----------------------|
| `apply_workflow_patch` | `jsonpatch.JsonPatch(patches).apply(current_workflow)` | Translate RFC6902 patches to `{node_id: {param: value}}` mutations, call `engine.mutate_workflow()`, update `_state["current_workflow"]` from `engine.to_api_json()` |
| `add_node` | Direct dict insertion | `engine.mutate_workflow({new_id: {"class_type": ct, **inputs}})`, update state |
| `connect_nodes` | Direct `inputs[to_input] = [from, idx]` | `engine.mutate_workflow({to_node: {to_input: [from_node, from_output]}})`, update state |
| `set_input` | Direct `inputs[name] = value` | `engine.mutate_workflow({node_id: {input_name: value}})`, update state |
| `undo_workflow_patch` | `history.pop()` | `engine.pop_delta()`, update `_state["current_workflow"]` from `engine.to_api_json()` |
| `reset_workflow` | `deepcopy(base)` | Reset engine (new instance from base), update state |

### Backward Compatibility Rules
1. `_state["current_workflow"]` is ALWAYS kept in sync with `engine.to_api_json()` after every mutation
2. `_state["base_workflow"]` remains unchanged (used for diff computation)
3. `_state["history"]` list is kept for backward compat but engine's delta stack is the source of truth
4. `get_current_workflow()` returns `engine.to_api_json()` when engine exists, falls back to `_state["current_workflow"]`
5. All return value JSON formats remain identical

---

## Test Specifications (for CRUCIBLE)

### Required Tests — `tests/test_cognitive_core.py`

1. **Link preservation**: Create workflow with `["4", 0]` link in inputs. Mutate a different input on same node. Verify link array survives unchanged.

2. **LIVRPS strongest-opinion-wins**: Apply L-opinion delta setting `cfg=7`, then S-opinion delta setting `cfg=1`. Resolve. Assert `cfg==1` (S wins).

3. **SHA-256 tamper detection**: Create delta, manually modify `delta.mutations`, call `verify_stack_integrity()`. Assert returns `(False, [error_msg])`.

4. **Temporal query rollback**: Push 3 deltas. `temporal_query(back_steps=1)` should match `get_resolved_graph(up_to_index=2)`. `temporal_query(back_steps=3)` should match base.

5. **Multi-node atomic mutations**: Single `mutate_workflow()` call modifying 3 different nodes. All 3 changes appear in resolved graph.

6. **Node injection**: Delta with `{"99": {"class_type": "NewNode", "param": "val"}}` for node "99" not in base. Verify node appears in resolved graph with correct class_type and inputs.

7. **Empty delta stack**: `get_resolved_graph()` with no deltas returns deep copy of base, not a reference.

8. **Same-opinion chronological ordering**: Two L-opinion deltas both setting `steps` on same node. Second delta's value wins (chronological order preserved by stable sort).

9. **Round-trip fidelity**: `from_api_json(data).to_api_json()` produces identical structure to input (modulo key ordering). Then mutate, resolve, serialize, parse again — compare.

10. **Deep copy isolation**: Get resolved graph, mutate it directly, verify original engine state unchanged.

11. **Regression**: Full existing test suite (2140) still passes.

---

## GATE: Design complete. Proceeding to FORGE per user instruction.
