"""Compute workflow topology complexity from raw workflow JSON.

Pure function — no side effects, no external calls.  Analyzes the
workflow graph structure (node count, edge count, branching factor,
depth) and maps it to a ``ComplexityLevel`` band.
"""

from __future__ import annotations

from .schemas import ComplexityLevel


# ---------------------------------------------------------------------------
# Thresholds (calibrated to ComfyUI workflows in the wild)
# ---------------------------------------------------------------------------

_NODE_THRESHOLDS: list[tuple[int, ComplexityLevel]] = [
    (5, ComplexityLevel.TRIVIAL),
    (15, ComplexityLevel.SIMPLE),
    (30, ComplexityLevel.MODERATE),
    (50, ComplexityLevel.COMPLEX),
    # Anything above 50 → EXTREME
]


def _is_connection(value: object) -> bool:
    """True if *value* is a ComfyUI connection ``[node_id, slot]``."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _count_edges(workflow: dict) -> int:
    """Count the total number of connection edges in the workflow."""
    edges = 0
    for node_data in workflow.values():
        inputs = node_data.get("inputs", {})
        for val in inputs.values():
            if _is_connection(val):
                edges += 1
    return edges


def _compute_branching_factor(workflow: dict) -> float:
    """Compute max fan-out (outputs consumed) for any single node."""
    if not workflow:
        return 0.0
    out_counts: dict[str, int] = {}
    for node_data in workflow.values():
        inputs = node_data.get("inputs", {})
        for val in inputs.values():
            if _is_connection(val):
                src_id = val[0]
                out_counts[src_id] = out_counts.get(src_id, 0) + 1
    return float(max(out_counts.values())) if out_counts else 0.0


def _compute_depth(workflow: dict) -> int:
    """Compute the longest path (in edges) through the workflow DAG.

    Uses iterative DFS with memoization to avoid stack overflow on
    deeply nested workflows.
    """
    if not workflow:
        return 0

    # Build adjacency: node_id -> list of downstream node_ids
    children: dict[str, list[str]] = {nid: [] for nid in workflow}
    for nid, node_data in workflow.items():
        inputs = node_data.get("inputs", {})
        for val in inputs.values():
            if _is_connection(val):
                src_id = val[0]
                if src_id in children:
                    children[src_id].append(nid)

    memo: dict[str, int] = {}

    def _depth(nid: str) -> int:
        if nid in memo:
            return memo[nid]
        kids = children.get(nid, [])
        if not kids:
            memo[nid] = 0
            return 0
        best = 0
        for kid in kids:
            d = _depth(kid)
            if d > best:
                best = d
        memo[nid] = best + 1
        return best + 1

    # Compute depth from every node, take max
    return max((_depth(nid) for nid in workflow), default=0)


def compute_complexity(workflow: dict) -> ComplexityLevel:
    """Derive ``ComplexityLevel`` from workflow topology.

    Args:
        workflow: ComfyUI API-format workflow JSON.

    Returns:
        The complexity band for the workflow.
    """
    node_count = len(workflow)
    if node_count == 0:
        return ComplexityLevel.TRIVIAL

    # Primary signal: node count
    level = ComplexityLevel.EXTREME
    for threshold, band in _NODE_THRESHOLDS:
        if node_count <= threshold:
            level = band
            break

    # Secondary signal: branching factor can bump complexity up by 1
    branching = _compute_branching_factor(workflow)
    depth = _compute_depth(workflow)
    edge_count = _count_edges(workflow)

    # High branching (>4) or deep graphs (>8) can push up one level
    if (branching > 4.0 or depth > 8) and level < ComplexityLevel.EXTREME:
        level = ComplexityLevel(level + 1)

    # Very dense graphs (edges/nodes > 2.5) can also push up
    ratio = edge_count / node_count if node_count > 0 else 0.0
    if ratio > 2.5 and level < ComplexityLevel.EXTREME:
        level = ComplexityLevel(level + 1)

    return level
