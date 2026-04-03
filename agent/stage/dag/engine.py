"""Core DAG engine — builds and evaluates the Workflow Intelligence DAG.

The DAG is a ``networkx.DiGraph`` where each node is a named computation
step and edges encode data dependencies.  ``evaluate_dag`` walks the
graph in topological order, calling each compute function with its
upstream results, and assembles the final ``WorkflowIntelligence``.

Dependency graph::

    complexity ──┐
                 ├──► model_reqs ──┬──► optimization
                 │                 ├──► risk ──► readiness
                 │                 │
    (workflow)───┘                 │
                                   │
    tool_scope ◄───────────────────┘

``tool_scope`` depends on ``risk`` and ``readiness`` (plus workflow-level
booleans) but is computed last so it can see the full picture.
"""

from __future__ import annotations

from typing import Any

try:
    import networkx as nx

    HAS_NX = True
except ImportError:  # pragma: no cover
    HAS_NX = False

from .compute_complexity import compute_complexity
from .compute_model_reqs import compute_model_reqs
from .compute_optimization import compute_optimization
from .compute_readiness import compute_readiness
from .compute_risk import compute_risk
from .compute_tool_scope import compute_tool_scope
from .schemas import (
    ComplexityLevel,
    ModelRequirements,
    OptimizationVector,
    ReadinessGrade,
    RiskLevel,
    WorkflowIntelligence,
)


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

# Node names in the DAG
_COMPLEXITY = "complexity"
_MODEL_REQS = "model_reqs"
_OPTIMIZATION = "optimization"
_RISK = "risk"
_READINESS = "readiness"
_TOOL_SCOPE = "tool_scope"


def build_dag() -> Any:
    """Build the Workflow Intelligence DAG.

    Returns a ``networkx.DiGraph`` encoding the dependency graph between
    compute steps.  Each node carries a ``compute`` attribute pointing to
    the function that produces its result.

    Raises:
        RuntimeError: If networkx is not installed.
    """
    if not HAS_NX:
        raise RuntimeError(
            "networkx is required for the Workflow Intelligence DAG. "
            "Install it with: pip install networkx"
        )

    dag: nx.DiGraph = nx.DiGraph()

    # Nodes (compute functions are attached as attributes for introspection
    # but are not called by networkx — evaluate_dag drives execution)
    dag.add_node(_COMPLEXITY, compute=compute_complexity)
    dag.add_node(_MODEL_REQS, compute=compute_model_reqs)
    dag.add_node(_OPTIMIZATION, compute=compute_optimization)
    dag.add_node(_RISK, compute=compute_risk)
    dag.add_node(_READINESS, compute=compute_readiness)
    dag.add_node(_TOOL_SCOPE, compute=compute_tool_scope)

    # Edges (data dependencies)
    dag.add_edge(_COMPLEXITY, _OPTIMIZATION)    # optimization needs complexity
    dag.add_edge(_MODEL_REQS, _OPTIMIZATION)    # optimization needs model_reqs
    dag.add_edge(_MODEL_REQS, _RISK)            # risk needs model_reqs
    dag.add_edge(_RISK, _READINESS)             # readiness needs risk
    dag.add_edge(_RISK, _TOOL_SCOPE)            # tool_scope needs risk
    dag.add_edge(_READINESS, _TOOL_SCOPE)       # tool_scope needs readiness

    assert nx.is_directed_acyclic_graph(dag), "Intelligence DAG has a cycle"
    return dag


# ---------------------------------------------------------------------------
# DAG evaluation
# ---------------------------------------------------------------------------


def evaluate_dag(
    dag: Any,
    workflow_json: dict,
    *,
    node_registry: dict | None = None,
    system_stats: dict | None = None,
    workflow_validated: bool = False,
    workflow_executed: bool = False,
) -> WorkflowIntelligence:
    """Evaluate all DAG nodes in topological order.

    Pure function — no side effects.  Each compute step receives only its
    declared upstream results plus the raw workflow JSON.

    Args:
        dag: The DiGraph returned by ``build_dag()``.
        workflow_json: ComfyUI API-format workflow JSON.
        node_registry: Optional dict mapping ``class_type`` to node info.
        system_stats: Optional system stats dict (``vram_total_gb``, etc.).
        workflow_validated: Whether the workflow has been validated.
        workflow_executed: Whether the workflow has been executed.

    Returns:
        Fully populated ``WorkflowIntelligence`` (with ``evaluated=True``).
    """
    if not HAS_NX:
        raise RuntimeError("networkx is required for DAG evaluation.")

    import networkx as nx

    results: dict[str, Any] = {}

    for node_name in nx.topological_sort(dag):
        if node_name == _COMPLEXITY:
            results[_COMPLEXITY] = compute_complexity(workflow_json)

        elif node_name == _MODEL_REQS:
            results[_MODEL_REQS] = compute_model_reqs(workflow_json)

        elif node_name == _OPTIMIZATION:
            results[_OPTIMIZATION] = compute_optimization(
                workflow_json,
                results[_COMPLEXITY],
                results[_MODEL_REQS],
                system_stats=system_stats,
            )

        elif node_name == _RISK:
            results[_RISK] = compute_risk(
                workflow_json,
                results[_MODEL_REQS],
                node_registry=node_registry,
                system_stats=system_stats,
            )

        elif node_name == _READINESS:
            # Gather missing nodes for readiness check
            missing: list[str] = []
            if node_registry is not None:
                seen: set[str] = set()
                for nd in workflow_json.values():
                    ct = nd.get("class_type", "")
                    if ct and ct not in seen:
                        seen.add(ct)
                        if ct not in node_registry:
                            missing.append(ct)
            results[_READINESS] = compute_readiness(
                results[_RISK],
                missing_nodes=missing or None,
            )

        elif node_name == _TOOL_SCOPE:
            workflow_loaded = bool(workflow_json)
            results[_TOOL_SCOPE] = compute_tool_scope(
                workflow_loaded=workflow_loaded,
                workflow_validated=workflow_validated,
                workflow_executed=workflow_executed,
                risk=results[_RISK],
                readiness=results[_READINESS],
            )

    return WorkflowIntelligence(
        complexity=results.get(_COMPLEXITY, ComplexityLevel.TRIVIAL),
        model_requirements=results.get(
            _MODEL_REQS, ModelRequirements()
        ),
        optimization=results.get(_OPTIMIZATION, OptimizationVector()),
        risk=results.get(_RISK, RiskLevel.SAFE),
        readiness=results.get(_READINESS, ReadinessGrade.READY),
        tool_scope=results.get(_TOOL_SCOPE, frozenset()),
        evaluated=True,
    )
