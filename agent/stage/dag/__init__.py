"""Workflow Intelligence DAG — Subsystem 1.

Builds a dependency graph of pure compute functions that analyze a
ComfyUI workflow JSON and produce a ``WorkflowIntelligence`` snapshot:
complexity, model requirements, optimization opportunities, risk,
readiness, and recommended tool scope.

Usage::

    from agent.stage.dag import build_dag, evaluate_dag

    dag = build_dag()
    intel = evaluate_dag(dag, workflow_json)
    print(intel.complexity, intel.risk, intel.readiness)
"""

from .engine import build_dag, evaluate_dag
from .schemas import (
    ComplexityLevel,
    ModelRequirements,
    OptimizationVector,
    ReadinessGrade,
    RiskLevel,
    WorkflowIntelligence,
)

__all__ = [
    "ComplexityLevel",
    "ModelRequirements",
    "OptimizationVector",
    "ReadinessGrade",
    "RiskLevel",
    "WorkflowIntelligence",
    "build_dag",
    "evaluate_dag",
]
