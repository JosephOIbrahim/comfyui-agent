"""compose_workflow — Build workflow from creative intent + experience.

New capability: given an intent description, compose a workflow
by selecting appropriate template, model, and parameters based
on accumulated experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompositionPlan:
    """Plan for composing a workflow from intent."""

    intent: str
    model_family: str = ""
    base_template: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class CompositionResult:
    """Result of workflow composition."""

    success: bool = True
    plan: CompositionPlan | None = None
    workflow_data: dict[str, Any] = field(default_factory=dict)
    mutations_applied: int = 0
    error: str = ""


def compose_workflow(
    intent: str,
    available_templates: list[dict[str, Any]] | None = None,
    experience_patterns: list[dict[str, Any]] | None = None,
    model_family: str | None = None,
) -> CompositionResult:
    """Compose a workflow from creative intent.

    This is the intelligent composition layer. It:
    1. Classifies the intent
    2. Selects a base template
    3. Applies experience-derived parameter choices
    4. Returns a ready-to-execute workflow

    Args:
        intent: Natural language description of desired output.
        available_templates: List of template metadata dicts.
        experience_patterns: Relevant patterns from experience accumulator.
        model_family: Force a specific model family (optional).

    Returns:
        CompositionResult with the composed workflow.
    """
    result = CompositionResult()

    if not intent.strip():
        result.success = False
        result.error = "Empty intent"
        return result

    # Build composition plan
    plan = CompositionPlan(intent=intent)

    # Detect desired model family from intent keywords
    intent_lower = intent.lower()
    if model_family:
        plan.model_family = model_family
    elif "flux" in intent_lower:
        plan.model_family = "Flux"
    elif "sdxl" in intent_lower or "xl" in intent_lower:
        plan.model_family = "SDXL"
    elif "sd3" in intent_lower:
        plan.model_family = "SD3"
    else:
        plan.model_family = "SD1.5"  # Default fallback

    # Extract quality/style hints
    if any(kw in intent_lower for kw in ("photorealistic", "photo", "realistic")):
        plan.parameters["cfg"] = 7.5
        plan.parameters["steps"] = 30
    elif any(kw in intent_lower for kw in ("dreamy", "soft", "ethereal")):
        plan.parameters["cfg"] = 5.0
        plan.parameters["steps"] = 35
    elif any(kw in intent_lower for kw in ("sharp", "crisp", "detailed")):
        plan.parameters["cfg"] = 9.0
        plan.parameters["steps"] = 25
    else:
        plan.parameters["cfg"] = 7.0
        plan.parameters["steps"] = 20

    # Apply experience patterns if available
    if experience_patterns:
        for pattern in experience_patterns:
            if pattern.get("confidence", 0) > 0.7:
                plan.parameters.update(pattern.get("parameters", {}))
                plan.confidence = max(plan.confidence, pattern["confidence"])

    plan.reasoning = (
        f"Selected {plan.model_family} family based on intent analysis. "
        f"Applied {len(plan.parameters)} parameter defaults."
    )
    result.plan = plan

    # Template selection
    if available_templates:
        for tmpl in available_templates:
            if tmpl.get("family", "").lower() == plan.model_family.lower():
                plan.base_template = tmpl.get("name", "")
                result.workflow_data = tmpl.get("data", {})
                break

    result.mutations_applied = len(plan.parameters)
    return result
