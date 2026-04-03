"""Adapter: IntentCollectorAgent <-> VisionAgent/verify data translation.

Pure functions — no side effects, no imports of brain agent classes.
"""

from __future__ import annotations


def intent_to_criteria(intent: dict) -> dict:
    """Convert captured artistic intent to verification success criteria.

    Intent (from IntentCollectorAgent)::

        {
            "description": str,
            "parameters": dict,    # e.g. {"cfg": 7, "steps": 30}
            "style": str           # e.g. "photorealistic", "painterly"
        }

    Criteria (for VisionAgent / verify tools)::

        {
            "expected_attributes": [str, ...],
            "quality_threshold": float,
            "style_match": str
        }
    """
    description = intent.get("description", "")
    parameters = intent.get("parameters", {})
    style = intent.get("style", "")

    # Build expected attributes from intent description and parameters
    expected_attributes: list[str] = []

    if description:
        expected_attributes.append(f"matches_description: {description[:200]}")

    # Map key generation parameters to quality expectations
    cfg = parameters.get("cfg")
    steps = parameters.get("steps")
    denoise = parameters.get("denoise")

    if cfg is not None:
        if isinstance(cfg, (int, float)) and cfg >= 8:
            expected_attributes.append("sharp_details")
        elif isinstance(cfg, (int, float)) and cfg <= 5:
            expected_attributes.append("creative_freedom")

    if steps is not None:
        if isinstance(steps, (int, float)) and steps >= 30:
            expected_attributes.append("high_detail")
        elif isinstance(steps, (int, float)) and steps <= 15:
            expected_attributes.append("speed_optimized")

    if denoise is not None:
        if isinstance(denoise, (int, float)) and denoise >= 0.8:
            expected_attributes.append("high_variation")
        elif isinstance(denoise, (int, float)) and denoise <= 0.3:
            expected_attributes.append("structure_preserved")

    if style:
        expected_attributes.append(f"style: {style}")

    # Derive quality threshold from step count / cfg combo
    quality_threshold = 0.6  # baseline
    if isinstance(steps, (int, float)) and steps >= 30:
        quality_threshold = max(quality_threshold, 0.75)
    if isinstance(cfg, (int, float)) and 6 <= cfg <= 10:
        quality_threshold = max(quality_threshold, 0.7)

    return {
        "expected_attributes": expected_attributes,
        "quality_threshold": round(quality_threshold, 2),
        "style_match": style or "unspecified",
    }


def verify_against_intent(verify_result: dict, criteria: dict) -> dict:
    """Check if execution output preserved the original intent.

    Args:
        verify_result: Output from vision analysis or verify_execution::

            {
                "analysis": str,
                "scores": {"quality": float, "composition": float, ...},
                "suggestions": [str, ...]
            }

        criteria: Output from ``intent_to_criteria``::

            {
                "expected_attributes": [str, ...],
                "quality_threshold": float,
                "style_match": str
            }

    Returns::

        {
            "intent_preserved": bool,
            "deviations": [str, ...],
            "score": float
        }
    """
    scores = verify_result.get("scores", {})
    analysis = verify_result.get("analysis", "")
    suggestions = verify_result.get("suggestions", [])

    threshold = criteria.get("quality_threshold", 0.6)
    style_match = criteria.get("style_match", "unspecified")
    expected_attrs = criteria.get("expected_attributes", [])

    # Compute average quality score
    numeric_scores = [
        v for v in scores.values() if isinstance(v, (int, float))
    ]
    avg_score = (
        sum(numeric_scores) / len(numeric_scores)
        if numeric_scores
        else 0.0
    )

    deviations: list[str] = []

    # Check quality threshold
    if avg_score < threshold:
        deviations.append(
            f"Quality below threshold: {avg_score:.2f} < {threshold:.2f}"
        )

    # Check style match (simple substring presence in analysis)
    if style_match and style_match != "unspecified":
        analysis_lower = analysis.lower()
        if style_match.lower() not in analysis_lower:
            # Not necessarily a deviation — just note the style wasn't
            # explicitly confirmed in the analysis text.
            deviations.append(
                f"Style '{style_match}' not confirmed in analysis"
            )

    # Check expected attributes against suggestions (deviations)
    for attr in expected_attrs:
        if attr.startswith("style:"):
            continue  # already handled above
        # If the attribute explicitly expects good detail but suggestions
        # mention detail issues, flag it
        attr_lower = attr.lower()
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if (
                "detail" in attr_lower
                and "detail" in suggestion_lower
                and ("lack" in suggestion_lower or "more" in suggestion_lower)
            ):
                deviations.append(
                    f"Expected '{attr}' but got suggestion: {suggestion[:100]}"
                )
                break

    intent_preserved = avg_score >= threshold and len(deviations) == 0

    return {
        "intent_preserved": intent_preserved,
        "deviations": deviations[:10],
        "score": round(avg_score, 3),
    }
