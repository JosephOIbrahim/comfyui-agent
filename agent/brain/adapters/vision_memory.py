"""Adapter: VisionAgent <-> MemoryAgent data translation.

Pure functions — no side effects, no imports of brain agent classes.
"""

from __future__ import annotations


def vision_to_outcome(vision_analysis: dict) -> dict:
    """Convert VisionAgent analyze_image output to MemoryAgent record_outcome format.

    VisionAgent returns::

        {
            "analysis": str,
            "scores": {"quality": float, "composition": float, ...},
            "suggestions": [str, ...]
        }

    MemoryAgent expects::

        {
            "session": str,
            "action": str,
            "result": str,
            "details": dict
        }
    """
    analysis_text = vision_analysis.get("analysis", "")
    scores = vision_analysis.get("scores", {})
    suggestions = vision_analysis.get("suggestions", [])

    # Derive a result category from scores (if present)
    avg_score = 0.0
    if scores:
        numeric = [v for v in scores.values() if isinstance(v, (int, float))]
        avg_score = sum(numeric) / len(numeric) if numeric else 0.0

    if avg_score >= 0.8:
        result = "success"
    elif avg_score >= 0.5:
        result = "partial"
    else:
        result = "needs_improvement"

    return {
        "session": vision_analysis.get("session", "default"),
        "action": "vision_analysis",
        "result": result,
        "details": {
            "analysis_summary": analysis_text[:500] if analysis_text else "",
            "scores": dict(sorted(scores.items())) if scores else {},
            "suggestion_count": len(suggestions),
            "suggestions": suggestions[:10],
            "average_score": round(avg_score, 3),
        },
    }


def patterns_to_vision_context(learned_patterns: dict) -> dict:
    """Convert MemoryAgent patterns to VisionAgent evaluation criteria.

    MemoryAgent returns::

        {
            "patterns": [
                {"pattern": str, "frequency": int, "outcome": str},
                ...
            ],
            "model_combos": {
                "combo_key": {"success_rate": float, ...},
                ...
            }
        }

    VisionAgent wants::

        {
            "known_issues": [str, ...],
            "expected_quality": float
        }
    """
    patterns = learned_patterns.get("patterns", [])
    model_combos = learned_patterns.get("model_combos", {})

    # Extract known issues from patterns that had negative outcomes
    known_issues: list[str] = []
    for pat in patterns:
        outcome = pat.get("outcome", "")
        if outcome in ("needs_improvement", "failure", "partial"):
            desc = pat.get("pattern", "")
            if desc:
                known_issues.append(desc)

    # Derive expected quality from model combo success rates
    success_rates: list[float] = []
    for combo_info in model_combos.values():
        rate = combo_info.get("success_rate")
        if isinstance(rate, (int, float)):
            success_rates.append(float(rate))

    expected_quality = (
        round(sum(success_rates) / len(success_rates), 3)
        if success_rates
        else 0.5
    )

    return {
        "known_issues": known_issues[:20],
        "expected_quality": expected_quality,
    }
