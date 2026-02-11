"""Memory module — outcome tracking and pattern learning.

Records every execution outcome. Over time, builds a knowledge base about
what works for this artist on this machine. Recommendations are aggregation-
based (not ML) — group by model combo, sampler, steps, etc.

Storage: append-only JSONL in sessions/{name}_outcomes.jsonl.
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from ..config import SESSIONS_DIR
from ..tools._util import to_json

log = logging.getLogger(__name__)

TOOLS: list[dict] = [
    {
        "name": "record_outcome",
        "description": (
            "Record the outcome of a workflow execution. Stores parameters, "
            "model combo, quality assessment, render time, and user feedback. "
            "Call after execute_workflow + analyze_image to build learning data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Session name. Defaults to 'default'.",
                },
                "workflow_summary": {
                    "type": "string",
                    "description": "Brief workflow description for quick reference.",
                },
                "key_params": {
                    "type": "object",
                    "description": (
                        "Key parameters: model, steps, cfg, sampler, scheduler, "
                        "resolution, etc."
                    ),
                },
                "model_combo": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of models used (checkpoint, lora, controlnet, etc.).",
                },
                "render_time_s": {
                    "type": "number",
                    "description": "Render time in seconds.",
                },
                "quality_score": {
                    "type": "number",
                    "description": "Quality score from vision analysis (0-1).",
                },
                "vision_notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Notes from vision analysis.",
                },
                "user_feedback": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "User's reaction to the output.",
                },
            },
            "required": ["key_params"],
        },
    },
    {
        "name": "get_learned_patterns",
        "description": (
            "Query outcome history for patterns. Returns aggregated insights: "
            "best model combos, optimal parameters, quality trends. Use to inform "
            "recommendations and avoid repeating past mistakes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Session name. Defaults to 'default'.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "What to look for: 'best_models', 'optimal_params', "
                        "'quality_trends', 'speed_analysis', or 'all'."
                    ),
                },
                "model_filter": {
                    "type": "string",
                    "description": "Filter outcomes by model name (substring match).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Get personalized recommendations based on outcome history. "
            "Cross-references past successes with current workflow context "
            "to suggest improvements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Session name. Defaults to 'default'.",
                },
                "current_model": {
                    "type": "string",
                    "description": "Current checkpoint model name.",
                },
                "current_params": {
                    "type": "object",
                    "description": "Current workflow parameters.",
                },
                "goal": {
                    "type": "string",
                    "description": "What the artist is trying to achieve.",
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _outcomes_path(session: str) -> Path:
    """Path to the outcomes JSONL file for a session."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"{session}_outcomes.jsonl"


def _load_outcomes(session: str) -> list[dict]:
    """Load all outcomes for a session."""
    path = _outcomes_path(session)
    if not path.exists():
        return []
    outcomes = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            try:
                outcomes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return outcomes


def _append_outcome(session: str, outcome: dict) -> None:
    """Append an outcome to the JSONL file."""
    path = _outcomes_path(session)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(outcome, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _workflow_hash(key_params: dict) -> str:
    """Hash workflow parameters for grouping."""
    canonical = json.dumps(key_params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _best_model_combos(outcomes: list[dict]) -> list[dict]:
    """Find model combos with highest average quality."""
    combo_scores = defaultdict(list)
    for o in outcomes:
        combo = tuple(sorted(o.get("model_combo", [])))
        if not combo:
            continue
        score = o.get("quality_score")
        if score is not None:
            combo_scores[combo].append(score)

    results = []
    for combo, scores in sorted(combo_scores.items()):
        results.append({
            "models": list(combo),
            "avg_quality": round(sum(scores) / len(scores), 3),
            "sample_count": len(scores),
            "best_score": round(max(scores), 3),
        })

    results.sort(key=lambda x: x["avg_quality"], reverse=True)
    return results[:10]


def _optimal_params(outcomes: list[dict]) -> dict:
    """Find parameter values correlated with high quality."""
    param_scores = defaultdict(lambda: defaultdict(list))

    for o in outcomes:
        score = o.get("quality_score")
        if score is None:
            continue
        params = o.get("key_params", {})
        for key, value in params.items():
            param_scores[key][str(value)].append(score)

    optimal = {}
    for param, values in sorted(param_scores.items()):
        best_value = None
        best_avg = -1
        for value, scores in values.items():
            avg = sum(scores) / len(scores)
            if avg > best_avg:
                best_avg = avg
                best_value = value
        if best_value is not None:
            optimal[param] = {
                "best_value": best_value,
                "avg_quality": round(best_avg, 3),
                "sample_count": len(values[best_value]),
            }

    return optimal


def _speed_analysis(outcomes: list[dict]) -> dict:
    """Analyze render time trends."""
    times = [o["render_time_s"] for o in outcomes if o.get("render_time_s") is not None]
    if not times:
        return {"message": "No render time data yet."}

    # Find fastest configs
    fast_outcomes = sorted(
        [o for o in outcomes if o.get("render_time_s") is not None],
        key=lambda x: x["render_time_s"],
    )

    return {
        "total_runs": len(times),
        "avg_render_s": round(sum(times) / len(times), 2),
        "fastest_s": round(min(times), 2),
        "slowest_s": round(max(times), 2),
        "fastest_config": fast_outcomes[0].get("key_params", {}) if fast_outcomes else {},
    }


def _quality_trends(outcomes: list[dict]) -> dict:
    """Track quality over time."""
    scored = [o for o in outcomes if o.get("quality_score") is not None]
    if not scored:
        return {"message": "No quality data yet."}

    # Split into halves for trend
    mid = len(scored) // 2
    if mid == 0:
        return {
            "total_scored": len(scored),
            "avg_quality": round(sum(o["quality_score"] for o in scored) / len(scored), 3),
            "trend": "insufficient_data",
        }

    first_half = scored[:mid]
    second_half = scored[mid:]
    avg_first = sum(o["quality_score"] for o in first_half) / len(first_half)
    avg_second = sum(o["quality_score"] for o in second_half) / len(second_half)

    trend = "improving" if avg_second > avg_first + 0.02 else (
        "declining" if avg_second < avg_first - 0.02 else "stable"
    )

    return {
        "total_scored": len(scored),
        "avg_quality": round(sum(o["quality_score"] for o in scored) / len(scored), 3),
        "recent_avg": round(avg_second, 3),
        "trend": trend,
        "positive_feedback": sum(1 for o in scored if o.get("user_feedback") == "positive"),
        "negative_feedback": sum(1 for o in scored if o.get("user_feedback") == "negative"),
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_record_outcome(tool_input: dict) -> str:
    session = tool_input.get("session", "default")

    outcome = {
        "timestamp": time.time(),
        "session": session,
        "workflow_summary": tool_input.get("workflow_summary", ""),
        "workflow_hash": _workflow_hash(tool_input.get("key_params", {})),
        "key_params": tool_input.get("key_params", {}),
        "model_combo": tool_input.get("model_combo", []),
        "render_time_s": tool_input.get("render_time_s"),
        "quality_score": tool_input.get("quality_score"),
        "vision_notes": tool_input.get("vision_notes", []),
        "user_feedback": tool_input.get("user_feedback", "neutral"),
    }

    _append_outcome(session, outcome)

    total = len(_load_outcomes(session))
    return to_json({
        "recorded": True,
        "total_outcomes": total,
        "workflow_hash": outcome["workflow_hash"],
        "message": f"Outcome recorded. {total} total outcomes for session '{session}'.",
    })


def _handle_get_learned_patterns(tool_input: dict) -> str:
    session = tool_input.get("session", "default")
    query = tool_input.get("query", "all")
    model_filter = tool_input.get("model_filter")

    outcomes = _load_outcomes(session)

    if model_filter:
        outcomes = [
            o for o in outcomes
            if any(model_filter.lower() in m.lower() for m in o.get("model_combo", []))
            or model_filter.lower() in str(o.get("key_params", {}).get("model", "")).lower()
        ]

    if not outcomes:
        return to_json({
            "outcomes_count": 0,
            "message": "No outcome data yet. Run some workflows first!",
        })

    result = {"outcomes_count": len(outcomes)}

    if query in ("best_models", "all"):
        result["best_model_combos"] = _best_model_combos(outcomes)
    if query in ("optimal_params", "all"):
        result["optimal_params"] = _optimal_params(outcomes)
    if query in ("quality_trends", "all"):
        result["quality_trends"] = _quality_trends(outcomes)
    if query in ("speed_analysis", "all"):
        result["speed_analysis"] = _speed_analysis(outcomes)

    return to_json(result)


def _handle_get_recommendations(tool_input: dict) -> str:
    session = tool_input.get("session", "default")
    current_model = tool_input.get("current_model", "")
    current_params = tool_input.get("current_params", {})
    outcomes = _load_outcomes(session)

    if not outcomes:
        return to_json({
            "recommendations": [],
            "message": "No outcome history yet. Run some workflows to build recommendations!",
        })

    recommendations = []

    # 1. Check if there's a better model combo
    best_combos = _best_model_combos(outcomes)
    if best_combos and current_model:
        top = best_combos[0]
        if current_model not in top["models"] and top["avg_quality"] > 0.7:
            recommendations.append({
                "type": "model_combo",
                "suggestion": f"Try model combo: {', '.join(top['models'])}",
                "reason": f"Avg quality {top['avg_quality']} over {top['sample_count']} runs",
                "confidence": min(top["sample_count"] / 5, 1.0),
            })

    # 2. Check for optimal parameter values
    optimal = _optimal_params(outcomes)
    for param, info in optimal.items():
        current_val = str(current_params.get(param, ""))
        if current_val and current_val != info["best_value"] and info["sample_count"] >= 3:
            recommendations.append({
                "type": "parameter",
                "parameter": param,
                "current": current_val,
                "suggested": info["best_value"],
                "reason": f"Avg quality {info['avg_quality']} over {info['sample_count']} runs",
                "confidence": min(info["sample_count"] / 5, 1.0),
            })

    # 3. Speed recommendations
    speed = _speed_analysis(outcomes)
    if isinstance(speed, dict) and "fastest_config" in speed:
        if speed.get("avg_render_s", 0) > 10:
            recommendations.append({
                "type": "performance",
                "suggestion": "Consider optimization — average render time is high",
                "avg_time": speed["avg_render_s"],
                "fastest_config": speed["fastest_config"],
                "confidence": 0.6,
            })

    # Sort by confidence
    recommendations.sort(key=lambda r: r.get("confidence", 0), reverse=True)

    return to_json({
        "recommendations": recommendations[:5],
        "based_on": len(outcomes),
        "message": f"{len(recommendations)} recommendations based on {len(outcomes)} past outcomes.",
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a memory tool call."""
    if name == "record_outcome":
        return _handle_record_outcome(tool_input)
    elif name == "get_learned_patterns":
        return _handle_get_learned_patterns(tool_input)
    elif name == "get_recommendations":
        return _handle_get_recommendations(tool_input)
    else:
        return to_json({"error": f"Unknown memory tool: {name}"})
