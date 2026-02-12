"""Memory module — outcome tracking and pattern learning.

Records every execution outcome. Over time, builds a knowledge base about
what works for this artist on this machine. Recommendations are aggregation-
based (not ML) — group by model combo, sampler, steps, etc.

Storage: append-only JSONL in sessions/{name}_outcomes.jsonl.
"""

import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

from ..config import SESSIONS_DIR
from ..tools._util import to_json

log = logging.getLogger(__name__)

OUTCOME_SCHEMA_VERSION = 1
OUTCOME_MAX_BYTES = 10_000_000   # 10 MB — rotate when exceeded
OUTCOME_BACKUP_COUNT = 5

# Temporal decay: outcomes older than this half-life contribute less to aggregations.
# Default: 7 days. Recent outcomes matter more than ancient ones.
DECAY_HALF_LIFE_S = 7 * 24 * 3600  # 604800 seconds


def _temporal_weight(timestamp: float, now: float | None = None) -> float:
    """Exponential decay weight based on outcome age.

    Returns 1.0 for brand-new outcomes, 0.5 at half-life age, asymptotically 0.
    Minimum weight 0.01 to never fully discard data.
    """
    if now is None:
        now = time.time()
    age = max(0.0, now - timestamp)
    weight = math.exp(-0.693147 * age / DECAY_HALF_LIFE_S)  # ln(2) = 0.693147
    return max(weight, 0.01)

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
                "goal_id": {
                    "type": "string",
                    "description": (
                        "Goal ID from planner (links outcome to the goal that "
                        "triggered it). Optional — auto-populated when planner is active."
                    ),
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
                "scope": {
                    "type": "string",
                    "enum": ["session", "global"],
                    "description": (
                        "Scope of pattern query. 'session' (default) uses current "
                        "session only. 'global' aggregates across all sessions for "
                        "cross-session learning."
                    ),
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
    {
        "name": "detect_implicit_feedback",
        "description": (
            "Analyze behavior patterns in the outcome history to infer user "
            "satisfaction without explicit ratings. Detects: reuse patterns "
            "(iterating on a config = positive), abandonment (switching away = "
            "negative), refinement bursts (small tweaks = refining), and "
            "parameter regression (reverting = negative on recent change). "
            "Returns enriched feedback signals per outcome."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Session name. Defaults to 'default'.",
                },
                "window": {
                    "type": "integer",
                    "description": (
                        "Number of recent outcomes to analyze. Default: 20."
                    ),
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


def _migrate_outcome(outcome: dict) -> dict:
    """Upgrade an outcome record from older schema versions to current.

    v0 -> v1: adds schema_version field.
    """
    if "schema_version" not in outcome:
        outcome["schema_version"] = 1
    return outcome


def _load_outcomes(session: str) -> list[dict]:
    """Load all outcomes for a session, migrating old records."""
    path = _outcomes_path(session)
    if not path.exists():
        return []
    outcomes = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            try:
                outcomes.append(_migrate_outcome(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return outcomes


def _load_all_outcomes() -> list[dict]:
    """Load outcomes from ALL sessions for cross-session learning.

    Scans SESSIONS_DIR for all *_outcomes.jsonl files, loads and merges them,
    then sorts by timestamp for temporal coherence.
    """
    if not SESSIONS_DIR.exists():
        return []
    all_outcomes = []
    for path in sorted(SESSIONS_DIR.glob("*_outcomes.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                try:
                    all_outcomes.append(_migrate_outcome(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    all_outcomes.sort(key=lambda o: o.get("timestamp", 0))
    return all_outcomes


def _rotate_outcomes(path: Path) -> None:
    """Rotate outcome JSONL files: file.jsonl -> file.jsonl.1, etc."""
    for i in range(OUTCOME_BACKUP_COUNT, 0, -1):
        src = Path(f"{path}.{i}") if i > 0 else path
        dst = Path(f"{path}.{i + 1}")
        if i == OUTCOME_BACKUP_COUNT:
            # Drop the oldest
            if src.exists():
                src.unlink()
        elif src.exists():
            src.rename(dst)
    # Rename current -> .1
    if path.exists():
        path.rename(Path(f"{path}.1"))
    log.info("Rotated outcome files for %s", path.stem)


def _append_outcome(session: str, outcome: dict) -> None:
    """Append an outcome to the JSONL file, rotating if size exceeded."""
    path = _outcomes_path(session)
    # Check if rotation needed before writing
    if path.exists():
        try:
            if path.stat().st_size > OUTCOME_MAX_BYTES:
                _rotate_outcomes(path)
        except OSError:
            pass
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
    """Find model combos with highest time-weighted average quality.

    Uses exponential temporal decay so recent outcomes count more than old ones.
    """
    now = time.time()
    combo_data = defaultdict(list)  # combo -> [(score, weight)]
    for o in outcomes:
        combo = tuple(sorted(o.get("model_combo", [])))
        if not combo:
            continue
        score = o.get("quality_score")
        if score is not None:
            w = _temporal_weight(o.get("timestamp", 0), now)
            combo_data[combo].append((score, w))

    results = []
    for combo, entries in sorted(combo_data.items()):
        total_w = sum(w for _, w in entries)
        if total_w == 0:
            continue
        weighted_avg = sum(s * w for s, w in entries) / total_w
        results.append({
            "models": list(combo),
            "avg_quality": round(weighted_avg, 3),
            "sample_count": len(entries),
            "best_score": round(max(s for s, _ in entries), 3),
        })

    results.sort(key=lambda x: x["avg_quality"], reverse=True)
    return results[:10]


def _optimal_params(outcomes: list[dict]) -> dict:
    """Find parameter values correlated with high quality.

    Uses exponential temporal decay so recent outcomes count more than old ones.
    """
    now = time.time()
    param_data = defaultdict(lambda: defaultdict(list))  # param -> value -> [(score, weight)]

    for o in outcomes:
        score = o.get("quality_score")
        if score is None:
            continue
        w = _temporal_weight(o.get("timestamp", 0), now)
        params = o.get("key_params", {})
        for key, value in params.items():
            param_data[key][str(value)].append((score, w))

    optimal = {}
    for param, values in sorted(param_data.items()):
        best_value = None
        best_avg = -1.0
        best_count = 0
        for value, entries in values.items():
            total_w = sum(w for _, w in entries)
            if total_w == 0:
                continue
            weighted_avg = sum(s * w for s, w in entries) / total_w
            if weighted_avg > best_avg:
                best_avg = weighted_avg
                best_value = value
                best_count = len(entries)
        if best_value is not None:
            optimal[param] = {
                "best_value": best_value,
                "avg_quality": round(best_avg, 3),
                "sample_count": best_count,
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
        "schema_version": OUTCOME_SCHEMA_VERSION,
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
        "goal_id": tool_input.get("goal_id"),
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
    scope = tool_input.get("scope", "session")

    outcomes = _load_all_outcomes() if scope == "global" else _load_outcomes(session)

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


def _avoid_negative_patterns(outcomes: list[dict]) -> list[dict]:
    """Identify parameter combos that led to negative feedback or low quality."""
    warnings = []
    param_failures = defaultdict(list)

    for o in outcomes:
        score = o.get("quality_score")
        feedback = o.get("user_feedback", "neutral")
        if (score is not None and score < 0.4) or feedback == "negative":
            for key, value in o.get("key_params", {}).items():
                param_failures[f"{key}={value}"].append(score or 0)

    for combo, scores in sorted(param_failures.items()):
        if len(scores) >= 2:
            warnings.append({
                "type": "avoid",
                "pattern": combo,
                "avg_quality": round(sum(scores) / len(scores), 3),
                "occurrences": len(scores),
                "reason": f"Led to low quality in {len(scores)} runs",
            })

    return warnings


def _workflow_context_recommendations(
    outcomes: list[dict],
    current_params: dict,
) -> list[dict]:
    """Generate recommendations based on workflow context patterns."""
    recs = []

    # Detect if current workflow is under-stepped
    current_steps = current_params.get("steps")
    if current_steps and isinstance(current_steps, (int, float)):
        high_quality = [
            o for o in outcomes
            if (o.get("quality_score") or 0) > 0.7
        ]
        if high_quality:
            step_values = [
                o["key_params"]["steps"] for o in high_quality
                if "steps" in o.get("key_params", {})
                and isinstance(o["key_params"]["steps"], (int, float))
            ]
            if step_values:
                avg_good_steps = sum(step_values) / len(step_values)
                if current_steps < avg_good_steps * 0.7:
                    recs.append({
                        "type": "context",
                        "suggestion": f"Increase steps from {current_steps} to ~{int(avg_good_steps)}",
                        "reason": f"High-quality outputs averaged {avg_good_steps:.0f} steps",
                        "confidence": min(len(step_values) / 5, 1.0),
                    })

    # Detect sampler mismatch with model
    current_sampler = current_params.get("sampler")
    current_model_name = str(current_params.get("model", "")).lower()
    if current_sampler and current_model_name:
        # Find what sampler worked best with similar models
        model_outcomes = [
            o for o in outcomes
            if current_model_name in str(o.get("key_params", {}).get("model", "")).lower()
            and (o.get("quality_score") or 0) > 0.6
        ]
        if model_outcomes:
            sampler_scores = defaultdict(list)
            for o in model_outcomes:
                s = o.get("key_params", {}).get("sampler")
                if s:
                    sampler_scores[s].append(o.get("quality_score", 0))
            if sampler_scores:
                best_sampler = max(sampler_scores, key=lambda s: sum(sampler_scores[s]) / len(sampler_scores[s]))
                if best_sampler != current_sampler and len(sampler_scores[best_sampler]) >= 2:
                    avg = sum(sampler_scores[best_sampler]) / len(sampler_scores[best_sampler])
                    recs.append({
                        "type": "context",
                        "suggestion": f"Try sampler '{best_sampler}' instead of '{current_sampler}'",
                        "reason": f"Avg quality {avg:.3f} with this model over {len(sampler_scores[best_sampler])} runs",
                        "confidence": min(len(sampler_scores[best_sampler]) / 4, 1.0),
                    })

    return recs


def _handle_get_recommendations(tool_input: dict) -> str:
    session = tool_input.get("session", "default")
    current_model = tool_input.get("current_model", "")
    current_params = tool_input.get("current_params", {})
    goal = tool_input.get("goal", "")
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

    # 4. Contextual recommendations (workflow-aware)
    if current_params:
        ctx_recs = _workflow_context_recommendations(outcomes, current_params)
        recommendations.extend(ctx_recs)

    # 5. Negative pattern avoidance
    avoidance = _avoid_negative_patterns(outcomes)
    if avoidance:
        for warning in avoidance[:2]:
            recommendations.append({
                "type": "warning",
                "suggestion": f"Avoid: {warning['pattern']}",
                "reason": warning["reason"],
                "confidence": min(warning["occurrences"] / 3, 1.0),
            })

    # 6. Goal-specific recommendations
    if goal:
        goal_lower = goal.lower()
        if "fast" in goal_lower or "speed" in goal_lower:
            if speed.get("fastest_config"):
                recommendations.append({
                    "type": "goal",
                    "suggestion": "Use fastest known config for speed",
                    "config": speed["fastest_config"],
                    "fastest_time": speed.get("fastest_s", 0),
                    "confidence": 0.7,
                })
        if "quality" in goal_lower or "detail" in goal_lower:
            high_q = [o for o in outcomes if (o.get("quality_score") or 0) > 0.8]
            if high_q:
                best = max(high_q, key=lambda o: o.get("quality_score", 0))
                recommendations.append({
                    "type": "goal",
                    "suggestion": "Use highest quality config",
                    "config": best.get("key_params", {}),
                    "quality_score": best.get("quality_score"),
                    "confidence": 0.8,
                })

    # Sort by confidence
    recommendations.sort(key=lambda r: r.get("confidence", 0), reverse=True)

    return to_json({
        "recommendations": recommendations[:8],
        "based_on": len(outcomes),
        "avoidance_patterns": len(avoidance),
        "message": f"{len(recommendations)} recommendations based on {len(outcomes)} past outcomes.",
    })


# ---------------------------------------------------------------------------
# Implicit feedback detection
# ---------------------------------------------------------------------------

def _params_similarity(p1: dict, p2: dict) -> float:
    """Compare two parameter dicts. Returns 0.0 (nothing in common) to 1.0 (identical)."""
    if not p1 or not p2:
        return 0.0
    all_keys = set(p1.keys()) | set(p2.keys())
    if not all_keys:
        return 0.0
    matches = sum(1 for k in all_keys if str(p1.get(k)) == str(p2.get(k)))
    return matches / len(all_keys)


def _detect_reuse(outcomes: list[dict]) -> list[dict]:
    """Detect reuse patterns — same model combo used repeatedly = positive signal."""
    signals = []
    combo_runs = defaultdict(list)

    for i, o in enumerate(outcomes):
        combo = tuple(sorted(o.get("model_combo", [])))
        if combo:
            combo_runs[combo].append(i)

    for combo, indices in sorted(combo_runs.items()):
        if len(indices) >= 2:
            signals.append({
                "type": "reuse",
                "signal": "positive",
                "models": list(combo),
                "run_count": len(indices),
                "strength": min(len(indices) / 5, 1.0),
                "reason": f"Model combo reused {len(indices)} times — implies satisfaction",
            })

    return signals


def _detect_abandonment(outcomes: list[dict]) -> list[dict]:
    """Detect abandonment — model combo used once then never again after others."""
    signals = []
    if len(outcomes) < 3:
        return signals

    combo_last_seen = {}
    for i, o in enumerate(outcomes):
        combo = tuple(sorted(o.get("model_combo", [])))
        if combo:
            combo_last_seen[combo] = i

    total = len(outcomes)
    for combo, last_idx in sorted(combo_last_seen.items()):
        # Count how many times this combo was used
        count = sum(
            1 for o in outcomes
            if tuple(sorted(o.get("model_combo", []))) == combo
        )
        # Used only once and there are many subsequent runs with other combos
        remaining = total - last_idx - 1
        if count == 1 and remaining >= 2:
            signals.append({
                "type": "abandonment",
                "signal": "negative",
                "models": list(combo),
                "used_once_at": last_idx,
                "runs_after": remaining,
                "strength": min(remaining / 5, 1.0),
                "reason": f"Used once then abandoned ({remaining} runs with other combos followed)",
            })

    return signals


def _detect_refinement_bursts(outcomes: list[dict]) -> list[dict]:
    """Detect rapid iteration with small param tweaks = refining (positive on approach)."""
    signals = []
    if len(outcomes) < 2:
        return signals

    # Look for clusters of outcomes with high parameter similarity
    burst_start = 0
    for i in range(1, len(outcomes)):
        similarity = _params_similarity(
            outcomes[i].get("key_params", {}),
            outcomes[i - 1].get("key_params", {}),
        )
        if similarity < 0.5:
            # End of a burst
            burst_len = i - burst_start
            if burst_len >= 3:
                signals.append({
                    "type": "refinement_burst",
                    "signal": "positive",
                    "outcome_range": [burst_start, i - 1],
                    "burst_length": burst_len,
                    "strength": min(burst_len / 6, 1.0),
                    "reason": f"Cluster of {burst_len} runs with similar params — active refinement",
                })
            burst_start = i

    # Check final burst
    burst_len = len(outcomes) - burst_start
    if burst_len >= 3:
        signals.append({
            "type": "refinement_burst",
            "signal": "positive",
            "outcome_range": [burst_start, len(outcomes) - 1],
            "burst_length": burst_len,
            "strength": min(burst_len / 6, 1.0),
            "reason": f"Current cluster of {burst_len} runs with similar params — active refinement",
        })

    return signals


def _detect_parameter_regression(outcomes: list[dict]) -> list[dict]:
    """Detect when a user reverts a parameter back to an earlier value."""
    signals = []
    if len(outcomes) < 3:
        return signals

    # Track parameter value history
    for i in range(2, len(outcomes)):
        curr = outcomes[i].get("key_params", {})
        prev = outcomes[i - 1].get("key_params", {})
        prev2 = outcomes[i - 2].get("key_params", {})

        for key in curr:
            curr_val = str(curr.get(key, ""))
            prev_val = str(prev.get(key, ""))
            prev2_val = str(prev2.get(key, ""))

            # Current value matches 2-ago but differs from 1-ago = regression
            if curr_val and curr_val == prev2_val and curr_val != prev_val:
                signals.append({
                    "type": "parameter_regression",
                    "signal": "negative",
                    "parameter": key,
                    "reverted_from": prev_val,
                    "reverted_to": curr_val,
                    "at_outcome": i,
                    "strength": 0.7,
                    "reason": f"Reverted '{key}' from {prev_val} back to {curr_val} — change was negative",
                })

    return signals


def _handle_detect_implicit_feedback(tool_input: dict) -> str:
    """Analyze behavior patterns to infer implicit feedback."""
    session = tool_input.get("session", "default")
    window = tool_input.get("window", 20)

    outcomes = _load_outcomes(session)
    if not outcomes:
        return to_json({
            "signals": [],
            "summary": {},
            "message": "No outcome history yet.",
        })

    # Use the most recent N outcomes
    recent = outcomes[-window:]

    # Run all detectors
    all_signals = []
    all_signals.extend(_detect_reuse(recent))
    all_signals.extend(_detect_abandonment(recent))
    all_signals.extend(_detect_refinement_bursts(recent))
    all_signals.extend(_detect_parameter_regression(recent))

    # Summarize
    positive_count = sum(1 for s in all_signals if s["signal"] == "positive")
    negative_count = sum(1 for s in all_signals if s["signal"] == "negative")
    avg_strength = (
        sum(s.get("strength", 0) for s in all_signals) / len(all_signals)
        if all_signals else 0
    )

    # Overall satisfaction inference
    if positive_count > negative_count * 2:
        satisfaction = "likely_satisfied"
    elif negative_count > positive_count * 2:
        satisfaction = "likely_unsatisfied"
    elif positive_count > 0 or negative_count > 0:
        satisfaction = "mixed"
    else:
        satisfaction = "insufficient_data"

    return to_json({
        "signals": all_signals,
        "summary": {
            "total_signals": len(all_signals),
            "positive": positive_count,
            "negative": negative_count,
            "avg_strength": round(avg_strength, 3),
            "inferred_satisfaction": satisfaction,
        },
        "outcomes_analyzed": len(recent),
        "total_outcomes": len(outcomes),
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
    elif name == "detect_implicit_feedback":
        return _handle_detect_implicit_feedback(tool_input)
    else:
        return to_json({"error": f"Unknown memory tool: {name}"})
