"""Hyperagent tools — MCP tool interface for the MetaAgent.

Four tools:
  propose_improvement    — propose a system improvement
  check_evolution_tier   — classify a change category into tier
  get_meta_history       — list improvement proposals and their status
  get_calibration_stats  — aggregate calibration/evolution statistics

Tool pattern: TOOLS list[dict] + handle(name, tool_input) -> str.
"""

from __future__ import annotations

import json as _json
import threading

from .hyperagent import MetaAgent


def _to_json(obj: object) -> str:
    """Deterministic JSON (mirrors agent.tools._util.to_json)."""
    return _json.dumps(obj, sort_keys=True)

# ---------------------------------------------------------------------------
# Module-level singleton (lazy, thread-safe)
# ---------------------------------------------------------------------------

_meta_agent: MetaAgent | None = None
_meta_agent_lock = threading.Lock()


def _get_meta_agent() -> MetaAgent:
    global _meta_agent
    if _meta_agent is None:
        with _meta_agent_lock:
            if _meta_agent is None:  # Double-check after acquiring lock
                _meta_agent = MetaAgent()
    return _meta_agent


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "propose_improvement",
        "description": (
            "Propose a system improvement to the meta-agent. "
            "The change is auto-classified into Tier 1 (auto-evolve), "
            "Tier 2 (ratchet-validated), or Tier 3 (human-gate) based "
            "on its category. Returns the proposal with tier and status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Category of the change. Examples: recipe, "
                        "routing_weight, agent_prompt_tuning, constitution, "
                        "scoring_function, anchor_parameter."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the change.",
                },
                "proposed_change": {
                    "type": "object",
                    "description": "The proposed change as a JSON object.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this change should be made.",
                },
            },
            "required": ["category", "description", "proposed_change"],
        },
    },
    {
        "name": "check_evolution_tier",
        "description": (
            "Classify a change category into its evolution tier. "
            "Tier 1: auto-evolve. Tier 2: ratchet-validated. "
            "Tier 3: human-gate (constitutional anchor)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The change category to classify.",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "get_meta_history",
        "description": (
            "List improvement proposals and their current status. "
            "Optionally filter by tier or status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tier": {
                    "type": "integer",
                    "description": "Filter by tier (1, 2, or 3).",
                    "enum": [1, 2, 3],
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status.",
                    "enum": [
                        "proposed", "testing", "accepted",
                        "rejected", "blocked",
                    ],
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_calibration_stats",
        "description": (
            "Get aggregate statistics over the meta-agent's calibration "
            "history. Shows total events, outcomes (improved/degraded/neutral), "
            "and average score delta."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_propose_improvement(tool_input: dict) -> str:
    ma = _get_meta_agent()
    improvement = ma.propose_improvement(
        category=tool_input["category"],
        description=tool_input["description"],
        proposed_change=tool_input["proposed_change"],
        rationale=tool_input.get("rationale", ""),
    )
    result = improvement.to_dict()
    result["can_auto_apply"] = ma.can_auto_apply(improvement)
    result["requires_human_gate"] = ma.requires_human_gate(improvement)
    result["requires_ratchet_test"] = ma.requires_ratchet_test(improvement)
    return _to_json(result)


def _handle_check_evolution_tier(tool_input: dict) -> str:
    ma = _get_meta_agent()
    category = tool_input["category"]
    tier = ma.classify_change(category)
    tier_labels = {1: "auto-evolve", 2: "ratchet-validated", 3: "human-gate"}
    return _to_json({
        "category": category,
        "tier": tier,
        "tier_label": tier_labels.get(tier, "unknown"),
    })


def _handle_get_meta_history(tool_input: dict) -> str:
    ma = _get_meta_agent()
    items = ma.history

    tier_filter = tool_input.get("tier")
    if tier_filter is not None:
        items = [i for i in items if i.tier == tier_filter]

    status_filter = tool_input.get("status")
    if status_filter is not None:
        items = [i for i in items if i.status == status_filter]

    return _to_json({
        "count": len(items),
        "items": [i.to_dict() for i in items],
    })


def _handle_get_calibration_stats(tool_input: dict) -> str:  # noqa: ARG001
    ma = _get_meta_agent()
    return _to_json(ma.get_calibration_stats())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "propose_improvement": _handle_propose_improvement,
    "check_evolution_tier": _handle_check_evolution_tier,
    "get_meta_history": _handle_get_meta_history,
    "get_calibration_stats": _handle_get_calibration_stats,
}


def handle(name: str, tool_input: dict) -> str:
    """Execute a hyperagent tool call. Returns JSON string."""
    try:
        handler = _DISPATCH.get(name)
        if handler is None:
            return _to_json({"error": f"Unknown tool: {name}"})
        return handler(tool_input)
    except Exception as exc:  # noqa: BLE001
        return _to_json({"error": str(exc)})
