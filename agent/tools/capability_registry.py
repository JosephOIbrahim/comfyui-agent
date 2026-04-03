"""Capability-matching dispatch registry for tool selection.

Indexes tool capabilities so the agent can query for tools by requirement
rather than by name. Parallel to _HANDLERS (which remains the fast path
for direct name-based lookup).

Example:
    registry.select({"requires_comfyui": True, "phase": "pilot", "max_risk": 1})
    → all tools that need ComfyUI, belong to the pilot phase, and have risk ≤ 1
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

_LATENCY_ORDER = {"realtime": 0, "interactive": 1, "batch": 2}


@dataclass(frozen=True)
class ToolCapability:
    """Immutable capability descriptor for a single tool."""

    tool_name: str
    requires_comfyui: bool = False
    latency_class: str = "interactive"  # "realtime" | "interactive" | "batch"
    input_requirements: frozenset[str] = frozenset()
    output_type: str = "json"  # "json" | "image_path" | "status" | "structured"
    phase: str = "any"  # "understand" | "discover" | "pilot" | "verify" | "any"
    mutates_workflow: bool = False
    risk_level: int = 0  # 0=read-only, 1=mutation, 2=execute, 3=provision, 4=destructive
    requires_brain: bool = False
    requires_stage: bool = False


class ToolCapabilityRegistry:
    """Indexes tool capabilities for query-based dispatch.

    Thread-safe. All mutation (register/register_batch) and reads (select/get)
    are serialized via an RLock so the registry can be populated at startup
    and queried from any thread.
    """

    def __init__(self) -> None:
        self._caps: dict[str, ToolCapability] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, cap: ToolCapability) -> None:
        """Register a single tool capability."""
        with self._lock:
            self._caps[cap.tool_name] = cap

    def register_batch(self, caps: list[ToolCapability]) -> None:
        """Register multiple capabilities at once."""
        with self._lock:
            for cap in caps:
                self._caps[cap.tool_name] = cap

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> ToolCapability | None:
        """Get capability by exact tool name."""
        with self._lock:
            return self._caps.get(tool_name)

    def all_tools(self) -> list[ToolCapability]:
        """Return all registered capabilities (sorted by name for determinism)."""
        with self._lock:
            return sorted(self._caps.values(), key=lambda c: c.tool_name)

    # ------------------------------------------------------------------
    # Query / selection
    # ------------------------------------------------------------------

    def select(self, requirements: dict) -> list[ToolCapability]:
        """Filter capabilities matching all specified requirements.

        Supported keys:
            requires_comfyui (bool), requires_brain (bool),
            requires_stage (bool), mutates_workflow (bool),
            phase (str), max_risk (int), latency_class (str),
            output_type (str)

        Results sorted by: risk_level ascending, then latency_class
        (realtime < interactive < batch), then tool_name.
        """
        with self._lock:
            candidates = list(self._caps.values())

        max_risk = requirements.get("max_risk")
        phase = requirements.get("phase")
        latency = requirements.get("latency_class")
        output = requirements.get("output_type")

        result: list[ToolCapability] = []
        for cap in candidates:
            if "requires_comfyui" in requirements:
                if cap.requires_comfyui != requirements["requires_comfyui"]:
                    continue
            if "requires_brain" in requirements:
                if cap.requires_brain != requirements["requires_brain"]:
                    continue
            if "requires_stage" in requirements:
                if cap.requires_stage != requirements["requires_stage"]:
                    continue
            if "mutates_workflow" in requirements:
                if cap.mutates_workflow != requirements["mutates_workflow"]:
                    continue
            if max_risk is not None and cap.risk_level > max_risk:
                continue
            if phase is not None and cap.phase != "any" and cap.phase != phase:
                continue
            if latency is not None and cap.latency_class != latency:
                continue
            if output is not None and cap.output_type != output:
                continue
            result.append(cap)

        result.sort(
            key=lambda c: (
                c.risk_level,
                _LATENCY_ORDER.get(c.latency_class, 99),
                c.tool_name,
            )
        )
        return result

    def select_best(self, requirements: dict) -> ToolCapability | None:
        """Return the single best match (lowest risk, fastest latency)."""
        matches = self.select(requirements)
        return matches[0] if matches else None

    def by_phase(self, phase: str) -> list[ToolCapability]:
        """All tools belonging to a specific phase (plus 'any' phase tools)."""
        with self._lock:
            candidates = list(self._caps.values())
        return sorted(
            [c for c in candidates if c.phase == phase or c.phase == "any"],
            key=lambda c: (c.risk_level, c.tool_name),
        )

    def by_risk(self, max_risk: int) -> list[ToolCapability]:
        """All tools at or below the given risk level."""
        with self._lock:
            candidates = list(self._caps.values())
        return sorted(
            [c for c in candidates if c.risk_level <= max_risk],
            key=lambda c: (c.risk_level, c.tool_name),
        )
