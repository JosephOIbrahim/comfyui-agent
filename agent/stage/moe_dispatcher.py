"""MoE Dispatcher — task classification, agent selection, and chain orchestration.

Classifies tasks by type (recon/design/provision/build/verify/analyze),
selects agent profile(s), filters tool access per profile, manages sequential
chains, tracks typed artifacts between agents, and enforces constitutional
commandments.

Extends concepts from agent/agents/router.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .moe_profiles import (
    DEFAULT_CHAIN,
    AgentProfile,
    get_profile,
)

# ---------------------------------------------------------------------------
# Task types and classification
# ---------------------------------------------------------------------------

TaskType = Literal[
    "recon", "design", "provision", "build", "verify", "analyze",
]

# Maps task type to the agent chain that handles it
TASK_CHAINS: dict[TaskType, tuple[str, ...]] = {
    "recon": ("scout",),
    "design": ("scout", "architect"),
    "provision": ("scout", "provisioner"),
    "build": ("scout", "architect", "forge", "crucible"),
    "verify": ("crucible",),
    "analyze": ("crucible", "vision"),
}

# Keywords for task classification
_TASK_KEYWORDS: dict[TaskType, list[str]] = {
    "recon": [
        "discover", "find", "list", "what nodes", "what models",
        "search", "check", "available", "installed",
    ],
    "design": [
        "plan", "design", "architect", "translate", "intent",
        "how would", "what if", "strategy",
    ],
    "provision": [
        "download", "install", "provision", "get model",
        "fetch", "acquire",
    ],
    "build": [
        "create", "generate", "make", "build", "modify",
        "change", "add node", "wire", "connect", "patch",
        "set", "update",
    ],
    "verify": [
        "execute", "run", "validate", "test", "check output",
        "queue",
    ],
    "analyze": [
        "analyze", "compare", "judge", "rate", "score",
        "evaluate", "review", "quality", "improve",
    ],
}


# ---------------------------------------------------------------------------
# Typed artifacts
# ---------------------------------------------------------------------------

@dataclass
class HandoffArtifact:
    """Typed artifact passed between agents in a chain."""

    artifact_type: str
    source_agent: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "artifact_type": self.artifact_type,
            "data": self.data,
            "source_agent": self.source_agent,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Chain execution state
# ---------------------------------------------------------------------------

@dataclass
class ChainState:
    """Tracks state of a running agent chain."""

    task_type: TaskType
    chain: tuple[str, ...]
    current_index: int = 0
    artifacts: list[HandoffArtifact] = field(default_factory=list)
    retries: dict[str, int] = field(default_factory=dict)
    status: Literal[
        "pending", "running", "completed", "blocked", "failed",
    ] = "pending"
    blocker_reason: str = ""

    @property
    def current_agent(self) -> str | None:
        if self.current_index < len(self.chain):
            return self.chain[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.chain)

    def to_dict(self) -> dict:
        return {
            "artifacts": [a.to_dict() for a in self.artifacts],
            "blocker_reason": self.blocker_reason,
            "chain": list(self.chain),
            "current_agent": self.current_agent,
            "current_index": self.current_index,
            "retries": self.retries,
            "status": self.status,
            "task_type": self.task_type,
        }


# ---------------------------------------------------------------------------
# Constitutional constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class MoEDispatcher:
    """Classify tasks, select agents, orchestrate chains."""

    def __init__(self, max_retries: int = MAX_RETRIES):
        self.max_retries = max_retries

    def classify_task(self, description: str) -> TaskType:
        """Classify a task description into a TaskType.

        Checks keywords in priority order: provision > build > analyze >
        design > verify > recon (default).
        """
        desc_lower = description.lower().strip()

        # Priority order matters — more specific first
        priority_order: list[TaskType] = [
            "provision", "build", "analyze", "design", "verify", "recon",
        ]

        for task_type in priority_order:
            for keyword in _TASK_KEYWORDS[task_type]:
                if keyword in desc_lower:
                    return task_type

        return "recon"

    def get_chain(self, task_type: TaskType) -> tuple[str, ...]:
        """Return the agent chain for a task type."""
        return TASK_CHAINS.get(task_type, ("scout",))

    def get_full_chain(self) -> tuple[str, ...]:
        """Return the full default chain."""
        return DEFAULT_CHAIN

    def select_profiles(self, task_type: TaskType) -> list[AgentProfile]:
        """Select agent profiles for a task type."""
        chain = self.get_chain(task_type)
        profiles = []
        for name in chain:
            profile = get_profile(name)
            if profile is not None:
                profiles.append(profile)
        return profiles

    def filter_tools_for_agent(
        self, agent_name: str, available_tools: list[str],
    ) -> list[str]:
        """Filter tools to only those allowed by the agent's profile."""
        profile = get_profile(agent_name)
        if profile is None:
            return []
        allowed = set(profile.allowed_tools)
        return [t for t in available_tools if t in allowed]

    def create_chain(self, task_type: TaskType) -> ChainState:
        """Create a new chain execution state."""
        chain = self.get_chain(task_type)
        return ChainState(task_type=task_type, chain=chain, status="pending")

    def start_chain(self, state: ChainState) -> ChainState:
        """Start a chain (set status to running)."""
        state.status = "running"
        return state

    def advance_chain(
        self,
        state: ChainState,
        artifact: HandoffArtifact | None = None,
    ) -> ChainState:
        """Advance to the next agent in the chain.

        Optionally records a handoff artifact from the current agent.
        """
        if state.is_complete:
            state.status = "completed"
            return state

        if artifact is not None:
            state.artifacts.append(artifact)

        state.current_index += 1

        if state.is_complete:
            state.status = "completed"

        return state

    def record_retry(self, state: ChainState, agent_name: str) -> bool:
        """Record a retry for an agent. Returns False if max retries exceeded.

        Constitutional commandment: bounded_failure — 3 retries then BLOCKER.
        """
        current = state.retries.get(agent_name, 0) + 1
        state.retries[agent_name] = current

        if current >= self.max_retries:
            state.status = "blocked"
            state.blocker_reason = (
                f"Agent '{agent_name}' exceeded {self.max_retries} retries"
            )
            return False

        return True

    def validate_handoff(
        self,
        artifact: HandoffArtifact,
        expected_type: str,
    ) -> bool:
        """Validate that a handoff artifact matches the expected type.

        Constitutional commandment: explicit_handoffs.
        """
        return artifact.artifact_type == expected_type

    def check_role_isolation(
        self, agent_name: str, tool_name: str,
    ) -> bool:
        """Check if an agent is allowed to use a specific tool.

        Constitutional commandment: role_isolation.
        """
        profile = get_profile(agent_name)
        if profile is None:
            return False
        return profile.can_use_tool(tool_name)

    def check_adversarial_verification(
        self, builder: str, verifier: str,
    ) -> bool:
        """Ensure builder and verifier are different agents.

        Constitutional commandment: adversarial_verification.
        """
        return builder != verifier

    def dispatch(self, description: str) -> ChainState:
        """Full dispatch: classify → create chain → start.

        Returns the initialized ChainState ready for execution.
        """
        task_type = self.classify_task(description)
        state = self.create_chain(task_type)
        return self.start_chain(state)
