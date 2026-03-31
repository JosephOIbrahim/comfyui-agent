"""ExperienceChunk — full (params -> outcome) tuple per generation.

Each chunk captures the complete experiment: what was the workflow
state, what was generated, and how good was the result.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityScore:
    """Multi-dimensional quality assessment."""

    overall: float = 0.0  # 0.0 - 1.0
    technical: float = 0.0  # Sharpness, noise, artifacts
    aesthetic: float = 0.0  # Composition, color, style
    prompt_adherence: float = 0.0  # How well it matches the prompt
    source: str = ""  # "vision", "rule", "human", "hash"

    @property
    def is_scored(self) -> bool:
        return self.overall > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall, "technical": self.technical,
            "aesthetic": self.aesthetic, "prompt_adherence": self.prompt_adherence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QualityScore:
        return cls(
            overall=d.get("overall", 0.0), technical=d.get("technical", 0.0),
            aesthetic=d.get("aesthetic", 0.0), prompt_adherence=d.get("prompt_adherence", 0.0),
            source=d.get("source", ""),
        )


@dataclass
class ExperienceChunk:
    """Full experiment record: params -> outcome.

    Captures the complete state of a generation experiment for
    later retrieval and learning.
    """

    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    # Input state
    model_family: str = ""
    checkpoint: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    # parameters: {node_id: {param: value}} — flat representation of the workflow state

    # Workflow identity
    workflow_hash: str = ""  # SHA-256 of the resolved workflow
    delta_count: int = 0  # Number of delta layers at time of execution

    # Outcome
    output_filenames: list[str] = field(default_factory=list)
    quality: QualityScore = field(default_factory=QualityScore)
    execution_time_ms: float = 0.0
    error: str = ""

    # Metadata
    tags: list[str] = field(default_factory=list)
    session_id: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.error and len(self.output_filenames) > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def decay_weight(self) -> float:
        """Temporal decay weight. Newer experiences weight more heavily.

        Uses exponential decay with a half-life of 7 days.
        """
        half_life_seconds = 7 * 24 * 3600  # 7 days
        return 2.0 ** (-self.age_seconds / half_life_seconds)

    def matches_context(self, other: ExperienceChunk, threshold: float = 0.5) -> bool:
        """Quick similarity check against another chunk."""
        score = 0.0
        checks = 0

        if self.model_family and other.model_family:
            checks += 1
            if self.model_family == other.model_family:
                score += 1.0

        if self.checkpoint and other.checkpoint:
            checks += 1
            if self.checkpoint == other.checkpoint:
                score += 1.0

        if self.parameters and other.parameters:
            checks += 1
            # Count matching parameter keys
            common_keys = set(self.parameters.keys()) & set(other.parameters.keys())
            if common_keys:
                matching = sum(
                    1 for k in common_keys
                    if self.parameters[k] == other.parameters[k]
                )
                score += matching / len(common_keys)

        if checks == 0:
            return False
        return (score / checks) >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for persistence."""
        return {
            "chunk_id": self.chunk_id,
            "timestamp": self.timestamp,
            "model_family": self.model_family,
            "checkpoint": self.checkpoint,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "parameters": self.parameters,
            "workflow_hash": self.workflow_hash,
            "delta_count": self.delta_count,
            "output_filenames": self.output_filenames,
            "quality": self.quality.to_dict(),
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "tags": self.tags,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperienceChunk:
        """Deserialize from a dict."""
        quality_data = d.get("quality", {})
        quality = QualityScore.from_dict(quality_data) if isinstance(quality_data, dict) else QualityScore()
        return cls(
            chunk_id=d.get("chunk_id", uuid.uuid4().hex[:12]),
            timestamp=d.get("timestamp", time.time()),
            model_family=d.get("model_family", ""),
            checkpoint=d.get("checkpoint", ""),
            prompt=d.get("prompt", ""),
            negative_prompt=d.get("negative_prompt", ""),
            parameters=d.get("parameters", {}),
            workflow_hash=d.get("workflow_hash", ""),
            delta_count=d.get("delta_count", 0),
            output_filenames=d.get("output_filenames", []),
            quality=quality,
            execution_time_ms=d.get("execution_time_ms", 0.0),
            error=d.get("error", ""),
            tags=d.get("tags", []),
            session_id=d.get("session_id", ""),
        )
