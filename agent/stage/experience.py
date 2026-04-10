"""Experience Accumulator — records and queries workflow outcomes in USD.

Each experience is a chunk recording what was tried, what happened, and how
well we predicted the outcome. Stored as USD prims under /experience/.

ExperienceChunk fields:
  initial_state          dict snapshot of workflow state before change
  decisions              list of decisions made (param changes, node swaps)
  outcome                6-axis scores (aesthetic, depth, normals, camera, segmentation, lighting)
  predicted_outcome      predicted 6-axis scores (if prediction was made)
  prediction_accuracy    float [0,1] — how close prediction was to reality
  context_signature_hash SHA-256 from WorkflowSignature
  experience_weight      float — starts at 1.0, decays over time

USD prim layout:
  /experience/
    /exp_{id}
      initial_state = JSON string
      decisions = JSON string
      outcome:aesthetic = float
      outcome:depth = float
      ...
      predicted:aesthetic = float
      ...
      prediction_accuracy = float
      context_signature_hash = string
      experience_weight = float
      timestamp = float
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from pxr import Usd  # noqa: F401 — HAS_USD guard
    HAS_USD = True
except ImportError:
    HAS_USD = False


# 6 scoring axes (matching ratchet.py)
OUTCOME_AXES = (
    "aesthetic", "depth", "normals", "camera", "segmentation", "lighting",
)

# Temporal decay: weight = initial * decay_rate ^ (age_days)
DEFAULT_DECAY_RATE = 0.98
DEFAULT_INITIAL_WEIGHT = 1.0


class ExperienceError(Exception):
    """Base error for experience operations."""


@dataclass
class ExperienceChunk:
    """Record of a single workflow experience."""

    chunk_id: str
    initial_state: dict[str, Any]
    decisions: list[dict[str, Any]]
    outcome: dict[str, float]  # axis -> score [0,1]
    predicted_outcome: dict[str, float] = field(default_factory=dict)
    prediction_accuracy: float = 0.0
    context_signature_hash: str = ""
    experience_weight: float = DEFAULT_INITIAL_WEIGHT
    timestamp: float = field(default_factory=time.time)

    def decayed_weight(
        self,
        now: float | None = None,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> float:
        """Compute temporally decayed weight.

        Args:
            now: Current time. Defaults to time.time().
            decay_rate: Per-day decay multiplier. Default 0.98.

        Returns:
            Decayed weight >= 0.
        """
        current = now if now is not None else time.time()
        age_days = max(0.0, (current - self.timestamp) / 86400.0)
        return self.experience_weight * (decay_rate ** age_days)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "chunk_id": self.chunk_id,
            "initial_state": self.initial_state,
            "decisions": self.decisions,
            "outcome": self.outcome,
            "predicted_outcome": self.predicted_outcome,
            "prediction_accuracy": self.prediction_accuracy,
            "context_signature_hash": self.context_signature_hash,
            "experience_weight": self.experience_weight,
            "timestamp": self.timestamp,
        }


def _compute_prediction_accuracy(
    outcome: dict[str, float],
    predicted: dict[str, float],
) -> float:
    """Compute accuracy as 1 - mean_absolute_error across shared axes.

    Returns 0.0 if no axes overlap.
    """
    shared = set(outcome) & set(predicted)
    if not shared:
        return 0.0
    total_error = sum(abs(outcome[a] - predicted[a]) for a in shared)
    mae = total_error / len(shared)
    return max(0.0, 1.0 - mae)


def _generate_chunk_id(
    context_hash: str,
    timestamp: float,
) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{context_hash}:{timestamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def record_experience(
    cws: Any,  # CognitiveWorkflowStage
    initial_state: dict[str, Any],
    decisions: list[dict[str, Any]],
    outcome: dict[str, float],
    context_signature_hash: str = "",
    predicted_outcome: dict[str, float] | None = None,
    timestamp: float | None = None,
) -> ExperienceChunk:
    """Record a workflow experience as a USD prim.

    Args:
        cws: CognitiveWorkflowStage instance.
        initial_state: Snapshot of workflow state before change.
        decisions: List of decisions made.
        outcome: 6-axis outcome scores.
        context_signature_hash: Hash from WorkflowSignature.
        predicted_outcome: Predicted scores (if available).
        timestamp: Override timestamp. Defaults to time.time().

    Returns:
        The created ExperienceChunk.

    Raises:
        ExperienceError: If outcome scores are invalid.
    """
    for axis, score in outcome.items():
        if not (0.0 <= score <= 1.0):
            raise ExperienceError(
                f"Outcome score for '{axis}' must be in [0, 1], got {score}"
            )

    ts = timestamp if timestamp is not None else time.time()
    predicted = predicted_outcome or {}

    for axis, score in predicted.items():
        if not (0.0 <= score <= 1.0):
            raise ExperienceError(
                f"Predicted score for '{axis}' must be in [0, 1], got {score}"
            )

    accuracy = _compute_prediction_accuracy(outcome, predicted)
    chunk_id = _generate_chunk_id(context_signature_hash, ts)

    chunk = ExperienceChunk(
        chunk_id=chunk_id,
        initial_state=initial_state,
        decisions=decisions,
        outcome=outcome,
        predicted_outcome=predicted,
        prediction_accuracy=accuracy,
        context_signature_hash=context_signature_hash,
        timestamp=ts,
    )

    # Write to USD prims
    prim_path = f"/experience/exp_{chunk_id}"

    cws.write(prim_path, "chunk_id", chunk_id)
    cws.write(
        prim_path, "initial_state",
        json.dumps(initial_state, sort_keys=True, allow_nan=False),  # Cycle 61: NaN-safe
    )
    cws.write(
        prim_path, "decisions",
        json.dumps(decisions, sort_keys=True, allow_nan=False),  # Cycle 61: NaN-safe
    )

    # Outcome axes
    for axis in OUTCOME_AXES:
        if axis in outcome:
            cws.write(prim_path, f"outcome:{axis}", outcome[axis])

    # Predicted axes
    for axis in OUTCOME_AXES:
        if axis in predicted:
            cws.write(prim_path, f"predicted:{axis}", predicted[axis])

    cws.write(prim_path, "prediction_accuracy", accuracy)
    cws.write(prim_path, "context_signature_hash", context_signature_hash)
    cws.write(prim_path, "experience_weight", chunk.experience_weight)
    cws.write(prim_path, "timestamp", ts)

    return chunk


def query_experience(
    cws: Any,  # CognitiveWorkflowStage
    context_signature_hash: str | None = None,
    min_weight: float = 0.0,
    limit: int = 50,
) -> list[ExperienceChunk]:
    """Query stored experiences from USD prims.

    Args:
        cws: CognitiveWorkflowStage instance.
        context_signature_hash: Filter by signature hash. None = all.
        min_weight: Minimum decayed weight to include.
        limit: Maximum results to return.

    Returns:
        List of ExperienceChunk, sorted by timestamp descending.
    """
    exp_prim = cws.stage.GetPrimAtPath("/experience")
    if not exp_prim.IsValid():
        return []

    now = time.time()
    results: list[ExperienceChunk] = []

    for child in exp_prim.GetChildren():
        chunk = _prim_to_chunk(child)
        if chunk is None:
            continue

        # Filter by signature hash
        if (context_signature_hash is not None
                and chunk.context_signature_hash != context_signature_hash):
            continue

        # Filter by decayed weight
        if chunk.decayed_weight(now) < min_weight:
            continue

        results.append(chunk)

    # Sort by timestamp descending (most recent first)
    results.sort(key=lambda c: c.timestamp, reverse=True)
    return results[:limit]


def get_statistics(
    cws: Any,  # CognitiveWorkflowStage
    context_signature_hash: str | None = None,
) -> dict[str, Any]:
    """Compute statistics over stored experiences.

    Args:
        cws: CognitiveWorkflowStage instance.
        context_signature_hash: Filter by signature hash. None = all.

    Returns:
        Dict with total_count, avg_outcome per axis, avg_prediction_accuracy,
        avg_weight, and unique_signatures.
    """
    chunks = query_experience(
        cws, context_signature_hash=context_signature_hash, limit=10000,
    )

    if not chunks:
        return {
            "total_count": 0,
            "avg_outcome": {},
            "avg_prediction_accuracy": 0.0,
            "avg_weight": 0.0,
            "unique_signatures": 0,
        }

    n = len(chunks)

    # Axis averages
    axis_sums: dict[str, float] = {}
    axis_counts: dict[str, int] = {}
    for chunk in chunks:
        for axis, score in chunk.outcome.items():
            axis_sums[axis] = axis_sums.get(axis, 0.0) + score
            axis_counts[axis] = axis_counts.get(axis, 0) + 1

    avg_outcome = {
        axis: axis_sums[axis] / axis_counts[axis]
        for axis in axis_sums
    }

    now = time.time()
    avg_accuracy = sum(c.prediction_accuracy for c in chunks) / n
    avg_weight = sum(c.decayed_weight(now) for c in chunks) / n
    unique_sigs = len({c.context_signature_hash for c in chunks})

    return {
        "total_count": n,
        "avg_outcome": avg_outcome,
        "avg_prediction_accuracy": avg_accuracy,
        "avg_weight": avg_weight,
        "unique_signatures": unique_sigs,
    }


def _prim_to_chunk(prim: Any) -> ExperienceChunk | None:
    """Convert a USD prim to an ExperienceChunk, or None if invalid."""
    chunk_id_attr = prim.GetAttribute("chunk_id")
    if not chunk_id_attr.IsValid():
        return None

    chunk_id = str(chunk_id_attr.Get())

    # Read JSON fields
    initial_state_attr = prim.GetAttribute("initial_state")
    try:  # Cycle 65: USD attribute may contain corrupted JSON
        initial_state = (
            json.loads(str(initial_state_attr.Get()))
            if initial_state_attr.IsValid() else {}
        )
    except (ValueError, TypeError):
        initial_state = {}

    decisions_attr = prim.GetAttribute("decisions")
    try:  # Cycle 65: USD attribute may contain corrupted JSON
        decisions = (
            json.loads(str(decisions_attr.Get()))
            if decisions_attr.IsValid() else []
        )
    except (ValueError, TypeError):
        decisions = []

    # Read outcome axes
    outcome: dict[str, float] = {}
    for axis in OUTCOME_AXES:
        attr = prim.GetAttribute(f"outcome:{axis}")
        if attr.IsValid():
            outcome[axis] = float(attr.Get())

    # Read predicted axes
    predicted: dict[str, float] = {}
    for axis in OUTCOME_AXES:
        attr = prim.GetAttribute(f"predicted:{axis}")
        if attr.IsValid():
            predicted[axis] = float(attr.Get())

    # Scalars
    def _read_float(name: str, default: float = 0.0) -> float:
        attr = prim.GetAttribute(name)
        return float(attr.Get()) if attr.IsValid() else default

    def _read_str(name: str, default: str = "") -> str:
        attr = prim.GetAttribute(name)
        return str(attr.Get()) if attr.IsValid() else default

    return ExperienceChunk(
        chunk_id=chunk_id,
        initial_state=initial_state,
        decisions=decisions,
        outcome=outcome,
        predicted_outcome=predicted,
        prediction_accuracy=_read_float("prediction_accuracy"),
        context_signature_hash=_read_str("context_signature_hash"),
        experience_weight=_read_float("experience_weight", DEFAULT_INITIAL_WEIGHT),
        timestamp=_read_float("timestamp"),
    )
