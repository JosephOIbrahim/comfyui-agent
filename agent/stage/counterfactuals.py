"""Counterfactual Simulation — "what if we had done X instead?"

At session close, generates counterfactual experiences for paths not taken.
Stored under /counterfactuals/pending/ with low initial confidence (0.3).
Later validation adjusts confidence; promoted counterfactuals move to
/counterfactuals/validated/.

USD prim layout:
  /counterfactuals/
    /pending/
      /cf_{id}     (hypothesis, predicted_outcome, confidence, source_chunk_id)
    /validated/
      /cf_{id}     (same attrs + validation_outcome, validation_timestamp)
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

from .experience import OUTCOME_AXES, ExperienceChunk

# Default confidence for new counterfactuals
DEFAULT_CONFIDENCE = 0.3
# Minimum confidence to promote
PROMOTION_THRESHOLD = 0.7


class CounterfactualError(Exception):
    """Base error for counterfactual operations."""


@dataclass
class Counterfactual:
    """A hypothetical alternative outcome."""

    cf_id: str
    source_chunk_id: str               # experience that spawned this
    hypothesis: dict[str, Any]         # what-if change description
    predicted_outcome: dict[str, float]  # predicted axis scores
    confidence: float = DEFAULT_CONFIDENCE
    status: str = "pending"             # pending | validated | rejected
    validation_outcome: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    validation_timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "cf_id": self.cf_id,
            "source_chunk_id": self.source_chunk_id,
            "hypothesis": self.hypothesis,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "status": self.status,
            "validation_outcome": self.validation_outcome,
            "timestamp": self.timestamp,
            "validation_timestamp": self.validation_timestamp,
        }


def _generate_cf_id(source_chunk_id: str, hypothesis: dict, timestamp: float) -> str:
    """Generate a deterministic counterfactual ID."""
    raw = f"{source_chunk_id}:{json.dumps(hypothesis, sort_keys=True)}:{timestamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_counterfactual(
    cws: Any,  # CognitiveWorkflowStage
    source_chunk: ExperienceChunk,
    hypothesis: dict[str, Any],
    predicted_outcome: dict[str, float],
    confidence: float = DEFAULT_CONFIDENCE,
    timestamp: float | None = None,
) -> Counterfactual:
    """Generate a counterfactual from a real experience.

    Stores under /counterfactuals/pending/ with low initial confidence.

    Args:
        cws: CognitiveWorkflowStage instance.
        source_chunk: The real experience this is an alternative to.
        hypothesis: Description of the alternative path.
        predicted_outcome: Predicted axis scores for the alternative.
        confidence: Initial confidence [0,1]. Default 0.3.
        timestamp: Override timestamp.

    Returns:
        The created Counterfactual.

    Raises:
        CounterfactualError: If scores are invalid.
    """
    for axis, score in predicted_outcome.items():
        if not (0.0 <= score <= 1.0):
            raise CounterfactualError(
                f"Predicted score for '{axis}' must be in [0, 1], got {score}"
            )
    if not (0.0 <= confidence <= 1.0):
        raise CounterfactualError(
            f"Confidence must be in [0, 1], got {confidence}"
        )

    ts = timestamp if timestamp is not None else time.time()
    cf_id = _generate_cf_id(source_chunk.chunk_id, hypothesis, ts)

    cf = Counterfactual(
        cf_id=cf_id,
        source_chunk_id=source_chunk.chunk_id,
        hypothesis=hypothesis,
        predicted_outcome=predicted_outcome,
        confidence=confidence,
        timestamp=ts,
    )

    # Write to USD
    prim_path = f"/counterfactuals/pending/cf_{cf_id}"
    cws.write(prim_path, "cf_id", cf_id)
    cws.write(prim_path, "source_chunk_id", source_chunk.chunk_id)
    cws.write(prim_path, "hypothesis", json.dumps(hypothesis, sort_keys=True))
    cws.write(prim_path, "confidence", confidence)
    cws.write(prim_path, "status", "pending")
    cws.write(prim_path, "timestamp", ts)

    for axis in OUTCOME_AXES:
        if axis in predicted_outcome:
            cws.write(prim_path, f"predicted:{axis}", predicted_outcome[axis])

    return cf


def validate_counterfactual(
    cws: Any,  # CognitiveWorkflowStage
    cf_id: str,
    actual_outcome: dict[str, float],
    timestamp: float | None = None,
) -> Counterfactual:
    """Validate a pending counterfactual against actual results.

    Adjusts confidence based on prediction accuracy. If the predicted
    outcome was close to the actual outcome, confidence increases.

    Args:
        cws: CognitiveWorkflowStage instance.
        cf_id: ID of the counterfactual to validate.
        actual_outcome: Actual axis scores observed.
        timestamp: Override validation timestamp.

    Returns:
        Updated Counterfactual.

    Raises:
        CounterfactualError: If cf_id not found or scores invalid.
    """
    for axis, score in actual_outcome.items():
        if not (0.0 <= score <= 1.0):
            raise CounterfactualError(
                f"Actual score for '{axis}' must be in [0, 1], got {score}"
            )

    ts = timestamp if timestamp is not None else time.time()

    # Read from pending
    prim_path = f"/counterfactuals/pending/cf_{cf_id}"
    if not cws.prim_exists(prim_path):
        raise CounterfactualError(f"Counterfactual not found: {cf_id}")

    # Read predicted outcome
    predicted: dict[str, float] = {}
    for axis in OUTCOME_AXES:
        val = cws.read(prim_path, f"predicted:{axis}")
        if val is not None:
            predicted[axis] = float(val)

    # Compute accuracy
    shared = set(predicted) & set(actual_outcome)
    if shared:
        mae = sum(abs(predicted[a] - actual_outcome[a]) for a in shared) / len(shared)
        accuracy = max(0.0, 1.0 - mae)
    else:
        accuracy = 0.0

    # Read current confidence and adjust
    old_confidence = float(cws.read(prim_path, "confidence") or DEFAULT_CONFIDENCE)
    # Move confidence toward accuracy
    new_confidence = 0.5 * old_confidence + 0.5 * accuracy

    # Determine status
    status = "validated" if new_confidence >= PROMOTION_THRESHOLD else "pending"

    # Update USD prim
    cws.write(prim_path, "confidence", new_confidence)
    cws.write(prim_path, "status", status)
    cws.write(prim_path, "validation_timestamp", ts)
    for axis, score in actual_outcome.items():
        cws.write(prim_path, f"validation:{axis}", score)

    # Read back other fields
    source_chunk_id = str(cws.read(prim_path, "source_chunk_id") or "")
    hypothesis_raw = str(cws.read(prim_path, "hypothesis") or "{}")

    return Counterfactual(
        cf_id=cf_id,
        source_chunk_id=source_chunk_id,
        hypothesis=json.loads(hypothesis_raw),
        predicted_outcome=predicted,
        confidence=new_confidence,
        status=status,
        validation_outcome=actual_outcome,
        timestamp=float(cws.read(prim_path, "timestamp") or 0.0),
        validation_timestamp=ts,
    )


def promote_validated(
    cws: Any,  # CognitiveWorkflowStage
) -> list[str]:
    """Move validated counterfactuals from /pending/ to /validated/.

    Reads all prims under /counterfactuals/pending/, and for those with
    status=="validated", copies attributes to /counterfactuals/validated/
    and removes from pending (by writing status="promoted" on pending).

    Returns:
        List of promoted cf_ids.
    """
    pending_prim = cws.stage.GetPrimAtPath("/counterfactuals/pending")
    if not pending_prim.IsValid():
        return []

    promoted: list[str] = []

    for child in pending_prim.GetChildren():
        status_attr = child.GetAttribute("status")
        if not status_attr.IsValid():
            continue
        if str(status_attr.Get()) != "validated":
            continue

        cf_id_attr = child.GetAttribute("cf_id")
        if not cf_id_attr.IsValid():
            continue
        cf_id = str(cf_id_attr.Get())

        # Copy all attributes to /validated/
        src_path = f"/counterfactuals/pending/cf_{cf_id}"
        dst_path = f"/counterfactuals/validated/cf_{cf_id}"

        for attr in child.GetAttributes():
            val = attr.Get()
            if val is not None:
                cws.write(dst_path, attr.GetName(), val)

        # Mark as promoted in pending
        cws.write(src_path, "status", "promoted")
        promoted.append(cf_id)

    return promoted


def list_pending(
    cws: Any,  # CognitiveWorkflowStage
) -> list[Counterfactual]:
    """List all pending counterfactuals.

    Returns:
        List of Counterfactual objects with status=="pending".
    """
    pending_prim = cws.stage.GetPrimAtPath("/counterfactuals/pending")
    if not pending_prim.IsValid():
        return []

    results: list[Counterfactual] = []
    for child in pending_prim.GetChildren():
        cf = _prim_to_cf(child)
        if cf and cf.status == "pending":
            results.append(cf)
    return results


def list_validated(
    cws: Any,  # CognitiveWorkflowStage
) -> list[Counterfactual]:
    """List all validated counterfactuals.

    Returns:
        List of Counterfactual objects from /counterfactuals/validated/.
    """
    val_prim = cws.stage.GetPrimAtPath("/counterfactuals/validated")
    if not val_prim.IsValid():
        return []

    results: list[Counterfactual] = []
    for child in val_prim.GetChildren():
        cf = _prim_to_cf(child)
        if cf:
            results.append(cf)
    return results


def _prim_to_cf(prim: Any) -> Counterfactual | None:
    """Convert a USD prim to a Counterfactual, or None if invalid."""
    cf_id_attr = prim.GetAttribute("cf_id")
    if not cf_id_attr.IsValid():
        return None

    cf_id = str(cf_id_attr.Get())

    def _read_str(name: str, default: str = "") -> str:
        attr = prim.GetAttribute(name)
        return str(attr.Get()) if attr.IsValid() else default

    def _read_float(name: str, default: float = 0.0) -> float:
        attr = prim.GetAttribute(name)
        return float(attr.Get()) if attr.IsValid() else default

    hypothesis_raw = _read_str("hypothesis", "{}")
    predicted: dict[str, float] = {}
    validation: dict[str, float] = {}

    for axis in OUTCOME_AXES:
        attr = prim.GetAttribute(f"predicted:{axis}")
        if attr.IsValid():
            predicted[axis] = float(attr.Get())
        v_attr = prim.GetAttribute(f"validation:{axis}")
        if v_attr.IsValid():
            validation[axis] = float(v_attr.Get())

    return Counterfactual(
        cf_id=cf_id,
        source_chunk_id=_read_str("source_chunk_id"),
        hypothesis=json.loads(hypothesis_raw),
        predicted_outcome=predicted,
        confidence=_read_float("confidence", DEFAULT_CONFIDENCE),
        status=_read_str("status", "pending"),
        validation_outcome=validation,
        timestamp=_read_float("timestamp"),
        validation_timestamp=_read_float("validation_timestamp"),
    )
