"""Ratchet — binary keep/discard decisions on USD sublayers with multi-axis scoring.

Each iteration of a generation pipeline produces a delta sublayer. The Ratchet
evaluates that delta across configurable perceptual axes and records a binary
keep/discard decision. Kept deltas can be flattened into a recipe sublayer for
export or replay.

Axes (all optional — score what you have):
  aesthetic      Perceptual quality / artistic fidelity
  depth          Depth map accuracy / scene geometry
  normals        Surface normal alignment
  camera         Camera placement / framing
  segmentation   Mask / region accuracy
  lighting       Lighting match / illumination quality

FORESIGHT integration (optional — ratchet works standalone without it):
  If a CWM (Cognitive World Model) is provided, the ratchet will:
  - Predict outcomes before experiments (logged alongside actuals)
  - Auto-record experience after every keep/discard decision
  - Generate counterfactuals at session close
  - Consult an Arbiter for surfacing mode (silent/soft_surface/explicit)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from pxr import Sdf, Usd
    HAS_USD = True
except ImportError:
    HAS_USD = False

log = logging.getLogger(__name__)


DEFAULT_WEIGHTS: dict[str, float] = {
    "aesthetic": 1.0,
    "depth": 1.0,
    "normals": 1.0,
    "camera": 1.0,
    "segmentation": 1.0,
    "lighting": 1.0,
}

_SCORE_MIN = 0.0
_SCORE_MAX = 1.0


class RatchetError(Exception):
    """Base error for Ratchet operations."""


@dataclass
class RatchetDecision:
    """Immutable record of a single keep/discard decision."""

    delta_id: str
    kept: bool          # True = keep, False = discard
    axis_scores: dict[str, float]
    composite: float
    timestamp: float = field(default_factory=time.time)
    # FORESIGHT fields (populated when CWM is wired in)
    predicted_scores: dict[str, float] | None = None
    prediction_accuracy: float | None = None
    arbiter_mode: str | None = None  # silent | soft_surface | explicit


class Ratchet:
    """Multi-axis scorer and binary keep/discard tracker for USD delta sublayers.

    Scores are per-axis floats in [0.0, 1.0]. A weighted average produces the
    composite score. Auto-decision compares the composite against a threshold.
    Explicit keep/discard methods bypass the threshold.

    Args:
        weights: Per-axis weights. Keys from DEFAULT_WEIGHTS are recognised.
                 Pass a subset to ignore other axes (e.g. {"aesthetic": 2.0}).
                 Defaults to DEFAULT_WEIGHTS (all axes, equal weight 1.0).
        threshold: Auto-decision cutoff in [0, 1]. composite >= threshold → keep.
                   Default 0.5.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        threshold: float = 0.5,
        cws: Any | None = None,
        cwm: Any | None = None,
        arbiter: Any | None = None,
        workflow_signature: Any | None = None,
    ) -> None:
        if not (_SCORE_MIN <= threshold <= _SCORE_MAX):
            raise RatchetError(
                f"threshold must be in [0, 1], got {threshold}"
            )
        self._weights: dict[str, float] = (
            dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        )
        self._threshold = threshold
        self._history: list[RatchetDecision] = []
        # FORESIGHT integration (all optional — degradation cascade)
        self._cws = cws                          # CognitiveWorkflowStage
        self._cwm = cwm                          # CWM predict() callable
        self._arbiter = arbiter                   # Arbiter instance
        self._workflow_signature = workflow_signature  # WorkflowSignature

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> dict[str, float]:
        """Current axis weights (copy)."""
        return dict(self._weights)

    @property
    def threshold(self) -> float:
        """Auto-decision threshold."""
        return self._threshold

    @property
    def history(self) -> list[RatchetDecision]:
        """All decisions recorded so far, oldest first (copy)."""
        return list(self._history)

    @property
    def has_foresight(self) -> bool:
        """True if CWM + CWS are both wired in."""
        return self._cwm is not None and self._cws is not None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_score(self, axis_scores: dict[str, float]) -> float:
        """Compute a weighted composite score from per-axis scores.

        Only axes present in *both* axis_scores and self._weights contribute.
        Missing axes are silently skipped (score what you have).

        Args:
            axis_scores: Map of axis name to score in [0.0, 1.0].

        Returns:
            Weighted average in [0.0, 1.0]. 0.0 if no axes match.

        Raises:
            RatchetError: If any score is outside [0.0, 1.0].
        """
        for axis, score in axis_scores.items():
            if not (_SCORE_MIN <= score <= _SCORE_MAX):
                raise RatchetError(
                    f"Score for '{axis}' must be in [0, 1], got {score}"
                )

        total_weight = 0.0
        weighted_sum = 0.0
        for axis, score in axis_scores.items():
            w = self._weights.get(axis, 0.0)
            if w > 0.0:
                weighted_sum += score * w
                total_weight += w

        return 0.0 if total_weight == 0.0 else weighted_sum / total_weight

    # ------------------------------------------------------------------
    # Decision recording
    # ------------------------------------------------------------------

    def decide(
        self,
        delta_id: str,
        axis_scores: dict[str, float],
        *,
        change_context: dict[str, Any] | None = None,
    ) -> bool:
        """Score a delta and auto-decide keep/discard via threshold.

        Records the decision in history.

        Args:
            delta_id: Identifier of the USD delta layer (e.g. Sdf.Layer.identifier).
            axis_scores: Map of axis name to score in [0.0, 1.0].
            change_context: Optional dict describing what changed (for FORESIGHT).
                Passed to CWM predict() and experience recording.

        Returns:
            True if kept (composite >= threshold), False if discarded.

        Raises:
            RatchetError: If any score is out of [0, 1].
        """
        composite = self.compute_score(axis_scores)
        kept = composite >= self._threshold

        # FORESIGHT: predict before recording
        predicted, accuracy, mode = self._foresight_predict(change_context)

        decision = RatchetDecision(
            delta_id=delta_id,
            kept=kept,
            axis_scores=dict(axis_scores),
            composite=composite,
            predicted_scores=predicted,
            prediction_accuracy=accuracy,
            arbiter_mode=mode,
        )
        self._history.append(decision)
        self._foresight_record(decision, change_context)
        return kept

    def keep(
        self,
        delta_id: str,
        axis_scores: dict[str, float] | None = None,
        *,
        change_context: dict[str, Any] | None = None,
    ) -> RatchetDecision:
        """Explicitly mark a delta as kept.

        Args:
            delta_id: Identifier of the USD delta layer.
            axis_scores: Optional scores to record. Composite computed if provided.
                         If None, composite defaults to 1.0.
            change_context: Optional dict describing what changed (for FORESIGHT).

        Returns:
            The recorded RatchetDecision.

        Raises:
            RatchetError: If any score is out of [0, 1].
        """
        scores = axis_scores or {}
        composite = self.compute_score(scores) if scores else 1.0

        predicted, accuracy, mode = self._foresight_predict(change_context)

        decision = RatchetDecision(
            delta_id=delta_id,
            kept=True,
            axis_scores=dict(scores),
            composite=composite,
            predicted_scores=predicted,
            prediction_accuracy=accuracy,
            arbiter_mode=mode,
        )
        self._history.append(decision)
        self._foresight_record(decision, change_context)
        return decision

    def discard(
        self,
        delta_id: str,
        axis_scores: dict[str, float] | None = None,
        *,
        change_context: dict[str, Any] | None = None,
    ) -> RatchetDecision:
        """Explicitly mark a delta as discarded.

        Args:
            delta_id: Identifier of the USD delta layer.
            axis_scores: Optional scores to record. Composite computed if provided.
                         If None, composite defaults to 0.0.
            change_context: Optional dict describing what changed (for FORESIGHT).

        Returns:
            The recorded RatchetDecision.

        Raises:
            RatchetError: If any score is out of [0, 1].
        """
        scores = axis_scores or {}
        composite = self.compute_score(scores) if scores else 0.0

        predicted, accuracy, mode = self._foresight_predict(change_context)

        decision = RatchetDecision(
            delta_id=delta_id,
            kept=False,
            axis_scores=dict(scores),
            composite=composite,
            predicted_scores=predicted,
            prediction_accuracy=accuracy,
            arbiter_mode=mode,
        )
        self._history.append(decision)
        self._foresight_record(decision, change_context)
        return decision

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def kept_ids(self) -> list[str]:
        """Return delta_ids of all kept decisions, in order."""
        return [d.delta_id for d in self._history if d.kept]

    def discarded_ids(self) -> list[str]:
        """Return delta_ids of all discarded decisions, in order."""
        return [d.delta_id for d in self._history if not d.kept]

    def best(self) -> RatchetDecision | None:
        """Return the decision with the highest composite score, or None."""
        return max(self._history, key=lambda d: d.composite) if self._history else None

    def worst(self) -> RatchetDecision | None:
        """Return the decision with the lowest composite score, or None."""
        return min(self._history, key=lambda d: d.composite) if self._history else None

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all decisions."""
        total = len(self._history)
        kept = sum(1 for d in self._history if d.kept)
        # FORESIGHT: prediction accuracy stats
        predicted = [
            d for d in self._history if d.prediction_accuracy is not None
        ]
        avg_accuracy = (
            sum(d.prediction_accuracy for d in predicted) / len(predicted)
            if predicted else None
        )
        return {
            "total": total,
            "kept": kept,
            "discarded": total - kept,
            "threshold": self._threshold,
            "best_composite": self.best().composite if self._history else None,
            "worst_composite": self.worst().composite if self._history else None,
            "foresight_enabled": self.has_foresight,
            "predictions_made": len(predicted),
            "avg_prediction_accuracy": avg_accuracy,
        }

    # ------------------------------------------------------------------
    # FORESIGHT integration (all methods degrade gracefully)
    # ------------------------------------------------------------------

    def _foresight_predict(
        self, change_context: dict[str, Any] | None,
    ) -> tuple[dict[str, float] | None, float | None, str | None]:
        """Query CWM + Arbiter if available. Returns (predicted, accuracy, mode).

        Degrades to (None, None, None) if FORESIGHT is not wired in.
        """
        if not self.has_foresight or change_context is None:
            return None, None, None

        try:
            from .cwm import predict
            prediction = predict(
                self._cws, change_context,
                current_signature=self._workflow_signature,
            )
            predicted = prediction.axis_scores

            # Arbiter consultation
            mode: str | None = None
            if self._arbiter is not None:
                current_composite = (
                    self.best().composite if self._history else 0.5
                )
                try:
                    ad = self._arbiter.prioritize_experiment(
                        prediction, current_composite,
                    )
                    mode = ad.mode
                except Exception:
                    log.debug("Arbiter consultation failed", exc_info=True)

            return predicted, None, mode  # accuracy computed after actual
        except Exception:
            log.debug("CWM prediction failed", exc_info=True)
            return None, None, None

    def _foresight_record(
        self,
        decision: RatchetDecision,
        change_context: dict[str, Any] | None,
    ) -> None:
        """Record experience after a decision. Updates prediction_accuracy.

        Degrades silently if FORESIGHT is not wired in.
        """
        if not self.has_foresight:
            return

        # Compute prediction accuracy if we have both predicted and actual
        if decision.predicted_scores and decision.axis_scores:
            shared = set(decision.predicted_scores) & set(decision.axis_scores)
            if shared:
                mae = sum(
                    abs(decision.predicted_scores[a] - decision.axis_scores[a])
                    for a in shared
                ) / len(shared)
                decision.prediction_accuracy = max(0.0, 1.0 - mae)

        # Record experience
        try:
            from .experience import record_experience
            sig_hash = (
                self._workflow_signature.signature_hash()
                if self._workflow_signature else ""
            )
            record_experience(
                self._cws,
                initial_state=change_context or {},
                decisions=[{
                    "delta_id": decision.delta_id,
                    "kept": decision.kept,
                }],
                outcome=decision.axis_scores,
                context_signature_hash=sig_hash,
                predicted_outcome=decision.predicted_scores,
            )
        except Exception:
            log.debug("Experience recording failed", exc_info=True)

    def close_session(self) -> list[str]:
        """Close the ratchet session: generate counterfactuals for best kept.

        Called at session end. Generates a counterfactual for the highest-
        impact kept experiment: "what if we had pushed further?"

        Returns:
            List of generated counterfactual IDs (empty if FORESIGHT off).
        """
        if not self.has_foresight:
            return []

        kept = [d for d in self._history if d.kept and d.axis_scores]
        if not kept:
            return []

        # Pick the best kept decision
        best_kept = max(kept, key=lambda d: d.composite)

        try:
            from .counterfactuals import generate_counterfactual
            from .experience import ExperienceChunk

            # Create a synthetic source chunk from the best decision
            sig_hash = (
                self._workflow_signature.signature_hash()
                if self._workflow_signature else ""
            )
            source = ExperienceChunk(
                chunk_id=best_kept.delta_id,
                initial_state={"delta_id": best_kept.delta_id},
                decisions=[{"kept": True, "composite": best_kept.composite}],
                outcome=best_kept.axis_scores,
                context_signature_hash=sig_hash,
                timestamp=best_kept.timestamp,
            )

            # Hypothesize: "what if we increased the best axes further?"
            boosted = {
                axis: min(1.0, score + 0.1)
                for axis, score in best_kept.axis_scores.items()
            }
            hypothesis = {
                "type": "push_further",
                "source_delta_id": best_kept.delta_id,
                "source_composite": best_kept.composite,
            }

            cf = generate_counterfactual(
                self._cws, source, hypothesis, boosted,
            )
            return [cf.cf_id]
        except Exception:
            log.debug("Counterfactual generation failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Recipe extraction
    # ------------------------------------------------------------------

    def extract_recipe(
        self,
        cws: Any,  # CognitiveWorkflowStage — avoids circular import
        *,
        kept_ids: list[str] | None = None,
        recipe_name: str = "recipe",
    ) -> str:
        """Flatten winning delta sublayers into a single recipe sublayer.

        Merges all kept deltas (or the provided kept_ids subset) into a new
        anonymous Sdf.Layer that is inserted at the front of the stage's
        sublayer stack (strongest opinion).

        Args:
            cws: CognitiveWorkflowStage instance.
            kept_ids: Override which delta ids to flatten. Defaults to kept_ids().
            recipe_name: Label embedded in the anonymous layer identifier.

        Returns:
            Identifier of the new recipe layer.

        Raises:
            RatchetError: If USD is unavailable, no kept deltas exist, or none
                          of the kept ids match layers in the stage.
        """
        if not HAS_USD:
            raise RatchetError(
                "USD not available. Install with: pip install usd-core"
            )

        ids_to_use = kept_ids if kept_ids is not None else self.kept_ids()
        if not ids_to_use:
            raise RatchetError("No kept deltas to extract into a recipe.")

        id_set = set(ids_to_use)
        matched_layers: list[Any] = [
            layer for layer in cws._agent_deltas
            if layer.identifier in id_set
        ]

        if not matched_layers:
            raise RatchetError(
                f"None of the kept delta ids were found in the stage. "
                f"Kept ids: {ids_to_use}"
            )

        # Build an ephemeral stage with only the matched layers (oldest = weakest).
        # reversed() puts oldest at the tail of subLayerPaths (weakest position).
        ephemeral = Usd.Stage.CreateInMemory("ephemeral_recipe.usda")
        ephemeral.GetRootLayer().subLayerPaths = [
            layer.identifier for layer in reversed(matched_layers)
        ]

        # Flatten into a clean single layer, then transfer into the recipe layer.
        recipe_layer = Sdf.Layer.CreateAnonymous(f"{recipe_name}.usda")
        recipe_layer.TransferContent(ephemeral.Flatten())

        # Insert at front of the CWS sublayer stack (strongest opinion).
        stage_root = cws.stage.GetRootLayer()
        paths = list(stage_root.subLayerPaths)
        paths.insert(0, recipe_layer.identifier)
        stage_root.subLayerPaths = paths

        # Keep the layer alive via the agent_deltas list.
        cws._agent_deltas.append(recipe_layer)

        return recipe_layer.identifier
