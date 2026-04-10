"""LIVRPS Mutation Bridge — Subsystem 5.

Routes all workflow mutations through LIVRPS composition in the
CognitiveWorkflowStage.  Each mutation creates an agent delta sublayer;
multiple agents' opinions compose deterministically (newest/strongest
wins per-attribute via USD sublayer ordering).

Thread-safe: delegates locking to the stage's internal mechanisms.

Graceful degradation: when USD is not available (HAS_USD = False),
the bridge stores deltas in a plain dict and returns them directly
in MutationResolution.resolved_values.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

try:
    from pxr import Usd  # noqa: F401

    HAS_USD = True
except ImportError:
    HAS_USD = False


@dataclass(frozen=True)
class MutationResolution:
    """Result of a mutation routed through LIVRPS composition."""

    operation: str
    agent_name: str
    delta_layer_id: str
    resolved_values: dict
    overridden_by: list[str]
    timestamp: float


class MutationBridge:
    """Routes workflow mutations through LIVRPS composition.

    All mutations create agent delta sublayers in the CognitiveWorkflowStage.
    Multiple agents' opinions compose deterministically: newest/strongest wins
    per-attribute (LIVRPS ordering).

    Thread-safe: uses an internal RLock for history and degraded-mode state.
    Stage operations rely on the stage's own thread safety.
    """

    def __init__(self, stage: object | None = None):
        """Initialize with a CognitiveWorkflowStage instance.

        Args:
            stage: A CognitiveWorkflowStage. Pass None for degraded mode
                   (no USD available).
        """
        self._stage = stage
        self._lock = threading.RLock()
        self._history: list[MutationResolution] = []
        # Degraded mode: track agent deltas as plain dicts
        self._degraded_deltas: dict[str, list[dict]] = {}
        # Map agent_name -> list of layer IDs for rollback
        self._agent_layers: dict[str, list[str]] = {}

    @property
    def has_stage(self) -> bool:
        """Whether a real USD stage is available."""
        return HAS_USD and self._stage is not None

    def mutate(
        self,
        operation: str,
        agent_name: str,
        delta: dict,
        *,
        workflow_path: str = "/workflows/current",
    ) -> MutationResolution:
        """Apply a mutation through LIVRPS composition.

        Args:
            operation: Type of mutation (set_input, apply_patch, etc.)
            agent_name: Agent requesting the mutation.
            delta: Dict of {prim_path:attr_name: value} to change.
            workflow_path: Base prim path for the workflow.

        Returns:
            MutationResolution with composed result and audit trail.
        """
        ts = time.time()

        if self.has_stage:
            return self._mutate_usd(operation, agent_name, delta, workflow_path, ts)
        return self._mutate_degraded(operation, agent_name, delta, workflow_path, ts)

    def _mutate_usd(
        self,
        operation: str,
        agent_name: str,
        delta: dict,
        workflow_path: str,
        ts: float,
    ) -> MutationResolution:
        """Apply mutation via real USD stage."""
        stage = self._stage  # type: ignore[union-attr]

        # Prefix delta keys with workflow_path if they aren't already absolute
        prefixed: dict[str, object] = {}
        for key, value in delta.items():
            if key.startswith("/"):
                prefixed[key] = value
            else:
                prefixed[f"{workflow_path}/{key}"] = value

        # Snapshot pre-mutation values for override detection
        pre_values: dict[str, object] = {}
        for key in prefixed:
            colon = key.find(":", key.rfind("/") + 1)
            if colon == -1:
                continue
            prim_path = key[:colon]
            attr_name = key[colon + 1 :]
            try:
                pre_values[key] = stage.read(prim_path, attr_name)
            except Exception as _e:  # Cycle 62: log instead of silently swallow
                log.debug("Could not read pre-mutation value for %r: %s", key, _e)
                pre_values[key] = None

        # Apply as agent delta sublayer
        layer_id = stage.add_agent_delta(agent_name, prefixed)

        # Track layer ownership
        with self._lock:
            self._agent_layers.setdefault(agent_name, []).append(layer_id)

        # Read back composed values to see what LIVRPS resolved
        resolved: dict[str, object] = {}
        overridden_by: list[str] = []

        for key in prefixed:
            colon = key.find(":", key.rfind("/") + 1)
            if colon == -1:
                continue
            prim_path = key[:colon]
            attr_name = key[colon + 1 :]
            try:
                composed_value = stage.read(prim_path, attr_name)
                resolved[key] = composed_value
                # If composed value differs from what we wrote, a stronger
                # layer overrode us.
                if composed_value != prefixed[key]:
                    overridden_by.append(key)
            except Exception:
                resolved[key] = prefixed[key]

        resolution = MutationResolution(
            operation=operation,
            agent_name=agent_name,
            delta_layer_id=layer_id,
            resolved_values=dict(resolved),
            overridden_by=overridden_by,
            timestamp=ts,
        )

        with self._lock:
            self._history.append(resolution)

        return resolution

    def _mutate_degraded(
        self,
        operation: str,
        agent_name: str,
        delta: dict,
        workflow_path: str,
        ts: float,
    ) -> MutationResolution:
        """Apply mutation in degraded mode (no USD)."""
        layer_id = f"degraded_{agent_name}_{int(ts * 1000)}"

        # Prefix delta keys
        prefixed: dict[str, object] = {}
        for key, value in delta.items():
            if key.startswith("/"):
                prefixed[key] = value
            else:
                prefixed[f"{workflow_path}/{key}"] = value

        with self._lock:
            self._degraded_deltas.setdefault(agent_name, []).append(prefixed)
            self._agent_layers.setdefault(agent_name, []).append(layer_id)

        resolution = MutationResolution(
            operation=operation,
            agent_name=agent_name,
            delta_layer_id=layer_id,
            resolved_values=dict(prefixed),
            overridden_by=[],
            timestamp=ts,
        )

        with self._lock:
            self._history.append(resolution)

        log.info(
            "MutationBridge degraded: %s by %s applied without USD composition.",
            operation,
            agent_name,
        )
        return resolution

    def list_resolutions(self) -> list[MutationResolution]:
        """Get history of all mutation resolutions in this session."""
        with self._lock:
            return list(self._history)

    def get_composed_state(
        self,
        workflow_path: str = "/workflows/current",
    ) -> dict:
        """Get the fully composed workflow state after all LIVRPS resolution.

        In degraded mode, returns the last-write-wins merge of all deltas.
        """
        if self.has_stage:
            stage = self._stage  # type: ignore[union-attr]
            try:
                prim_attrs = stage.get_prim_attrs(workflow_path)
                if prim_attrs:
                    return prim_attrs
                # Try flattening children
                result: dict[str, object] = {}
                for child_name in stage.list_children(workflow_path):
                    child_path = f"{workflow_path}/{child_name}"
                    attrs = stage.get_prim_attrs(child_path)
                    for attr_name, val in attrs.items():
                        result[f"{child_path}:{attr_name}"] = val
                return result
            except Exception as exc:
                log.warning("get_composed_state failed on stage: %s", exc)
                return {}

        # Degraded: merge all deltas, last write wins
        with self._lock:
            merged: dict[str, object] = {}
            for agent_deltas in self._degraded_deltas.values():
                for delta in agent_deltas:
                    merged.update(delta)
            return dict(merged)

    def rollback_agent(self, agent_name: str) -> int:
        """Remove all delta layers from a specific agent.

        Returns:
            Number of deltas removed.
        """
        with self._lock:
            layer_ids = self._agent_layers.pop(agent_name, [])

        if not layer_ids:
            return 0

        if self.has_stage:
            stage = self._stage  # type: ignore[union-attr]
            removed = 0
            # The stage tracks deltas by position. We need to use
            # the layer identifiers to find and remove them.
            # CognitiveWorkflowStage stores deltas in _agent_deltas list.
            # We remove matching layers from the sublayer stack.
            try:
                layer_id_set = set(layer_ids)
                # Access internal delta list to find matching layers
                remaining_deltas = []
                removed_ids = set()
                for layer in stage._agent_deltas:
                    if layer.identifier in layer_id_set:
                        removed_ids.add(layer.identifier)
                        removed += 1
                    else:
                        remaining_deltas.append(layer)
                stage._agent_deltas = remaining_deltas

                # Update sublayer paths on the root layer
                root = stage.root_layer
                root.subLayerPaths = [p for p in root.subLayerPaths if p not in removed_ids]
            except Exception as exc:
                log.warning("rollback_agent stage cleanup failed: %s", exc)
                removed = len(layer_ids)
        else:
            # Degraded mode
            with self._lock:
                self._degraded_deltas.pop(agent_name, None)
            removed = len(layer_ids)

        # Remove from history
        with self._lock:
            self._history = [r for r in self._history if r.agent_name != agent_name]

        log.info("Rolled back %d delta(s) for agent '%s'.", removed, agent_name)
        return removed
