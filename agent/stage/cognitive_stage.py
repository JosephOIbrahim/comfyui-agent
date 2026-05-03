"""CognitiveWorkflowStage — USD-native composed stage for the ecosystem.

All agent state, workflow data, recipes, and execution history live in a
single pxr.Usd.Stage with real LIVRPS composition.

Architecture:
  Root layer (empty — manages sublayer composition only)
    subLayerPaths:
      [0] newest_agent_delta    (strongest opinion)
      [1] older_agent_delta
      ...
      [N] base_layer            (weakest — workflow data, bootstrap prims)

  write() targets the base layer. Agent deltas insert before it.
  Root layer never has authored attributes, so sublayers compose correctly.

Requires usd-core: pip install usd-core
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from pxr import Sdf, Usd
    HAS_USD = True
except ImportError:
    HAS_USD = False

from .anchors import check_anchor

_log = logging.getLogger(__name__)


class StageError(Exception):
    """Base error for stage operations."""


@dataclass(frozen=True)
class StageEvent:
    """Typed event emitted by CognitiveWorkflowStage on every observable op.

    External consumers (MCP resources, Moneta adapter, telemetry) subscribe
    via CognitiveWorkflowStage.subscribe() to receive these.
    """

    op: str  # one of: "write", "add_delta", "rollback", "flush"
    prim_path: str | None = None
    attr_name: str | None = None
    layer_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    payload: dict | None = None  # op-specific extras

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "prim_path": self.prim_path,
            "attr_name": self.attr_name,
            "layer_id": self.layer_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


# Top-level hierarchy prims
STAGE_HIERARCHY = (
    "/workflows",
    "/recipes",
    "/executions",
    "/agents",
    "/models",
    "/scenes",
)

# Map Python types to USD value types (checked in order; bool before int)
_TYPE_MAP: list[tuple[type, Any]] = []  # Populated after import guard


def _init_type_map() -> None:
    """Populate type map after confirming pxr is available."""
    global _TYPE_MAP
    _TYPE_MAP = [
        (bool, Sdf.ValueTypeNames.Bool),
        (int, Sdf.ValueTypeNames.Int64),
        (float, Sdf.ValueTypeNames.Double),
        (str, Sdf.ValueTypeNames.String),
    ]


def _sdf_type_for(value: Any) -> Any:
    """Return the Sdf.ValueTypeName for a Python value, or None."""
    for py_type, sdf_type in _TYPE_MAP:
        if isinstance(value, py_type):
            return sdf_type
    return None


class CognitiveWorkflowStage:
    """USD-native composed stage for the entire ecosystem.

    Supports LIVRPS composition via sublayer ordering:
      L (Local)      Agent's direct edit (delta sublayer)  STRONGEST
      I (Inherit)    Agent role constraints
      V (Variants)   Creative profiles
      R (References) Learned recipes
      P (Payloads)   Lazy-loaded components
      S (Specialize) Base workflow templates (base sublayer) WEAKEST
    """

    def __init__(self, root_path: str | Path | None = None):
        """Create or open a stage.

        Args:
            root_path: Path to .usda/.usdc file. None creates in-memory stage.
                       Existing file is loaded into the base layer.
        """
        if not HAS_USD:
            raise StageError(
                "USD not available. Install with: pip install usd-core"
            )
        if not _TYPE_MAP:
            _init_type_map()

        self._root_path = Path(root_path) if root_path else None
        self._agent_deltas: list[Any] = []  # list[Sdf.Layer]

        # W2.1 — subscriber registry. Callbacks fire post-mutation, dispatched
        # by a SINGLE consumer thread reading from a bounded queue, so the
        # writer never blocks on a slow subscriber and we never spawn one
        # thread per event. Lazily started on first subscribe().
        self._subscribers: dict[int, Callable[[StageEvent], None]] = {}
        self._sub_lock = threading.Lock()
        self._sub_next_id = 0
        self._dispatch_queue: queue.Queue | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._dispatch_sentinel = object()
        self._dispatch_drops = 0  # number of events dropped due to full queue

        # Always in-memory. Root layer manages sublayer composition only.
        self._stage = Usd.Stage.CreateInMemory("cognitive_stage.usda")
        self._base_layer = Sdf.Layer.CreateAnonymous("base.usda")

        # Base layer is last sublayer (weakest opinion)
        self._stage.GetRootLayer().subLayerPaths = [
            self._base_layer.identifier
        ]

        if root_path and Path(root_path).exists():
            existing = Sdf.Layer.FindOrOpen(str(root_path))
            if existing:
                self._base_layer.TransferContent(existing)
        else:
            if root_path:
                Path(root_path).parent.mkdir(parents=True, exist_ok=True)
            self._bootstrap()

    def _bootstrap(self) -> None:
        """Create hierarchy prims in the base layer."""
        self._stage.SetEditTarget(Usd.EditTarget(self._base_layer))
        try:
            for path in STAGE_HIERARCHY:
                if not self._stage.GetPrimAtPath(path):
                    self._stage.DefinePrim(path, "Scope")
        finally:
            self._stage.SetEditTarget(self._stage.GetRootLayer())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stage(self) -> Usd.Stage:
        """Direct access to the underlying pxr.Usd.Stage."""
        return self._stage

    @property
    def root_layer(self) -> Sdf.Layer:
        """The root layer of the stage."""
        return self._stage.GetRootLayer()

    @property
    def base_layer(self) -> Sdf.Layer:
        """The base sublayer (weakest opinion, holds workflow data)."""
        return self._base_layer

    @property
    def delta_count(self) -> int:
        """Number of active agent delta sublayers."""
        return len(self._agent_deltas)

    # ------------------------------------------------------------------
    # W2.1/W2.3 — Subscriber registry
    # ------------------------------------------------------------------

    # Bounded queue size. Big enough that bursty traffic doesn't drop
    # under normal load, small enough that a stuck consumer doesn't pin
    # unbounded memory. Drops are logged so the WARN trail is visible.
    _DISPATCH_QUEUE_MAXSIZE: int = 10_000

    def subscribe(self, callback: Callable[[StageEvent], None]) -> int:
        """Register a callback for stage mutation events.

        Returns an integer handle that can be passed to unsubscribe(). The
        callback is invoked once per write / add_delta / rollback / flush,
        on a single dedicated daemon thread so subscriber failures cannot
        block the writer or corrupt stage state, and a 1000-event burst
        does not spawn 1000 threads.
        """
        with self._sub_lock:
            handle = self._sub_next_id
            self._sub_next_id += 1
            self._subscribers[handle] = callback
            self._ensure_dispatcher_locked()
            return handle

    def unsubscribe(self, handle: int) -> bool:
        """Remove a subscriber by handle. Returns True if removed.

        If this was the last subscriber, the dispatcher thread is left
        running but idle (it'll consume any in-flight events and then
        block on the queue). Use `close_subscribers()` to tear it down.
        """
        with self._sub_lock:
            return self._subscribers.pop(handle, None) is not None

    def close_subscribers(self) -> None:
        """Stop the dispatcher thread and clear all subscribers.

        Safe to call multiple times; safe to call before any subscribe().
        Tests use this in teardown so daemon threads don't accumulate
        across pytest runs.
        """
        with self._sub_lock:
            self._subscribers.clear()
            t = self._dispatch_thread
            q = self._dispatch_queue
            self._dispatch_thread = None
            self._dispatch_queue = None
        if q is not None:
            try:
                q.put(self._dispatch_sentinel, timeout=1.0)
            except queue.Full:
                pass
        if t is not None:
            t.join(timeout=2.0)

    def _ensure_dispatcher_locked(self) -> None:
        """Lazily spawn the single dispatcher thread. Call with _sub_lock held."""
        if self._dispatch_thread is not None:
            return
        self._dispatch_queue = queue.Queue(maxsize=self._DISPATCH_QUEUE_MAXSIZE)
        t = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="cozy-stage-dispatch",
        )
        t.start()
        self._dispatch_thread = t

    def _dispatch_loop(self) -> None:
        """Single consumer thread — drains the queue and fans out to subs.

        Exits cleanly when it sees `_dispatch_sentinel`. Never lets a
        subscriber's exception escape (each callback is wrapped in
        try/except per Article V).
        """
        q = self._dispatch_queue
        if q is None:
            return
        while True:
            event = q.get()
            if event is self._dispatch_sentinel:
                return
            with self._sub_lock:
                callbacks = list(self._subscribers.values())
            for cb in callbacks:
                try:
                    cb(event)
                except Exception as exc:
                    _log.warning(
                        "Stage subscriber %r failed on %s: %s",
                        cb, event.op, exc,
                    )

    def _emit(self, event: StageEvent) -> None:
        """Notify all subscribers of an event. Never raises.

        Non-blocking — events are enqueued for the dispatcher thread.
        On queue overflow (slow/stuck consumer) the event is DROPPED with
        a WARN log; the writer never blocks. This violates Article V's
        "no silent state changes" if it ever fires, so the drop counter
        is exposed via `dispatch_drops` for observability.
        """
        if self._dispatch_queue is None:
            # No subscribers ever attached — fast path, no work.
            return
        try:
            self._dispatch_queue.put_nowait(event)
        except queue.Full:
            self._dispatch_drops += 1
            _log.warning(
                "Stage subscriber queue full (drops=%d) — dropping %s event",
                self._dispatch_drops, event.op,
            )

    @property
    def dispatch_drops(self) -> int:
        """Total events dropped because the dispatcher queue was full."""
        return self._dispatch_drops

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def read(self, prim_path: str, attr_name: str | None = None) -> Any:
        """Read an attribute value or check prim existence.

        Returns the composed/resolved value (all LIVRPS applied).

        Args:
            prim_path: USD prim path (e.g. "/workflows/my_workflow").
            attr_name: Attribute name. If None, returns True/False for existence.

        Returns:
            Resolved attribute value, True (prim exists), or None (not found).
        """
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        if attr_name is None:
            return True
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            return None
        return attr.Get()

    def write(
        self,
        prim_path: str,
        attr_name: str,
        value: Any,
        *,
        node_type: str | None = None,
    ) -> None:
        """Write an attribute value to the base layer (weakest opinion).

        Agent deltas override these values via sublayer composition.

        Args:
            prim_path: USD prim path.
            attr_name: Attribute name.
            value: Value to write.
            node_type: If provided, checks anchor immunity before writing.

        Raises:
            AnchorViolationError: If writing to a protected parameter.
            StageError: If value type is unsupported.
        """
        if node_type:
            check_anchor(node_type, attr_name)

        self._stage.SetEditTarget(Usd.EditTarget(self._base_layer))
        try:
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                prim = self._stage.DefinePrim(prim_path)

            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.Set(value)
            else:
                sdf_type = _sdf_type_for(value)
                if sdf_type is None:
                    raise StageError(
                        f"Unsupported type {type(value).__name__} "
                        f"for '{attr_name}'"
                    )
                prim.CreateAttribute(attr_name, sdf_type).Set(value)
        finally:
            self._stage.SetEditTarget(self._stage.GetRootLayer())

        self._emit(StageEvent(
            op="write",
            prim_path=prim_path,
            attr_name=attr_name,
        ))

    # ------------------------------------------------------------------
    # Agent Deltas (LIVRPS Local opinion)
    # ------------------------------------------------------------------

    def add_agent_delta(
        self, agent_name: str, delta_dict: dict[str, Any]
    ) -> str:
        """Apply agent modifications as a new sublayer.

        Each delta is a separate anonymous sublayer inserted before the
        base layer (stronger opinion). Clean rollback via rollback_to().

        Args:
            agent_name: Agent identifier (e.g. "forge", "scout").
            delta_dict: Maps "prim_path:attr_name" to value.
                        Example: {"/workflows/w1/nodes/node_3:steps": 30}

        Returns:
            Layer identifier of the created delta.
        """
        layer = Sdf.Layer.CreateAnonymous(f"{agent_name}_delta.usda")

        for key, value in delta_dict.items():
            # Split prim_path from attr_name at the first colon after
            # the last slash. Attr names can contain colons (USD namespaces
            # like "input:steps"), but prim paths cannot.
            last_slash = key.rfind("/")
            colon_pos = key.find(":", last_slash + 1)
            if colon_pos == -1:
                raise StageError(
                    f"Delta key must be 'prim_path:attr_name', got '{key}'"
                )
            prim_path = key[:colon_pos]
            attr_name = key[colon_pos + 1:]

            # Build prim spec hierarchy with Over specifier
            parts = prim_path.strip("/").split("/")
            current = None
            for i in range(len(parts)):
                spec_path = "/" + "/".join(parts[: i + 1])
                current = Sdf.CreatePrimInLayer(layer, spec_path)
                current.specifier = Sdf.SpecifierOver

            if current is None:
                continue

            sdf_type = _sdf_type_for(value)
            if sdf_type is None:
                raise StageError(
                    f"Unsupported type {type(value).__name__} for '{key}'"
                )
            Sdf.AttributeSpec(current, attr_name, sdf_type).default = value

        # Insert at front of sublayer list (strongest position).
        # Base layer is always last.
        root = self._stage.GetRootLayer()
        paths = list(root.subLayerPaths)
        paths.insert(0, layer.identifier)
        root.subLayerPaths = paths

        self._agent_deltas.append(layer)

        self._emit(StageEvent(
            op="add_delta",
            layer_id=layer.identifier,
            payload={"agent_name": agent_name, "keys": list(delta_dict.keys())},
        ))
        return layer.identifier

    def rollback_to(self, n_deltas: int) -> int:
        """Remove the top N most-recent agent delta sublayers.

        Args:
            n_deltas: Number of deltas to remove.

        Returns:
            Number of deltas actually removed.
        """
        to_remove = min(n_deltas, len(self._agent_deltas))
        if to_remove == 0:
            return 0

        # Most recent delta is self._agent_deltas[-1], which is at
        # subLayerPaths[0] (strongest position). Pop from the end.
        removed_ids = set()
        for _ in range(to_remove):
            removed = self._agent_deltas.pop()
            removed_ids.add(removed.identifier)

        root = self._stage.GetRootLayer()
        root.subLayerPaths = [
            p for p in root.subLayerPaths if p not in removed_ids
        ]

        self._emit(StageEvent(
            op="rollback",
            payload={"removed_count": to_remove},
        ))
        return to_remove

    def list_deltas(self) -> list[str]:
        """List identifiers of all agent delta sublayers (oldest first)."""
        return [layer.identifier for layer in self._agent_deltas]

    # ------------------------------------------------------------------
    # Variant Profiles
    # ------------------------------------------------------------------

    def select_profile(
        self, prim_path: str, variant_set: str, profile_name: str
    ) -> None:
        """Select a variant on a prim (creative profiles, model configs).

        Args:
            prim_path: Path to the prim with variant sets.
            variant_set: Name of the variant set (e.g. "style").
            profile_name: Variant to select (e.g. "explore").
        """
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise StageError(f"Prim not found: {prim_path}")
        vsets = prim.GetVariantSets()
        if variant_set not in vsets.GetNames():
            raise StageError(
                f"Variant set '{variant_set}' not found on {prim_path}"
            )
        vset = vsets.GetVariantSet(variant_set)
        # Some pxr versions raise on invalid names; others silently accept
        # them. Cross-check `variantNames` upfront so behavior is uniform.
        valid_names = list(vset.GetVariantNames())
        if profile_name not in valid_names:
            raise StageError(
                f"Could not select variant '{profile_name}' on "
                f"{prim_path}/{variant_set}: not in {valid_names}"
            )
        try:
            vset.SetVariantSelection(profile_name)
        except Exception as _e:
            raise StageError(
                f"Could not select variant '{profile_name}' on "
                f"{prim_path}/{variant_set}: {_e}"
            ) from _e

    # ------------------------------------------------------------------
    # Reconstruction & Inspection
    # ------------------------------------------------------------------

    def reconstruct_clean(self) -> dict[str, dict[str, Any]]:
        """Read base layer content without agent deltas.

        Temporarily sets sublayers to just the base layer to see only
        the base-authored opinions. Safe via try/finally.

        Returns:
            Dict of {prim_path: {attr_name: value}} from base layer only.
        """
        root = self._stage.GetRootLayer()
        saved = list(root.subLayerPaths)
        root.subLayerPaths = [self._base_layer.identifier]
        try:
            result: dict[str, dict[str, Any]] = {}
            for prim in self._stage.Traverse():
                path = str(prim.GetPath())
                attrs = {}
                for attr in prim.GetAttributes():
                    val = attr.Get()
                    if val is not None:
                        attrs[attr.GetName()] = val
                if attrs:
                    result[path] = attrs
            return result
        finally:
            root.subLayerPaths = saved

    def prim_exists(self, prim_path: str) -> bool:
        """Check if a prim exists at the given path."""
        return self._stage.GetPrimAtPath(prim_path).IsValid()

    def list_children(self, prim_path: str) -> list[str]:
        """List child prim names under a path."""
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return []
        return [child.GetName() for child in prim.GetChildren()]

    def get_prim_attrs(self, prim_path: str) -> dict[str, Any]:
        """Get all attributes on a prim as a dict."""
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {}
        result = {}
        for attr in prim.GetAttributes():
            val = attr.Get()
            if val is not None:
                result[attr.GetName()] = val
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self, output_path: str | Path | None = None) -> str:
        """Save the composed (flattened) stage to disk.

        Flattens all sublayers into a single file. Agent deltas are
        baked in. Use export_flat() for the same behavior explicitly.

        The write is atomic: the flattened stage is exported to a sibling
        `<path>.tmp` file and then `os.replace`d into place. A SIGKILL
        between Export and replace leaves the .tmp orphaned but the
        canonical path either intact (pre-flush) or fully rewritten
        (post-flush) — never partially written. Per Article IV of the
        Cozy Constitution, checkpoint integrity requires this.

        Args:
            output_path: Override path. Uses root_path if not provided.

        Returns:
            Path the stage was saved to.

        Raises:
            StageError: If no output path available (in-memory stage).
        """
        path = str(output_path) if output_path else (
            str(self._root_path) if self._root_path else None
        )
        if path is None:
            raise StageError(
                "No output path. Stage is in-memory — provide output_path."
            )
        tmp_path = path + ".tmp"
        try:
            self._stage.Flatten().Export(tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort cleanup of the orphaned tmp file. If the cleanup
            # itself fails, the next flush will overwrite it.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._emit(StageEvent(
            op="flush",
            payload={"path": path},
        ))
        return path

    def export_flat(self, output_path: str | Path) -> str:
        """Export a flattened stage (all composition resolved).

        Args:
            output_path: Where to write the flattened stage.

        Returns:
            Path the flattened stage was saved to.
        """
        path = str(output_path)
        self._stage.Flatten().Export(path)
        return path

    def to_usda(self) -> str:
        """Return the root layer as a USDA text string (shows composition)."""
        return self._stage.GetRootLayer().ExportToString()
