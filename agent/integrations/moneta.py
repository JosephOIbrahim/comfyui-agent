"""Moneta reference adapter — bidirectional file-watch transport.

This module is the placeholder for the eventual Moneta memory-substrate
integration. We don't yet have Moneta's wire format, so the reference
adapter uses a transport-agnostic shape:

  outbox/  — every StageEvent is appended to a daily JSONL file.
             Moneta (or any external consumer) tails this file.

  inbox/   — Moneta drops `*.delta.json` files containing
             `{"agent_name": str, "delta": {...}}`. The watcher reads
             each file, applies it as a stage agent_delta, and atomically
             renames the file to `*.applied`.

When Moneta's actual API contract lands, replace the file-watch transport
in `_emit()` and `_poll_inbox()` with HTTP/RPC; the StageEvent → record
mapping in `_event_to_record()` stays the same.

Per Article V of the Cozy Constitution, this adapter is a passive
subscriber — failures here cannot block stage writes (the subscriber
fan-out runs on a daemon thread with try/except isolation; see
`CognitiveWorkflowStage._emit()`).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stage.cognitive_stage import StageEvent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MonetaAdapterConfig:
    """Configuration for the Moneta reference adapter."""

    outbox_dir: Path
    """Directory where outbound StageEvents are appended as JSONL."""

    inbox_dir: Path | None = None
    """Optional directory where Moneta writes inbound delta files. None
    disables the inbox watcher (export-only mode)."""

    poll_interval_seconds: float = 2.0
    """How often the inbox watcher scans for new delta files."""

    schema_version: str = "moneta-v0"
    """Schema version stamp embedded in every outbound record. Bump
    when the wire format changes so consumers can migrate gracefully."""

    rotate_daily: bool = True
    """When True, outbound JSONL files rotate daily (`stage-events-YYYYMMDD.jsonl`).
    When False, all events go to a single `stage-events.jsonl`."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MonetaAdapter:
    """Bidirectional file-watch transport between Cozy stage and Moneta.

    Usage:
        adapter = MonetaAdapter(config, stage)
        adapter.start()
        ... stage operations emit events; deltas are ingested ...
        adapter.stop()

    The adapter never raises from inside its callbacks — the stage
    subscriber fan-out tolerates exceptions, but we additionally swallow
    them at the adapter level so transient FS issues don't fill the log.
    """

    def __init__(self, config: MonetaAdapterConfig, stage: Any):
        self._config = config
        self._stage = stage
        self._sub_handle: int | None = None
        self._inbox_thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._outbox_lock = threading.Lock()  # serialize JSONL appends
        self._events_emitted = 0
        self._deltas_ingested = 0
        self._ingest_failures = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to stage events and (if inbox configured) watch deltas."""
        self._config.outbox_dir.mkdir(parents=True, exist_ok=True)

        # Outbound: subscribe to stage events
        self._sub_handle = self._stage.subscribe(self._on_stage_event)

        # Inbound: optional inbox watcher thread
        if self._config.inbox_dir is not None:
            self._config.inbox_dir.mkdir(parents=True, exist_ok=True)
            t = threading.Thread(
                target=self._inbox_loop, daemon=True, name="moneta-inbox",
            )
            t.start()
            self._inbox_thread = t

    def stop(self) -> None:
        """Unsubscribe and stop the inbox watcher. Safe to call repeatedly."""
        self._shutdown.set()
        if self._sub_handle is not None:
            try:
                self._stage.unsubscribe(self._sub_handle)
            except Exception:
                pass
            self._sub_handle = None
        if self._inbox_thread is not None:
            self._inbox_thread.join(timeout=self._config.poll_interval_seconds + 1.0)
            self._inbox_thread = None

    # ------------------------------------------------------------------
    # Stats (for observability / tests)
    # ------------------------------------------------------------------

    @property
    def events_emitted(self) -> int:
        return self._events_emitted

    @property
    def deltas_ingested(self) -> int:
        return self._deltas_ingested

    @property
    def ingest_failures(self) -> int:
        return self._ingest_failures

    # ------------------------------------------------------------------
    # Outbound: stage event → JSONL
    # ------------------------------------------------------------------

    def _on_stage_event(self, event: StageEvent) -> None:
        """Subscriber callback — append the event to today's JSONL file."""
        try:
            record = self._event_to_record(event)
            self._append_jsonl(record)
            self._events_emitted += 1
        except Exception as exc:
            # Article V: failures here MUST NOT block writers. Log and move on.
            log.warning("Moneta outbound emit failed: %s", exc)

    def _event_to_record(self, event: StageEvent) -> dict[str, Any]:
        """Translate a StageEvent into the Moneta wire record.

        Schema (v0):
          {
            "schema": "moneta-v0",
            "ts": float (epoch seconds),
            "op": "write" | "add_delta" | "rollback" | "flush",
            "prim_path": str | null,
            "attr_name": str | null,
            "layer_id": str | null,
            "payload": object | null,
          }

        When the real Moneta wire format lands, change this method only.
        """
        return {
            "schema": self._config.schema_version,
            "ts": event.timestamp,
            "op": event.op,
            "prim_path": event.prim_path,
            "attr_name": event.attr_name,
            "layer_id": event.layer_id,
            "payload": event.payload,
        }

    def _outbox_path(self) -> Path:
        """Resolve the JSONL file for today (or the static file if no rotation)."""
        if self._config.rotate_daily:
            day = time.strftime("%Y%m%d", time.gmtime())
            return self._config.outbox_dir / f"stage-events-{day}.jsonl"
        return self._config.outbox_dir / "stage-events.jsonl"

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        """Append one JSON record to the outbox file. Lock-serialized."""
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with self._outbox_lock:
            with open(self._outbox_path(), "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

    # ------------------------------------------------------------------
    # Inbound: Moneta delta files → stage agent_delta
    # ------------------------------------------------------------------

    def _inbox_loop(self) -> None:
        """Poll the inbox directory for `*.delta.json` files."""
        assert self._config.inbox_dir is not None
        inbox = self._config.inbox_dir
        while not self._shutdown.is_set():
            try:
                # Sort so we ingest in deterministic order — useful for tests
                # and for upstream Moneta's "send-and-forget" semantics.
                for path in sorted(inbox.glob("*.delta.json")):
                    if self._shutdown.is_set():
                        break
                    self._ingest_one(path)
            except Exception as exc:
                log.warning("Moneta inbox scan failed: %s", exc)
                self._ingest_failures += 1
            self._shutdown.wait(self._config.poll_interval_seconds)

    def _ingest_one(self, path: Path) -> None:
        """Read one delta file, apply it, then mark it as applied.

        File format:
          {
            "agent_name": "moneta",
            "delta": {"<prim_path>:<attr_name>": <value>, ...}
          }

        On success, rename `<name>.delta.json` → `<name>.delta.json.applied`.
        On failure, rename to `<name>.delta.json.failed` so the file isn't
        retried on the next scan (caller should inspect and re-stage).
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            agent_name = doc.get("agent_name", "moneta")
            delta = doc.get("delta")
            if not isinstance(delta, dict) or not delta:
                raise ValueError(f"delta must be a non-empty dict, got {type(delta).__name__}")
            self._stage.add_agent_delta(agent_name, delta)
            path.rename(path.with_suffix(path.suffix + ".applied"))
            self._deltas_ingested += 1
            log.info("Moneta delta ingested: %s (%d keys)", path.name, len(delta))
        except Exception as exc:
            log.warning("Moneta delta ingest failed for %s: %s", path.name, exc)
            self._ingest_failures += 1
            try:
                path.rename(path.with_suffix(path.suffix + ".failed"))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience constructor from .env config
# ---------------------------------------------------------------------------

def from_env(stage: Any) -> MonetaAdapter | None:
    """Build a MonetaAdapter from environment variables, or None if disabled.

    Reads:
      MONETA_OUTBOX_DIR  — required to enable; absolute path
      MONETA_INBOX_DIR   — optional; enables bidirectional ingest
      MONETA_POLL_SECONDS — optional; default 2.0

    Raises:
      MonetaConfigError if either env-supplied directory fails the
      sandbox check from `agent/tools/_util.py:validate_path`. Pre-fix
      (T7 from the 5x review), an env var like MONETA_OUTBOX_DIR=/etc
      would have created/written JSONL into /etc unchallenged. The
      validate_path gate restricts directories to the configured
      sandbox (project dir, COMFYUI_DATABASE, sessions, workflows,
      tempdir).
    """
    import os
    outbox = os.getenv("MONETA_OUTBOX_DIR", "").strip()
    if not outbox:
        return None
    inbox = os.getenv("MONETA_INBOX_DIR", "").strip()
    poll = float(os.getenv("MONETA_POLL_SECONDS", "2.0"))

    # T7: validate the env-supplied directories. validate_path returns an
    # error string if rejected, None if accepted.
    from ..tools._util import validate_path
    err = validate_path(outbox)
    if err:
        raise MonetaConfigError(
            f"MONETA_OUTBOX_DIR rejected: {err}"
        )
    if inbox:
        err = validate_path(inbox)
        if err:
            raise MonetaConfigError(
                f"MONETA_INBOX_DIR rejected: {err}"
            )

    config = MonetaAdapterConfig(
        outbox_dir=Path(outbox),
        inbox_dir=Path(inbox) if inbox else None,
        poll_interval_seconds=poll,
    )
    return MonetaAdapter(config, stage)


class MonetaConfigError(ValueError):
    """Raised when MONETA_* env vars fail validation."""
