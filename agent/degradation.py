"""Degradation manager with per-subsystem fallback and kill switches.

Tracks subsystem health, composes with the existing CircuitBreaker
infrastructure, and ensures the MCP server never crashes due to a
single subsystem failure.  Thread-safe (RLock).

Usage::

    from agent.degradation import DegradationManager
    from agent.circuit_breaker import get_breaker

    dm = DegradationManager()
    dm.register("comfyui", fallback=lambda *a, **k: '{"error":"offline"}',
                breaker=get_breaker("comfyui"))

    result = dm.with_fallback("comfyui", risky_http_call, url, timeout=5)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .circuit_breaker import CircuitBreaker

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SubsystemStatus:
    """Health snapshot for a single subsystem."""

    name: str
    healthy: bool = True
    circuit_state: str = "closed"
    last_error: str | None = None
    fallback_invocations: int = 0
    last_checked: float = 0.0


class _SubsystemEntry:
    """Internal bookkeeping for a registered subsystem (not exported)."""

    __slots__ = (
        "name", "fallback", "breaker",
        "healthy", "last_error", "fallback_invocations", "last_checked",
    )

    def __init__(
        self,
        name: str,
        fallback: Callable[..., Any],
        breaker: CircuitBreaker | None,
    ) -> None:
        self.name = name
        self.fallback = fallback
        self.breaker = breaker
        self.healthy: bool = True
        self.last_error: str | None = None
        self.fallback_invocations: int = 0
        self.last_checked: float = 0.0

    def snapshot(self) -> SubsystemStatus:
        circuit = "closed"
        if self.breaker is not None:
            circuit = self.breaker.state
        return SubsystemStatus(
            name=self.name,
            healthy=self.healthy,
            circuit_state=circuit,
            last_error=self.last_error,
            fallback_invocations=self.fallback_invocations,
            last_checked=self.last_checked,
        )


class DegradationManager:
    """Tracks subsystem health and provides automatic fallback.

    Thread-safe.  Each subsystem has an independent fallback function so
    that failures in one area (e.g. ComfyUI HTTP) never cascade into
    unrelated tools (e.g. session persistence).
    """

    def __init__(self) -> None:
        self._entries: dict[str, _SubsystemEntry] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        fallback: Callable[..., Any],
        breaker: CircuitBreaker | None = None,
    ) -> None:
        """Register a subsystem with its fallback and optional breaker.

        If a subsystem with the same name is already registered it is
        replaced (useful for testing).
        """
        with self._lock:
            self._entries[name] = _SubsystemEntry(name, fallback, breaker)
        log.debug("Registered subsystem '%s' (breaker=%s)", name, breaker)

    # ------------------------------------------------------------------
    # Execution with fallback
    # ------------------------------------------------------------------

    def with_fallback(
        self,
        name: str,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute *fn* with automatic fallback on failure.

        Flow:
        1. If a circuit breaker is registered and open, skip straight to
           fallback (fast-fail).
        2. Try ``fn(*args, **kwargs)``.
        3. On success: mark healthy, record_success on breaker.
        4. On exception: mark unhealthy, record_failure on breaker, invoke
           fallback with the same arguments, increment counter.

        Raises ``KeyError`` if *name* is not registered.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(
                    f"Subsystem '{name}' not registered. "
                    f"Available: {sorted(self._entries)}"
                )
            breaker = entry.breaker

        # Fast-fail if circuit is open
        if breaker is not None and not breaker.allow_request():
            log.debug(
                "Subsystem '%s': circuit open, invoking fallback directly",
                name,
            )
            with self._lock:
                entry.fallback_invocations += 1
                entry.last_checked = time.monotonic()
            return entry.fallback(*args, **kwargs)

        # Try the real function
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            # Record failure
            if breaker is not None:
                breaker.record_failure()
            with self._lock:
                entry.healthy = False
                entry.last_error = f"{type(exc).__name__}: {exc}"
                entry.fallback_invocations += 1
                entry.last_checked = time.monotonic()
            log.warning(
                "Subsystem '%s' failed (%s), invoking fallback (#%d)",
                name, entry.last_error, entry.fallback_invocations,
            )
            return entry.fallback(*args, **kwargs)

        # Record success
        if breaker is not None:
            breaker.record_success()
        with self._lock:
            entry.healthy = True
            entry.last_error = None
            entry.last_checked = time.monotonic()

        return result

    # ------------------------------------------------------------------
    # Health queries
    # ------------------------------------------------------------------

    def is_healthy(self, name: str) -> bool:
        """Check if a subsystem is currently marked healthy."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return False
            return entry.healthy

    def status(self) -> dict[str, SubsystemStatus]:
        """Get health status of all registered subsystems."""
        with self._lock:
            return {
                name: entry.snapshot()
                for name, entry in sorted(self._entries.items())
            }

    def reset(self, name: str) -> None:
        """Reset a subsystem to healthy state (e.g. after manual fix).

        Also resets the associated circuit breaker if one exists.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(f"Subsystem '{name}' not registered.")
            entry.healthy = True
            entry.last_error = None
            entry.fallback_invocations = 0
            entry.last_checked = time.monotonic()
            if entry.breaker is not None:
                entry.breaker.reset()
        log.info("Subsystem '%s' reset to healthy", name)

    def summary(self) -> str:
        """Human-readable health summary for MCP tool responses."""
        statuses = self.status()
        if not statuses:
            return "No subsystems registered."

        lines: list[str] = []
        healthy_count = sum(1 for s in statuses.values() if s.healthy)
        total = len(statuses)
        lines.append(f"Health: {healthy_count}/{total} subsystems OK")

        for name, ss in statuses.items():
            icon = "OK" if ss.healthy else "DEGRADED"
            line = f"  [{icon}] {name} (circuit={ss.circuit_state})"
            if ss.fallback_invocations > 0:
                line += f" fallbacks={ss.fallback_invocations}"
            if ss.last_error:
                line += f" last_err={ss.last_error[:80]}"
            lines.append(line)

        return "\n".join(lines)
