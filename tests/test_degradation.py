"""CRUCIBLE tests for degradation manager (agent/degradation.py).

Adversarial: tests circuit breaker integration, thread safety,
unregistered subsystem errors, fallback counting, and reset behavior.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent.circuit_breaker import CircuitBreaker
from agent.degradation import DegradationManager, SubsystemStatus


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestDegradationManagerBasics:
    def test_fresh_manager_no_subsystems(self):
        dm = DegradationManager()
        assert dm.status() == {}

    def test_register_subsystem(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "fallback")
        status = dm.status()
        assert "test" in status

    def test_healthy_by_default(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "fallback")
        assert dm.is_healthy("test") is True
        ss = dm.status()["test"]
        assert ss.healthy is True

    def test_unregistered_is_not_healthy(self):
        dm = DegradationManager()
        assert dm.is_healthy("nonexistent") is False


# ---------------------------------------------------------------------------
# with_fallback
# ---------------------------------------------------------------------------


class TestWithFallback:
    def test_success_returns_fn_result(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda x: "FALLBACK")
        result = dm.with_fallback("test", lambda x: x * 2, 21)
        assert result == 42

    def test_failure_returns_fallback(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "FALLBACK")

        def fail():
            raise RuntimeError("boom")

        result = dm.with_fallback("test", fail)
        assert result == "FALLBACK"

    def test_failure_tracks_error(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "ok")

        def fail():
            raise ValueError("specific error")

        dm.with_fallback("test", fail)
        ss = dm.status()["test"]
        assert ss.healthy is False
        assert ss.last_error is not None
        assert "ValueError" in ss.last_error
        assert "specific error" in ss.last_error

    def test_failure_increments_fallback_count(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "ok")

        for i in range(5):
            dm.with_fallback("test", lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        ss = dm.status()["test"]
        assert ss.fallback_invocations == 5

    def test_success_after_failure_restores_healthy(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "fallback")

        # Fail
        dm.with_fallback("test", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert dm.is_healthy("test") is False

        # Succeed
        dm.with_fallback("test", lambda: 42)
        assert dm.is_healthy("test") is True
        ss = dm.status()["test"]
        assert ss.last_error is None

    def test_unregistered_subsystem_raises(self):
        dm = DegradationManager()
        with pytest.raises(KeyError, match="not registered"):
            dm.with_fallback("nonexistent", lambda: 42)

    def test_fallback_receives_same_args(self):
        """Fallback should receive the same *args and **kwargs as fn."""
        dm = DegradationManager()
        captured: list = []

        def fallback(*args, **kwargs):
            captured.append((args, kwargs))
            return "FALLBACK"

        dm.register("test", fallback=fallback)

        def fail(a, b, key=None):
            raise RuntimeError("boom")

        dm.with_fallback("test", fail, "x", "y", key="z")
        assert len(captured) == 1
        assert captured[0] == (("x", "y"), {"key": "z"})


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    def test_breaker_open_skips_fn(self):
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=999)
        dm = DegradationManager()
        call_count = 0

        def real_fn():
            nonlocal call_count
            call_count += 1
            return "REAL"

        dm.register("test", fallback=lambda: "FALLBACK", breaker=breaker)

        # Trip the breaker
        breaker.record_failure()
        assert breaker.state == "open"

        result = dm.with_fallback("test", real_fn)
        assert result == "FALLBACK"
        assert call_count == 0, "fn should not have been called with open breaker"

    def test_breaker_records_success(self):
        breaker = CircuitBreaker("test", failure_threshold=3)
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "FALLBACK", breaker=breaker)

        # Record 2 failures (not enough to trip)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "closed"

        # Successful call should reset
        dm.with_fallback("test", lambda: "ok")
        assert breaker.state == "closed"

    def test_breaker_records_failure(self):
        breaker = CircuitBreaker("test", failure_threshold=2)
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "FALLBACK", breaker=breaker)

        dm.with_fallback("test", lambda: (_ for _ in ()).throw(RuntimeError("1")))
        dm.with_fallback("test", lambda: (_ for _ in ()).throw(RuntimeError("2")))
        assert breaker.state == "open"

    def test_circuit_state_in_status(self):
        breaker = CircuitBreaker("test", failure_threshold=1)
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "ok", breaker=breaker)
        breaker.record_failure()
        ss = dm.status()["test"]
        assert ss.circuit_state == "open"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_error(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "ok")
        dm.with_fallback("test", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert dm.is_healthy("test") is False

        dm.reset("test")
        assert dm.is_healthy("test") is True
        ss = dm.status()["test"]
        assert ss.last_error is None
        assert ss.fallback_invocations == 0

    def test_reset_clears_breaker(self):
        breaker = CircuitBreaker("test", failure_threshold=1)
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "ok", breaker=breaker)
        breaker.record_failure()
        assert breaker.state == "open"

        dm.reset("test")
        assert breaker.state == "closed"

    def test_reset_unregistered_raises(self):
        dm = DegradationManager()
        with pytest.raises(KeyError, match="not registered"):
            dm.reset("nonexistent")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_empty(self):
        dm = DegradationManager()
        assert "No subsystems registered" in dm.summary()

    def test_summary_human_readable(self):
        dm = DegradationManager()
        dm.register("comfyui", fallback=lambda: "offline")
        dm.register("session", fallback=lambda: "no_session")
        summary = dm.summary()
        assert isinstance(summary, str)
        assert "comfyui" in summary
        assert "session" in summary
        assert "2/2" in summary  # both healthy

    def test_summary_shows_degraded(self):
        dm = DegradationManager()
        dm.register("broken", fallback=lambda: "ok")
        dm.with_fallback("broken", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        summary = dm.summary()
        assert "DEGRADED" in summary
        assert "0/1" in summary


# ---------------------------------------------------------------------------
# SubsystemStatus
# ---------------------------------------------------------------------------


class TestSubsystemStatus:
    def test_status_fields(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "ok")
        ss = dm.status()["test"]
        assert isinstance(ss, SubsystemStatus)
        assert ss.name == "test"
        assert ss.healthy is True
        assert ss.circuit_state == "closed"
        assert ss.last_error is None
        assert ss.fallback_invocations == 0

    def test_register_replaces_existing(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda: "v1")
        dm.with_fallback("test", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert dm.is_healthy("test") is False
        # Re-register should replace
        dm.register("test", fallback=lambda: "v2")
        assert dm.is_healthy("test") is True


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestDegradationThreadSafety:
    def test_concurrent_with_fallback(self):
        dm = DegradationManager()
        dm.register("test", fallback=lambda x: f"fallback_{x}")
        errors: list[str] = []
        results: list = []
        lock = threading.Lock()

        def run_many(thread_id: int) -> None:
            try:
                for i in range(50):
                    if i % 3 == 0:
                        # Fail sometimes
                        def fail(x):
                            raise RuntimeError(f"thread_{thread_id}_{i}")
                        r = dm.with_fallback("test", fail, i)
                    else:
                        r = dm.with_fallback("test", lambda x: x * 2, i)
                    with lock:
                        results.append(r)
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=run_many, args=(tid,))
            for tid in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 500
        # Some should be fallback results, some real results
        fallback_results = [r for r in results if isinstance(r, str) and "fallback" in str(r)]
        real_results = [r for r in results if isinstance(r, int)]
        assert len(fallback_results) > 0
        assert len(real_results) > 0

    def test_concurrent_register_and_query(self):
        dm = DegradationManager()
        errors: list[str] = []

        def register_many(start: int) -> None:
            try:
                for i in range(20):
                    dm.register(f"sub_{start}_{i}", fallback=lambda: "ok")
            except Exception as exc:
                errors.append(str(exc))

        def query_status() -> None:
            try:
                for _ in range(20):
                    dm.status()
                    dm.summary()
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=register_many, args=(tid * 100,))
            for tid in range(5)
        ] + [
            threading.Thread(target=query_status)
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
