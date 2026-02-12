"""Tests for circuit_breaker.py — circuit breaker pattern for external services."""

import threading
import time

import pytest

from agent.circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    COMFYUI_BREAKER,
    CircuitBreaker,
    get_breaker,
    reset_all,
)


@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset global breaker registry between tests."""
    yield
    reset_all()


class TestCircuitBreakerStates:
    def test_starts_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CLOSED

    def test_allows_requests_when_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.allow_request() is True

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        assert cb.state == CLOSED
        cb.record_failure()
        assert cb.state == CLOSED
        cb.record_failure()
        assert cb.state == OPEN

    def test_rejects_requests_when_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == OPEN
        assert cb.allow_request() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CLOSED
        # Should need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CLOSED

    def test_transitions_to_half_open_after_recovery(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.state == OPEN
        time.sleep(0.06)
        assert cb.allow_request() is True
        assert cb.state == HALF_OPEN

    def test_half_open_limits_requests(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05,
                            half_open_max=1)
        cb.record_failure()
        time.sleep(0.06)
        # First request in half-open allowed
        assert cb.allow_request() is True
        # Second blocked
        assert cb.allow_request() is False

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_success()
        assert cb.state == CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_failure()
        assert cb.state == OPEN


class TestCircuitBreakerReset:
    def test_reset_returns_to_initial(self):
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == OPEN
        cb.reset()
        assert cb.state == CLOSED
        assert cb.allow_request() is True


class TestGlobalRegistry:
    def test_get_breaker_returns_singleton(self):
        b1 = get_breaker("my_service")
        b2 = get_breaker("my_service")
        assert b1 is b2

    def test_different_names_different_instances(self):
        b1 = get_breaker("service_a")
        b2 = get_breaker("service_b")
        assert b1 is not b2

    def test_comfyui_breaker_defaults(self):
        cb = COMFYUI_BREAKER()
        assert cb.name == "comfyui"
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0

    def test_comfyui_breaker_singleton(self):
        assert COMFYUI_BREAKER() is COMFYUI_BREAKER()


class TestThreadSafety:
    def test_concurrent_failures(self):
        """Multiple threads recording failures should be safe."""
        cb = CircuitBreaker(name="test_threads", failure_threshold=100)
        errors = []

        def record_many():
            try:
                for _ in range(50):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cb.state == OPEN  # 200 failures > threshold of 100

    def test_concurrent_allow_request(self):
        """Concurrent allow_request calls should not crash."""
        cb = CircuitBreaker(name="test_allow", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)

        results = []

        def check():
            results.append(cb.allow_request())

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should be True (the first to transition to HALF_OPEN)
        assert any(results)
