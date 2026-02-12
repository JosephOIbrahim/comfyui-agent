"""Tests for agent/rate_limiter.py — token bucket rate limiter."""

import threading
import time

import pytest

from agent.rate_limiter import (
    GlobalRateLimiter,
    RateLimiter,
    CIVITAI_LIMITER,
    HUGGINGFACE_LIMITER,
    VISION_LIMITER,
)


@pytest.fixture(autouse=True)
def reset_global():
    """Reset the global registry between tests."""
    GlobalRateLimiter.reset()
    yield
    GlobalRateLimiter.reset()


class TestRateLimiter:
    def test_acquire_succeeds_with_capacity(self):
        limiter = RateLimiter(rate=1.0, capacity=5)
        assert limiter.acquire() is True

    def test_acquire_depletes_tokens(self):
        limiter = RateLimiter(rate=0.01, capacity=2)  # Very slow refill
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        # Third should fail (non-blocking)
        assert limiter.acquire() is False

    def test_acquire_with_timeout_succeeds_after_refill(self):
        limiter = RateLimiter(rate=100.0, capacity=1)  # Fast refill
        limiter.acquire()  # Drain
        # Should refill quickly
        assert limiter.acquire(timeout=1.0) is True

    def test_acquire_with_timeout_fails_when_slow_refill(self):
        limiter = RateLimiter(rate=0.01, capacity=1)  # Very slow
        limiter.acquire()  # Drain
        # Should time out
        assert limiter.acquire(timeout=0.05) is False

    def test_refill_over_time(self):
        limiter = RateLimiter(rate=100.0, capacity=5)
        # Drain all tokens
        for _ in range(5):
            limiter.acquire()
        # Wait for refill
        time.sleep(0.05)
        assert limiter.acquire() is True

    def test_available_property(self):
        limiter = RateLimiter(rate=1.0, capacity=3)
        assert limiter.available == 3.0
        limiter.acquire()
        assert limiter.available < 3.0

    def test_capacity_is_ceiling(self):
        limiter = RateLimiter(rate=1000.0, capacity=5)
        time.sleep(0.1)  # Would refill way past capacity
        assert limiter.available <= 5.0

    def test_thread_safety(self):
        """Multiple threads acquiring concurrently should not over-allocate."""
        limiter = RateLimiter(rate=0.01, capacity=10)  # 10 tokens, near-zero refill
        acquired = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            result = limiter.acquire()
            acquired.append(result)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 10 tokens available, all 10 should succeed
        assert sum(acquired) == 10


class TestGlobalRateLimiter:
    def test_singleton_behavior(self):
        a = GlobalRateLimiter.get("test", rate=1.0, capacity=5)
        b = GlobalRateLimiter.get("test", rate=1.0, capacity=5)
        assert a is b

    def test_different_resources_different_limiters(self):
        a = GlobalRateLimiter.get("res_a", rate=1.0, capacity=5)
        b = GlobalRateLimiter.get("res_b", rate=2.0, capacity=10)
        assert a is not b

    def test_reset_clears_all(self):
        a = GlobalRateLimiter.get("test", rate=1.0, capacity=5)
        GlobalRateLimiter.reset()
        b = GlobalRateLimiter.get("test", rate=1.0, capacity=5)
        assert a is not b


class TestPreConfiguredLimiters:
    def test_civitai_limiter(self):
        limiter = CIVITAI_LIMITER()
        assert isinstance(limiter, RateLimiter)
        assert limiter.acquire() is True

    def test_huggingface_limiter(self):
        limiter = HUGGINGFACE_LIMITER()
        assert isinstance(limiter, RateLimiter)
        assert limiter.acquire() is True

    def test_vision_limiter(self):
        limiter = VISION_LIMITER()
        assert isinstance(limiter, RateLimiter)
        assert limiter.acquire() is True

    def test_same_resource_returns_same_instance(self):
        a = CIVITAI_LIMITER()
        b = CIVITAI_LIMITER()
        assert a is b
