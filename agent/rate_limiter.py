"""Thread-safe token bucket rate limiter with global singleton registry.

Prevents hammering external APIs (CivitAI, HuggingFace, Claude Vision)
by enforcing configurable rate limits. No external dependencies.

Usage:
    from .rate_limiter import CIVITAI_LIMITER
    if not CIVITAI_LIMITER().acquire(timeout=5.0):
        return error("Rate limited — try again shortly.")
"""

import threading
import time


class RateLimiter:
    """Token bucket rate limiter — thread-safe, blocking or non-blocking.

    Args:
        rate: Tokens refilled per second.
        capacity: Maximum burst size (bucket capacity).
    """

    def __init__(self, rate: float, capacity: int):
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Acquire tokens. Blocks up to `timeout` seconds if needed.

        Args:
            tokens: Number of tokens to consume (default 1).
            timeout: Max seconds to wait. None = non-blocking (return immediately).

        Returns:
            True if tokens were acquired, False if timed out or unavailable.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            # Non-blocking mode — fail immediately
            if deadline is None:
                return False

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                return False

            # Sleep briefly and retry
            wait = min(tokens / self._rate, deadline - now, 0.1)
            time.sleep(wait)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called under lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available(self) -> float:
        """Current token count (approximate — may change between read and use)."""
        with self._lock:
            self._refill()
            return self._tokens


class GlobalRateLimiter:
    """Singleton registry of named rate limiters.

    Ensures only one limiter per resource name exists across the process.
    """

    _instances: dict[str, RateLimiter] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, resource: str, rate: float, capacity: int) -> RateLimiter:
        """Get or create a rate limiter for the named resource."""
        with cls._lock:
            if resource not in cls._instances:
                cls._instances[resource] = RateLimiter(rate, capacity)
            return cls._instances[resource]

    @classmethod
    def reset(cls) -> None:
        """Clear all limiters. Use in tests for isolation."""
        with cls._lock:
            cls._instances.clear()


# ---------------------------------------------------------------------------
# Pre-configured limiters for each external API
# ---------------------------------------------------------------------------

def CIVITAI_LIMITER() -> RateLimiter:
    """CivitAI API: 1 req/s sustained, burst of 5."""
    return GlobalRateLimiter.get("civitai", rate=1.0, capacity=5)


def HUGGINGFACE_LIMITER() -> RateLimiter:
    """HuggingFace API: 2 req/s sustained, burst of 10."""
    return GlobalRateLimiter.get("huggingface", rate=2.0, capacity=10)


def VISION_LIMITER() -> RateLimiter:
    """Claude Vision API: 0.5 req/s sustained, burst of 2."""
    return GlobalRateLimiter.get("vision", rate=0.5, capacity=2)
