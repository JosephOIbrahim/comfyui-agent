"""Circuit breaker pattern for external service calls.

Prevents cascading failures when ComfyUI or other services are down.
Three states: CLOSED (normal), OPEN (fast-fail), HALF_OPEN (test one request).

Thread-safe. Used by comfy_api.py to protect against repeated connection failures.
"""

import logging
import threading
import time

log = logging.getLogger(__name__)

# States
CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Args:
        name: Identifier for logging.
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout: Seconds to wait in OPEN state before trying HALF_OPEN.
        half_open_max: Max concurrent requests allowed in HALF_OPEN state.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_count = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns True if request can proceed, False if circuit is open.
        """
        with self._lock:
            if self._state == CLOSED:
                return True

            if self._state == OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = HALF_OPEN
                    self._half_open_count = 0
                    log.info("Circuit breaker '%s': OPEN -> HALF_OPEN (testing)", self.name)
                    self._half_open_count += 1
                    return True
                return False

            if self._state == HALF_OPEN:
                if self._half_open_count < self.half_open_max:
                    self._half_open_count += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record a successful request. Resets failure count, closes circuit."""
        with self._lock:
            if self._state != CLOSED:
                log.info("Circuit breaker '%s': %s -> CLOSED (success)", self.name, self._state)
            self._state = CLOSED
            self._failure_count = 0
            self._half_open_count = 0

    def record_failure(self) -> None:
        """Record a failed request. May open the circuit."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == HALF_OPEN:
                self._state = OPEN
                log.warning(
                    "Circuit breaker '%s': HALF_OPEN -> OPEN (test request failed)", self.name
                )
            elif self._failure_count >= self.failure_threshold:
                if self._state != OPEN:
                    log.warning(
                        "Circuit breaker '%s': CLOSED -> OPEN (%d consecutive failures)",
                        self.name,
                        self._failure_count,
                    )
                self._state = OPEN

    def reset(self) -> None:
        """Reset to initial state. For testing."""
        with self._lock:
            self._state = CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._half_open_count = 0


# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------

_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(
    name: str,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """Get or create a named circuit breaker (singleton per name)."""
    with _registry_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _breakers[name]


def reset_all() -> None:
    """Reset all circuit breakers. For testing."""
    with _registry_lock:
        _breakers.clear()


# Pre-configured breakers
def COMFYUI_BREAKER() -> CircuitBreaker:
    """Circuit breaker for ComfyUI HTTP API calls."""
    return get_breaker("comfyui", failure_threshold=3, recovery_timeout=30.0)
