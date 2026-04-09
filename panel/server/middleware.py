"""Panel route guards — auth, rate limiting, request size."""

import logging
import time

from aiohttp import web

from agent.config import MCP_AUTH_TOKEN

log = logging.getLogger("comfy-cozy")

# Rate limit categories (tokens/sec, burst capacity)
_RATE_CONFIGS = {
    "execute": (1.0, 3),  # 1 execution/sec, burst of 3
    "discover": (5.0, 15),  # 5 searches/sec
    "download": (0.5, 2),  # 1 download every 2 sec
    "mutation": (10.0, 30),  # 10 modifications/sec
    "read": (50.0, 100),  # 50 reads/sec
    "chat": (5.0, 10),  # 5 messages/sec
}

_MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10 MB

# Simple token bucket per category
_buckets: dict[str, "_TokenBucket"] = {}


class _TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.monotonic()

    def acquire(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def _get_bucket(category: str) -> _TokenBucket:
    if category not in _buckets:
        rate, cap = _RATE_CONFIGS.get(category, (50.0, 100))
        _buckets[category] = _TokenBucket(rate, cap)
    return _buckets[category]


def check_auth(request: web.Request) -> web.Response | None:
    """Check Bearer token if MCP_AUTH_TOKEN is configured. Returns 401 response or None."""
    if not MCP_AUTH_TOKEN:
        return None  # Auth not configured — passthrough
    # Skip auth for health check
    if request.path.endswith("/health"):
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != MCP_AUTH_TOKEN:
        return web.json_response({"error": "Unauthorized"}, status=401)
    return None


def check_rate_limit(category: str) -> web.Response | None:
    """Check rate limit for category. Returns 429 response or None."""
    bucket = _get_bucket(category)
    if not bucket.acquire():
        return web.json_response(
            {"error": "Rate limited", "retry_after": 1.0 / bucket.rate},
            status=429,
            headers={"Retry-After": str(int(1.0 / bucket.rate))},
        )
    return None


def check_size(request: web.Request) -> web.Response | None:
    """Check request body size. Returns 413 response or None."""
    if request.content_length and request.content_length > _MAX_REQUEST_BYTES:
        return web.json_response({"error": "Payload too large"}, status=413)
    return None


def reset_buckets() -> None:
    """Clear all token buckets. Use in tests for isolation."""
    _buckets.clear()
