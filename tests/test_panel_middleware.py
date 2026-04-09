"""Tests for panel route guards — auth, rate limiting, request size."""

from unittest.mock import MagicMock

import pytest

from panel.server.middleware import (
    _TokenBucket,
    check_auth,
    check_rate_limit,
    check_size,
    reset_buckets,
)


def _make_request(
    path="/comfy-cozy/graph-state",
    auth_header=None,
    content_length=None,
):
    """Build a minimal mock request matching what the guards inspect."""
    req = MagicMock()
    req.path = path
    req.headers = {}
    if auth_header is not None:
        req.headers["Authorization"] = auth_header
    req.content_length = content_length
    return req


@pytest.fixture(autouse=True)
def _clean_buckets():
    """Reset token buckets between tests so state doesn't leak."""
    reset_buckets()
    yield
    reset_buckets()


# ── check_auth ─────────────────────────────────────────────────


class TestCheckAuth:
    def test_passthrough_when_no_token_configured(self, monkeypatch):
        monkeypatch.setattr("panel.server.middleware.MCP_AUTH_TOKEN", None)
        req = _make_request()
        assert check_auth(req) is None

    def test_passthrough_when_token_empty_string(self, monkeypatch):
        monkeypatch.setattr("panel.server.middleware.MCP_AUTH_TOKEN", "")
        req = _make_request()
        assert check_auth(req) is None

    def test_rejects_missing_header(self, monkeypatch):
        monkeypatch.setattr("panel.server.middleware.MCP_AUTH_TOKEN", "secret-abc")
        req = _make_request()
        resp = check_auth(req)
        assert resp is not None
        assert resp.status == 401

    def test_rejects_wrong_token(self, monkeypatch):
        monkeypatch.setattr("panel.server.middleware.MCP_AUTH_TOKEN", "secret-abc")
        req = _make_request(auth_header="Bearer wrong-token")
        resp = check_auth(req)
        assert resp is not None
        assert resp.status == 401

    def test_accepts_correct_token(self, monkeypatch):
        monkeypatch.setattr("panel.server.middleware.MCP_AUTH_TOKEN", "secret-abc")
        req = _make_request(auth_header="Bearer secret-abc")
        assert check_auth(req) is None

    def test_health_endpoint_skips_auth(self, monkeypatch):
        monkeypatch.setattr("panel.server.middleware.MCP_AUTH_TOKEN", "secret-abc")
        req = _make_request(path="/comfy-cozy/health")
        assert check_auth(req) is None

    def test_rejects_non_bearer_scheme(self, monkeypatch):
        monkeypatch.setattr("panel.server.middleware.MCP_AUTH_TOKEN", "secret-abc")
        req = _make_request(auth_header="Basic secret-abc")
        resp = check_auth(req)
        assert resp is not None
        assert resp.status == 401


# ── check_rate_limit ───────────────────────────────────────────


class TestCheckRateLimit:
    def test_allows_up_to_capacity(self):
        # "execute" has capacity 3
        for _ in range(3):
            assert check_rate_limit("execute") is None

    def test_rejects_over_capacity(self):
        for _ in range(3):
            check_rate_limit("execute")
        resp = check_rate_limit("execute")
        assert resp is not None
        assert resp.status == 429

    def test_different_categories_independent(self):
        # Exhaust execute
        for _ in range(3):
            check_rate_limit("execute")
        # discover should still work
        assert check_rate_limit("discover") is None

    def test_unknown_category_uses_default(self):
        # Unknown category gets default (50.0, 100)
        assert check_rate_limit("unknown_category_xyz") is None

    def test_429_includes_retry_after(self):
        for _ in range(3):
            check_rate_limit("execute")
        resp = check_rate_limit("execute")
        assert resp.headers["Retry-After"] == "1"


# ── check_size ─────────────────────────────────────────────────


class TestCheckSize:
    def test_allows_small_request(self):
        req = _make_request(content_length=1024)
        assert check_size(req) is None

    def test_allows_no_content_length(self):
        req = _make_request(content_length=None)
        assert check_size(req) is None

    def test_rejects_oversized(self):
        req = _make_request(content_length=20 * 1024 * 1024)
        resp = check_size(req)
        assert resp is not None
        assert resp.status == 413

    def test_allows_exactly_at_limit(self):
        req = _make_request(content_length=10 * 1024 * 1024)
        assert check_size(req) is None


# ── _TokenBucket internals ─────────────────────────────────────


class TestTokenBucket:
    def test_acquire_returns_true_when_tokens_available(self):
        b = _TokenBucket(1.0, 5)
        assert b.acquire() is True

    def test_acquire_returns_false_when_empty(self):
        b = _TokenBucket(1.0, 2)
        assert b.acquire() is True
        assert b.acquire() is True
        assert b.acquire() is False

    def test_refill_over_time(self):
        import time

        b = _TokenBucket(100.0, 5)  # 100 tokens/sec so refill is fast
        b.acquire()
        b.acquire()
        b.acquire()
        b.acquire()
        b.acquire()
        assert b.acquire() is False
        time.sleep(0.06)  # ~6 tokens refilled at 100/s
        assert b.acquire() is True
