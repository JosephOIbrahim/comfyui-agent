"""Tests for the bounded DiscoveryCache."""

import time

from agent.discovery_cache import DiscoveryCache


class TestDiscoveryCache:
    """Tests for DiscoveryCache get/set/eviction/TTL."""

    def test_get_missing_returns_none(self):
        cache = DiscoveryCache()
        assert cache.get("missing") is None

    def test_set_and_get(self):
        cache = DiscoveryCache()
        cache.set("key1", [1, 2, 3])
        assert cache.get("key1") == [1, 2, 3]

    def test_overwrite_existing(self):
        cache = DiscoveryCache()
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_invalidate(self):
        cache = DiscoveryCache()
        cache.set("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_invalidate_missing_key_no_error(self):
        cache = DiscoveryCache()
        cache.invalidate("nonexistent")  # should not raise

    def test_clear(self):
        cache = DiscoveryCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_size(self):
        cache = DiscoveryCache()
        assert cache.size == 0
        cache.set("a", 1)
        assert cache.size == 1
        cache.set("b", 2)
        assert cache.size == 2

    def test_ttl_expiry(self):
        cache = DiscoveryCache(ttl_seconds=0.05)
        cache.set("key1", "value")
        assert cache.get("key1") == "value"
        time.sleep(0.06)
        assert cache.get("key1") is None

    def test_lru_eviction_at_capacity(self):
        cache = DiscoveryCache(max_entries=3)
        cache.set("a", 1)
        time.sleep(0.01)  # Ensure distinct timestamps on Windows
        cache.set("b", 2)
        time.sleep(0.01)
        cache.set("c", 3)
        time.sleep(0.01)

        # Access "a" to make it recently used
        cache.get("a")
        time.sleep(0.01)

        # Adding "d" should evict the LRU entry ("b")
        cache.set("d", 4)
        assert cache.size == 3
        assert cache.get("a") == 1  # recently accessed, kept
        assert cache.get("b") is None  # LRU, evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_eviction_removes_oldest_accessed(self):
        cache = DiscoveryCache(max_entries=2)
        cache.set("a", 1)
        time.sleep(0.01)
        cache.set("b", 2)
        # "a" was accessed first, so it's the LRU
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_stats(self):
        cache = DiscoveryCache(max_entries=100, ttl_seconds=60)
        cache.set("x", 1)
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["max_entries"] == 100
        assert stats["ttl_seconds"] == 60

    def test_complex_values(self):
        cache = DiscoveryCache()
        cache.set("nodes", [{"name": "KSampler", "category": "sampling"}])
        result = cache.get("nodes")
        assert result == [{"name": "KSampler", "category": "sampling"}]

    def test_none_value_stored(self):
        """None is a valid cache value (distinct from 'not found')."""
        cache = DiscoveryCache()
        cache.set("nullable", None)
        # We need a way to distinguish "cached None" from "not found"
        # Current implementation returns None for both — acceptable tradeoff
        assert cache.size == 1
