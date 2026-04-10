

# ---------------------------------------------------------------------------
# Cycle 43 — _install_locks FIFO eviction cap
# ---------------------------------------------------------------------------

class TestInstallLocksCap:
    """_install_locks must be capped at _MAX_INSTALL_LOCKS entries."""

    def test_locks_capped_at_max(self):
        """After _MAX_INSTALL_LOCKS+1 unique paths, dict never exceeds the cap."""
        from agent.tools.comfy_provision import (
            _get_install_lock, _install_locks, _install_locks_mutex, _MAX_INSTALL_LOCKS,
        )

        # Clear state first
        with _install_locks_mutex:
            _install_locks.clear()

        # Register MAX + 10 unique paths
        for i in range(_MAX_INSTALL_LOCKS + 10):
            _get_install_lock(f"/fake/path/{i}")

        with _install_locks_mutex:
            actual = len(_install_locks)

        assert actual <= _MAX_INSTALL_LOCKS, (
            f"_install_locks grew to {actual}, must stay ≤ {_MAX_INSTALL_LOCKS}"
        )

    def test_max_install_locks_constant_exists(self):
        """_MAX_INSTALL_LOCKS constant must be defined and positive."""
        from agent.tools.comfy_provision import _MAX_INSTALL_LOCKS
        assert isinstance(_MAX_INSTALL_LOCKS, int)
        assert _MAX_INSTALL_LOCKS > 0

    def test_existing_path_returns_same_lock_object(self):
        """Calling _get_install_lock twice for same path returns the same lock."""
        from agent.tools.comfy_provision import _get_install_lock, _install_locks_mutex, _install_locks

        with _install_locks_mutex:
            _install_locks.clear()

        lock_a = _get_install_lock("/same/path")
        lock_b = _get_install_lock("/same/path")
        assert lock_a is lock_b, "Same path must return same lock object"
