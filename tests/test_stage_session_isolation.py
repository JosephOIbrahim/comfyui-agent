"""Parity tests for cycle 12 — session isolation across 4 stage modules.

Verifies that foresight_tools, provision_tools, compositor_tools, and
hyperagent_tools all read the _conn_session ContextVar so each MCP
connection / sidebar conversation gets its own isolated per-session state
instead of sharing a single process-wide "default" slot.
"""

from __future__ import annotations

import contextvars as _cv
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Shared helpers (mirror the TestStageGateInteraction pattern from
# tests/test_conn_ctx.py)
# ---------------------------------------------------------------------------


def _run_in_session(session_id: str, fn):
    """Run fn() with _conn_session set to session_id in a copied context."""
    from agent._conn_ctx import _conn_session

    copied = _cv.copy_context()

    def _runner():
        _conn_session.set(session_id)
        return fn()

    return copied.run(_runner)


_TEST_SIDS = [
    "fs_session_a",
    "fs_session_b",
    "prov_a",
    "prov_b",
    "comp_a",
    "comp_b",
    "hyper_a",
    "hyper_b",
]


def _clean_all_test_sessions():
    """Wipe WorkflowSession, SessionContext, and stage-module per-session caches."""
    from agent.session_context import _registry as _ctx_registry
    from agent.workflow_session import _registry_lock, _sessions

    with _registry_lock:
        for sid in _TEST_SIDS:
            _sessions.pop(sid, None)
    for sid in _TEST_SIDS:
        _ctx_registry.destroy(sid)

    # Stage-module per-session caches.
    try:
        from agent.stage import provision_tools as _pt

        with _pt._prov_lock:
            for sid in _TEST_SIDS:
                _pt._provisioners.pop(sid, None)
            _pt._provisioners.pop("default", None)
    except Exception:
        pass

    try:
        from agent.stage import compositor_tools as _ct

        with _ct._scenes_lock:
            for sid in _TEST_SIDS:
                _ct._scenes.pop(sid, None)
    except Exception:
        pass

    try:
        from agent.stage import hyperagent_tools as _ht

        with _ht._meta_agents_lock:
            for sid in _TEST_SIDS:
                _ht._meta_agents.pop(sid, None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Foresight
# ---------------------------------------------------------------------------


class TestForesightSessionIsolation:
    def setup_method(self):
        _clean_all_test_sessions()

    def teardown_method(self):
        _clean_all_test_sessions()

    def test_get_stage_reads_contextvar(self):
        """_get_stage() must route through _conn_session so two sessions get
        distinct SessionContext objects (not a shared "default")."""
        from agent.session_context import get_session_context
        from agent.stage import foresight_tools as ft

        # We compare via _get_ctx (same code path) because _get_stage may
        # return None when usd-core is unavailable — but the contextvar
        # plumbing lives in _get_ctx / get_session_context.
        def _get_a():
            return ft._get_ctx()

        def _get_b():
            return ft._get_ctx()

        ctx_a = _run_in_session("fs_session_a", _get_a)
        ctx_b = _run_in_session("fs_session_b", _get_b)

        # Neither may be the "default" context — they must be the ones
        # registered under the per-session ids.
        expected_a = get_session_context("fs_session_a")
        expected_b = get_session_context("fs_session_b")

        assert ctx_a is expected_a
        assert ctx_b is expected_b
        assert id(ctx_a) != id(ctx_b)

        # And just to be sure, they aren't the "default" SessionContext.
        default_ctx = get_session_context("default")
        assert id(ctx_a) != id(default_ctx)
        assert id(ctx_b) != id(default_ctx)

    def test_get_stage_default_when_unset(self):
        """Without a contextvar set (CLI / bare pytest), _get_stage falls
        back to the "default" session via current_conn_session()."""
        from agent._conn_ctx import current_conn_session
        from agent.session_context import get_session_context
        from agent.stage import foresight_tools as ft

        # Sanity: bare current_conn_session() returns "default" when unset.
        assert current_conn_session() == "default"

        ctx = ft._get_ctx()
        assert ctx is get_session_context("default")


# ---------------------------------------------------------------------------
# Provision
# ---------------------------------------------------------------------------


class TestProvisionSessionIsolation:
    def setup_method(self):
        _clean_all_test_sessions()

    def teardown_method(self):
        _clean_all_test_sessions()

    def test_get_provisioner_reads_contextvar(self):
        """_get_provisioner() must cache per-session, keyed off _conn_session.

        Provisioner may fail to construct (USD/usd-core missing) and the
        function returns None — but the per-session cache behavior is still
        observable via the _provisioners dict: either both sessions register
        distinct entries (correct) or neither does (both None fall-throughs).
        The important negative assertion is that NEITHER call populates a
        "default" entry when a real contextvar value is set.
        """
        from agent.stage import provision_tools as pt

        def _call():
            return pt._get_provisioner()

        prov_a = _run_in_session("prov_a", _call)
        prov_b = _run_in_session("prov_b", _call)

        # Contract: "default" key must NOT be populated by either call,
        # because both calls ran with a real contextvar set.
        with pt._prov_lock:
            # If construction succeeded for either session, it must be
            # cached under that specific session_id (not "default").
            if prov_a is not None:
                assert "prov_a" in pt._provisioners
                assert pt._provisioners["prov_a"] is prov_a
            if prov_b is not None:
                assert "prov_b" in pt._provisioners
                assert pt._provisioners["prov_b"] is prov_b

            # If both constructed successfully, they must be DIFFERENT
            # instances (isolation).
            if prov_a is not None and prov_b is not None:
                assert prov_a is not prov_b

            # Negative: neither call should have written to "default".
            assert "default" not in pt._provisioners or (
                prov_a is None and prov_b is None
            )

    def test_provision_status_handler_uses_contextvar(self):
        """_handle_provision_status degraded path must call
        get_session_context(<contextvar value>), not the literal "default".

        This guards the line-200 fix: before cycle 12 the handler used a
        hardcoded "default" sid, which broke per-session isolation.
        """
        from agent.stage import provision_tools as pt

        # Force the degraded path: make _get_provisioner return None so
        # the handler falls into the get_session_context branch.
        captured_sids: list[str] = []

        def _fake_get_ctx(sid):
            captured_sids.append(sid)

            class _Ctx:
                stage = None

            return _Ctx()

        with patch.object(pt, "_get_provisioner", return_value=None), patch(
            "agent.session_context.get_session_context", side_effect=_fake_get_ctx
        ):
            _run_in_session(
                "prov_a",
                lambda: pt._handle_provision_status({"prim_path": "/Test/foo"}),
            )

        assert captured_sids, "get_session_context was never called"
        # Must have used the contextvar value, NOT the literal "default".
        assert "prov_a" in captured_sids
        assert "default" not in captured_sids


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------


class TestCompositorSessionIsolation:
    def setup_method(self):
        _clean_all_test_sessions()

    def teardown_method(self):
        _clean_all_test_sessions()

    def test_set_get_scene_isolated_per_session(self):
        """_set_scene / _get_scene must be keyed off _conn_session so two
        sessions can't see each other's scenes."""
        from agent.stage import compositor_tools as ct

        scene_a = {"name": "scene_a_sentinel"}
        scene_b = {"name": "scene_b_sentinel"}

        # Session A: set scene_a, verify retrievable.
        def _set_a():
            ct._set_scene(scene_a)
            return ct._get_scene()

        got_a = _run_in_session("comp_a", _set_a)
        assert got_a is scene_a

        # Session B: fresh session, _get_scene must return None
        # (no cross-contamination from A).
        def _get_empty_b():
            return ct._get_scene()

        got_b_empty = _run_in_session("comp_b", _get_empty_b)
        assert got_b_empty is None, f"Expected None, got: {got_b_empty!r}"

        # Session B: set scene_b, verify retrievable.
        def _set_b():
            ct._set_scene(scene_b)
            return ct._get_scene()

        got_b = _run_in_session("comp_b", _set_b)
        assert got_b is scene_b

        # Back to session A: scene_a must still be there (not overwritten).
        def _get_a():
            return ct._get_scene()

        still_a = _run_in_session("comp_a", _get_a)
        assert still_a is scene_a, (
            f"Session A scene was overwritten! Got: {still_a!r}"
        )

        # And session B still has its own.
        still_b = _run_in_session("comp_b", _get_a)
        assert still_b is scene_b


# ---------------------------------------------------------------------------
# Hyperagent
# ---------------------------------------------------------------------------


class TestHyperagentSessionIsolation:
    def setup_method(self):
        _clean_all_test_sessions()

    def teardown_method(self):
        _clean_all_test_sessions()

    def test_get_meta_agent_isolated_per_session(self):
        """_get_meta_agent must return a distinct MetaAgent per session,
        cached per session_id, with no cross-contamination."""
        try:
            from agent.stage import hyperagent_tools as ht
        except Exception as e:  # pragma: no cover
            import pytest

            pytest.skip(f"hyperagent_tools unavailable: {e}")

        def _call():
            return ht._get_meta_agent()

        try:
            ma_a1 = _run_in_session("hyper_a", _call)
            ma_b = _run_in_session("hyper_b", _call)
            ma_a2 = _run_in_session("hyper_a", _call)
        except Exception as e:  # pragma: no cover
            import pytest

            pytest.skip(f"MetaAgent construction failed: {e}")

        # Two different sessions → two different instances.
        assert id(ma_a1) != id(ma_b), (
            "hyper_a and hyper_b returned the same MetaAgent — "
            "session isolation broken"
        )

        # Same session called twice → same cached instance.
        assert ma_a1 is ma_a2, (
            "Second call inside hyper_a returned a different MetaAgent — "
            "per-session caching broken"
        )

        # And neither should be the "default" slot's agent.
        assert "hyper_a" in ht._meta_agents
        assert "hyper_b" in ht._meta_agents
        assert ht._meta_agents["hyper_a"] is ma_a1
        assert ht._meta_agents["hyper_b"] is ma_b
