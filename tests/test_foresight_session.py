"""Tests for FORESIGHT session context wiring + experience persistence."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.session_context import SessionContext, SessionRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return SessionContext(session_id="foresight_test")


@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


# ---------------------------------------------------------------------------
# SessionContext FORESIGHT properties
# ---------------------------------------------------------------------------

class TestSessionContextForesight:
    def test_cwm_starts_none(self, ctx):
        assert ctx.cwm is None

    def test_arbiter_starts_none(self, ctx):
        assert ctx.arbiter is None

    def test_workflow_signature_starts_none(self, ctx):
        assert ctx.workflow_signature is None

    def test_ensure_cwm_returns_predict(self, ctx):
        result = ctx.ensure_cwm()
        # Should import cwm.predict successfully
        from agent.stage.cwm import predict
        assert result is predict

    def test_ensure_cwm_idempotent(self, ctx):
        a = ctx.ensure_cwm()
        b = ctx.ensure_cwm()
        assert a is b

    def test_ensure_arbiter_returns_arbiter(self, ctx):
        result = ctx.ensure_arbiter()
        from agent.stage.arbiter import Arbiter
        assert isinstance(result, Arbiter)

    def test_ensure_arbiter_idempotent(self, ctx):
        a = ctx.ensure_arbiter()
        b = ctx.ensure_arbiter()
        assert a is b

    def test_workflow_signature_settable(self, ctx):
        from agent.stage.workflow_signature import WorkflowSignature
        sig = WorkflowSignature(model_family="sdxl")
        ctx.workflow_signature = sig
        assert ctx.workflow_signature is sig


class TestSessionContextRatchetForesightWiring:
    """ensure_ratchet() wires FORESIGHT into the Ratchet."""

    def test_ratchet_has_foresight(self, ctx):
        pytest.importorskip("pxr", reason="usd-core not installed")
        r = ctx.ensure_ratchet()
        assert r is not None
        assert r.has_foresight is True

    def test_ratchet_without_usd_returns_none(self, ctx):
        with patch("agent.session_context.SessionContext.ensure_stage",
                    return_value=None):
            ctx._ratchet = None  # reset
            r = ctx.ensure_ratchet()
            assert r is None

    def test_ratchet_idempotent(self, ctx):
        pytest.importorskip("pxr", reason="usd-core not installed")
        a = ctx.ensure_ratchet()
        b = ctx.ensure_ratchet()
        assert a is b


# ---------------------------------------------------------------------------
# Experience persistence (save/load)
# ---------------------------------------------------------------------------

class TestExperiencePersistence:
    def test_save_experience(self, usd_stage, tmp_path):
        from agent.stage.experience import record_experience

        record_experience(
            usd_stage,
            initial_state={"steps": 20},
            decisions=[],
            outcome={"aesthetic": 0.8},
            timestamp=1000.0,
        )

        with patch("agent.memory.session._sessions_dir", return_value=tmp_path):
            from agent.memory.session import save_experience
            result = save_experience("test_sess", usd_stage)

        assert "saved_experience" in result
        assert result["count"] == 1

    def test_load_experience(self, usd_stage, tmp_path):
        from agent.stage.experience import record_experience, query_experience

        # Record into source stage
        record_experience(
            usd_stage,
            initial_state={"steps": 20},
            decisions=[],
            outcome={"aesthetic": 0.8},
            timestamp=1000.0,
        )

        # Save
        with patch("agent.memory.session._sessions_dir", return_value=tmp_path):
            from agent.memory.session import save_experience, load_experience
            save_experience("test_sess", usd_stage)

            # Load into fresh stage
            from agent.stage.cognitive_stage import CognitiveWorkflowStage
            fresh = CognitiveWorkflowStage()
            count = load_experience("test_sess", fresh)

        assert count == 1
        exps = query_experience(fresh)
        assert len(exps) == 1

    def test_load_nonexistent(self, tmp_path):
        with patch("agent.memory.session._sessions_dir", return_value=tmp_path):
            from agent.memory.session import load_experience
            count = load_experience("nonexistent", MagicMock())
        assert count == 0

    def test_save_empty_stage(self, usd_stage, tmp_path):
        with patch("agent.memory.session._sessions_dir", return_value=tmp_path):
            from agent.memory.session import save_experience
            result = save_experience("empty", usd_stage)
        assert result["count"] == 0

    def test_roundtrip_preserves_signature_hash(self, usd_stage, tmp_path):
        from agent.stage.experience import record_experience, query_experience

        record_experience(
            usd_stage,
            initial_state={},
            decisions=[],
            outcome={"aesthetic": 0.5},
            context_signature_hash="hash_abc",
            timestamp=2000.0,
        )

        with patch("agent.memory.session._sessions_dir", return_value=tmp_path):
            from agent.memory.session import save_experience, load_experience
            save_experience("sig_test", usd_stage)

            from agent.stage.cognitive_stage import CognitiveWorkflowStage
            fresh = CognitiveWorkflowStage()
            load_experience("sig_test", fresh)

        exps = query_experience(fresh)
        assert exps[0].context_signature_hash == "hash_abc"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryForesight:
    def test_get_or_create_has_foresight_properties(self):
        reg = SessionRegistry()
        ctx = reg.get_or_create("foresight_reg")
        assert ctx.cwm is None  # not yet initialized
        assert ctx.arbiter is None
        reg.clear()
