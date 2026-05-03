"""Tests for agent/stage/moe_dispatcher.py — all mocked, no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.moe_dispatcher import (
    HandoffArtifact,
    MoEDispatcher,
    TASK_CHAINS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dispatcher():
    return MoEDispatcher()


# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

class TestClassifyTask:
    def test_recon(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("discover what nodes are available") == "recon"

    def test_design(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("plan a new workflow") == "design"

    def test_provision(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("download the SDXL model") == "provision"

    def test_build(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("create a new image pipeline") == "build"

    def test_verify(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("execute the workflow") == "verify"

    def test_analyze(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("analyze the output quality") == "analyze"

    def test_default_recon(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("xyzzy foobar") == "recon"

    def test_empty_string(self, dispatcher: MoEDispatcher):
        assert dispatcher.classify_task("") == "recon"


# ---------------------------------------------------------------------------
# Chain selection
# ---------------------------------------------------------------------------

class TestChainSelection:
    def test_recon_chain(self, dispatcher: MoEDispatcher):
        assert dispatcher.get_chain("recon") == ("scout",)

    def test_build_chain(self, dispatcher: MoEDispatcher):
        # Mutating chains end with scribe per Cozy Constitution Article II.
        assert dispatcher.get_chain("build") == (
            "scout", "architect", "forge", "crucible", "scribe",
        )

    def test_analyze_chain(self, dispatcher: MoEDispatcher):
        # Vision records experience prims, so analyze mutates state and
        # must end with scribe.
        assert dispatcher.get_chain("analyze") == ("crucible", "vision", "scribe")

    def test_full_chain(self, dispatcher: MoEDispatcher):
        assert dispatcher.get_full_chain() == (
            "scout", "architect", "provisioner",
            "forge", "crucible", "vision", "scribe",
        )

    def test_all_task_types_have_chains(self):
        expected = {"recon", "design", "provision", "build", "verify", "analyze"}
        assert set(TASK_CHAINS.keys()) == expected


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------

class TestSelectProfiles:
    def test_recon_profiles(self, dispatcher: MoEDispatcher):
        profiles = dispatcher.select_profiles("recon")
        assert len(profiles) == 1
        assert profiles[0].name == "scout"

    def test_build_profiles(self, dispatcher: MoEDispatcher):
        profiles = dispatcher.select_profiles("build")
        assert [p.name for p in profiles] == [
            "scout", "architect", "forge", "crucible", "scribe",
        ]


# ---------------------------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------------------------

class TestToolFiltering:
    def test_scout_filter(self, dispatcher: MoEDispatcher):
        all_tools = ["discover", "execute_workflow", "add_node"]
        filtered = dispatcher.filter_tools_for_agent("scout", all_tools)
        assert filtered == ["discover"]

    def test_unknown_agent_returns_empty(self, dispatcher: MoEDispatcher):
        filtered = dispatcher.filter_tools_for_agent("fake", ["discover"])
        assert filtered == []


# ---------------------------------------------------------------------------
# Chain lifecycle
# ---------------------------------------------------------------------------

class TestChainLifecycle:
    def test_create_chain(self, dispatcher: MoEDispatcher):
        state = dispatcher.create_chain("build")
        assert state.task_type == "build"
        assert state.status == "pending"
        assert state.current_agent == "scout"

    def test_start_chain(self, dispatcher: MoEDispatcher):
        state = dispatcher.create_chain("recon")
        dispatcher.start_chain(state)
        assert state.status == "running"

    def test_advance_chain(self, dispatcher: MoEDispatcher):
        state = dispatcher.create_chain("build")
        dispatcher.start_chain(state)
        assert state.current_agent == "scout"

        artifact = HandoffArtifact(
            artifact_type="recon_report",
            source_agent="scout",
            data={"nodes": 42},
        )
        dispatcher.advance_chain(state, artifact)
        assert state.current_agent == "architect"
        assert len(state.artifacts) == 1

    def test_chain_completes(self, dispatcher: MoEDispatcher):
        state = dispatcher.create_chain("recon")  # 1-step chain
        dispatcher.start_chain(state)
        dispatcher.advance_chain(state)
        assert state.is_complete
        assert state.status == "completed"

    def test_chain_to_dict(self, dispatcher: MoEDispatcher):
        state = dispatcher.create_chain("recon")
        d = state.to_dict()
        assert "task_type" in d
        assert "chain" in d
        assert "status" in d


# ---------------------------------------------------------------------------
# Retry / bounded failure
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retry_within_bounds(self, dispatcher: MoEDispatcher):
        state = dispatcher.create_chain("build")
        assert dispatcher.record_retry(state, "scout") is True
        assert dispatcher.record_retry(state, "scout") is True

    def test_retry_exceeds_bounds(self, dispatcher: MoEDispatcher):
        state = dispatcher.create_chain("build")
        dispatcher.record_retry(state, "scout")
        dispatcher.record_retry(state, "scout")
        result = dispatcher.record_retry(state, "scout")
        assert result is False
        assert state.status == "blocked"
        assert "scout" in state.blocker_reason


# ---------------------------------------------------------------------------
# Constitutional checks
# ---------------------------------------------------------------------------

class TestConstitutionalChecks:
    def test_validate_handoff(self, dispatcher: MoEDispatcher):
        artifact = HandoffArtifact(
            artifact_type="recon_report",
            source_agent="scout",
            data={},
        )
        assert dispatcher.validate_handoff(artifact, "recon_report") is True
        assert dispatcher.validate_handoff(artifact, "design_spec") is False

    def test_role_isolation(self, dispatcher: MoEDispatcher):
        assert dispatcher.check_role_isolation("scout", "discover") is True
        assert dispatcher.check_role_isolation("scout", "execute_workflow") is False

    def test_adversarial_verification(self, dispatcher: MoEDispatcher):
        assert dispatcher.check_adversarial_verification("forge", "crucible") is True
        assert dispatcher.check_adversarial_verification("forge", "forge") is False


# ---------------------------------------------------------------------------
# Full dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_dispatch_returns_running_state(self, dispatcher: MoEDispatcher):
        state = dispatcher.dispatch("discover what's available")
        assert state.status == "running"
        assert state.task_type == "recon"

    def test_dispatch_build(self, dispatcher: MoEDispatcher):
        state = dispatcher.dispatch("create a new workflow")
        assert state.task_type == "build"
        assert state.current_agent == "scout"


# ---------------------------------------------------------------------------
# HandoffArtifact
# ---------------------------------------------------------------------------

class TestHandoffArtifact:
    def test_to_dict(self):
        a = HandoffArtifact(
            artifact_type="recon_report",
            source_agent="scout",
            data={"nodes": 10},
            timestamp=1000.0,
        )
        d = a.to_dict()
        assert d["artifact_type"] == "recon_report"
        assert d["source_agent"] == "scout"
        assert d["data"] == {"nodes": 10}
        assert d["timestamp"] == 1000.0
