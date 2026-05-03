"""Tests for agent/stage/moe_profiles.py — all mocked, no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.moe_profiles import (
    ALL_PROFILES,
    ARCHITECT,
    CRUCIBLE,
    DEFAULT_CHAIN,
    FORGE,
    PROFILE_NAMES,
    PROVISIONER,
    SCOUT,
    VISION,
    filter_tools,
    get_allowed_tools,
    get_profile,
)


# ---------------------------------------------------------------------------
# Profile basics
# ---------------------------------------------------------------------------

class TestAgentProfile:
    def test_profile_is_frozen(self):
        with pytest.raises(AttributeError):
            SCOUT.name = "hacked"  # type: ignore[misc]

    def test_to_dict_keys(self):
        d = SCOUT.to_dict()
        assert set(d.keys()) == {
            "name", "system_prompt_fragment", "allowed_tools",
            "authority_rules", "handoff_artifact_type",
        }

    def test_can_use_tool_allowed(self):
        assert SCOUT.can_use_tool("discover") is True

    def test_can_use_tool_denied(self):
        assert SCOUT.can_use_tool("execute_workflow") is False

    def test_owns(self):
        assert SCOUT.owns("environment_discovery") is True
        assert SCOUT.owns("workflow_mutation") is False

    def test_is_forbidden(self):
        assert SCOUT.is_forbidden("modify_workflow") is True
        assert SCOUT.is_forbidden("environment_discovery") is False


# ---------------------------------------------------------------------------
# All 7 profiles exist (6 originals + scribe added per Cozy Constitution Article II)
# ---------------------------------------------------------------------------

class TestAllProfiles:
    EXPECTED_NAMES = {
        "scout", "architect", "provisioner", "forge", "crucible", "vision",
        "scribe",
    }

    def test_seven_profiles(self):
        assert len(ALL_PROFILES) == 7

    def test_profile_names(self):
        assert set(ALL_PROFILES.keys()) == self.EXPECTED_NAMES

    def test_profile_names_tuple(self):
        assert set(PROFILE_NAMES) == self.EXPECTED_NAMES

    def test_default_chain_order(self):
        # Scribe terminates the default chain — Article II of the Cozy Constitution
        # mandates that every state-mutating chain ends with a flush.
        assert DEFAULT_CHAIN == (
            "scout", "architect", "provisioner", "forge", "crucible", "vision",
            "scribe",
        )

    def test_each_profile_has_allowed_tools(self):
        for name, profile in ALL_PROFILES.items():
            assert len(profile.allowed_tools) > 0, f"{name} has no tools"

    def test_each_profile_has_authority_rules(self):
        for name, profile in ALL_PROFILES.items():
            assert "owns" in profile.authority_rules, f"{name} missing 'owns'"
            assert "cannot" in profile.authority_rules, f"{name} missing 'cannot'"

    def test_each_profile_has_handoff_type(self):
        expected_types = {
            "scout": "recon_report",
            "architect": "design_spec",
            "provisioner": "provision_manifest",
            "forge": "build_artifact",
            "crucible": "execution_result",
            "vision": "quality_report",
            "scribe": "persistence_receipt",
        }
        for name, profile in ALL_PROFILES.items():
            assert profile.handoff_artifact_type == expected_types[name]

    def test_each_profile_has_system_prompt(self):
        for name, profile in ALL_PROFILES.items():
            assert len(profile.system_prompt_fragment) > 20, f"{name} prompt too short"


# ---------------------------------------------------------------------------
# Authority isolation
# ---------------------------------------------------------------------------

class TestAuthorityIsolation:
    def test_scout_cannot_modify(self):
        assert SCOUT.is_forbidden("modify_workflow")

    def test_architect_cannot_execute(self):
        assert ARCHITECT.is_forbidden("execute_workflow")

    def test_provisioner_cannot_modify(self):
        assert PROVISIONER.is_forbidden("modify_workflow")

    def test_forge_cannot_execute(self):
        assert FORGE.is_forbidden("execute_workflow")

    def test_crucible_cannot_modify(self):
        assert CRUCIBLE.is_forbidden("modify_workflow")

    def test_vision_cannot_execute(self):
        assert VISION.is_forbidden("execute_workflow")

    def test_no_profile_owns_everything(self):
        for name, profile in ALL_PROFILES.items():
            assert len(profile.authority_rules["cannot"]) > 0, (
                f"{name} has no restrictions"
            )


# ---------------------------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------------------------

class TestToolFiltering:
    def test_scout_gets_discover(self):
        result = filter_tools("scout", ["discover", "execute_workflow", "add_node"])
        assert result == ["discover"]

    def test_forge_gets_add_node(self):
        result = filter_tools("forge", ["discover", "add_node", "set_input"])
        assert result == ["add_node", "set_input"]

    def test_unknown_profile_returns_empty(self):
        result = filter_tools("nonexistent", ["discover"])
        assert result == []


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def test_get_profile_found(self):
        assert get_profile("scout") is SCOUT

    def test_get_profile_not_found(self):
        assert get_profile("nonexistent") is None

    def test_get_allowed_tools_found(self):
        tools = get_allowed_tools("scout")
        assert "discover" in tools

    def test_get_allowed_tools_not_found(self):
        assert get_allowed_tools("nonexistent") == ()
