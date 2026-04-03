"""CRUCIBLE tests for the pre-dispatch gate (agent/gate/).

Adversarial: tests every risk level, every GateDecision path,
circuit breaker interactions, consent requirements, and scope checks.
"""

from __future__ import annotations

import pytest

from agent.gate import (
    GateDecision,
    GateResult,
    RiskLevel,
    get_risk_level,
    pre_dispatch_check,
)
from agent.gate.risk_levels import TOOL_RISK_LEVELS


# ---------------------------------------------------------------------------
# Risk level classification tests
# ---------------------------------------------------------------------------


class TestRiskLevels:
    def test_read_only_tools(self):
        read_only = [
            "is_comfyui_running", "get_all_nodes", "get_node_info",
            "get_system_stats", "get_queue_status", "get_history",
            "list_custom_nodes", "list_models", "load_workflow",
            "validate_workflow", "discover", "get_civitai_model",
            "validate_before_execute", "comfyui_agent_ping",
        ]
        for tool in read_only:
            assert get_risk_level(tool) == RiskLevel.READ_ONLY, f"{tool} not READ_ONLY"

    def test_reversible_tools(self):
        reversible = [
            "apply_workflow_patch", "add_node", "connect_nodes",
            "set_input", "save_workflow", "undo_workflow_patch",
            "save_session", "add_note", "plan_goal", "capture_intent",
        ]
        for tool in reversible:
            assert get_risk_level(tool) == RiskLevel.REVERSIBLE, f"{tool} not REVERSIBLE"

    def test_execution_tools(self):
        execution = [
            "execute_workflow", "execute_with_progress",
            "analyze_image", "compare_outputs",
        ]
        for tool in execution:
            assert get_risk_level(tool) == RiskLevel.EXECUTION, f"{tool} not EXECUTION"

    def test_provision_tools(self):
        provision = ["install_node_pack", "download_model", "provision_download"]
        for tool in provision:
            assert get_risk_level(tool) == RiskLevel.PROVISION, f"{tool} not PROVISION"

    def test_destructive_tools(self):
        destructive = [
            "uninstall_node_pack", "reset_workflow",
            "stage_rollback", "migrate_deprecated_nodes",
        ]
        for tool in destructive:
            assert get_risk_level(tool) == RiskLevel.DESTRUCTIVE, f"{tool} not DESTRUCTIVE"

    def test_unknown_tool_defaults_reversible(self):
        assert get_risk_level("completely_fake_tool") == RiskLevel.REVERSIBLE

    def test_risk_level_classification_comprehensive(self):
        """Spot-check that 20+ tools have correct risk levels."""
        checks = {
            "get_all_nodes": RiskLevel.READ_ONLY,
            "list_models": RiskLevel.READ_ONLY,
            "get_editable_fields": RiskLevel.READ_ONLY,
            "classify_workflow": RiskLevel.READ_ONLY,
            "get_workflow_diff": RiskLevel.READ_ONLY,
            "check_registry_freshness": RiskLevel.READ_ONLY,
            "get_trending_models": RiskLevel.READ_ONLY,
            "get_learned_patterns": RiskLevel.READ_ONLY,
            "get_plan": RiskLevel.READ_ONLY,
            "get_current_intent": RiskLevel.READ_ONLY,
            "set_input": RiskLevel.REVERSIBLE,
            "connect_nodes": RiskLevel.REVERSIBLE,
            "record_outcome": RiskLevel.REVERSIBLE,
            "start_demo": RiskLevel.REVERSIBLE,
            "apply_optimization": RiskLevel.REVERSIBLE,
            "execute_workflow": RiskLevel.EXECUTION,
            "hash_compare_images": RiskLevel.EXECUTION,
            "run_pipeline": RiskLevel.EXECUTION,
            "install_node_pack": RiskLevel.PROVISION,
            "download_model": RiskLevel.PROVISION,
            "uninstall_node_pack": RiskLevel.DESTRUCTIVE,
            "reset_workflow": RiskLevel.DESTRUCTIVE,
        }
        for tool, expected in checks.items():
            actual = get_risk_level(tool)
            assert actual == expected, f"{tool}: expected {expected.name}, got {actual.name}"


# ---------------------------------------------------------------------------
# Gate decision tests
# ---------------------------------------------------------------------------


class TestGateDecisions:
    def test_read_only_tool_bypasses_gate(self):
        result = pre_dispatch_check("get_all_nodes", {})
        assert result.decision == GateDecision.ALLOW
        assert all(result.checks.values()), "All checks should pass for READ_ONLY"
        assert result.risk_level == RiskLevel.READ_ONLY

    def test_destructive_tool_always_locked(self):
        result = pre_dispatch_check("uninstall_node_pack", {"pack_name": "x"})
        assert result.decision == GateDecision.LOCKED
        assert result.risk_level == RiskLevel.DESTRUCTIVE

    def test_provision_tool_returns_escalate(self):
        result = pre_dispatch_check(
            "install_node_pack",
            {"pack_url": "https://example.com"},
            session_active=True,
        )
        assert result.decision == GateDecision.ESCALATE
        assert result.risk_level == RiskLevel.PROVISION

    def test_mutation_tool_all_checks_pass(self):
        result = pre_dispatch_check(
            "set_input",
            {"node_id": "1", "field": "seed", "value": 42},
            breaker_state="closed",
            session_active=True,
            has_undo=True,
            action_history=["load_workflow"],  # satisfy scout_before_act
        )
        assert result.decision == GateDecision.ALLOW
        assert all(result.checks.values())

    def test_mutation_tool_circuit_open_denied(self):
        result = pre_dispatch_check(
            "set_input",
            {"node_id": "1", "field": "seed", "value": 42},
            breaker_state="open",
            session_active=True,
            has_undo=True,
        )
        assert result.decision == GateDecision.DENY
        assert result.checks["system_health"] is False

    def test_execution_tool_without_validation_denied(self):
        result = pre_dispatch_check(
            "execute_workflow",
            {},
            breaker_state="closed",
            session_active=True,
            validated=False,
            has_undo=True,
        )
        assert result.decision == GateDecision.DENY
        assert result.checks["consent"] is False

    def test_execution_tool_with_validation_allowed(self):
        result = pre_dispatch_check(
            "execute_workflow",
            {},
            breaker_state="closed",
            session_active=True,
            validated=True,
            has_undo=True,
            action_history=["load_workflow"],  # satisfy scout_before_act
        )
        assert result.decision == GateDecision.ALLOW

    def test_all_five_checks_reported(self):
        result = pre_dispatch_check(
            "set_input",
            {},
            session_active=True,
            has_undo=True,
        )
        assert len(result.checks) == 5
        expected_keys = {"system_health", "consent", "constitution", "reversibility", "scope"}
        assert set(result.checks.keys()) == expected_keys

    def test_single_check_failure_denies(self):
        """One bad check should result in DENY, not ALLOW."""
        # No undo -> reversibility fails for REVERSIBLE tool
        result = pre_dispatch_check(
            "set_input",
            {},
            session_active=True,
            has_undo=False,
        )
        assert result.decision == GateDecision.DENY
        assert result.checks["reversibility"] is False

    def test_gate_result_is_frozen(self):
        result = pre_dispatch_check("get_all_nodes", {})
        with pytest.raises(AttributeError):
            result.decision = GateDecision.DENY  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scope check tests (path sanitization)
# ---------------------------------------------------------------------------


class TestScopeChecks:
    def test_path_traversal_blocked(self):
        result = pre_dispatch_check(
            "set_input",
            {"path": "../../etc/passwd"},
            session_active=True,
            has_undo=True,
        )
        assert result.decision == GateDecision.DENY
        assert result.checks["scope"] is False

    def test_system_path_blocked(self):
        result = pre_dispatch_check(
            "set_input",
            {"file_path": "C:\\Windows\\System32\\evil.dll"},
            session_active=True,
            has_undo=True,
        )
        assert result.decision == GateDecision.DENY
        assert result.checks["scope"] is False

    def test_safe_path_allowed(self):
        result = pre_dispatch_check(
            "set_input",
            {"file_path": "G:/COMFYUI_Database/workflows/test.json"},
            session_active=True,
            has_undo=True,
        )
        assert result.checks["scope"] is True

    def test_non_path_keys_ignored(self):
        """Keys not in _PATH_KEYS should not trigger scope check."""
        result = pre_dispatch_check(
            "set_input",
            {"value": "../../etc/passwd", "seed": 42},
            session_active=True,
            has_undo=True,
        )
        assert result.checks["scope"] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGateEdgeCases:
    def test_half_open_breaker_allows(self):
        result = pre_dispatch_check(
            "set_input",
            {},
            breaker_state="half_open",
            session_active=True,
            has_undo=True,
        )
        assert result.checks["system_health"] is True

    def test_no_session_denies_mutation(self):
        result = pre_dispatch_check(
            "set_input",
            {},
            session_active=False,
            has_undo=True,
        )
        assert result.decision == GateDecision.DENY
        assert result.checks["consent"] is False

    def test_empty_tool_input(self):
        result = pre_dispatch_check(
            "set_input",
            {},
            session_active=True,
            has_undo=True,
        )
        assert result.checks["scope"] is True

    def test_provision_with_path_traversal(self):
        """Provision tool with path traversal should still ESCALATE (consent fails first)."""
        result = pre_dispatch_check(
            "install_node_pack",
            {"path": "../../evil"},
            session_active=True,
        )
        # Consent check fails first for PROVISION, returning ESCALATE
        assert result.decision == GateDecision.ESCALATE

    def test_reason_populated_on_deny(self):
        result = pre_dispatch_check(
            "set_input",
            {},
            breaker_state="open",
            session_active=True,
            has_undo=True,
        )
        assert result.decision == GateDecision.DENY
        assert len(result.reason) > 0
