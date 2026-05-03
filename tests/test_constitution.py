"""Tests for agent/stage/constitution.py — all mocked, no real I/O."""

from __future__ import annotations

import pytest

from agent.stage.constitution import (
    ALL_COMMANDMENTS,
    CheckResult,
    adversarial_verification,
    all_passed,
    bounded_failure,
    complete_output,
    explicit_handoffs,
    failed_checks,
    human_gates,
    role_isolation,
    run_post_checks,
    run_pre_checks,
    scout_before_act,
    verify_after_mutation,
)


# ---------------------------------------------------------------------------
# CheckResult basics
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_to_dict(self):
        r = CheckResult(passed=True, commandment="test", reason="ok")
        d = r.to_dict()
        assert d == {"passed": True, "commandment": "test", "reason": "ok"}

    def test_frozen(self):
        r = CheckResult(passed=True, commandment="test", reason="ok")
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Commandment 1: scout_before_act
# ---------------------------------------------------------------------------

class TestScoutBeforeAct:
    def test_first_action_recon_passes(self):
        r = scout_before_act([], "discover")
        assert r.passed is True

    def test_empty_history_passes_when_not_tracked(self):
        """Empty action_history means 'not tracked', not 'no actions taken'."""
        r = scout_before_act([], "add_node")
        assert r.passed is True
        assert "not tracked" in r.reason.lower()

    def test_non_first_action_passes(self):
        r = scout_before_act(["discover"], "add_node")
        assert r.passed is True


# ---------------------------------------------------------------------------
# Commandment 2: verify_after_mutation
# ---------------------------------------------------------------------------

class TestVerifyAfterMutation:
    def test_mutation_then_verify_passes(self):
        r = verify_after_mutation("apply_workflow_patch", "validate_workflow", False)
        assert r.passed is True

    def test_mutation_then_mutation_fails(self):
        r = verify_after_mutation("apply_workflow_patch", "add_node", False)
        assert r.passed is False

    def test_mutation_already_verified_passes(self):
        r = verify_after_mutation("apply_workflow_patch", "add_node", True)
        assert r.passed is True

    def test_no_last_action_passes(self):
        r = verify_after_mutation(None, "add_node", False)
        assert r.passed is True

    def test_recon_after_mutation_passes(self):
        r = verify_after_mutation("apply_workflow_patch", "get_editable_fields", False)
        assert r.passed is True


# ---------------------------------------------------------------------------
# Commandment 3: bounded_failure
# ---------------------------------------------------------------------------

class TestBoundedFailure:
    def test_zero_retries_passes(self):
        assert bounded_failure(0).passed is True

    def test_within_bounds_passes(self):
        assert bounded_failure(2).passed is True

    def test_at_limit_fails(self):
        assert bounded_failure(3).passed is False

    def test_over_limit_fails(self):
        assert bounded_failure(5).passed is False

    def test_custom_limit(self):
        assert bounded_failure(4, max_retries=5).passed is True
        assert bounded_failure(5, max_retries=5).passed is False


# ---------------------------------------------------------------------------
# Commandment 4: complete_output
# ---------------------------------------------------------------------------

class TestCompleteOutput:
    def test_clean_output_passes(self):
        assert complete_output("All nodes connected successfully.").passed is True

    def test_todo_fails(self):
        r = complete_output("TODO: finish this later")
        assert r.passed is False
        assert "TODO" in r.reason

    def test_fixme_fails(self):
        assert complete_output("FIXME: broken").passed is False

    def test_stub_fails(self):
        assert complete_output("This is a stub implementation").passed is False

    def test_not_implemented_fails(self):
        assert complete_output("raise NotImplementedError").passed is False

    def test_xxx_fails(self):
        assert complete_output("XXX temporary hack").passed is False


# ---------------------------------------------------------------------------
# Commandment 5: role_isolation
# ---------------------------------------------------------------------------

class TestRoleIsolation:
    def test_allowed_tool_passes(self):
        r = role_isolation("scout", "discover", ("discover", "list_models"))
        assert r.passed is True

    def test_disallowed_tool_fails(self):
        r = role_isolation("scout", "execute_workflow", ("discover",))
        assert r.passed is False
        assert "not allowed" in r.reason


# ---------------------------------------------------------------------------
# Commandment 6: explicit_handoffs
# ---------------------------------------------------------------------------

class TestExplicitHandoffs:
    def test_matching_artifact_passes(self):
        artifact = {"artifact_type": "recon_report", "data": {}}
        r = explicit_handoffs(artifact, "recon_report")
        assert r.passed is True

    def test_mismatched_artifact_fails(self):
        artifact = {"artifact_type": "design_spec", "data": {}}
        r = explicit_handoffs(artifact, "recon_report")
        assert r.passed is False

    def test_no_artifact_fails(self):
        r = explicit_handoffs(None, "recon_report")
        assert r.passed is False


# ---------------------------------------------------------------------------
# Commandment 7: adversarial_verification
# ---------------------------------------------------------------------------

class TestAdversarialVerification:
    def test_different_agents_passes(self):
        assert adversarial_verification("forge", "crucible").passed is True

    def test_same_agent_fails(self):
        r = adversarial_verification("forge", "forge")
        assert r.passed is False
        assert "same agent" in r.reason


# ---------------------------------------------------------------------------
# Commandment 8: human_gates
# ---------------------------------------------------------------------------

class TestHumanGates:
    def test_non_irreversible_passes(self):
        assert human_gates("discover").passed is True

    def test_irreversible_unapproved_fails(self):
        r = human_gates("reset_workflow", human_approved=False)
        assert r.passed is False
        assert "irreversible" in r.reason

    def test_irreversible_approved_passes(self):
        assert human_gates("reset_workflow", human_approved=True).passed is True


# ---------------------------------------------------------------------------
# Aggregate checks
# ---------------------------------------------------------------------------

class TestAggregateChecks:
    def test_all_commandments_count(self):
        # 10 commandments: original 8 + persistence_durability + self_healing_ladder
        # added in the Cozy Constitution revision (Article VIII).
        assert len(ALL_COMMANDMENTS) == 10

    def test_run_pre_checks_all_pass(self):
        results = run_pre_checks(
            agent_name="scout",
            proposed_tool="discover",
            allowed_tools=("discover",),
            action_history=["get_all_nodes"],
            last_action="get_all_nodes",
            verified_since_mutation=True,
            retry_count=0,
        )
        assert all_passed(results)

    def test_run_pre_checks_role_violation(self):
        results = run_pre_checks(
            agent_name="scout",
            proposed_tool="execute_workflow",
            allowed_tools=("discover",),
            action_history=["discover"],
            last_action="discover",
            verified_since_mutation=True,
        )
        assert not all_passed(results)
        fails = failed_checks(results)
        assert any(f.commandment == "role_isolation" for f in fails)

    def test_run_post_checks_clean_output(self):
        results = run_post_checks(
            output="Workflow validated successfully.",
            builder="forge",
            verifier="crucible",
        )
        assert all_passed(results)

    def test_run_post_checks_stub_output(self):
        results = run_post_checks(output="TODO: implement this")
        assert not all_passed(results)

    def test_run_post_checks_same_builder_verifier(self):
        results = run_post_checks(
            output="Clean output.",
            builder="forge",
            verifier="forge",
        )
        assert not all_passed(results)
        fails = failed_checks(results)
        assert any(f.commandment == "adversarial_verification" for f in fails)

    def test_run_post_checks_with_handoff(self):
        results = run_post_checks(
            output="Done.",
            handoff_artifact={"artifact_type": "recon_report"},
            expected_artifact_type="recon_report",
        )
        assert all_passed(results)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_all_passed_true(self):
        results = [
            CheckResult(passed=True, commandment="a", reason="ok"),
            CheckResult(passed=True, commandment="b", reason="ok"),
        ]
        assert all_passed(results) is True

    def test_all_passed_false(self):
        results = [
            CheckResult(passed=True, commandment="a", reason="ok"),
            CheckResult(passed=False, commandment="b", reason="fail"),
        ]
        assert all_passed(results) is False

    def test_failed_checks_filters(self):
        results = [
            CheckResult(passed=True, commandment="a", reason="ok"),
            CheckResult(passed=False, commandment="b", reason="fail"),
            CheckResult(passed=False, commandment="c", reason="fail2"),
        ]
        fails = failed_checks(results)
        assert len(fails) == 2
        assert all(not f.passed for f in fails)
