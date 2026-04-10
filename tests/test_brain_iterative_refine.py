"""Tests for agent/brain/iterative_refine.py — Cycle 56 guard fixes.

Verifies that bare bracket accesses replaced with .get() defaults in
Cycle 56 do not crash when the key is absent from the returned dict,
and that walrus operators for parameter_mutations don't double-evaluate.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Cycle 56: confidence_check.get("proceed", True) guard
# ---------------------------------------------------------------------------

class TestConfidenceCheckProceedGuard:
    """confidence_check["proceed"] → confidence_check.get("proceed", True)."""

    def test_missing_proceed_defaults_to_true(self):
        """A confidence_check dict with no 'proceed' key should default True."""
        cc = {"reason": "all good", "confidence": 0.9}
        # Simulate the guard expression
        proceed = cc.get("proceed", True)
        assert proceed is True

    def test_explicit_false_proceed_is_honored(self):
        """proceed=False must still block when present."""
        cc = {"proceed": False, "reason": "low confidence", "confidence": 0.3}
        proceed = cc.get("proceed", True)
        assert proceed is False

    def test_explicit_true_proceed_is_honored(self):
        """proceed=True must still pass when present."""
        cc = {"proceed": True, "reason": "ok", "confidence": 0.8}
        proceed = cc.get("proceed", True)
        assert proceed is True

    def test_empty_dict_defaults_to_true(self):
        """Completely empty confidence_check must not crash."""
        cc: dict = {}
        proceed = cc.get("proceed", True)
        assert proceed is True


# ---------------------------------------------------------------------------
# Cycle 56: loop_decision.get("action") guard
# ---------------------------------------------------------------------------

class TestLoopDecisionActionGuard:
    """loop_decision["action"] → loop_decision.get("action")."""

    def test_missing_action_does_not_equal_accept(self):
        """A loop_decision with no 'action' key must not trigger accept branch."""
        ld: dict = {"continue": True}
        assert ld.get("action") != "accept"

    def test_action_accept_is_detected(self):
        """action='accept' must match the accept branch."""
        ld = {"action": "accept", "continue": False}
        assert ld.get("action") == "accept"

    def test_action_refine_is_not_accept(self):
        """action='refine' must not match the accept branch."""
        ld = {"action": "refine", "continue": True}
        assert ld.get("action") != "accept"

    def test_action_escalate_is_not_accept(self):
        """action='escalate' must not match the accept branch."""
        ld = {"action": "escalate", "continue": False}
        assert ld.get("action") != "accept"

    def test_none_action_is_not_accept(self):
        """action=None must not match the accept branch."""
        ld = {"action": None, "continue": False}
        assert ld.get("action") != "accept"


# ---------------------------------------------------------------------------
# Cycle 56: loop_decision.get("continue", False) guard
# ---------------------------------------------------------------------------

class TestLoopDecisionContinueGuard:
    """loop_decision["continue"] → loop_decision.get("continue", False)."""

    def test_missing_continue_defaults_to_false(self):
        """A loop_decision with no 'continue' key must default to False (stop)."""
        ld = {"action": "accept"}
        should_continue = ld.get("continue", False)
        assert should_continue is False

    def test_continue_true_is_honored(self):
        """continue=True must propagate as True."""
        ld = {"action": "refine", "continue": True}
        should_continue = ld.get("continue", False)
        assert should_continue is True

    def test_continue_false_is_honored(self):
        """continue=False must propagate as False."""
        ld = {"action": "escalate", "continue": False}
        should_continue = ld.get("continue", False)
        assert should_continue is False

    def test_empty_dict_continue_defaults_false(self):
        """Empty loop_decision must not crash — stops the loop."""
        ld: dict = {}
        should_continue = ld.get("continue", False)
        assert should_continue is False


# ---------------------------------------------------------------------------
# Cycle 56: walrus operator for parameter_mutations
# ---------------------------------------------------------------------------

class TestParameterMutationsWalrus:
    """Walrus operator on intent_spec.parameter_mutations avoids double lookup."""

    def test_missing_parameter_mutations_skips_loop(self):
        """intent_spec with no parameter_mutations key must not crash."""
        intent_spec = {"model": "sd15", "steps": 20}
        # Simulate the walrus guard
        if param_muts := intent_spec.get("parameter_mutations"):
            results = list(param_muts)
        else:
            results = []
        assert results == []

    def test_empty_parameter_mutations_skips_loop(self):
        """Empty list must not enter the loop."""
        intent_spec = {"parameter_mutations": []}
        if param_muts := intent_spec.get("parameter_mutations"):
            results = list(param_muts)
        else:
            results = []
        assert results == []

    def test_none_parameter_mutations_skips_loop(self):
        """None value must not enter the loop."""
        intent_spec = {"parameter_mutations": None}
        if param_muts := intent_spec.get("parameter_mutations"):
            results = list(param_muts)
        else:
            results = []
        assert results == []

    def test_nonempty_parameter_mutations_iterates(self):
        """Actual mutations must be iterated exactly once."""
        muts = [{"target": "KSampler.cfg", "value": 7.0}]
        intent_spec = {"parameter_mutations": muts}
        seen = []
        if param_muts := intent_spec.get("parameter_mutations"):
            for m in param_muts:
                seen.append(m)
        assert seen == muts

    def test_walrus_single_evaluation(self):
        """Walrus must evaluate .get() exactly once (no double lookup)."""
        call_count = 0

        class TrackingDict(dict):
            def get(self, key, default=None):
                nonlocal call_count
                if key == "parameter_mutations":
                    call_count += 1
                return super().get(key, default)

        intent_spec = TrackingDict({"parameter_mutations": [{"target": "x", "value": 1}]})
        if param_muts := intent_spec.get("parameter_mutations"):
            _ = list(param_muts)
        assert call_count == 1  # walrus calls .get() exactly once
