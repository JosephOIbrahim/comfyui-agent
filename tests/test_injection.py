"""Tests for agent/stage/injection.py — no real I/O."""

from __future__ import annotations

import time

import pytest

from agent.stage.injection import (
    InjectionError,
    InjectionState,
    get_state,
    inject,
    inject_none,
    modulated_gain,
)


@pytest.fixture
def usd_stage():
    pytest.importorskip("pxr", reason="usd-core not installed")
    from agent.stage.cognitive_stage import CognitiveWorkflowStage
    return CognitiveWorkflowStage()


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset injection state between tests."""
    inject_none()
    yield
    inject_none()


class TestInjectionState:
    def test_defaults(self):
        s = InjectionState()
        assert s.profile_name == "none"
        assert s.alpha == 0.0
        assert s.active is False

    def test_compute_alpha_inactive(self):
        s = InjectionState()
        assert s.compute_alpha() == 0.0

    def test_compute_alpha_at_start(self):
        now = time.time()
        s = InjectionState(
            active=True, start_time=now, peak=1.0,
            onset_tau=100.0, plateau_duration=600.0,
        )
        alpha = s.compute_alpha(now)
        assert alpha < 0.05  # Near zero at t=0

    def test_compute_alpha_after_onset(self):
        now = time.time()
        s = InjectionState(
            active=True, start_time=now - 300.0, peak=1.0,
            onset_tau=100.0, plateau_duration=600.0,
        )
        alpha = s.compute_alpha(now)
        assert alpha > 0.9  # Near peak after 3*tau

    def test_compute_alpha_during_offset(self):
        now = time.time()
        s = InjectionState(
            active=True, start_time=now - 1200.0, peak=1.0,
            onset_tau=100.0, plateau_duration=600.0, offset_tau=200.0,
        )
        alpha = s.compute_alpha(now)
        assert alpha < 0.5  # Decaying after plateau

    def test_phase_baseline(self):
        s = InjectionState()
        assert s.phase == "baseline"

    def test_to_dict(self):
        s = InjectionState(profile_name="explore", alpha=0.5)
        d = s.to_dict()
        assert d["profile_name"] == "explore"
        assert "phase" in d


class TestInject:
    def test_inject_activates(self):
        state = inject(profile_name="explore")
        assert state.active is True
        assert state.profile_name == "explore"

    def test_inject_with_cws(self, usd_stage):
        state = inject(usd_stage, "creative")
        assert state.profile_name == "creative"
        assert state.active is True

    def test_inject_invalid_peak(self):
        with pytest.raises(InjectionError, match="Peak"):
            inject(profile_name="explore", peak=1.5)

    def test_inject_invalid_profile(self):
        from agent.stage.creative_profiles import ProfileError
        with pytest.raises(ProfileError):
            inject(profile_name="nonexistent")

    def test_inject_custom_params(self):
        state = inject(
            profile_name="radical",
            peak=0.8, onset_tau=60.0,
            plateau_duration=300.0, offset_tau=150.0,
        )
        assert state.peak == 0.8
        assert state.onset_tau == 60.0

    def test_inject_updates_global_state(self):
        inject(profile_name="creative")
        s = get_state()
        assert s.profile_name == "creative"
        assert s.active is True


class TestInjectNone:
    def test_deactivates(self):
        inject(profile_name="radical")
        state = inject_none()
        assert state.active is False
        assert state.alpha == 0.0

    def test_resets_global_state(self):
        inject(profile_name="creative")
        inject_none()
        s = get_state()
        assert s.active is False

    def test_with_cws(self, usd_stage):
        inject(usd_stage, "radical")
        state = inject_none(usd_stage)
        assert state.active is False


class TestModulatedGain:
    def test_baseline_returns_one(self):
        assert modulated_gain("cfg") == 1.0

    def test_active_injection_modulates(self):
        inject(profile_name="radical")
        # Force alpha to 1.0 for testing
        s = get_state()
        s.alpha = 1.0
        s.start_time = time.time() - 500  # past onset
        gain = modulated_gain("sampler")
        # radical.sampler_gain = 3.0, alpha ≈ high
        assert gain > 1.0

    def test_unknown_group_returns_one(self):
        inject(profile_name="explore")
        s = get_state()
        s.alpha = 1.0
        assert modulated_gain("unknown_group") == 1.0

    def test_integration_reduces_gain(self):
        inject(profile_name="integration")
        s = get_state()
        s.alpha = 1.0
        s.start_time = time.time() - 500
        gain = modulated_gain("cfg")
        # integration.cfg_gain = 0.6, so gain < 1.0
        assert gain < 1.0


# ---------------------------------------------------------------------------
# Cycle 62: stage profile select failure must log at DEBUG
# ---------------------------------------------------------------------------

class TestProfileSelectLogging:
    """Stage profile select failure → log.debug (Cycle 62)."""

    def test_inject_stage_failure_logs_debug(self, caplog, _reset_state):
        """When cws profile select raises during inject(), debug message appears."""
        import logging
        from unittest.mock import MagicMock, patch

        cws_mock = MagicMock()

        with patch("agent.stage.injection.read_active_profile", return_value="explore"), \
             patch("agent.stage.injection.select_profile",
                   side_effect=RuntimeError("USD prim error")), \
             caplog.at_level(logging.DEBUG, logger="agent.stage.injection"):
            inject(profile_name="explore", cws=cws_mock)

        assert any("profile" in r.message.lower() or "degrad" in r.message.lower()
                   for r in caplog.records), "Expected debug log on profile select failure"

    def test_inject_none_stage_failure_logs_debug(self, caplog, _reset_state):
        """When cws profile reset raises during inject_none(), debug message appears."""
        import logging
        from unittest.mock import MagicMock, patch

        cws_mock = MagicMock()

        with patch("agent.stage.injection.select_profile",
                   side_effect=RuntimeError("USD write error")), \
             caplog.at_level(logging.DEBUG, logger="agent.stage.injection"):
            inject_none(cws=cws_mock)

        assert any("profile" in r.message.lower() or "degrad" in r.message.lower()
                   for r in caplog.records), "Expected debug log on profile reset failure"

    def test_inject_stage_failure_does_not_raise(self, _reset_state):
        """inject() must succeed even when the USD stage select raises."""
        from unittest.mock import MagicMock, patch

        cws_mock = MagicMock()
        with patch("agent.stage.injection.read_active_profile", return_value="explore"), \
             patch("agent.stage.injection.select_profile",
                   side_effect=RuntimeError("no USD")):
            state = inject(profile_name="explore", cws=cws_mock)

        assert state.active is True

    def test_inject_none_stage_failure_does_not_raise(self, _reset_state):
        """inject_none() must succeed even when the USD stage reset raises."""
        from unittest.mock import MagicMock, patch

        cws_mock = MagicMock()
        with patch("agent.stage.injection.select_profile",
                   side_effect=RuntimeError("no USD")):
            state = inject_none(cws=cws_mock)

        assert state.active is False
