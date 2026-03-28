"""Injection — applies creative profiles with pharmacokinetic alpha curves.

inject(stage, profile_name) activates a creative profile and modulates
the ratchet exploration range via a time-varying alpha curve:

  alpha(t) = peak * (1 - exp(-t/onset_tau)) * exp(-max(0, t-plateau)/offset_tau)

Phases:
  onset     Exponential rise to peak (0 → peak)
  plateau   Hold at peak intensity
  offset    Exponential decay back to baseline

inject_none() returns to baseline (alpha=0, default profile gains).

From the digital injection framework patent.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from .creative_profiles import (
    PROFILES,
    get_profile,
    select_profile,
    store_as_variant_set,
    read_active_profile,
)


class InjectionError(Exception):
    """Base error for injection operations."""


@dataclass
class InjectionState:
    """Current injection state with pharmacokinetic alpha curve."""

    profile_name: str = "none"
    alpha: float = 0.0          # Current intensity [0, 1]
    peak: float = 1.0           # Maximum alpha
    onset_tau: float = 120.0    # Onset time constant (seconds)
    plateau_duration: float = 600.0   # Plateau hold (seconds)
    offset_tau: float = 300.0   # Offset time constant (seconds)
    start_time: float = 0.0     # When injection started
    active: bool = False

    def compute_alpha(self, now: float | None = None) -> float:
        """Compute current alpha from pharmacokinetic curve.

        Args:
            now: Current time. Defaults to time.time().

        Returns:
            Alpha value [0, peak].
        """
        if not self.active:
            return 0.0

        t = (now if now is not None else time.time()) - self.start_time
        if t < 0:
            return 0.0

        # Onset: exponential rise
        onset = 1.0 - math.exp(-t / self.onset_tau) if self.onset_tau > 0 else 1.0

        # Offset: exponential decay after plateau
        t_after_plateau = t - self.plateau_duration
        if t_after_plateau > 0:
            offset = math.exp(-t_after_plateau / self.offset_tau) if self.offset_tau > 0 else 0.0
        else:
            offset = 1.0

        self.alpha = self.peak * onset * offset
        return self.alpha

    @property
    def phase(self) -> str:
        """Current phase: onset, plateau, offset, or baseline."""
        if not self.active:
            return "baseline"
        t = time.time() - self.start_time
        if t < self.onset_tau * 3:  # ~95% of onset reached at 3*tau
            return "onset"
        if t < self.plateau_duration:
            return "plateau"
        return "offset"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "alpha": self.alpha,
            "peak": self.peak,
            "phase": self.phase,
            "active": self.active,
            "onset_tau": self.onset_tau,
            "plateau_duration": self.plateau_duration,
            "offset_tau": self.offset_tau,
        }


# Module-level injection state (will be session-scoped later)
_state = InjectionState()


def get_state() -> InjectionState:
    """Get the current injection state."""
    return _state


def inject(
    cws: Any | None = None,
    profile_name: str = "explore",
    *,
    peak: float = 1.0,
    onset_tau: float = 120.0,
    plateau_duration: float = 600.0,
    offset_tau: float = 300.0,
) -> InjectionState:
    """Activate a creative profile with pharmacokinetic alpha curve.

    Args:
        cws: CognitiveWorkflowStage (optional, for variant selection).
        profile_name: Profile to activate.
        peak: Maximum alpha [0, 1]. Default 1.0.
        onset_tau: Onset time constant in seconds. Default 120.
        plateau_duration: How long to hold at peak. Default 600.
        offset_tau: Offset decay time constant. Default 300.

    Returns:
        Updated InjectionState.

    Raises:
        InjectionError: If profile_name is invalid or peak out of range.
    """
    global _state

    if not (0.0 <= peak <= 1.0):
        raise InjectionError(f"Peak must be in [0, 1], got {peak}")

    get_profile(profile_name)  # validates name, raises if unknown

    if cws is not None:
        # Ensure variant set exists, then select
        try:
            if read_active_profile(cws) is None:
                store_as_variant_set(cws)
            select_profile(cws, profile_name)
        except Exception:
            pass  # Degrade gracefully

    _state = InjectionState(
        profile_name=profile_name,
        alpha=0.0,
        peak=peak,
        onset_tau=onset_tau,
        plateau_duration=plateau_duration,
        offset_tau=offset_tau,
        start_time=time.time(),
        active=True,
    )

    return _state


def inject_none(cws: Any | None = None) -> InjectionState:
    """Return to baseline — deactivate all injection.

    Args:
        cws: CognitiveWorkflowStage (optional).

    Returns:
        Reset InjectionState at baseline.
    """
    global _state

    if cws is not None:
        try:
            select_profile(cws, "explore")  # Default/neutral
        except Exception:
            pass

    _state = InjectionState()
    return _state


def modulated_gain(param_group: str) -> float:
    """Get the current gain for a parameter group, modulated by alpha.

    The gain is interpolated between 1.0 (baseline) and the profile's
    gain for this group, scaled by the current alpha.

    Returns:
        Effective gain multiplier.
    """
    if not _state.active:
        return 1.0

    profile = PROFILES.get(_state.profile_name)
    if profile is None:
        return 1.0

    alpha = _state.compute_alpha()
    profile_gain = profile.get_gain(param_group)

    # Interpolate: baseline(1.0) → profile_gain, scaled by alpha
    return 1.0 + alpha * (profile_gain - 1.0)
