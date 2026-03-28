"""Creative Profiles — gain modulation for artistic exploration.

Profiles control the "exploration radius" of parameter modifications.
Each profile defines per-parameter-group gain multipliers that scale
how much the system adjusts parameters during autoresearch.

Four profiles:
  explore       Mild widening — slightly larger parameter changes
  creative      Moderate — noticeable creative range expansion
  radical       Full dissolution — maximum exploration
  integration   Focused — tighter, more precise adjustments

Stored as USD variant sets for clean switching via LIVRPS composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from pxr import Sdf
    HAS_USD = True
except ImportError:
    HAS_USD = False


class ProfileError(Exception):
    """Base error for creative profile operations."""


@dataclass(frozen=True)
class CreativeProfile:
    """Gain modulation profile for parameter groups."""

    name: str
    sampler_gain: float = 1.0     # Multiplier for sampler step range
    cfg_gain: float = 1.0         # Multiplier for CFG scale range
    steps_gain: float = 1.0       # Multiplier for step count range
    denoise_gain: float = 1.0     # Multiplier for denoise strength range
    lora_gain: float = 1.0        # Multiplier for LoRA weight range
    description: str = ""

    def apply(self, base_value: float, param_group: str) -> float:
        """Apply gain to a base value for a parameter group.

        Args:
            base_value: The base parameter change amount.
            param_group: One of sampler, cfg, steps, denoise, lora.

        Returns:
            Gain-modulated value.
        """
        gain = self.get_gain(param_group)
        return base_value * gain

    def get_gain(self, param_group: str) -> float:
        """Get the gain for a parameter group."""
        gains = {
            "sampler": self.sampler_gain,
            "cfg": self.cfg_gain,
            "steps": self.steps_gain,
            "denoise": self.denoise_gain,
            "lora": self.lora_gain,
        }
        return gains.get(param_group, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sampler_gain": self.sampler_gain,
            "cfg_gain": self.cfg_gain,
            "steps_gain": self.steps_gain,
            "denoise_gain": self.denoise_gain,
            "lora_gain": self.lora_gain,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

EXPLORE = CreativeProfile(
    name="explore",
    sampler_gain=1.3,
    cfg_gain=1.2,
    steps_gain=1.2,
    denoise_gain=1.1,
    lora_gain=1.2,
    description="Mild widening — slightly larger parameter changes",
)

CREATIVE = CreativeProfile(
    name="creative",
    sampler_gain=1.8,
    cfg_gain=1.6,
    steps_gain=1.5,
    denoise_gain=1.4,
    lora_gain=1.5,
    description="Moderate — noticeable creative range expansion",
)

RADICAL = CreativeProfile(
    name="radical",
    sampler_gain=3.0,
    cfg_gain=2.5,
    steps_gain=2.0,
    denoise_gain=2.0,
    lora_gain=2.5,
    description="Full dissolution — maximum exploration",
)

INTEGRATION = CreativeProfile(
    name="integration",
    sampler_gain=0.5,
    cfg_gain=0.6,
    steps_gain=0.7,
    denoise_gain=0.5,
    lora_gain=0.6,
    description="Focused — tighter, more precise adjustments",
)

PROFILES: dict[str, CreativeProfile] = {
    "explore": EXPLORE,
    "creative": CREATIVE,
    "radical": RADICAL,
    "integration": INTEGRATION,
}

GAIN_ATTRS = ("sampler_gain", "cfg_gain", "steps_gain", "denoise_gain", "lora_gain")


def get_profile(name: str) -> CreativeProfile:
    """Get a profile by name.

    Raises:
        ProfileError: If name is not a known profile.
    """
    if name not in PROFILES:
        raise ProfileError(
            f"Unknown profile: {name}. "
            f"Available: {', '.join(PROFILES.keys())}"
        )
    return PROFILES[name]


def store_as_variant_set(
    cws: Any,  # CognitiveWorkflowStage
    profiles: list[CreativeProfile] | None = None,
    *,
    prim_path: str = "/agents/creative_profile",
    variant_set_name: str = "profile",
) -> str:
    """Store creative profiles as a USD variant set.

    Args:
        cws: CognitiveWorkflowStage instance.
        profiles: Profiles to store. Defaults to all built-in profiles.
        prim_path: Where to create the variant set prim.
        variant_set_name: Name of the variant set.

    Returns:
        The prim path of the variant set.

    Raises:
        ProfileError: If USD is not available.
    """
    if not HAS_USD:
        raise ProfileError("USD not available")

    profiles = profiles or list(PROFILES.values())

    stage = cws.stage
    prim = stage.DefinePrim(prim_path)
    vsets = prim.GetVariantSets()
    vset = vsets.AddVariantSet(variant_set_name)

    for profile in profiles:
        vset.AddVariant(profile.name)
        vset.SetVariantSelection(profile.name)
        with vset.GetVariantEditContext():
            for attr_name in GAIN_ATTRS:
                val = getattr(profile, attr_name)
                attr = prim.GetAttribute(attr_name)
                if not attr.IsValid():
                    prim.CreateAttribute(
                        attr_name, Sdf.ValueTypeNames.Double
                    ).Set(val)
                else:
                    attr.Set(val)
            # Store description
            desc_attr = prim.GetAttribute("description")
            if not desc_attr.IsValid():
                prim.CreateAttribute(
                    "description", Sdf.ValueTypeNames.String
                ).Set(profile.description)
            else:
                desc_attr.Set(profile.description)

    # Default to explore
    vset.SetVariantSelection("explore")
    return prim_path


def select_profile(
    cws: Any,  # CognitiveWorkflowStage
    name: str,
    *,
    prim_path: str = "/agents/creative_profile",
    variant_set_name: str = "profile",
) -> CreativeProfile:
    """Switch the active creative profile via variant selection.

    Args:
        cws: CognitiveWorkflowStage instance.
        name: Profile name to activate.
        prim_path: Path to the variant set prim.
        variant_set_name: Name of the variant set.

    Returns:
        The selected CreativeProfile.

    Raises:
        ProfileError: If name is unknown or prim doesn't exist.
    """
    profile = get_profile(name)

    if HAS_USD:
        prim = cws.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            vsets = prim.GetVariantSets()
            if variant_set_name in vsets.GetNames():
                vsets.GetVariantSet(variant_set_name).SetVariantSelection(name)

    return profile


def read_active_profile(
    cws: Any,  # CognitiveWorkflowStage
    *,
    prim_path: str = "/agents/creative_profile",
    variant_set_name: str = "profile",
) -> str | None:
    """Read the currently active profile name from the stage.

    Returns:
        Profile name string, or None if not set.
    """
    if not HAS_USD:
        return None

    prim = cws.stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    vsets = prim.GetVariantSets()
    if variant_set_name not in vsets.GetNames():
        return None

    return vsets.GetVariantSet(variant_set_name).GetVariantSelection()
