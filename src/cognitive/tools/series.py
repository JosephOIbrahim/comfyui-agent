"""generate_series — Multi-image with style consistency.

New capability: generate a series of images with consistent
style, varying specific parameters (seed, prompt elements)
while keeping others locked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeriesConfig:
    """Configuration for a generation series."""

    base_workflow: dict[str, Any] = field(default_factory=dict)
    vary_params: dict[str, list[Any]] = field(default_factory=dict)
    lock_params: dict[str, Any] = field(default_factory=dict)
    count: int = 1
    style_lock: bool = True  # Keep style-related params consistent


@dataclass
class SeriesResult:
    """Result of a series generation."""

    planned_count: int = 0
    variations: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and len(self.variations) > 0


def generate_series(config: SeriesConfig) -> SeriesResult:
    """Plan a series of workflow variations for batch generation.

    Creates a list of mutation sets that vary specified parameters
    while keeping others locked. Actual execution is delegated to
    the execution pipeline.

    Args:
        config: SeriesConfig with base workflow and variation spec.

    Returns:
        SeriesResult with planned variations.
    """
    result = SeriesResult()

    if not config.base_workflow:
        result.error = "No base workflow provided"
        return result

    if not config.vary_params and config.count <= 1:
        result.error = "Nothing to vary — provide vary_params or count > 1"
        return result

    result.planned_count = config.count

    # Generate variation sets
    for i in range(config.count):
        variation: dict[str, Any] = {"index": i, "mutations": {}}

        # Apply varying parameters
        for param_path, values in config.vary_params.items():
            if values:
                # Cycle through values
                val = values[i % len(values)]
                # Parse param_path as "node_id.param_name"
                if "." in param_path:
                    node_id, param = param_path.split(".", 1)
                    variation["mutations"].setdefault(node_id, {})[param] = val

        # Apply locked parameters
        for param_path, val in config.lock_params.items():
            if "." in param_path:
                node_id, param = param_path.split(".", 1)
                variation["mutations"].setdefault(node_id, {})[param] = val

        result.variations.append(variation)

    return result
