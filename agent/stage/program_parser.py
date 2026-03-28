"""Program.md Parser — extracts structured autoresearch specs from markdown.

Reads a program.md file and extracts:
  objective        What the autoresearch is trying to achieve
  parameter_axes   Tunable parameters with ranges and current values
  anchor_params    Parameters that must not be modified
  strategy_hints   Guidance for the optimizer
  success_criteria How to measure success

Returns a ProgramSpec dataclass.

Expected markdown format:
```
# Objective
Generate photorealistic portraits with consistent lighting.

# Parameters
- steps: 20-50, current=30
- cfg: 5.0-12.0, current=7.5
- denoise: 0.3-0.9, current=0.6

# Anchors
- checkpoint: realisticVisionV60.safetensors
- resolution: 1024x1024

# Strategy
- Prioritize aesthetic quality over speed
- Try DPM++ 2M Karras first

# Success Criteria
- aesthetic >= 0.8
- lighting >= 0.7
```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ParseError(Exception):
    """Error parsing program.md."""


@dataclass
class ParameterAxis:
    """A tunable parameter with range and current value."""

    name: str
    min_value: float = 0.0
    max_value: float = 1.0
    current: float = 0.5
    step: float | None = None  # Optional step size

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "min": self.min_value,
            "max": self.max_value,
            "current": self.current,
        }
        if self.step is not None:
            d["step"] = self.step
        return d


@dataclass
class ProgramSpec:
    """Structured autoresearch program specification."""

    objective: str = ""
    parameter_axes: list[ParameterAxis] = field(default_factory=list)
    anchor_params: dict[str, str] = field(default_factory=dict)
    strategy_hints: list[str] = field(default_factory=list)
    success_criteria: dict[str, float] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "parameter_axes": [p.to_dict() for p in self.parameter_axes],
            "anchor_params": self.anchor_params,
            "strategy_hints": self.strategy_hints,
            "success_criteria": self.success_criteria,
        }

    @property
    def param_names(self) -> list[str]:
        """List of tunable parameter names."""
        return [p.name for p in self.parameter_axes]


# Regex patterns for parameter lines
_PARAM_RE = re.compile(
    r"-\s+(\w+)\s*:\s*"
    r"([\d.]+)\s*-\s*([\d.]+)"
    r"(?:\s*,\s*current\s*=\s*([\d.]+))?"
    r"(?:\s*,\s*step\s*=\s*([\d.]+))?"
)

# Regex for anchor lines: "- name: value"
_ANCHOR_RE = re.compile(r"-\s+(\w+)\s*:\s*(.+)")

# Regex for success criteria: "- axis >= threshold"
_CRITERIA_RE = re.compile(r"-\s+(\w+)\s*>=\s*([\d.]+)")


def parse_program(path: str | Path) -> ProgramSpec:
    """Parse a program.md file into a ProgramSpec.

    Args:
        path: Path to the markdown file.

    Returns:
        ProgramSpec with extracted fields.

    Raises:
        ParseError: If file cannot be read.
    """
    p = Path(path)
    if not p.exists():
        raise ParseError(f"File not found: {path}")

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        raise ParseError(f"Cannot read {path}: {e}") from e

    return parse_program_text(text)


def parse_program_text(text: str) -> ProgramSpec:
    """Parse program specification from markdown text.

    Args:
        text: Markdown text content.

    Returns:
        ProgramSpec with extracted fields.
    """
    spec = ProgramSpec(raw_text=text)

    sections = _split_sections(text)

    # Objective
    obj_text = sections.get("objective", "").strip()
    if obj_text:
        spec.objective = obj_text

    # Parameters
    params_text = sections.get("parameters", "")
    for match in _PARAM_RE.finditer(params_text):
        name = match.group(1)
        min_val = float(match.group(2))
        max_val = float(match.group(3))
        current = float(match.group(4)) if match.group(4) else (min_val + max_val) / 2
        step = float(match.group(5)) if match.group(5) else None
        spec.parameter_axes.append(ParameterAxis(
            name=name, min_value=min_val, max_value=max_val,
            current=current, step=step,
        ))

    # Anchors
    anchors_text = sections.get("anchors", "")
    for match in _ANCHOR_RE.finditer(anchors_text):
        spec.anchor_params[match.group(1)] = match.group(2).strip()

    # Strategy
    strategy_text = sections.get("strategy", "")
    for line in strategy_text.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            spec.strategy_hints.append(line[2:].strip())

    # Success Criteria
    criteria_text = sections.get("success criteria", sections.get("success", ""))
    for match in _CRITERIA_RE.finditer(criteria_text):
        spec.success_criteria[match.group(1)] = float(match.group(2))

    return spec


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown text into sections by # headers.

    Returns dict mapping lowercase section title to content.
    """
    sections: dict[str, str] = {}
    current_title = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            if current_title:
                sections[current_title] = "\n".join(current_lines)
            current_title = stripped[2:].strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title:
        sections[current_title] = "\n".join(current_lines)

    return sections
