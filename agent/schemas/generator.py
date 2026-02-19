"""Schema Generator — infer output schemas from example dicts.

Given a concrete example of agent output, the generator produces a YAML
schema definition that can be used with ``loader.validate_output()``.
This lets artists create custom schemas by providing example outputs
rather than hand-writing YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InferredField:
    """A single field inferred from an example value."""

    name: str
    type: str
    required: bool = True
    description: str = ""
    example_value: Any = None
    nested_fields: list[InferredField] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_field(name: str, value: Any) -> InferredField:
    """Infer a field definition from a name-value pair.

    Type inference rules:
    - ``str`` -> ``"string"``
    - ``bool`` -> ``"boolean"`` (checked before int since bool is int subclass)
    - ``int`` -> ``"integer"``
    - ``float`` -> ``"float"`` (with auto range [0.0, 1.0] if value is in that range)
    - ``list`` -> ``"list[string]"`` if all str, ``"list"`` with item_schema if
      all dicts, ``"list"`` otherwise
    - ``dict`` -> ``"object"`` with nested fields
    - ``None`` / other -> ``"any"``
    """
    if isinstance(value, str):
        return InferredField(name=name, type="string", example_value=value)

    if isinstance(value, bool):
        return InferredField(name=name, type="boolean", example_value=value)

    if isinstance(value, int):
        return InferredField(name=name, type="integer", example_value=value)

    if isinstance(value, float):
        f = InferredField(name=name, type="float", example_value=value)
        # Auto-detect [0, 1] range for floats in that range
        if 0.0 <= value <= 1.0:
            f.description = "Auto-detected range [0.0, 1.0]"
        return f

    if isinstance(value, list):
        if not value:
            return InferredField(name=name, type="list", example_value=[])

        # Check if all strings
        if all(isinstance(v, str) for v in value):
            return InferredField(
                name=name, type="list[string]", example_value=value
            )

        # Check if all ints (not bools)
        if all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            return InferredField(
                name=name, type="list[integer]", example_value=value
            )

        # Check if all dicts -> list with item_schema
        if all(isinstance(v, dict) for v in value):
            # Use first item as representative
            nested = []
            if value:
                for k, v in value[0].items():
                    nested.append(_infer_field(k, v))
            return InferredField(
                name=name,
                type="list",
                example_value=value,
                nested_fields=nested,
            )

        return InferredField(name=name, type="list", example_value=value)

    if isinstance(value, dict):
        nested = [_infer_field(k, v) for k, v in value.items()]
        return InferredField(
            name=name,
            type="object",
            example_value=value,
            nested_fields=nested,
        )

    return InferredField(name=name, type="any", example_value=value)


def _field_to_schema(f: InferredField) -> dict[str, Any]:
    """Convert an InferredField to a schema field dict."""
    result: dict[str, Any] = {"type": f.type, "required": f.required}

    if f.description:
        result["description"] = f.description

    # Float range auto-detection
    if f.type == "float" and isinstance(f.example_value, float):
        if 0.0 <= f.example_value <= 1.0:
            result["range"] = [0.0, 1.0]

    # List with item_schema (nested dicts)
    if f.type == "list" and f.nested_fields:
        item_schema: dict[str, Any] = {}
        for nf in f.nested_fields:
            item_schema[nf.name] = _field_to_schema(nf)
        result["item_schema"] = item_schema

    # Object with nested fields
    if f.type == "object" and f.nested_fields:
        nested: dict[str, Any] = {}
        for nf in f.nested_fields:
            nested[nf.name] = _field_to_schema(nf)
        result["nested_fields"] = nested

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def infer_schema_from_example(
    example: dict[str, Any],
    agent: str,
    schema_name: str,
    extends: str | None = "default",
) -> dict[str, Any]:
    """Create a schema dict from an example output dict.

    Parameters
    ----------
    example:
        A concrete example of agent output.
    agent:
        Agent name (e.g. ``"intent"``, ``"execution"``, ``"verify"``).
    schema_name:
        Name for the generated schema.
    extends:
        Base schema to inherit from.  Set to ``None`` to skip
        inheritance.

    Returns
    -------
    dict
        A complete schema dict ready for ``write_schema()``.
    """
    schema: dict[str, Any] = {
        "schema": {
            "name": schema_name,
            "version": "1.0",
            "agent": agent,
            "description": f"Auto-generated schema from example for {agent}",
        },
        "fields": {},
    }

    if extends is not None:
        schema["extends"] = extends

    for key, value in example.items():
        inferred = _infer_field(key, value)
        schema["fields"][key] = _field_to_schema(inferred)

    return schema


def write_schema(schema: dict[str, Any], path: Path | str) -> None:
    """Write a schema dict to a YAML file.

    Uses ``sort_keys=False`` to preserve field ordering for readability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            schema,
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
