"""Schema Registry — structured output contracts for specialist agents.

Each agent (intent, execution, verify) has a default schema that defines
the shape of its output.  Artists can override schemas by placing custom
YAML files in the ``custom/`` subdirectory under each agent folder.

Resolution order for ``load_schema(agent, schema_name)``:

1. ``{agent}/custom/{schema_name}.yaml``
2. ``{agent}/{schema_name}.yaml``
3. ``{agent}/default.yaml``

Schemas support single-level inheritance via an ``extends`` key in the
YAML root.  The base schema fields are merged with the extension, and
the extension wins on conflicts.

Thread-safe caching mirrors the pattern in ``profiles/loader.py``.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMAS_DIR: Path = Path(__file__).parent
"""Directory where agent schema subdirectories live."""

# ---------------------------------------------------------------------------
# Thread-safe cache
# ---------------------------------------------------------------------------

_cache: dict[str, dict[str, Any]] = {}
_cache_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file with explicit UTF-8 encoding."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Schema at {path} did not parse to a dict")
    return data


def _resolve_schema(agent: str, schema_name: str) -> dict[str, Any]:
    """Find and load a schema file using the resolution chain.

    Resolution order:
    1. ``{agent}/custom/{schema_name}.yaml``
    2. ``{agent}/{schema_name}.yaml``
    3. ``{agent}/default.yaml``

    Raises ``FileNotFoundError`` if no schema can be resolved.
    """
    agent_dir = SCHEMAS_DIR / agent

    # 1. Custom override
    custom_path = agent_dir / "custom" / f"{schema_name}.yaml"
    if custom_path.is_file():
        return _load_yaml(custom_path)

    # 2. Standard name
    standard_path = agent_dir / f"{schema_name}.yaml"
    if standard_path.is_file():
        return _load_yaml(standard_path)

    # 3. Default fallback (only if we weren't already looking for default)
    if schema_name != "default":
        default_path = agent_dir / "default.yaml"
        if default_path.is_file():
            return _load_yaml(default_path)

    raise FileNotFoundError(
        f"No schema found for agent={agent!r}, schema_name={schema_name!r}. "
        f"Searched: {custom_path}, {standard_path}"
    )


def _merge_schemas(
    base: dict[str, Any], extension: dict[str, Any]
) -> dict[str, Any]:
    """Merge *extension* onto *base* (single-level inheritance).

    - ``schema`` metadata is shallow-merged (extension wins).
    - ``fields`` are merged key-by-key; extension fields override base
      fields with the same name.
    - Any other top-level keys from the extension are added.
    """
    merged: dict[str, Any] = copy.deepcopy(base)

    # Merge schema metadata
    if "schema" in extension:
        merged.setdefault("schema", {})
        merged["schema"].update(extension["schema"])

    # Merge fields
    if "fields" in extension:
        merged.setdefault("fields", {})
        merged["fields"].update(copy.deepcopy(extension["fields"]))

    # Copy any other top-level keys from extension
    for key in extension:
        if key not in ("schema", "fields", "extends"):
            merged[key] = copy.deepcopy(extension[key])

    return merged


def _validate_type(
    name: str, value: Any, field_def: dict[str, Any]
) -> list[str]:
    """Validate a single value against its field definition.

    Returns a list of error strings (empty means valid).
    """
    errors: list[str] = []
    field_type = field_def.get("type", "any")

    if field_type == "any":
        return errors

    # --- Scalar types ---
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "float": (int, float),
        "boolean": bool,
    }

    if field_type in type_map:
        expected = type_map[field_type]
        # bool is subclass of int in Python, exclude it for int/float
        if field_type in ("integer", "float") and isinstance(value, bool):
            errors.append(
                f"Field {name!r}: expected {field_type}, got bool"
            )
        elif not isinstance(value, expected):
            errors.append(
                f"Field {name!r}: expected {field_type}, "
                f"got {type(value).__name__}"
            )

    # --- Range check for float/integer ---
    if field_type in ("float", "integer") and "range" in field_def:
        lo, hi = field_def["range"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < lo or value > hi:
                errors.append(
                    f"Field {name!r}: value {value} outside "
                    f"range [{lo}, {hi}]"
                )

    # --- Enum ---
    if field_type == "enum":
        allowed = field_def.get("values", [])
        if value not in allowed:
            errors.append(
                f"Field {name!r}: value {value!r} not in "
                f"allowed values {allowed}"
            )

    # --- list (generic) ---
    if field_type == "list":
        if not isinstance(value, list):
            errors.append(
                f"Field {name!r}: expected list, "
                f"got {type(value).__name__}"
            )
        elif "item_schema" in field_def:
            item_schema = field_def["item_schema"]
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    for sub_name, sub_def in item_schema.items():
                        if isinstance(sub_def, dict):
                            sub_required = sub_def.get("required", True)
                            if sub_name not in item:
                                if sub_required:
                                    errors.append(
                                        f"Field {name!r}[{i}]: "
                                        f"missing required sub-field "
                                        f"{sub_name!r}"
                                    )
                            else:
                                errors.extend(
                                    _validate_type(
                                        f"{name}[{i}].{sub_name}",
                                        item[sub_name],
                                        sub_def,
                                    )
                                )

    # --- list[string] ---
    if field_type == "list[string]":
        if not isinstance(value, list):
            errors.append(
                f"Field {name!r}: expected list[string], "
                f"got {type(value).__name__}"
            )
        else:
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    errors.append(
                        f"Field {name!r}[{i}]: expected string, "
                        f"got {type(item).__name__}"
                    )

    # --- list[integer] ---
    if field_type == "list[integer]":
        if not isinstance(value, list):
            errors.append(
                f"Field {name!r}: expected list[integer], "
                f"got {type(value).__name__}"
            )
        else:
            for i, item in enumerate(value):
                if not isinstance(item, int) or isinstance(item, bool):
                    errors.append(
                        f"Field {name!r}[{i}]: expected integer, "
                        f"got {type(item).__name__}"
                    )

    # --- object ---
    if field_type == "object":
        if not isinstance(value, dict):
            errors.append(
                f"Field {name!r}: expected object, "
                f"got {type(value).__name__}"
            )

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_schema(
    agent: str, schema_name: str = "default"
) -> dict[str, Any]:
    """Load (and cache) an agent output schema.

    Resolution order:
    1. ``{agent}/custom/{schema_name}.yaml``
    2. ``{agent}/{schema_name}.yaml``
    3. ``{agent}/default.yaml``

    If the schema declares ``extends: "other_name"``, the base schema is
    loaded and merged first.

    The returned dict is a deep copy so callers can mutate freely.
    """
    cache_key = f"{agent}:{schema_name}"

    with _cache_lock:
        if cache_key in _cache:
            return copy.deepcopy(_cache[cache_key])

    # --- Resolve outside the lock (I/O) --------------------------------
    schema = _resolve_schema(agent, schema_name)

    # --- Handle inheritance ---
    if "extends" in schema:
        base_name = schema["extends"]
        base = _resolve_schema(agent, base_name)
        schema = _merge_schemas(base, schema)

    # --- Store and return ------------------------------------------------
    with _cache_lock:
        _cache[cache_key] = schema

    return copy.deepcopy(schema)


def validate_output(
    output: dict[str, Any],
    agent: str,
    schema_name: str = "default",
) -> list[str]:
    """Validate an agent output dict against its schema.

    Returns a list of error strings.  An empty list means the output is
    valid.
    """
    schema = load_schema(agent, schema_name)
    fields = schema.get("fields", {})
    errors: list[str] = []

    for field_name, field_def in fields.items():
        if not isinstance(field_def, dict):
            continue

        required = field_def.get("required", False)

        if field_name not in output:
            if required:
                errors.append(
                    f"Missing required field: {field_name!r}"
                )
            continue

        errors.extend(
            _validate_type(field_name, output[field_name], field_def)
        )

    return errors


def list_schemas(agent: str) -> list[str]:
    """List all available schema names for an agent.

    Returns a deterministically sorted list of schema names (without
    the ``.yaml`` extension).
    """
    agent_dir = SCHEMAS_DIR / agent
    names: set[str] = set()

    # Standard schemas
    if agent_dir.is_dir():
        for path in agent_dir.glob("*.yaml"):
            names.add(path.stem)

    # Custom schemas
    custom_dir = agent_dir / "custom"
    if custom_dir.is_dir():
        for path in custom_dir.glob("*.yaml"):
            names.add(path.stem)

    return sorted(names)


def clear_cache() -> None:
    """Drop all cached schemas.  Intended for testing."""
    with _cache_lock:
        _cache.clear()
