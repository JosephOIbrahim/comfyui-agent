"""Schema cache for ComfyUI /object_info.

Parses the raw /object_info response into typed NodeSchema objects
and provides mutation validation BEFORE patches reach the graph engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputSpec:
    """Specification for a single node input."""

    name: str
    input_type: str  # e.g. "INT", "FLOAT", "STRING", "MODEL", "CLIP", combo list, etc.
    required: bool = True
    default: Any = None
    min_val: Any = None
    max_val: Any = None
    valid_values: list[str] | None = None  # For combo/enum types

    @classmethod
    def from_object_info(cls, name: str, spec: Any, required: bool = True) -> InputSpec:
        """Parse a single input spec from /object_info format.

        ComfyUI input specs are typically: [type_name, {options}]
        For combo types: [["option1", "option2", ...], {options}]
        """
        if not isinstance(spec, (list, tuple)) or len(spec) == 0:
            return cls(name=name, input_type="UNKNOWN", required=required)

        type_or_values = spec[0]
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}

        if isinstance(type_or_values, list):
            # Combo/enum type
            return cls(
                name=name,
                input_type="COMBO",
                required=required,
                default=options.get("default"),
                valid_values=type_or_values,
            )

        input_type = str(type_or_values)
        return cls(
            name=name,
            input_type=input_type,
            required=required,
            default=options.get("default"),
            min_val=options.get("min"),
            max_val=options.get("max"),
        )


@dataclass
class NodeSchema:
    """Typed schema for a ComfyUI node class."""

    class_type: str
    display_name: str = ""
    category: str = ""
    description: str = ""
    inputs: dict[str, InputSpec] = field(default_factory=dict)
    output_types: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    @classmethod
    def from_object_info(cls, class_type: str, info: dict[str, Any]) -> NodeSchema:
        """Parse a single node from /object_info response."""
        inputs: dict[str, InputSpec] = {}

        for section, required in [("required", True), ("optional", False)]:
            section_data = info.get("input", {}).get(section, {})
            if isinstance(section_data, dict):
                for inp_name, inp_spec in section_data.items():
                    inputs[inp_name] = InputSpec.from_object_info(
                        inp_name, inp_spec, required=required,
                    )

        return cls(
            class_type=class_type,
            display_name=info.get("display_name", class_type),
            category=info.get("category", ""),
            description=info.get("description", ""),
            inputs=inputs,
            output_types=info.get("output", []),
            output_names=info.get("output_name", []),
        )


class SchemaCache:
    """In-memory cache of parsed ComfyUI node schemas.

    Provides mutation validation: check whether a proposed parameter
    change is valid BEFORE it reaches the graph engine.
    """

    def __init__(self):
        self._schemas: dict[str, NodeSchema] = {}
        self._last_refresh: float = 0.0
        self._raw_data: dict[str, Any] = {}

    @property
    def is_populated(self) -> bool:
        """True if the cache has been populated at least once."""
        return len(self._schemas) > 0

    @property
    def node_count(self) -> int:
        """Number of node schemas in the cache."""
        return len(self._schemas)

    @property
    def last_refresh(self) -> float:
        """Timestamp of last refresh."""
        return self._last_refresh

    def refresh(self, object_info: dict[str, Any]) -> int:
        """Parse /object_info response into typed schemas.

        Args:
            object_info: Raw response from GET /object_info.

        Returns:
            Number of node schemas parsed.
        """
        schemas: dict[str, NodeSchema] = {}
        for class_type, info in object_info.items():
            if isinstance(info, dict):
                schemas[class_type] = NodeSchema.from_object_info(class_type, info)
        self._schemas = schemas
        self._raw_data = object_info
        self._last_refresh = time.time()
        return len(schemas)

    async def async_refresh(self, api_client) -> int:
        """Fetch /object_info via an async client and refresh.

        Args:
            api_client: An httpx.AsyncClient with base_url set.

        Returns:
            Number of node schemas parsed.
        """
        resp = await api_client.get("/object_info", timeout=30.0)
        resp.raise_for_status()
        return self.refresh(resp.json())

    def get_schema(self, class_type: str) -> NodeSchema | None:
        """Get schema for a node class, or None if not cached."""
        return self._schemas.get(class_type)

    def validate_mutation(
        self,
        class_type: str,
        param_name: str,
        param_value: Any,
    ) -> tuple[bool, str]:
        """Validate a proposed mutation against the schema.

        Returns:
            (is_valid, reason) — reason is empty string when valid.
        """
        schema = self._schemas.get(class_type)
        if schema is None:
            return (False, f"Unknown node type: {class_type!r}")

        spec = schema.inputs.get(param_name)
        if spec is None:
            # Could be a connection input not in the schema
            return (True, "")

        # Combo validation: check if value is in valid_values
        if spec.valid_values is not None:
            if param_value not in spec.valid_values:
                return (
                    False,
                    f"{param_name} must be one of {spec.valid_values[:10]}, "
                    f"got {param_value!r}",
                )

        # Numeric range validation
        if spec.input_type in ("INT", "FLOAT"):
            try:
                num = float(param_value)
                if spec.min_val is not None and num < float(spec.min_val):
                    return (False, f"{param_name} minimum is {spec.min_val}, got {param_value}")
                if spec.max_val is not None and num > float(spec.max_val):
                    return (False, f"{param_name} maximum is {spec.max_val}, got {param_value}")
            except (ValueError, TypeError):
                pass

        return (True, "")

    def get_valid_values(self, class_type: str, param_name: str) -> list[str] | None:
        """Get valid values for a combo/enum parameter, or None."""
        schema = self._schemas.get(class_type)
        if schema is None:
            return None
        spec = schema.inputs.get(param_name)
        if spec is None or spec.valid_values is None:
            return None
        return list(spec.valid_values)

    def get_connectable_nodes(
        self,
        target_class_type: str,
        target_input: str,
    ) -> list[str]:
        """Find node types whose outputs can connect to a given input.

        Returns class_types that have an output matching the input's type.
        """
        schema = self._schemas.get(target_class_type)
        if schema is None:
            return []
        spec = schema.inputs.get(target_input)
        if spec is None:
            return []

        required_type = spec.input_type
        if required_type in ("COMBO", "UNKNOWN", "INT", "FLOAT", "STRING", "BOOLEAN"):
            return []  # Literal types, not connectable

        result = []
        for ct, s in sorted(self._schemas.items()):
            if required_type in s.output_types:
                result.append(ct)
        return result

    def list_node_types(self) -> list[str]:
        """List all cached node class_types."""
        return sorted(self._schemas.keys())
