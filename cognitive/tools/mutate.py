"""mutate_workflow — Schema-validated non-destructive mutation.

Absorbs PILOT tools into a single validated mutation interface.
All mutations go through schema validation (when available)
and the CognitiveGraphEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MutationResult:
    """Result of a workflow mutation."""

    success: bool = True
    changes: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    delta_layer_id: str = ""
    error: str = ""


def mutate_workflow(
    engine: Any,
    mutations: dict[str, dict[str, Any]],
    opinion: str = "L",
    description: str = "",
    schema_cache: Any | None = None,
) -> MutationResult:
    """Apply schema-validated mutations through the graph engine.

    Args:
        engine: CognitiveGraphEngine instance.
        mutations: {node_id: {param: value, ...}, ...}
        opinion: LIVRPS tier.
        description: Human-readable description.
        schema_cache: Optional SchemaCache for pre-validation.

    Returns:
        MutationResult with success status and details.
    """
    result = MutationResult()

    # Pre-validate against schema if available
    if schema_cache is not None and hasattr(schema_cache, "validate_mutation"):
        resolved = engine.to_api_json()
        for node_id, params in mutations.items():
            class_type = None
            if node_id in resolved:
                class_type = resolved[node_id].get("class_type")
            elif "class_type" in params:
                class_type = params["class_type"]

            if class_type is None:
                continue

            for param_name, param_value in params.items():
                if param_name == "class_type":
                    continue
                valid, reason = schema_cache.validate_mutation(
                    class_type, param_name, param_value,
                )
                if not valid:
                    result.validation_errors.append(
                        f"Node {node_id} ({class_type}).{param_name}: {reason}"
                    )

    if result.validation_errors:
        result.success = False
        result.error = f"{len(result.validation_errors)} validation error(s)"
        return result

    # Apply through engine
    try:
        delta = engine.mutate_workflow(
            mutations,
            opinion=opinion,
            description=description,
        )
        result.delta_layer_id = delta.layer_id

        # Build change report
        for node_id, params in mutations.items():
            for param_name, param_value in params.items():
                if param_name == "class_type":
                    continue
                result.changes.append({
                    "node_id": node_id,
                    "param": param_name,
                    "value": param_value,
                })

    except Exception as e:
        result.success = False
        result.error = str(e)

    return result
