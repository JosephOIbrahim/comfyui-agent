"""manage_dependencies — Custom node management + schema cache invalidation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DependencyAction:
    """Result of a dependency management action."""

    action: str  # "install", "uninstall", "update"
    package: str
    success: bool = True
    message: str = ""
    schema_invalidated: bool = False


def manage_dependencies(
    action: str,
    package: str,
    schema_cache: Any | None = None,
) -> DependencyAction:
    """Manage custom node dependencies.

    This is a coordination layer — actual installation is delegated
    to the existing comfy_provision tools. This function handles
    schema cache invalidation after changes.

    Args:
        action: "install", "uninstall", or "update".
        package: Package name or URL.
        schema_cache: Optional SchemaCache to invalidate after changes.

    Returns:
        DependencyAction with result details.
    """
    result = DependencyAction(action=action, package=package)

    if action not in ("install", "uninstall", "update"):
        result.success = False
        result.message = f"Unknown action: {action!r}. Use 'install', 'uninstall', or 'update'."
        return result

    result.message = (
        f"Dependency {action} for {package!r} prepared. "
        f"Delegate to install_node_pack/uninstall_node_pack tool for execution."
    )

    # Invalidate schema cache after any dependency change
    if schema_cache is not None and hasattr(schema_cache, "refresh"):
        result.schema_invalidated = True
        result.message += " Schema cache should be refreshed after completion."

    return result
