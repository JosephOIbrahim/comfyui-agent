"""Consolidated macro-tools for the cognitive layer.

8 macro-tools that compose existing granular tools into
higher-level operations. The existing 108 granular tools
remain available via MCP for LLM consumers.
"""

from .analyze import analyze_workflow
from .mutate import mutate_workflow
from .query import query_environment
from .dependencies import manage_dependencies
from .execute import execute_workflow
from .compose import compose_workflow
from .series import generate_series
from .research import autoresearch

__all__ = [
    "analyze_workflow",
    "mutate_workflow",
    "query_environment",
    "manage_dependencies",
    "execute_workflow",
    "compose_workflow",
    "generate_series",
    "autoresearch",
]
