"""Schema System — structured output contracts for specialist agents.

Re-exports the loader and generator functions so callers can do::

    from agent.schemas import load_schema, validate_output
"""

from .generator import infer_schema_from_example, write_schema
from .loader import clear_cache, list_schemas, load_schema, validate_output

__all__ = [
    "clear_cache",
    "infer_schema_from_example",
    "list_schemas",
    "load_schema",
    "validate_output",
    "write_schema",
]
