"""Creative Metadata Layer — PNG metadata read/write/reconstruct.

Embeds artistic intent, iteration history, and session context into
PNG tEXt chunks alongside ComfyUI's native workflow data. Uses a
distinct chunk key ('comfyui_agent') that never overwrites ComfyUI's
'prompt' or 'workflow' chunks.

Schema versioned (v1) for forward compatibility.
"""

import json
import logging

from ._util import to_json, validate_path

log = logging.getLogger(__name__)

try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

METADATA_CHUNK_KEY = "comfyui_agent"

METADATA_SCHEMA_V1 = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "timestamp": {"type": "number"},
        "intent": {
            "type": "object",
            "properties": {
                "user_request": {"type": "string"},
                "interpretation": {"type": "string"},
                "style_references": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "session_context": {"type": "string"},
            },
        },
        "iterations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "iteration": {"type": "integer"},
                    "type": {
                        "type": "string",
                        "enum": ["initial", "refinement", "variation", "rollback"],
                    },
                    "trigger": {"type": "string"},
                    "patches": {"type": "array"},
                    "params": {"type": "object"},
                    "feedback": {"type": "string"},
                    "observation": {"type": "string"},
                },
            },
        },
        "accepted_iteration": {"type": "integer"},
        "session": {
            "type": "object",
            "properties": {
                "session_name": {"type": "string"},
                "workflow_path": {"type": "string"},
                "workflow_hash": {"type": "string"},
                "key_params": {"type": "object"},
                "model_combo": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "required": ["schema_version", "timestamp"],
}

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "write_image_metadata",
        "description": (
            "Write creative metadata (artistic intent, iteration history, "
            "session context) to a PNG image's tEXt chunk. Preserves "
            "ComfyUI's native 'prompt' and 'workflow' chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the PNG image.",
                },
                "metadata": {
                    "type": "object",
                    "description": (
                        "Creative metadata object. Must include 'schema_version': 1 "
                        "and 'timestamp'. Optional keys: 'intent', 'iterations', "
                        "'accepted_iteration', 'session'."
                    ),
                },
            },
            "required": ["image_path", "metadata"],
        },
    },
    {
        "name": "read_image_metadata",
        "description": (
            "Read creative metadata from a PNG image's tEXt chunk. "
            "Returns the 'comfyui_agent' metadata if present, or "
            "an empty result if not."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the PNG image.",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "reconstruct_context",
        "description": (
            "Read creative metadata from a PNG and reconstruct the full "
            "artistic context: what the artist wanted, how we got there, "
            "and what session state was active. Use this when loading an "
            "image to brief the agent on prior work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the PNG image.",
                },
            },
            "required": ["image_path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_metadata(metadata: dict) -> str | None:
    """Basic validation of metadata schema. Returns error or None."""
    if not isinstance(metadata, dict):
        return "Metadata must be a dict"
    if metadata.get("schema_version") != 1:
        return "schema_version must be 1"
    if "timestamp" not in metadata:
        return "timestamp is required"
    return None


def _read_png_metadata(image_path: str) -> tuple[dict | None, dict]:
    """Read all PNG tEXt chunks. Returns (our_metadata, all_chunks)."""
    if not _HAS_PIL:
        return None, {}

    img = Image.open(image_path)
    text_chunks = {}
    if hasattr(img, "text"):
        text_chunks = dict(img.text)
    img.close()

    our_data = None
    raw = text_chunks.get(METADATA_CHUNK_KEY)
    if raw:
        try:
            our_data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("Failed to parse %s chunk from %s", METADATA_CHUNK_KEY, image_path)

    return our_data, text_chunks


def _write_png_metadata(image_path: str, metadata: dict) -> None:
    """Write our metadata to PNG, preserving all existing chunks."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required for PNG metadata operations")

    img = Image.open(image_path)

    # Preserve existing text chunks
    existing_text = {}
    if hasattr(img, "text"):
        existing_text = dict(img.text)

    # Build new PngInfo with all existing chunks + ours
    png_info = PngInfo()
    for key, value in sorted(existing_text.items()):
        if key != METADATA_CHUNK_KEY:
            png_info.add_text(key, value)

    # Add our chunk
    png_info.add_text(METADATA_CHUNK_KEY, to_json(metadata))

    # Re-save
    img.save(image_path, pnginfo=png_info)
    img.close()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_write(tool_input: dict) -> str:
    image_path = tool_input["image_path"]
    metadata = tool_input["metadata"]

    if not _HAS_PIL:
        return to_json({"error": "Pillow not installed. Install with: pip install Pillow"})

    path_err = validate_path(image_path, must_exist=True)
    if path_err:
        return to_json({"error": path_err})

    if not image_path.lower().endswith(".png"):
        return to_json({"error": "Only PNG files are supported for metadata embedding"})

    # Validate schema
    schema_err = _validate_metadata(metadata)
    if schema_err:
        return to_json({"error": f"Invalid metadata: {schema_err}"})

    try:
        _write_png_metadata(image_path, metadata)
    except Exception as e:
        return to_json({"error": f"Failed to write metadata: {e}"})

    return to_json({
        "status": "ok",
        "image_path": image_path,
        "chunk_key": METADATA_CHUNK_KEY,
        "metadata_keys": sorted(metadata.keys()),
    })


def _handle_read(tool_input: dict) -> str:
    image_path = tool_input["image_path"]

    if not _HAS_PIL:
        return to_json({"error": "Pillow not installed. Install with: pip install Pillow"})

    path_err = validate_path(image_path, must_exist=True)
    if path_err:
        return to_json({"error": path_err})

    try:
        our_data, all_chunks = _read_png_metadata(image_path)
    except Exception as e:
        return to_json({"error": f"Failed to read metadata: {e}"})

    has_comfyui = "prompt" in all_chunks or "workflow" in all_chunks

    return to_json({
        "image_path": image_path,
        "has_creative_metadata": our_data is not None,
        "metadata": our_data,
        "has_comfyui_native": has_comfyui,
        "native_chunk_keys": sorted(k for k in all_chunks if k != METADATA_CHUNK_KEY),
    })


def _handle_reconstruct(tool_input: dict) -> str:
    """Reconstruct artistic context from PNG metadata."""
    image_path = tool_input["image_path"]

    if not _HAS_PIL:
        return to_json({"error": "Pillow not installed. Install with: pip install Pillow"})

    path_err = validate_path(image_path, must_exist=True)
    if path_err:
        return to_json({"error": path_err})

    try:
        our_data, all_chunks = _read_png_metadata(image_path)
    except Exception as e:
        return to_json({"error": f"Failed to read metadata: {e}"})

    if our_data is None:
        return to_json({
            "image_path": image_path,
            "has_context": False,
            "summary": "No creative metadata found in this image.",
            "context": None,
        })

    # Reconstruct structured context
    context: dict = {
        "schema_version": our_data.get("schema_version"),
        "created_at": our_data.get("timestamp"),
    }

    # Intent layer
    intent = our_data.get("intent")
    if intent:
        context["intent"] = {
            "what_artist_wanted": intent.get("user_request", ""),
            "how_agent_interpreted": intent.get("interpretation", ""),
            "style_references": intent.get("style_references", []),
            "session_context": intent.get("session_context", ""),
        }

    # Iteration history
    iterations = our_data.get("iterations", [])
    accepted = our_data.get("accepted_iteration")
    if iterations:
        context["iteration_history"] = {
            "total_iterations": len(iterations),
            "accepted_iteration": accepted,
            "journey": [
                {
                    "step": it.get("iteration"),
                    "type": it.get("type"),
                    "trigger": it.get("trigger", ""),
                    "feedback": it.get("feedback", ""),
                }
                for it in iterations
            ],
        }

    # Session context
    session = our_data.get("session")
    if session:
        context["session"] = {
            "name": session.get("session_name", ""),
            "workflow": session.get("workflow_path", ""),
            "key_params": session.get("key_params", {}),
            "models_used": session.get("model_combo", []),
        }

    # Build human-readable summary
    summary_parts = []
    if intent:
        req = intent.get("user_request", "")
        if req:
            summary_parts.append(f"Artist wanted: {req}")
    if iterations:
        summary_parts.append(
            f"Went through {len(iterations)} iteration(s), "
            f"accepted #{accepted if accepted is not None else 'unknown'}"
        )
    if session:
        model_combo = session.get("model_combo", [])
        if model_combo:
            summary_parts.append(f"Using: {', '.join(model_combo)}")

    summary = ". ".join(summary_parts) if summary_parts else "Metadata present but minimal."

    return to_json({
        "image_path": image_path,
        "has_context": True,
        "summary": summary,
        "context": context,
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute an image_metadata tool call."""
    try:
        if name == "write_image_metadata":
            return _handle_write(tool_input)
        elif name == "read_image_metadata":
            return _handle_read(tool_input)
        elif name == "reconstruct_context":
            return _handle_reconstruct(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
