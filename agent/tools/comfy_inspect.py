"""Filesystem inspection tools.

Reads the local ComfyUI installation to discover installed
custom nodes, models, and their on-disk details — without
needing ComfyUI to be running.
"""

import json
from pathlib import Path

from ..config import CUSTOM_NODES_DIR, MODELS_DIR
from ._util import to_json

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "list_custom_nodes",
        "description": (
            "List all installed ComfyUI custom node packs by scanning "
            "the Custom_Nodes directory. Returns pack names, whether they "
            "have a README or requirements.txt, and node count if detectable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name_filter": {
                    "type": "string",
                    "description": "Optional substring filter on pack name (case-insensitive).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_models",
        "description": (
            "List model files in a specific model subdirectory "
            "(e.g. 'checkpoints', 'loras', 'controlnet', 'vae'). "
            "Returns filenames and sizes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": (
                        "Subdirectory under models/ to scan. "
                        "Common: checkpoints, loras, vae, controlnet, "
                        "clip, upscale_models, embeddings, unet, diffusion_models."
                    ),
                },
            },
            "required": ["model_type"],
        },
    },
    {
        "name": "get_models_summary",
        "description": (
            "Get a summary of all model directories: "
            "which types exist and how many files each contains."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_node_source",
        "description": (
            "Read the source code of a custom node pack's entry point "
            "(__init__.py) to understand what nodes it registers. "
            "Useful for discovering NODE_CLASS_MAPPINGS."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_pack": {
                    "type": "string",
                    "description": "Name of the custom node pack directory.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to read (default 200).",
                },
            },
            "required": ["node_pack"],
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}


def _human_size(nbytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _is_node_pack(path: Path) -> bool:
    """Check if a directory looks like a custom node pack."""
    if not path.is_dir():
        return False
    name = path.name
    # Skip __pycache__, hidden dirs, temp dirs
    if name.startswith((".", "__", "tmp")):
        return False
    # Must have Python files
    return any(path.glob("*.py"))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_list_custom_nodes(tool_input: dict) -> str:
    name_filter = (tool_input.get("name_filter") or "").lower()

    if not CUSTOM_NODES_DIR.exists():
        return to_json({
            "error": f"Custom nodes directory not found: {CUSTOM_NODES_DIR}",
        })

    packs = []
    for item in sorted(CUSTOM_NODES_DIR.iterdir()):
        if not _is_node_pack(item):
            continue
        if name_filter and name_filter not in item.name.lower():
            continue

        info = {"name": item.name}

        # Check for README
        for readme_name in ("README.md", "readme.md", "README.rst"):
            if (item / readme_name).exists():
                info["has_readme"] = True
                break

        # Check for requirements
        info["has_requirements"] = (item / "requirements.txt").exists()

        # Try to count registered nodes from __init__.py
        init_file = item / "__init__.py"
        if init_file.exists():
            try:
                content = init_file.read_text(encoding="utf-8", errors="replace")
                if "NODE_CLASS_MAPPINGS" in content:
                    info["registers_nodes"] = True
            except Exception:
                pass

        packs.append(info)

    return to_json({
        "directory": str(CUSTOM_NODES_DIR),
        "count": len(packs),
        "packs": packs,
    })


def _handle_list_models(tool_input: dict) -> str:
    model_type = tool_input["model_type"]
    model_dir = MODELS_DIR / model_type

    if not model_dir.exists():
        # List available types
        available = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
        return to_json({
            "error": f"Model directory '{model_type}' not found.",
            "available_types": sorted(available),
        })

    models = []
    for f in sorted(model_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in _MODEL_EXTENSIONS:
            continue
        # Path relative to model_dir
        rel = f.relative_to(model_dir)
        models.append({
            "name": str(rel),
            "size": _human_size(f.stat().st_size),
            "size_bytes": f.stat().st_size,
        })

    return to_json({
        "model_type": model_type,
        "directory": str(model_dir),
        "count": len(models),
        "models": models,
    })


def _handle_get_models_summary() -> str:
    if not MODELS_DIR.exists():
        return to_json({"error": f"Models directory not found: {MODELS_DIR}"})

    summary = {}
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        count = sum(
            1 for f in d.rglob("*")
            if f.is_file() and f.suffix.lower() in _MODEL_EXTENSIONS
        )
        if count > 0:
            summary[d.name] = count

    return to_json({
        "directory": str(MODELS_DIR),
        "types": summary,
        "total_types": len(summary),
    })


def _handle_read_node_source(tool_input: dict) -> str:
    node_pack = tool_input["node_pack"]
    max_lines = tool_input.get("max_lines", 200)

    pack_dir = CUSTOM_NODES_DIR / node_pack
    if not pack_dir.exists():
        # Suggest similar names
        available = [
            d.name for d in CUSTOM_NODES_DIR.iterdir()
            if _is_node_pack(d) and node_pack.lower() in d.name.lower()
        ]
        return to_json({
            "error": f"Node pack '{node_pack}' not found.",
            "similar": available[:10],
        })

    init_file = pack_dir / "__init__.py"
    if not init_file.exists():
        # Try to find main Python files
        py_files = sorted(pack_dir.glob("*.py"))
        return to_json({
            "error": f"No __init__.py in '{node_pack}'.",
            "python_files": [f.name for f in py_files[:20]],
        })

    try:
        lines = init_file.read_text(encoding="utf-8", errors="replace").splitlines()
        truncated = len(lines) > max_lines
        source = "\n".join(lines[:max_lines])
    except Exception as e:
        return to_json({"error": f"Could not read file: {e}"})

    return to_json({
        "pack": node_pack,
        "file": str(init_file),
        "total_lines": len(lines),
        "truncated": truncated,
        "source": source,
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a comfy_inspect tool call."""
    try:
        if name == "list_custom_nodes":
            return _handle_list_custom_nodes(tool_input)
        elif name == "list_models":
            return _handle_list_models(tool_input)
        elif name == "get_models_summary":
            return _handle_get_models_summary()
        elif name == "read_node_source":
            return _handle_read_node_source(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
