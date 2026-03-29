"""Provisioning tools -- install node packs and download models.

Bridges the gap between discovery (finding things) and usage (having them).
These are the "make it happen" tools that SuperDuper's repair and download
actions invoke.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from ..config import CUSTOM_NODES_DIR, MODELS_DIR
from ._util import to_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "install_node_pack",
        "description": (
            "Install a custom node pack by cloning its git repository into "
            "Custom_Nodes. After installing, ComfyUI must be restarted for "
            "the new nodes to be available. Use discover or find_missing_nodes "
            "to get the repository URL first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Git repository URL to clone "
                        "(e.g. 'https://github.com/author/ComfyUI-PackName')."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Optional folder name override. If omitted, uses the "
                        "repository name from the URL."
                    ),
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "download_model",
        "description": (
            "Download a model file from a URL to the correct models subdirectory. "
            "Supports checkpoints, LoRAs, VAEs, ControlNets, etc. Shows progress "
            "during download. Use discover to find the download URL first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Direct download URL for the model file.",
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Filename to save as (e.g. 'my_lora.safetensors'). "
                        "If omitted, derived from the URL."
                    ),
                },
                "model_type": {
                    "type": "string",
                    "description": (
                        "Model category directory: checkpoints, loras, vae, "
                        "controlnet, clip, clip_vision, upscale_models, "
                        "embeddings, diffusion_models, text_encoders, etc."
                    ),
                },
                "subfolder": {
                    "type": "string",
                    "description": (
                        "Optional subfolder within the model_type directory "
                        "(e.g. 'LTX2' inside loras/)."
                    ),
                },
            },
            "required": ["url", "model_type"],
        },
    },
    {
        "name": "uninstall_node_pack",
        "description": (
            "Remove a custom node pack from Custom_Nodes by renaming it with a "
            "disabled prefix. This is non-destructive -- the pack can be "
            "re-enabled by removing the prefix. ComfyUI restart required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the node pack folder in Custom_Nodes.",
                },
            },
            "required": ["name"],
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_GIT_HOSTS = frozenset([
    "github.com", "gitlab.com", "bitbucket.org",
    "huggingface.co", "codeberg.org",
])

_MODEL_EXTENSIONS = frozenset([
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx",
])


def _validate_git_url(url: str) -> str | None:
    """Validate git URL is from an allowed host. Returns error or None."""
    url_lower = url.lower().strip()
    if not url_lower.startswith("https://"):
        return "Only HTTPS URLs are allowed for security."
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url_lower)
        if parsed.hostname not in _ALLOWED_GIT_HOSTS:
            return (
                f"Host '{parsed.hostname}' not in allowed list: "
                f"{', '.join(sorted(_ALLOWED_GIT_HOSTS))}."
            )
    except Exception:
        return "Invalid URL format."
    return None


def _folder_name_from_url(url: str) -> str:
    """Extract folder name from git URL."""
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def _filename_from_url(url: str) -> str:
    """Extract filename from download URL."""
    from urllib.parse import urlparse, unquote
    parsed = urlparse(url)
    path = unquote(parsed.path)
    name = path.split("/")[-1]
    # Strip query params from name
    if "?" in name:
        name = name.split("?")[0]
    return name or "model.safetensors"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_install_node_pack(tool_input: dict) -> str:
    url = tool_input["url"].strip()
    name = tool_input.get("name") or _folder_name_from_url(url)

    # Validate URL
    err = _validate_git_url(url)
    if err:
        return to_json({"error": err})

    # Check if already installed
    target = CUSTOM_NODES_DIR / name
    if target.exists():
        return to_json({
            "error": f"Node pack '{name}' is already installed at {target}.",
            "hint": "If it's not working, try restarting ComfyUI.",
        })

    # Check Custom_Nodes dir exists
    if not CUSTOM_NODES_DIR.exists():
        return to_json({"error": f"Custom_Nodes directory not found: {CUSTOM_NODES_DIR}"})

    # Clone the repository
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(CUSTOM_NODES_DIR),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return to_json({
                "error": f"git clone failed: {stderr[:300]}",
                "hint": "Check the URL is correct and accessible.",
            })
    except FileNotFoundError:
        return to_json({
            "error": "git is not installed or not on PATH.",
            "hint": "Install git from https://git-scm.com/",
        })
    except subprocess.TimeoutExpired:
        # Clean up partial clone
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        return to_json({"error": "git clone timed out after 120 seconds."})

    # Check for requirements.txt and suggest pip install
    requirements = target / "requirements.txt"
    has_requirements = requirements.exists()

    # Install requirements if present
    pip_result = None
    if has_requirements:
        try:
            pip_proc = subprocess.run(
                ["pip", "install", "-r", str(requirements)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if pip_proc.returncode == 0:
                pip_result = "Dependencies installed successfully."
            else:
                pip_result = f"pip install had issues: {pip_proc.stderr[:200]}"
        except Exception as e:
            pip_result = f"Could not install dependencies: {e}"

    return to_json({
        "installed": name,
        "path": str(target),
        "has_requirements": has_requirements,
        "pip_result": pip_result,
        "restart_required": True,
        "message": (
            f"Node pack '{name}' installed successfully. "
            "Restart ComfyUI to load the new nodes."
        ),
    })


def _handle_download_model(tool_input: dict) -> str:
    url = tool_input["url"].strip()
    model_type = tool_input["model_type"].strip()
    subfolder = tool_input.get("subfolder", "").strip()
    filename = tool_input.get("filename") or _filename_from_url(url)

    # Validate model type directory
    type_dir = MODELS_DIR / model_type
    if subfolder:
        type_dir = type_dir / subfolder

    # Create directory if needed
    type_dir.mkdir(parents=True, exist_ok=True)

    target = type_dir / filename

    # Check if already exists
    if target.exists():
        size_gb = target.stat().st_size / (1024 ** 3)
        return to_json({
            "error": f"Model already exists: {target}",
            "size_gb": round(size_gb, 2),
            "hint": "Delete it first if you want to re-download.",
        })

    # Validate extension
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in _MODEL_EXTENSIONS:
        return to_json({
            "error": f"Unexpected file extension '{suffix}'.",
            "hint": f"Expected one of: {', '.join(sorted(_MODEL_EXTENSIONS))}",
        })

    # Download with progress
    temp_path = target.with_suffix(target.suffix + ".download")
    start_time = time.time()

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
            if response.status_code != 200:
                return to_json({
                    "error": f"Download failed: HTTP {response.status_code}",
                    "url": url,
                })

            downloaded = 0

            with open(temp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)

        # Rename temp to final
        temp_path.rename(target)
        elapsed = time.time() - start_time
        size_gb = downloaded / (1024 ** 3)
        speed_mbps = (downloaded / (1024 ** 2)) / max(elapsed, 0.1)

        return to_json({
            "downloaded": filename,
            "path": str(target),
            "model_type": model_type,
            "size_gb": round(size_gb, 2),
            "elapsed_seconds": round(elapsed, 1),
            "speed_mbps": round(speed_mbps, 1),
            "message": (
                f"Downloaded '{filename}' ({size_gb:.1f} GB) to {model_type}/. "
                f"It should be available immediately -- no restart needed."
            ),
        })

    except httpx.TimeoutException:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return to_json({
            "error": "Download timed out. The file may be very large.",
            "hint": "Try again or download manually.",
        })
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return to_json({"error": f"Download failed: {e}"})


def _handle_uninstall_node_pack(tool_input: dict) -> str:
    name = tool_input["name"].strip()

    source = CUSTOM_NODES_DIR / name
    if not source.exists():
        return to_json({
            "error": f"Node pack '{name}' not found in {CUSTOM_NODES_DIR}.",
        })

    # Non-destructive: rename with disabled prefix
    disabled_name = f"_disabled_{name}"
    target = CUSTOM_NODES_DIR / disabled_name

    if target.exists():
        return to_json({
            "error": f"Disabled version already exists: {disabled_name}",
            "hint": "Delete the disabled folder manually if you want to re-disable.",
        })

    try:
        source.rename(target)
    except Exception as e:
        return to_json({"error": f"Failed to disable: {e}"})

    return to_json({
        "disabled": name,
        "renamed_to": disabled_name,
        "path": str(target),
        "restart_required": True,
        "message": (
            f"Node pack '{name}' disabled (renamed to {disabled_name}). "
            "Restart ComfyUI to take effect. "
            "To re-enable, rename it back to remove the '_disabled_' prefix."
        ),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def handle(name: str, tool_input: dict) -> str:
    """Execute a provisioning tool call."""
    try:
        if name == "install_node_pack":
            return _handle_install_node_pack(tool_input)
        elif name == "download_model":
            return _handle_download_model(tool_input)
        elif name == "uninstall_node_pack":
            return _handle_uninstall_node_pack(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        log.error("Provisioning tool %s failed: %s", name, e, exc_info=True)
        return to_json({"error": str(e)})
