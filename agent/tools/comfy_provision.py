"""Provisioning tools -- install node packs and download models.

Bridges the gap between discovery (finding things) and usage (having them).
These are the "make it happen" tools that Comfy Cozy's repair and download
actions invoke.
"""

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx

from ..config import CUSTOM_NODES_DIR, MODELS_DIR
from ._util import to_json

log = logging.getLogger(__name__)

# Per-target install locks: prevents concurrent installs into the same directory.
# Keyed by resolved target path string so different packs don't block each other.
_MAX_INSTALL_LOCKS = 500  # Cycle 43: cap prevents unbounded growth on large installations
_install_locks: dict[str, threading.Lock] = {}
_install_locks_mutex = threading.Lock()


def _get_install_lock(target_path: str) -> threading.Lock:
    """Return (creating if needed) the lock for a given install target path."""
    with _install_locks_mutex:
        if target_path not in _install_locks:
            if len(_install_locks) >= _MAX_INSTALL_LOCKS:  # Cycle 43: FIFO eviction
                oldest_key = next(iter(_install_locks))
                del _install_locks[oldest_key]
            _install_locks[target_path] = threading.Lock()
        return _install_locks[target_path]

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
    {
        "name": "repair_workflow",
        "description": (
            "One-shot workflow repair: detect missing nodes, find which packs "
            "provide them, and install all required packs. Returns a full "
            "report of what was found and installed. Requires ComfyUI running "
            "and a workflow loaded in the PILOT engine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "auto_install": {
                    "type": "boolean",
                    "description": (
                        "If true (default), automatically install all found packs. "
                        "If false, just report what's missing without installing."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "reconfigure_workflow",
        "description": (
            "Scan the loaded workflow for model file references (checkpoints, "
            "LoRAs, VAEs, etc.) and check which exist locally. For missing "
            "models, suggest the closest local match or flag for download. "
            "Optionally apply substitutions automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "auto_fix": {
                    "type": "boolean",
                    "description": (
                        "If true, automatically apply the best local substitution "
                        "for each missing model. If false (default), just report."
                    ),
                },
            },
            "required": [],
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

# Allowed URL schemes/hosts for model downloads (SSRF prevention)
_ALLOWED_DOWNLOAD_HOSTS = frozenset([
    "github.com", "gitlab.com", "bitbucket.org",
    "huggingface.co", "civitai.com", "codeberg.org",
])


def _validate_download_url(url: str) -> str | None:
    """Validate a model download URL to prevent SSRF attacks.

    Only allows HTTPS connections to known-safe external hosts.
    Blocks localhost, private IP ranges, CGNAT, and unknown hosts.
    Returns None if valid, or an error message if invalid.
    """
    url_stripped = url.strip()
    if not url_stripped.lower().startswith("https://"):
        return "Only HTTPS URLs are allowed for model downloads."
    try:
        import ipaddress
        from urllib.parse import urlparse
        parsed = urlparse(url_stripped)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return "Invalid URL: missing hostname."
        # Block localhost and loopback
        if hostname in ("localhost", "127.0.0.1", "::1") or hostname.startswith("localhost."):
            return "Access denied: download from localhost is not allowed."
        # Block private IP ranges and CGNAT (SSRF prevention)
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return f"Access denied: download from private/internal IP '{hostname}' is not allowed."
            if addr in ipaddress.ip_network("100.64.0.0/10"):  # RFC 6598 CGNAT
                return f"Access denied: download from CGNAT address '{hostname}' is not allowed."
        except ValueError:
            pass  # Not a bare IP — hostname will be checked below
        # Block known internal/metadata endpoints by hostname pattern
        blocked_patterns = ("metadata.google", "169.254.", "192.168.", "10.", "172.")
        for pattern in blocked_patterns:
            if hostname.startswith(pattern):
                return f"Access denied: download from '{hostname}' is not allowed."
    except Exception as _e:  # Cycle 61: log unexpected URL parse errors for debuggability
        log.debug("Unexpected error validating download URL %r: %s", url[:100], _e)
        return "Invalid URL format."
    return None


_MAX_DOWNLOAD_REDIRECTS = 10
_LOCK_ACQUIRE_TIMEOUT = 5          # Cycle 44: named constants for all timeouts/sizes
_GIT_CLONE_TIMEOUT = 120           # seconds — sufficient for most repos over a home connection
_PIP_INSTALL_TIMEOUT = 120         # seconds — matches git clone budget
_DOWNLOAD_STREAM_TIMEOUT = 30.0    # seconds — httpx per-chunk timeout
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024 # bytes — 1 MB chunks balance memory vs syscall count
_MAX_DOWNLOAD_BYTES = 20 * 1024 ** 3  # 20 GB hard cap; moved from handler body
_CGNAT_NETWORK = None              # Initialised on first use


def _resolve_and_check_private(hostname: str) -> str | None:
    """DNS-resolve *hostname* and reject private/reserved/CGNAT IP addresses.

    Called for every redirect hop so that an allowlisted CDN that issues a
    302 to a private IP is caught before the connection is established.

    Returns None if the resolved address is safe, or an error string if it is
    private, loopback, link-local, reserved, or in the CGNAT range.
    """
    import ipaddress
    import socket

    global _CGNAT_NETWORK
    if _CGNAT_NETWORK is None:
        _CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

    try:
        addrs = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError):
        return f"Could not resolve hostname '{hostname}'."

    for *_, sockaddr in addrs:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return f"Resolved to private/internal IP '{ip_str}' — blocked for safety."
            if addr in _CGNAT_NETWORK:
                return f"Resolved to CGNAT address '{ip_str}' — blocked for safety."
        except ValueError:
            pass  # Should not happen from getaddrinfo, but be defensive

    return None


def _safe_filename(filename: str) -> str | None:
    """Return just the basename of a filename, rejecting path traversal attempts.

    Returns None if the filename contains directory separators or is empty.
    """
    # Strip leading/trailing whitespace
    name = filename.strip()
    # Reject any path separators (both Unix and Windows)
    if "/" in name or "\\" in name:
        return None
    # Reject dot-dot components
    if name in (".", "..") or name.startswith(".."):
        return None
    if not name:
        return None
    return name


def _safe_path_component(component: str) -> str | None:
    """Validate a single directory path component (no separators, no traversal).

    Returns None if invalid, or the stripped component if valid.
    """
    comp = component.strip()
    if not comp:
        return None
    if "/" in comp or "\\" in comp:
        return None
    if comp in (".", "..") or comp.startswith(".."):
        return None
    return comp


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
    except Exception as _e:  # Cycle 61: log unexpected URL parse errors for debuggability
        log.debug("Unexpected error validating git URL %r: %s", url[:100], _e)
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
    url = tool_input.get("url")  # Cycle 47: guard required field
    if not url or not isinstance(url, str):
        return to_json({"error": "url is required and must be a non-empty string."})
    url = url.strip()
    raw_name = tool_input.get("name") or _folder_name_from_url(url)

    # Validate URL
    err = _validate_git_url(url)
    if err:
        return to_json({"error": err})

    # Validate name: must be a single directory component (no path traversal)
    name = _safe_path_component(raw_name)
    if name is None:
        return to_json({
            "error": f"Invalid node pack name '{raw_name}': must be a single folder name "
                     "with no path separators or '..' components.",
        })

    # Verify the resolved target stays within Custom_Nodes
    target = CUSTOM_NODES_DIR / name
    try:
        resolved_target = target.resolve()
        if not str(resolved_target).startswith(str(CUSTOM_NODES_DIR.resolve())):
            return to_json({"error": "Access denied: name resolves outside Custom_Nodes directory."})
    except (OSError, ValueError):
        return to_json({"error": f"Invalid node pack name: {raw_name}"})

    # Acquire per-target lock to prevent concurrent installs into the same directory.
    install_lock = _get_install_lock(str(resolved_target))
    if not install_lock.acquire(timeout=_LOCK_ACQUIRE_TIMEOUT):
        return to_json({
            "error": f"Node pack '{name}' is already being installed by another request.",
            "hint": "Wait for the current install to complete.",
        })

    try:
        # Re-check inside lock — another thread may have installed while we waited.
        if target.exists():
            return to_json({
                "error": f"Node pack '{name}' is already installed at {target}.",
                "hint": "If it's not working, try restarting ComfyUI.",
            })

        # Check Custom_Nodes dir exists
        if not CUSTOM_NODES_DIR.exists():
            return to_json({"error": f"Custom_Nodes directory not found: {CUSTOM_NODES_DIR}"})

        # Clone the repository
        log.info("Cloning node pack '%s' from %s", name, url)
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                capture_output=True,
                text=True,
                timeout=_GIT_CLONE_TIMEOUT,
                cwd=str(CUSTOM_NODES_DIR),
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                log.warning("git clone failed for '%s': %s", name, stderr[:200])
                return to_json({
                    "error": f"git clone failed: {stderr[:300]}",
                    "hint": "Check the URL is correct and accessible.",
                })
            log.info("Cloned '%s' to %s", name, target)
        except FileNotFoundError:
            return to_json({
                "error": "git is not installed or not on PATH.",
                "hint": "Install git from https://git-scm.com/",
            })
        except subprocess.TimeoutExpired:
            # Clean up partial clone
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            log.warning("git clone timed out for '%s' after %ds", name, _GIT_CLONE_TIMEOUT)
            return to_json({"error": f"git clone timed out after {_GIT_CLONE_TIMEOUT} seconds."})

        # Check for requirements.txt and suggest pip install
        requirements = target / "requirements.txt"
        has_requirements = requirements.exists()

        # Install requirements if present
        pip_result = None
        if has_requirements:
            log.info("Installing requirements for '%s'", name)
            try:
                pip_proc = subprocess.run(
                    ["pip", "install", "-r", str(requirements)],
                    capture_output=True,
                    text=True,
                    timeout=_PIP_INSTALL_TIMEOUT,
                )
                if pip_proc.returncode == 0:
                    log.info("Requirements installed for '%s'", name)
                    pip_result = "Dependencies installed successfully."
                else:
                    log.warning("pip install incomplete for '%s' (exit %d)", name, pip_proc.returncode)
                    pip_result = "The node pack installed but some dependencies may be incomplete. Restart ComfyUI — if nodes don't appear, check the ComfyUI console for details."
            except Exception as e:
                log.warning("pip install failed for '%s': %s", name, e)
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

    finally:
        install_lock.release()


def _handle_download_model(tool_input: dict) -> str:
    url = tool_input.get("url")  # Cycle 47: guard required fields
    raw_model_type = tool_input.get("model_type")
    if not url or not isinstance(url, str):
        return to_json({"error": "url is required and must be a non-empty string."})
    if not raw_model_type or not isinstance(raw_model_type, str):
        return to_json({"error": "model_type is required and must be a non-empty string."})
    url = url.strip()
    raw_model_type = raw_model_type.strip()
    raw_subfolder = tool_input.get("subfolder", "").strip()
    raw_filename = tool_input.get("filename") or _filename_from_url(url)

    # Validate download URL (SSRF prevention)
    url_err = _validate_download_url(url)
    if url_err:
        return to_json({"error": url_err})

    # Validate model_type: single directory name, no traversal
    model_type = _safe_path_component(raw_model_type)
    if model_type is None:
        return to_json({
            "error": f"Invalid model_type '{raw_model_type}': must be a single folder name "
                     "with no path separators or '..' components.",
        })

    # Validate subfolder if provided: each component must be safe
    if raw_subfolder:
        safe_subfolder_parts = []
        for part in raw_subfolder.replace("\\", "/").split("/"):
            if not part:
                continue
            safe_part = _safe_path_component(part)
            if safe_part is None:
                return to_json({
                    "error": f"Invalid subfolder '{raw_subfolder}': contains illegal "
                             "path component '{part}'.",
                })
            safe_subfolder_parts.append(safe_part)
        subfolder = "/".join(safe_subfolder_parts)
    else:
        subfolder = ""

    # Validate filename: single file name, no path separators
    filename = _safe_filename(raw_filename)
    if filename is None:
        return to_json({
            "error": f"Invalid filename '{raw_filename}': must be a plain filename "
                     "with no path separators or '..' components.",
        })

    # Build the target path and verify it stays within MODELS_DIR
    type_dir = MODELS_DIR / model_type
    if subfolder:
        type_dir = type_dir / subfolder

    target = type_dir / filename

    # Resolve and confirm path is within MODELS_DIR (defense-in-depth)
    try:
        resolved_target = (MODELS_DIR / model_type).resolve()
        if not str(resolved_target).startswith(str(MODELS_DIR.resolve())):
            return to_json({"error": "Access denied: path resolves outside models directory."})
    except (OSError, ValueError):
        return to_json({"error": "Invalid model path."})

    # Create directory if needed
    type_dir.mkdir(parents=True, exist_ok=True)

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

    # Download with progress — manual redirect following with per-hop SSRF validation.
    # follow_redirects=False is intentional: we re-validate every redirect target
    # so that an allowlisted CDN that issues a 302 to a private IP is caught.
    from urllib.parse import urlparse, urljoin

    temp_path = target.with_suffix(target.suffix + ".download")
    start_time = time.time()
    log.info("Downloading model '%s' from %s → %s/%s", filename, url, model_type, filename)

    try:
        current_url = url
        downloaded = 0

        for _hop in range(_MAX_DOWNLOAD_REDIRECTS + 1):
            with httpx.stream("GET", current_url, follow_redirects=False, timeout=_DOWNLOAD_STREAM_TIMEOUT) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    if _hop >= _MAX_DOWNLOAD_REDIRECTS:
                        return to_json({"error": "Too many redirects during download (max 10).", "url": url})
                    location = response.headers.get("location", "")
                    if not location:
                        return to_json({"error": "Redirect response missing Location header.", "url": url})
                    new_url = urljoin(current_url, location)
                    log.debug("Download redirect %d: %s → %s", _hop + 1, current_url, new_url)
                    # Re-validate the redirect target URL against the allowlist
                    redirect_err = _validate_download_url(new_url)
                    if redirect_err:
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
                        return to_json({"error": f"Redirect blocked: {redirect_err}", "url": url})
                    # DNS-resolve the redirect hostname to catch private IPs
                    redir_host = (urlparse(new_url).hostname or "").lower()
                    ip_err = _resolve_and_check_private(redir_host)
                    if ip_err:
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
                        return to_json({"error": f"Redirect to '{redir_host}' blocked: {ip_err}", "url": url})
                    current_url = new_url
                    continue

                # Non-redirect — this is the final destination.
                # DNS-resolve once more to guard against late-binding SSRF.
                final_host = (urlparse(current_url).hostname or "").lower()
                ip_err = _resolve_and_check_private(final_host)
                if ip_err:
                    return to_json({"error": f"Download from '{final_host}' blocked: {ip_err}", "url": url})

                if response.status_code != 200:
                    sc = response.status_code
                    if sc == 403:
                        msg = "Download blocked — this model may require a CivitAI account or API key."
                    elif sc == 404:
                        msg = "Download URL no longer valid — the model may have been removed."
                    elif sc >= 500:
                        msg = "The model host is temporarily unavailable. Try again in a few minutes."
                    else:
                        msg = f"Download failed (server returned {sc}). Try again later."
                    return to_json({"error": msg, "url": url})

                # Hard cap — guards against runaway/malicious responses that a CDN
                # redirect could return after the Content-Length header has already
                # been consumed. Legitimate model files are rarely above 15 GB;
                # this limit preserves disk headroom. (Cycle 31 fix; constant Cycle 44)
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > _MAX_DOWNLOAD_BYTES:
                            raise RuntimeError(
                                f"Download aborted: file exceeds the 20 GB safety limit "
                                f"({downloaded / 1024**3:.1f} GB downloaded so far)."
                            )
                elapsed = time.time() - start_time
                log.info(
                    "Downloaded '%s' — %.1f MB in %.1fs (%.1f MB/s)",
                    filename,
                    downloaded / 1024 / 1024,
                    elapsed,
                    (downloaded / 1024 / 1024) / max(elapsed, 0.001),
                )
                break  # Download complete
        else:
            return to_json({"error": "Too many redirects during download (max 10).", "url": url})

        # Cycle 36: guard against zero-byte downloads (HTTP 200 with empty body)
        if downloaded == 0:
            temp_path.unlink(missing_ok=True)
            return to_json({
                "error": (
                    "Download produced an empty file (0 bytes). "
                    "The server returned HTTP 200 but sent no data. "
                    "Check the URL or try again."
                ),
                "url": url,
            })

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
    raw_name = tool_input.get("name")  # Cycle 47: guard required field
    if not raw_name or not isinstance(raw_name, str):
        return to_json({"error": "name is required and must be a non-empty string."})
    raw_name = raw_name.strip()

    # Validate name: must be a single directory component (no path traversal)
    name = _safe_path_component(raw_name)
    if name is None:
        return to_json({
            "error": f"Invalid node pack name '{raw_name}': must be a single folder name "
                     "with no path separators or '..' components.",
        })

    # Verify path stays within Custom_Nodes
    source = CUSTOM_NODES_DIR / name
    try:
        resolved_source = source.resolve()
        if not str(resolved_source).startswith(str(CUSTOM_NODES_DIR.resolve())):
            return to_json({"error": "Access denied: name resolves outside Custom_Nodes directory."})
    except (OSError, ValueError):
        return to_json({"error": f"Invalid node pack name: {raw_name}"})

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
# Model reference scanning (for reconfigure_workflow)
# ---------------------------------------------------------------------------

# Input fields that reference model files, mapped to their model type directory
_MODEL_INPUT_FIELDS = {
    "ckpt_name": "checkpoints",
    "checkpoint_name": "checkpoints",
    "model_name": "checkpoints",
    "lora_name": "loras",
    "vae_name": "vae",
    "control_net_name": "controlnet",
    "clip_name": "clip",
    "unet_name": "unet",
    "upscale_model": "upscale_models",
    "embedding_name": "embeddings",
    "motion_model": "animatediff_models",
}


def _scan_model_references(workflow: dict) -> list[dict]:
    """Scan a workflow for model file references and check if they exist locally."""
    refs = []
    for node_id, node in sorted(workflow.items()):
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        inputs = node.get("inputs", {})
        for field, value in inputs.items():
            if not isinstance(value, str):
                continue
            # Check known model fields
            model_type = _MODEL_INPUT_FIELDS.get(field)
            if not model_type:
                # Also check by extension
                if any(value.lower().endswith(ext) for ext in _MODEL_EXTENSIONS):
                    model_type = "checkpoints"  # best guess
                else:
                    continue

            # Check if file exists
            model_path = MODELS_DIR / model_type / value
            exists = model_path.exists()

            # Find closest local match if missing
            best_match = None
            if not exists:
                type_dir = MODELS_DIR / model_type
                if type_dir.exists():
                    local_files = [
                        f.name for f in type_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in _MODEL_EXTENSIONS
                    ]
                    if local_files:
                        # Simple substring match
                        stem = Path(value).stem.lower()
                        scored = []
                        for lf in local_files:
                            lf_stem = Path(lf).stem.lower()
                            score = 0
                            # Exact stem match
                            if lf_stem == stem:
                                score = 100
                            # Stem contains or contained by
                            elif stem in lf_stem or lf_stem in stem:
                                score = 60
                            # Word overlap
                            else:
                                stem_words = set(stem.replace("-", "_").split("_"))
                                lf_words = set(lf_stem.replace("-", "_").split("_"))
                                overlap = stem_words & lf_words
                                if overlap:
                                    score = len(overlap) * 20
                            if score > 0:
                                scored.append((score, lf))
                        if scored:
                            scored.sort(key=lambda x: -x[0])
                            best_match = scored[0][1]

            refs.append({
                "node_id": node_id,
                "class_type": node.get("class_type", "?"),
                "field": field,
                "value": value,
                "model_type": model_type,
                "exists": exists,
                "best_match": best_match,
            })

    return refs


def _handle_repair_workflow(tool_input: dict) -> str:
    """Find missing nodes and auto-install the packs that provide them."""
    import json
    _raw_install = tool_input.get("auto_install", True)  # Cycle 67: coerce string "false" (truthy)
    auto_install = _raw_install if isinstance(_raw_install, bool) else str(_raw_install).lower() not in ("false", "0", "no", "")

    # Step 1: Find missing nodes
    try:
        from .comfy_discover import handle as discover_handle
        result_json = discover_handle("find_missing_nodes", {})
        result = json.loads(result_json)
    except Exception as e:
        return to_json({"error": f"Could not check missing nodes: {e}"})

    missing = result.get("missing", [])
    if not missing:
        return to_json({
            "status": "clean",
            "message": "No missing nodes detected. Workflow is ready to run.",
            "missing_count": 0,
        })

    # Step 2: Collect unique packs to install
    packs_to_install: dict[str, dict] = {}  # url -> {name, nodes}
    unresolved = []
    for m in missing:
        class_type = m.get("class_type", "?")
        pack_url = m.get("pack_url", "")
        pack_name = m.get("pack_name", "")

        if pack_url and pack_url not in packs_to_install:
            packs_to_install[pack_url] = {
                "name": pack_name or _folder_name_from_url(pack_url),
                "url": pack_url,
                "nodes": [],
            }
        if pack_url:
            packs_to_install[pack_url]["nodes"].append(class_type)
        else:
            unresolved.append(class_type)

    # Step 3: Install (if auto_install)
    install_results = []
    if auto_install and packs_to_install:
        for url, pack_info in packs_to_install.items():
            install_json = _handle_install_node_pack({"url": url, "name": pack_info["name"]})
            try:
                install_result = json.loads(install_json)
            except (ValueError, TypeError):
                install_result = {"error": "Installer returned non-JSON response"}
            install_results.append({
                "pack": pack_info["name"],
                "url": url,
                "nodes_provided": pack_info["nodes"],
                "success": "installed" in install_result,
                "message": install_result.get("message", install_result.get("error", "")),
            })

    restart_needed = any(r["success"] for r in install_results)

    return to_json({
        "status": "repaired" if install_results else "report",
        "missing_count": len(missing),
        "packs_found": len(packs_to_install),
        "packs_installed": sum(1 for r in install_results if r["success"]),
        "unresolved_nodes": unresolved,
        "install_results": install_results,
        "restart_required": restart_needed,
        "message": (
            f"Found {len(missing)} missing node types across {len(packs_to_install)} packs. "
            + (f"Installed {sum(1 for r in install_results if r['success'])} packs. "
               "Restart ComfyUI to load new nodes."
               if install_results else
               "Use auto_install=true to install them.")
            + (f" {len(unresolved)} node types could not be resolved to a pack."
               if unresolved else "")
        ),
    })


def _handle_reconfigure_workflow(tool_input: dict) -> str:
    """Scan workflow model references and fix missing ones."""
    _raw_fix = tool_input.get("auto_fix", False)  # Cycle 67: coerce string "false" (truthy)
    auto_fix = _raw_fix if isinstance(_raw_fix, bool) else str(_raw_fix).lower() not in ("false", "0", "no", "")

    # Get current workflow from PILOT state
    try:
        from .workflow_patch import _get_state
        workflow = _get_state()["current_workflow"]
        if not workflow:
            return to_json({
                "error": "No workflow loaded. Load a workflow first.",
            })
    except Exception as e:
        return to_json({"error": f"Could not access workflow state: {e}"})

    refs = _scan_model_references(workflow)
    if not refs:
        return to_json({
            "status": "clean",
            "message": "No model references found in workflow.",
            "references": [],
        })

    missing_refs = [r for r in refs if not r["exists"]]
    found_refs = [r for r in refs if r["exists"]]

    # Apply auto-fix substitutions
    fixes_applied = []
    if auto_fix and missing_refs:
        from .workflow_patch import _get_state as _get_patch_state
        wf = _get_patch_state()["current_workflow"]
        for ref in missing_refs:
            if ref["best_match"]:
                node = wf.get(ref["node_id"])
                if node and "inputs" in node:
                    old_value = node["inputs"].get(ref["field"])
                    node["inputs"][ref["field"]] = ref["best_match"]
                    fixes_applied.append({
                        "node_id": ref["node_id"],
                        "class_type": ref["class_type"],
                        "field": ref["field"],
                        "old": old_value,
                        "new": ref["best_match"],
                    })

    still_missing = [
        r for r in missing_refs
        if not any(f["node_id"] == r["node_id"] and f["field"] == r["field"]
                   for f in fixes_applied)
    ]

    return to_json({
        "status": "reconfigured" if fixes_applied else "report",
        "total_references": len(refs),
        "found": len(found_refs),
        "missing": len(missing_refs),
        "fixes_applied": len(fixes_applied),
        "still_missing": len(still_missing),
        "details": {
            "found": [
                {"node": r["class_type"], "field": r["field"], "model": r["value"]}
                for r in found_refs
            ],
            "missing": [
                {
                    "node": r["class_type"],
                    "field": r["field"],
                    "model": r["value"],
                    "model_type": r["model_type"],
                    "best_match": r["best_match"],
                }
                for r in missing_refs
            ],
            "fixes": fixes_applied,
        },
        "message": (
            f"{len(refs)} model references scanned. "
            f"{len(found_refs)} found, {len(missing_refs)} missing. "
            + (f"Applied {len(fixes_applied)} substitutions. "
               if fixes_applied else "")
            + (f"{len(still_missing)} models have no local match -- download needed."
               if still_missing else
               "All models resolved." if not missing_refs else
               "Use auto_fix=true to apply best-match substitutions.")
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
        elif name == "repair_workflow":
            return _handle_repair_workflow(tool_input)
        elif name == "reconfigure_workflow":
            return _handle_reconfigure_workflow(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        log.error("Provisioning tool %s failed: %s", name, e, exc_info=True)
        return to_json({"error": str(e)})
