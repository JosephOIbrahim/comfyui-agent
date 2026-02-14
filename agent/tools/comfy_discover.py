"""Discovery tools — search for custom node packs and models.

Primary data source: ComfyUI Manager's local JSON registries
(custom-node-list.json, extension-node-map.json, model-list.json).
These ship with ComfyUI-Manager and cover 4,000+ packs and 31,000+ node types.

Secondary: HuggingFace Hub API for broader model search.

Includes freshness tracking to detect stale registry data and suggest updates.
"""

import json
import time
from pathlib import Path

import httpx

from ..config import COMFYUI_URL, CUSTOM_NODES_DIR, MODELS_DIR
from ..rate_limiter import HUGGINGFACE_LIMITER
from ._util import to_json

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "search_custom_nodes",
        "description": (
            "Search for custom node packs by name/description, or find which "
            "pack provides a specific node type. Uses ComfyUI Manager's "
            "registry (4,000+ packs, 31,000+ mapped node types). "
            "Shows whether each result is already installed locally."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search term — either a pack name/keyword "
                        "(e.g. 'IPAdapter', 'upscaler', 'video') or a specific "
                        "node class_type (e.g. 'IPAdapterUnifiedLoader')."
                    ),
                },
                "by": {
                    "type": "string",
                    "enum": ["name", "node_type"],
                    "description": (
                        "Search mode: 'name' searches pack titles and descriptions, "
                        "'node_type' finds which pack provides a specific node class. "
                        "Default: 'name'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_models",
        "description": (
            "Search for models by name, type, or base model. Searches the "
            "ComfyUI Manager model registry (500+ curated entries with direct "
            "download URLs) or HuggingFace Hub for broader results. "
            "Cross-references with locally installed model files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search term — model name, type, or keyword "
                        "(e.g. 'FLUX', 'anime lora', 'controlnet depth')."
                    ),
                },
                "model_type": {
                    "type": "string",
                    "description": (
                        "Filter by model type: checkpoint, lora, vae, controlnet, "
                        "clip, upscale, embedding, etc. Matches against the "
                        "'type' field in the registry."
                    ),
                },
                "source": {
                    "type": "string",
                    "enum": ["registry", "huggingface"],
                    "description": (
                        "Where to search: 'registry' uses ComfyUI Manager's "
                        "curated list (fast, offline), 'huggingface' searches "
                        "HuggingFace Hub (broader, needs internet). Default: 'registry'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_missing_nodes",
        "description": (
            "Analyze a workflow to find which custom nodes are missing and "
            "suggest where to get them. Checks every node class_type against "
            "the live ComfyUI instance, then looks up missing ones in the "
            "ComfyUI Manager registry. Essential for importing shared workflows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to workflow JSON file. If omitted, uses the "
                        "currently loaded workflow from workflow_patch."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_install_instructions",
        "description": (
            "Get installation instructions for a custom node pack or model. "
            "Given a query and source, returns the specific commands needed to "
            "install (git clone for node packs, download path for models). "
            "Bridges the gap between discovery and actual installation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Name or identifier of the item to install — a node pack "
                        "name/URL, a model name, or a node class_type."
                    ),
                },
                "source": {
                    "type": "string",
                    "enum": ["registry", "civitai", "huggingface"],
                    "description": (
                        "Where to look up install info: 'registry' for ComfyUI "
                        "Manager node packs and models, 'civitai' for CivitAI "
                        "models, 'huggingface' for HuggingFace models. "
                        "Default: 'registry'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_registry_freshness",
        "description": (
            "Check how fresh the local discovery data is. Reports the age of "
            "ComfyUI Manager registry files (custom-node-list, extension-node-map, "
            "model-list) and local model directories. Helps decide whether to use "
            "local cache or fetch updated data from CivitAI/HuggingFace. "
            "Use refresh=true to clear the in-memory cache and force a reload."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": (
                        "If true, clears the in-memory registry cache so the next "
                        "search reloads from disk. Default: false."
                    ),
                },
            },
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Registry cache (loaded once per session from local ComfyUI Manager files)
# ---------------------------------------------------------------------------

_MANAGER_DIR = CUSTOM_NODES_DIR / "ComfyUI-Manager"

_cache: dict = {
    "custom_nodes": None,     # list[dict] from custom-node-list.json
    "extension_map": None,    # dict from extension-node-map.json
    "node_to_pack": None,     # dict[str, dict] — node_type -> pack info
    "model_list": None,       # list[dict] from model-list.json
}

# Freshness tracking — records when each cache was last loaded
_freshness: dict = {
    "custom_nodes_loaded_at": None,   # epoch timestamp
    "extension_map_loaded_at": None,
    "model_list_loaded_at": None,
}

# Staleness thresholds (seconds)
_STALE_THRESHOLD = 7 * 24 * 3600   # 7 days — registry files update infrequently
_WARN_THRESHOLD = 30 * 24 * 3600   # 30 days — strongly suggest update


def _load_custom_nodes() -> list[dict]:
    """Load and cache custom-node-list.json."""
    if _cache["custom_nodes"] is not None:
        return _cache["custom_nodes"]

    path = _MANAGER_DIR / "custom-node-list.json"
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _cache["custom_nodes"] = data.get("custom_nodes", [])
        _freshness["custom_nodes_loaded_at"] = time.time()
    except Exception:
        _cache["custom_nodes"] = []

    return _cache["custom_nodes"]


def _load_extension_map() -> dict:
    """Load and cache extension-node-map.json."""
    if _cache["extension_map"] is not None:
        return _cache["extension_map"]

    path = _MANAGER_DIR / "extension-node-map.json"
    if not path.exists():
        return {}

    try:
        _cache["extension_map"] = json.loads(path.read_text(encoding="utf-8"))
        _freshness["extension_map_loaded_at"] = time.time()
    except Exception:
        _cache["extension_map"] = {}

    return _cache["extension_map"]


def _build_node_to_pack() -> dict[str, dict]:
    """Build reverse index: node_type -> {url, title, node_types}."""
    if _cache["node_to_pack"] is not None:
        return _cache["node_to_pack"]

    ext_map = _load_extension_map()
    index = {}
    for url, entry in sorted(ext_map.items()):  # He2025: sorted iteration for deterministic collision resolution
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        node_types = entry[0] if isinstance(entry[0], list) else []
        meta = entry[1] if isinstance(entry[1], dict) else {}
        title = meta.get("title_aux", url.split("/")[-1] if "/" in url else url)

        pack_info = {"url": url, "title": title, "node_count": len(node_types)}
        for nt in node_types:
            index[nt] = pack_info

    _cache["node_to_pack"] = index
    return index


def _load_model_list() -> list[dict]:
    """Load and cache model-list.json."""
    if _cache["model_list"] is not None:
        return _cache["model_list"]

    path = _MANAGER_DIR / "model-list.json"
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _cache["model_list"] = data.get("models", [])
        _freshness["model_list_loaded_at"] = time.time()
    except Exception:
        _cache["model_list"] = []

    return _cache["model_list"]


def _is_pack_installed(reference_url: str) -> bool:
    """Check if a custom node pack is installed locally."""
    if not CUSTOM_NODES_DIR.exists():
        return False
    # Extract folder name from git URL
    folder = reference_url.rstrip("/").split("/")[-1]
    if folder.endswith(".git"):
        folder = folder[:-4]
    return (CUSTOM_NODES_DIR / folder).is_dir()


def _is_model_installed(filename: str, save_path: str) -> bool:
    """Check if a model file exists in the expected directory."""
    if not MODELS_DIR.exists():
        return False
    target = MODELS_DIR / save_path / filename
    return target.exists()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_search_custom_nodes(tool_input: dict) -> str:
    query = tool_input["query"]
    by = tool_input.get("by", "name")
    max_results = tool_input.get("max_results", 10)

    if by == "node_type":
        return _search_by_node_type(query, max_results)
    else:
        return _search_by_name(query, max_results)


def _search_by_node_type(query: str, max_results: int) -> str:
    """Find which pack provides a specific node type."""
    index = _build_node_to_pack()

    if not index:
        return to_json({
            "error": (
                "ComfyUI Manager not found. Install it from "
                "https://github.com/ltdrdata/ComfyUI-Manager"
            ),
        })

    # Exact match first
    if query in index:
        pack = index[query]
        return to_json({
            "match": "exact",
            "node_type": query,
            "pack": {
                "title": pack["title"],
                "url": pack["url"],
                "node_count": pack["node_count"],
                "installed": _is_pack_installed(pack["url"]),
            },
        })

    # Fuzzy match — case-insensitive substring, sorted for determinism
    query_lower = query.lower()
    matches = []
    seen_packs = set()
    for nt, pack in sorted(index.items()):
        if query_lower in nt.lower() and pack["url"] not in seen_packs:
            seen_packs.add(pack["url"])
            matches.append({
                "node_type": nt,
                "pack_title": pack["title"],
                "pack_url": pack["url"],
                "installed": _is_pack_installed(pack["url"]),
            })
            if len(matches) >= max_results:
                break

    if not matches:
        return to_json({
            "match": "none",
            "node_type": query,
            "message": f"No pack found providing '{query}'. Check the spelling.",
            "total_known_types": len(index),
        })

    return to_json({
        "match": "fuzzy",
        "query": query,
        "results": matches,
        "total_matches": len(matches),
    })


def _search_by_name(query: str, max_results: int) -> str:
    """Search custom node packs by title/description/author."""
    packs = _load_custom_nodes()

    if not packs:
        return to_json({
            "error": (
                "ComfyUI Manager not found. Install it from "
                "https://github.com/ltdrdata/ComfyUI-Manager"
            ),
        })

    query_lower = query.lower()
    query_words = query_lower.split()
    scored = []

    for pack in packs:
        title = pack.get("title", "")
        desc = pack.get("description", "")
        author = pack.get("author", "")
        searchable = f"{title} {desc} {author}".lower()

        # Score: all words match > some words match
        word_hits = sum(1 for w in query_words if w in searchable)
        if word_hits == 0:
            continue

        # Boost exact title match
        score = word_hits
        if query_lower in title.lower():
            score += 10

        ref = pack.get("reference", "")
        scored.append((score, {
            "title": title,
            "author": author,
            "description": desc[:200] if desc else "",
            "url": ref,
            "install_type": pack.get("install_type", ""),
            "installed": _is_pack_installed(ref) if ref else False,
        }))

    scored.sort(key=lambda x: (-x[0], x[1].get("title", "")))
    results = [r for _, r in scored[:max_results]]

    return to_json({
        "query": query,
        "results": results,
        "total_matches": len(scored),
        "showing": len(results),
    })


def _handle_search_models(tool_input: dict) -> str:
    query = tool_input["query"]
    model_type = tool_input.get("model_type")
    source = tool_input.get("source", "registry")
    max_results = tool_input.get("max_results", 10)

    if source == "huggingface":
        return _search_huggingface(query, model_type, max_results)
    else:
        return _search_model_registry(query, model_type, max_results)


def _search_model_registry(query: str, model_type: str | None, max_results: int) -> str:
    """Search ComfyUI Manager's model-list.json."""
    models = _load_model_list()

    if not models:
        return to_json({
            "error": (
                "ComfyUI Manager model list not found. Install ComfyUI Manager "
                "or try source='huggingface' for online search."
            ),
        })

    query_lower = query.lower()
    query_words = query_lower.split()
    scored = []

    for model in models:
        name = model.get("name", "")
        mtype = model.get("type", "")
        base = model.get("base", "")
        desc = model.get("description", "")
        searchable = f"{name} {mtype} {base} {desc}".lower()

        # Apply type filter if given
        if model_type and model_type.lower() not in mtype.lower():
            continue

        word_hits = sum(1 for w in query_words if w in searchable)
        if word_hits == 0:
            continue

        score = word_hits
        if query_lower in name.lower():
            score += 10

        filename = model.get("filename", "")
        save_path = model.get("save_path", "")

        scored.append((score, {
            "name": name,
            "type": mtype,
            "base": base,
            "description": desc[:200] if desc else "",
            "filename": filename,
            "url": model.get("url", ""),
            "size": model.get("size", ""),
            "save_path": save_path,
            "installed": _is_model_installed(filename, save_path) if filename and save_path else False,
        }))

    scored.sort(key=lambda x: (-x[0], x[1].get("name", "")))
    results = [r for _, r in scored[:max_results]]

    return to_json({
        "source": "registry",
        "query": query,
        "type_filter": model_type,
        "results": results,
        "total_matches": len(scored),
        "showing": len(results),
    })


def _search_huggingface(query: str, model_type: str | None, max_results: int) -> str:
    """Search HuggingFace Hub API for models."""
    params = {
        "search": query,
        "sort": "downloads",
        "direction": "-1",
        "limit": max_results,
    }

    # Map model types to HuggingFace tags
    if model_type:
        type_tags = {
            "checkpoint": "diffusers",
            "lora": "lora",
            "vae": "vae",
            "controlnet": "controlnet",
            "embedding": "textual-inversion",
        }
        tag = type_tags.get(model_type.lower())
        if tag:
            params["filter"] = tag

    if not HUGGINGFACE_LIMITER().acquire(timeout=5.0):
        return to_json({"error": "Rate limited — too many HuggingFace requests. Try again shortly."})

    try:
        with httpx.Client() as client:
            resp = client.get(
                "https://huggingface.co/api/models",
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return to_json({"error": "Could not reach HuggingFace API. Check your internet connection."})
    except httpx.HTTPStatusError as e:
        return to_json({"error": f"HuggingFace API returned {e.response.status_code}."})
    except Exception as e:
        return to_json({"error": f"HuggingFace search failed: {e}"})

    results = []
    for item in data:
        model_id = item.get("modelId", item.get("id", ""))
        results.append({
            "name": model_id,
            "author": model_id.split("/")[0] if "/" in model_id else "",
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "tags": item.get("tags", [])[:5],
            "url": f"https://huggingface.co/{model_id}",
            "last_modified": item.get("lastModified", ""),
        })

    return to_json({
        "source": "huggingface",
        "query": query,
        "type_filter": model_type,
        "results": results,
        "showing": len(results),
    })


def _handle_find_missing_nodes(tool_input: dict) -> str:
    path_str = tool_input.get("path")

    # Get workflow
    if path_str:
        path = Path(path_str)
        if not path.exists():
            return to_json({"error": f"File not found: {path_str}"})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return to_json({"error": f"Invalid JSON: {e}"})

        # Extract API-format nodes
        if "nodes" in data and isinstance(data["nodes"], list):
            prompt_data = data.get("extra", {}).get("prompt")
            if prompt_data and isinstance(prompt_data, dict):
                workflow = {
                    k: v for k, v in prompt_data.items()
                    if isinstance(v, dict) and "class_type" in v
                }
            else:
                # UI-only: extract class_types from nodes array
                workflow = {}
                for node in data["nodes"]:
                    nid = str(node.get("id", ""))
                    ntype = node.get("type", "")
                    if ntype:
                        workflow[nid] = {"class_type": ntype, "inputs": {}}
        else:
            workflow = {
                k: v for k, v in data.items()
                if isinstance(v, dict) and "class_type" in v
            }
    else:
        from .workflow_patch import get_current_workflow
        workflow = get_current_workflow()
        if workflow is None:
            return to_json({
                "error": (
                    "No workflow loaded. Either provide a 'path' or load one "
                    "with apply_workflow_patch first."
                ),
            })

    # Collect unique class_types
    class_types = set()
    for node in workflow.values():
        ct = node.get("class_type")
        if ct:
            class_types.add(ct)

    if not class_types:
        return to_json({"error": "No nodes found in workflow."})

    # Check which are available in live ComfyUI
    available = set()
    try:
        with httpx.Client() as client:
            resp = client.get(f"{COMFYUI_URL}/object_info", timeout=15.0)
            resp.raise_for_status()
            object_info = resp.json()
            available = set(object_info.keys())
    except httpx.ConnectError:
        return to_json({
            "error": "ComfyUI not reachable. Start ComfyUI to check node availability.",
        })
    except Exception as e:
        return to_json({"error": f"Could not fetch node info: {e}"})

    # Classify
    installed = sorted(class_types & available)
    missing = sorted(class_types - available)

    # Look up missing nodes in extension-node-map
    index = _build_node_to_pack()
    missing_report = []
    packs_needed = {}  # url -> {title, nodes}

    for node_type in missing:
        pack_info = index.get(node_type)
        if pack_info:
            url = pack_info["url"]
            if url not in packs_needed:
                packs_needed[url] = {
                    "title": pack_info["title"],
                    "url": url,
                    "installed": _is_pack_installed(url),
                    "missing_nodes": [],
                }
            packs_needed[url]["missing_nodes"].append(node_type)

            missing_report.append({
                "node_type": node_type,
                "pack_title": pack_info["title"],
                "pack_url": url,
                "pack_installed": _is_pack_installed(url),
            })
        else:
            missing_report.append({
                "node_type": node_type,
                "pack_title": None,
                "pack_url": None,
                "message": "Not found in ComfyUI Manager registry.",
            })

    # Summary
    if not missing:
        return to_json({
            "status": "all_installed",
            "total_node_types": len(class_types),
            "message": "All node types in this workflow are available.",
        })

    return to_json({
        "status": "missing_nodes",
        "total_node_types": len(class_types),
        "installed_count": len(installed),
        "missing_count": len(missing),
        "missing_nodes": missing_report,
        "packs_to_install": list(packs_needed.values()),
        "install_hint": (
            "Install missing packs via ComfyUI Manager or git clone into "
            f"{CUSTOM_NODES_DIR}"
        ),
    })


def _handle_get_install_instructions(tool_input: dict) -> str:
    """Get install instructions for a node pack or model."""
    query = tool_input["query"]
    source = tool_input.get("source", "registry")

    if source == "civitai":
        return to_json({
            "source": "civitai",
            "query": query,
            "instructions": (
                f"To install a CivitAI model, use the CivitAI tools to find the "
                f"model first (search_civitai), then download the file to the "
                f"appropriate directory under {MODELS_DIR}. "
                f"CivitAI models are typically .safetensors files."
            ),
            "steps": [
                f"1. Use search_civitai to find '{query}' and get the download URL",
                "2. Note the model type (checkpoint, lora, controlnet, etc.)",
                f"3. Download to {MODELS_DIR}/<type>/ directory",
                "4. Restart ComfyUI or refresh the model list",
            ],
        })

    if source == "huggingface":
        return to_json({
            "source": "huggingface",
            "query": query,
            "instructions": (
                f"To install a HuggingFace model, download the .safetensors file "
                f"to the appropriate directory under {MODELS_DIR}."
            ),
            "steps": [
                f"1. Use search_models with source='huggingface' to find '{query}'",
                "2. Note the model type and download the .safetensors file",
                f"3. Place in {MODELS_DIR}/<type>/ directory",
                "4. Restart ComfyUI or refresh the model list",
            ],
        })

    # Registry source — look up node packs and models
    # Try node pack first (by node_type)
    index = _build_node_to_pack()
    if query in index:
        pack = index[query]
        url = pack["url"]
        installed = _is_pack_installed(url)
        folder = url.rstrip("/").split("/")[-1]
        if folder.endswith(".git"):
            folder = folder[:-4]
        return to_json({
            "source": "registry",
            "type": "node_pack",
            "query": query,
            "pack_title": pack["title"],
            "installed": installed,
            "install_commands": [
                f"cd {CUSTOM_NODES_DIR}",
                f"git clone {url}",
                f"pip install -r {CUSTOM_NODES_DIR / folder / 'requirements.txt'} "
                f"(if exists)",
                "Restart ComfyUI",
            ] if not installed else [],
            "message": "Already installed." if installed else f"Clone {url} into Custom_Nodes.",
        })

    # Try node pack by name search
    packs = _load_custom_nodes()
    query_lower = query.lower()
    for pack in packs:
        title = pack.get("title", "")
        ref = pack.get("reference", "")
        if query_lower in title.lower() or query_lower in ref.lower():
            installed = _is_pack_installed(ref) if ref else False
            folder = ref.rstrip("/").split("/")[-1] if ref else ""
            if folder.endswith(".git"):
                folder = folder[:-4]
            return to_json({
                "source": "registry",
                "type": "node_pack",
                "query": query,
                "pack_title": title,
                "installed": installed,
                "install_commands": [
                    f"cd {CUSTOM_NODES_DIR}",
                    f"git clone {ref}",
                    f"pip install -r {CUSTOM_NODES_DIR / folder / 'requirements.txt'} "
                    f"(if exists)",
                    "Restart ComfyUI",
                ] if not installed and ref else [],
                "message": "Already installed." if installed else f"Clone {ref} into Custom_Nodes.",
            })

    # Try model registry
    models = _load_model_list()
    for model in models:
        name = model.get("name", "")
        if query_lower in name.lower():
            filename = model.get("filename", "")
            save_path = model.get("save_path", "")
            url = model.get("url", "")
            installed = _is_model_installed(filename, save_path) if filename and save_path else False
            return to_json({
                "source": "registry",
                "type": "model",
                "query": query,
                "model_name": name,
                "installed": installed,
                "install_commands": [
                    f"Download: {url}",
                    f"Save to: {MODELS_DIR / save_path / filename}",
                    "Restart ComfyUI or refresh model list",
                ] if not installed and url else [],
                "message": "Already installed." if installed else f"Download {filename} to {save_path}/.",
            })

    return to_json({
        "source": "registry",
        "query": query,
        "error": f"Could not find '{query}' in the registry. Try search_custom_nodes or search_models first.",
    })


def _clear_cache():
    """Clear all in-memory caches, forcing reload from disk on next access."""
    _cache["custom_nodes"] = None
    _cache["extension_map"] = None
    _cache["node_to_pack"] = None
    _cache["model_list"] = None
    _freshness["custom_nodes_loaded_at"] = None
    _freshness["extension_map_loaded_at"] = None
    _freshness["model_list_loaded_at"] = None


def _file_age_info(path: Path) -> dict:
    """Get age info for a file. Returns dict with exists, modified_at, age_s, age_human."""
    if not path.exists():
        return {"exists": False, "path": str(path)}

    mtime = path.stat().st_mtime
    age_s = time.time() - mtime
    age_days = age_s / 86400

    if age_days < 1:
        age_human = f"{age_s / 3600:.1f} hours"
    else:
        age_human = f"{age_days:.1f} days"

    if age_s > _WARN_THRESHOLD:
        status = "very_stale"
    elif age_s > _STALE_THRESHOLD:
        status = "stale"
    else:
        status = "fresh"

    return {
        "exists": True,
        "path": str(path),
        "age_seconds": round(age_s),
        "age_human": age_human,
        "status": status,
    }


def _model_dir_stats() -> dict:
    """Count model files by type in the models directory."""
    if not MODELS_DIR.exists():
        return {"exists": False, "path": str(MODELS_DIR)}

    types = {}
    for subdir in sorted(MODELS_DIR.iterdir()):
        if subdir.is_dir():
            count = sum(1 for f in subdir.iterdir() if f.is_file() and f.suffix in (
                ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
            ))
            if count > 0:
                types[subdir.name] = count

    return {
        "exists": True,
        "path": str(MODELS_DIR),
        "types": types,
        "total_files": sum(types.values()),
    }


def _handle_check_freshness(tool_input: dict) -> str:
    """Check freshness of registry data and model files."""
    if tool_input.get("refresh"):
        _clear_cache()

    # Check registry files
    registry_files = {
        "custom_node_list": _MANAGER_DIR / "custom-node-list.json",
        "extension_node_map": _MANAGER_DIR / "extension-node-map.json",
        "model_list": _MANAGER_DIR / "model-list.json",
    }

    registries = {}
    any_stale = False
    any_missing = False
    for key, path in sorted(registry_files.items()):
        info = _file_age_info(path)
        registries[key] = info
        if not info.get("exists"):
            any_missing = True
        elif info.get("status") in ("stale", "very_stale"):
            any_stale = True

    # Cache status
    cache_status = {
        "custom_nodes_cached": _cache["custom_nodes"] is not None,
        "extension_map_cached": _cache["extension_map"] is not None,
        "node_to_pack_cached": _cache["node_to_pack"] is not None,
        "model_list_cached": _cache["model_list"] is not None,
    }

    # Model directory stats
    models = _model_dir_stats()

    # Build recommendations
    recommendations = []
    if any_missing:
        recommendations.append(
            "Install ComfyUI Manager to get registry data: "
            "https://github.com/ltdrdata/ComfyUI-Manager"
        )
    if any_stale:
        recommendations.append(
            "Registry files are stale. Open ComfyUI Manager in the browser "
            "and click 'Update ComfyUI Manager' to refresh."
        )
    if tool_input.get("refresh"):
        recommendations.append("In-memory cache cleared. Next search will reload from disk.")

    return to_json({
        "registries": registries,
        "cache": cache_status,
        "models": models,
        "manager_installed": _MANAGER_DIR.exists(),
        "recommendations": recommendations,
        "refreshed": bool(tool_input.get("refresh")),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a comfy_discover tool call."""
    try:
        if name == "search_custom_nodes":
            return _handle_search_custom_nodes(tool_input)
        elif name == "search_models":
            return _handle_search_models(tool_input)
        elif name == "find_missing_nodes":
            return _handle_find_missing_nodes(tool_input)
        elif name == "get_install_instructions":
            return _handle_get_install_instructions(tool_input)
        elif name == "check_registry_freshness":
            return _handle_check_freshness(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
