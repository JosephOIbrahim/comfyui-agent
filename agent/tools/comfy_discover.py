"""Discovery tools — search for custom node packs and models.

Primary data source: ComfyUI Manager's local JSON registries
(custom-node-list.json, extension-node-map.json, model-list.json).
These ship with ComfyUI-Manager and cover 4,000+ packs and 31,000+ node types.

Secondary: HuggingFace Hub API for broader model search.

Includes freshness tracking to detect stale registry data and suggest updates.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path

import httpx

from ..config import COMFYUI_URL, CUSTOM_NODES_DIR, MODELS_DIR, MODEL_CATALOG_PATH
from ..rate_limiter import HUGGINGFACE_LIMITER
from ._util import to_json, validate_path

log = logging.getLogger(__name__)

# Pre-compiled pattern for UUID-style component instance node class_types.
# These are not real node classes and should be excluded from missing-node checks.
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _check_deprecated(class_type: str) -> dict | None:
    """Check if a node class_type has a known replacement.

    Returns replacement info dict or None. Best-effort — does not
    fail if node_replacement module isn't available or endpoint is down.
    """
    try:
        from .node_replacement import _fetch_replacements
        replacements = _fetch_replacements()
        if class_type in replacements:
            reps = replacements[class_type]
            if reps:
                return {
                    "deprecated": True,
                    "replacement": reps[0].get("new_node_id", "unknown"),
                    "all_replacements": [r.get("new_node_id") for r in reps],
                }
    except Exception as e:
        log.warning("_check_deprecated failed for %s: %s", class_type, e)
    return None

# ---------------------------------------------------------------------------
# Partner Node registry — officially supported by Comfy-Org + provider
# Update quarterly or when new partnerships are announced.
# ---------------------------------------------------------------------------

PARTNER_NODES: dict[str, dict] = {
    "Hunyuan3D": {
        "provider": "Tencent",
        "capabilities": "text/image/sketch to production 3D mesh",
        "best_for": "final production assets, high-quality 3D generation",
        "url": "https://github.com/Tencent/Hunyuan3D-2",
        "keywords": ["hunyuan3d", "hunyuan 3d", "tencent"],
    },
    "Meshy": {
        "provider": "Meshy",
        "capabilities": "AI mesh generation (stylized assets)",
        "best_for": "game-ready stylized assets, fast stylized generation",
        "url": "https://github.com/meshyai/comfyui-meshy",
        "keywords": ["meshy"],
    },
    "Tripo": {
        "provider": "Tripo",
        "capabilities": "fast 3D prototyping and iteration",
        "best_for": "rapid prototyping, quick iterations",
        "url": "https://github.com/VAST-AI-Research/ComfyUI-Tripo",
        "keywords": ["tripo", "vast-ai"],
    },
    "Rodin": {
        "provider": "Deemos",
        "capabilities": "high-quality 3D generation with geometric detail",
        "best_for": "realistic/detailed models, high geometric fidelity",
        "url": "https://github.com/Deemos-Technology/ComfyUI-Rodin",
        "keywords": ["rodin", "deemos"],
    },
}

# Core node packs that ship with ComfyUI or are maintained by Comfy-Org
_CORE_URLS = frozenset([
    "https://github.com/comfyanonymous/ComfyUI",
])


def _get_source_tier(title: str, url: str) -> str:
    """Determine source tier: 'core', 'partner', or 'community'."""
    url_lower = (url or "").lower()
    title_lower = (title or "").lower()

    # Check core
    if url_lower in _CORE_URLS or "comfyanonymous" in url_lower:
        return "core"

    # Check partner nodes
    for partner_info in PARTNER_NODES.values():
        if partner_info["url"].lower() == url_lower:
            return "partner"
        if any(kw in title_lower for kw in partner_info["keywords"]):
            return "partner"

    return "community"

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "discover",
        "description": (
            "Unified search across local model catalog, ComfyUI Manager registry, "
            "CivitAI, and HuggingFace for custom node packs and models. Returns "
            "results ranked by relevance with installed status. Use this for any "
            "'find me X' or 'what's available for Y' question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for — model name, node pack, keyword, "
                        "or specific node class_type "
                        "(e.g. 'IPAdapter', 'anime LoRA for SDXL', 'depth controlnet')."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": ["nodes", "models", "all"],
                    "description": (
                        "What to search: 'nodes' for custom node packs, "
                        "'models' for checkpoints/LoRAs/etc, 'all' for both. "
                        "Default: 'all'."
                    ),
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["registry", "civitai", "huggingface"],
                    },
                    "description": (
                        "Which sources to search. Default: all applicable "
                        "(registry for nodes, registry+civitai+huggingface for models)."
                    ),
                },
                "model_type": {
                    "type": "string",
                    "description": (
                        "Filter models by type: checkpoint, lora, vae, controlnet, "
                        "embedding, upscaler. Ignored for node searches."
                    ),
                },
                "base_model": {
                    "type": "string",
                    "description": (
                        "Filter by base model: 'sd15', 'sdxl', 'flux', 'sd3', 'pony'. "
                        "Applied to CivitAI; ignored for registry/HuggingFace."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["most_downloaded", "highest_rated", "newest"],
                    "description": (
                        "CivitAI sort order. Default: 'most_downloaded'. "
                        "Ignored for non-CivitAI sources."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results per source (default 5).",
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

# Single lock protecting all _cache and _freshness writes. Read-heavy workload
# (each key written once per session), so a plain Lock is sufficient.
_cache_lock = threading.Lock()


def _load_custom_nodes() -> list[dict]:
    """Load and cache custom-node-list.json. Thread-safe: double-checked lock."""
    if _cache["custom_nodes"] is not None:
        return _cache["custom_nodes"]

    with _cache_lock:
        if _cache["custom_nodes"] is not None:  # Re-check after acquiring lock
            return _cache["custom_nodes"]

        path = _MANAGER_DIR / "custom-node-list.json"
        if not path.exists():
            _cache["custom_nodes"] = []
            return _cache["custom_nodes"]

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _cache["custom_nodes"] = data.get("custom_nodes", [])
            _freshness["custom_nodes_loaded_at"] = time.time()
        except Exception:
            log.warning("Failed to load custom-node-list.json — registry will be empty", exc_info=True)
            _cache["custom_nodes"] = []

        return _cache["custom_nodes"]


def _load_extension_map() -> dict:
    """Load and cache extension-node-map.json. Thread-safe: double-checked lock."""
    if _cache["extension_map"] is not None:
        return _cache["extension_map"]

    with _cache_lock:
        if _cache["extension_map"] is not None:
            return _cache["extension_map"]

        path = _MANAGER_DIR / "extension-node-map.json"
        if not path.exists():
            _cache["extension_map"] = {}
            return _cache["extension_map"]

        try:
            _cache["extension_map"] = json.loads(path.read_text(encoding="utf-8"))
            _freshness["extension_map_loaded_at"] = time.time()
        except Exception:
            log.warning("Failed to load extension-node-map.json — node-to-pack index will be empty", exc_info=True)
            _cache["extension_map"] = {}

        return _cache["extension_map"]


def _build_node_to_pack() -> dict[str, dict]:
    """Build reverse index: node_type -> {url, title, node_types}. Thread-safe.

    Extension map is loaded BEFORE acquiring _cache_lock to avoid deadlock
    (_load_extension_map also acquires _cache_lock; Lock is not re-entrant).
    """
    if _cache["node_to_pack"] is not None:
        return _cache["node_to_pack"]

    # Load dependency outside the lock to prevent deadlock.
    ext_map = _load_extension_map()

    with _cache_lock:
        if _cache["node_to_pack"] is not None:  # Re-check: another thread may have built it
            return _cache["node_to_pack"]

        index = {}
        for url, entry in sorted(ext_map.items()):  # He2025: sorted for deterministic collision resolution
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
    """Load and cache model-list.json. Thread-safe: double-checked lock."""
    if _cache["model_list"] is not None:
        return _cache["model_list"]

    with _cache_lock:
        if _cache["model_list"] is not None:
            return _cache["model_list"]

        path = _MANAGER_DIR / "model-list.json"
        if not path.exists():
            _cache["model_list"] = []
            return _cache["model_list"]

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _cache["model_list"] = data.get("models", [])
            _freshness["model_list_loaded_at"] = time.time()
        except Exception:
            log.warning("Failed to load model-list.json — model registry will be empty", exc_info=True)
            _cache["model_list"] = []

        return _cache["model_list"]


# ---------------------------------------------------------------------------
# Local model catalog (COMFYUI_Database/model_catalog.json)
# ---------------------------------------------------------------------------

_catalog_cache: list[dict] | None = None


def _load_model_catalog() -> list[dict]:
    """Load model_catalog.json — local enriched model metadata. Thread-safe."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    with _cache_lock:
        if _catalog_cache is not None:  # Re-check after lock
            return _catalog_cache

        if not MODEL_CATALOG_PATH.exists():
            _catalog_cache = []
            return _catalog_cache

        try:
            data = json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))
            # Flatten categories into a single list
            models = []
            for category, info in data.get("categories", {}).items():
                for model in info.get("models", []):
                    model["_category"] = category
                    models.append(model)
            _catalog_cache = models
        except Exception:
            log.warning("Failed to load model_catalog.json — local catalog will be empty", exc_info=True)
            _catalog_cache = []

        return _catalog_cache


def _search_catalog_unified(
    query: str, model_type: str | None, max_results: int,
) -> list[dict]:
    """Search model_catalog.json for locally known models."""
    models = _load_model_catalog()
    if not models:
        return []

    query_lower = query.lower()
    query_words = query_lower.split()
    scored = []

    for model in models:
        filename = model.get("filename", "")
        category = model.get("_category", "")
        optimized = " ".join(model.get("optimized", []))
        searchable = f"{filename} {category} {optimized}".lower()

        if model_type and model_type.lower() not in category.lower():
            continue

        word_hits = sum(1 for w in query_words if w in searchable)
        if word_hits == 0:
            continue

        score = word_hits / max(len(query_words), 1)
        if query_lower in filename.lower():
            score = min(score + 0.5, 1.0)

        # Catalog models are always installed (that's how they got cataloged)
        size_gb = model.get("size", 0)
        scored.append((score, _normalize_result(
            name=filename,
            result_type="model",
            source="local_catalog",
            relevance_score=score,
            installed=True,
            url="",
            model_type=category or None,
            size_gb=round(size_gb, 2) if size_gb else None,
            optimization=optimized or None,
        )))

    scored.sort(key=lambda x: (-x[0], x[1]["name"]))
    return [r for _, r in scored[:max_results]]


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
# Unified discover — helpers
# ---------------------------------------------------------------------------

def _normalize_result(
    name: str,
    result_type: str,
    source: str,
    relevance_score: float,
    installed: bool,
    url: str,
    **extra,
) -> dict:
    """Common result schema for unified discovery."""
    result = {
        "installed": installed,
        "name": name,
        "relevance_score": round(relevance_score, 3),
        "source": source,
        "source_tier": _get_source_tier(name, url),
        "type": result_type,
        "url": url,
    }
    # He2025: sorted keys via sorted extra insertion
    for k in sorted(extra):
        if extra[k] is not None:
            result[k] = extra[k]
    return result


def _search_nodes_unified(query: str, max_results: int) -> list[dict]:
    """Search custom node packs, return normalized results."""
    packs = _load_custom_nodes()
    if not packs:
        return []

    query_lower = query.lower()
    query_words = query_lower.split()
    scored = []

    for pack in packs:
        title = pack.get("title", "")
        desc = pack.get("description", "")
        author = pack.get("author", "")
        searchable = f"{title} {desc} {author}".lower()

        word_hits = sum(1 for w in query_words if w in searchable)
        if word_hits == 0:
            continue

        score = word_hits / max(len(query_words), 1)
        if query_lower in title.lower():
            score = min(score + 0.5, 1.0)

        ref = pack.get("reference", "")
        scored.append((score, _normalize_result(
            name=title,
            result_type="node_pack",
            source="registry",
            relevance_score=score,
            installed=_is_pack_installed(ref) if ref else False,
            url=ref,
            author=author,
            description=desc[:200] if desc else None,
        )))

    scored.sort(key=lambda x: (-x[0], x[1]["name"]))
    return [r for _, r in scored[:max_results]]


def _search_models_unified(
    query: str, model_type: str | None, max_results: int,
) -> list[dict]:
    """Search model registry, return normalized results."""
    models = _load_model_list()
    if not models:
        return []

    query_lower = query.lower()
    query_words = query_lower.split()
    scored = []

    for model in models:
        name = model.get("name", "")
        mtype = model.get("type", "")
        base = model.get("base", "")
        desc = model.get("description", "")
        searchable = f"{name} {mtype} {base} {desc}".lower()

        if model_type and model_type.lower() not in mtype.lower():
            continue

        word_hits = sum(1 for w in query_words if w in searchable)
        if word_hits == 0:
            continue

        score = word_hits / max(len(query_words), 1)
        if query_lower in name.lower():
            score = min(score + 0.5, 1.0)

        filename = model.get("filename", "")
        save_path = model.get("save_path", "")

        scored.append((score, _normalize_result(
            name=name,
            result_type="model",
            source="registry",
            relevance_score=score,
            installed=_is_model_installed(filename, save_path) if filename and save_path else False,
            url=model.get("url", ""),
            base_model=base or None,
            description=desc[:200] if desc else None,
            model_type=mtype or None,
        )))

    scored.sort(key=lambda x: (-x[0], x[1]["name"]))
    return [r for _, r in scored[:max_results]]


def _search_civitai_unified(
    query: str,
    model_type: str | None,
    base_model: str | None,
    sort: str,
    max_results: int,
) -> tuple[list[dict], str | None]:
    """Search CivitAI, return (normalized_results, error_or_None)."""
    from .civitai_api import _handle_search_civitai

    try:
        raw = json.loads(_handle_search_civitai({
            "query": query,
            "model_type": model_type,
            "base_model": base_model,
            "sort": sort,
            "max_results": max_results,
        }))
    except (ValueError, TypeError) as _e:  # Cycle 43: guard non-JSON from civitai handler
        return [], f"CivitAI search returned non-JSON: {_e}"
    if "error" in raw:
        return [], raw["error"]

    results = []
    for item in raw.get("results", []):
        # Normalize relevance from downloads (log scale)
        downloads = item.get("downloads", 0)
        # Guard: API may return downloads as string or None. (Cycle 33 fix)
        if not isinstance(downloads, (int, float)):
            try:
                downloads = float(downloads)
            except (ValueError, TypeError):
                downloads = 0
        rel = min(1.0, (downloads / 1_000_000) ** 0.3) if downloads > 0 else 0.1
        results.append(_normalize_result(
            name=item.get("name", ""),
            result_type="model",
            source="civitai",
            relevance_score=rel,
            installed=item.get("installed", False),
            url=item.get("url", ""),
            base_model=item.get("base_model") or None,
            downloads=downloads or None,
            model_type=item.get("type") or None,
            rating=item.get("rating") or None,
        ))
    return results, None


def _search_hf_unified(
    query: str, model_type: str | None, max_results: int,
) -> tuple[list[dict], str | None]:
    """Search HuggingFace, return (normalized_results, error_or_None)."""
    try:
        raw = json.loads(_search_huggingface(query, model_type, max_results))
    except (ValueError, TypeError) as _e:  # Cycle 43: guard non-JSON from HuggingFace handler
        return [], f"HuggingFace search returned non-JSON: {_e}"
    if "error" in raw:
        return [], raw["error"]

    results = []
    for item in raw.get("results", []):
        downloads = item.get("downloads", 0)
        # Guard: API may return downloads as string or None. (Cycle 33 fix)
        if not isinstance(downloads, (int, float)):
            try:
                downloads = float(downloads)
            except (ValueError, TypeError):
                downloads = 0
        rel = min(1.0, (downloads / 1_000_000) ** 0.3) if downloads > 0 else 0.1
        results.append(_normalize_result(
            name=item.get("name", ""),
            result_type="model",
            source="huggingface",
            relevance_score=rel,
            installed=False,
            url=item.get("url", ""),
            author=item.get("author") or None,
            downloads=downloads or None,
        ))
    return results, None


def _deduplicate(results: list[dict]) -> list[dict]:
    """Deduplicate by lowercased name. Keep highest-scored, note also_found_on."""
    seen: dict[str, dict] = {}  # lowercase name -> result dict
    for r in results:
        key = r["name"].lower().strip()
        if key in seen:
            existing = seen[key]
            # Track sources
            if r["source"] not in existing.get("also_found_on", []):
                existing.setdefault("also_found_on", []).append(r["source"])
                existing["also_found_on"] = sorted(existing["also_found_on"])
            # Keep higher score
            if r["relevance_score"] > existing["relevance_score"]:
                also = existing.get("also_found_on", [])
                also_with_old = sorted(set(also) | {existing["source"]})
                r["also_found_on"] = [s for s in also_with_old if s != r["source"]]
                if r["also_found_on"]:
                    r["also_found_on"] = sorted(r["also_found_on"])
                else:
                    r.pop("also_found_on", None)
                seen[key] = r
        else:
            seen[key] = r
    return list(seen.values())


_TIER_RANK = {"core": 0, "partner": 1, "community": 2}


def _rank_results(results: list[dict]) -> list[dict]:
    """Rank: installed first, tier (core>partner>community), relevance, name."""
    return sorted(
        results,
        key=lambda r: (
            not r.get("installed", False),            # installed first
            _TIER_RANK.get(r.get("source_tier", "community"), 2),  # tier rank
            -r.get("relevance_score", 0),              # higher score first
            r.get("name", ""),                         # alphabetical tiebreaker
        ),
    )


def _annotate_provision_hints(results: list[dict]) -> None:
    """Add provision_hint to uninstalled model results.

    When a model is not installed, the hint tells the caller they can
    register it in the CognitiveWorkflowStage and download it via
    provision_download. Modifies results in place.
    """
    for r in results:
        if r.get("type") == "model" and not r.get("installed", False):
            url = r.get("url", "")
            name = r.get("name", "unknown")
            model_type = r.get("model_type", r.get("save_path", "checkpoints"))
            if url:
                r["provision_hint"] = {
                    "action": "provision_download",
                    "description": (
                        f"Register '{name}' in the stage model registry, "
                        f"then call provision_download to fetch it."
                    ),
                    "register_args": {
                        "model_type": model_type,
                        "filename": name,
                        "source_url": url,
                    },
                }


def _handle_discover(tool_input: dict) -> str:
    """Unified discovery across registry, CivitAI, and HuggingFace."""
    query = tool_input.get("query")  # Cycle 46: guard required field
    if not query or not isinstance(query, str):
        return to_json({"error": "query is required and must be a non-empty string."})
    category = tool_input.get("category", "all")
    sources = tool_input.get("sources")
    model_type = tool_input.get("model_type")
    base_model = tool_input.get("base_model")
    sort = tool_input.get("sort", "most_downloaded")
    max_results = tool_input.get("max_results", 5)

    all_results: list[dict] = []
    sources_searched: list[str] = []
    errors: list[dict] = []

    # Determine which sources to search
    if sources is None:
        search_registry = True
        search_civitai = category in ("models", "all")
        search_hf = category in ("models", "all")
    else:
        search_registry = "registry" in sources
        search_civitai = "civitai" in sources
        search_hf = "huggingface" in sources

    # 1. Registry nodes (if applicable)
    if search_registry and category in ("nodes", "all"):
        node_results = _search_nodes_unified(query, max_results)
        all_results.extend(node_results)
        if "registry_nodes" not in sources_searched:
            sources_searched.append("registry_nodes")

    # 2. Registry models (if applicable)
    if search_registry and category in ("models", "all"):
        model_results = _search_models_unified(query, model_type, max_results)
        all_results.extend(model_results)
        if "registry_models" not in sources_searched:
            sources_searched.append("registry_models")

    # 3. Local model catalog (enriched metadata)
    if search_registry and category in ("models", "all"):
        catalog_results = _search_catalog_unified(query, model_type, max_results)
        if catalog_results:
            all_results.extend(catalog_results)
            if "local_catalog" not in sources_searched:
                sources_searched.append("local_catalog")

    # 4. CivitAI (models only)
    if search_civitai and category in ("models", "all"):
        civitai_results, civitai_err = _search_civitai_unified(
            query, model_type, base_model, sort, max_results,
        )
        all_results.extend(civitai_results)
        sources_searched.append("civitai")
        if civitai_err:
            errors.append({"source": "civitai", "error": civitai_err})

    # 5. HuggingFace (models only)
    if search_hf and category in ("models", "all"):
        hf_results, hf_err = _search_hf_unified(query, model_type, max_results)
        all_results.extend(hf_results)
        sources_searched.append("huggingface")
        if hf_err:
            errors.append({"source": "huggingface", "error": hf_err})

    # Deduplicate and rank
    deduped = _deduplicate(all_results)
    ranked = _rank_results(deduped)

    # Annotate uninstalled models with provision hints
    _annotate_provision_hints(ranked)

    response: dict = {
        "query": query,
        "category": category,
        "results": ranked,
        "sources_searched": sorted(sources_searched),
        "total": len(ranked),
    }
    if errors:
        response["errors"] = errors

    return to_json(response)


# ---------------------------------------------------------------------------
# Internal handlers (kept for backward compat with internal callers)
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
        result = {
            "match": "exact",
            "node_type": query,
            "pack": {
                "title": pack["title"],
                "url": pack["url"],
                "node_count": pack["node_count"],
                "installed": _is_pack_installed(pack["url"]),
            },
        }
        dep_info = _check_deprecated(query)
        if dep_info:
            result["deprecated"] = True
            result["replacement_available"] = dep_info["replacement"]
        return to_json(result)

    # Fuzzy match — case-insensitive substring, sorted for determinism
    query_lower = query.lower()
    matches = []
    seen_packs = set()
    for nt, pack in sorted(index.items()):
        if query_lower in nt.lower() and pack["url"] not in seen_packs:
            seen_packs.add(pack["url"])
            match_item = {
                "node_type": nt,
                "pack_title": pack["title"],
                "pack_url": pack["url"],
                "installed": _is_pack_installed(pack["url"]),
            }
            dep_info = _check_deprecated(nt)
            if dep_info:
                match_item["deprecated"] = True
                match_item["replacement_available"] = dep_info["replacement"]
            matches.append(match_item)
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
            "3d": "image-to-3d",
            "image-to-3d": "image-to-3d",
            "text-to-3d": "text-to-3d",
            "text-to-speech": "text-to-speech",
            "tts": "text-to-speech",
            "audio": "text-to-audio",
            "text-to-audio": "text-to-audio",
            "text-to-video": "text-to-video",
            "video": "text-to-video",
            "image-to-video": "image-to-video",
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

    # Guard against unexpected API shape — resp.json() returns Any.
    # If the API returns a dict (e.g., {"results": [...]}) or a string,
    # iterating directly would produce wrong data without error. (Cycle 29 fix)
    if not isinstance(data, list):
        return to_json({
            "error": "Unexpected HuggingFace API response format (expected list).",
            "hint": "The API may have changed. Try again or check the endpoint.",
        })

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
        path_err = validate_path(path_str, must_exist=True)  # Cycle 46: path traversal guard
        if path_err:
            return to_json({"error": path_err})
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

    # Collect unique class_types from top-level nodes
    class_types = set()
    for node in workflow.values():
        ct = node.get("class_type")
        if ct:
            class_types.add(ct)

    # Also collect class_types from component/subgraph definitions.
    # Component instance nodes use a UUID as their class_type; the actual
    # internal nodes live under definitions.subgraphs[].nodes[].
    from .workflow_parse import _extract_subgraph_nodes, _all_subgraph_class_types
    subgraph_info = _extract_subgraph_nodes(data) if path_str else {}
    subgraph_class_types = _all_subgraph_class_types(subgraph_info)
    class_types.update(subgraph_class_types)

    # Remove UUID-style component types — they aren't real node classes
    class_types = {ct for ct in class_types if not _UUID_PATTERN.match(ct)}

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

            report_item = {
                "node_type": node_type,
                "pack_title": pack_info["title"],
                "pack_url": url,
                "pack_installed": _is_pack_installed(url),
            }
            dep_info = _check_deprecated(node_type)
            if dep_info:
                report_item["deprecated"] = True
                report_item["replacement_available"] = dep_info["replacement"]
            missing_report.append(report_item)
        else:
            report_item = {
                "node_type": node_type,
                "pack_title": None,
                "pack_url": None,
                "message": "Not found in ComfyUI Manager registry.",
            }
            dep_info = _check_deprecated(node_type)
            if dep_info:
                report_item["deprecated"] = True
                report_item["replacement_available"] = dep_info["replacement"]
            missing_report.append(report_item)

    # Summary
    component_note = None
    if subgraph_info:
        component_note = (
            f"Workflow contains {len(subgraph_info)} component/subgraph "
            f"definition(s). Internal nodes were also checked for availability."
        )

    if not missing:
        result = {
            "status": "all_installed",
            "total_node_types": len(class_types),
            "message": "All node types in this workflow are available.",
        }
        if component_note:
            result["component_note"] = component_note
        return to_json(result)

    result = {
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
    }
    if component_note:
        result["component_note"] = component_note
    return to_json(result)


def _handle_get_install_instructions(tool_input: dict) -> str:
    """Get install instructions for a node pack or model."""
    query = tool_input.get("query")  # Cycle 48: guard required field
    if not query or not isinstance(query, str):
        return to_json({"error": "query is required and must be a non-empty string."})
    source = tool_input.get("source", "registry")

    if source == "civitai":
        return to_json({
            "source": "civitai",
            "query": query,
            "instructions": (
                f"To install a CivitAI model, use the CivitAI tools to find the "
                f"model first (discover), then download the file to the "
                f"appropriate directory under {MODELS_DIR}. "
                f"CivitAI models are typically .safetensors files."
            ),
            "steps": [
                f"1. Use discover to find '{query}' and get the download URL",
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
                f"1. Use discover to find '{query}'",
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
        "error": f"Could not find '{query}' in the registry. Try discover first.",
    })


def _clear_cache():
    """Clear all in-memory caches, forcing reload from disk on next access."""
    global _catalog_cache
    with _cache_lock:
        _cache["custom_nodes"] = None
        _cache["extension_map"] = None
        _cache["node_to_pack"] = None
        _cache["model_list"] = None
        _freshness["custom_nodes_loaded_at"] = None
        _freshness["extension_map_loaded_at"] = None
        _freshness["model_list_loaded_at"] = None
        _catalog_cache = None


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
        if name == "discover":
            return _handle_discover(tool_input)
        elif name == "find_missing_nodes":
            return _handle_find_missing_nodes(tool_input)
        elif name == "get_install_instructions":
            return _handle_get_install_instructions(tool_input)
        elif name == "check_registry_freshness":
            return _handle_check_freshness(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        log.error("Unhandled error in comfy_discover tool %s: %s", name, e, exc_info=True)  # Cycle 49: add logging
        return to_json({  # Cycle 49: user-friendly context
            "error": f"Discovery failed ({type(e).__name__}): {e}",
            "tool": name,
        })
