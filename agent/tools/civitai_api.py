"""CivitAI discovery — search trending models, LoRAs, ControlNets.

The largest community hub for Stable Diffusion models. Public API,
no authentication required for search. Complements ComfyUI Manager's
curated registry with community ratings, download counts, and trending data.

API reference: https://github.com/civitai/civitai/wiki/REST-API-Reference
"""

import logging
import os
import re

import httpx

from ..config import MODELS_DIR
from ..rate_limiter import CIVITAI_LIMITER
from ._util import to_json

log = logging.getLogger(__name__)

CIVITAI_API = "https://civitai.com/api/v1"
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY")  # Optional — improves rate limits

# Our lowercase types -> CivitAI API enum values
_TYPE_MAP = {
    "checkpoint": "Checkpoint",
    "lora": "LORA",
    "controlnet": "Controlnet",
    "embedding": "TextualInversion",
    "vae": "VAE",
    "hypernetwork": "Hypernetwork",
    "upscaler": "Upscaler",
    "poses": "Poses",
}

# CivitAI type -> local model subdirectory
_TYPE_TO_SUBDIR = {
    "Checkpoint": "checkpoints",
    "LORA": "loras",
    "Controlnet": "controlnet",
    "VAE": "vae",
    "TextualInversion": "embeddings",
    "Upscaler": "upscale_models",
    "Hypernetwork": "hypernetworks",
}

# Shorthand -> CivitAI base model filter values
_BASE_MODEL_MAP = {
    "sd15": "SD 1.5",
    "sd1.5": "SD 1.5",
    "sdxl": "SDXL 1.0",
    "sdxl_turbo": "SDXL Turbo",
    "flux": "Flux.1 D",
    "flux_dev": "Flux.1 D",
    "flux_schnell": "Flux.1 S",
    "sd3": "SD 3",
    "sd3.5": "SD 3.5",
    "pony": "Pony",
    "cascade": "Stable Cascade",
}

_SORT_MAP = {
    "most_downloaded": "Most Downloaded",
    "highest_rated": "Highest Rated",
    "newest": "Newest",
}

_PERIOD_MAP = {
    "day": "Day",
    "week": "Week",
    "month": "Month",
    "year": "Year",
    "all_time": "AllTime",
}

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "search_civitai",
        "description": (
            "Search CivitAI for models, LoRAs, ControlNets, and embeddings. "
            "Returns community ratings, download counts, and compatibility info. "
            "CivitAI is the largest community hub — use this to discover what's "
            "popular and trending in the Stable Diffusion ecosystem."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term — model name, style, or keyword.",
                },
                "model_type": {
                    "type": "string",
                    "enum": ["checkpoint", "lora", "controlnet", "embedding", "vae", "upscaler"],
                    "description": "Filter by model type.",
                },
                "base_model": {
                    "type": "string",
                    "description": (
                        "Filter by base model: 'sd15', 'sdxl', 'flux', 'sd3', 'pony'."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["most_downloaded", "highest_rated", "newest"],
                    "description": "Sort order. Default: 'most_downloaded'.",
                },
                "period": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year", "all_time"],
                    "description": "Time period for sorting. Default: 'all_time'.",
                },
                "nsfw": {
                    "type": "boolean",
                    "description": "Include NSFW results. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (1-20). Default: 10.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_civitai_model",
        "description": (
            "Get detailed info for a specific CivitAI model by ID. Returns "
            "all versions, files, example images, description, and stats. "
            "Use after search_civitai to get full details before recommending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "integer",
                    "description": "CivitAI model ID (from search results).",
                },
            },
            "required": ["model_id"],
        },
    },
    {
        "name": "get_trending_models",
        "description": (
            "Get trending models on CivitAI this week. Quick way to discover "
            "what's popular right now. Useful for proactive recommendations: "
            "'A new LoRA is trending and compatible with your workflow.'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "enum": ["checkpoint", "lora", "controlnet", "embedding", "vae", "upscaler"],
                    "description": "Filter by type. Default: all types.",
                },
                "base_model": {
                    "type": "string",
                    "description": (
                        "Filter by base: 'sd15', 'sdxl', 'flux', 'sd3', 'pony'."
                    ),
                },
                "period": {
                    "type": "string",
                    "enum": ["day", "week", "month"],
                    "description": "Trending period. Default: 'week'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (1-20). Default: 10.",
                },
            },
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags from CivitAI descriptions."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _build_headers() -> dict:
    """Build request headers with optional auth."""
    headers = {"Content-Type": "application/json"}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    return headers


def _check_installed(model_type: str, files: list[dict]) -> bool:
    """Check if any file from a CivitAI model version is installed locally."""
    if not MODELS_DIR.exists():
        return False
    subdir = _TYPE_TO_SUBDIR.get(model_type, "")
    if subdir:
        target_dir = MODELS_DIR / subdir
        if target_dir.exists():
            for f in files:
                name = f.get("name", "")
                if name and (target_dir / name).exists():
                    return True
    # Fallback: search all model subdirectories
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir():
            for f in files:
                name = f.get("name", "")
                if name and (d / name).exists():
                    return True
    return False


def _parse_model(model: dict) -> dict:
    """Extract key fields from a CivitAI model response."""
    stats = model.get("stats", {})
    creator = model.get("creator", {})
    versions = model.get("modelVersions", [])
    model_type = model.get("type", "")

    latest = versions[0] if versions else {}
    latest_files = latest.get("files", [])

    return {
        "id": model.get("id"),
        "name": model.get("name", ""),
        "type": model_type,
        "creator": creator.get("username", ""),
        "rating": round(stats.get("rating", 0), 2),
        "downloads": stats.get("downloadCount", 0),
        "favorites": stats.get("favoriteCount", 0),
        "tags": sorted(model.get("tags", []))[:8],
        "base_model": latest.get("baseModel", ""),
        "latest_version": latest.get("name", ""),
        "nsfw": model.get("nsfw", False),
        "url": f"https://civitai.com/models/{model.get('id', '')}",
        "installed": _check_installed(model_type, latest_files),
    }


def _parse_model_detail(model: dict) -> dict:
    """Parse full model detail including versions and files."""
    stats = model.get("stats", {})
    creator = model.get("creator", {})
    model_type = model.get("type", "")

    versions = []
    for v in model.get("modelVersions", [])[:5]:
        files = []
        for f in v.get("files", []):
            files.append({
                "name": f.get("name", ""),
                "size_kb": round(f.get("sizeKB", 0)),
                "format": f.get("metadata", {}).get("format", ""),
            })
        images = [
            img.get("url", "")
            for img in v.get("images", [])[:3]
            if img.get("url")
        ]
        versions.append({
            "id": v.get("id"),
            "name": v.get("name", ""),
            "base_model": v.get("baseModel", ""),
            "created_at": v.get("createdAt", "")[:10],
            "download_url": v.get("downloadUrl", ""),
            "files": files,
            "example_images": images,
            "installed": _check_installed(model_type, v.get("files", [])),
        })

    return {
        "id": model.get("id"),
        "name": model.get("name", ""),
        "type": model_type,
        "creator": creator.get("username", ""),
        "description": _strip_html(model.get("description", ""))[:500],
        "tags": sorted(model.get("tags", []))[:10],
        "nsfw": model.get("nsfw", False),
        "stats": {
            "downloads": stats.get("downloadCount", 0),
            "favorites": stats.get("favoriteCount", 0),
            "rating": round(stats.get("rating", 0), 2),
            "rating_count": stats.get("ratingCount", 0),
        },
        "versions": versions,
        "url": f"https://civitai.com/models/{model.get('id', '')}",
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_search_civitai(tool_input: dict) -> str:
    query = tool_input["query"]
    model_type = tool_input.get("model_type")
    base_model = tool_input.get("base_model")
    sort = tool_input.get("sort", "most_downloaded")
    period = tool_input.get("period", "all_time")
    nsfw = tool_input.get("nsfw", False)
    max_results = min(tool_input.get("max_results", 10), 20)

    params: dict = {
        "query": query,
        "limit": max_results,
        "sort": _SORT_MAP.get(sort, "Most Downloaded"),
        "period": _PERIOD_MAP.get(period, "AllTime"),
        "nsfw": str(nsfw).lower(),
    }

    if model_type:
        civitai_type = _TYPE_MAP.get(model_type.lower())
        if civitai_type:
            params["types"] = civitai_type

    if base_model:
        bm = _BASE_MODEL_MAP.get(base_model.lower(), base_model)
        params["baseModels"] = bm

    if not CIVITAI_LIMITER().acquire(timeout=5.0):
        return to_json({"error": "Rate limited — too many CivitAI requests. Try again shortly."})

    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{CIVITAI_API}/models",
                params=params,
                headers=_build_headers(),
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return to_json({"error": "Could not reach CivitAI API. Check internet connection."})
    except httpx.HTTPStatusError as e:
        return to_json({"error": f"CivitAI API returned {e.response.status_code}."})
    except Exception as e:
        return to_json({"error": f"CivitAI search failed: {e}"})

    items = data.get("items", [])
    results = [_parse_model(m) for m in items]

    return to_json({
        "source": "civitai",
        "query": query,
        "type_filter": model_type,
        "base_model_filter": base_model,
        "sort": sort,
        "period": period,
        "results": results,
        "showing": len(results),
        "total_available": data.get("metadata", {}).get("totalItems", len(results)),
    })


def _handle_get_civitai_model(tool_input: dict) -> str:
    model_id = tool_input["model_id"]

    if not CIVITAI_LIMITER().acquire(timeout=5.0):
        return to_json({"error": "Rate limited — too many CivitAI requests. Try again shortly."})

    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{CIVITAI_API}/models/{model_id}",
                headers=_build_headers(),
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return to_json({"error": "Could not reach CivitAI API. Check internet connection."})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return to_json({"error": f"Model {model_id} not found on CivitAI."})
        return to_json({"error": f"CivitAI API returned {e.response.status_code}."})
    except Exception as e:
        return to_json({"error": f"CivitAI request failed: {e}"})

    return to_json(_parse_model_detail(data))


def _handle_get_trending_models(tool_input: dict) -> str:
    model_type = tool_input.get("model_type")
    base_model = tool_input.get("base_model")
    period = tool_input.get("period", "week")
    max_results = min(tool_input.get("max_results", 10), 20)

    params: dict = {
        "limit": max_results,
        "sort": "Most Downloaded",
        "period": _PERIOD_MAP.get(period, "Week"),
        "nsfw": "false",
    }

    if model_type:
        civitai_type = _TYPE_MAP.get(model_type.lower())
        if civitai_type:
            params["types"] = civitai_type

    if base_model:
        bm = _BASE_MODEL_MAP.get(base_model.lower(), base_model)
        params["baseModels"] = bm

    if not CIVITAI_LIMITER().acquire(timeout=5.0):
        return to_json({"error": "Rate limited — too many CivitAI requests. Try again shortly."})

    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{CIVITAI_API}/models",
                params=params,
                headers=_build_headers(),
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return to_json({"error": "Could not reach CivitAI API. Check internet connection."})
    except httpx.HTTPStatusError as e:
        return to_json({"error": f"CivitAI API returned {e.response.status_code}."})
    except Exception as e:
        return to_json({"error": f"CivitAI trending request failed: {e}"})

    items = data.get("items", [])
    results = [_parse_model(m) for m in items]

    return to_json({
        "source": "civitai_trending",
        "period": period,
        "type_filter": model_type,
        "base_model_filter": base_model,
        "results": results,
        "showing": len(results),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(name: str, tool_input: dict) -> str:
    """Execute a CivitAI tool call."""
    try:
        if name == "search_civitai":
            return _handle_search_civitai(tool_input)
        elif name == "get_civitai_model":
            return _handle_get_civitai_model(tool_input)
        elif name == "get_trending_models":
            return _handle_get_trending_models(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
