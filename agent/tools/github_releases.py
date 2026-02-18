"""GitHub release tracking for installed custom node packs.

Checks for updates to installed ComfyUI custom nodes by querying the
GitHub Releases API. Helps artists and TDs keep their node ecosystem
current without manually visiting each repo.

Requires: git on PATH (for local version detection).
Optional: GITHUB_API_TOKEN env var for higher rate limits.
"""

import json
import logging
import os
import re
import subprocess

import httpx

from ..rate_limiter import GITHUB_LIMITER
from ._util import to_json

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_API_TOKEN")

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "check_node_updates",
        "description": (
            "Check all installed custom node packs for available updates. "
            "Scans each pack's git remote, queries GitHub for the latest release, "
            "and reports which packs have newer versions available. "
            "No parameters required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_repo_releases",
        "description": (
            "Get recent releases for a specific GitHub repository. "
            "Useful after discovering a node pack to check its release history. "
            "Example: get_repo_releases(repo='comfyanonymous/ComfyUI')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "GitHub repo in 'owner/name' format.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max releases to return (1-20). Default: 5.",
                },
            },
            "required": ["repo"],
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_headers() -> dict:
    """Build GitHub API request headers."""
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


_GH_REMOTE_RE = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"
)


def _extract_github_repo(remote_url: str) -> str | None:
    """Extract 'owner/repo' from a GitHub remote URL."""
    m = _GH_REMOTE_RE.search(remote_url)
    if m:
        return f"{m.group('owner')}/{m.group('repo')}"
    return None


def _get_git_remote(pack_path: str) -> str | None:
    """Get the git remote origin URL for a pack directory."""
    try:
        result = subprocess.run(
            ["git", "-C", pack_path, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_git_head_short(pack_path: str) -> str | None:
    """Get short HEAD commit hash for a local git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", pack_path, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _fetch_latest_release(repo: str) -> dict | None:
    """Fetch the latest release for a GitHub repo."""
    if not GITHUB_LIMITER().acquire(timeout=5.0):
        return None

    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{GITHUB_API}/repos/{repo}/releases/latest",
                headers=_build_headers(),
                timeout=15.0,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, Exception) as e:
        log.debug("Failed to fetch release for %s: %s", repo, e)
        return None


def _fetch_releases(repo: str, limit: int = 5) -> list[dict]:
    """Fetch recent releases for a GitHub repo."""
    if not GITHUB_LIMITER().acquire(timeout=5.0):
        return []

    try:
        with httpx.Client() as client:
            resp = client.get(
                f"{GITHUB_API}/repos/{repo}/releases",
                params={"per_page": min(limit, 20)},
                headers=_build_headers(),
                timeout=15.0,
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, Exception) as e:
        log.debug("Failed to fetch releases for %s: %s", repo, e)
        return []


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_check_node_updates(tool_input: dict) -> str:
    """Check installed custom node packs for updates."""
    from . import comfy_inspect

    packs_result = json.loads(comfy_inspect.handle("list_custom_nodes", {}))
    if "error" in packs_result:
        return to_json({"error": f"Could not list custom nodes: {packs_result['error']}"})

    updates = []
    checked = 0
    up_to_date = 0
    skipped = 0

    for pack in packs_result.get("packs", []):
        pack_path = pack.get("path", "")
        if not pack_path:
            skipped += 1
            continue

        remote_url = _get_git_remote(pack_path)
        if not remote_url:
            skipped += 1
            continue

        repo = _extract_github_repo(remote_url)
        if not repo:
            skipped += 1
            continue

        checked += 1
        release = _fetch_latest_release(repo)
        if not release:
            up_to_date += 1
            continue

        local_head = _get_git_head_short(pack_path)
        tag = release.get("tag_name", "")
        published = release.get("published_at", "")[:10]

        updates.append({
            "name": pack["name"],
            "repo": repo,
            "local_head": local_head,
            "latest_tag": tag,
            "latest_name": release.get("name", tag),
            "published": published,
            "url": release.get("html_url", ""),
        })

    return to_json({
        "updates": updates,
        "checked": checked,
        "up_to_date": up_to_date,
        "has_updates": len(updates),
        "skipped": skipped,
    })


def _handle_get_repo_releases(tool_input: dict) -> str:
    """Get releases for a specific GitHub repo."""
    repo = tool_input["repo"]
    limit = min(tool_input.get("limit", 5), 20)

    # Validate repo format
    if "/" not in repo or repo.count("/") != 1:
        return to_json({"error": f"Invalid repo format: '{repo}'. Use 'owner/name'."})

    releases = _fetch_releases(repo, limit)
    if not releases:
        return to_json({
            "repo": repo,
            "releases": [],
            "message": "No releases found (repo may not use GitHub Releases).",
        })

    parsed = []
    for r in releases:
        body = r.get("body", "") or ""
        parsed.append({
            "tag": r.get("tag_name", ""),
            "name": r.get("name", ""),
            "published": r.get("published_at", "")[:10],
            "body_preview": body[:300] + ("..." if len(body) > 300 else ""),
            "url": r.get("html_url", ""),
            "prerelease": r.get("prerelease", False),
        })

    return to_json({
        "repo": repo,
        "releases": parsed,
        "showing": len(parsed),
    })


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def handle(name: str, tool_input: dict) -> str:
    """Execute a GitHub releases tool call."""
    try:
        if name == "check_node_updates":
            return _handle_check_node_updates(tool_input)
        elif name == "get_repo_releases":
            return _handle_get_repo_releases(tool_input)
        else:
            return to_json({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return to_json({"error": str(e)})
