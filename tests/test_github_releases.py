"""Tests for GitHub release tracking tools."""

import json
from unittest.mock import patch, MagicMock

from agent.tools import github_releases


class TestExtractGitHubRepo:
    def test_https_url(self):
        url = "https://github.com/comfyanonymous/ComfyUI.git"
        assert github_releases._extract_github_repo(url) == "comfyanonymous/ComfyUI"

    def test_https_url_no_git_suffix(self):
        url = "https://github.com/owner/repo"
        assert github_releases._extract_github_repo(url) == "owner/repo"

    def test_ssh_url(self):
        url = "git@github.com:owner/repo.git"
        assert github_releases._extract_github_repo(url) == "owner/repo"

    def test_non_github_url(self):
        url = "https://gitlab.com/owner/repo.git"
        assert github_releases._extract_github_repo(url) is None

    def test_empty_string(self):
        assert github_releases._extract_github_repo("") is None


class TestBuildHeaders:
    def test_headers_without_token(self):
        with patch.object(github_releases, "GITHUB_TOKEN", None):
            h = github_releases._build_headers()
            assert "Authorization" not in h
            assert h["Accept"] == "application/vnd.github+json"

    def test_headers_with_token(self):
        with patch.object(github_releases, "GITHUB_TOKEN", "ghp_test123"):
            h = github_releases._build_headers()
            assert h["Authorization"] == "Bearer ghp_test123"


class TestGetRepoReleases:
    def test_basic_fetch(self):
        mock_releases = [
            {
                "tag_name": "v1.2.0",
                "name": "Release 1.2.0",
                "published_at": "2025-01-15T00:00:00Z",
                "body": "Bug fixes and improvements",
                "html_url": "https://github.com/owner/repo/releases/tag/v1.2.0",
                "prerelease": False,
            },
        ]
        with patch("agent.tools.github_releases._fetch_releases", return_value=mock_releases):
            result = json.loads(github_releases.handle("get_repo_releases", {
                "repo": "owner/repo",
            }))
        assert result["repo"] == "owner/repo"
        assert len(result["releases"]) == 1
        assert result["releases"][0]["tag"] == "v1.2.0"

    def test_invalid_repo_format(self):
        result = json.loads(github_releases.handle("get_repo_releases", {
            "repo": "invalid-no-slash",
        }))
        assert "error" in result

    def test_repo_not_found(self):
        with patch("agent.tools.github_releases._fetch_releases", return_value=[]):
            result = json.loads(github_releases.handle("get_repo_releases", {
                "repo": "owner/nonexistent",
            }))
        assert result["releases"] == []
        assert "No releases" in result.get("message", "")

    def test_body_preview_truncated(self):
        long_body = "A" * 500
        mock_releases = [
            {
                "tag_name": "v1.0",
                "name": "v1.0",
                "published_at": "2025-01-01T00:00:00Z",
                "body": long_body,
                "html_url": "https://github.com/o/r/releases/tag/v1.0",
                "prerelease": False,
            },
        ]
        with patch("agent.tools.github_releases._fetch_releases", return_value=mock_releases):
            result = json.loads(github_releases.handle("get_repo_releases", {
                "repo": "o/r",
            }))
        preview = result["releases"][0]["body_preview"]
        assert len(preview) <= 303 + 3  # 300 chars + "..."
        assert preview.endswith("...")


class TestCheckNodeUpdates:
    def test_with_updates(self):
        packs = {
            "packs": [
                {"name": "ComfyUI-Impact-Pack", "path": "/fake/path"},
            ],
            "count": 1,
        }
        release = {
            "tag_name": "v5.0",
            "name": "Impact Pack v5.0",
            "published_at": "2025-02-01T00:00:00Z",
            "html_url": "https://github.com/ltdrdata/ComfyUI-Impact-Pack/releases/tag/v5.0",
        }
        with (
            patch("agent.tools.comfy_inspect.handle", return_value=json.dumps(packs)),
            patch("agent.tools.github_releases._get_git_remote", return_value="https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"),
            patch("agent.tools.github_releases._get_git_head_short", return_value="abc1234"),
            patch("agent.tools.github_releases._fetch_latest_release", return_value=release),
        ):
            result = json.loads(github_releases.handle("check_node_updates", {}))
        assert result["checked"] == 1
        assert result["has_updates"] == 1
        assert result["updates"][0]["name"] == "ComfyUI-Impact-Pack"
        assert result["updates"][0]["latest_tag"] == "v5.0"

    def test_no_git_dir_skipped(self):
        packs = {
            "packs": [
                {"name": "SomeNode", "path": "/fake/path"},
            ],
            "count": 1,
        }
        with (
            patch("agent.tools.comfy_inspect.handle", return_value=json.dumps(packs)),
            patch("agent.tools.github_releases._get_git_remote", return_value=None),
        ):
            result = json.loads(github_releases.handle("check_node_updates", {}))
        assert result["checked"] == 0
        assert result["skipped"] == 1

    def test_inspect_error(self):
        with patch(
            "agent.tools.comfy_inspect.handle",
            return_value=json.dumps({"error": "Not found"}),
        ):
            result = json.loads(github_releases.handle("check_node_updates", {}))
        assert "error" in result

    def test_no_packs(self):
        packs = {"packs": [], "count": 0}
        with patch(
            "agent.tools.comfy_inspect.handle",
            return_value=json.dumps(packs),
        ):
            result = json.loads(github_releases.handle("check_node_updates", {}))
        assert result["checked"] == 0
        assert result["has_updates"] == 0


class TestToolRegistration:
    def test_tools_exported(self):
        assert len(github_releases.TOOLS) == 2
        names = {t["name"] for t in github_releases.TOOLS}
        assert "check_node_updates" in names
        assert "get_repo_releases" in names

    def test_handle_unknown_tool(self):
        result = json.loads(github_releases.handle("nonexistent", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 30: JSON response type guard tests
# ---------------------------------------------------------------------------

class TestResponseTypeGuards:
    """_fetch_latest_release and _fetch_releases must guard against non-dict/non-list."""

    def _make_client(self, mock_client_cls, status=200, data=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = mock_resp
        mock_client_cls.return_value = mc
        return mc

    def test_fetch_latest_release_list_returns_none(self):
        """If GitHub returns a list for /releases/latest, treat as no release."""
        with patch("agent.tools.github_releases.GITHUB_LIMITER") as lim, \
             patch("httpx.Client") as mock_client:
            lim.return_value.return_value.acquire.return_value = True
            self._make_client(mock_client, data=["not", "a", "dict"])
            result = github_releases._fetch_latest_release("owner/repo")
        assert result is None

    def test_fetch_latest_release_dict_passes(self):
        """If GitHub returns a valid dict, return it unchanged."""
        with patch("agent.tools.github_releases.GITHUB_LIMITER") as lim, \
             patch("httpx.Client") as mock_client:
            lim.return_value.return_value.acquire.return_value = True
            self._make_client(mock_client, data={"tag_name": "v1.0.0", "name": "Release 1.0"})
            result = github_releases._fetch_latest_release("owner/repo")
        assert isinstance(result, dict)
        assert result["tag_name"] == "v1.0.0"

    def test_fetch_releases_dict_returns_empty(self):
        """If GitHub returns a dict for /releases (not a list), return []."""
        with patch("agent.tools.github_releases.GITHUB_LIMITER") as lim, \
             patch("httpx.Client") as mock_client:
            lim.return_value.return_value.acquire.return_value = True
            self._make_client(mock_client, data={"message": "Not Found"})
            result = github_releases._fetch_releases("owner/repo")
        assert result == []

    def test_fetch_releases_list_passes(self):
        """If GitHub returns a list, return it."""
        with patch("agent.tools.github_releases.GITHUB_LIMITER") as lim, \
             patch("httpx.Client") as mock_client:
            lim.return_value.return_value.acquire.return_value = True
            self._make_client(mock_client, data=[{"tag_name": "v1.0"}, {"tag_name": "v0.9"}])
            result = github_releases._fetch_releases("owner/repo")
        assert isinstance(result, list)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Cycle 42 — check_node_updates non-JSON guard
# ---------------------------------------------------------------------------

class TestCheckNodeUpdatesNonJsonGuard:
    """Guard against non-JSON returned by comfy_inspect.handle."""

    def test_non_json_from_inspect_returns_error(self):
        """If comfy_inspect.handle returns non-JSON, return structured error."""
        from unittest.mock import patch
        import json

        with patch("agent.tools.comfy_inspect.handle", return_value="NOT JSON AT ALL"):
            result = json.loads(github_releases.handle("check_node_updates", {}))
        assert "error" in result
        assert "parse" in result["error"].lower() or "json" in result["error"].lower()

    def test_inspect_returns_valid_json_with_error_field(self):
        """If inspect returns JSON with error key, surface it."""
        from unittest.mock import patch
        import json

        error_payload = json.dumps({"error": "ComfyUI not running"})
        with patch("agent.tools.comfy_inspect.handle", return_value=error_payload):
            result = json.loads(github_releases.handle("check_node_updates", {}))
        assert "error" in result
        assert "comfyui" in result["error"].lower() or "custom nodes" in result["error"].lower()


# ---------------------------------------------------------------------------
# Cycle 43 — intent_collector required field guards
# ---------------------------------------------------------------------------
# (Testing in github_releases test file would be wrong; create proper file via handle)

class TestIntentCollectorRequiredFields:
    """capture_intent must guard missing/wrong-type required fields."""

    def test_missing_user_request_returns_error(self):
        """capture_intent with no user_request returns structured error."""
        from agent.brain import handle
        import json
        result = json.loads(handle("capture_intent", {
            "interpretation": "Lower CFG to 5",
        }))
        assert "error" in result
        assert "user_request" in result["error"].lower()

    def test_missing_interpretation_returns_error(self):
        """capture_intent with no interpretation returns structured error."""
        from agent.brain import handle
        import json
        result = json.loads(handle("capture_intent", {
            "user_request": "Make it dreamier",
        }))
        assert "error" in result
        assert "interpretation" in result["error"].lower()

    def test_empty_string_user_request_returns_error(self):
        """capture_intent with empty user_request string returns error."""
        from agent.brain import handle
        import json
        result = json.loads(handle("capture_intent", {
            "user_request": "",
            "interpretation": "Do something",
        }))
        assert "error" in result

    def test_both_required_present_succeeds(self):
        """capture_intent with both required fields succeeds."""
        from agent.brain import handle
        import json
        result = json.loads(handle("capture_intent", {
            "user_request": "Make it dreamier",
            "interpretation": "Lower CFG to 5, switch to DPM++ 2M Karras",
        }))
        assert "error" not in result
        assert result.get("status") == "captured"

    def test_brain_shim_unknown_tool_returns_error_not_recursion(self):
        """Module-level handle() with unknown tool returns JSON error, not infinite recursion."""
        from agent.brain import intent_collector
        import json
        result = json.loads(intent_collector.handle("nonexistent_tool_xyz", {}))
        assert "error" in result
        assert "unknown" in result["error"].lower()


# ---------------------------------------------------------------------------
# Cycle 47 — get_node_releases required field guard
# ---------------------------------------------------------------------------

class TestGetNodeReleasesRequiredField:
    """get_node_releases must return structured error when repo is missing or invalid."""

    def test_missing_repo_returns_error(self):
        import json
        from agent.tools import github_releases
        result = json.loads(github_releases.handle("get_repo_releases", {}))
        assert "error" in result
        assert "repo" in result["error"].lower()

    def test_empty_repo_returns_error(self):
        import json
        from agent.tools import github_releases
        result = json.loads(github_releases.handle("get_repo_releases", {"repo": ""}))
        assert "error" in result

    def test_none_repo_returns_error(self):
        import json
        from agent.tools import github_releases
        result = json.loads(github_releases.handle("get_repo_releases", {"repo": None}))
        assert "error" in result

    def test_valid_repo_passes_guard(self):
        """A valid 'owner/name' format must not be blocked by the missing-field guard."""
        import json
        from unittest.mock import patch
        from agent.tools import github_releases
        # Format validation fires ("/" check) — guard error must not appear
        with patch.object(github_releases, "_fetch_releases", return_value=[]):
            result = json.loads(github_releases.handle("get_repo_releases", {
                "repo": "owner/name",
            }))
        assert "repo" not in result.get("error", "").lower() or "required" not in result.get("error", "").lower()
