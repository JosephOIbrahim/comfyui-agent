"""Tests for GitHub release tracking tools."""

import json
from unittest.mock import patch

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
