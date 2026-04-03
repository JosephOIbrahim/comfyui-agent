"""Tests for agent.health module."""

from unittest.mock import MagicMock, patch

import httpx
import pytest


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    """Clear LLM provider cache between tests."""
    from agent.llm import _provider_cache

    _provider_cache.clear()
    yield
    _provider_cache.clear()


def _mock_system_stats_response(devices=None):
    """Build a mock httpx.Response for /system_stats."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"devices": devices or []}
    return resp


class TestHealthAllOk:
    def test_health_all_ok(self):
        """Both ComfyUI and LLM reachable -> status ok."""
        mock_resp = _mock_system_stats_response()

        with (
            patch("agent.health.httpx.Client") as mock_client_cls,
            patch("agent.health._check_llm") as mock_llm,
        ):
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_llm.return_value = {
                "status": "ok",
                "provider": "anthropic",
                "class": "AnthropicProvider",
            }

            from agent.health import check_health

            result = check_health()

        assert result["status"] == "ok"
        assert result["comfyui"]["status"] == "ok"
        assert result["llm"]["status"] == "ok"
        assert "uptime_seconds" in result
        assert "llm_provider" in result


class TestHealthComfyUIDown:
    def test_health_comfyui_down(self):
        """ComfyUI connection refused -> status degraded."""
        with (
            patch("agent.health.httpx.Client") as mock_client_cls,
            patch("agent.health._check_llm") as mock_llm,
        ):
            client_instance = MagicMock()
            client_instance.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=client_instance)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_llm.return_value = {
                "status": "ok",
                "provider": "anthropic",
                "class": "AnthropicProvider",
            }

            from agent.health import check_health

            result = check_health()

        assert result["status"] == "degraded"
        assert result["comfyui"]["status"] == "error"
        assert "Connection refused" in result["comfyui"]["error"]


class TestHealthComfyUITimeout:
    def test_health_comfyui_timeout(self):
        """ComfyUI timeout -> status degraded."""
        with (
            patch("agent.health.httpx.Client") as mock_client_cls,
            patch("agent.health._check_llm") as mock_llm,
        ):
            client_instance = MagicMock()
            client_instance.get.side_effect = httpx.ReadTimeout("timed out")
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=client_instance)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_llm.return_value = {
                "status": "ok",
                "provider": "anthropic",
                "class": "AnthropicProvider",
            }

            from agent.health import check_health

            result = check_health()

        assert result["status"] == "degraded"
        assert result["comfyui"]["status"] == "error"
        assert "Timeout" in result["comfyui"]["error"]


class TestHealthLLMError:
    def test_health_llm_error(self):
        """LLM provider init fails -> status degraded."""
        mock_resp = _mock_system_stats_response()

        with (
            patch("agent.health.httpx.Client") as mock_client_cls,
            patch("agent.llm.get_provider", side_effect=ValueError("Unknown provider")),
        ):
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            from agent.health import check_health

            result = check_health()

        assert result["status"] == "degraded"
        assert result["llm"]["status"] == "error"
        assert "Unknown provider" in result["llm"]["error"]


class TestHealthUptime:
    def test_health_includes_uptime(self):
        """uptime_seconds is a non-negative number."""
        with (
            patch("agent.health._check_comfyui") as mock_comfy,
            patch("agent.health._check_llm") as mock_llm,
        ):
            mock_comfy.return_value = {"status": "ok"}
            mock_llm.return_value = {"status": "ok"}

            from agent.health import check_health

            result = check_health()

        assert isinstance(result["uptime_seconds"], int)
        assert result["uptime_seconds"] >= 0


class TestHealthGPUInfo:
    def test_health_gpu_info(self):
        """GPU info extracted from ComfyUI system_stats devices."""
        devices = [
            {
                "name": "NVIDIA RTX 4090",
                "vram_total": 25_769_803_776,
                "vram_free": 20_000_000_000,
            }
        ]
        mock_resp = _mock_system_stats_response(devices=devices)

        with patch("agent.health.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            from agent.health import _check_comfyui

            result = _check_comfyui()

        assert result["status"] == "ok"
        assert result["gpu"]["name"] == "NVIDIA RTX 4090"
        assert result["gpu"]["vram_total_gb"] == 25.8
        assert result["gpu"]["vram_free_gb"] == 20.0
