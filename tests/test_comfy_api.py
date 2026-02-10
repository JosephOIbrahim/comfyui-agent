"""Tests for comfy_api tool — mocked HTTP, no real ComfyUI needed."""

import json
import pytest
from unittest.mock import patch, MagicMock
from agent.tools import comfy_api


def _mock_response(data: dict, status_code: int = 200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    return resp


class TestIsRunning:
    def test_running(self):
        mock_stats = {
            "system": {"python_version": "3.11.0"},
            "devices": [{"name": "NVIDIA RTX 4090", "type": "cuda"}],
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_stats):
            result = json.loads(comfy_api.handle("is_comfyui_running", {}))
            assert result["running"] is True
            assert "4090" in result["gpu"]

    def test_not_running(self):
        with patch("agent.tools.comfy_api._get", side_effect=ConnectionError("nope")):
            result = json.loads(comfy_api.handle("is_comfyui_running", {}))
            assert result["running"] is False


class TestGetAllNodes:
    def test_summary_no_filter(self):
        mock_data = {
            "KSampler": {"category": "sampling", "display_name": "KSampler"},
            "CLIPTextEncode": {"category": "conditioning", "display_name": "CLIP Text Encode"},
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_all_nodes", {}))
            assert result["count"] == 2
            assert "KSampler" in result["nodes"]

    def test_name_filter(self):
        mock_data = {
            "KSampler": {
                "category": "sampling",
                "display_name": "KSampler",
                "description": "",
                "input": {"required": {"model": ["MODEL"]}},
                "output": ["LATENT"],
            },
            "CLIPTextEncode": {
                "category": "conditioning",
                "display_name": "CLIP Text Encode",
                "description": "",
                "input": {"required": {"text": ["STRING"]}},
                "output": ["CONDITIONING"],
            },
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_all_nodes", {"name_filter": "sampler"}))
            assert result["count"] == 1
            assert "KSampler" in result["nodes"]

    def test_category_filter(self):
        mock_data = {
            "KSampler": {
                "category": "sampling",
                "display_name": "KSampler",
                "description": "",
                "input": {"required": {}},
                "output": [],
            },
            "CLIPTextEncode": {
                "category": "conditioning",
                "display_name": "CLIP Text Encode",
                "description": "",
                "input": {"required": {}},
                "output": [],
            },
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(
                comfy_api.handle("get_all_nodes", {"category_filter": "conditioning"})
            )
            assert result["count"] == 1
            assert "CLIPTextEncode" in result["nodes"]


class TestGetNodeInfo:
    def test_found(self):
        mock_data = {
            "KSampler": {
                "display_name": "KSampler",
                "category": "sampling",
                "description": "Runs a sampler",
                "input": {
                    "required": {
                        "model": ["MODEL"],
                        "seed": ["INT", {"default": 0, "min": 0, "max": 2**32}],
                        "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    },
                },
                "output": ["LATENT"],
                "output_name": ["LATENT"],
                "output_is_list": [False],
            },
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_node_info", {"node_type": "KSampler"}))
            assert result["class_type"] == "KSampler"
            assert "seed" in result["input"]["required"]

    def test_not_found_suggests(self):
        # First call returns empty (specific node), second returns all
        def side_effect(path, **kwargs):
            if "/object_info/NotReal" in path:
                return {}
            return {"KSampler": {}, "KSamplerAdvanced": {}}

        with patch("agent.tools.comfy_api._get", side_effect=side_effect):
            result = json.loads(comfy_api.handle("get_node_info", {"node_type": "NotReal"}))
            assert "error" in result
            assert "similar_nodes" in result


class TestGetSystemStats:
    def test_returns_stats(self):
        mock_data = {
            "system": {"python_version": "3.11.0", "embedded_python": False},
            "devices": [{"name": "RTX 4090", "type": "cuda", "vram_total": 25769803776}],
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_system_stats", {}))
            assert "system" in result
            assert "devices" in result


class TestGetQueue:
    def test_empty_queue(self):
        mock_data = {"queue_running": [], "queue_pending": []}
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_queue_status", {}))
            assert result["running_count"] == 0
            assert result["pending_count"] == 0


class TestGetHistory:
    def test_specific_prompt(self):
        mock_data = {
            "abc123": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "9": {
                        "images": [{"filename": "out_00001.png", "subfolder": "", "type": "output"}],
                    },
                },
            },
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(
                comfy_api.handle("get_history", {"prompt_id": "abc123"})
            )
            assert "abc123" in result
            assert result["abc123"]["outputs"][0]["filename"] == "out_00001.png"


class TestConnectionError:
    def test_connect_error(self):
        import httpx
        with patch(
            "agent.tools.comfy_api._get",
            side_effect=httpx.ConnectError("refused"),
        ):
            result = json.loads(comfy_api.handle("get_system_stats", {}))
            assert "error" in result
            assert "running" in result["error"].lower() or "connect" in result["error"].lower()
