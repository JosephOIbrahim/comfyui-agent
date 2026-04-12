"""Integration tests — discovery and node inspection tool handlers.

All tests mock the actual HTTP calls to ComfyUI. They exercise the tool
handler dispatch logic (input validation, response shaping, error paths)
without requiring a live server.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


class TestDiscoverModels:
    """discover tool with model-oriented queries."""

    def test_discover_models(self, comfyui_available: str) -> None:
        """Discover with a common query returns model names."""
        with patch(
            "agent.tools.comfy_discover._search_civitai",
            return_value=[
                {"name": "SDXL Base 1.0", "type": "checkpoint"},
                {"name": "SDXL Refiner 1.0", "type": "checkpoint"},
            ],
        ), patch(
            "agent.tools.comfy_discover._search_local_models",
            return_value=[],
        ), patch(
            "agent.tools.comfy_discover._search_hf",
            return_value=[],
        ):
            from agent.tools.comfy_discover import handle

            raw = handle("discover", {"query": "SDXL", "scope": "models"})
            result = json.loads(raw)
            assert "error" not in result
            # Should contain at least one result from the mock
            results = result.get("results", result.get("models", []))
            assert len(results) >= 1


class TestDiscoverNodes:
    """discover tool with node-oriented queries."""

    def test_discover_nodes(self, comfyui_available: str) -> None:
        """Discover custom nodes returns structured response."""
        with patch(
            "agent.tools.comfy_discover._search_registry",
            return_value=[
                {
                    "name": "ComfyUI-Manager",
                    "description": "ComfyUI node manager",
                    "url": "https://github.com/example/ComfyUI-Manager",
                },
            ],
        ), patch(
            "agent.tools.comfy_discover._search_local_nodes",
            return_value=[],
        ):
            from agent.tools.comfy_discover import handle

            raw = handle("discover", {"query": "manager", "scope": "nodes"})
            result = json.loads(raw)
            assert "error" not in result


class TestGetModelsSummary:
    """get_models_summary tool handler."""

    def test_get_models_summary(self, comfyui_available: str) -> None:
        """get_models_summary returns dict with model categories."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": [["model_a.safetensors", "model_b.safetensors"]]
                    }
                }
            }
        }

        with patch("agent.tools.comfy_api._get", return_value=mock_response):
            from agent.tools.comfy_api import handle as api_handle

            raw = api_handle("get_models_summary", {})
            result = json.loads(raw)
            assert isinstance(result, dict)
            assert "error" not in result


class TestGetNodeInfo:
    """get_node_info tool handler."""

    def test_get_node_info_ksampler(self, comfyui_available: str) -> None:
        """get_node_info for KSampler returns inputs and outputs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "KSampler": {
                "input": {
                    "required": {
                        "seed": ["INT", {"default": 0}],
                        "steps": ["INT", {"default": 20}],
                        "cfg": ["FLOAT", {"default": 8.0}],
                        "sampler_name": [["euler", "dpm_2"]],
                        "scheduler": [["normal", "karras"]],
                        "denoise": ["FLOAT", {"default": 1.0}],
                        "model": ["MODEL"],
                        "positive": ["CONDITIONING"],
                        "negative": ["CONDITIONING"],
                        "latent_image": ["LATENT"],
                    }
                },
                "output": ["LATENT"],
                "output_name": ["LATENT"],
                "name": "KSampler",
                "display_name": "KSampler",
                "category": "sampling",
            }
        }

        with patch("agent.tools.comfy_api._get", return_value=mock_response):
            from agent.tools.comfy_api import handle as api_handle

            raw = api_handle("get_node_info", {"node_type": "KSampler"})
            result = json.loads(raw)
            assert "error" not in result
            # Should have input information
            node_data = result.get("KSampler", result)
            assert "input" in node_data or "inputs" in node_data or "required" in str(result)
