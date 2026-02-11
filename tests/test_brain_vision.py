"""Tests for brain/vision.py — image analysis via Claude Vision."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.brain import vision


@pytest.fixture
def fake_image(tmp_path):
    """Create a tiny PNG for testing."""
    # Minimal valid PNG (1x1 red pixel)
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
        b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
        b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    img_path = tmp_path / "test_output.png"
    img_path.write_bytes(png_data)
    return str(img_path)


class TestAnalyzeImage:
    def test_file_not_found(self):
        result = json.loads(vision.handle("analyze_image", {
            "image_path": "/nonexistent/image.png",
        }))
        assert "error" in result

    def test_analyze_calls_vision_api(self, fake_image):
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = json.dumps({
            "quality_score": 0.85,
            "artifacts": [],
            "composition": "Good composition",
            "prompt_adherence": 0.9,
            "strengths": ["sharp details"],
            "suggestions": ["try higher CFG"],
        })
        mock_response.content = [mock_block]

        with patch("agent.brain.vision.anthropic.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = json.loads(vision.handle("analyze_image", {
                "image_path": fake_image,
                "prompt_used": "a red pixel",
            }))

        assert result["quality_score"] == 0.85
        assert result["image_path"] == fake_image
        # Verify the API was called with image content
        call_args = mock_client.messages.create.call_args
        assert call_args is not None

    def test_handles_non_json_response(self, fake_image):
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "This is not JSON"
        mock_response.content = [mock_block]

        with patch("agent.brain.vision.anthropic.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = json.loads(vision.handle("analyze_image", {
                "image_path": fake_image,
            }))

        assert "raw_analysis" in result
        assert "parse_error" in result


class TestCompareOutputs:
    def test_both_files_missing(self):
        result = json.loads(vision.handle("compare_outputs", {
            "image_a": "/fake/a.png",
            "image_b": "/fake/b.png",
        }))
        assert "error" in result

    def test_compare_calls_api(self, fake_image):
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = json.dumps({
            "improved": True,
            "differences": ["sharper details"],
            "quality_delta": 0.15,
            "recommendation": "The change improved quality.",
        })
        mock_response.content = [mock_block]

        with patch("agent.brain.vision.anthropic.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = json.loads(vision.handle("compare_outputs", {
                "image_a": fake_image,
                "image_b": fake_image,
                "change_description": "Increased steps from 20 to 30",
            }))

        assert result["improved"] is True
        assert result["quality_delta"] == 0.15


class TestSuggestImprovements:
    def test_suggest_calls_api(self, fake_image):
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = json.dumps({
            "suggestions": [
                {
                    "parameter": "steps",
                    "current_value": "20",
                    "suggested_value": "30",
                    "reason": "More detail",
                    "expected_impact": "Sharper output",
                    "confidence": 0.8,
                },
            ],
            "priority_order": ["steps"],
        })
        mock_response.content = [mock_block]

        with patch("agent.brain.vision.anthropic.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = json.loads(vision.handle("suggest_improvements", {
                "image_path": fake_image,
                "workflow_summary": "SDXL, 20 steps, Euler, CFG 7.0",
                "goal": "more detail",
            }))

        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["parameter"] == "steps"
