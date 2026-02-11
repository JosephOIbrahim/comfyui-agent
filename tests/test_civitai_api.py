"""Tests for CivitAI API discovery tools."""

import json
from unittest.mock import patch, MagicMock


from agent.tools import civitai_api


def _mock_response(data, status_code=200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.json.return_value = data
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client(response):
    """Create a mock httpx.Client context manager."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get.return_value = response
    return client


SAMPLE_MODEL = {
    "id": 12345,
    "name": "Realistic Vision V6.0",
    "type": "Checkpoint",
    "nsfw": False,
    "tags": ["realistic", "photorealistic"],
    "creator": {"username": "SG_161222"},
    "stats": {
        "downloadCount": 1543210,
        "favoriteCount": 89012,
        "rating": 4.82,
        "ratingCount": 2341,
    },
    "modelVersions": [
        {
            "id": 67890,
            "name": "V6.0 B1",
            "baseModel": "SD 1.5",
            "createdAt": "2024-01-15T12:00:00Z",
            "downloadUrl": "https://civitai.com/api/download/models/67890",
            "files": [
                {
                    "name": "realisticVisionV60B1_v60B1.safetensors",
                    "sizeKB": 2048000,
                    "metadata": {"format": "SafeTensor"},
                },
            ],
            "images": [
                {"url": "https://image.civitai.com/example1.jpg"},
                {"url": "https://image.civitai.com/example2.jpg"},
            ],
        },
    ],
}


class TestSearchCivitai:
    @patch("agent.tools.civitai_api.httpx.Client")
    def test_basic_search(self, mock_client_cls):
        mock_client_cls.return_value = _mock_client(
            _mock_response({"items": [SAMPLE_MODEL], "metadata": {"totalItems": 1}})
        )
        result = json.loads(civitai_api.handle("search_civitai", {"query": "realistic"}))
        assert result["source"] == "civitai"
        assert result["showing"] == 1
        assert result["results"][0]["name"] == "Realistic Vision V6.0"
        assert result["results"][0]["rating"] == 4.82
        assert result["results"][0]["downloads"] == 1543210

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_type_filter(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api.handle("search_civitai", {
            "query": "portrait",
            "model_type": "lora",
        })
        # Verify the type parameter was passed
        call_args = client.get.call_args
        assert call_args[1]["params"]["types"] == "LORA"

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_base_model_filter(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api.handle("search_civitai", {
            "query": "anime",
            "base_model": "sdxl",
        })
        call_args = client.get.call_args
        assert call_args[1]["params"]["baseModels"] == "SDXL 1.0"

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_sort_and_period(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api.handle("search_civitai", {
            "query": "landscape",
            "sort": "highest_rated",
            "period": "month",
        })
        call_args = client.get.call_args
        assert call_args[1]["params"]["sort"] == "Highest Rated"
        assert call_args[1]["params"]["period"] == "Month"

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_connection_error(self, mock_client_cls):
        client = _mock_client(None)
        client.get.side_effect = civitai_api.httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = client
        result = json.loads(civitai_api.handle("search_civitai", {"query": "test"}))
        assert "error" in result
        assert "CivitAI" in result["error"]

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_max_results_capped(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api.handle("search_civitai", {
            "query": "test",
            "max_results": 100,
        })
        call_args = client.get.call_args
        assert call_args[1]["params"]["limit"] == 20

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_installed_detection(self, mock_client_cls):
        mock_client_cls.return_value = _mock_client(
            _mock_response({"items": [SAMPLE_MODEL], "metadata": {"totalItems": 1}})
        )
        # Model not installed by default (MODELS_DIR doesn't exist in test)
        result = json.loads(civitai_api.handle("search_civitai", {"query": "realistic"}))
        assert result["results"][0]["installed"] is False


class TestGetCivitaiModel:
    @patch("agent.tools.civitai_api.httpx.Client")
    def test_get_detail(self, mock_client_cls):
        mock_client_cls.return_value = _mock_client(_mock_response(SAMPLE_MODEL))
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": 12345}))
        assert result["id"] == 12345
        assert result["name"] == "Realistic Vision V6.0"
        assert len(result["versions"]) == 1
        assert result["versions"][0]["base_model"] == "SD 1.5"
        assert len(result["versions"][0]["example_images"]) == 2
        assert result["stats"]["downloads"] == 1543210

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_model_not_found(self, mock_client_cls):
        import httpx as real_httpx
        resp = MagicMock()
        resp.status_code = 404
        client = _mock_client(resp)
        client.get.side_effect = real_httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=resp,
        )
        mock_client_cls.return_value = client
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": 99999}))
        assert "error" in result
        assert "not found" in result["error"]

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_description_stripped(self, mock_client_cls):
        model = dict(SAMPLE_MODEL)
        model["description"] = "<p>This is a <strong>great</strong> model.</p>"
        mock_client_cls.return_value = _mock_client(_mock_response(model))
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": 12345}))
        assert "<" not in result["description"]
        assert "great" in result["description"]


class TestGetTrending:
    @patch("agent.tools.civitai_api.httpx.Client")
    def test_trending_default(self, mock_client_cls):
        client = _mock_client(
            _mock_response({"items": [SAMPLE_MODEL], "metadata": {"totalItems": 1}})
        )
        mock_client_cls.return_value = client
        result = json.loads(civitai_api.handle("get_trending_models", {}))
        assert result["source"] == "civitai_trending"
        assert result["period"] == "week"
        assert result["showing"] == 1
        # Verify sort params
        call_args = client.get.call_args
        assert call_args[1]["params"]["sort"] == "Most Downloaded"
        assert call_args[1]["params"]["period"] == "Week"

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_trending_with_filters(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api.handle("get_trending_models", {
            "model_type": "lora",
            "base_model": "flux",
            "period": "day",
        })
        call_args = client.get.call_args
        assert call_args[1]["params"]["types"] == "LORA"
        assert call_args[1]["params"]["baseModels"] == "Flux.1 D"
        assert call_args[1]["params"]["period"] == "Day"


class TestHelpers:
    def test_strip_html(self):
        assert civitai_api._strip_html("<p>Hello <b>world</b></p>") == "Hello world"
        assert civitai_api._strip_html("") == ""
        assert civitai_api._strip_html("no tags") == "no tags"

    def test_type_map_coverage(self):
        """All our types should map to CivitAI types."""
        for t in ["checkpoint", "lora", "controlnet", "embedding", "vae"]:
            assert t in civitai_api._TYPE_MAP

    def test_base_model_map_coverage(self):
        for bm in ["sd15", "sdxl", "flux", "sd3", "pony"]:
            assert bm in civitai_api._BASE_MODEL_MAP

    def test_parse_model_summary(self):
        result = civitai_api._parse_model(SAMPLE_MODEL)
        assert result["id"] == 12345
        assert result["creator"] == "SG_161222"
        assert result["base_model"] == "SD 1.5"
        assert "civitai.com" in result["url"]


class TestRegistration:
    def test_tools_registered(self):
        names = [t["name"] for t in civitai_api.TOOLS]
        assert "search_civitai" in names
        assert "get_civitai_model" in names
        assert "get_trending_models" in names

    def test_dispatch_unknown(self):
        result = json.loads(civitai_api.handle("nonexistent", {}))
        assert "error" in result
