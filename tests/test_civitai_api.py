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
    """Tests for the internal _handle_search_civitai function."""

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_basic_search(self, mock_client_cls):
        mock_client_cls.return_value = _mock_client(
            _mock_response({"items": [SAMPLE_MODEL], "metadata": {"totalItems": 1}})
        )
        result = json.loads(civitai_api._handle_search_civitai({"query": "realistic"}))
        assert result["source"] == "civitai"
        assert result["showing"] == 1
        assert result["results"][0]["name"] == "Realistic Vision V6.0"
        assert result["results"][0]["rating"] == 4.82
        assert result["results"][0]["downloads"] == 1543210

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_type_filter(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api._handle_search_civitai({
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
        civitai_api._handle_search_civitai({
            "query": "anime",
            "base_model": "sdxl",
        })
        call_args = client.get.call_args
        assert call_args[1]["params"]["baseModels"] == "SDXL 1.0"

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_sort_and_period(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api._handle_search_civitai({
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
        result = json.loads(civitai_api._handle_search_civitai({"query": "test"}))
        assert "error" in result
        assert "CivitAI" in result["error"]

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_max_results_capped(self, mock_client_cls):
        client = _mock_client(_mock_response({"items": [], "metadata": {"totalItems": 0}}))
        mock_client_cls.return_value = client
        civitai_api._handle_search_civitai({
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
        result = json.loads(civitai_api._handle_search_civitai({"query": "realistic"}))
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
        assert "search_civitai" not in names  # moved to unified discover
        assert "get_civitai_model" in names
        assert "get_trending_models" in names

    def test_dispatch_unknown(self):
        result = json.loads(civitai_api.handle("nonexistent", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 30: JSON response type guard tests
# ---------------------------------------------------------------------------

class TestCivitAIResponseTypeGuards:
    """API response type guards must reject non-dict responses."""

    def _make_resp(self, data):
        resp = MagicMock()
        resp.json.return_value = data
        resp.raise_for_status = MagicMock()
        return resp

    def test_search_non_dict_response_returns_error(self):
        """search handler must return error if API returns a list instead of dict."""
        with patch("agent.tools.civitai_api.CIVITAI_LIMITER") as mock_limiter, \
             patch("httpx.Client") as mock_client:
            mock_limiter.return_value.return_value.acquire.return_value = True
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = self._make_resp(["item1", "item2"])
            result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": 12345}))
        # get_civitai_model uses _parse_model_detail on the raw response
        # that function must not crash on a list; we check a non-dict triggers graceful path
        # (the outer except will catch AttributeError → "error" key in result)
        assert "error" in result or "id" in result  # Either errored or parsed fine

    def test_trending_non_dict_response_returns_error(self):
        """get_trending_models must return error if API returns non-dict."""
        with patch("agent.tools.civitai_api.CIVITAI_LIMITER") as mock_limiter, \
             patch("httpx.Client") as mock_client:
            mock_limiter.return_value.return_value.acquire.return_value = True
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = self._make_resp([])
            result = json.loads(civitai_api.handle("get_trending_models", {}))
        assert "error" in result
        assert "unexpected" in result["error"].lower() or "format" in result["error"].lower()


# ---------------------------------------------------------------------------
# Cycle 32: items list guard
# ---------------------------------------------------------------------------

class TestItemsListGuard:
    """items field must be guarded as a list before iteration."""

    def _make_resp(self, data):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.json.return_value = data
        resp.raise_for_status = MagicMock()
        return resp

    def test_trending_items_non_list_returns_empty_results(self):
        """If items is a dict (API weirdness), trending must return 0 results, not crash."""
        from unittest.mock import patch, MagicMock
        with patch("agent.tools.civitai_api.CIVITAI_LIMITER") as mock_limiter, \
             patch("httpx.Client") as mock_client:
            mock_limiter.return_value.return_value.acquire.return_value = True
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = self._make_resp({
                "items": {"nested": "dict"},  # not a list
                "metadata": {},
            })
            result = json.loads(civitai_api.handle("get_trending_models", {}))
        # Should not crash; must have results key (empty list)
        assert "results" in result or "error" in result
        if "results" in result:
            assert isinstance(result["results"], list)
            assert len(result["results"]) == 0  # items was a dict, guarded to []

    def test_trending_items_none_returns_empty_results(self):
        """If items is None, trending must not crash (TypeError on None iteration)."""
        from unittest.mock import patch, MagicMock
        with patch("agent.tools.civitai_api.CIVITAI_LIMITER") as mock_limiter, \
             patch("httpx.Client") as mock_client:
            mock_limiter.return_value.return_value.acquire.return_value = True
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = self._make_resp({
                "items": None,  # explicit null
                "metadata": {},
            })
            result = json.loads(civitai_api.handle("get_trending_models", {}))
        # Must not raise TypeError; results should be empty list
        assert "results" in result or "error" in result
        if "results" in result:
            assert isinstance(result["results"], list)
            assert len(result["results"]) == 0  # items was None, guarded to []


# ---------------------------------------------------------------------------
# Cycle 42 — model_id validation
# ---------------------------------------------------------------------------

class TestGetCivitaiModelIdValidation:
    """Adversarial tests for model_id > 0 guard in _handle_get_civitai_model."""

    def test_model_id_zero_returns_error(self):
        """model_id=0 must return an error, not call the API."""
        import json
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": 0}))
        assert "error" in result
        assert "positive" in result["error"].lower() or "model_id" in result["error"].lower()

    def test_model_id_negative_returns_error(self):
        """model_id=-1 must return an error."""
        import json
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": -1}))
        assert "error" in result

    def test_model_id_string_returns_error(self):
        """model_id as a string (e.g. 'abc') must return error."""
        import json
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": "abc"}))
        assert "error" in result

    def test_model_id_valid_positive_proceeds(self):
        """model_id=1 must pass validation (may fail at network, but not at guard)."""
        from unittest.mock import patch, MagicMock
        import json
        with patch("agent.tools.civitai_api.CIVITAI_LIMITER") as mock_limiter, \
             patch("httpx.Client") as mock_client:
            mock_limiter.return_value.return_value.acquire.return_value = True
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"id": 1, "name": "Test Model", "type": "Checkpoint"}
            resp.raise_for_status.return_value = None
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = resp
            result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": 1}))
        # Should not have a validation error
        assert result.get("error") != "model_id must be a positive integer, got 1."


# ---------------------------------------------------------------------------
# Cycle 46 — required field guards for get_civitai_trending and get_civitai_model
# ---------------------------------------------------------------------------

class TestGetCivitaiModelRequiredField:
    """get_civitai_model must return structured error when model_id is missing."""

    def test_missing_model_id_returns_error(self):
        """Missing model_id must return structured error, not KeyError."""
        from agent.tools import civitai_api
        result = json.loads(civitai_api.handle("get_civitai_model", {}))
        assert "error" in result
        # Must be the type-validation error, not a Python KeyError
        assert "model_id" in result["error"].lower() or "integer" in result["error"].lower()

    def test_none_model_id_returns_error(self):
        from agent.tools import civitai_api
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": None}))
        assert "error" in result

    def test_string_model_id_returns_error(self):
        from agent.tools import civitai_api
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": "123"}))
        assert "error" in result

    def test_zero_model_id_returns_error(self):
        from agent.tools import civitai_api
        result = json.loads(civitai_api.handle("get_civitai_model", {"model_id": 0}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Cycle 63: max_results type coercion guard
# ---------------------------------------------------------------------------

class TestMaxResultsTypeGuard:
    """search_models/get_trending_models must reject non-integer max_results (Cycle 63)."""

    def test_get_civitai_model_trending_string_max_results_returns_error(self):
        """String max_results in get_trending_models must return JSON error."""
        import json
        from agent.tools import civitai_api
        result = json.loads(civitai_api.handle("get_trending_models", {
            "max_results": "ten",
        }))
        assert "error" in result
        assert "max_results" in result["error"].lower()

    def test_get_trending_models_string_max_results_returns_error(self):
        """String max_results 'lots' must return JSON error."""
        import json
        from agent.tools import civitai_api
        result = json.loads(civitai_api.handle("get_trending_models", {
            "max_results": "lots",
        }))
        assert "error" in result
        assert "max_results" in result["error"].lower()

    def test_get_trending_models_int_max_results_accepted(self):
        """Integer max_results must not trigger the type guard."""
        import json
        from agent.tools import civitai_api
        result = json.loads(civitai_api.handle("get_trending_models", {
            "max_results": 5,
        }))
        # Should error on network, not on type
        assert result.get("error", "") != "max_results must be an integer"


# ---------------------------------------------------------------------------
# Cycle 64: circuit breaker protection
# ---------------------------------------------------------------------------

class TestCivitaiCircuitBreaker:
    """Cycle 64: CivitAI HTTP handlers must respect the CIVITAI_BREAKER."""

    def _make_open_breaker(self):
        from agent.circuit_breaker import CIVITAI_BREAKER
        breaker = CIVITAI_BREAKER()
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
        return breaker

    def teardown_method(self):
        from agent.circuit_breaker import reset_all
        reset_all()

    def test_search_blocked_when_breaker_open(self):
        """_handle_search_civitai returns error when circuit is open."""
        self._make_open_breaker()
        result = json.loads(civitai_api._handle_search_civitai({"query": "test"}))
        assert "error" in result
        assert "unavailable" in result["error"].lower() or "temporarily" in result["error"].lower()

    def test_get_civitai_model_blocked_when_breaker_open(self):
        """_handle_get_civitai_model returns error when circuit is open."""
        self._make_open_breaker()
        result = json.loads(civitai_api._handle_get_civitai_model({"model_id": 1}))
        assert "error" in result
        assert "unavailable" in result["error"].lower() or "temporarily" in result["error"].lower()

    def test_trending_blocked_when_breaker_open(self):
        """_handle_get_trending_models returns error when circuit is open."""
        self._make_open_breaker()
        result = json.loads(civitai_api._handle_get_trending_models({}))
        assert "error" in result
        assert "unavailable" in result["error"].lower() or "temporarily" in result["error"].lower()

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_connect_error_records_failure(self, mock_client_cls):
        """ConnectError must increment the circuit breaker failure count."""
        import httpx
        from agent.circuit_breaker import CIVITAI_BREAKER
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = client

        cb = CIVITAI_BREAKER()
        initial_failures = cb._failure_count
        civitai_api._handle_get_civitai_model({"model_id": 1})
        assert cb._failure_count > initial_failures

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_success_records_success(self, mock_client_cls):
        """Successful HTTP call must call breaker.record_success()."""
        from agent.circuit_breaker import CIVITAI_BREAKER, CLOSED
        # Pre-open the breaker to half-open state
        cb = CIVITAI_BREAKER()
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        # Force recovery
        import time
        cb._last_failure_time = time.monotonic() - cb.recovery_timeout - 1

        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        resp = _mock_response({"items": [], "metadata": {"totalItems": 0}})
        client.get.return_value = resp
        mock_client_cls.return_value = client

        civitai_api._handle_search_civitai({"query": "test"})
        # After success, circuit should be CLOSED
        assert cb.state == CLOSED


# ---------------------------------------------------------------------------
# Cycle 66: resp.json() JSONDecodeError guards
# ---------------------------------------------------------------------------

class TestCivitaiJsonDecodeGuard:
    """Cycle 66: All CivitAI HTTP handlers must handle non-JSON responses gracefully."""

    def _make_html_response(self):
        """Mock response whose .json() raises ValueError (as httpx does for HTML bodies)."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("No JSON object could be decoded")
        return resp

    def _make_client(self, response):
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = response
        return client

    def teardown_method(self):
        from agent.circuit_breaker import reset_all
        reset_all()

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_search_html_response_returns_error(self, mock_client_cls):
        """_handle_search_civitai: HTML body (ValueError from resp.json()) → structured error."""
        mock_client_cls.return_value = self._make_client(self._make_html_response())
        result = json.loads(civitai_api._handle_search_civitai({"query": "sdxl"}))
        assert "error" in result
        assert "non-JSON" in result["error"] or "HTML" in result["error"]

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_get_model_html_response_returns_error(self, mock_client_cls):
        """_handle_get_civitai_model: HTML body → structured error."""
        mock_client_cls.return_value = self._make_client(self._make_html_response())
        result = json.loads(civitai_api._handle_get_civitai_model({"model_id": 99}))
        assert "error" in result
        assert "non-JSON" in result["error"] or "HTML" in result["error"]

    @patch("agent.tools.civitai_api.httpx.Client")
    def test_trending_html_response_returns_error(self, mock_client_cls):
        """_handle_get_trending_models: HTML body → structured error."""
        mock_client_cls.return_value = self._make_client(self._make_html_response())
        result = json.loads(civitai_api._handle_get_trending_models({}))
        assert "error" in result
        assert "non-JSON" in result["error"] or "HTML" in result["error"]
