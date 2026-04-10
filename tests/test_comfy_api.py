"""Tests for comfy_api tool — mocked HTTP, no real ComfyUI needed."""

import json
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


class TestAutogrowNodeInfo:
    """Test that get_node_info annotates COMFY_AUTOGROW_V3 inputs."""

    def test_autogrow_hints_present(self):
        mock_data = {
            "ComfyMathExpression": {
                "display_name": "Math Expression",
                "category": "math",
                "description": "Evaluate math expressions",
                "input": {
                    "required": {
                        "expression": ["STRING", {"default": "a + b", "multiline": True}],
                        "values": [
                            "COMFY_AUTOGROW_V3",
                            {
                                "template": {
                                    "input": {
                                        "required": {
                                            "value": ["FLOAT,INT", {}],
                                        },
                                    },
                                },
                                "names": ["a", "b", "c", "d", "e"],
                                "min": 1,
                            },
                        ],
                    },
                    "optional": {},
                },
                "output": ["FLOAT", "INT"],
                "output_name": ["FLOAT", "INT"],
                "output_is_list": [False, False],
            },
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_node_info", {
                "node_type": "ComfyMathExpression",
            }))
        assert "autogrow_inputs" in result
        hints = result["autogrow_inputs"]
        assert "values" in hints
        assert hints["values"]["type"] == "COMFY_AUTOGROW_V3"
        assert hints["values"]["sub_input_type"] == "FLOAT,INT"
        assert hints["values"]["min"] == 1
        assert "values.a" in hints["values"]["usage"]
        assert "a" in hints["values"]["template_names"]

    def test_no_autogrow_hints_for_normal_nodes(self):
        mock_data = {
            "KSampler": {
                "display_name": "KSampler",
                "category": "sampling",
                "description": "",
                "input": {
                    "required": {
                        "model": ["MODEL", {}],
                        "seed": ["INT", {"default": 0}],
                    },
                    "optional": {},
                },
                "output": ["LATENT"],
                "output_name": ["LATENT"],
                "output_is_list": [False],
            },
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_node_info", {
                "node_type": "KSampler",
            }))
        assert "autogrow_inputs" not in result


# ---------------------------------------------------------------------------
# Cycle 42 — malformed node entry guard
# ---------------------------------------------------------------------------

class TestGetAllNodesMalformedEntry:
    """Guard against non-dict entries in /object_info response."""

    def test_non_dict_node_entry_is_skipped(self):
        """If a node entry is a string (not dict), it must be skipped, not crash."""
        import json
        mock_data = {
            "GoodNode": {
                "input": {"required": {}, "optional": {}},
                "output": [], "output_name": [], "output_is_list": [],
                "name": "GoodNode", "display_name": "Good Node",
                "description": "", "category": "sampling",
                "output_node": False,
            },
            "BadNode": "this should be a dict but is a string",
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_all_nodes", {"format": "names_only"}))
        assert "error" not in result
        assert "GoodNode" in result["nodes"]
        assert "BadNode" not in result["nodes"]

    def test_none_node_entry_is_skipped(self):
        """If a node entry is None, it must be skipped."""
        import json
        mock_data = {
            "ValidNode": {
                "input": {"required": {}, "optional": {}},
                "output": [], "output_name": [], "output_is_list": [],
                "name": "ValidNode", "display_name": "Valid Node",
                "description": "", "category": "latent",
                "output_node": False,
            },
            "NullNode": None,
        }
        with patch("agent.tools.comfy_api._get", return_value=mock_data):
            result = json.loads(comfy_api.handle("get_all_nodes", {"format": "names_only"}))
        assert "error" not in result
        assert "ValidNode" in result["nodes"]
        assert "NullNode" not in result["nodes"]


# ---------------------------------------------------------------------------
# Cycle 43 — resp.json() JSONDecodeError guard
# ---------------------------------------------------------------------------

class TestGetJsonDecodeGuard:
    """_get() must convert non-JSON response bodies into ConnectError, not crash."""

    def _make_bad_json_resp(self):
        """Create a mock response whose .json() raises ValueError (HTML page etc)."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("No JSON object could be decoded")
        return resp

    def test_non_json_response_raises_connect_error(self):
        """_get() with non-JSON body raises httpx.ConnectError, not ValueError."""
        import httpx

        with patch("agent.tools.comfy_api._get_client") as mock_client, \
             patch("agent.tools.comfy_api.COMFYUI_BREAKER") as mock_breaker:
            mock_breaker.return_value.allow_request.return_value = True
            mock_breaker.return_value.record_success.return_value = None
            mock_client.return_value.get.return_value = self._make_bad_json_resp()

            try:
                from agent.tools.comfy_api import _get
                _get("/test_path")
                assert False, "Should have raised"
            except httpx.ConnectError as e:
                assert "non-json" in str(e).lower() or "json" in str(e).lower()
            except ValueError:
                assert False, "_get() leaked a raw ValueError — must be ConnectError"

    def test_valid_json_response_passes_through(self):
        """_get() with valid JSON response returns the parsed dict unchanged."""
        with patch("agent.tools.comfy_api._get_client") as mock_client, \
             patch("agent.tools.comfy_api.COMFYUI_BREAKER") as mock_breaker:
            mock_breaker.return_value.allow_request.return_value = True
            mock_breaker.return_value.record_success.return_value = None
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"status": "ok", "nodes": 42}
            mock_client.return_value.get.return_value = resp

            from agent.tools.comfy_api import _get
            result = _get("/valid_path")
        assert result == {"status": "ok", "nodes": 42}


# ---------------------------------------------------------------------------
# Cycle 46 — get_node_info required field guard
# ---------------------------------------------------------------------------

class TestGetNodeInfoRequiredField:
    """get_node_info must return a structured error when node_type is missing or invalid."""

    def test_missing_node_type_returns_error(self):
        from agent.tools import comfy_api
        result = json.loads(comfy_api.handle("get_node_info", {}))
        assert "error" in result
        assert "node_type" in result["error"].lower()

    def test_empty_node_type_returns_error(self):
        from agent.tools import comfy_api
        result = json.loads(comfy_api.handle("get_node_info", {"node_type": ""}))
        assert "error" in result

    def test_none_node_type_returns_error(self):
        from agent.tools import comfy_api
        result = json.loads(comfy_api.handle("get_node_info", {"node_type": None}))
        assert "error" in result

    def test_integer_node_type_returns_error(self):
        from agent.tools import comfy_api
        result = json.loads(comfy_api.handle("get_node_info", {"node_type": 42}))
        assert "error" in result

    def test_valid_node_type_not_blocked(self):
        """Guard must not block valid string node_type values (ComfyUI offline → HTTP error)."""
        from unittest.mock import patch
        from agent.tools import comfy_api
        import httpx
        # Simulate ComfyUI being offline — the guard should be passed (error from HTTP, not guard)
        with patch("agent.tools.comfy_api._get", side_effect=httpx.ConnectError("offline")):
            result = json.loads(comfy_api.handle("get_node_info", {"node_type": "KSampler"}))
        assert "node_type" not in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# Cycle 54 — comfy_api null-safety and type guards
# ---------------------------------------------------------------------------

class TestGetQueueNullSafety:
    """get_queue_status must handle null API values for queue_running / queue_pending."""

    def test_null_queue_running_does_not_crash(self):
        from unittest.mock import patch
        from agent.tools import comfy_api
        with patch("agent.tools.comfy_api._get", return_value={
            "queue_running": None,
            "queue_pending": None,
        }):
            result = json.loads(comfy_api.handle("get_queue_status", {}))
        assert "error" not in result
        assert result["running_count"] == 0
        assert result["pending_count"] == 0

    def test_missing_queue_keys_default_to_empty(self):
        from unittest.mock import patch
        from agent.tools import comfy_api
        with patch("agent.tools.comfy_api._get", return_value={}):
            result = json.loads(comfy_api.handle("get_queue_status", {}))
        assert result["running_count"] == 0
        assert result["pending_count"] == 0


class TestGetHistoryOptionalField:
    """get_history must type-check optional prompt_id."""

    def test_integer_prompt_id_returns_error(self):
        from agent.tools import comfy_api
        result = json.loads(comfy_api.handle("get_history", {"prompt_id": 42}))
        assert "error" in result
        assert "prompt_id" in result["error"].lower()

    def test_none_prompt_id_allowed(self):
        """prompt_id=None must be allowed (fetches full history)."""
        from unittest.mock import patch
        from agent.tools import comfy_api
        with patch("agent.tools.comfy_api._get", return_value={}):
            result = json.loads(comfy_api.handle("get_history", {"prompt_id": None}))
        assert "prompt_id" not in result.get("error", "").lower()

    def test_omitted_prompt_id_allowed(self):
        """Omitting prompt_id must fetch full history without error."""
        from unittest.mock import patch
        from agent.tools import comfy_api
        with patch("agent.tools.comfy_api._get", return_value={}):
            result = json.loads(comfy_api.handle("get_history", {}))
        assert "prompt_id" not in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# Cycle 67: input validation guards
# ---------------------------------------------------------------------------

class TestGetHistoryCycle67Guards:
    """Cycle 67: get_history max_items must reject non-integer values."""

    def test_string_max_items_returns_error(self):
        """String max_items ('ten') must return JSON error, not TypeError in slice."""
        from agent.tools import comfy_api
        result = json.loads(comfy_api.handle("get_history", {"max_items": "ten"}))
        assert "error" in result
        assert "max_items" in result["error"].lower()

    def test_float_string_max_items_returns_error(self):
        """Float string '5.5' must return error (int() rejects it)."""
        from agent.tools import comfy_api
        result = json.loads(comfy_api.handle("get_history", {"max_items": "5.5"}))
        assert "error" in result
        assert "max_items" in result["error"].lower()

    def test_integer_max_items_not_blocked(self):
        """Integer max_items must not trigger the type guard."""
        from unittest.mock import patch
        from agent.tools import comfy_api
        with patch("agent.tools.comfy_api._get", return_value={}):
            result = json.loads(comfy_api.handle("get_history", {"max_items": 3}))
        assert result.get("error", "") != "max_items must be an integer."


class TestIsRunningCycle67Guards:
    """Cycle 67: is_comfyui_running must handle non-dict device entries."""

    def test_non_dict_device_does_not_crash(self):
        """devices list with string element must not raise AttributeError."""
        from unittest.mock import patch
        from agent.tools import comfy_api
        malformed_stats = {"devices": ["gpu_name_string"], "system": {}}
        with patch("agent.tools.comfy_api._get", return_value=malformed_stats):
            result = json.loads(comfy_api.handle("is_comfyui_running", {}))
        assert result.get("running") is True
        assert "no gpu" in result.get("gpu", "").lower()

    def test_none_device_entry_does_not_crash(self):
        """devices list with None element must not raise AttributeError."""
        from unittest.mock import patch
        from agent.tools import comfy_api
        malformed_stats = {"devices": [None], "system": {}}
        with patch("agent.tools.comfy_api._get", return_value=malformed_stats):
            result = json.loads(comfy_api.handle("is_comfyui_running", {}))
        assert result.get("running") is True
        assert "no gpu" in result.get("gpu", "").lower()


# ---------------------------------------------------------------------------
# Cycle 68: is_comfyui_running ConnectError gives actionable message
# ---------------------------------------------------------------------------

class TestIsRunningCycle68ErrorMessage:
    """Cycle 68: ConnectError must return a user-actionable message, not str(e)."""

    def test_connect_error_returns_actionable_message(self):
        """httpx.ConnectError on is_comfyui_running must mention 'Start ComfyUI'."""
        import httpx
        from agent.tools import comfy_api
        with patch("agent.tools.comfy_api._get", side_effect=httpx.ConnectError("refused")):
            result = json.loads(comfy_api.handle("is_comfyui_running", {}))
        assert result.get("running") is False
        error_msg = result.get("error", "")
        assert "start comfyui" in error_msg.lower() or "not running" in error_msg.lower()

    def test_connect_error_does_not_expose_raw_socket_error(self):
        """ConnectError must not pass raw socket error text to the artist."""
        import httpx
        from agent.tools import comfy_api
        with patch("agent.tools.comfy_api._get",
                   side_effect=httpx.ConnectError("ECONNREFUSED 127.0.0.1:8188")):
            result = json.loads(comfy_api.handle("is_comfyui_running", {}))
        error_msg = result.get("error", "")
        assert "econnrefused" not in error_msg.lower()
        assert "connecterror" not in error_msg.lower()
