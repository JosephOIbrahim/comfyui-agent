"""Tests for the transport hardening layer.

[TRANSPORT x CRUCIBLE] — Schema cache, execution events, interrupt.
All tests mocked — no live ComfyUI dependency.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from src.cognitive.transport.schema_cache import SchemaCache, NodeSchema, InputSpec
from src.cognitive.transport.events import ExecutionEvent, EventType
from src.cognitive.transport.interrupt import interrupt_execution, get_system_stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_object_info():
    """Minimal /object_info response for testing."""
    return {
        "KSampler": {
            "display_name": "KSampler",
            "category": "sampling",
            "description": "Samples from a model.",
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "seed": ["INT", {"default": 0, "min": 0, "max": 2**32}],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "cfg": ["FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}],
                    "sampler_name": [
                        ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"],
                        {"default": "euler"},
                    ],
                    "scheduler": [
                        ["normal", "karras", "exponential", "sgm_uniform"],
                        {"default": "normal"},
                    ],
                    "positive": ["CONDITIONING"],
                    "negative": ["CONDITIONING"],
                    "latent_image": ["LATENT"],
                    "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
                },
                "optional": {},
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
        },
        "CheckpointLoaderSimple": {
            "display_name": "Load Checkpoint",
            "category": "loaders",
            "description": "Loads a checkpoint.",
            "input": {
                "required": {
                    "ckpt_name": [
                        ["v1-5-pruned.safetensors", "sdxl-base.safetensors"],
                        {},
                    ],
                },
            },
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
        },
        "CLIPTextEncode": {
            "display_name": "CLIP Text Encode",
            "category": "conditioning",
            "description": "Encodes text.",
            "input": {
                "required": {
                    "text": ["STRING", {"multiline": True}],
                    "clip": ["CLIP"],
                },
            },
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
        },
    }


@pytest.fixture
def cache(sample_object_info):
    """A populated SchemaCache."""
    c = SchemaCache()
    c.refresh(sample_object_info)
    return c


# ---------------------------------------------------------------------------
# SchemaCache Tests
# ---------------------------------------------------------------------------

class TestSchemaCache:

    def test_refresh_populates(self, cache):
        assert cache.is_populated is True
        assert cache.node_count == 3

    def test_refresh_returns_count(self, sample_object_info):
        c = SchemaCache()
        count = c.refresh(sample_object_info)
        assert count == 3

    def test_empty_cache(self):
        c = SchemaCache()
        assert c.is_populated is False
        assert c.node_count == 0

    def test_get_schema(self, cache):
        schema = cache.get_schema("KSampler")
        assert schema is not None
        assert schema.class_type == "KSampler"
        assert schema.category == "sampling"

    def test_get_schema_missing(self, cache):
        assert cache.get_schema("NonExistent") is None

    def test_list_node_types(self, cache):
        types = cache.list_node_types()
        assert "CheckpointLoaderSimple" in types
        assert "KSampler" in types
        assert "CLIPTextEncode" in types

    def test_last_refresh_timestamp(self, cache):
        assert cache.last_refresh > 0


class TestSchemaValidation:

    def test_valid_combo_value(self, cache):
        ok, reason = cache.validate_mutation("KSampler", "sampler_name", "euler")
        assert ok is True
        assert reason == ""

    def test_invalid_combo_value(self, cache):
        ok, reason = cache.validate_mutation("KSampler", "sampler_name", "nonexistent_sampler")
        assert ok is False
        assert "must be one of" in reason

    def test_valid_numeric_range(self, cache):
        ok, reason = cache.validate_mutation("KSampler", "steps", 50)
        assert ok is True

    def test_numeric_below_min(self, cache):
        ok, reason = cache.validate_mutation("KSampler", "steps", 0)
        assert ok is False
        assert "minimum" in reason

    def test_numeric_above_max(self, cache):
        ok, reason = cache.validate_mutation("KSampler", "denoise", 2.0)
        assert ok is False
        assert "maximum" in reason

    def test_unknown_node_type(self, cache):
        ok, reason = cache.validate_mutation("FakeNode", "param", "val")
        assert ok is False
        assert "Unknown node type" in reason

    def test_unknown_param_passes(self, cache):
        """Unknown params pass validation (could be connection inputs)."""
        ok, reason = cache.validate_mutation("KSampler", "unknown_param", "val")
        assert ok is True

    def test_get_valid_values(self, cache):
        vals = cache.get_valid_values("KSampler", "sampler_name")
        assert "euler" in vals
        assert "dpmpp_2m" in vals

    def test_get_valid_values_non_combo(self, cache):
        assert cache.get_valid_values("KSampler", "steps") is None

    def test_get_valid_values_missing_node(self, cache):
        assert cache.get_valid_values("FakeNode", "param") is None


class TestConnectableNodes:

    def test_find_model_producers(self, cache):
        """CheckpointLoaderSimple outputs MODEL, so it can connect to KSampler.model."""
        connectable = cache.get_connectable_nodes("KSampler", "model")
        assert "CheckpointLoaderSimple" in connectable

    def test_find_conditioning_producers(self, cache):
        connectable = cache.get_connectable_nodes("KSampler", "positive")
        assert "CLIPTextEncode" in connectable

    def test_literal_type_returns_empty(self, cache):
        """INT/FLOAT/STRING inputs are not connectable."""
        connectable = cache.get_connectable_nodes("KSampler", "steps")
        assert connectable == []

    def test_missing_node_returns_empty(self, cache):
        assert cache.get_connectable_nodes("FakeNode", "input") == []


# ---------------------------------------------------------------------------
# InputSpec Tests
# ---------------------------------------------------------------------------

class TestInputSpec:

    def test_combo_type(self):
        spec = InputSpec.from_object_info(
            "sampler",
            [["euler", "dpm"], {"default": "euler"}],
        )
        assert spec.input_type == "COMBO"
        assert spec.valid_values == ["euler", "dpm"]
        assert spec.default == "euler"

    def test_int_type(self):
        spec = InputSpec.from_object_info(
            "steps",
            ["INT", {"default": 20, "min": 1, "max": 100}],
        )
        assert spec.input_type == "INT"
        assert spec.default == 20
        assert spec.min_val == 1
        assert spec.max_val == 100

    def test_connection_type(self):
        spec = InputSpec.from_object_info("model", ["MODEL"])
        assert spec.input_type == "MODEL"
        assert spec.valid_values is None

    def test_empty_spec(self):
        spec = InputSpec.from_object_info("x", [])
        assert spec.input_type == "UNKNOWN"

    def test_non_list_spec(self):
        spec = InputSpec.from_object_info("x", "not_a_list")
        assert spec.input_type == "UNKNOWN"


# ---------------------------------------------------------------------------
# ExecutionEvent Tests
# ---------------------------------------------------------------------------

class TestExecutionEvent:

    def test_progress_event(self):
        msg = {"type": "progress", "data": {"value": 5, "max": 20, "prompt_id": "abc"}}
        event = ExecutionEvent.from_ws_message(msg, started_at=time.time() - 1.0)
        assert event.event_type == EventType.PROGRESS
        assert event.progress_value == 5
        assert event.progress_max == 20
        assert event.progress_pct == 25.0
        assert event.elapsed_ms > 0

    def test_executing_with_node(self):
        msg = {"type": "executing", "data": {"node": "4", "prompt_id": "abc"}}
        event = ExecutionEvent.from_ws_message(msg)
        assert event.event_type == EventType.EXECUTING
        assert event.node_id == "4"

    def test_executing_none_node_is_complete(self):
        """executing with node=None signals completion."""
        msg = {"type": "executing", "data": {"node": None, "prompt_id": "abc"}}
        event = ExecutionEvent.from_ws_message(msg)
        assert event.event_type == EventType.EXECUTION_COMPLETE
        assert event.is_terminal is True
        assert event.is_error is False

    def test_error_event(self):
        msg = {"type": "execution_error", "data": {"prompt_id": "abc", "node": "4"}}
        event = ExecutionEvent.from_ws_message(msg)
        assert event.event_type == EventType.EXECUTION_ERROR
        assert event.is_terminal is True
        assert event.is_error is True

    def test_start_event_sets_started_at(self):
        msg = {"type": "execution_start", "data": {"prompt_id": "abc"}}
        event = ExecutionEvent.from_ws_message(msg)
        assert event.event_type == EventType.EXECUTION_START
        assert event.started_at > 0

    def test_unknown_event_type(self):
        msg = {"type": "some_new_type", "data": {}}
        event = ExecutionEvent.from_ws_message(msg)
        assert event.event_type == EventType.UNKNOWN

    def test_progress_pct_zero_max(self):
        event = ExecutionEvent(event_type=EventType.PROGRESS, progress_max=0, progress_value=5)
        assert event.progress_pct == 0.0

    def test_progress_pct_capped_at_100(self):
        event = ExecutionEvent(
            event_type=EventType.PROGRESS, progress_max=10, progress_value=20,
        )
        assert event.progress_pct == 100.0

    def test_elapsed_ms_no_start(self):
        event = ExecutionEvent(event_type=EventType.PROGRESS, started_at=0.0)
        assert event.elapsed_ms == 0.0

    def test_cached_event(self):
        msg = {"type": "execution_cached", "data": {"nodes": ["1", "2"], "prompt_id": "abc"}}
        event = ExecutionEvent.from_ws_message(msg)
        assert event.event_type == EventType.EXECUTION_CACHED


# ---------------------------------------------------------------------------
# Interrupt Tests (mocked)
# ---------------------------------------------------------------------------

class TestInterrupt:

    @patch("src.cognitive.transport.interrupt.httpx.post")
    def test_successful_interrupt(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        ok, msg = interrupt_execution()
        assert ok is True
        assert "interrupted" in msg.lower()

    @patch("src.cognitive.transport.interrupt.httpx.post")
    def test_interrupt_http_error(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        ok, msg = interrupt_execution()
        assert ok is False
        assert "500" in msg

    @patch("src.cognitive.transport.interrupt.httpx.post")
    def test_interrupt_connect_error(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.ConnectError("refused")
        ok, msg = interrupt_execution()
        assert ok is False
        assert "Could not connect" in msg

    @patch("src.cognitive.transport.interrupt.httpx.get")
    def test_system_stats_success(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"devices": [{"name": "RTX 4090"}]},
        )
        mock_get.return_value.raise_for_status = lambda: None
        stats = get_system_stats()
        assert "devices" in stats

    @patch("src.cognitive.transport.interrupt.httpx.get")
    def test_system_stats_error(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.ConnectError("refused")
        stats = get_system_stats()
        assert "error" in stats
