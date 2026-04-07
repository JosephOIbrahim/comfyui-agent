"""Tests for the consolidated macro-tools.

[AUTONOMY x CRUCIBLE] — Tests for all 8 macro-tools.
"""

import pytest

from cognitive.tools.analyze import analyze_workflow
from cognitive.tools.mutate import mutate_workflow
from cognitive.tools.query import query_environment
from cognitive.tools.dependencies import manage_dependencies
from cognitive.tools.execute import execute_workflow, ExecutionStatus
from cognitive.tools.compose import compose_workflow
from cognitive.tools.series import generate_series, SeriesConfig
from cognitive.tools.research import autoresearch, AutoresearchConfig
from cognitive.core.graph import CognitiveGraphEngine
from cognitive.transport.schema_cache import SchemaCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_workflow():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a landscape", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "ugly", "clip": ["1", 1]}},
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42, "steps": 20, "cfg": 7.5,
                "sampler_name": "euler", "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0], "positive": ["2", 0],
                "negative": ["3", 0], "latent_image": ["5", 0],
            },
        },
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
    }


@pytest.fixture
def engine(sample_workflow):
    return CognitiveGraphEngine(sample_workflow)


@pytest.fixture
def schema_cache():
    cache = SchemaCache()
    cache.refresh({
        "KSampler": {
            "input": {
                "required": {
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "cfg": ["FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}],
                    "sampler_name": [["euler", "dpmpp_2m"], {}],
                },
            },
            "output": ["LATENT"],
        },
    })
    return cache


# ---------------------------------------------------------------------------
# analyze_workflow
# ---------------------------------------------------------------------------

class TestAnalyzeWorkflow:

    def test_basic_analysis(self, sample_workflow):
        result = analyze_workflow(sample_workflow)
        assert result.node_count == 6
        assert "KSampler" in result.node_types
        assert len(result.connections) > 0
        assert len(result.editable_fields) > 0

    def test_classification_txt2img(self, sample_workflow):
        result = analyze_workflow(sample_workflow)
        assert result.classification == "txt2img"

    def test_model_family_detection(self, sample_workflow):
        result = analyze_workflow(sample_workflow)
        assert result.model_family == "SD1.5"

    def test_summary_present(self, sample_workflow):
        result = analyze_workflow(sample_workflow)
        assert "txt2img" in result.summary
        assert "6 nodes" in result.summary

    def test_schema_validation(self, sample_workflow, schema_cache):
        # Inject invalid value
        sample_workflow["4"]["inputs"]["sampler_name"] = "bad_sampler"
        result = analyze_workflow(sample_workflow, schema_cache=schema_cache)
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_empty_workflow(self):
        result = analyze_workflow({})
        assert result.node_count == 0

    def test_sdxl_detection(self):
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl-base.safetensors"}},
        }
        result = analyze_workflow(wf)
        assert result.model_family == "SDXL"


# ---------------------------------------------------------------------------
# mutate_workflow
# ---------------------------------------------------------------------------

class TestMutateWorkflow:

    def test_valid_mutation(self, engine):
        result = mutate_workflow(engine, {"4": {"steps": 30}})
        assert result.success is True
        assert len(result.changes) == 1
        assert result.delta_layer_id != ""

    def test_schema_validated_mutation(self, engine, schema_cache):
        result = mutate_workflow(
            engine, {"4": {"sampler_name": "bad"}}, schema_cache=schema_cache,
        )
        assert result.success is False
        assert len(result.validation_errors) > 0

    def test_valid_with_schema(self, engine, schema_cache):
        result = mutate_workflow(
            engine, {"4": {"sampler_name": "euler"}}, schema_cache=schema_cache,
        )
        assert result.success is True

    def test_multi_node_mutation(self, engine):
        result = mutate_workflow(engine, {
            "4": {"steps": 30, "cfg": 5.0},
            "2": {"text": "new prompt"},
        })
        assert result.success is True
        assert len(result.changes) == 3


# ---------------------------------------------------------------------------
# query_environment
# ---------------------------------------------------------------------------

class TestQueryEnvironment:

    def test_full_snapshot(self):
        snap = query_environment(
            system_stats={"devices": [{"name": "RTX 4090", "vram_total": 25769803776, "vram_free": 20000000000}]},
            queue_info={"queue_running": [1], "queue_pending": [2, 3]},
            node_packs=["ComfyUI-Manager", "ComfyUI-Impact-Pack"],
            models={"checkpoints": ["v1-5.safetensors"]},
        )
        assert snap.comfyui_running is True
        assert snap.gpu_name == "RTX 4090"
        assert snap.vram_total_mb > 0
        assert snap.queue_running == 1
        assert snap.queue_pending == 2
        assert len(snap.installed_node_packs) == 2

    def test_empty_snapshot(self):
        snap = query_environment()
        assert snap.comfyui_running is False
        assert snap.gpu_name == ""

    def test_with_schema_cache(self, schema_cache):
        snap = query_environment(schema_cache=schema_cache)
        assert snap.schema_cached is True
        assert snap.node_count == 1


# ---------------------------------------------------------------------------
# manage_dependencies
# ---------------------------------------------------------------------------

class TestManageDependencies:

    def test_install_action(self):
        result = manage_dependencies("install", "ComfyUI-Impact-Pack")
        assert result.success is True
        assert result.action == "install"

    def test_invalid_action(self):
        result = manage_dependencies("delete", "some-pack")
        assert result.success is False

    def test_schema_invalidation(self, schema_cache):
        result = manage_dependencies("install", "pack", schema_cache=schema_cache)
        assert result.schema_invalidated is True


# ---------------------------------------------------------------------------
# execute_workflow
# ---------------------------------------------------------------------------

class TestExecuteWorkflow:

    def test_empty_workflow(self):
        result = execute_workflow({})
        assert result.status == ExecutionStatus.FAILED
        assert "Empty" in result.error

    def test_no_nodes_workflow(self):
        result = execute_workflow({"metadata": "not a node"})
        assert result.status == ExecutionStatus.FAILED
        assert "No nodes" in result.error

    def test_valid_workflow_returns_pending(self, sample_workflow):
        result = execute_workflow(sample_workflow)
        assert result.status == ExecutionStatus.PENDING
        assert result.prompt_id != ""

    def test_callback_called(self, sample_workflow):
        called = []
        execute_workflow(sample_workflow, on_complete=lambda r: called.append(r))
        assert len(called) == 1


# ---------------------------------------------------------------------------
# compose_workflow
# ---------------------------------------------------------------------------

class TestComposeWorkflow:

    def test_empty_intent(self):
        result = compose_workflow("")
        assert result.success is False

    def test_basic_composition(self):
        result = compose_workflow("a beautiful sunset")
        assert result.success is True
        assert result.plan is not None
        assert result.plan.model_family == "SD1.5"

    def test_flux_detection(self):
        result = compose_workflow("flux style portrait")
        assert result.plan.model_family == "Flux"

    def test_sdxl_detection(self):
        result = compose_workflow("SDXL quality landscape")
        assert result.plan.model_family == "SDXL"

    def test_photorealistic_params(self):
        result = compose_workflow("photorealistic portrait")
        assert result.plan.parameters.get("cfg") == 7.5
        assert result.plan.parameters.get("steps") == 30

    def test_dreamy_params(self):
        result = compose_workflow("dreamy ethereal scene")
        assert result.plan.parameters.get("cfg") == 5.0

    def test_experience_patterns_applied(self):
        patterns = [{"confidence": 0.9, "parameters": {"steps": 40}}]
        result = compose_workflow("test", experience_patterns=patterns)
        assert result.plan.parameters["steps"] == 40
        assert result.plan.confidence == 0.9

    def test_explicit_model_family(self):
        result = compose_workflow("test", model_family="SD3")
        assert result.plan.model_family == "SD3"


# ---------------------------------------------------------------------------
# generate_series
# ---------------------------------------------------------------------------

class TestGenerateSeries:

    def test_empty_workflow(self):
        result = generate_series(SeriesConfig())
        assert result.success is False

    def test_no_variation(self):
        result = generate_series(SeriesConfig(base_workflow={"1": {"class_type": "N", "inputs": {}}}))
        assert result.success is False

    def test_basic_series(self):
        config = SeriesConfig(
            base_workflow={"1": {"class_type": "KSampler", "inputs": {"seed": 42}}},
            vary_params={"1.seed": [1, 2, 3, 4]},
            count=4,
        )
        result = generate_series(config)
        assert result.success is True
        assert result.planned_count == 4
        assert len(result.variations) == 4

    def test_locked_params(self):
        config = SeriesConfig(
            base_workflow={"1": {"class_type": "KSampler", "inputs": {}}},
            vary_params={"1.seed": [1, 2]},
            lock_params={"1.cfg": 7.0},
            count=2,
        )
        result = generate_series(config)
        for v in result.variations:
            assert v["mutations"]["1"]["cfg"] == 7.0


# ---------------------------------------------------------------------------
# autoresearch
# ---------------------------------------------------------------------------

class TestAutoresearch:

    def test_no_evaluator(self, engine):
        config = AutoresearchConfig(max_steps=5)
        result = autoresearch(engine, config, initial_quality=0.5)
        assert result.stopped_reason == "no_evaluator"
        assert result.steps_taken == 1

    def test_quality_threshold(self, engine):
        config = AutoresearchConfig(max_steps=10, quality_threshold=0.5)
        result = autoresearch(engine, config, initial_quality=0.8)
        assert result.stopped_reason == "quality_threshold_reached"
        assert result.steps_taken == 0

    def test_max_steps(self, engine):
        config = AutoresearchConfig(
            max_steps=3,
            quality_evaluator=lambda: 0.5,
        )
        result = autoresearch(engine, config, initial_quality=0.3)
        assert result.stopped_reason == "max_steps_reached"
        assert result.steps_taken == 3
