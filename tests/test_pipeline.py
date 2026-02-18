"""Tests for agent/tools/pipeline.py -- multi-workflow chaining engine."""

import json
import threading

import pytest

from agent.tools import pipeline
from agent.tools.pipeline import (
    _apply_input_mappings,
    _apply_param_overrides,
    _pipeline_lock,
    _pipeline_state,
    _validate_pipeline_definition,
)


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Reset module-level state between tests."""
    with _pipeline_lock:
        _pipeline_state["current_pipeline"] = None
        _pipeline_state["execution_results"] = []
        _pipeline_state["status"] = "idle"
        _pipeline_state["error"] = None
    yield


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestValidatePipelineDefinition:
    def test_valid_simple_pipeline(self):
        p = {"stages": [
            {"stage_id": "s1", "workflow_source": "template:txt2img_sdxl"},
        ]}
        assert _validate_pipeline_definition(p) is None

    def test_valid_two_stage_with_mapping(self):
        p = {"stages": [
            {"stage_id": "gen", "workflow_source": "template:txt2img_sdxl"},
            {"stage_id": "up", "workflow_source": "template:img2img",
             "input_mappings": [{"node_id": "2", "input_name": "image",
                                 "from_stage": "gen"}]},
        ]}
        assert _validate_pipeline_definition(p) is None

    def test_empty_stages_error(self):
        assert _validate_pipeline_definition({"stages": []}) is not None

    def test_missing_stages_error(self):
        assert _validate_pipeline_definition({}) is not None

    def test_stages_not_array_error(self):
        assert _validate_pipeline_definition({"stages": "nope"}) is not None

    def test_duplicate_stage_id_error(self):
        p = {"stages": [
            {"stage_id": "s1", "workflow_source": "a.json"},
            {"stage_id": "s1", "workflow_source": "b.json"},
        ]}
        err = _validate_pipeline_definition(p)
        assert "Duplicate" in err

    def test_missing_stage_id_error(self):
        p = {"stages": [{"workflow_source": "a.json"}]}
        err = _validate_pipeline_definition(p)
        assert "stage_id" in err

    def test_missing_workflow_source_error(self):
        p = {"stages": [{"stage_id": "s1"}]}
        err = _validate_pipeline_definition(p)
        assert "workflow_source" in err

    def test_forward_reference_error(self):
        p = {"stages": [
            {"stage_id": "s1", "workflow_source": "a.json",
             "input_mappings": [{"node_id": "1", "input_name": "x",
                                 "from_stage": "s2"}]},
            {"stage_id": "s2", "workflow_source": "b.json"},
        ]}
        err = _validate_pipeline_definition(p)
        assert "hasn't been defined yet" in err

    def test_stage_not_dict_error(self):
        p = {"stages": ["not_a_dict"]}
        err = _validate_pipeline_definition(p)
        assert "must be an object" in err


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------


class TestApplyParamOverrides:
    def test_valid_override(self):
        wf = {"3": {"class_type": "KSampler", "inputs": {"seed": 0}}}
        errors = _apply_param_overrides(wf, {"3.seed": 42})
        assert errors == []
        assert wf["3"]["inputs"]["seed"] == 42

    def test_creates_inputs_dict(self):
        wf = {"3": {"class_type": "KSampler"}}
        errors = _apply_param_overrides(wf, {"3.seed": 42})
        assert errors == []
        assert wf["3"]["inputs"]["seed"] == 42

    def test_invalid_key_format(self):
        wf = {"3": {"class_type": "KSampler", "inputs": {}}}
        errors = _apply_param_overrides(wf, {"bad_key": 42})
        assert len(errors) == 1
        assert "node_id.input_name" in errors[0]

    def test_missing_node(self):
        wf = {"3": {"class_type": "KSampler", "inputs": {}}}
        errors = _apply_param_overrides(wf, {"99.seed": 42})
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_multiple_overrides(self):
        wf = {
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 0, "steps": 20}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": ""}},
        }
        errors = _apply_param_overrides(wf, {
            "3.seed": 42, "3.steps": 30, "6.text": "a cat",
        })
        assert errors == []
        assert wf["3"]["inputs"]["seed"] == 42
        assert wf["3"]["inputs"]["steps"] == 30
        assert wf["6"]["inputs"]["text"] == "a cat"


# ---------------------------------------------------------------------------
# Input mappings
# ---------------------------------------------------------------------------


class TestApplyInputMappings:
    def test_valid_mapping(self):
        wf = {"2": {"class_type": "LoadImage",
                     "inputs": {"image": "placeholder"}}}
        stage_outputs = {
            "gen": [{"filename": "ComfyUI_00001_.png",
                     "absolute_path": "/out/ComfyUI_00001_.png"}],
        }
        errors = _apply_input_mappings(wf, [
            {"node_id": "2", "input_name": "image",
             "from_stage": "gen", "output_index": 0},
        ], stage_outputs)
        assert errors == []
        assert wf["2"]["inputs"]["image"] == "ComfyUI_00001_.png"

    def test_missing_stage_output(self):
        wf = {"2": {"class_type": "LoadImage", "inputs": {}}}
        errors = _apply_input_mappings(wf, [
            {"node_id": "2", "input_name": "image",
             "from_stage": "missing"},
        ], {})
        assert len(errors) == 1
        assert "no captured outputs" in errors[0]

    def test_output_index_out_of_range(self):
        wf = {"2": {"class_type": "LoadImage", "inputs": {}}}
        stage_outputs = {"gen": [{"filename": "test.png"}]}
        errors = _apply_input_mappings(wf, [
            {"node_id": "2", "input_name": "image",
             "from_stage": "gen", "output_index": 5},
        ], stage_outputs)
        assert len(errors) == 1
        assert "out of range" in errors[0]

    def test_missing_target_node(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        stage_outputs = {"gen": [{"filename": "test.png"}]}
        errors = _apply_input_mappings(wf, [
            {"node_id": "99", "input_name": "image",
             "from_stage": "gen"},
        ], stage_outputs)
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_empty_filename(self):
        wf = {"2": {"class_type": "LoadImage", "inputs": {}}}
        stage_outputs = {"gen": [{"filename": ""}]}
        errors = _apply_input_mappings(wf, [
            {"node_id": "2", "input_name": "image",
             "from_stage": "gen"},
        ], stage_outputs)
        assert len(errors) == 1
        assert "no filename" in errors[0]

    def test_default_output_index(self):
        wf = {"2": {"class_type": "LoadImage", "inputs": {}}}
        stage_outputs = {"gen": [{"filename": "test.png"}]}
        errors = _apply_input_mappings(wf, [
            {"node_id": "2", "input_name": "image",
             "from_stage": "gen"},
        ], stage_outputs)
        assert errors == []
        assert wf["2"]["inputs"]["image"] == "test.png"


# ---------------------------------------------------------------------------
# create_pipeline tool
# ---------------------------------------------------------------------------


class TestCreatePipeline:
    def test_simple_pipeline(self):
        result = json.loads(pipeline.handle("create_pipeline", {
            "stages": [
                {"stage_id": "gen",
                 "workflow_source": "template:txt2img_sdxl"},
            ],
            "name": "test_pipe",
        }))
        assert result["created"] is True
        assert result["name"] == "test_pipe"
        assert result["stage_count"] == 1

    def test_two_stage_pipeline(self):
        result = json.loads(pipeline.handle("create_pipeline", {
            "stages": [
                {"stage_id": "gen",
                 "workflow_source": "template:txt2img_sdxl"},
                {"stage_id": "upscale",
                 "workflow_source": "template:img2img",
                 "input_mappings": [
                     {"node_id": "2", "input_name": "image",
                      "from_stage": "gen"},
                 ]},
            ],
            "name": "txt2img_upscale",
        }))
        assert result["created"] is True
        assert result["stage_count"] == 2
        assert result["stages"][1]["receives_from"] == ["gen"]

    def test_with_param_overrides(self):
        result = json.loads(pipeline.handle("create_pipeline", {
            "stages": [
                {"stage_id": "gen",
                 "workflow_source": "template:txt2img_sdxl",
                 "param_overrides": {"3.seed": 42, "6.text": "a cat"}},
            ],
        }))
        assert result["created"] is True
        assert result["stages"][0]["override_count"] == 2

    def test_stores_in_module_state(self):
        pipeline.handle("create_pipeline", {
            "stages": [{"stage_id": "s1",
                         "workflow_source": "template:txt2img_sdxl"}],
            "name": "stored_test",
        })
        with _pipeline_lock:
            assert _pipeline_state["current_pipeline"] is not None
            assert _pipeline_state["status"] == "idle"

    def test_error_empty_stages(self):
        result = json.loads(pipeline.handle("create_pipeline", {
            "stages": [],
        }))
        assert "error" in result

    def test_error_duplicate_ids(self):
        result = json.loads(pipeline.handle("create_pipeline", {
            "stages": [
                {"stage_id": "s1", "workflow_source": "a.json"},
                {"stage_id": "s1", "workflow_source": "b.json"},
            ],
        }))
        assert "error" in result
        assert "Duplicate" in result["error"]

    def test_error_forward_reference(self):
        result = json.loads(pipeline.handle("create_pipeline", {
            "stages": [
                {"stage_id": "s1", "workflow_source": "a.json",
                 "input_mappings": [{"node_id": "1", "input_name": "x",
                                     "from_stage": "s2"}]},
                {"stage_id": "s2", "workflow_source": "b.json"},
            ],
        }))
        assert "error" in result


# ---------------------------------------------------------------------------
# run_pipeline tool
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def _make_pipeline(self, stages):
        return {
            "name": "test", "stages": stages,
            "stage_count": len(stages),
        }

    def test_run_single_stage(self):
        from unittest.mock import patch

        fake_wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        fake_exec = {
            "status": "complete", "prompt_id": "p1",
            "outputs": [{"type": "image", "filename": "out.png",
                         "subfolder": ""}],
            "total_time_s": 5.0,
        }
        fake_resolved = [{
            "filename": "out.png",
            "absolute_path": "/out/out.png",
            "exists": True, "size_bytes": 1024, "type": "image",
        }]

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=(fake_wf, None)), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   return_value=fake_exec), \
             patch("agent.tools.pipeline._resolve_stage_outputs",
                   return_value=fake_resolved), \
             patch("agent.tools.handle", return_value="{}"):

            result = json.loads(pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "gen",
                     "workflow_source": "template:txt2img_sdxl"},
                ]),
            }))

        assert result["status"] == "complete"
        assert result["stages_completed"] == 1
        assert result["stages_failed"] == 0
        assert len(result["final_outputs"]) == 1
        assert result["final_outputs"][0]["filename"] == "out.png"

    def test_run_two_stages_with_chaining(self):
        from unittest.mock import patch

        call_count = {"n": 0}

        def mock_load(source):
            call_count["n"] += 1
            return ({"1": {"class_type": "X", "inputs": {}}}, None)

        def mock_exec(wf, timeout):
            return {
                "status": "complete",
                "prompt_id": f"p{call_count['n']}",
                "outputs": [{"type": "image",
                             "filename": f"out{call_count['n']}.png",
                             "subfolder": ""}],
                "total_time_s": 3.0,
            }

        def mock_resolve(exec_result, output_key):
            fn = exec_result["outputs"][0]["filename"]
            return [{"filename": fn, "absolute_path": f"/out/{fn}",
                     "exists": True, "size_bytes": 1024,
                     "type": "image"}]

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   side_effect=mock_load), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   side_effect=mock_exec), \
             patch("agent.tools.pipeline._resolve_stage_outputs",
                   side_effect=mock_resolve), \
             patch("agent.tools.handle", return_value="{}"):

            result = json.loads(pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "gen",
                     "workflow_source": "template:txt2img_sdxl"},
                    {"stage_id": "upscale",
                     "workflow_source": "template:img2img",
                     "input_mappings": [
                         {"node_id": "1", "input_name": "image",
                          "from_stage": "gen"},
                     ]},
                ]),
            }))

        assert result["status"] == "complete"
        assert result["stages_completed"] == 2

    def test_run_with_param_overrides(self):
        from unittest.mock import patch

        applied_wf = {}

        def mock_exec(wf, timeout):
            applied_wf.update(wf)
            return {
                "status": "complete", "prompt_id": "p1",
                "outputs": [],
            }

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=(
                       {"3": {"class_type": "KSampler",
                              "inputs": {"seed": 0}}}, None)), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   side_effect=mock_exec), \
             patch("agent.tools.pipeline._resolve_stage_outputs",
                   return_value=[]), \
             patch("agent.tools.handle", return_value="{}"):

            pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "gen",
                     "workflow_source": "template:txt2img_sdxl",
                     "param_overrides": {"3.seed": 42}},
                ]),
            })

        assert applied_wf["3"]["inputs"]["seed"] == 42

    def test_stop_on_error_true(self):
        from unittest.mock import patch

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=(None, "Load failed")):
            result = json.loads(pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "s1", "workflow_source": "bad.json"},
                    {"stage_id": "s2", "workflow_source": "good.json"},
                ]),
                "stop_on_error": True,
            }))

        assert result["status"] == "error"
        assert result["stages_completed"] == 0
        assert result["stages_failed"] == 1
        assert len(result["stage_results"]) == 1

    def test_stop_on_error_false_continues(self):
        from unittest.mock import patch

        call_count = {"n": 0}

        def mock_load(source):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return (None, "Load failed")
            return ({"1": {"class_type": "X", "inputs": {}}}, None)

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   side_effect=mock_load), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   return_value={"status": "complete",
                                 "prompt_id": "p2",
                                 "outputs": []}), \
             patch("agent.tools.pipeline._resolve_stage_outputs",
                   return_value=[]), \
             patch("agent.tools.handle", return_value="{}"):

            result = json.loads(pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "s1",
                     "workflow_source": "bad.json"},
                    {"stage_id": "s2",
                     "workflow_source": "good.json"},
                ]),
                "stop_on_error": False,
            }))

        assert result["stages_failed"] == 1
        assert result["stages_completed"] == 1
        assert len(result["stage_results"]) == 2

    def test_execution_error(self):
        from unittest.mock import patch

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=({"1": {"class_type": "X",
                                        "inputs": {}}}, None)), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   return_value={"error": "CUDA out of memory"}):

            result = json.loads(pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "s1",
                     "workflow_source": "a.json"},
                ]),
            }))

        assert result["status"] == "error"
        assert result["stages_failed"] == 1

    def test_execution_timeout(self):
        from unittest.mock import patch

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=({"1": {"class_type": "X",
                                        "inputs": {}}}, None)), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   return_value={"status": "timeout",
                                 "prompt_id": "p1"}):

            result = json.loads(pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "s1",
                     "workflow_source": "a.json"},
                ]),
            }))

        assert result["status"] == "error"
        assert result["stage_results"][0]["status"] == "timeout"

    def test_no_pipeline_error(self):
        result = json.loads(pipeline._handle_run_pipeline({}))
        assert "error" in result

    def test_inline_pipeline(self):
        from unittest.mock import patch

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=({"1": {"class_type": "X",
                                        "inputs": {}}}, None)), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   return_value={"status": "complete",
                                 "prompt_id": "p1",
                                 "outputs": []}), \
             patch("agent.tools.pipeline._resolve_stage_outputs",
                   return_value=[]), \
             patch("agent.tools.handle", return_value="{}"):

            result = json.loads(pipeline._handle_run_pipeline({
                "pipeline": self._make_pipeline([
                    {"stage_id": "s1",
                     "workflow_source": "a.json"},
                ]),
            }))

        assert result["status"] == "complete"


# ---------------------------------------------------------------------------
# get_pipeline_status tool
# ---------------------------------------------------------------------------


class TestGetPipelineStatus:
    def test_no_pipeline(self):
        result = json.loads(pipeline.handle("get_pipeline_status", {}))
        assert result["status"] == "no_pipeline"

    def test_after_create(self):
        pipeline.handle("create_pipeline", {
            "stages": [{"stage_id": "s1",
                         "workflow_source": "a.json"}],
            "name": "my_pipe",
        })
        result = json.loads(pipeline.handle("get_pipeline_status", {}))
        assert result["status"] == "idle"
        assert result["pipeline_name"] == "my_pipe"
        assert result["stages_total"] == 1

    def test_after_successful_run(self):
        from unittest.mock import patch

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=({"1": {"class_type": "X",
                                        "inputs": {}}}, None)), \
             patch("agent.tools.pipeline._execute_stage_workflow",
                   return_value={
                       "status": "complete", "prompt_id": "p1",
                       "outputs": [{"type": "image",
                                    "filename": "out.png",
                                    "subfolder": ""}],
                   }), \
             patch("agent.tools.pipeline._resolve_stage_outputs",
                   return_value=[{
                       "filename": "out.png",
                       "absolute_path": "/out/out.png",
                       "exists": True, "size_bytes": 1024,
                       "type": "image",
                   }]), \
             patch("agent.tools.handle", return_value="{}"):

            pipeline.handle("create_pipeline", {
                "stages": [{"stage_id": "s1",
                             "workflow_source": "a.json"}],
            })
            pipeline._handle_run_pipeline({})

        result = json.loads(pipeline.handle("get_pipeline_status", {}))
        assert result["status"] == "complete"
        assert result["stages_completed"] == 1
        assert len(result["all_outputs"]) == 1

    def test_after_failed_run(self):
        from unittest.mock import patch

        with patch("agent.tools.pipeline._load_workflow_for_stage",
                   return_value=(None, "Load failed")):
            pipeline.handle("create_pipeline", {
                "stages": [{"stage_id": "s1",
                             "workflow_source": "bad.json"}],
            })
            pipeline._handle_run_pipeline({})

        result = json.loads(pipeline.handle("get_pipeline_status", {}))
        assert result["status"] == "error"
        assert result["stages_failed"] == 1


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_unknown_tool(self):
        result = json.loads(pipeline.handle("nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_dispatch_create(self):
        result = json.loads(pipeline.handle("create_pipeline", {
            "stages": [{"stage_id": "s1",
                         "workflow_source": "a.json"}],
        }))
        assert result["created"] is True

    def test_dispatch_status(self):
        result = json.loads(pipeline.handle("get_pipeline_status", {}))
        assert "status" in result


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_lock_exists(self):
        assert isinstance(_pipeline_lock, type(threading.Lock()))

    def test_concurrent_create(self):
        results = []

        def create(name):
            r = json.loads(pipeline.handle("create_pipeline", {
                "stages": [{"stage_id": "s1",
                             "workflow_source": "a.json"}],
                "name": name,
            }))
            results.append(r)

        threads = [
            threading.Thread(target=create, args=(f"pipe_{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(r["created"] is True for r in results)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_tool_count(self):
        assert len(pipeline.TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in pipeline.TOOLS}
        assert names == {
            "create_pipeline", "run_pipeline", "get_pipeline_status",
        }

    def test_schemas_have_required_fields(self):
        for tool in pipeline.TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_create_pipeline_requires_stages(self):
        schema = next(
            t for t in pipeline.TOOLS
            if t["name"] == "create_pipeline"
        )
        assert "stages" in schema["input_schema"]["required"]


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


class TestLoadWorkflowForStage:
    def test_template_prefix(self):
        wf, err = pipeline._load_workflow_for_stage(
            "template:txt2img_sdxl",
        )
        assert err is None
        assert wf is not None
        assert isinstance(wf, dict)

    def test_template_not_found(self):
        wf, err = pipeline._load_workflow_for_stage(
            "template:nonexistent_template",
        )
        assert wf is None
        assert "not found" in err

    def test_file_path(self, tmp_path):
        import json as json_mod
        wf_data = {
            "1": {"class_type": "KSampler",
                  "inputs": {"seed": 42}},
        }
        path = tmp_path / "test_wf.json"
        path.write_text(json_mod.dumps(wf_data), encoding="utf-8")

        wf, err = pipeline._load_workflow_for_stage(str(path))
        assert err is None
        assert wf is not None
        assert wf["1"]["inputs"]["seed"] == 42

    def test_file_not_found(self):
        wf, err = pipeline._load_workflow_for_stage(
            "/nonexistent/path/to/wf.json",
        )
        assert wf is None
        assert err is not None
