"""Tests for workflow_parse tool — both synthetic and real workflows."""

import json
import pytest
from unittest.mock import patch
from agent.tools import workflow_parse


# ---------------------------------------------------------------------------
# Fixtures: synthetic workflows
# ---------------------------------------------------------------------------

@pytest.fixture
def api_workflow(tmp_path):
    """Minimal API-format workflow: loader → encoder → sampler → save."""
    data = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd15.safetensors",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "a beautiful sunset",
                "clip": ["1", 1],  # connection: node 1, output 1 (CLIP)
            },
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1,
            },
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],       # MODEL from loader
                "positive": ["2", 0],    # CONDITIONING from encoder
                "negative": ["5", 0],    # CONDITIONING from neg encoder
                "latent_image": ["3", 0],  # LATENT from empty
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["1", 1],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["4", 0],
                "vae": ["1", 2],
            },
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "output",
            },
        },
    }
    path = tmp_path / "api_workflow.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def ui_workflow_with_prompt(tmp_path):
    """UI-format workflow with embedded extra.prompt."""
    data = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple"},
            {"id": 2, "type": "CLIPTextEncode"},
        ],
        "links": [[1, 1, 2, 0, 1]],
        "extra": {
            "prompt": {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "sd15.safetensors"},
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "hello world", "clip": ["1", 1]},
                },
            },
        },
    }
    path = tmp_path / "ui_with_prompt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def ui_only_workflow(tmp_path):
    """UI-format workflow WITHOUT extra.prompt."""
    data = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["sd15.safetensors"]},
            {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["a cat"]},
        ],
        "links": [],
        "extra": {"ds": {"scale": 1.0}},
    }
    path = tmp_path / "ui_only.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_workflow tests
# ---------------------------------------------------------------------------

class TestLoadWorkflow:
    def test_api_format(self, api_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(api_workflow)})
        )
        assert result["format"] == "api"
        assert result["node_count"] == 7
        assert result["connection_count"] > 0
        assert result["editable_field_count"] > 0

    def test_api_nodes_detected(self, api_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(api_workflow)})
        )
        nodes = result["nodes"]
        assert nodes["1"]["class_type"] == "CheckpointLoaderSimple"
        assert nodes["4"]["class_type"] == "KSampler"
        assert nodes["7"]["class_type"] == "SaveImage"

    def test_connections_traced(self, api_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(api_workflow)})
        )
        connections = result["connections"]
        # KSampler receives model from loader
        model_conn = next(
            (c for c in connections
             if c["to_class"] == "KSampler" and c["to_input"] == "model"),
            None,
        )
        assert model_conn is not None
        assert model_conn["from_class"] == "CheckpointLoaderSimple"
        assert model_conn["from_output"] == 0

    def test_editable_fields_found(self, api_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(api_workflow)})
        )
        fields = result["editable_fields"]
        # Should find text prompts, seed, steps, cfg, etc.
        field_names = [(f["class_type"], f["field"]) for f in fields]
        assert ("CLIPTextEncode", "text") in field_names
        assert ("KSampler", "seed") in field_names
        assert ("KSampler", "steps") in field_names
        assert ("KSampler", "cfg") in field_names
        assert ("EmptyLatentImage", "width") in field_names

    def test_summary_generated(self, api_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(api_workflow)})
        )
        summary = result["summary"]
        assert "7 nodes" in summary
        assert "connections" in summary

    def test_ui_with_prompt(self, ui_workflow_with_prompt):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(ui_workflow_with_prompt)})
        )
        assert result["format"] == "ui_with_api"
        assert result["node_count"] == 2
        # Should find the text field
        fields = result["editable_fields"]
        texts = [f for f in fields if f["field"] == "text"]
        assert len(texts) == 1
        assert texts[0]["value"] == "hello world"

    def test_ui_only(self, ui_only_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(ui_only_workflow)})
        )
        assert result["format"] == "ui_only"
        assert result["node_count"] == 2
        assert "UI-only" in result["summary"]

    def test_file_not_found(self):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": "/nonexistent/file.json"})
        )
        assert "error" in result

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json {{{", encoding="utf-8")
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(bad)})
        )
        assert "error" in result


class TestConnectionTracing:
    def test_full_pipeline_traced(self, api_workflow):
        """Verify the full loader→encoder→sampler→decode→save chain."""
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(api_workflow)})
        )
        conns = result["connections"]

        # Build adjacency: target_class.input → source_class
        edges = {
            (c["to_class"], c["to_input"]): c["from_class"]
            for c in conns
        }

        # Check pipeline chain
        assert edges[("CLIPTextEncode", "clip")] == "CheckpointLoaderSimple"
        assert edges[("KSampler", "model")] == "CheckpointLoaderSimple"
        assert edges[("KSampler", "positive")] == "CLIPTextEncode"
        assert edges[("KSampler", "latent_image")] == "EmptyLatentImage"
        assert edges[("VAEDecode", "samples")] == "KSampler"
        assert edges[("VAEDecode", "vae")] == "CheckpointLoaderSimple"
        assert edges[("SaveImage", "images")] == "VAEDecode"

    def test_connection_count(self, api_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(api_workflow)})
        )
        # 2 CLIPTextEncode.clip + 1 model + 1 positive + 1 negative
        # + 1 latent + 1 samples + 1 vae + 1 images = 9
        assert result["connection_count"] == 9


class TestGetEditableFields:
    def test_all_fields(self, api_workflow):
        result = json.loads(
            workflow_parse.handle("get_editable_fields", {"path": str(api_workflow)})
        )
        assert result["total_fields"] > 0
        assert "KSampler" in result["fields_by_class"]

    def test_class_filter(self, api_workflow):
        result = json.loads(
            workflow_parse.handle(
                "get_editable_fields",
                {"path": str(api_workflow), "class_filter": "sampler"},
            )
        )
        assert "KSampler" in result["fields_by_class"]
        # Should NOT include other classes
        assert "CLIPTextEncode" not in result["fields_by_class"]
        assert "EmptyLatentImage" not in result["fields_by_class"]

    def test_ui_only_returns_error(self, ui_only_workflow):
        result = json.loads(
            workflow_parse.handle("get_editable_fields", {"path": str(ui_only_workflow)})
        )
        assert "error" in result


class TestValidateWorkflow:
    def test_validation_requires_comfyui(self, api_workflow):
        """If ComfyUI isn't running, should return a clear error."""
        import httpx

        with patch(
            "agent.tools.workflow_parse.httpx.Client"
        ) as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.get.side_effect = httpx.ConnectError("refused")

            result = json.loads(
                workflow_parse.handle("validate_workflow", {"path": str(api_workflow)})
            )
            assert "error" in result
            assert "running" in result["error"].lower()

    def test_ui_only_rejected(self, ui_only_workflow):
        result = json.loads(
            workflow_parse.handle("validate_workflow", {"path": str(ui_only_workflow)})
        )
        assert "error" in result
        assert "UI-only" in result["error"]

    def test_validation_with_mock_comfyui(self, api_workflow):
        """Mock a successful validation against object_info."""

        mock_object_info = {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [["sd15.safetensors"]]}},
                "output": ["MODEL", "CLIP", "VAE"],
            },
            "CLIPTextEncode": {
                "input": {"required": {"text": ["STRING"], "clip": ["CLIP"]}},
                "output": ["CONDITIONING"],
            },
            "EmptyLatentImage": {
                "input": {"required": {
                    "width": ["INT"], "height": ["INT"], "batch_size": ["INT"],
                }},
                "output": ["LATENT"],
            },
            "KSampler": {
                "input": {"required": {
                    "model": ["MODEL"],
                    "positive": ["CONDITIONING"],
                    "negative": ["CONDITIONING"],
                    "latent_image": ["LATENT"],
                    "seed": ["INT"],
                    "steps": ["INT"],
                    "cfg": ["FLOAT"],
                    "sampler_name": [["euler", "dpmpp_2m"]],
                    "scheduler": [["normal", "karras"]],
                    "denoise": ["FLOAT"],
                }},
                "output": ["LATENT"],
            },
            "VAEDecode": {
                "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
                "output": ["IMAGE"],
            },
            "SaveImage": {
                "input": {"required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING"],
                }},
                "output": [],
            },
        }

        mock_resp = type("MockResp", (), {
            "json": lambda self: mock_object_info,
            "raise_for_status": lambda self: None,
        })()

        with patch("agent.tools.workflow_parse.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            result = json.loads(
                workflow_parse.handle("validate_workflow", {"path": str(api_workflow)})
            )
            assert result["valid"] is True
            assert len(result["errors"]) == 0

    def test_validation_detects_missing_node(self, tmp_path):
        """Workflow with a node type that doesn't exist."""
        data = {
            "1": {
                "class_type": "NonExistentNode",
                "inputs": {"text": "hello"},
            },
        }
        wf = tmp_path / "bad_node.json"
        wf.write_text(json.dumps(data), encoding="utf-8")

        mock_object_info = {"KSampler": {"input": {"required": {}}, "output": []}}
        mock_resp = type("MockResp", (), {
            "json": lambda self: mock_object_info,
            "raise_for_status": lambda self: None,
        })()

        with patch("agent.tools.workflow_parse.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            result = json.loads(
                workflow_parse.handle("validate_workflow", {"path": str(wf)})
            )
            assert result["valid"] is False
            assert any("NonExistentNode" in e for e in result["errors"])

    def test_validation_detects_type_mismatch(self, tmp_path):
        """Connection where output type doesn't match input type."""
        data = {
            "1": {
                "class_type": "SaveImage",
                "inputs": {"images": "literal_not_connection", "filename_prefix": "out"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "hello",
                    "clip": ["1", 0],  # SaveImage output 0 → CLIP input (type mismatch)
                },
            },
        }
        wf = tmp_path / "mismatch.json"
        wf.write_text(json.dumps(data), encoding="utf-8")

        mock_object_info = {
            "SaveImage": {
                "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}},
                "output": ["IMAGE"],  # outputs IMAGE
            },
            "CLIPTextEncode": {
                "input": {"required": {"text": ["STRING"], "clip": ["CLIP"]}},
                "output": ["CONDITIONING"],
            },
        }
        mock_resp = type("MockResp", (), {
            "json": lambda self: mock_object_info,
            "raise_for_status": lambda self: None,
        })()

        with patch("agent.tools.workflow_parse.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            result = json.loads(
                workflow_parse.handle("validate_workflow", {"path": str(wf)})
            )
            assert result["valid"] is False
            assert any("Type mismatch" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# 3D workflow tests (Graft A: 3D data type support)
# ---------------------------------------------------------------------------

@pytest.fixture
def hunyuan3d_workflow(tmp_path):
    """Minimal Hunyuan3D workflow: loader -> conditioner -> sampler -> mesh decoder -> save."""
    data = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "input.png",
            },
        },
        "2": {
            "class_type": "Hunyuan3DLoader",
            "inputs": {
                "model_name": "hunyuan3d-v2.safetensors",
            },
        },
        "3": {
            "class_type": "Hunyuan3DConditioner",
            "inputs": {
                "image": ["1", 0],
                "model": ["2", 0],
            },
        },
        "4": {
            "class_type": "Hunyuan3DSampler",
            "inputs": {
                "conditioning": ["3", 0],
                "model": ["2", 0],
                "steps": 50,
                "guidance_scale": 7.5,
            },
        },
        "5": {
            "class_type": "Hunyuan3DMeshDecoder",
            "inputs": {
                "triplane": ["4", 0],
                "resolution": 512,
            },
        },
        "6": {
            "class_type": "SaveGLB",
            "inputs": {
                "mesh": ["5", 0],
                "filename_prefix": "output_3d",
            },
        },
    }
    path = tmp_path / "hunyuan3d_workflow.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def splat_to_mesh_workflow(tmp_path):
    """Gaussian splat to mesh conversion workflow."""
    data = {
        "1": {
            "class_type": "Load3DGaussian",
            "inputs": {
                "file_path": "scene.ply",
            },
        },
        "2": {
            "class_type": "GaussianToMesh",
            "inputs": {
                "point_cloud": ["1", 0],
                "marching_cubes_resolution": 256,
            },
        },
        "3": {
            "class_type": "SaveGLB",
            "inputs": {
                "mesh": ["2", 0],
                "filename_prefix": "converted_mesh",
            },
        },
    }
    path = tmp_path / "splat_to_mesh.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestHunyuan3DWorkflow:
    def test_parse_3d_workflow(self, hunyuan3d_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(hunyuan3d_workflow)})
        )
        assert result["format"] == "api"
        assert result["node_count"] == 6
        assert result["connection_count"] == 6  # image, 2x model, conditioning, triplane, mesh

    def test_3d_nodes_categorized(self, hunyuan3d_workflow):
        """Summary should categorize 3D nodes under '3D Processing'."""
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(hunyuan3d_workflow)})
        )
        summary = result["summary"]
        assert "3D Processing" in summary
        assert "Hunyuan3DMeshDecoder" in summary
        assert "Hunyuan3DSampler" in summary

    def test_3d_connections_traced(self, hunyuan3d_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(hunyuan3d_workflow)})
        )
        conns = result["connections"]
        edges = {
            (c["to_class"], c["to_input"]): c["from_class"]
            for c in conns
        }
        assert edges[("Hunyuan3DMeshDecoder", "triplane")] == "Hunyuan3DSampler"
        assert edges[("SaveGLB", "mesh")] == "Hunyuan3DMeshDecoder"

    def test_3d_editable_fields(self, hunyuan3d_workflow):
        result = json.loads(
            workflow_parse.handle("get_editable_fields", {"path": str(hunyuan3d_workflow)})
        )
        fields = result["editable_fields"] if "editable_fields" in result else []
        by_class = result.get("fields_by_class", {})
        assert "Hunyuan3DSampler" in by_class
        sampler_fields = by_class["Hunyuan3DSampler"]
        field_names = [f["field"] for f in sampler_fields]
        assert "steps" in field_names
        assert "guidance_scale" in field_names


class TestSplatToMeshWorkflow:
    def test_parse_splat_conversion(self, splat_to_mesh_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(splat_to_mesh_workflow)})
        )
        assert result["format"] == "api"
        assert result["node_count"] == 3

    def test_splat_nodes_categorized(self, splat_to_mesh_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(splat_to_mesh_workflow)})
        )
        summary = result["summary"]
        # Both gaussian and mesh nodes should be in 3D Processing
        assert "3D Processing" in summary

    def test_splat_conversion_chain(self, splat_to_mesh_workflow):
        result = json.loads(
            workflow_parse.handle("load_workflow", {"path": str(splat_to_mesh_workflow)})
        )
        conns = result["connections"]
        edges = {
            (c["to_class"], c["to_input"]): c["from_class"]
            for c in conns
        }
        assert edges[("GaussianToMesh", "point_cloud")] == "Load3DGaussian"
        assert edges[("SaveGLB", "mesh")] == "GaussianToMesh"


class TestValidation3DWorkflow:
    def test_validation_with_3d_types(self, hunyuan3d_workflow):
        """Mock validation with 3D node types and MESH/TRIPLANE types."""
        mock_object_info = {
            "LoadImage": {
                "input": {"required": {"image": ["STRING"]}},
                "output": ["IMAGE", "MASK"],
            },
            "Hunyuan3DLoader": {
                "input": {"required": {"model_name": [["hunyuan3d-v2.safetensors"]]}},
                "output": ["MODEL"],
            },
            "Hunyuan3DConditioner": {
                "input": {"required": {"image": ["IMAGE"], "model": ["MODEL"]}},
                "output": ["CONDITIONING"],
            },
            "Hunyuan3DSampler": {
                "input": {"required": {
                    "conditioning": ["CONDITIONING"],
                    "model": ["MODEL"],
                    "steps": ["INT"],
                    "guidance_scale": ["FLOAT"],
                }},
                "output": ["TRIPLANE"],
            },
            "Hunyuan3DMeshDecoder": {
                "input": {"required": {"triplane": ["TRIPLANE"], "resolution": ["INT"]}},
                "output": ["MESH"],
            },
            "SaveGLB": {
                "input": {"required": {"mesh": ["MESH"], "filename_prefix": ["STRING"]}},
                "output": [],
            },
        }

        mock_resp = type("MockResp", (), {
            "json": lambda self: mock_object_info,
            "raise_for_status": lambda self: None,
        })()

        with patch("agent.tools.workflow_parse.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            result = json.loads(
                workflow_parse.handle("validate_workflow", {"path": str(hunyuan3d_workflow)})
            )
            assert result["valid"] is True
            assert len(result["errors"]) == 0

    def test_validation_detects_3d_type_mismatch(self, tmp_path):
        """IMAGE -> MESH input should be a type mismatch."""
        data = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "input.png"},
            },
            "2": {
                "class_type": "SaveGLB",
                "inputs": {
                    "mesh": ["1", 0],  # IMAGE -> MESH (mismatch)
                    "filename_prefix": "out",
                },
            },
        }
        wf = tmp_path / "mismatch_3d.json"
        wf.write_text(json.dumps(data), encoding="utf-8")

        mock_object_info = {
            "LoadImage": {
                "input": {"required": {"image": ["STRING"]}},
                "output": ["IMAGE", "MASK"],
            },
            "SaveGLB": {
                "input": {"required": {"mesh": ["MESH"], "filename_prefix": ["STRING"]}},
                "output": [],
            },
        }

        mock_resp = type("MockResp", (), {
            "json": lambda self: mock_object_info,
            "raise_for_status": lambda self: None,
        })()

        with patch("agent.tools.workflow_parse.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            result = json.loads(
                workflow_parse.handle("validate_workflow", {"path": str(wf)})
            )
            assert result["valid"] is False
            assert any("Type mismatch" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Tool registry integration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_tools_registered(self):
        from agent.tools import ALL_TOOLS
        names = {t["name"] for t in ALL_TOOLS}
        assert "load_workflow" in names
        assert "validate_workflow" in names
        assert "get_editable_fields" in names

    def test_dispatch_works(self):
        from agent.tools import handle
        result = json.loads(handle("load_workflow", {"path": "/nonexistent"}))
        assert "error" in result
