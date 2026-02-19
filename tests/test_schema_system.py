"""Tests for the schema system — loader, validator, and generator."""

from __future__ import annotations

import threading

import pytest
import yaml

from agent.schemas import (
    clear_cache,
    infer_schema_from_example,
    list_schemas,
    load_schema,
    validate_output,
    write_schema,
)
from agent.schemas.loader import (
    SCHEMAS_DIR,
    _merge_schemas,
    _validate_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    """Ensure each test starts with a clean cache."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Schema Loading
# ---------------------------------------------------------------------------


class TestLoadSchema:
    """Tests for load_schema resolution and caching."""

    def test_load_intent_default(self):
        schema = load_schema("intent")
        assert schema["schema"]["name"] == "IntentSpecification"
        assert schema["schema"]["agent"] == "intent"
        assert "fields" in schema
        assert "model_id" in schema["fields"]

    def test_load_execution_default(self):
        schema = load_schema("execution")
        assert schema["schema"]["name"] == "ExecutionResult"
        assert "status" in schema["fields"]

    def test_load_verify_default(self):
        schema = load_schema("verify")
        assert schema["schema"]["name"] == "VerificationResult"
        assert "overall_score" in schema["fields"]
        assert "decision" in schema["fields"]

    def test_schema_is_deep_copy(self):
        s1 = load_schema("intent")
        s2 = load_schema("intent")
        assert s1 == s2
        s1["fields"]["model_id"]["type"] = "MUTATED"
        s3 = load_schema("intent")
        assert s3["fields"]["model_id"]["type"] == "string"

    def test_missing_agent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_agent_xyz")

    def test_explicit_default_name(self):
        schema = load_schema("intent", "default")
        assert schema["schema"]["name"] == "IntentSpecification"


class TestSchemaCache:
    """Tests for thread-safe caching."""

    def test_cache_hit(self):
        s1 = load_schema("intent")
        s2 = load_schema("intent")
        assert s1 == s2

    def test_clear_cache(self):
        load_schema("intent")
        clear_cache()
        # Should still work — reloads from disk
        s = load_schema("intent")
        assert s["schema"]["name"] == "IntentSpecification"

    def test_thread_safety(self):
        """Multiple threads loading schemas concurrently."""
        results: list[dict] = []
        errors: list[Exception] = []

        def load_in_thread():
            try:
                s = load_schema("intent")
                results.append(s)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=load_in_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        # All results should be equivalent
        for r in results:
            assert r["schema"]["name"] == "IntentSpecification"


class TestSchemaResolution:
    """Tests for custom -> standard -> default fallback."""

    def test_custom_override(self, tmp_path):
        """Custom schema takes priority over default."""
        import agent.schemas.loader as loader_mod

        original_dir = loader_mod.SCHEMAS_DIR
        try:
            # Set up temp schema directory
            loader_mod.SCHEMAS_DIR = tmp_path
            agent_dir = tmp_path / "test_agent"
            agent_dir.mkdir()
            (agent_dir / "custom").mkdir()

            # Write default
            default = {
                "schema": {"name": "Default", "version": "1.0", "agent": "test_agent"},
                "fields": {"a": {"type": "string", "required": True}},
            }
            with open(agent_dir / "default.yaml", "w", encoding="utf-8") as f:
                yaml.dump(default, f)

            # Write custom
            custom = {
                "schema": {"name": "Custom", "version": "1.0", "agent": "test_agent"},
                "fields": {
                    "a": {"type": "string", "required": True},
                    "b": {"type": "integer", "required": False},
                },
            }
            with open(
                agent_dir / "custom" / "myschema.yaml", "w", encoding="utf-8"
            ) as f:
                yaml.dump(custom, f)

            clear_cache()
            schema = load_schema("test_agent", "myschema")
            assert schema["schema"]["name"] == "Custom"
            assert "b" in schema["fields"]
        finally:
            loader_mod.SCHEMAS_DIR = original_dir

    def test_fallback_to_default(self, tmp_path):
        """Unknown schema_name falls back to default.yaml."""
        import agent.schemas.loader as loader_mod

        original_dir = loader_mod.SCHEMAS_DIR
        try:
            loader_mod.SCHEMAS_DIR = tmp_path
            agent_dir = tmp_path / "test_agent"
            agent_dir.mkdir()

            default = {
                "schema": {"name": "Fallback", "version": "1.0", "agent": "test_agent"},
                "fields": {"x": {"type": "string", "required": True}},
            }
            with open(agent_dir / "default.yaml", "w", encoding="utf-8") as f:
                yaml.dump(default, f)

            clear_cache()
            schema = load_schema("test_agent", "nonexistent")
            assert schema["schema"]["name"] == "Fallback"
        finally:
            loader_mod.SCHEMAS_DIR = original_dir


class TestSchemaInheritance:
    """Tests for extends-based schema merging."""

    def test_merge_adds_fields(self):
        base = {
            "schema": {"name": "Base", "version": "1.0"},
            "fields": {"a": {"type": "string", "required": True}},
        }
        ext = {
            "fields": {"b": {"type": "integer", "required": False}},
        }
        merged = _merge_schemas(base, ext)
        assert "a" in merged["fields"]
        assert "b" in merged["fields"]

    def test_merge_extension_wins_conflict(self):
        base = {
            "schema": {"name": "Base", "version": "1.0"},
            "fields": {"a": {"type": "string", "required": True}},
        }
        ext = {
            "schema": {"name": "Extended"},
            "fields": {"a": {"type": "integer", "required": False}},
        }
        merged = _merge_schemas(base, ext)
        assert merged["fields"]["a"]["type"] == "integer"
        assert merged["schema"]["name"] == "Extended"
        # Version preserved from base
        assert merged["schema"]["version"] == "1.0"

    def test_extends_field_in_yaml(self, tmp_path):
        """Schema with extends key loads and merges the base."""
        import agent.schemas.loader as loader_mod

        original_dir = loader_mod.SCHEMAS_DIR
        try:
            loader_mod.SCHEMAS_DIR = tmp_path
            agent_dir = tmp_path / "test_agent"
            agent_dir.mkdir()

            base = {
                "schema": {"name": "Base", "version": "1.0", "agent": "test_agent"},
                "fields": {
                    "a": {"type": "string", "required": True},
                    "b": {"type": "float", "required": True},
                },
            }
            with open(agent_dir / "default.yaml", "w", encoding="utf-8") as f:
                yaml.dump(base, f)

            ext = {
                "extends": "default",
                "schema": {"name": "Extended"},
                "fields": {
                    "c": {"type": "boolean", "required": False},
                },
            }
            with open(agent_dir / "extended.yaml", "w", encoding="utf-8") as f:
                yaml.dump(ext, f)

            clear_cache()
            schema = load_schema("test_agent", "extended")
            assert "a" in schema["fields"]
            assert "b" in schema["fields"]
            assert "c" in schema["fields"]
            assert schema["schema"]["name"] == "Extended"
        finally:
            loader_mod.SCHEMAS_DIR = original_dir

    def test_merge_preserves_base_immutability(self):
        base = {
            "schema": {"name": "Base"},
            "fields": {"a": {"type": "string"}},
        }
        ext = {"fields": {"a": {"type": "integer"}}}
        _merge_schemas(base, ext)
        # Base should be unchanged
        assert base["fields"]["a"]["type"] == "string"


# ---------------------------------------------------------------------------
# Output Validation
# ---------------------------------------------------------------------------


class TestValidateOutput:
    """Tests for validate_output against schemas."""

    def test_valid_intent_output(self):
        output = {
            "model_id": "sdxl-base",
            "parameter_mutations": [
                {
                    "target": "sampler.steps",
                    "action": "set",
                    "value": 25,
                    "reason": "Better quality",
                }
            ],
            "prompt_mutations": [
                {
                    "target": "positive_prompt",
                    "action": "append",
                    "value": "cinematic lighting",
                    "reason": "User asked for dreamier",
                }
            ],
            "confidence": 0.85,
        }
        errors = validate_output(output, "intent")
        assert errors == []

    def test_missing_required_field(self):
        output = {
            "parameter_mutations": [],
            "prompt_mutations": [],
            "confidence": 0.5,
        }
        errors = validate_output(output, "intent")
        assert any("model_id" in e for e in errors)

    def test_type_mismatch_string(self):
        output = {
            "model_id": 123,  # should be string
            "parameter_mutations": [],
            "prompt_mutations": [],
            "confidence": 0.5,
        }
        errors = validate_output(output, "intent")
        assert any("model_id" in e and "string" in e for e in errors)

    def test_type_mismatch_float(self):
        output = {
            "model_id": "test",
            "parameter_mutations": [],
            "prompt_mutations": [],
            "confidence": "high",  # should be float
        }
        errors = validate_output(output, "intent")
        assert any("confidence" in e for e in errors)

    def test_range_violation(self):
        output = {
            "model_id": "test",
            "parameter_mutations": [],
            "prompt_mutations": [],
            "confidence": 1.5,  # out of [0.0, 1.0]
        }
        errors = validate_output(output, "intent")
        assert any("range" in e for e in errors)

    def test_enum_violation(self):
        output = {
            "status": "unknown_status",
            "patches_applied": [],
        }
        errors = validate_output(output, "execution")
        assert any("status" in e and "not in" in e for e in errors)

    def test_valid_execution_output(self):
        output = {
            "status": "success",
            "patches_applied": [{"op": "replace", "path": "/3/inputs/steps"}],
            "output_images": ["/tmp/out.png"],
            "execution_time": 12.5,
        }
        errors = validate_output(output, "execution")
        assert errors == []

    def test_valid_verify_output(self):
        output = {
            "overall_score": 0.82,
            "intent_alignment": 0.9,
            "technical_quality": 0.75,
            "decision": "accept",
        }
        errors = validate_output(output, "verify")
        assert errors == []

    def test_optional_field_absent_ok(self):
        output = {
            "model_id": "test",
            "parameter_mutations": [],
            "prompt_mutations": [],
            "confidence": 0.5,
            # warnings and conflicts_resolved are optional
        }
        errors = validate_output(output, "intent")
        assert errors == []

    def test_list_string_type_validation(self):
        output = {
            "model_id": "test",
            "parameter_mutations": [],
            "prompt_mutations": [],
            "confidence": 0.5,
            "warnings": [123, 456],  # should be list[string]
        }
        errors = validate_output(output, "intent")
        assert any("warnings" in e for e in errors)

    def test_boolean_not_treated_as_int(self):
        errors = _validate_type(
            "test_field", True, {"type": "integer"}
        )
        assert len(errors) == 1
        assert "bool" in errors[0]

    def test_int_accepted_for_float(self):
        errors = _validate_type(
            "score", 1, {"type": "float", "range": [0.0, 1.0]}
        )
        assert errors == []

    def test_range_lower_bound(self):
        errors = _validate_type(
            "score", -0.1, {"type": "float", "range": [0.0, 1.0]}
        )
        assert any("range" in e for e in errors)

    def test_object_type_validation(self):
        errors = _validate_type("data", "not_a_dict", {"type": "object"})
        assert any("object" in e for e in errors)

    def test_any_type_accepts_everything(self):
        for val in ["str", 42, 3.14, True, None, [], {}]:
            errors = _validate_type("x", val, {"type": "any"})
            assert errors == []


# ---------------------------------------------------------------------------
# Schema Generator
# ---------------------------------------------------------------------------


class TestSchemaGenerator:
    """Tests for infer_schema_from_example and write_schema."""

    def test_infer_string(self):
        schema = infer_schema_from_example(
            {"name": "hello"}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["name"]["type"] == "string"

    def test_infer_integer(self):
        schema = infer_schema_from_example(
            {"count": 42}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["count"]["type"] == "integer"

    def test_infer_float(self):
        schema = infer_schema_from_example(
            {"ratio": 3.14}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["ratio"]["type"] == "float"

    def test_infer_float_range_auto(self):
        schema = infer_schema_from_example(
            {"score": 0.85}, "test", "test_schema", extends=None
        )
        field = schema["fields"]["score"]
        assert field["type"] == "float"
        assert field.get("range") == [0.0, 1.0]

    def test_infer_float_no_range_outside_01(self):
        schema = infer_schema_from_example(
            {"temp": 2.5}, "test", "test_schema", extends=None
        )
        assert "range" not in schema["fields"]["temp"]

    def test_infer_boolean(self):
        schema = infer_schema_from_example(
            {"active": True}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["active"]["type"] == "boolean"

    def test_infer_list_string(self):
        schema = infer_schema_from_example(
            {"tags": ["a", "b", "c"]}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["tags"]["type"] == "list[string]"

    def test_infer_list_integer(self):
        schema = infer_schema_from_example(
            {"ids": [1, 2, 3]}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["ids"]["type"] == "list[integer]"

    def test_infer_list_dict(self):
        schema = infer_schema_from_example(
            {"items": [{"name": "a", "value": 1}]},
            "test",
            "test_schema",
            extends=None,
        )
        field = schema["fields"]["items"]
        assert field["type"] == "list"
        assert "item_schema" in field
        assert "name" in field["item_schema"]
        assert "value" in field["item_schema"]

    def test_infer_dict(self):
        schema = infer_schema_from_example(
            {"config": {"key": "val"}}, "test", "test_schema", extends=None
        )
        field = schema["fields"]["config"]
        assert field["type"] == "object"
        assert "nested_fields" in field

    def test_infer_none_is_any(self):
        schema = infer_schema_from_example(
            {"unknown": None}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["unknown"]["type"] == "any"

    def test_infer_empty_dict(self):
        schema = infer_schema_from_example(
            {}, "test", "test_schema", extends=None
        )
        assert schema["fields"] == {}

    def test_infer_empty_list(self):
        schema = infer_schema_from_example(
            {"items": []}, "test", "test_schema", extends=None
        )
        assert schema["fields"]["items"]["type"] == "list"

    def test_extends_included(self):
        schema = infer_schema_from_example(
            {"x": 1}, "test", "test_schema", extends="default"
        )
        assert schema["extends"] == "default"

    def test_extends_none_omitted(self):
        schema = infer_schema_from_example(
            {"x": 1}, "test", "test_schema", extends=None
        )
        assert "extends" not in schema

    def test_schema_metadata(self):
        schema = infer_schema_from_example(
            {"x": 1}, "myagent", "myschema", extends=None
        )
        assert schema["schema"]["agent"] == "myagent"
        assert schema["schema"]["name"] == "myschema"
        assert schema["schema"]["version"] == "1.0"

    def test_write_and_read_roundtrip(self, tmp_path):
        schema = infer_schema_from_example(
            {"name": "test", "score": 0.9, "tags": ["a"]},
            "test",
            "roundtrip",
            extends=None,
        )
        path = tmp_path / "roundtrip.yaml"
        write_schema(schema, path)

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert loaded["schema"]["name"] == "roundtrip"
        assert "name" in loaded["fields"]
        assert "score" in loaded["fields"]

    def test_roundtrip_validation(self, tmp_path):
        """Example -> schema -> validate example = no errors."""
        import agent.schemas.loader as loader_mod

        original_dir = loader_mod.SCHEMAS_DIR
        try:
            loader_mod.SCHEMAS_DIR = tmp_path
            agent_dir = tmp_path / "roundtrip_agent"
            agent_dir.mkdir()

            example = {
                "status": "done",
                "score": 0.75,
                "items": ["a", "b"],
                "count": 5,
                "active": True,
            }
            schema = infer_schema_from_example(
                example, "roundtrip_agent", "default", extends=None
            )
            write_schema(schema, agent_dir / "default.yaml")

            clear_cache()
            errors = validate_output(example, "roundtrip_agent")
            assert errors == []
        finally:
            loader_mod.SCHEMAS_DIR = original_dir

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "schema.yaml"
        schema = {"schema": {"name": "test"}, "fields": {}}
        write_schema(schema, path)
        assert path.is_file()


# ---------------------------------------------------------------------------
# list_schemas
# ---------------------------------------------------------------------------


class TestListSchemas:
    """Tests for list_schemas."""

    def test_list_intent_schemas(self):
        names = list_schemas("intent")
        assert "default" in names

    def test_list_execution_schemas(self):
        names = list_schemas("execution")
        assert "default" in names

    def test_list_verify_schemas(self):
        names = list_schemas("verify")
        assert "default" in names

    def test_list_includes_custom(self, tmp_path):
        import agent.schemas.loader as loader_mod

        original_dir = loader_mod.SCHEMAS_DIR
        try:
            loader_mod.SCHEMAS_DIR = tmp_path
            agent_dir = tmp_path / "test_agent"
            agent_dir.mkdir()
            (agent_dir / "custom").mkdir()

            with open(
                agent_dir / "default.yaml", "w", encoding="utf-8"
            ) as f:
                yaml.dump({"schema": {"name": "D"}, "fields": {}}, f)

            with open(
                agent_dir / "custom" / "special.yaml", "w", encoding="utf-8"
            ) as f:
                yaml.dump({"schema": {"name": "S"}, "fields": {}}, f)

            names = list_schemas("test_agent")
            assert "default" in names
            assert "special" in names
        finally:
            loader_mod.SCHEMAS_DIR = original_dir

    def test_list_sorted(self):
        names = list_schemas("intent")
        assert names == sorted(names)

    def test_list_nonexistent_agent(self):
        names = list_schemas("no_such_agent_xyz")
        assert names == []


# ---------------------------------------------------------------------------
# YAML file integrity
# ---------------------------------------------------------------------------


class TestYamlFiles:
    """Tests that all default YAML files parse correctly."""

    @pytest.mark.parametrize("agent", ["intent", "execution", "verify"])
    def test_default_schema_parses(self, agent):
        schema = load_schema(agent)
        assert "schema" in schema
        assert "fields" in schema
        assert isinstance(schema["fields"], dict)

    @pytest.mark.parametrize("agent", ["intent", "execution", "verify"])
    def test_default_schema_has_metadata(self, agent):
        schema = load_schema(agent)
        meta = schema["schema"]
        assert "name" in meta
        assert "version" in meta
        assert "agent" in meta
        assert meta["agent"] == agent

    def test_field_types_yaml_valid(self):
        path = SCHEMAS_DIR / "_meta" / "field_types.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "types" in data
        for type_name, type_def in data["types"].items():
            assert "python" in type_def
            assert "description" in type_def

    @pytest.mark.parametrize("agent", ["intent", "execution", "verify"])
    def test_required_fields_are_present(self, agent):
        """Each schema has at least one required field."""
        schema = load_schema(agent)
        required = [
            name
            for name, fdef in schema["fields"].items()
            if isinstance(fdef, dict) and fdef.get("required")
        ]
        assert len(required) > 0
