"""Tests for brain/_sdk.py — BrainConfig, BrainAgent, and SDK integration."""

import json
from pathlib import Path

import pytest

from agent.brain._sdk import (
    BrainAgent,
    BrainConfig,
    _NullLimiter,
    _default_to_json,
    _default_validate_path,
    get_integrated_config,
    reset_integrated_config,
)


class TestBrainConfig:
    def test_default_values(self):
        cfg = BrainConfig()
        assert cfg.comfyui_url == "http://127.0.0.1:8188"
        assert cfg.sessions_dir == Path("./sessions")
        assert cfg.tool_dispatcher is None
        assert cfg.get_workflow_state is None
        assert cfg.patch_handle is None

    def test_custom_values(self):
        cfg = BrainConfig(
            comfyui_url="http://localhost:9999",
            sessions_dir=Path("/tmp/test"),
            agent_model="test-model",
        )
        assert cfg.comfyui_url == "http://localhost:9999"
        assert cfg.sessions_dir == Path("/tmp/test")
        assert cfg.agent_model == "test-model"

    def test_to_json_default(self):
        cfg = BrainConfig()
        result = cfg.to_json({"b": 2, "a": 1})
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_custom_to_json(self):
        def my_json(obj, **kw):
            return json.dumps(obj)

        cfg = BrainConfig(to_json=my_json)
        # Custom serializer doesn't sort keys
        result = cfg.to_json({"b": 2, "a": 1})
        assert result == '{"b": 2, "a": 1}'


class TestDefaultHelpers:
    def test_default_to_json_sort_keys(self):
        result = _default_to_json({"c": 3, "a": 1, "b": 2})
        assert result == '{"a": 1, "b": 2, "c": 3}'

    def test_default_validate_path_valid(self):
        assert _default_validate_path(".") is None

    def test_default_validate_path_must_exist_missing(self):
        result = _default_validate_path("/nonexistent/path/xyz123", must_exist=True)
        assert result is not None
        assert "not found" in result.lower() or "invalid" in result.lower()

    def test_null_limiter(self):
        limiter = _NullLimiter()
        assert limiter.acquire() is True
        assert limiter.acquire(tokens=100, timeout=0.0) is True


class TestBrainAgent:
    def test_base_class_raises(self):
        agent = BrainAgent(config=BrainConfig())
        with pytest.raises(NotImplementedError):
            agent.handle("test", {})

    def test_subclass_receives_config(self):
        class MyAgent(BrainAgent):
            TOOLS = [{"name": "test_tool", "description": "test", "input_schema": {}}]

            def handle(self, name, tool_input):
                return self.to_json({"ok": True})

        cfg = BrainConfig(comfyui_url="http://custom:1234")
        agent = MyAgent(config=cfg)
        assert agent.cfg.comfyui_url == "http://custom:1234"
        result = json.loads(agent.handle("test_tool", {}))
        assert result["ok"] is True

    def test_to_json_on_instance(self):
        agent = BrainAgent(config=BrainConfig())
        result = agent.to_json({"key": "value"})
        assert json.loads(result) == {"key": "value"}


class TestIntegratedConfig:
    def setup_method(self):
        reset_integrated_config()

    def teardown_method(self):
        reset_integrated_config()

    def test_integrated_config_caching(self):
        cfg1 = get_integrated_config()
        cfg2 = get_integrated_config()
        assert cfg1 is cfg2

    def test_integrated_config_has_real_values(self):
        cfg = get_integrated_config()
        assert cfg.comfyui_url  # non-empty
        assert cfg.sessions_dir.name  # has a name
        assert cfg.tool_dispatcher is not None
        assert cfg.get_workflow_state is not None
        assert cfg.patch_handle is not None

    def test_reset_allows_new_config(self):
        cfg1 = get_integrated_config()
        reset_integrated_config()
        cfg2 = get_integrated_config()
        assert cfg1 is not cfg2


class TestStandaloneInstantiation:
    """Verify agent classes can be created with custom config (no parent imports at call time)."""

    def test_demo_agent_standalone(self):
        from agent.brain.demo import DemoAgent

        cfg = BrainConfig()
        agent = DemoAgent(config=cfg)
        result = json.loads(agent.handle("start_demo", {"scenario": "list"}))
        assert "available_scenarios" in result
        assert result["count"] >= 4

    def test_planner_agent_standalone(self, tmp_path):
        from agent.brain.planner import PlannerAgent

        cfg = BrainConfig(sessions_dir=tmp_path)
        agent = PlannerAgent(config=cfg)
        assert agent.cfg.sessions_dir == tmp_path
        assert len(agent.TOOLS) == 4

    def test_memory_agent_standalone(self, tmp_path):
        from agent.brain.memory import MemoryAgent

        cfg = BrainConfig(sessions_dir=tmp_path)
        agent = MemoryAgent(config=cfg)
        assert agent.cfg.sessions_dir == tmp_path
        assert len(agent.TOOLS) == 4

    def test_orchestrator_agent_standalone(self):
        from agent.brain.orchestrator import OrchestratorAgent

        cfg = BrainConfig()
        agent = OrchestratorAgent(config=cfg)
        result = json.loads(agent.handle("check_subtasks", {}))
        assert "tasks" in result

    def test_optimizer_agent_standalone(self):
        from agent.brain.optimizer import OptimizerAgent

        cfg = BrainConfig()
        agent = OptimizerAgent(config=cfg)
        assert len(agent.TOOLS) == 4


class TestBackwardCompat:
    """Verify module-level APIs still work (existing tests depend on these)."""

    def test_module_tools_match_class_tools(self):
        from agent.brain import demo, memory, optimizer, orchestrator, planner, vision

        assert demo.TOOLS is demo.DemoAgent.TOOLS
        assert planner.TOOLS is planner.PlannerAgent.TOOLS
        assert memory.TOOLS is memory.MemoryAgent.TOOLS
        assert vision.TOOLS is vision.VisionAgent.TOOLS
        assert orchestrator.TOOLS is orchestrator.OrchestratorAgent.TOOLS
        assert optimizer.TOOLS is optimizer.OptimizerAgent.TOOLS

    def test_module_handle_delegates(self):
        from agent.brain import demo

        result = json.loads(demo.handle("start_demo", {"scenario": "list"}))
        assert "available_scenarios" in result

    def test_package_exports(self):
        from agent.brain import (
            BrainAgent,
            BrainConfig,
            DemoAgent,
            MemoryAgent,
            OptimizerAgent,
            OrchestratorAgent,
            PlannerAgent,
            VisionAgent,
        )

        assert BrainConfig is not None
        assert issubclass(DemoAgent, BrainAgent)
        assert issubclass(PlannerAgent, BrainAgent)
        assert issubclass(MemoryAgent, BrainAgent)
        assert issubclass(VisionAgent, BrainAgent)
        assert issubclass(OrchestratorAgent, BrainAgent)
        assert issubclass(OptimizerAgent, BrainAgent)
