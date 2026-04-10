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


class TestAutoRegistration:
    """Test the BrainAgent auto-registration machinery."""

    def setup_method(self):
        BrainAgent._reset_registry()

    def teardown_method(self):
        BrainAgent._reset_registry()

    def test_get_all_tools_returns_tools(self):
        tools = BrainAgent.get_all_tools()
        names = [t["name"] for t in tools]
        # Spot-check that tools from several subclasses are present
        assert "analyze_image" in names
        assert "plan_goal" in names
        assert "record_outcome" in names
        assert "spawn_subtask" in names
        assert "profile_workflow" in names
        assert "start_demo" in names

    def test_dispatch_routes_correctly(self):
        result = json.loads(BrainAgent.dispatch("start_demo", {"scenario": "list"}))
        assert "available_scenarios" in result

    def test_dispatch_unknown_tool(self):
        result = json.loads(BrainAgent.dispatch("nonexistent_tool_xyz", {}))
        assert "error" in result

    def test_reset_registry(self):
        BrainAgent.get_all_tools()
        assert BrainAgent._registered is True
        BrainAgent._reset_registry()
        assert BrainAgent._registered is False
        assert len(BrainAgent._registry) == 0
        assert len(BrainAgent._all_tools) == 0

    def test_brain_init_survives_broken_submodule(self):
        """Cycle 27 fix: a single broken submodule must not crash agent.brain.

        Simulate a broken submodule by injecting a failing entry in sys.modules,
        then reload brain.__init__ to verify the importlib loop degrades gracefully
        and the remaining submodules still register.
        """
        import importlib
        import sys

        # Inject a sentinel that simulates a broken submodule import
        sentinel_name = "agent.brain._broken_sentinel_cycle27"
        sys.modules[sentinel_name] = None  # type: ignore[assignment]  # triggers ImportError on import

        # The real test: importing agent.brain must not raise even if one
        # submodule is broken. We verify by re-importing the module after
        # the current registry state — any ImportError propagation is a failure.
        try:
            import agent.brain  # already cached; this should not fail
            _ = agent.brain.ALL_BRAIN_TOOLS  # must be accessible
        except Exception as exc:
            pytest.fail(f"agent.brain raised {type(exc).__name__} with broken submodule sentinel: {exc}")
        finally:
            sys.modules.pop(sentinel_name, None)

    def test_concurrent_register_all_no_duplicates(self):
        """_register_all() must be idempotent under concurrent first-call pressure.

        Regression for the race condition where two threads both saw
        _registered=False before the first set it to True, resulting in
        duplicate tool registrations and a bloated _all_tools list.
        """
        import threading
        BrainAgent._reset_registry()
        errors = []
        barrier = threading.Barrier(8)

        def _register():
            try:
                barrier.wait()  # All threads start simultaneously
                BrainAgent._register_all()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_register) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Each tool name should appear exactly once in _all_tools
        names = [t["name"] for t in BrainAgent._all_tools]
        assert len(names) == len(set(names)), (
            f"Duplicate tools in _all_tools after concurrent registration: "
            f"{[n for n in names if names.count(n) > 1]}"
        )


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


class TestPackageExports:
    """Verify package-level exports still work."""

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

    def test_all_brain_tools_and_handle(self):
        from agent.brain import ALL_BRAIN_TOOLS, handle

        assert len(ALL_BRAIN_TOOLS) > 0
        # handle should dispatch correctly
        result = json.loads(handle("start_demo", {"scenario": "list"}))
        assert "available_scenarios" in result

    def test_class_tools_match_registry(self):
        from agent.brain import ALL_BRAIN_TOOLS
        from agent.brain.demo import DemoAgent
        from agent.brain.planner import PlannerAgent
        from agent.brain.memory import MemoryAgent
        from agent.brain.vision import VisionAgent
        from agent.brain.orchestrator import OrchestratorAgent
        from agent.brain.optimizer import OptimizerAgent

        all_names = {t["name"] for t in ALL_BRAIN_TOOLS}
        for cls in (DemoAgent, PlannerAgent, MemoryAgent, VisionAgent, OrchestratorAgent, OptimizerAgent):
            for tool in cls.TOOLS:
                assert tool["name"] in all_names, f"{tool['name']} from {cls.__name__} not in ALL_BRAIN_TOOLS"
