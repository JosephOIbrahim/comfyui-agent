"""Tests for tool authority scoping — Commandment 5 (Role Isolation)."""

import json

from agent.tool_scope import (
    SCOPE_EXECUTION,
    SCOPE_INTENT,
    SCOPE_VERIFY,
    SCOPES,
    FullScopeDispatcher,
    ScopedDispatcher,
    ToolScope,
)


class TestToolScope:
    """Tests for the ToolScope dataclass."""

    def test_check_allowed(self):
        scope = ToolScope(name="test", allowed_tools=frozenset({"a", "b"}))
        assert scope.check("a") is True
        assert scope.check("b") is True

    def test_check_not_allowed(self):
        scope = ToolScope(name="test", allowed_tools=frozenset({"a"}))
        assert scope.check("c") is False

    def test_denied_overrides_allowed(self):
        scope = ToolScope(
            name="test",
            allowed_tools=frozenset({"a", "b"}),
            denied_tools=frozenset({"b"}),
        )
        assert scope.check("a") is True
        assert scope.check("b") is False

    def test_describe(self):
        scope = ToolScope(
            name="test",
            allowed_tools=frozenset({"a", "b", "c"}),
            denied_tools=frozenset({"x"}),
        )
        desc = scope.describe()
        assert desc["name"] == "test"
        assert desc["allowed_count"] == 3
        assert desc["denied_count"] == 1

    def test_frozen(self):
        """ToolScope is immutable."""
        scope = ToolScope(name="test", allowed_tools=frozenset({"a"}))
        try:
            scope.name = "changed"
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestScopedDispatcher:
    """Tests for ScopedDispatcher enforcement."""

    def _make_mock_dispatch(self, return_value='{"ok": true}'):
        calls = []

        def mock_dispatch(name, tool_input, **kwargs):
            calls.append({"name": name, "input": tool_input, "kwargs": kwargs})
            return return_value

        return mock_dispatch, calls

    def test_allowed_call_passes_through(self):
        dispatch, calls = self._make_mock_dispatch()
        scope = ToolScope(name="test", allowed_tools=frozenset({"get_node_info"}))
        sd = ScopedDispatcher(dispatch, scope)
        result = sd("get_node_info", {"node_name": "KSampler"})
        assert result == '{"ok": true}'
        assert len(calls) == 1
        assert calls[0]["name"] == "get_node_info"

    def test_denied_call_returns_error(self):
        dispatch, calls = self._make_mock_dispatch()
        scope = ToolScope(name="verify", allowed_tools=frozenset({"analyze_image"}))
        sd = ScopedDispatcher(dispatch, scope)
        result = sd("execute_workflow", {"prompt_id": "abc"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not authorized" in parsed["error"]
        assert parsed["scope"] == "verify"
        assert len(calls) == 0  # dispatch was NOT called

    def test_ctx_forwarded(self):
        dispatch, calls = self._make_mock_dispatch()
        scope = ToolScope(name="test", allowed_tools=frozenset({"get_node_info"}))
        fake_ctx = {"session_id": "test-session"}
        sd = ScopedDispatcher(dispatch, scope, ctx=fake_ctx)
        sd("get_node_info", {})
        assert calls[0]["kwargs"]["ctx"] == fake_ctx


class TestPredefinedScopes:
    """Tests for the predefined intent/execution/verify scopes."""

    def test_intent_cannot_execute(self):
        assert SCOPE_INTENT.check("execute_workflow") is False
        assert SCOPE_INTENT.check("execute_with_progress") is False

    def test_intent_cannot_mutate(self):
        assert SCOPE_INTENT.check("apply_workflow_patch") is False
        assert SCOPE_INTENT.check("set_input") is False
        assert SCOPE_INTENT.check("add_node") is False
        assert SCOPE_INTENT.check("connect_nodes") is False

    def test_intent_cannot_judge_quality(self):
        assert SCOPE_INTENT.check("analyze_image") is False
        assert SCOPE_INTENT.check("compare_outputs") is False

    def test_intent_can_read(self):
        assert SCOPE_INTENT.check("get_node_info") is True
        assert SCOPE_INTENT.check("get_editable_fields") is True
        assert SCOPE_INTENT.check("list_models") is True
        assert SCOPE_INTENT.check("discover") is True

    def test_intent_can_capture(self):
        assert SCOPE_INTENT.check("capture_intent") is True
        assert SCOPE_INTENT.check("get_current_intent") is True

    def test_execution_cannot_judge(self):
        assert SCOPE_EXECUTION.check("analyze_image") is False
        assert SCOPE_EXECUTION.check("compare_outputs") is False
        assert SCOPE_EXECUTION.check("suggest_improvements") is False

    def test_execution_can_mutate(self):
        assert SCOPE_EXECUTION.check("add_node") is True
        assert SCOPE_EXECUTION.check("connect_nodes") is True
        assert SCOPE_EXECUTION.check("set_input") is True
        assert SCOPE_EXECUTION.check("apply_workflow_patch") is True

    def test_execution_can_execute(self):
        assert SCOPE_EXECUTION.check("execute_workflow") is True
        assert SCOPE_EXECUTION.check("validate_before_execute") is True

    def test_verify_cannot_mutate(self):
        assert SCOPE_VERIFY.check("apply_workflow_patch") is False
        assert SCOPE_VERIFY.check("set_input") is False
        assert SCOPE_VERIFY.check("add_node") is False
        assert SCOPE_VERIFY.check("connect_nodes") is False

    def test_verify_cannot_execute(self):
        assert SCOPE_VERIFY.check("execute_workflow") is False
        assert SCOPE_VERIFY.check("execute_with_progress") is False

    def test_verify_can_analyze(self):
        assert SCOPE_VERIFY.check("analyze_image") is True
        assert SCOPE_VERIFY.check("compare_outputs") is True
        assert SCOPE_VERIFY.check("suggest_improvements") is True
        assert SCOPE_VERIFY.check("hash_compare_images") is True

    def test_verify_can_record_outcomes(self):
        assert SCOPE_VERIFY.check("record_outcome") is True
        assert SCOPE_VERIFY.check("record_iteration_step") is True
        assert SCOPE_VERIFY.check("finalize_iterations") is True

    def test_all_scopes_can_read_nodes(self):
        """All scopes should be able to inspect nodes (fundamental operation)."""
        for scope_name in ("intent", "execution", "verify"):
            scope = SCOPES[scope_name]
            assert scope.check("get_node_info") is True, f"{scope_name} can't read nodes"
            assert scope.check("get_all_nodes") is True, f"{scope_name} can't list nodes"

    def test_all_scopes_can_discover(self):
        """All scopes should be able to search for models/nodes."""
        for scope_name in ("intent", "execution", "verify"):
            scope = SCOPES[scope_name]
            assert scope.check("discover") is True, f"{scope_name} can't discover"

    def test_scopes_dict_complete(self):
        # Cycle 64: "full" removed — FullScopeDispatcher is the production path
        assert set(SCOPES.keys()) == {"intent", "execution", "verify"}

    def test_no_tool_in_all_denied_lists(self):
        """No tool should be denied in ALL scopes (unreachable tool)."""
        all_denied = set()
        for scope in (SCOPE_INTENT, SCOPE_EXECUTION, SCOPE_VERIFY):
            if not all_denied:
                all_denied = set(scope.denied_tools)
            else:
                all_denied &= set(scope.denied_tools)
        # There should be no tools denied by ALL three scopes
        assert len(all_denied) == 0, f"Tools unreachable from any scope: {all_denied}"


class TestFullScopeDispatcher:
    """Tests for the unrestricted MCP-facing dispatcher."""

    def test_allows_everything(self):
        calls = []

        def mock_dispatch(name, tool_input, **kwargs):
            calls.append(name)
            return '{"ok": true}'

        fsd = FullScopeDispatcher(mock_dispatch)
        result = fsd("execute_workflow", {"prompt_id": "abc"})
        assert result == '{"ok": true}'
        assert len(calls) == 1

    def test_forwards_ctx(self):
        calls = []

        def mock_dispatch(name, tool_input, **kwargs):
            calls.append(kwargs)
            return '{"ok": true}'

        fsd = FullScopeDispatcher(mock_dispatch, ctx="test-ctx")
        fsd("any_tool", {})
        assert calls[0]["ctx"] == "test-ctx"


# ---------------------------------------------------------------------------
# Cycle 61 — tool_scope error JSON is allow_nan=False safe
# ---------------------------------------------------------------------------

class TestScopedDispatcherNaNSafety:
    """Cycle 61: ScopedDispatcher error JSON must be NaN-safe (allow_nan=False)."""

    def test_unauthorized_tool_returns_valid_json(self):
        """Unauthorized tool call must return parseable JSON (not NaN-polluted)."""
        import json
        from agent.tool_scope import ToolScope, ScopedDispatcher
        scope = ToolScope("test_scope", allowed_tools=frozenset({"allowed_tool"}))
        dispatcher = ScopedDispatcher(lambda n, t, **k: '{"ok": true}', scope)
        result = dispatcher("forbidden_tool", {})
        parsed = json.loads(result)  # Must not raise JSONDecodeError
        assert "error" in parsed
        assert "test_scope" in parsed["error"] or "test_scope" in parsed.get("scope", "")
