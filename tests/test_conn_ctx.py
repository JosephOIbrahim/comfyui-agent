"""Tests for agent._conn_ctx — per-connection session name isolation (P0-F).

Covers:
- current_conn_session() returns a stable name within one context
- Different asyncio contexts get different names
- Name propagates correctly into ThreadPoolExecutor threads
- Monkey-patch of agent.tools.handle is visible to parallel workers (P0-H)
"""

import asyncio
import concurrent.futures
import contextvars
import threading


class TestCurrentConnSession:
    """current_conn_session() — returns 'default' outside MCP, UUID inside MCP thread."""

    def test_returns_default_when_contextvar_not_set(self):
        """Outside an MCP handler the function returns 'default' (safe fallback)."""
        import contextvars as _cv
        from agent._conn_ctx import current_conn_session

        # Fresh context with ContextVar never set
        ctx = _cv.copy_context()
        result = ctx.run(current_conn_session)
        assert result == "default"

    def test_returns_contextvar_value_when_set(self):
        """Returns whatever was explicitly set in the ContextVar."""
        import contextvars as _cv
        from agent._conn_ctx import _conn_session, current_conn_session

        ctx = _cv.copy_context()

        def _run():
            _conn_session.set("conn_abcd1234")
            return current_conn_session()

        result = ctx.run(_run)
        assert result == "conn_abcd1234"

    def test_explicit_set_in_executor_thread_works(self):
        """Explicitly setting ContextVar inside executor thread is visible via current_conn_session."""
        from agent._conn_ctx import _conn_session, current_conn_session

        captured = []
        conn_name = "conn_test1234"

        def _worker():
            # mcp_server._handler sets ContextVar explicitly before calling the tool
            _conn_session.set(conn_name)
            captured.append(current_conn_session())

        with concurrent.futures.ThreadPoolExecutor() as ex:
            ex.submit(_worker).result()

        assert captured[0] == conn_name

    def test_default_when_not_set_in_executor_thread(self):
        """Executor thread with no ContextVar set returns 'default', not a UUID."""
        from agent._conn_ctx import current_conn_session

        captured = []

        def _worker():
            captured.append(current_conn_session())

        with concurrent.futures.ThreadPoolExecutor() as ex:
            ex.submit(_worker).result()

        assert captured[0] == "default"

    def test_concurrent_threads_no_bleed(self):
        """Two concurrent threads with different ContextVar values don't bleed into each other."""
        from agent._conn_ctx import _conn_session, current_conn_session

        results = {}
        barrier = threading.Barrier(2)

        def _worker(conn_name, slot):
            _conn_session.set(conn_name)
            barrier.wait()  # Both threads set their value, then both read
            results[slot] = current_conn_session()

        t1 = threading.Thread(target=_worker, args=("conn_aaa", "t1"))
        t2 = threading.Thread(target=_worker, args=("conn_bbb", "t2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "conn_aaa", f"t1 bled: got {results['t1']}"
        assert results["t2"] == "conn_bbb", f"t2 bled: got {results['t2']}"


class TestParallelToolDispatch:
    """P0-H — monkey-patch of agent.tools.handle is visible to executor workers."""

    def test_patch_agent_tools_handle_seen_by_workers(self):
        """Patching agent.tools.handle must be visible to ThreadPoolExecutor."""
        from unittest.mock import patch
        from agent.main import run_agent_turn
        from agent.llm import ToolUseBlock
        from unittest.mock import MagicMock

        call_log = []

        def _patched_handle(name, tool_input, **kw):
            call_log.append(name)
            return '{"ok": true}'

        # Two tool calls → parallel executor path in run_agent_turn
        def _make_response():
            response = MagicMock()
            response.stop_reason = "tool_use"
            response.content = [
                ToolUseBlock(id="t1", name="get_all_nodes", input={}),
                ToolUseBlock(id="t2", name="get_system_stats", input={}),
            ]
            return response

        with patch("agent.main._stream_with_retry", return_value=_make_response()):
            with patch("agent.tools.handle", side_effect=_patched_handle):
                client = MagicMock()
                messages = [{"role": "user", "content": "go"}]
                run_agent_turn(client, messages, "system")

        assert "get_all_nodes" in call_log
        assert "get_system_stats" in call_log

    def test_main_no_longer_exports_handle_tool(self):
        """agent.main must not export handle_tool (removed alias → live module ref)."""
        import agent.main as _main
        assert not hasattr(_main, "handle_tool"), (
            "agent.main.handle_tool was removed — patch agent.tools.handle instead"
        )

    def test_chat_uses_progress_parameter_not_monkey_patch(self):
        """panel/server/chat.py must pass progress= to run_agent_turn, not monkey-patch handle."""
        import ast
        from pathlib import Path

        chat_path = Path(__file__).parent.parent / "panel" / "server" / "chat.py"
        src = chat_path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        found_wrong_patch = False
        found_progress_kwarg = False

        for node in ast.walk(tree):
            # Check no global handle assignment (old monkey-patch pattern)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        attr = f"{ast.unparse(target.value)}.{target.attr}"
                        if "handle_tool" in attr and "main" in attr:
                            found_wrong_patch = True
                        if "agent_tools" in attr and target.attr == "handle":
                            found_wrong_patch = True

            # Check that run_agent_turn is called with progress= keyword
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name == "run_agent_turn":
                    kw_names = {kw.arg for kw in node.keywords}
                    if "progress" in kw_names:
                        found_progress_kwarg = True

        assert not found_wrong_patch, "chat.py must NOT monkey-patch agent.tools.handle or agent.main.handle_tool"
        assert found_progress_kwarg, "chat.py must call run_agent_turn(..., progress=...) to forward the progress reporter"


class TestWorkflowPatchSessionIsolation:
    """workflow_patch._get_state() must read from _conn_session ContextVar.

    Before the fix, _get_state() always returned get_session("default"),
    so the sidebar and the MCP server trampled each other's workflow state.
    After the fix, each connection gets its own WorkflowSession.
    """

    def test_get_state_reads_from_contextvar(self):
        """_get_state() returns different sessions when contextvar differs."""
        import contextvars as _cv
        from agent._conn_ctx import _conn_session
        from agent.tools.workflow_patch import _get_state

        ctx_alpha = _cv.copy_context()
        ctx_beta = _cv.copy_context()

        def _capture_alpha():
            _conn_session.set("sess_alpha")
            return id(_get_state())

        def _capture_beta():
            _conn_session.set("sess_beta")
            return id(_get_state())

        alpha_id = ctx_alpha.run(_capture_alpha)
        beta_id = ctx_beta.run(_capture_beta)

        # Different sessions → different WorkflowSession instances
        assert alpha_id != beta_id

    def test_two_sessions_do_not_trample_each_other(self):
        """Loading a workflow under sess_alpha does not appear in sess_beta."""
        import contextvars as _cv
        from agent._conn_ctx import _conn_session
        from agent.tools.workflow_patch import _get_state
        from agent.workflow_session import _sessions, _registry_lock

        # Clean slate — drop any leftover test sessions
        with _registry_lock:
            for sid in ("sess_alpha_iso", "sess_beta_iso"):
                _sessions.pop(sid, None)

        wf_alpha = {"node_1": {"class_type": "AlphaCheckpoint", "inputs": {}}}
        wf_beta = {"node_2": {"class_type": "BetaCheckpoint", "inputs": {}}}

        ctx_alpha = _cv.copy_context()
        ctx_beta = _cv.copy_context()

        def _write_alpha():
            _conn_session.set("sess_alpha_iso")
            state = _get_state()
            state["current_workflow"] = wf_alpha
            state["loaded_path"] = "/tmp/alpha.json"

        def _write_beta():
            _conn_session.set("sess_beta_iso")
            state = _get_state()
            state["current_workflow"] = wf_beta
            state["loaded_path"] = "/tmp/beta.json"

        def _read_alpha():
            _conn_session.set("sess_alpha_iso")
            return _get_state().get("current_workflow"), _get_state().get("loaded_path")

        ctx_alpha.run(_write_alpha)
        ctx_beta.run(_write_beta)

        alpha_wf, alpha_path = ctx_alpha.run(_read_alpha)

        # Alpha must still see ITS workflow, not beta's
        assert alpha_wf == wf_alpha
        assert alpha_path == "/tmp/alpha.json"

        # Cleanup
        with _registry_lock:
            _sessions.pop("sess_alpha_iso", None)
            _sessions.pop("sess_beta_iso", None)

    def test_default_session_unchanged_when_contextvar_unset(self):
        """CLI / test code with no contextvar still hits the 'default' session."""
        from agent.tools.workflow_patch import _get_state
        from agent.workflow_session import get_session

        # No contextvar set → _get_state() should equal get_session("default")
        assert _get_state() is get_session("default")


class TestSharedSessionHelpers:
    """agent/_session_helpers.py — shared transport helpers used by both
    ui/server/routes.py (sidebar) and panel/server/chat.py (panel).

    These tests prove the helpers correctly set _conn_session AND the
    thread-local correlation ID inside the worker, so both transports get
    the same per-conversation isolation guarantees.
    """

    def test_spawn_with_session_sets_both_contextvar_and_corr_id(self):
        """spawn_with_session must set _conn_session + correlation ID inside the worker."""
        from agent._session_helpers import spawn_with_session
        from agent._conn_ctx import current_conn_session
        from agent.logging_config import get_correlation_id

        captured = {}

        def _target():
            captured["session"] = current_conn_session()
            captured["corr"] = get_correlation_id()

        thread = spawn_with_session(_target, args=(), session_id="conv_panel_a")
        thread.start()
        thread.join(timeout=2.0)

        assert captured["session"] == "conv_panel_a"
        assert captured["corr"] == "conv_panel_a"

    def test_spawn_with_session_isolates_concurrent_workers(self):
        """Two concurrent spawns with different session_ids must not bleed."""
        from agent._session_helpers import spawn_with_session
        from agent._conn_ctx import current_conn_session

        results = {}
        barrier = threading.Barrier(2)

        def _target(slot):
            barrier.wait()  # Both threads enter at the same time
            results[slot] = current_conn_session()

        t1 = spawn_with_session(_target, args=("t1",), session_id="conv_alpha")
        t2 = spawn_with_session(_target, args=("t2",), session_id="conv_beta")
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        assert results["t1"] == "conv_alpha"
        assert results["t2"] == "conv_beta"

    def test_run_in_executor_with_session_sets_contextvar(self):
        """run_in_executor_with_session must set _conn_session inside the executor."""
        from agent._session_helpers import run_in_executor_with_session
        from agent._conn_ctx import current_conn_session

        captured = {}

        def _target():
            captured["session"] = current_conn_session()

        async def _go():
            loop = asyncio.get_running_loop()
            await run_in_executor_with_session(loop, _target, session_id="conv_executor_x")

        asyncio.run(_go())
        assert captured["session"] == "conv_executor_x"


class TestCLISessionContextvar:
    """agent run --session foo must thread the session_id through
    _conn_session ContextVar so all tool calls inside run_interactive
    see the right session.

    Cycle 15 fix: cycles 0+1+4+12 fixed sidebar/MCP/panel/stage but the
    CLI was the last transport that didn't set the contextvar. Without
    this fix, `agent run --session foo` was functionally equivalent to
    `agent run` for tool isolation purposes.
    """

    def teardown_method(self, method):
        """Reset the global _shutdown flag in agent.main after each test.

        cli.run calls _save_and_exit() at the end of run() which invokes
        request_shutdown() — that sets agent.main._shutdown. Subsequent
        TestRunAgentTurn tests see the flag set and bail out early. We
        clear it here so the leak doesn't propagate.
        """
        from agent.main import _shutdown
        _shutdown.clear()

    def test_cli_run_sets_conn_session_from_session_flag(self):
        """`agent run --session foo` must set _conn_session to 'foo' before run_interactive."""
        from unittest.mock import patch, MagicMock
        from typer.testing import CliRunner
        from agent.cli import app
        from agent._conn_ctx import current_conn_session

        captured = {}

        def _capture_session(*args, **kwargs):
            # When run_interactive is called from inside cli.run, the
            # contextvar should already be set.
            captured["session"] = current_conn_session()

        runner = CliRunner()
        # Mock everything that would do real work — we only care about the
        # contextvar being set when run_interactive fires.
        with patch("agent.cli.session_tools") as mock_session_tools, \
             patch("agent.main.run_interactive", side_effect=_capture_session), \
             patch("agent.main.create_client", return_value=MagicMock()), \
             patch("agent.cli.ANTHROPIC_API_KEY", "sk-test"), \
             patch("agent.cli.signal.signal"), \
             patch("agent.cli.atexit.register"):
            mock_session_tools.handle.return_value = '{"saved_at": "now"}'
            result = runner.invoke(app, ["run", "--session", "foo_test_sid"])

        # The capture happens inside cli.run, before the finally block
        # resets the contextvar. So at capture time, the session_id is set.
        assert "session" in captured, (
            f"run_interactive was not called. CLI exit code: {result.exit_code}, "
            f"output: {result.output}"
        )
        assert captured["session"] == "foo_test_sid", (
            f"Expected 'foo_test_sid', got {captured['session']!r}"
        )

    def test_cli_run_without_session_uses_default(self):
        """`agent run` (no --session flag) sets contextvar to 'default'."""
        from unittest.mock import patch, MagicMock
        from typer.testing import CliRunner
        from agent.cli import app
        from agent._conn_ctx import current_conn_session

        captured = {}

        def _capture_session(*args, **kwargs):
            captured["session"] = current_conn_session()

        runner = CliRunner()
        with patch("agent.main.run_interactive", side_effect=_capture_session), \
             patch("agent.main.create_client", return_value=MagicMock()), \
             patch("agent.cli.ANTHROPIC_API_KEY", "sk-test"), \
             patch("agent.cli.signal.signal"), \
             patch("agent.cli.atexit.register"):
            result = runner.invoke(app, ["run"])

        assert "session" in captured, (
            f"run_interactive was not called. CLI exit code: {result.exit_code}, "
            f"output: {result.output}"
        )
        assert captured["session"] == "default"

    def test_cli_resets_contextvar_after_run_interactive(self):
        """The try/finally in cli.run must reset _conn_session after the loop exits."""
        from unittest.mock import patch, MagicMock
        from typer.testing import CliRunner
        from agent.cli import app
        from agent._conn_ctx import _conn_session

        runner = CliRunner()
        with patch("agent.main.run_interactive", return_value=None), \
             patch("agent.main.create_client", return_value=MagicMock()), \
             patch("agent.cli.ANTHROPIC_API_KEY", "sk-test"), \
             patch("agent.cli.signal.signal"), \
             patch("agent.cli.atexit.register"):
            runner.invoke(app, ["run", "--session", "cleanup_test"])

        # After cli.run returns, the contextvar should be reset to its
        # prior state (whatever the autouse _reset_conn_session fixture
        # set, typically "default" or absent).
        try:
            current = _conn_session.get()
            # The autouse fixture pre-sets "default" before each test,
            # so post-cli the contextvar should still be "default".
            assert current == "default", (
                f"Expected contextvar reset to 'default', got {current!r}"
            )
        except LookupError:
            pass  # Also acceptable — original state was unset


class TestOrchestratorSubtaskContext:
    """orchestrator.spawn_subtask must propagate _conn_session to the
    spawned worker thread via contextvars.copy_context(), otherwise
    subtasks fall back to the "default" session.
    """

    def test_subtask_inherits_parent_contextvar(self):
        """A subtask spawned inside a session must see that session inside the worker."""
        from agent._conn_ctx import _conn_session, current_conn_session

        # Build a minimal subtask thread the same way orchestrator does:
        # parent_ctx = contextvars.copy_context() → Thread(target=parent_ctx.run, args=(_worker,))
        captured = {}

        def _worker():
            captured["session"] = current_conn_session()

        def _spawn_in_parent_ctx():
            parent_ctx = contextvars.copy_context()
            t = threading.Thread(target=parent_ctx.run, args=(_worker,), daemon=True)
            t.start()
            t.join(timeout=2.0)

        # Set the parent contextvar, then spawn — worker should inherit it.
        ctx = contextvars.copy_context()
        def _runner():
            _conn_session.set("conv_parent_session")
            _spawn_in_parent_ctx()

        ctx.run(_runner)

        assert captured["session"] == "conv_parent_session"


class TestStageGateInteraction:
    """Stage tools must respect the gate AND let the gate see stage state.

    Before the fix, the gate's session_active/has_undo calculation only
    looked at workflow_patch state. A REVERSIBLE stage tool like stage_write
    was incorrectly DENIED when a USD stage existed but no workflow was
    loaded. After the fix, the gate falls back to checking
    SessionContext.stage for stage_* tool names.
    """

    def _run_in_session(self, session_id: str, fn):
        """Run fn() with _conn_session set to session_id in a copied context."""
        import contextvars as _cv
        from agent._conn_ctx import _conn_session

        copied = _cv.copy_context()

        def _runner():
            _conn_session.set(session_id)
            return fn()

        return copied.run(_runner)

    def _clean_session(self, session_id: str):
        """Wipe both the WorkflowSession and SessionContext for this session_id."""
        from agent.workflow_session import _sessions, _registry_lock
        from agent.session_context import _registry as _ctx_registry

        with _registry_lock:
            _sessions.pop(session_id, None)
        _ctx_registry.destroy(session_id)

    def test_stage_read_always_allowed_read_only_bypass(self):
        """stage_read is READ_ONLY → gate bypasses all checks regardless of state."""
        from agent import tools as _tools

        sid = "test_stage_gate_read"
        self._clean_session(sid)

        def _call():
            return _tools.handle("stage_read", {
                "prim_path": "/Test/foo",
                "attr_name": "bar",
            })

        result = self._run_in_session(sid, _call)

        # READ_ONLY tools never produce a "Gate denied" — even if usd-core
        # isn't installed and the handler returns _NO_STAGE, the gate must
        # not block.
        assert "Gate denied" not in result
        self._clean_session(sid)

    def test_stage_write_denied_when_no_workflow_and_no_stage(self):
        """stage_write is REVERSIBLE → denied when neither workflow nor stage exists."""
        from agent import tools as _tools

        sid = "test_stage_gate_write_empty"
        self._clean_session(sid)

        def _call():
            return _tools.handle("stage_write", {
                "prim_path": "/Test/foo",
                "attr_name": "bar",
                "value": 42,
            })

        result = self._run_in_session(sid, _call)

        # Gate should DENY — no workflow loaded, no stage attached
        assert "Gate denied" in result or "no active session" in result.lower()
        self._clean_session(sid)

    def test_stage_write_allowed_when_stage_exists_without_workflow(self):
        """stage_write should pass the gate when SessionContext.stage exists."""
        from agent import tools as _tools
        from agent.session_context import get_session_context

        sid = "test_stage_gate_write_with_stage"
        self._clean_session(sid)

        # Mock stage that records the write call
        class _MockStage:
            def __init__(self):
                self.writes = []
                self.delta_count = 0

            def write(self, prim_path, attr_name, value, node_type=None):
                self.writes.append((prim_path, attr_name, value, node_type))

            def list_deltas(self):
                return []

        mock_stage = _MockStage()

        def _call():
            sess_ctx = get_session_context(sid)
            sess_ctx.stage = mock_stage
            return _tools.handle("stage_write", {
                "prim_path": "/Test/foo",
                "attr_name": "bar",
                "value": 42,
            })

        result = self._run_in_session(sid, _call)

        # Gate must NOT deny — stage exists, so the stage-state fallback
        # in tools/__init__.py should set session_active=True.
        assert "Gate denied" not in result, f"Gate incorrectly denied: {result}"
        # Handler should have been reached and the mock recorded the write.
        assert len(mock_stage.writes) == 1
        assert mock_stage.writes[0] == ("/Test/foo", "bar", 42, None)
        self._clean_session(sid)

    def test_stage_rollback_always_locked_destructive(self):
        """stage_rollback is DESTRUCTIVE → gate returns LOCKED even with stage."""
        from agent import tools as _tools
        from agent.session_context import get_session_context

        sid = "test_stage_gate_rollback"
        self._clean_session(sid)

        # Even with a stage attached, DESTRUCTIVE tools return LOCKED.
        class _MockStage:
            delta_count = 5

            def rollback_to(self, n):
                return n

        def _call():
            sess_ctx = get_session_context(sid)
            sess_ctx.stage = _MockStage()
            return _tools.handle("stage_rollback", {"n_deltas": 1})

        result = self._run_in_session(sid, _call)

        # LOCKED → "destructive operation" in the error message per
        # agent/tools/__init__.py:253-259.
        assert "destructive" in result.lower(), f"Expected LOCKED, got: {result}"
        self._clean_session(sid)
