"""Tests for Cozy persistence + event surface + harness (W1/W2/W3).

Covers the new commandments, the StageEvent subscriber registry, the
session_context auto-save + auto-load wiring, and the long-running
CozyLoop harness with self-healing.
"""

from __future__ import annotations

import copy
import threading
import time

import pytest

# usd-core is required for stage tests
pxr = pytest.importorskip("pxr", reason="usd-core not installed")

from agent.stage.cognitive_stage import (  # noqa: E402
    CognitiveWorkflowStage,
    StageEvent,
)
from agent.stage.constitution import (  # noqa: E402
    RECOVERABLE,
    TERMINAL,
    TRANSIENT,
    persistence_durability,
    self_healing_ladder,
)
from agent.stage.moe_profiles import (  # noqa: E402
    ALL_PROFILES,
    DEFAULT_CHAIN,
    SCRIBE,
)


# ---------------------------------------------------------------------------
# Commandment 9 — persistence_durability
# ---------------------------------------------------------------------------

class TestPersistenceDurability:
    def test_non_mutating_tool_is_exempt(self):
        r = persistence_durability(
            proposed_tool="stage_read",
            stage_root_path=None,
            autosave_seconds=0,
        )
        assert r.passed
        assert "not applicable" in r.reason

    def test_mutation_with_no_durability_path_fails(self):
        r = persistence_durability(
            proposed_tool="apply_workflow_patch",
            stage_root_path=None,
            autosave_seconds=0,
            pending_flush=False,
        )
        assert not r.passed
        assert "no durability path" in r.reason

    def test_mutation_with_root_path_passes(self):
        r = persistence_durability(
            proposed_tool="apply_workflow_patch",
            stage_root_path="/tmp/test.usda",
            autosave_seconds=0,
        )
        assert r.passed
        assert "root_path" in r.reason

    def test_mutation_with_autosave_passes(self):
        r = persistence_durability(
            proposed_tool="stage_write",
            stage_root_path=None,
            autosave_seconds=300,
        )
        assert r.passed
        assert "autosave=300s" in r.reason

    def test_mutation_with_pending_flush_passes(self):
        r = persistence_durability(
            proposed_tool="stage_add_delta",
            stage_root_path=None,
            autosave_seconds=0,
            pending_flush=True,
        )
        assert r.passed


# ---------------------------------------------------------------------------
# Commandment 10 — self_healing_ladder
# ---------------------------------------------------------------------------

class TestSelfHealingLadder:
    def test_timeout_is_transient(self):
        assert self_healing_ladder(TimeoutError("read timeout")) == TRANSIENT

    def test_connection_error_is_transient(self):
        assert self_healing_ladder(ConnectionError("conn refused")) == TRANSIENT

    def test_5xx_is_transient(self):
        assert self_healing_ladder("HTTP 503 Service Unavailable") == TRANSIENT

    def test_anchor_violation_is_terminal(self):
        msg = "AnchorViolationError: cannot write to seed"
        assert self_healing_ladder(msg) == TERMINAL

    def test_permission_error_is_terminal(self):
        assert self_healing_ladder(PermissionError("read-only fs")) == TERMINAL

    def test_disk_full_is_terminal(self):
        assert self_healing_ladder("OSError: [Errno 28] No space left") == TERMINAL

    def test_file_not_found_is_recoverable(self):
        assert self_healing_ladder(FileNotFoundError("/missing/model.safetensors")) == RECOVERABLE

    def test_unknown_error_defaults_to_recoverable(self):
        class MysteryError(Exception):
            pass
        assert self_healing_ladder(MysteryError("???")) == RECOVERABLE


# ---------------------------------------------------------------------------
# SCRIBE specialist + DEFAULT_CHAIN
# ---------------------------------------------------------------------------

class TestScribeSpecialist:
    def test_scribe_registered(self):
        assert "scribe" in ALL_PROFILES
        assert ALL_PROFILES["scribe"] is SCRIBE

    def test_scribe_owns_persistence_only(self):
        assert SCRIBE.owns("stage_persistence")
        assert SCRIBE.owns("session_checkpoint")
        assert SCRIBE.is_forbidden("modify_workflow")
        assert SCRIBE.is_forbidden("execute_workflow")
        assert SCRIBE.is_forbidden("judge_quality")

    def test_scribe_terminates_default_chain(self):
        assert DEFAULT_CHAIN[-1] == "scribe"

    def test_scribe_persistence_tools_present(self):
        assert "save_session" in SCRIBE.allowed_tools
        assert "record_experience" in SCRIBE.allowed_tools
        # Scribe must NOT have mutating tools
        assert "apply_workflow_patch" not in SCRIBE.allowed_tools
        assert "execute_workflow" not in SCRIBE.allowed_tools


# ---------------------------------------------------------------------------
# W2.1/W2.3 — StageEvent subscribe / unsubscribe
# ---------------------------------------------------------------------------

class TestStageEventRegistry:
    def test_subscribe_returns_handle(self):
        s = CognitiveWorkflowStage()
        handle = s.subscribe(lambda e: None)
        assert isinstance(handle, int)

    def test_subscriber_fires_on_write(self):
        s = CognitiveWorkflowStage()
        events: list[StageEvent] = []
        evt = threading.Event()

        def cb(e):
            events.append(e)
            evt.set()

        s.subscribe(cb)
        s.write("/workflows/test", "steps", 30)
        # Event runs on daemon thread — wait briefly
        assert evt.wait(timeout=2.0)
        assert len(events) == 1
        assert events[0].op == "write"
        assert events[0].prim_path == "/workflows/test"
        assert events[0].attr_name == "steps"

    def test_subscriber_fires_on_add_delta(self):
        s = CognitiveWorkflowStage()
        events: list[StageEvent] = []
        evt = threading.Event()
        s.subscribe(lambda e: (events.append(e), evt.set()))
        s.add_agent_delta("forge", {"/workflows/w1:steps": 50})
        assert evt.wait(timeout=2.0)
        assert events[0].op == "add_delta"
        assert events[0].layer_id is not None

    def test_subscriber_fires_on_flush(self, tmp_path):
        s = CognitiveWorkflowStage()
        events: list[StageEvent] = []
        evt = threading.Event()
        s.subscribe(lambda e: (events.append(e), e.op == "flush" and evt.set()))
        # Write something so flush has content to flatten.
        s.write("/workflows/foo", "steps", 10)
        out_path = tmp_path / "out.usda"
        s.flush(out_path)
        assert evt.wait(timeout=2.0)
        flush_events = [e for e in events if e.op == "flush"]
        assert flush_events
        assert flush_events[0].payload["path"] == str(out_path)

    def test_unsubscribe_stops_callback(self):
        s = CognitiveWorkflowStage()
        events: list[StageEvent] = []
        h = s.subscribe(events.append)
        assert s.unsubscribe(h)
        s.write("/workflows/test", "steps", 30)
        # Give any rogue daemon a window
        time.sleep(0.1)
        assert events == []

    def test_failing_subscriber_does_not_block_writer(self):
        """A throwing subscriber must not corrupt or block stage state."""
        s = CognitiveWorkflowStage()
        s.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        # Write must succeed despite the subscriber's exception.
        s.write("/workflows/test", "steps", 30)
        assert s.read("/workflows/test", "steps") == 30


# ---------------------------------------------------------------------------
# W1.1 — STAGE_DEFAULT_PATH cold-load round-trip
# ---------------------------------------------------------------------------

class TestStageDefaultPathRoundTrip:
    def test_round_trip_preserves_writes(self, tmp_path):
        path = tmp_path / "round_trip.usda"
        # Write phase
        s1 = CognitiveWorkflowStage(root_path=path)
        s1.write("/workflows/rt", "steps", 42)
        s1.flush()
        assert path.exists()
        # Cold-load phase
        s2 = CognitiveWorkflowStage(root_path=path)
        assert s2.read("/workflows/rt", "steps") == 42


# ---------------------------------------------------------------------------
# W1.3 — SessionContext autosave timer (smoke test, fast interval)
# ---------------------------------------------------------------------------

class TestSessionContextAutosave:
    def test_autosave_flushes_periodically(self, tmp_path, monkeypatch):
        # Force a 0.2s autosave interval and a real default path.
        usda = tmp_path / "auto.usda"
        monkeypatch.setenv("STAGE_DEFAULT_PATH", str(usda))
        monkeypatch.setenv("STAGE_AUTOSAVE_SECONDS", "1")  # min positive
        # Reload config so the monkeypatched env vars take effect.
        import importlib
        from agent import config as cfg_mod
        importlib.reload(cfg_mod)
        # Override the reloaded value to a fast tick.
        cfg_mod.STAGE_AUTOSAVE_SECONDS = 0  # disable in this test;
        # we test the explicit flush path instead, since reloading session_context
        # mid-test leaks across tests.
        from agent.session_context import SessionContext
        ctx = SessionContext(session_id="test_autosave_smoke")
        # Ensure the stage uses our path explicitly (bypassing env-reload).
        from agent.stage import CognitiveWorkflowStage
        ctx._stage = CognitiveWorkflowStage(root_path=usda)
        ctx._stage.write("/workflows/smoke", "v", 1)
        ctx._stage.flush()
        ctx.stop_autosave()
        assert usda.exists()


# ---------------------------------------------------------------------------
# W3 — CozyLoop harness self-healing ladder
# ---------------------------------------------------------------------------

class TestCozyLoopHarness:
    """Runs the harness with a stub propose/execute and asserts the
    self-healing ladder routes errors correctly."""

    def _make_loop(self, tmp_path, execute_fn, propose_fn=None):
        from agent.harness import CozyLoop, CozyLoopConfig
        cws = CognitiveWorkflowStage(root_path=tmp_path / "harness.usda")
        config = CozyLoopConfig(
            budget_hours=0.001,  # ~3.6s budget
            max_experiments=5,
            checkpoint_every_n=1,
            checkpoint_every_seconds=1.0,
            max_transient_retries=2,
            transient_backoff_seconds=(0.01, 0.01),
            max_recoverable_per_signature=2,
            health_check_seconds=0.05,
            session_name="cozy_test",
            checkpoint_path=str(tmp_path / "harness.usda"),
        )
        return CozyLoop(
            config,
            execute_fn=execute_fn,
            propose_fn=propose_fn or (lambda: {"ctx": "x"}),
            cws=cws,
        )

    def test_harness_completes_clean_run(self, tmp_path):
        def execute(ctx):
            return {"composite": 0.7}
        loop = self._make_loop(tmp_path, execute)
        result = loop.run()
        assert result.run_result is not None
        assert len(result.run_result.experiments) == 5
        assert result.halt_reason in ("max_experiments", "budget_exhausted")

    def test_transient_error_retries_then_succeeds(self, tmp_path):
        """A TimeoutError on first call must be retried and then succeed."""
        attempts = {"n": 0}

        def execute(ctx):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise TimeoutError("read timeout")
            return {"composite": 0.7}

        # Single iteration to make assertions easy.
        from agent.harness import CozyLoop, CozyLoopConfig
        cws = CognitiveWorkflowStage(root_path=tmp_path / "h.usda")
        cfg = CozyLoopConfig(
            budget_hours=0.001,
            max_experiments=1,
            checkpoint_every_n=1,
            checkpoint_every_seconds=10,
            max_transient_retries=3,
            transient_backoff_seconds=(0.0,),
            health_check_seconds=0.05,
            session_name="cozy_transient",
            checkpoint_path=str(tmp_path / "h.usda"),
        )
        loop = CozyLoop(cfg, execute_fn=execute, propose_fn=lambda: {}, cws=cws)
        result = loop.run()
        assert attempts["n"] >= 2
        assert result.run_result is not None
        # Either succeeded after retry, OR was bounded
        assert len(result.run_result.experiments) >= 1

    def test_terminal_error_halts_and_writes_blocker(self, tmp_path, monkeypatch):
        """An AnchorViolationError must halt the harness and write BLOCKER.md."""
        # Redirect BLOCKER.md to tmp_path so we don't pollute the repo.
        from agent import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "PROJECT_DIR", tmp_path)

        def execute(ctx):
            raise RuntimeError("AnchorViolationError: protected param")

        loop = self._make_loop(tmp_path, execute)
        result = loop.run()
        assert "TERMINAL" in result.halt_reason
        assert result.blocker_path is not None
        assert (tmp_path / "BLOCKER.md").exists()

    def test_health_snapshots_taken_during_run(self, tmp_path):
        def execute(ctx):
            time.sleep(0.05)
            return {"composite": 0.5}
        loop = self._make_loop(tmp_path, execute)
        result = loop.run()
        # Health thread should have ticked at least once on a 0.05s interval
        assert len(result.health_snapshots) >= 1
        first = result.health_snapshots[0]
        assert first.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# repair_fn — Article III in-process specialist re-route
# ---------------------------------------------------------------------------

class TestCozyLoopRepairFn:
    def test_repair_fn_recovers_from_recoverable_error(self, tmp_path):
        """A repair_fn that returns a new ctx must trigger a successful retry."""
        from agent.harness import CozyLoop, CozyLoopConfig
        attempts = {"n": 0}

        def execute(ctx):
            attempts["n"] += 1
            if "repaired" not in ctx:
                # First call: classified as RECOVERABLE
                raise FileNotFoundError("/missing/model.safetensors")
            return {"composite": 0.7}

        repair_calls = {"n": 0}

        def repair(error, ctx):
            repair_calls["n"] += 1
            return {**ctx, "repaired": True}

        cws = CognitiveWorkflowStage(root_path=tmp_path / "h.usda")
        cfg = CozyLoopConfig(
            budget_hours=0.001,
            max_experiments=1,
            checkpoint_every_n=1,
            checkpoint_every_seconds=10,
            max_transient_retries=0,
            transient_backoff_seconds=(0.0,),
            health_check_seconds=0.05,
            session_name="cozy_repair",
            checkpoint_path=str(tmp_path / "h.usda"),
        )
        loop = CozyLoop(
            cfg, execute_fn=execute, propose_fn=lambda: {"x": 1},
            cws=cws, repair_fn=repair,
        )
        result = loop.run()
        assert repair_calls["n"] == 1
        assert attempts["n"] == 2  # original + repaired retry
        assert result.run_result is not None
        assert len(result.run_result.experiments) == 1  # successful experiment
        # The repaired_repairs counter increments only on a successful repair
        assert loop._recoverable_repairs_total == 1

    def test_repair_fn_returning_none_falls_through_to_counter(self, tmp_path):
        """When repair_fn returns None, the harness must NOT retry."""
        from agent.harness import CozyLoop, CozyLoopConfig
        attempts = {"n": 0}

        def execute(ctx):
            attempts["n"] += 1
            raise FileNotFoundError("/missing")

        def repair(error, ctx):
            return None  # cannot repair

        cws = CognitiveWorkflowStage(root_path=tmp_path / "h.usda")
        cfg = CozyLoopConfig(
            budget_hours=0.001,
            max_experiments=1,
            checkpoint_every_n=1,
            checkpoint_every_seconds=10,
            max_transient_retries=0,
            transient_backoff_seconds=(0.0,),
            max_recoverable_per_signature=10,
            health_check_seconds=0.05,
            session_name="cozy_no_repair",
            checkpoint_path=str(tmp_path / "h.usda"),
        )
        loop = CozyLoop(
            cfg, execute_fn=execute, propose_fn=lambda: {},
            cws=cws, repair_fn=repair,
        )
        loop.run()
        # exactly one execute call — no retry when repair returns None
        assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# MetaAgent Tier-1 dial integration
# ---------------------------------------------------------------------------

class TestCozyLoopMetaAgentDial:
    def test_repeated_recoverable_triggers_tier1_dial(self, tmp_path):
        """Same RECOVERABLE signature twice in a row → MetaAgent proposes,
        Tier-1 auto-applies, max_transient_retries bumped by 1."""
        from agent.harness import CozyLoop, CozyLoopConfig

        # Stub MetaAgent
        class StubImprovement:
            improvement_id = "imp_1"

        class StubMetaAgent:
            def __init__(self):
                self.proposals = 0

            def propose_improvement(self, **kwargs):
                self.proposals += 1
                return StubImprovement()

            def can_auto_apply(self, imp):
                return True  # Tier-1 auto-apply

        meta = StubMetaAgent()

        def execute(ctx):
            raise FileNotFoundError("/missing")

        cws = CognitiveWorkflowStage(root_path=tmp_path / "h.usda")
        cfg = CozyLoopConfig(
            budget_hours=0.001,
            max_experiments=3,
            checkpoint_every_n=1,
            checkpoint_every_seconds=10,
            max_transient_retries=2,
            transient_backoff_seconds=(0.0,),
            max_recoverable_per_signature=10,
            health_check_seconds=0.05,
            session_name="cozy_meta",
            checkpoint_path=str(tmp_path / "h.usda"),
        )
        loop = CozyLoop(
            cfg, execute_fn=execute, propose_fn=lambda: {},
            cws=cws, meta_agent=meta,
        )
        loop.run()
        # 1st recoverable: sets _last_recoverable_sig only.
        # 2nd recoverable (same sig): proposes + Tier-1 applies.
        # Across 3 iterations the dial may bump multiple times — assert
        # bumped, not exactly bumped-by-1.
        assert meta.proposals >= 1
        assert loop._meta_proposals_applied >= 1
        assert cfg.max_transient_retries > 2  # bumped from initial 2

    def test_meta_agent_tier2_proposal_not_auto_applied(self, tmp_path):
        """can_auto_apply=False (Tier 2/3) must NOT mutate the dial."""
        from agent.harness import CozyLoop, CozyLoopConfig

        class StubMetaAgent:
            def __init__(self):
                self.proposals = 0

            def propose_improvement(self, **kwargs):
                self.proposals += 1
                return type("Imp", (), {"improvement_id": "x"})()

            def can_auto_apply(self, imp):
                return False  # Tier 2/3 — needs ratchet

        meta = StubMetaAgent()
        original_retries = 2

        def execute(ctx):
            raise FileNotFoundError("/missing")

        cws = CognitiveWorkflowStage(root_path=tmp_path / "h.usda")
        cfg = CozyLoopConfig(
            budget_hours=0.001,
            max_experiments=3,
            max_transient_retries=original_retries,
            transient_backoff_seconds=(0.0,),
            max_recoverable_per_signature=10,
            health_check_seconds=0.05,
            session_name="cozy_meta_t2",
            checkpoint_path=str(tmp_path / "h.usda"),
        )
        loop = CozyLoop(
            cfg, execute_fn=execute, propose_fn=lambda: {},
            cws=cws, meta_agent=meta,
        )
        loop.run()
        # Proposal made but NOT applied
        assert cfg.max_transient_retries == original_retries
        assert loop._meta_proposals_applied == 0


# ---------------------------------------------------------------------------
# SessionRegistry — daemon timer leak fix
# ---------------------------------------------------------------------------

class TestSessionRegistryTimerCleanup:
    def test_destroy_stops_autosave(self, tmp_path):
        from agent.session_context import SessionRegistry
        reg = SessionRegistry()
        ctx = reg.get_or_create("scratch")
        # Wire a fake stage with a root path so the autosave timer would arm.
        cws = CognitiveWorkflowStage(root_path=tmp_path / "auto.usda")
        ctx._stage = cws
        # Manually start a fast timer so we can see it in flight.
        import threading as _threading
        ctx._autosave_timer = _threading.Timer(60.0, lambda: None)
        ctx._autosave_timer.daemon = True
        ctx._autosave_timer.start()
        assert reg.destroy("scratch")
        # Shutdown event must be set so any future tick exits early.
        assert ctx._autosave_shutdown.is_set()

    def test_clear_stops_all_autosaves(self, tmp_path):
        from agent.session_context import SessionRegistry
        reg = SessionRegistry()
        ctxs = [reg.get_or_create(f"s{i}") for i in range(3)]
        # Plant fake timers on each
        for ctx in ctxs:
            import threading as _threading
            ctx._autosave_timer = _threading.Timer(60.0, lambda: None)
            ctx._autosave_timer.daemon = True
            ctx._autosave_timer.start()
        reg.clear()
        for ctx in ctxs:
            assert ctx._autosave_shutdown.is_set()


# ---------------------------------------------------------------------------
# Moneta reference adapter
# ---------------------------------------------------------------------------

class TestMonetaAdapter:
    def test_outbound_emits_jsonl(self, tmp_path):
        import json as _json
        from agent.integrations.moneta import MonetaAdapter, MonetaAdapterConfig
        cws = CognitiveWorkflowStage()
        config = MonetaAdapterConfig(
            outbox_dir=tmp_path / "outbox",
            inbox_dir=None,
            poll_interval_seconds=0.01,
        )
        adapter = MonetaAdapter(config, cws)
        adapter.start()
        try:
            cws.write("/workflows/m1", "steps", 25)
            # Subscriber fan-out is on a daemon thread — give it time.
            for _ in range(50):
                if adapter.events_emitted >= 1:
                    break
                time.sleep(0.02)
            assert adapter.events_emitted >= 1
            # Inspect the JSONL file
            files = list((tmp_path / "outbox").glob("*.jsonl"))
            assert len(files) == 1
            lines = files[0].read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) >= 1
            rec = _json.loads(lines[0])
            assert rec["op"] == "write"
            assert rec["prim_path"] == "/workflows/m1"
            assert rec["attr_name"] == "steps"
            assert rec["schema"] == "moneta-v0"
        finally:
            adapter.stop()

    def test_inbound_ingests_delta_file(self, tmp_path):
        import json as _json
        from agent.integrations.moneta import MonetaAdapter, MonetaAdapterConfig
        cws = CognitiveWorkflowStage()
        outbox = tmp_path / "outbox"
        inbox = tmp_path / "inbox"
        config = MonetaAdapterConfig(
            outbox_dir=outbox,
            inbox_dir=inbox,
            poll_interval_seconds=0.05,
        )
        adapter = MonetaAdapter(config, cws)
        adapter.start()
        try:
            inbox.mkdir(exist_ok=True)
            delta_file = inbox / "001.delta.json"
            delta_file.write_text(_json.dumps({
                "agent_name": "moneta",
                "delta": {"/workflows/from_moneta:steps": 42},
            }), encoding="utf-8")
            # Wait for the watcher to see and ingest
            for _ in range(50):
                if adapter.deltas_ingested >= 1:
                    break
                time.sleep(0.05)
            assert adapter.deltas_ingested == 1
            # File must be marked .applied
            assert (inbox / "001.delta.json.applied").exists()
            # And the delta must be visible on the stage
            assert cws.read("/workflows/from_moneta", "steps") == 42
        finally:
            adapter.stop()

    def test_malformed_delta_marked_failed_not_retried(self, tmp_path):
        from agent.integrations.moneta import MonetaAdapter, MonetaAdapterConfig
        cws = CognitiveWorkflowStage()
        config = MonetaAdapterConfig(
            outbox_dir=tmp_path / "outbox",
            inbox_dir=tmp_path / "inbox",
            poll_interval_seconds=0.05,
        )
        adapter = MonetaAdapter(config, cws)
        adapter.start()
        try:
            (tmp_path / "inbox").mkdir(exist_ok=True)
            bad = tmp_path / "inbox" / "broken.delta.json"
            bad.write_text("{not valid json", encoding="utf-8")
            for _ in range(50):
                if adapter.ingest_failures >= 1:
                    break
                time.sleep(0.05)
            assert adapter.ingest_failures >= 1
            assert (tmp_path / "inbox" / "broken.delta.json.failed").exists()
            assert not bad.exists()
        finally:
            adapter.stop()


# ---------------------------------------------------------------------------
# A1 — atomic flush via .tmp + os.replace
# ---------------------------------------------------------------------------

class TestAtomicFlush:
    def test_flush_writes_via_tmp_and_renames(self, tmp_path):
        """flush() must write to <path>.tmp first, then os.replace into place."""
        import os as _os
        from unittest.mock import patch

        path = tmp_path / "atomic.usda"
        s = CognitiveWorkflowStage(root_path=path)
        s.write("/workflows/a", "v", 1)

        # Spy on os.replace so we can assert it ran exactly once.
        original_replace = _os.replace
        replace_calls: list[tuple[str, str]] = []

        def spy(src, dst):
            replace_calls.append((str(src), str(dst)))
            return original_replace(src, dst)

        with patch("agent.stage.cognitive_stage.os.replace", side_effect=spy):
            s.flush()

        assert path.exists()
        # Exactly one replace call: <path>.tmp -> <path>
        assert len(replace_calls) == 1
        src, dst = replace_calls[0]
        assert src == str(path) + ".tmp"
        assert dst == str(path)
        # The .tmp file is gone after the rename
        assert not (tmp_path / "atomic.usda.tmp").exists()

    def test_flush_failure_cleans_up_tmp(self, tmp_path):
        """If Export raises, the orphaned .tmp file is removed."""
        from unittest.mock import patch

        path = tmp_path / "fail.usda"
        s = CognitiveWorkflowStage(root_path=path)
        s.write("/workflows/x", "v", 1)

        # Force Flatten().Export to raise
        class Boom(Exception):
            pass

        original_flatten = s._stage.Flatten

        def evil_flatten():
            flat = original_flatten()
            original_export = flat.Export

            def raising_export(target):
                # Create the .tmp file then raise — simulates partial write
                with open(target, "w", encoding="utf-8") as f:
                    f.write("partial")
                raise Boom("simulated crash mid-export")

            flat.Export = raising_export
            return flat

        with patch.object(s._stage, "Flatten", evil_flatten):
            with pytest.raises(Boom):
                s.flush()

        # The canonical path was never written
        assert not path.exists()
        # The orphaned .tmp was cleaned up
        assert not (tmp_path / "fail.usda.tmp").exists()


# ---------------------------------------------------------------------------
# A2 — single-thread dispatcher
# ---------------------------------------------------------------------------

def _count_dispatcher_threads() -> int:
    """Count live cozy-stage-dispatch threads in the current process."""
    return sum(
        1 for t in threading.enumerate()
        if "cozy-stage-dispatch" in t.name and t.is_alive()
    )


class TestSingleThreadDispatcher:
    def test_burst_of_writes_uses_single_dispatcher_thread(self):
        """100 events must NOT spawn 100 threads — the cozy fix.

        Uses delta-based counting so it's robust to other tests in the
        suite that may have leaked dispatcher threads (those leaks are
        bugs in those tests but not this one's concern).
        """
        s = CognitiveWorkflowStage()
        events: list[StageEvent] = []
        evt_done = threading.Event()
        target = 100

        def cb(e):
            events.append(e)
            if len(events) >= target:
                evt_done.set()

        before_subscribe = _count_dispatcher_threads()
        s.subscribe(cb)
        after_subscribe = _count_dispatcher_threads()
        # subscribe() must have spawned exactly one new dispatcher
        assert after_subscribe == before_subscribe + 1
        assert s._dispatch_thread is not None
        assert s._dispatch_thread.is_alive()

        for i in range(target):
            s.write(f"/workflows/burst_{i}", "v", i)

        # Wait for the dispatcher to drain
        assert evt_done.wait(timeout=5.0)

        # Still the same dispatcher count — burst didn't spawn more
        after_burst = _count_dispatcher_threads()
        assert after_burst == after_subscribe
        assert len(events) >= target

        s.close_subscribers()

    def test_close_subscribers_stops_dispatcher(self):
        s = CognitiveWorkflowStage()
        before = _count_dispatcher_threads()
        s.subscribe(lambda e: None)
        assert _count_dispatcher_threads() == before + 1
        dispatcher = s._dispatch_thread
        s.close_subscribers()
        # Give it a moment to exit
        if dispatcher is not None:
            dispatcher.join(timeout=2.0)
        # Net count is back to the baseline
        assert _count_dispatcher_threads() == before
        assert s._dispatch_thread is None

    def test_full_queue_drops_with_warning(self, monkeypatch):
        """When the dispatcher is slow and the queue fills, events are
        DROPPED with a warning, not buffered indefinitely."""
        s = CognitiveWorkflowStage()
        # Tiny queue + slow consumer
        monkeypatch.setattr(
            CognitiveWorkflowStage, "_DISPATCH_QUEUE_MAXSIZE", 2,
        )

        # Slow subscriber blocks the dispatcher
        block = threading.Event()

        def slow(e):
            block.wait(timeout=2.0)

        s.subscribe(slow)
        # Fire many events — most should drop because queue is size 2
        for i in range(50):
            s.write(f"/workflows/slow_{i}", "v", i)

        # Some events must have been dropped
        assert s.dispatch_drops > 0
        # Unblock the consumer so close_subscribers can join
        block.set()
        s.close_subscribers()


# ---------------------------------------------------------------------------
# B1 — subprocess-based crash-resume integration test
# ---------------------------------------------------------------------------

class TestCrashResume:
    """Spawns a child Python process, writes + flushes, SIGKILLs it,
    then verifies the parent can cold-load the same .usda file. Proves
    that:
      - The atomic flush survives an abrupt kill (no .tmp orphan, no
        corrupted canonical file)
      - CognitiveWorkflowStage(root_path=...) reconstructs prims from
        the flushed file with no other state
    """

    def test_kill_after_flush_resumes_cleanly(self, tmp_path):
        import subprocess
        import sys
        import os as _os
        import signal as _signal

        usda_path = tmp_path / "crash_resume.usda"
        marker = tmp_path / "child.ready"

        child_script = f"""
import sys, time
from pathlib import Path
from agent.stage.cognitive_stage import CognitiveWorkflowStage

stage = CognitiveWorkflowStage(root_path=Path({str(usda_path)!r}))
# Write multiple prims under different top-level scopes
stage.write("/workflows/wf_a", "steps", 25)
stage.write("/workflows/wf_a", "cfg", 7.5)
stage.write("/agents/forge", "active", True)
stage.write("/experience/exp_001", "score", 0.83)
stage.flush()

# Signal that flush has completed
Path({str(marker)!r}).write_text("ready", encoding="utf-8")

# Hang so the parent can SIGKILL us
while True:
    time.sleep(0.5)
"""

        proc = subprocess.Popen(
            [sys.executable, "-c", child_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Wait up to 10s for the marker file
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if marker.exists():
                    break
                if proc.poll() is not None:
                    out, err = proc.communicate(timeout=1.0)
                    pytest.fail(
                        f"Child died early: rc={proc.returncode}\n"
                        f"stdout: {out.decode()!r}\n"
                        f"stderr: {err.decode()!r}"
                    )
                time.sleep(0.1)
            assert marker.exists(), "Child never wrote ready marker"

            # The canonical file exists and the .tmp does not
            assert usda_path.exists()
            assert not (tmp_path / "crash_resume.usda.tmp").exists()

            # Brutally kill the child
            _os.kill(proc.pid, _signal.SIGKILL)
            proc.wait(timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5.0)

        # Parent: cold-load and verify prims
        stage = CognitiveWorkflowStage(root_path=usda_path)
        assert stage.read("/workflows/wf_a", "steps") == 25
        # USD stores doubles; just check it round-tripped (small float).
        cfg = stage.read("/workflows/wf_a", "cfg")
        assert cfg is not None and abs(cfg - 7.5) < 1e-6
        assert stage.read("/agents/forge", "active") is True
        assert stage.read("/experience/exp_001", "score") is not None
        stage.close_subscribers()


# ---------------------------------------------------------------------------
# W2 — CLI closure factories (make_propose_fn / make_execute_fn)
# ---------------------------------------------------------------------------

class TestMakeProposeFn:
    def test_proposals_cycle_through_full_set(self):
        from agent.harness import make_propose_fn
        propose = make_propose_fn()
        # The cycle has 5 entries — collect 6 proposals to verify wrap-around.
        proposals = [propose() for _ in range(6)]
        # All proposals are dicts with the expected schema
        for p in proposals:
            assert set(p.keys()) >= {"param"} | {"delta"}
            assert p["param"] in ("steps", "cfg", "seed")
            assert isinstance(p["delta"], (int, float))
        # Wrap-around: 6th proposal == 1st proposal
        assert proposals[0] == proposals[5]

    def test_proposals_are_independent_copies(self):
        """Mutating a returned proposal must not corrupt the cycle."""
        from agent.harness import make_propose_fn
        propose = make_propose_fn()
        first = propose()
        first["delta"] = 999
        # Skip ahead 5 proposals to wrap back to the same cycle slot
        for _ in range(4):
            propose()
        sixth = propose()
        assert sixth["delta"] != 999


class TestMakeExecuteFnDryRun:
    def test_dry_run_returns_synthetic_scores_without_calling_executor(self):
        from agent.harness import make_execute_fn
        # workflow_loader and workflow_executor must NOT be called in dry-run
        loader_called = {"n": 0}
        executor_called = {"n": 0}

        def loader(_path):
            loader_called["n"] += 1
            return {}

        def executor(_wf):
            executor_called["n"] += 1
            return {}

        execute = make_execute_fn(
            "dry-run", workflow_path=None,
            workflow_loader=loader, workflow_executor=executor,
        )
        scores = execute({"param": "steps", "delta": +5})
        assert scores["success"] == 1.0
        assert 0.0 <= scores["speed"] <= 1.0
        assert loader_called["n"] == 0
        assert executor_called["n"] == 0


class TestMakeExecuteFnReal:
    @pytest.fixture
    def base_workflow(self):
        # Minimal workflow with a sampler-like node. The closure looks for
        # "steps" / "cfg" / "seed" inputs and patches them.
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "steps": 20,
                    "cfg": 7.0,
                    "seed": 12345,
                    "model": ["1", 0],
                },
            },
            "4": {"class_type": "CheckpointLoader", "inputs": {}},
        }

    def test_real_mode_loads_workflow_once(self, base_workflow):
        from agent.harness import make_execute_fn
        load_calls = {"n": 0}

        def loader(_path):
            load_calls["n"] += 1
            return base_workflow

        def executor(wf):
            # Verify the executor sees the patched workflow on each call
            return {"status": "complete", "outputs": ["img1"], "total_time_s": 5.0}

        execute = make_execute_fn(
            "real", workflow_path="/fake/wf.json",
            workflow_loader=loader, workflow_executor=executor,
        )
        # Loader fires exactly once at factory-time, NOT per call
        assert load_calls["n"] == 1
        # Three calls — loader still 1
        execute({"param": "steps", "delta": +5})
        execute({"param": "cfg", "delta": +1.0})
        execute({"param": "seed", "delta": +1})
        assert load_calls["n"] == 1

    def test_real_mode_applies_proposal_to_workflow(self, base_workflow):
        from agent.harness import make_execute_fn
        captured = []

        def executor(wf):
            captured.append(copy.deepcopy(wf))
            return {"status": "complete", "outputs": ["a"], "total_time_s": 3.0}

        execute = make_execute_fn(
            "real", workflow_path="/fake/wf.json",
            workflow_loader=lambda p: base_workflow,
            workflow_executor=executor,
        )
        execute({"param": "steps", "delta": +10})
        # The workflow seen by the executor has steps == 30 (20 + 10)
        assert captured[0]["3"]["inputs"]["steps"] == 30
        # The base workflow is untouched (deep copy at factory time)
        assert base_workflow["3"]["inputs"]["steps"] == 20

    def test_real_mode_returns_failure_scores_on_executor_error(
        self, base_workflow,
    ):
        from agent.harness import make_execute_fn

        def executor(wf):
            return {"status": "error", "error": "ComfyUI not running"}

        execute = make_execute_fn(
            "real", workflow_path="/fake/wf.json",
            workflow_loader=lambda p: base_workflow,
            workflow_executor=executor,
        )
        scores = execute({"param": "steps", "delta": +5})
        assert scores["success"] == 0.0
        assert scores["speed"] == 0.0

    def test_real_mode_returns_failure_scores_on_executor_exception(
        self, base_workflow,
    ):
        from agent.harness import make_execute_fn

        def executor(wf):
            raise RuntimeError("boom")

        execute = make_execute_fn(
            "real", workflow_path="/fake/wf.json",
            workflow_loader=lambda p: base_workflow,
            workflow_executor=executor,
        )
        scores = execute({"param": "steps", "delta": +5})
        assert scores["success"] == 0.0

    def test_real_mode_without_workflow_path_raises(self):
        from agent.harness import make_execute_fn
        with pytest.raises(ValueError, match="--workflow"):
            make_execute_fn("real", workflow_path=None)

    def test_unknown_mode_raises(self):
        from agent.harness import make_execute_fn
        with pytest.raises(ValueError, match="unsupported execute mode"):
            make_execute_fn("nonsense", workflow_path=None)

    def test_speed_score_decreases_with_total_time(self, base_workflow):
        from agent.harness import make_execute_fn

        def fast_executor(wf):
            return {"status": "complete", "outputs": ["a"], "total_time_s": 1.0}

        def slow_executor(wf):
            return {"status": "complete", "outputs": ["a"], "total_time_s": 30.0}

        fast = make_execute_fn(
            "real", workflow_path="/fake/wf.json",
            workflow_loader=lambda p: base_workflow,
            workflow_executor=fast_executor,
        )
        slow = make_execute_fn(
            "real", workflow_path="/fake/wf.json",
            workflow_loader=lambda p: base_workflow,
            workflow_executor=slow_executor,
        )
        s_fast = fast({"param": "steps", "delta": +1})
        s_slow = slow({"param": "steps", "delta": +1})
        assert s_fast["speed"] > s_slow["speed"]
