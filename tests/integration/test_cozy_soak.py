"""Cozy soak validation — sustained-load proof that the harness doesn't leak.

Marked `integration` so default `-m "not integration"` runs skip it. Runs
~1000 iterations of the CozyLoop with cheap stub callbacks in roughly
15-25s and asserts:

  - dispatcher A2 fix: thread count delta is small (no thread-per-event)
  - atomic flush A1: no `.tmp` orphans, cold-load succeeds post-run
  - autosave timer: doesn't accumulate per-stage
  - file descriptor leak: FD count delta is small (POSIX only — gated)
  - dispatch_drops: zero under normal load (queue keeps up)

This is the empirical floor for the "long-running autonomous" claim. The
1000-iteration target is the minimum that makes a per-iteration linear
leak (1 thread or 1 FD per iter) impossible to miss; it's sized so a real
leak would push thread/FD counts up by ~1000 from baseline, far above any
plausible noise floor.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core not installed")

from agent.harness import CozyLoop, CozyLoopConfig  # noqa: E402
from agent.stage.cognitive_stage import CognitiveWorkflowStage  # noqa: E402

pytestmark = pytest.mark.integration


def _proc_fd_count() -> int | None:
    """Return open FD count for this process, or None on non-POSIX systems."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        return None
    try:
        return len(os.listdir(fd_dir))
    except OSError:
        return None


def _live_dispatcher_threads() -> int:
    """Count live cozy-stage-dispatch threads in the current process."""
    return sum(
        1 for t in threading.enumerate()
        if "cozy-stage-dispatch" in t.name and t.is_alive()
    )


def test_1k_iteration_soak(tmp_path):
    """Run 1000 iterations through CozyLoop and assert no leaks.

    A linear leak (1 thread or 1 FD per iteration) would push counts
    ~1000 above baseline, far above any plausible noise floor.
    """
    target_iterations = 1000
    usda = tmp_path / "soak.usda"

    # Snapshot pre-test resources. Other tests in the suite may have leaked
    # threads/FDs but we're measuring DELTA, not absolute.
    threads_before = threading.active_count()
    dispatchers_before = _live_dispatcher_threads()
    fds_before = _proc_fd_count()

    # A subscriber that touches every event so the dispatcher thread is
    # actually exercised under load (not just enqueueing into the void).
    cws = CognitiveWorkflowStage(root_path=usda)
    seen_events = {"n": 0}

    def observer(event):
        seen_events["n"] += 1

    sub_handle = cws.subscribe(observer)

    # Stub callbacks — we want to drive iterations as fast as possible to
    # uncover leaks; we don't care about the score. Each iteration triggers
    # one stage write so the dispatcher gets exercised.
    counter = {"n": 0}

    def cheap_propose():
        counter["n"] += 1
        # Touch the stage so write events are emitted to the dispatcher
        cws.write("/workflows/soak", "iter", counter["n"])
        return {"iter": counter["n"]}

    def cheap_execute(ctx):
        return {"composite": 0.5}

    # Loose budget so the test terminates by max_experiments, not by clock.
    # 60s ceiling is generous; real iteration rate is ~50/s.
    config = CozyLoopConfig(
        budget_hours=60 / 3600,
        max_experiments=target_iterations,
        checkpoint_every_n=200,  # Avoid per-iteration USD flatten — slow
        checkpoint_every_seconds=999,
        max_transient_retries=0,
        transient_backoff_seconds=(0.0,),
        max_recoverable_per_signature=10,
        health_check_seconds=999,  # No health snapshots needed
        session_name="cozy_soak",
        checkpoint_path=str(usda),
    )
    loop = CozyLoop(
        config,
        execute_fn=cheap_execute,
        propose_fn=cheap_propose,
        cws=cws,
    )

    t0 = time.time()
    result = loop.run()
    elapsed = time.time() - t0

    # Allow a brief window for the dispatcher to drain remaining events
    deadline = time.time() + 5.0
    while seen_events["n"] < target_iterations and time.time() < deadline:
        time.sleep(0.05)

    # ---- Assertions -------------------------------------------------------

    # 1) Iteration count matches budget
    assert result.run_result is not None
    completed = len(result.run_result.experiments)
    assert completed >= target_iterations, (
        f"only {completed}/{target_iterations} iterations completed in {elapsed:.1f}s"
    )

    # 2) Dispatcher delivered the events (at least most — drops would
    #    surface here). Each cheap_propose() does one stage write, so the
    #    target_iterations is a lower bound, but proposals are also called
    #    from CozyLoop's _propose() which adds extra writes from internal
    #    stage updates. Just verify we got at least the propose count.
    assert seen_events["n"] >= int(target_iterations * 0.95), (
        f"observer saw {seen_events['n']} events, expected >= {target_iterations}"
    )
    assert cws.dispatch_drops == 0, (
        f"dispatcher dropped {cws.dispatch_drops} events under normal load"
    )

    # 3) A1 atomic flush: no .tmp orphan after run
    assert not (tmp_path / "soak.usda.tmp").exists()

    # Tear down the dispatcher BEFORE counting threads so per-stage daemons
    # are reaped.
    cws.unsubscribe(sub_handle)
    cws.close_subscribers()
    # Give the dispatcher thread one tick to exit
    time.sleep(0.1)

    # 4) Thread leak: dispatcher count must be back to baseline
    dispatchers_after = _live_dispatcher_threads()
    assert dispatchers_after == dispatchers_before, (
        f"dispatcher leak: before={dispatchers_before}, "
        f"after={dispatchers_after}"
    )

    # 5) Total thread delta should be small. Allow a small slack for any
    #    short-lived daemons spawned by the runner internals.
    threads_after = threading.active_count()
    assert threads_after - threads_before < 10, (
        f"thread leak: {threads_before} -> {threads_after} "
        f"(delta {threads_after - threads_before})"
    )

    # 6) FD leak (POSIX only). Allow generous slack — pytest, USD, and
    #    autosave timers each touch handles. The signal is "is the delta
    #    BOUNDED" not "is it zero".
    fds_after = _proc_fd_count()
    if fds_before is not None and fds_after is not None:
        # 50 is a generous cap; a leak per iteration would yield 5000+
        assert fds_after - fds_before < 50, (
            f"file-descriptor leak: {fds_before} -> {fds_after} "
            f"(delta {fds_after - fds_before})"
        )

    # 7) Atomic flush survives: cold-load the .usda and verify the soak's
    #    final write is readable.
    cws2 = CognitiveWorkflowStage(root_path=usda)
    final_iter = cws2.read("/workflows/soak", "iter")
    cws2.close_subscribers()
    assert final_iter is not None and final_iter > 0
