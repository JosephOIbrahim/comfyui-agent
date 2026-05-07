# Incident — 2026-05-02 — Pre-existing test failure on master, xfail recommended for inside-out pass

**Phase:** SCOUT (pre-flight)
**Sequence:** 1
**Date:** 2026-05-02
**Branch:** architecture/inside-out-pass
**Approved by:** Joe (Option 1: PROCEED-WITH-XFAIL — see "Status" below for landing state)

## Resolved

**Resolution date:** 2026-05-07
**Fixing commit:** `bf5d0c3d47e3ba61f4a470fcbe61667af1d092a3`
**Summary:** Added `vset.GetVariantNames()` pre-check in `agent/stage/cognitive_stage.py:select_profile` — `test_invalid_variant_name_raises_stage_error` now passes; baseline restored to 2717 passing.

## What this incident covers

The pytest baseline check in `RUN_INSIDE_OUT_PASS.md` Section 10 surfaced a single failing test that pre-exists on `master`:

```
FAILED tests/test_cognitive_stage.py::TestSelectProfileExceptionWrapping::test_invalid_variant_name_raises_stage_error
  tests/test_cognitive_stage.py:533
  Expected: select_profile(prim, set, "nonexistent_variant_xyz") raises StageError
  Actual:   call returns silently — DID NOT RAISE
```

## Provenance

- Test introduced in `0e0e70e [VERIFY] Cycle 64: ... USD exception wrapping`
- Implementation `agent/stage/cognitive_stage.py:select_profile` (lines 320–343) introduced in same commit
- Subsequent touches: `790e97f` (lint-only)
- The failure exists on `master` — not introduced by `architecture/inside-out-pass` (which is one docs-only commit ahead)

## Root cause (read-only diagnosis, not fixed)

`select_profile` wraps the call to `pxr.Usd.VariantSet.SetVariantSelection` in `try/except` expecting `pxr` to raise on an invalid variant name. **USD does not raise.** `SetVariantSelection` writes the selection metadata as an opaque string regardless of whether the variant was registered via `AddVariant`. The `try/except` therefore never fires, and the test (which expects `StageError`) sees no exception.

The fix would be a pre-check before calling `SetVariantSelection`:

```python
vset = vsets.GetVariantSet(variant_set)
if profile_name not in vset.GetVariantNames():
    raise StageError(
        f"Could not select variant '{profile_name}' on {prim_path}/{variant_set}: "
        f"not in {vset.GetVariantNames()}"
    )
vset.SetVariantSelection(profile_name)
```

## Why this is xfail'd, not fixed

Constitution Rule 5 (role isolation): SCOUT does not mutate implementation code.
Constitution Rule 4 (no half-finished work): the fix above is small but it is implementation work, not scout work. Scope creep.
The bug is unrelated to the inside-out architecture migration.

## Status — recommended remediation, NOT YET APPLIED

The xfail decorator below was approved as the remediation but **has not been landed in the test code as of 2026-05-02**. `tests/test_cognitive_stage.py:533` currently has no `@pytest.mark.xfail` decorator, so `pytest` returns `2716 passed, 1 failed` until either the decorator or the impl fix lands.

The pre-flight assertions in `RUN_INSIDE_OUT_PASS.md` and `INSIDE_OUT_RUN_PLAYBOOK.md` are stated against the carve-out baseline (2716 passing + 1 known pre-existing fail), which is satisfiable in the current state without any test/impl mutation.

When/if the xfail decorator is landed (separate branch, separate PR), the recommended form is:

```python
@pytest.mark.xfail(
    reason="Pre-existing on master; pxr.SetVariantSelection does not raise on invalid name. "
           "See docs/incidents/incident_2026-05-02_unauthorized_xfail_during_preflight.md. "
           "Fix is out of scope for inside-out pass.",
    strict=False,
)
```

`strict=False` so a future impl fix producing an `XPASS` warns rather than turning the suite red.

## Recovery procedure

When the impl is fixed (separate branch, separate PR, post-inside-out-pass):

1. Apply the validation pre-check shown above to `agent/stage/cognitive_stage.py:select_profile`
2. Remove the `@pytest.mark.xfail` decorator from the test (if it was landed in the meantime; the carve-out baseline in the runbook does not require it)
3. Verify `pytest tests/test_cognitive_stage.py` is fully green
4. Update the baseline assertions in `RUN_INSIDE_OUT_PASS.md` and `INSIDE_OUT_RUN_PLAYBOOK.md` from "2716 passing + 1 known fail" back to a single passing count (now 2717)
5. Update or close this incident report (mark resolved with the fixing commit hash; do not delete unless all references in the runbook + playbook also drop)

## What I am NOT doing

- Not fixing the impl bug (out of scope)
- Not fixing the second failure observed in partial output at ~37% — re-escalating to Joe if the post-xfail pytest run confirms it as a separate unrelated failure
- Not silencing other failures
- Not skipping the test (xfail still runs the test, just doesn't fail the suite on it)
