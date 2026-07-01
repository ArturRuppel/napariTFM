# Parallel Batch Progressive Stage Fill — Design

**Date:** 2026-07-01
**Status:** Approved (design); ready for implementation planning

## Problem

During a parallel Run-all (`num_workers > 1`), every enabled stage dot on a
row's mini-rail (`MiniRail` in `_experiments_list.py`) flips to a flat amber
`"running"` the instant that folder's worker is submitted
(`ExperimentsList.mark_running()`, driven by `_on_batch_progress` in
`_widget.py`), and stays that flat colour for the worker's *entire* runtime —
often several minutes. With several workers running concurrently, several rows
show this identical undifferentiated look at once, and it's indistinguishable
from a hung run.

This is a known, documented trade-off (see `_run_all_experiments_parallel`'s
docstring: "the per-stage progress rail simply does not update live during a
parallel run"), because a `ProcessPoolExecutor` worker
(`_run_position_headless`) runs in a separate spawned process with no sink and
reports nothing back to the parent until it fully finishes.

Contrast with the sequential path: the single-experiment detail panel's
`StageSpine` widget already renders a growing pie-wedge fill
(`StageSpine.set_progress(fraction)`), fed in real time by
`ViewerSink`'s `on_stage_progress(stage, status, fraction)` callback, which
itself is driven by `PipelineSink` hooks (`stage_started` /  `stage_frame` /
`stage_finished`) that `process_folder` already calls unconditionally at every
stage boundary — sequential mode just happens to run in-process, so a Qt
signal can reach the widget directly.

Goal: give parallel-run rows the same real, progressive, per-stage (and
per-frame) fill, instead of a flat "running" glow for the whole worker
lifetime.

## Approach

Since `process_folder` already emits `stage_started` / `stage_frame` /
`stage_finished` unconditionally via `self._emit(...)` (a no-op when
`sink is None`), no changes are needed to the compute path itself. We only need
to get those existing hooks across the process boundary and route them to the
correct row's correct dot.

### 1. IPC mechanism: a plain `multiprocessing.Queue`

Three options were considered:

1. **Plain `multiprocessing.Queue`** (chosen). Created once per parallel run,
   using the same spawn `mp_context` already used for the
   `ProcessPoolExecutor`, and passed as an extra argument into every submitted
   `_run_position_headless` call. Multiple worker processes `put()`
   concurrently — exactly its designed usage. The parent drains it
   non-blockingly (`get_nowait()` loop) from the existing 150ms poll timer.
2. **`multiprocessing.Manager().Queue()`** — a proxy queue backed by an extra
   manager server process. Same semantics as #1 with an added IPC hop per
   message, for no capability actually needed (nothing here requires sharing
   the queue with objects created after the workers are spawned).
3. **Shared-memory counters** (`multiprocessing.Array` of per-slot floats) —
   avoids message passing, but `ProcessPoolExecutor` doesn't expose a stable
   "which physical worker owns which folder right now" mapping (futures are
   reassigned to free workers dynamically as slots free up), so reliably
   tagging a shared-memory slot with the *current* folder/stage would need
   more bookkeeping than a self-describing message queue, for no real benefit
   at this message volume.

No throttling: every `stage_frame` call enqueues one small tuple, mirroring
`ViewerSink` exactly. The messages are tiny and draining is O(queue depth) per
150ms tick; real per-frame TFM compute cost vastly dominates the IPC cost.

### 2. `QueueProgressSink`

A new small `PipelineSink` subclass (backend-side, no Qt/napari import, same
constraint `PipelineSink` already respects) wraps the queue plus this worker's
`folder` path. It reuses the exact fraction math `ViewerSink` already uses:

- `stage_started(stage, num_frames, info)` → enqueue
  `(folder, stage, "running", 0.0)`
- `stage_frame(stage, frame_index, frame)` → enqueue
  `(folder, stage, "running", (frame_index + 1) / max(num_frames, 1))`
- `stage_finished(stage, result)` → enqueue `(folder, stage, "done", None)`

`_run_position_headless` constructs one `QueueProgressSink(queue, folder)` per
worker invocation and passes it as `BatchAnalysis(config, sink=...)` — the
only change to that function is threading the queue through and constructing
the sink instead of passing `sink=None`.

### 3. Draining and routing

- `start_parallel` creates the queue and stores it alongside
  `self._executor` / `self._pending_futures`.
- `poll_parallel_progress` additionally drains the queue fully via a
  `get_nowait()` loop on every call (in addition to its existing
  `futures_wait(timeout=0)` check for folder completion), and returns these as
  a new `stage_events: list[tuple[str, str, str, Optional[float]]]` alongside
  today's `(folder, status)` `events` list.
- `_widget.py`'s `_run_all_experiments_parallel` poll callback forwards each
  stage event to a new `_on_batch_stage_progress(folder, stage, status,
  fraction)` handler, mirroring the existing `_on_run_all_stage_progress` used
  by the sequential path.
- `_on_batch_stage_progress` calls a new `ExperimentsList.set_row_stage_progress(
  path, stage, status, fraction)`, which updates *only that one row's one
  stage dot* — reusing the existing per-stage-status plumbing
  (`ExperimentRow.set_stage_statuses`) for the status, plus a new fractional
  field on `MiniRail` for the dot's paint.

### 4. Submission-time and error semantics

- `mark_running()` still fires at submission time as today's transient
  placeholder (a queued-but-not-yet-scheduled worker still reads as "running"
  briefly). The first real `stage_started` event for that folder overwrites it
  moments later with the true first-stage state.
- No new error-forwarding path. On worker failure, the existing
  `apply_row_statuses(folder, self._experiment_stage_status(folder))` call
  (already wired to the `"error"` branch of `_on_batch_progress`) reads real
  on-disk per-stage truth once the worker returns, so the failed stage settles
  to a true `"error"` there. Only the *in-flight* view up to that point
  changes — real per-stage progress instead of flat amber — not the
  end-of-run reconciliation.

### 5. Rendering: fractional fill in `MiniRail`

`MiniRail` gains the same fractional pie-wedge logic `StageSpine` already has:

- A per-stage `_progress: dict[str, Optional[float]]`, defaulting to `None`.
- `set_stage_progress(stage, fraction)` sets it; cleared back to `None`
  whenever that stage's status changes away from `"running"` (mirrors
  `StageSpine.set_status`'s stale-progress guard, so a dot never flashes a
  stale partial wedge on its next run).
- `paintEvent` draws a pie slice (same geometry as `StageSpine`: a ring plus a
  clockwise-growing wedge from 12 o'clock) instead of a flat fill when
  `status == "running" and progress is not None`, at `MiniRail`'s existing
  `DOT_R = 4` radius.

The 4px radius is notably smaller than `StageSpine`'s 6px node; whether a pie
wedge reads clearly at that size (versus just noise) will be checked visually
via the running app once built, with radius/stroke adjusted if needed. This is
an implementation-time tuning detail, not a scope change.

## Testing

- Extend `_FakeAnalyzer` in `tests/test_workflow_shell.py` to also emit canned
  stage-progress events (not just folder-level done/error), matching the new
  `stage_events` return value of `poll_parallel_progress`.
- Add assertions that a row's individual stage dot updates mid-run, distinctly
  from its sibling dots on the same row and from other rows.
- Add a `MiniRail` paint/`appearance()`-level test for the fractional-fill
  state (progress set, then cleared on status change away from `"running"`).
- Existing tests asserting flat `mark_running()` behaviour at submission time
  should continue to pass unchanged — that call site isn't removed, only
  superseded moments later by real events.

## Out of scope

- No changes to the sequential Run-all path or `StageSpine` itself — this is
  additive to the parallel path only.
- No throttling of the progress queue (see rationale above).
- No new error-signalling path from inside a worker — existing end-of-run
  disk-truth reconciliation is retained as-is.
