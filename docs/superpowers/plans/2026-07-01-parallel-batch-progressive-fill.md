# Parallel Batch Progressive Stage Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give parallel Run-all rows the same real, progressive, per-stage (and per-frame) mini-rail fill that the single-experiment detail panel's `StageSpine` already shows during a sequential run, instead of a flat amber "running" dot for a worker's entire lifetime.

**Architecture:** Each parallel worker process attaches a new `QueueProgressSink` (a `PipelineSink` subclass) instead of no sink at all; it puts `(folder, stage, status, fraction)` tuples onto a `multiprocessing.Queue` shared by every worker in the run. The GUI's existing 150ms poll timer drains that queue every tick (`BatchAnalysis.poll_parallel_progress`) and routes each tuple to a new `ExperimentsList.set_row_stage_progress`, which updates one row's one stage dot. `MiniRail` gains the same fractional pie-wedge paint `StageSpine` already has, so the dot itself renders the fill.

**Tech Stack:** Python, `multiprocessing.Queue` (spawn context), Qt (`qtpy`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-parallel-batch-progressive-fill-design.md` (approved).

---

## Task 1: `QueueProgressSink` — cross-process progress reporting

**Files:**
- Create: `napariTFM/backend/queue_progress_sink.py`
- Test: `tests/test_queue_progress_sink.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_queue_progress_sink.py`:

```python
"""QueueProgressSink: the process-boundary counterpart to ViewerSink.

A parallel Run-all worker (backend/batch_analysis.py's _run_position_headless)
attaches one of these instead of a ViewerSink, so its stage/frame lifecycle
hooks reach the parent process via a plain queue instead of a Qt signal.
"""

import queue

from napariTFM.backend.pipeline_sink import PipelineSink
from napariTFM.backend.queue_progress_sink import QueueProgressSink


def test_stage_started_enqueues_running_at_zero():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")

    sink.stage_started("displacement", 4)

    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 0.0)


def test_stage_frame_enqueues_growing_fraction():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")
    sink.stage_started("displacement", 4)
    q.get_nowait()  # discard the stage_started message

    sink.stage_frame("displacement", 0, None)
    sink.stage_frame("displacement", 1, None)
    sink.stage_frame("displacement", 3, None)

    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 0.25)
    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 0.5)
    assert q.get_nowait() == ("/data/pos_00", "displacement", "running", 1.0)


def test_stage_frame_falls_back_to_one_frame_when_num_frames_is_zero():
    """Mirrors ViewerSink's max(self._stage_num_frames, 1) guard: a
    zero-frame stage still reports a defined fraction instead of dividing by
    zero."""
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")
    sink.stage_started("stress", 0)
    q.get_nowait()

    sink.stage_frame("stress", 0, None)

    assert q.get_nowait() == ("/data/pos_00", "stress", "running", 1.0)


def test_stage_finished_enqueues_done_with_no_fraction():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_00")

    sink.stage_finished("force", object())

    assert q.get_nowait() == ("/data/pos_00", "force", "done", None)


def test_folder_is_stamped_on_every_message():
    q = queue.Queue()
    sink = QueueProgressSink(q, "/data/pos_07")

    sink.stage_started("preprocessing", 1)

    folder, *_ = q.get_nowait()
    assert folder == "/data/pos_07"


def test_is_a_pipeline_sink():
    assert issubclass(QueueProgressSink, PipelineSink)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_queue_progress_sink.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'napariTFM.backend.queue_progress_sink'`

- [ ] **Step 3: Write the implementation**

Create `napariTFM/backend/queue_progress_sink.py`:

```python
"""A PipelineSink that reports stage/frame progress across a process boundary.

Used by parallel Run-all workers (see ``batch_analysis._run_position_headless``):
each worker process attaches one ``QueueProgressSink`` wrapping a
``multiprocessing.Queue`` shared with the parent process, so the parent's
150ms poll timer (``BatchAnalysis.poll_parallel_progress``) can drain real
per-stage, per-frame progress instead of only learning a folder's terminal
``done``/``error`` status when its worker fully returns.

Mirrors ``ViewerSink``'s fraction math exactly (see
``napariTFM.utilities.viewer_sink.ViewerSink``) so both sinks compute "how far
into this stage are we" identically -- one delivers it via a Qt signal
in-process, this one via a multiprocessing queue.
"""

from typing import Any, Optional

from napariTFM.backend.pipeline_sink import PipelineSink


class QueueProgressSink(PipelineSink):
    """Puts ``(folder, stage, status, fraction)`` tuples onto a shared queue.

    Parameters
    ----------
    queue
        A ``multiprocessing.Queue`` created by the parent process (spawn
        context matching the ``ProcessPoolExecutor``) and passed into this
        worker's task. Multiple worker processes ``put()`` onto it
        concurrently; that is the queue's designed usage.
    folder
        This worker's experiment folder path, stamped onto every message so
        the parent can route it to the right row.
    """

    def __init__(self, queue, folder: str):
        self._queue = queue
        self._folder = folder
        self._stage_num_frames = 0

    def stage_started(self, stage: str, num_frames: int, info: Optional[dict] = None) -> None:
        self._stage_num_frames = num_frames
        self._queue.put((self._folder, stage, "running", 0.0))

    def stage_frame(self, stage: str, frame_index: int, frame: Any) -> None:
        fraction = (frame_index + 1) / max(self._stage_num_frames, 1)
        self._queue.put((self._folder, stage, "running", fraction))

    def stage_finished(self, stage: str, result: Any) -> None:
        self._queue.put((self._folder, stage, "done", None))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_queue_progress_sink.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add napariTFM/backend/queue_progress_sink.py tests/test_queue_progress_sink.py
git commit -m "Add QueueProgressSink: cross-process stage progress reporting"
```

---

## Task 2: Thread the progress queue through `batch_analysis.py`

**Files:**
- Modify: `napariTFM/backend/batch_analysis.py`
- Modify: `tests/test_batch_parallel.py`

- [ ] **Step 1: Update `tests/test_batch_parallel.py` to the new signatures (red)**

Add these imports at the top of `tests/test_batch_parallel.py` (after the existing `from napariTFM.utilities.batch_output import resolve_output_plan` line):

```python
import queue as queue_module

from napariTFM.backend.queue_progress_sink import QueueProgressSink
```

Replace `_stub_run_position_headless`:

```python
def _stub_run_position_headless(monkeypatch, status="done", err=None):
    """Replace the real (heavy, sleep(2)-banner-bearing) pipeline with an
    instant stand-in for orchestration tests, which care about
    future/event bookkeeping, not pipeline behaviour -- that's covered by
    the dedicated ``_run_position_headless`` unit tests above.
    """
    monkeypatch.setattr(
        ba, "_run_position_headless",
        lambda config, folder, output_dir, queue: (folder, status, err),
    )
```

Replace the three `_run_position_headless` direct-call tests:

```python
def test_run_position_headless_returns_done_on_success(tmp_path):
    config = _minimal_config()
    folder = str(tmp_path / "input")
    output_dir = str(tmp_path / "TFM_data" / "input")

    result = _run_position_headless(config, folder, output_dir, queue_module.Queue())

    assert result == (folder, "done", None)


def test_run_position_headless_returns_error_with_message_on_failure(tmp_path):
    # Malformed config: 'analysis_steps' is missing entirely, so
    # process_folder raises a bare KeyError out of
    # _handle_preprocessing_execution -- a real failure mode, not an injected
    # mock -- which _run_position_headless must catch and report rather than
    # propagate (a single bad position must not take the worker down).
    config = {"visualizations": {}, "parameters": {}, "input_files": {}}
    folder = str(tmp_path / "input")
    output_dir = str(tmp_path / "TFM_data" / "input")

    folder_out, status, err = _run_position_headless(
        config, folder, output_dir, queue_module.Queue()
    )

    assert folder_out == folder
    assert status == "error"
    assert err is not None and "analysis_steps" in err


def test_run_position_headless_constructs_queue_progress_sink(monkeypatch, tmp_path):
    captured = {}

    def _fake_process_folder(self, folder, output_dir=None):
        captured["sink"] = self._sink
        captured["progress_callback"] = self._progress_callback

    monkeypatch.setattr(BatchAnalysis, "process_folder", _fake_process_folder)

    folder = str(tmp_path / "input")
    output_dir = str(tmp_path / "TFM_data" / "input")
    q = queue_module.Queue()
    result = _run_position_headless(_minimal_config(), folder, output_dir, q)

    assert result == (folder, "done", None)
    assert isinstance(captured["sink"], QueueProgressSink)
    assert captured["sink"]._queue is q
    assert captured["sink"]._folder == folder
    assert captured["progress_callback"] is None
```

Replace `test_start_parallel_passes_config_and_output_dir_to_worker` and add a new test right after it:

```python
def test_start_parallel_passes_config_and_output_dir_to_worker(tmp_path, monkeypatch):
    a = str(tmp_path / "a")
    analysis = _analysis([a])
    _stub_run_position_headless(monkeypatch)
    fake = _FakeExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: fake)
    plan = _plan_for([a], tmp_path)

    analysis.start_parallel(plan, num_workers=1)

    (fn_config, fn_folder, fn_output_dir, fn_queue), = [args for args in fake.submitted_args]
    assert fn_config is analysis.config
    assert fn_folder == a
    assert fn_output_dir == str(plan.output_dirs[a])
    assert fn_queue is analysis._progress_queue


def test_start_parallel_shares_one_progress_queue_across_all_workers(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    analysis = _analysis([a, b])
    _stub_run_position_headless(monkeypatch)
    fake = _FakeExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: fake)
    plan = _plan_for([a, b], tmp_path)

    analysis.start_parallel(plan, num_workers=2)

    queues_passed = [args[3] for args in fake.submitted_args]
    assert queues_passed[0] is analysis._progress_queue
    assert queues_passed[1] is analysis._progress_queue
```

Update every `poll_parallel_progress()` call site in this file to unpack the new 3-tuple, and add one new drain test. The five existing calls become:

```python
def test_poll_parallel_progress_reports_done_and_finishes(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    events = []
    analysis = _analysis([a, b])
    analysis._progress_callback = lambda folder, status: events.append((folder, status))

    _stub_run_position_headless(monkeypatch)
    fake = _FakeExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: fake)
    plan = _plan_for([a, b], tmp_path)
    analysis.start_parallel(plan, num_workers=2)

    new_events, new_stage_events, finished = analysis.poll_parallel_progress()

    assert finished is True
    assert set(new_events) == {(a, "done"), (b, "done")}
    assert new_stage_events == []
    assert analysis._pending_futures == {}
    assert fake.shutdown_calls == [False]
    # "running" (at submit) then "done" (at completion) for each folder.
    assert events.count((a, "running")) == 1
    assert events.count((a, "done")) == 1


def test_poll_parallel_progress_reports_error_status(tmp_path, monkeypatch):
    a = str(tmp_path / "a")
    analysis = _analysis([a])
    fake = _FakeExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: fake)
    plan = _plan_for([a], tmp_path)

    # Patch _run_position_headless itself so the "worker" reports an error
    # without needing a real failing pipeline.
    monkeypatch.setattr(
        ba, "_run_position_headless",
        lambda config, folder, output_dir, queue: (folder, "error", "boom"),
    )

    analysis.start_parallel(plan, num_workers=1)
    events, stage_events, finished = analysis.poll_parallel_progress()

    assert finished is True
    assert events == [(a, "error")]
    assert stage_events == []


def test_poll_parallel_progress_is_non_blocking_when_nothing_finished(tmp_path, monkeypatch):
    a = str(tmp_path / "a")
    analysis = _analysis([a])
    queued = _QueuedExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: queued)
    plan = _plan_for([a], tmp_path)

    analysis.start_parallel(plan, num_workers=1)

    events, stage_events, finished = analysis.poll_parallel_progress()

    assert events == []
    assert stage_events == []
    assert finished is False
    assert len(analysis._pending_futures) == 1
    assert queued.shutdown_calls == []  # not finished yet -> no shutdown


def test_poll_parallel_progress_drains_queued_stage_events(tmp_path):
    a = str(tmp_path / "a")
    analysis = _analysis([a])
    analysis._pending_futures = {}
    analysis._executor = None
    analysis._progress_queue = queue_module.Queue()
    analysis._progress_queue.put((a, "displacement", "running", 0.5))
    analysis._progress_queue.put((a, "displacement", "running", 1.0))

    events, stage_events, finished = analysis.poll_parallel_progress()

    assert stage_events == [
        (a, "displacement", "running", 0.5),
        (a, "displacement", "running", 1.0),
    ]
    assert events == []
    assert finished is True
```

And the cancellation pair:

```python
def test_poll_parallel_progress_cancels_not_yet_started_futures(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    events = []
    analysis = _analysis([a, b])
    analysis._progress_callback = lambda folder, status: events.append((folder, status))

    queued = _QueuedExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: queued)
    plan = _plan_for([a, b], tmp_path)
    analysis.start_parallel(plan, num_workers=1)

    analysis.request_cancel()
    new_events, new_stage_events, finished = analysis.poll_parallel_progress()

    # Neither future had started ("run") yet, so both are cancellable.
    assert set(new_events) == {(a, "cancelled"), (b, "cancelled")}
    assert new_stage_events == []
    assert finished is True
    assert (a, "cancelled") in events
    assert (b, "cancelled") in events


def test_poll_parallel_progress_lets_running_future_finish_after_cancel(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    analysis = _analysis([a, b])

    queued = _QueuedExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: queued)
    monkeypatch.setattr(
        ba, "_run_position_headless",
        lambda config, folder, output_dir, queue: (folder, "done", None),
    )
    plan = _plan_for([a, b], tmp_path)
    analysis.start_parallel(plan, num_workers=1)

    # Folder a's worker has already finished (e.g. it was running when cancel
    # was requested); folder b's is still queued.
    queued.run(0)

    analysis.request_cancel()
    events, stage_events, finished = analysis.poll_parallel_progress()

    assert set(events) == {(a, "done"), (b, "cancelled")}
    assert stage_events == []
    assert finished is True
```

(Place `test_poll_parallel_progress_drains_queued_stage_events` directly after `test_poll_parallel_progress_is_non_blocking_when_nothing_finished`, before the `# --- poll_parallel_progress: cancellation ---` comment.)

- [ ] **Step 2: Run the test file to verify it fails**

Run: `pytest tests/test_batch_parallel.py -v`
Expected: Multiple FAILs (`TypeError: <lambda>() missing 1 required positional argument: 'queue'`, unpacking `ValueError: not enough values to unpack`, `ModuleNotFoundError` for `queue_progress_sink`) -- confirms the tests now target the not-yet-built behavior.

- [ ] **Step 3: Update `napariTFM/backend/batch_analysis.py`**

Add two imports. Change:

```python
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, wait as futures_wait
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from time import sleep
from time import time
```

to:

```python
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, wait as futures_wait
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from queue import Empty
from time import sleep
from time import time
```

Change:

```python
from napariTFM.backend.preprocessing import preprocess_frame, preprocess_stack
from napariTFM.utilities import ntfm
```

to:

```python
from napariTFM.backend.preprocessing import preprocess_frame, preprocess_stack
from napariTFM.backend.queue_progress_sink import QueueProgressSink
from napariTFM.utilities import ntfm
```

Replace `_run_position_headless` in full:

```python
def _run_position_headless(
    config: dict, folder: str, output_dir: str, queue,
) -> tuple[str, str, Optional[str]]:
    """Process one position headlessly, in its own (spawned) worker process.

    Module-level (not a method or closure) so it is picklable for
    ``ProcessPoolExecutor`` under the ``spawn`` start method, which has to
    ship the callable itself to the worker. Constructs a ``BatchAnalysis``
    wired with a ``QueueProgressSink`` -- so this worker's real per-stage/
    per-frame progress reaches the parent process via *queue* -- and runs the
    same per-position pipeline as the sequential path (``process_folder``).

    Args:
        config: the run's plain-dict config (pickled across the process
            boundary; must not contain Qt objects -- see
            ``widgets/_run_config.build_run_config``).
        folder: the input folder path for this position.
        output_dir: the resolved ``TFM_data/`` output directory for this
            position (one entry of ``OutputPlan.output_dirs``).
        queue: the ``multiprocessing.Queue`` shared by every worker this run
            submitted (created once by ``start_parallel``), for this
            worker's ``QueueProgressSink`` to report progress on.

    Returns:
        ``(folder, status, error_message)`` where ``status`` is ``"done"``
        or ``"error"``, and ``error_message`` is ``None`` on success or the
        stringified exception on failure. Never raises: a failure inside
        ``process_folder`` is caught and reported as the ``"error"`` status
        instead, so a single bad position can't take down the worker (which
        would otherwise surface as a ``BrokenProcessPool`` for every other
        future in the pool).
    """
    sink = QueueProgressSink(queue, folder)
    analysis = BatchAnalysis(config, sink=sink)
    try:
        analysis.process_folder(folder, output_dir)
    except Exception as e:
        return folder, "error", str(e)
    return folder, "done", None
```

In `BatchAnalysis.__init__`, change:

```python
        self._cancelled = False
        # Parallel-mode pool state (populated by start_parallel).
        self._executor = None
        self._pending_futures = {}
```

to:

```python
        self._cancelled = False
        # Parallel-mode pool state (populated by start_parallel).
        self._executor = None
        self._pending_futures = {}
        self._progress_queue = None
```

Replace `start_parallel` in full:

```python
    def start_parallel(self, plan, num_workers: int) -> None:
        """Submit one headless task per folder to a process pool and return
        immediately (non-blocking; for GUI callers that poll progress via
        :meth:`poll_parallel_progress` from e.g. a Qt timer).

        Uses a ``ProcessPoolExecutor`` (spawn start method -- see module
        docstring/plan rationale: forking a GUI process with live Qt/BLAS/
        OpenMP threads is a deadlock hazard) rather than threads, because the
        pipeline is CPU-bound (numba/numpy/scipy) and would not parallelize
        across a single GIL.

        Futures are submitted in ``self.config['root_folders']`` order (top
        positions first). A pool with ``max_workers=num_workers`` starts the
        first N immediately and pulls the next queued folder in FIFO order as
        a slot frees, which alone satisfies "top positions first" -- no
        explicit priority queue is needed.

        Also creates one ``multiprocessing.Queue`` (same spawn context as the
        pool) shared by every worker this run submits, and passes it into
        each ``_run_position_headless`` call so a worker's
        ``QueueProgressSink`` can report real per-stage/per-frame progress
        back across the process boundary; :meth:`poll_parallel_progress`
        drains it every call.

        Stores the executor, a ``{Future: folder}`` map, and the progress
        queue on ``self._executor`` / ``self._pending_futures`` /
        ``self._progress_queue`` for :meth:`poll_parallel_progress` to drain.
        """
        ctx = multiprocessing.get_context("spawn")
        self._executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
        self._pending_futures: dict[Future, str] = {}
        self._progress_queue = ctx.Queue()
        for folder in self.config['root_folders']:
            output_dir = str(plan.output_dirs[folder])
            future = self._executor.submit(
                _run_position_headless, self.config, folder, output_dir, self._progress_queue,
            )
            self._pending_futures[future] = folder
            # "running" at submission time is an acceptable approximation (a
            # queued-but-not-yet-started task reports "running" slightly
            # early), matching the "running" semantics used elsewhere in this
            # file -- the queue above reports real per-stage progress once the
            # worker actually starts.
            self._report_progress(folder, "running")
```

Replace `poll_parallel_progress` in full:

```python
    def poll_parallel_progress(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, Optional[float]]], bool]:
        """Drain whatever parallel-mode futures have completed, and whatever
        per-stage progress messages have arrived, since the last call --
        without blocking. Intended to be called repeatedly (e.g. from a Qt
        timer) until ``finished`` is ``True``.

        Returns:
            ``(events, stage_events, finished)`` --

            - ``events``: a list of ``(folder, status)`` pairs that newly
              completed during *this* call, where ``status`` is one of
              ``"done"``, ``"error"``, or ``"cancelled"``. Each pair is also
              reported through :meth:`_report_progress` (the same hook the
              sequential path already uses), so a caller only needs to
              listen on one channel; the returned list is for callers (e.g.
              the GUI) that want to react inline without a callback, such as
              reloading a finished position's ``.ntfm`` from disk.
            - ``stage_events``: a list of ``(folder, stage, status, fraction)``
              tuples drained from the shared progress queue (one entry per
              ``QueueProgressSink`` message a worker put since the last poll)
              -- ``status`` is ``"running"`` (with a growing ``fraction``) or
              ``"done"`` (``fraction`` is then ``None``), mirroring
              ``ViewerSink.on_stage_progress``'s shape. Always drained fully,
              even on a call where ``events``/``finished`` report nothing new.
            - ``finished``: ``True`` once every submitted folder has been
              accounted for (no futures left pending after this call's
              bookkeeping). When it flips to ``True`` the executor is shut
              down via ``shutdown(wait=False)`` (non-blocking -- workers are
              already done by construction at that point).

        Cancellation: checks ``self._cancelled`` (set by
        :meth:`request_cancel`) on every call. When set, every not-yet-started
        future has ``.cancel()`` called on it -- this only succeeds for
        futures still queued (``Future.cancel()`` returns ``True``/``False``)
        -- and those folders are reported/returned with status
        ``"cancelled"``. There is no new submission after :meth:`start_parallel`
        runs (everything is submitted up front), so "stop submitting" is
        automatic. Already-running workers are *not* forcefully terminated;
        they finish naturally and report their real ``done``/``error`` status
        on a later poll.
        """
        stage_events: list[tuple[str, str, str, Optional[float]]] = []
        progress_queue = getattr(self, "_progress_queue", None)
        if progress_queue is not None:
            while True:
                try:
                    stage_events.append(progress_queue.get_nowait())
                except Empty:
                    break

        events: list[tuple[str, str]] = []

        if getattr(self, "_cancelled", False):
            for future in list(self._pending_futures):
                if future.cancel():
                    folder = self._pending_futures.pop(future)
                    self._report_progress(folder, "cancelled")
                    events.append((folder, "cancelled"))

        if self._pending_futures:
            done, _pending = futures_wait(list(self._pending_futures), timeout=0)
            for future in done:
                folder = self._pending_futures.pop(future)
                try:
                    _folder, status, _err = future.result()
                except Exception as e:
                    # A worker process crashed outright (e.g. BrokenProcessPool)
                    # rather than returning its usual (folder, status, err)
                    # tuple -- still report this folder as failed instead of
                    # raising out of a non-blocking poll.
                    status = "error"
                    print(f"Parallel worker for {folder} failed: {str(e)}")
                self._report_progress(folder, status)
                events.append((folder, status))

        finished = not self._pending_futures
        if finished and getattr(self, "_executor", None) is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        return events, stage_events, finished
```

In `_process_all_folders_parallel`, change:

```python
        self.start_parallel(plan, num_workers)
        finished = False
        while not finished:
            _events, finished = self.poll_parallel_progress()
            if not finished:
                sleep(0.05)
```

to:

```python
        self.start_parallel(plan, num_workers)
        finished = False
        while not finished:
            _events, _stage_events, finished = self.poll_parallel_progress()
            if not finished:
                sleep(0.05)
```

- [ ] **Step 4: Run the test file to verify it passes**

Run: `pytest tests/test_batch_parallel.py tests/test_batch_progress.py -v`
Expected: all passed

- [ ] **Step 5: Run the real-process integration test too**

Run: `pytest tests/test_batch_parallel_real_pool.py -v`
Expected: 1 passed (proves the shared `multiprocessing.Queue` pickles cleanly across a real spawned process boundary; both analysis steps are disabled in this test so no stage events fire, but the plumbing must not break the run)

- [ ] **Step 6: Commit**

```bash
git add napariTFM/backend/batch_analysis.py tests/test_batch_parallel.py
git commit -m "Thread a shared progress queue through parallel batch workers"
```

---

## Task 3: `MiniRail` fractional pie-wedge rendering

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py`
- Test: `tests/test_experiments_list.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_experiments_list.py`, insert these four tests directly after `test_minirail_off_dot_is_recessed_and_distinct_from_not_started` (before `test_minirail_click_emits_the_stage_under_the_cursor`):

```python
def test_minirail_set_stage_progress_stores_clamped_fraction(app):
    rail = MiniRail()
    rail.set_statuses({"force": "running"})
    rail.set_stage_progress("force", 0.5)
    assert rail._progress["force"] == 0.5
    rail.set_stage_progress("force", 1.4)
    assert rail._progress["force"] == 1.0
    rail.set_stage_progress("force", -0.2)
    assert rail._progress["force"] == 0.0


def test_minirail_set_stage_progress_accepts_none(app):
    rail = MiniRail()
    rail.set_statuses({"force": "running"})
    rail.set_stage_progress("force", 0.5)
    rail.set_stage_progress("force", None)
    assert rail._progress["force"] is None


def test_minirail_status_change_away_from_running_clears_progress(app):
    """A finished/restarted stage must not carry over its previous fill."""
    rail = MiniRail()
    rail.set_statuses({"force": "running"})
    rail.set_stage_progress("force", 0.7)
    rail.set_statuses({"force": "done"})
    assert rail._progress["force"] is None

    rail.set_statuses({"force": "running"})
    assert rail._progress["force"] is None


def test_minirail_paints_without_error_at_various_progress(app):
    """Smoke test: the pie-wedge paint path doesn't raise for edge fractions."""
    rail = MiniRail()
    rail.set_statuses({"force": "running"})
    for fraction in (0.0, 0.25, 0.99, 1.0):
        rail.set_stage_progress("force", fraction)
        rail.show()
        app.processEvents()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_experiments_list.py -v -k minirail_set_stage_progress or minirail_status_change or minirail_paints_without_error`
Expected: FAIL with `AttributeError: 'MiniRail' object has no attribute 'set_stage_progress'`

- [ ] **Step 3: Implement in `napariTFM/widgets/_experiments_list.py`**

In `MiniRail.__init__`, change:

```python
    def __init__(self, stages=PIPELINE_STAGES, parent=None):
        super().__init__(parent)
        self.stages = tuple(stages)
        self._statuses = {key: "not_started" for key in self.stages}
        # Index of the dot under the cursor (-1 = none), driving the hover halo.
        self._hover_idx = -1
        self.setFixedSize(self.DOT_GAP * len(self.stages), 2 * self.DOT_R + 6)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Track motion so the dots can light up and swap the cursor per-dot; the
        # tiny row is otherwise indistinguishable from a static status readout.
        self.setMouseTracking(True)

    def set_statuses(self, statuses: dict[str, str]) -> None:
        for key in self.stages:
            if key in statuses:
                self._statuses[key] = statuses[key]
        self.update()
```

to:

```python
    def __init__(self, stages=PIPELINE_STAGES, parent=None):
        super().__init__(parent)
        self.stages = tuple(stages)
        self._statuses = {key: "not_started" for key in self.stages}
        # Fractional completion (0..1) of an in-flight "running" stage, or None
        # when no per-frame progress is known (mirrors StageSpine._progress) --
        # fed by a parallel Run-all's real per-stage/per-frame events instead of
        # the flat placeholder mark_running() paints at submission time.
        self._progress: dict[str, Optional[float]] = {key: None for key in self.stages}
        # Index of the dot under the cursor (-1 = none), driving the hover halo.
        self._hover_idx = -1
        self.setFixedSize(self.DOT_GAP * len(self.stages), 2 * self.DOT_R + 6)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Track motion so the dots can light up and swap the cursor per-dot; the
        # tiny row is otherwise indistinguishable from a static status readout.
        self.setMouseTracking(True)

    def set_statuses(self, statuses: dict[str, str]) -> None:
        for key in self.stages:
            if key in statuses:
                self._statuses[key] = statuses[key]
                if statuses[key] != "running":
                    # Stale progress must not leak into this stage's next run
                    # (mirrors StageSpine.set_status's same guard).
                    self._progress[key] = None
        self.update()

    def set_stage_progress(self, stage: str, fraction: Optional[float]) -> None:
        """Set the in-flight fractional completion (0..1) of one stage's dot.

        Only visible while that stage's status is ``"running"``; harmless to
        call at other times since :meth:`paintEvent` ignores it then. Pass
        ``None`` to fall back to the plain solid-fill "running" dot.
        """
        self._progress[stage] = None if fraction is None else max(0.0, min(1.0, fraction))
        self.update()
```

Replace `MiniRail.paintEvent` in full:

```python
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2.0
        r = self.DOT_R
        for i, stage in enumerate(self.stages):
            cx = self.DOT_GAP * i + self.DOT_GAP / 2.0
            status = self._statuses[stage]
            fill, ring = _node_style(status, stage_accent(stage))
            if status == "off":
                painter.setPen(QPen(ring, 2, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
                continue
            if i == self._hover_idx:
                # Light the dot up under the cursor so it reads as a button.
                halo = QColor(ring)
                halo.setAlpha(70)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(halo))
                hr = r + 3
                painter.drawEllipse(QRectF(cx - hr, cy - hr, 2 * hr, 2 * hr))
            rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            progress = self._progress[stage]
            if status == "running" and progress is not None:
                # A pie wedge growing clockwise from 12 o'clock reads as a fill
                # level (mirrors StageSpine), so a parallel run's dot shows its
                # real per-stage progress instead of a flat "something is
                # happening" dot for the worker's entire runtime.
                painter.setPen(QPen(ring, 1.5))
                painter.setBrush(QBrush(self.palette().color(self.backgroundRole())))
                painter.drawEllipse(rect)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(ring))
                span = -round(360 * 16 * progress)
                painter.drawPie(rect, 90 * 16, span)
                continue
            centre = fill if fill is not None else self.palette().color(self.backgroundRole())
            painter.setPen(QPen(ring, 1.5))
            painter.setBrush(QBrush(centre))
            painter.drawEllipse(rect)
        painter.end()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_experiments_list.py -v`
Expected: all passed (existing `MiniRail`/`ExperimentsList` tests plus the four new ones)

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Add fractional pie-wedge rendering to MiniRail's stage dots"
```

---

## Task 4: `ExperimentsList.set_row_stage_progress`

**Files:**
- Modify: `napariTFM/widgets/_experiments_list.py`
- Test: `tests/test_experiments_list.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_experiments_list.py`, insert these three tests directly after `test_apply_row_statuses_paints_one_row_without_a_disk_read` (before `test_on_row_stage_clicked_requests_that_stage_load`):

```python
def test_set_row_stage_progress_updates_one_dot_only(app):
    widget = ExperimentsList(status_fn=lambda path: {
        "preprocessing": "not_started", "displacement": "not_started",
        "force": "not_started", "stress": "off",
    })
    widget.set_experiments(["/data/a", "/data/b"])

    widget.set_row_stage_progress("/data/a", "displacement", "running", 0.5)

    row_a, row_b = widget._rows
    assert row_a.mini_rail._statuses["displacement"] == "running"
    assert row_a.mini_rail._progress["displacement"] == 0.5
    # Sibling dot on the same row, and the other row entirely, are untouched.
    assert row_a.mini_rail._statuses["force"] == "not_started"
    assert row_b.mini_rail._progress["displacement"] is None


def test_set_row_stage_progress_ignores_unknown_path(app):
    widget = ExperimentsList(status_fn=lambda path: {
        "preprocessing": "not_started", "displacement": "not_started",
        "force": "not_started", "stress": "off",
    })
    widget.set_experiments(["/data/a"])

    # Must not raise for a path no longer in the table (e.g. a stale event
    # arriving after the row was deleted mid-run).
    widget.set_row_stage_progress("/data/gone", "force", "running", 0.5)


def test_set_row_stage_progress_clears_on_stage_finish(app):
    widget = ExperimentsList(status_fn=lambda path: {
        "preprocessing": "not_started", "displacement": "not_started",
        "force": "not_started", "stress": "off",
    })
    widget.set_experiments(["/data/a"])

    widget.set_row_stage_progress("/data/a", "displacement", "running", 0.6)
    widget.set_row_stage_progress("/data/a", "displacement", "done", None)

    row_a = widget._rows[0]
    assert row_a.mini_rail._statuses["displacement"] == "done"
    assert row_a.mini_rail._progress["displacement"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_experiments_list.py -v -k set_row_stage_progress`
Expected: FAIL with `AttributeError: 'ExperimentsList' object has no attribute 'set_row_stage_progress'`

- [ ] **Step 3: Implement in `napariTFM/widgets/_experiments_list.py`**

Add this method directly after `mark_running` (which ends with `row.set_stage_statuses(statuses)` / `return`), before `_on_run_all_clicked`:

```python
    def set_row_stage_progress(
        self, path: str, stage: str, status: str, fraction: Optional[float]
    ) -> None:
        """Paint one row's one stage dot with real in-flight progress (P4/#10).

        Fed by a parallel Run-all's per-stage/per-frame events (routed through
        the shell's ``_on_batch_stage_progress``), so a parallel-mode row's dot
        fills the same way the single-experiment detail panel's ``StageSpine``
        already does, instead of sitting on the flat ``mark_running()``
        placeholder for the worker's entire runtime. A no-op for a path not in
        the table.
        """
        for row in self._rows:
            if row.path != path:
                continue
            row.set_stage_statuses({stage: status})
            row.mini_rail.set_stage_progress(stage, fraction)
            return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_experiments_list.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_experiments_list.py tests/test_experiments_list.py
git commit -m "Add ExperimentsList.set_row_stage_progress for per-dot updates"
```

---

## Task 5: Wire the parallel poll loop to the new row-progress path

**Files:**
- Modify: `napariTFM/widgets/_widget.py`
- Modify: `tests/test_workflow_shell.py`

- [ ] **Step 1: Update `_FakeParallelBatchAnalysis` and add the failing test**

In `tests/test_workflow_shell.py`, change:

```python
    def queue_poll_result(self, events, finished):
        self._poll_queue.append((events, finished))

    def poll_parallel_progress(self):
        events, finished = self._poll_queue.pop(0)
        for folder, status in events:
            if self.progress_callback:
                self.progress_callback(folder, status)
        return events, finished
```

to:

```python
    def queue_poll_result(self, events, finished, stage_events=None):
        self._poll_queue.append((events, finished, stage_events or []))

    def poll_parallel_progress(self):
        events, finished, stage_events = self._poll_queue.pop(0)
        for folder, status in events:
            if self.progress_callback:
                self.progress_callback(folder, status)
        return events, stage_events, finished
```

(This keeps every existing 2-argument `queue_poll_result(events, finished)` call site in this file working unchanged -- `stage_events` defaults to `[]`.)

Add this new test after `test_run_all_parallel_keeps_polling_after_cancel_until_finished`, before `test_run_all_sequential_path_is_unchanged_for_default_num_workers`:

```python
def test_run_all_parallel_routes_stage_events_to_row_progress(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/exp_a", "/data/exp_b"])
    widget.experiments_list._num_workers_spinbox.setValue(2)

    seen = []
    monkeypatch.setattr(
        widget.experiments_list, "set_row_stage_progress",
        lambda path, stage, status, fraction: seen.append((path, stage, status, fraction)),
    )

    widget.experiments_list.run_all_requested.emit()

    analyzer = _FakeParallelBatchAnalysis.last_instance
    timer = _FakeTimer.instances[-1]

    analyzer.queue_poll_result(
        [], False,
        stage_events=[
            ("/data/exp_a", "displacement", "running", 0.5),
            ("/data/exp_b", "force", "running", 0.25),
        ],
    )
    timer.fire()

    assert seen == [
        ("/data/exp_a", "displacement", "running", 0.5),
        ("/data/exp_b", "force", "running", 0.25),
    ]
```

- [ ] **Step 2: Run tests to verify the new test fails**

Run: `pytest tests/test_workflow_shell.py -v -k stage_events`
Expected: FAIL (`ValueError: not enough values to unpack` inside `_widget.py`'s `_poll`, since it still does `events, finished = analyzer.poll_parallel_progress()`)

- [ ] **Step 3: Implement in `napariTFM/widgets/_widget.py`**

In `_run_all_experiments_parallel`'s `_poll` closure, change:

```python
        def _poll():
            events, finished = analyzer.poll_parallel_progress()
            for folder, status in events:
```

to:

```python
        def _poll():
            events, stage_events, finished = analyzer.poll_parallel_progress()
            for folder, stage, status, fraction in stage_events:
                self._on_batch_stage_progress(folder, stage, status, fraction)
            for folder, status in events:
```

(Everything after that `for folder, status in events:` line is unchanged.)

Add this new method directly after `_on_run_all_stage_progress` (which ends with `section.set_progress(fraction)`), before `_on_batch_progress`:

```python
    def _on_batch_stage_progress(
        self, folder: str, stage: str, status: str, fraction: float | None
    ) -> None:
        """Route one parallel-mode worker's real per-stage progress onto its row.

        The parallel poll timer's per-tick stage events land here and go
        straight to that one folder's one mini-rail dot -- the parallel-mode
        sibling of ``_on_run_all_stage_progress``, which does the equivalent
        for the single-experiment detail panel's ``StageSpine`` during a
        sequential run.
        """
        self.experiments_list.set_row_stage_progress(folder, stage, status, fraction)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workflow_shell.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add napariTFM/widgets/_widget.py tests/test_workflow_shell.py
git commit -m "Route parallel batch stage events to per-row progress dots"
```

---

## Task 6: Full regression run and live visual check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q`
Expected: all passed. (Project memory notes the full suite has an occasional flaky offscreen-Qt SIGABRT unrelated to this change -- if the run aborts outright rather than reporting a specific failing test, re-run just the touched files: `pytest tests/test_queue_progress_sink.py tests/test_batch_parallel.py tests/test_batch_parallel_real_pool.py tests/test_experiments_list.py tests/test_workflow_shell.py tests/test_pipeline_sink.py tests/test_viewer_sink.py tests/test_stage_spine.py -q`.)

- [ ] **Step 2: Manually verify in the running app**

Green tests confirm the wiring and paint logic in isolation, not that the dots read clearly at real size on screen -- that needs a human looking at it. Launch napari with the plugin, add at least 2 experiment folders with `preprocessing`/`displacement`/`force` enabled, set the worker-count spinbox to 2+, and click "Run all". Confirm each row's dots now visibly grow (pie-wedge fill) stage by stage instead of snapping straight to flat amber, and that a finished stage settles to its normal solid "done" dot.

If the wedge is too small to read clearly at `MiniRail.DOT_R = 4` (the spec flagged this as an open tuning question), adjust `DOT_R` and/or the `1.5`-width ring pen in `MiniRail.paintEvent` and re-run Step 2 until it reads clearly. This is expected to be an iterative, eyes-on step -- do not report the task complete without having actually watched a parallel run in the live app.

- [ ] **Step 3: Commit any visual tuning**

Only if Step 2 required changes:

```bash
git add napariTFM/widgets/_experiments_list.py
git commit -m "Tune MiniRail progress-wedge geometry for readability at 4px"
```
