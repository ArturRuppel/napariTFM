"""Parallel batch-worker processing (ProcessPoolExecutor-backed run-all).

Two layers under test:

- ``_run_position_headless``: the module-level, picklable per-position task a
  worker process actually runs. It's plain code, so it's tested directly --
  no process pool involved.
- The orchestration (``start_parallel`` / ``poll_parallel_progress`` /
  ``_process_all_folders_parallel``): tested with a synchronous fake standing
  in for ``ProcessPoolExecutor`` (monkeypatched at the module level), so these
  tests never spawn a real OS process and stay sub-second. The real
  multi-process integration is a separate task.
"""

from concurrent.futures import Future

from napariTFM.backend import batch_analysis as ba
from napariTFM.backend.batch_analysis import BatchAnalysis, _run_position_headless
from napariTFM.utilities.batch_output import resolve_output_plan


def _minimal_config(**overrides):
    """The smallest config that lets ``process_folder`` run to completion
    with every analysis step disabled (no real TFM input needed)."""
    config = {
        "analysis_steps": {
            "preprocessing": False,
            "displacement": False,
            "force": False,
            "stress": False,
        },
        "visualizations": {},
        "parameters": {},
        "input_files": {},
    }
    config.update(overrides)
    return config


def _analysis(root_folders, **config_overrides):
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = _minimal_config(root_folders=root_folders, **config_overrides)
    analysis._progress_callback = None
    analysis._sink = None
    analysis._cancelled = False
    return analysis


# --- _run_position_headless: a plain, picklable function ------------------

def test_run_position_headless_returns_done_on_success(tmp_path):
    config = _minimal_config()
    folder = str(tmp_path / "input")
    output_dir = str(tmp_path / "TFM_data" / "input")

    result = _run_position_headless(config, folder, output_dir)

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

    folder_out, status, err = _run_position_headless(config, folder, output_dir)

    assert folder_out == folder
    assert status == "error"
    assert err is not None and "analysis_steps" in err


def test_run_position_headless_constructs_sinkless_batch_analysis(monkeypatch, tmp_path):
    captured = {}

    def _fake_process_folder(self, folder, output_dir=None):
        captured["sink"] = self._sink
        captured["progress_callback"] = self._progress_callback

    monkeypatch.setattr(BatchAnalysis, "process_folder", _fake_process_folder)

    folder = str(tmp_path / "input")
    output_dir = str(tmp_path / "TFM_data" / "input")
    result = _run_position_headless(_minimal_config(), folder, output_dir)

    assert result == (folder, "done", None)
    assert captured["sink"] is None
    assert captured["progress_callback"] is None


# --- a synchronous fake standing in for ProcessPoolExecutor ----------------

class _FakeExecutor:
    """Records submissions; runs each task immediately and wraps the result
    in an already-completed ``Future`` -- like ``ProcessPoolExecutor`` but
    without ever leaving this process. Constructor args are accepted and
    ignored so it's a drop-in for ``ProcessPoolExecutor(max_workers=..,
    mp_context=..)``.
    """

    instances = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.submitted_args = []
        self.shutdown_calls = []
        _FakeExecutor.instances.append(self)

    def submit(self, fn, *args):
        self.submitted_args.append(args)
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as e:  # pragma: no cover - defensive
            future.set_exception(e)
        return future

    def shutdown(self, wait=True):
        self.shutdown_calls.append(wait)


class _QueuedExecutor:
    """Like ``_FakeExecutor`` but does *not* run the task on submit -- the
    test drives completion explicitly via ``.run(index)``. This is what lets
    a test exercise "cancel a not-yet-started future", since a future that's
    already done can never be cancelled.
    """

    def __init__(self, *args, **kwargs):
        self.calls = []  # list of (future, fn, args)
        self.shutdown_calls = []

    def submit(self, fn, *args):
        future = Future()
        self.calls.append((future, fn, args))
        return future

    def run(self, index):
        future, fn, args = self.calls[index]
        try:
            future.set_result(fn(*args))
        except Exception as e:
            future.set_exception(e)

    def shutdown(self, wait=True):
        self.shutdown_calls.append(wait)


def _plan_for(root_folders, tmp_path):
    return resolve_output_plan(root_folders, str(tmp_path / "processed"))


def _stub_run_position_headless(monkeypatch, status="done", err=None):
    """Replace the real (heavy, sleep(2)-banner-bearing) pipeline with an
    instant stand-in for orchestration tests, which care about
    future/event bookkeeping, not pipeline behaviour -- that's covered by
    the dedicated ``_run_position_headless`` unit tests above.
    """
    monkeypatch.setattr(
        ba, "_run_position_headless",
        lambda config, folder, output_dir: (folder, status, err),
    )


# --- start_parallel ---------------------------------------------------------

def test_start_parallel_submits_in_root_folders_order_and_reports_running(tmp_path, monkeypatch):
    a, b, c = (str(tmp_path / n) for n in ("a", "b", "c"))
    events = []
    analysis = _analysis([a, b, c])
    analysis._progress_callback = lambda folder, status: events.append((folder, status))

    _stub_run_position_headless(monkeypatch)
    fake = _FakeExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: fake)
    plan = _plan_for([a, b, c], tmp_path)

    analysis.start_parallel(plan, num_workers=2)

    # Submitted in the same order as root_folders ("top positions first").
    submitted_folders = [args[1] for args in fake.submitted_args]
    assert submitted_folders == [a, b, c]
    # Every folder reports "running" at submission time.
    assert events == [(a, "running"), (b, "running"), (c, "running")]
    assert set(analysis._pending_futures.values()) == {a, b, c}


def test_start_parallel_passes_config_and_output_dir_to_worker(tmp_path, monkeypatch):
    a = str(tmp_path / "a")
    analysis = _analysis([a])
    _stub_run_position_headless(monkeypatch)
    fake = _FakeExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: fake)
    plan = _plan_for([a], tmp_path)

    analysis.start_parallel(plan, num_workers=1)

    (fn_config, fn_folder, fn_output_dir), = [args for args in fake.submitted_args]
    assert fn_config is analysis.config
    assert fn_folder == a
    assert fn_output_dir == str(plan.output_dirs[a])


# --- poll_parallel_progress: normal completion -----------------------------

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

    new_events, finished = analysis.poll_parallel_progress()

    assert finished is True
    assert set(new_events) == {(a, "done"), (b, "done")}
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
    monkeypatch.setattr(ba, "_run_position_headless", lambda config, folder, output_dir: (folder, "error", "boom"))

    analysis.start_parallel(plan, num_workers=1)
    events, finished = analysis.poll_parallel_progress()

    assert finished is True
    assert events == [(a, "error")]


def test_poll_parallel_progress_is_non_blocking_when_nothing_finished(tmp_path, monkeypatch):
    a = str(tmp_path / "a")
    analysis = _analysis([a])
    queued = _QueuedExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: queued)
    plan = _plan_for([a], tmp_path)

    analysis.start_parallel(plan, num_workers=1)

    events, finished = analysis.poll_parallel_progress()

    assert events == []
    assert finished is False
    assert len(analysis._pending_futures) == 1
    assert queued.shutdown_calls == []  # not finished yet -> no shutdown


# --- poll_parallel_progress: cancellation -----------------------------------

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
    new_events, finished = analysis.poll_parallel_progress()

    # Neither future had started ("run") yet, so both are cancellable.
    assert set(new_events) == {(a, "cancelled"), (b, "cancelled")}
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
        lambda config, folder, output_dir: (folder, "done", None),
    )
    plan = _plan_for([a, b], tmp_path)
    analysis.start_parallel(plan, num_workers=1)

    # Folder a's worker has already finished (e.g. it was running when cancel
    # was requested); folder b's is still queued.
    queued.run(0)

    analysis.request_cancel()
    events, finished = analysis.poll_parallel_progress()

    assert set(events) == {(a, "done"), (b, "cancelled")}
    assert finished is True


# --- _process_all_folders_parallel: blocking CLI/test variant --------------

def test_process_all_folders_parallel_blocks_until_all_done(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    events = []
    analysis = _analysis([a, b])
    analysis._progress_callback = lambda folder, status: events.append((folder, status))

    _stub_run_position_headless(monkeypatch)
    fake = _FakeExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: fake)
    plan = _plan_for([a, b], tmp_path)

    analysis._process_all_folders_parallel(plan, num_workers=2)

    assert set(events) == {(a, "running"), (a, "done"), (b, "running"), (b, "done")}
    assert analysis._pending_futures == {}


def test_process_all_folders_parallel_honors_cancellation(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    events = []
    analysis = _analysis([a, b])
    analysis._progress_callback = lambda folder, status: events.append((folder, status))
    # Already cancelled before the blocking call starts (e.g. the GUI's
    # Cancel control was clicked just before "Run all" finished submitting).
    analysis._cancelled = True

    # A queued (never auto-run) executor: if cancellation weren't honored,
    # the blocking loop would spin forever waiting for futures that no one
    # ever completes.
    queued = _QueuedExecutor()
    monkeypatch.setattr(ba, "ProcessPoolExecutor", lambda *a, **kw: queued)
    plan = _plan_for([a, b], tmp_path)

    analysis._process_all_folders_parallel(plan, num_workers=2)

    assert (a, "cancelled") in events
    assert (b, "cancelled") in events
    assert analysis._pending_futures == {}


def test_process_all_folders_routes_to_parallel_when_num_workers_set(tmp_path, monkeypatch):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    analysis = _analysis([a, b], num_workers=2)

    called = {}
    monkeypatch.setattr(
        BatchAnalysis, "_process_all_folders_parallel",
        lambda self, plan, num_workers: called.update(plan=plan, num_workers=num_workers),
    )
    monkeypatch.setattr(
        BatchAnalysis, "_process_all_folders_sequential",
        lambda self, plan: called.update(sequential=True),
    )

    analysis.process_all_folders()

    assert called.get("num_workers") == 2
    assert "sequential" not in called


def test_process_all_folders_routes_to_sequential_by_default(tmp_path, monkeypatch):
    a = str(tmp_path / "a")
    analysis = _analysis([a])  # no num_workers -> defaults to 1

    called = {}
    monkeypatch.setattr(
        BatchAnalysis, "_process_all_folders_sequential",
        lambda self, plan: called.update(sequential=True),
    )
    monkeypatch.setattr(
        BatchAnalysis, "_process_all_folders_parallel",
        lambda self, plan, num_workers: called.update(parallel=True),
    )

    analysis.process_all_folders()

    assert called == {"sequential": True}
