"""Drift-regression guard for the sealed run/cancel lifecycle (A-2).

The three stage controllers used to hand-copy their run/cancel/freeze logic and
it drifted into bugs (a stage that forgot to freeze on run; three divergent
cancels; a cancel that hung on an unbounded wait). The lifecycle now lives once
in ``BaseAnalysisController`` as a sealed template method. These tests pin the
invariants directly, driving ``run()``/``cancel()`` with a fake worker so the
guarantees are checked without a real Qt background thread:

  * run() always freezes the UI, validates, begins streaming, starts the worker;
  * the UI is unfrozen on EVERY terminal path — success, error, cancel, and a
    synchronous setup failure that never starts a worker;
  * the final result is committed on ``returned``;
  * cancel() is cooperative (``quit()`` only, no wait/terminate), frees the UI
    inline, and orphans the worker (disconnects yielded/returned/errored) so no
    late frame, result, or error can race into the torn-down stage;
  * a cancelled worker's late ``finished`` is a guarded no-op — it does not
    unfreeze a fresh run the user has since started;
  * ``run`` and ``cancel`` cannot be overridden by a subclass.
"""

import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._base_widget import BaseAnalysisController


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _silence_dialogs(monkeypatch):
    # _handle_error pops a modal QMessageBox.critical, which would block offscreen.
    from qtpy.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)


class _FakeSignal:
    """Minimal stand-in for a worker's Qt signal: connect / disconnect / emit."""

    def __init__(self):
        self._cbs = []

    def connect(self, cb):
        self._cbs.append(cb)

    def disconnect(self, cb=None):
        if cb is None:
            self._cbs.clear()
        elif cb in self._cbs:
            self._cbs.remove(cb)

    def emit(self, *args):
        for cb in list(self._cbs):
            cb(*args)


class _FakeWorker:
    def __init__(self):
        self.yielded = _FakeSignal()
        self.returned = _FakeSignal()
        self.errored = _FakeSignal()
        self.finished = _FakeSignal()
        self.started = False
        self.quit_called = False

    def start(self):
        self.started = True

    def quit(self):
        self.quit_called = True


class _LifecycleController(BaseAnalysisController):
    """Concrete controller with trivial hooks, so we exercise only the template."""

    def __init__(self, *args, validate_error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = _FakeWorker()
        self._validate_error = validate_error
        self.validated = False
        self.began = False
        self.finalized = "UNSET"
        self.cleaned = False

    def _validate(self):
        self.validated = True
        if self._validate_error is not None:
            raise self._validate_error

    def _run_params(self):
        return object()

    def _begin_stream(self, params):
        self.began = True

    def _build_worker(self, params):
        return self.worker

    def _on_frame_processed(self, payload):
        pass

    def _finalize(self, result):
        self.finalized = result

    def _on_cancel_cleanup(self):
        self.cleaned = True


def _make(app, **kwargs):
    ctrl = _LifecycleController(
        viewer=None, data_manager=None, parameter_manager=None,
        visualization_manager=None, **kwargs,
    )
    frozen = []
    ctrl.ui_frozen.connect(frozen.append)
    return ctrl, frozen


def test_run_freezes_validates_begins_and_starts(app):
    ctrl, frozen = _make(app)
    ctrl.run()
    assert frozen == [True]                    # froze on run, no premature unfreeze
    assert ctrl.validated and ctrl.began
    assert ctrl.worker.started
    assert ctrl.worker in ctrl.active_workers


def test_finished_is_the_single_unfreeze_chokepoint(app):
    ctrl, frozen = _make(app)
    ctrl.run()
    ctrl.worker.finished.emit()                # terminal: success/abort/error all land here
    assert frozen[-1] is False
    assert ctrl.active_workers == []           # worker forgotten


def test_returned_commits_the_result(app):
    ctrl, _ = _make(app)
    ctrl.run()
    sentinel = object()
    ctrl.worker.returned.emit(sentinel)
    assert ctrl.finalized is sentinel


def test_synchronous_setup_failure_unfreezes_and_never_starts(app):
    ctrl, frozen = _make(app, validate_error=ValueError("no data"))
    ctrl.run()
    assert ctrl.validated
    assert not ctrl.worker.started             # validate raised before start
    assert frozen == [True, False]             # frozen then unfrozen — never left frozen
    assert ctrl.active_workers == []


def test_cancel_frees_ui_inline_and_orphans_the_worker(app):
    ctrl, frozen = _make(app)
    ctrl.run()
    worker = ctrl.worker
    ctrl.cancel()

    assert worker.quit_called                  # cooperative quit(), no wait/terminate
    # Orphaned: no late frame, result, or error can reach the torn-down stage.
    assert worker.yielded._cbs == []
    assert worker.returned._cbs == []
    assert worker.errored._cbs == []
    assert ctrl.active_workers == []           # forgotten now, not deferred
    assert ctrl.cleaned                        # stage cleanup ran inline
    assert frozen[-1] is False                 # UI freed immediately — no waiting


def test_cancelled_workers_late_finish_does_not_unfreeze_a_fresh_run(app):
    ctrl, frozen = _make(app)
    ctrl.run()
    cancelled = ctrl.worker
    ctrl.cancel()
    assert frozen[-1] is False                 # freed by the cancel

    # User immediately starts a new run; it freezes and owns the UI.
    ctrl.worker = _FakeWorker()
    ctrl.run()
    assert frozen[-1] is True

    # The cancelled worker's in-flight frame now finishes, late.
    cancelled.finished.emit()
    assert frozen[-1] is True                   # must NOT unfreeze the live run
    assert ctrl.worker in ctrl.active_workers


def test_cancel_with_no_active_worker_unfreezes_inline(app):
    ctrl, frozen = _make(app)
    ctrl.cancel()                              # nothing running
    assert ctrl.cleaned
    assert frozen[-1] is False


def test_run_and_cancel_are_sealed():
    with pytest.raises(TypeError):
        class _OverridesRun(BaseAnalysisController):
            def run(self):
                pass

    with pytest.raises(TypeError):
        class _OverridesCancel(BaseAnalysisController):
            def cancel(self):
                pass
