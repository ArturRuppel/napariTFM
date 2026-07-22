"""Previews run off the GUI thread.

Each stage's single-frame preview is a full solve whose cost occasionally spikes
(GPU contention, a cold kernel); run inline that spell freezes the whole window.
These tests pin the two halves of the async split without ever starting a real
worker thread — matching the streaming tests, which drive the controller hooks
directly rather than spinning the Qt event loop:

* the shared ``_start_preview_worker`` plumbing wires a worker onto the run
  lifecycle's terminal-state slots (freeze on start, register for ``cancel``,
  route the result to the main-thread paint callback, unfreeze on finish, and
  surface a paint error rather than crashing the thread); and
* each stage's paint half issues the exact visualization calls the old inline
  preview did — a regression guard on the refactor.
"""
import sys
import types

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

# The stress backend imports optional FE libs at import time; stub them so the
# controller module imports on a machine without them (mirrors the streaming test).
for _name in ["gmsh", "solidspy", "solidspy.assemutil",
              "solidspy.postprocesor", "solidspy.solutil"]:
    sys.modules.setdefault(_name, types.ModuleType(_name))


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


# --- thread-free fakes ----------------------------------------------------

class _FakeSignal:
    """A napari-worker signal stand-in: connect records slots, emit calls them."""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def disconnect(self, slot):
        if slot not in self._slots:
            raise TypeError("slot is not connected")
        self._slots.remove(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class _FakeWorker:
    """A configured-but-inert ``@thread_worker`` — ``start`` runs no thread."""

    def __init__(self):
        self.returned = _FakeSignal()
        self.errored = _FakeSignal()
        self.finished = _FakeSignal()
        self.started = False

    def start(self):
        self.started = True


class _RecordingViz:
    """Captures the preview visualization calls each paint half makes."""

    def __init__(self):
        self.calls = []

    def visualize_displacement_preview(self, field, *args, **kwargs):
        self.calls.append(("displacement", field.shape, args, kwargs))

    def visualize_force_preview(self, field, *args, **kwargs):
        self.calls.append(("force", field.shape, args, kwargs))

    def visualize_stress_preview(self, tensor, **kwargs):
        self.calls.append(("stress", np.shape(tensor), kwargs))

    def bring_layers_to_front(self, layers):
        self.calls.append(("front", tuple(layers)))


class _Result:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_controller(cls, viz):
    return cls(viewer=object(), data_manager=object(),
               parameter_manager=object(), visualization_manager=viz)


# --- the shared plumbing --------------------------------------------------

def test_start_preview_worker_wires_run_plumbing(app):
    from napariTFM.widgets.stress_widget import StressController

    ctrl = _make_controller(StressController, _RecordingViz())
    frozen = []
    ctrl.ui_frozen.connect(frozen.append)

    worker = _FakeWorker()
    seen = []
    ctrl._start_preview_worker(worker, seen.append, status="Working...")

    # Froze the UI, registered for cancel(), and started — in that order.
    assert frozen == [True]
    assert worker in ctrl.active_workers
    assert worker.started

    # The result reaches the main-thread paint callback via `returned`.
    worker.returned.emit("RESULT")
    assert seen == ["RESULT"]

    # `finished` forgets the worker and unfreezes once none remain.
    worker.finished.emit()
    assert ctrl.active_workers == []
    assert worker not in ctrl._preview_returned_slots
    assert frozen == [True, False]


def test_start_preview_worker_surfaces_paint_error(app, monkeypatch):
    from napariTFM.widgets.stress_widget import StressController

    ctrl = _make_controller(StressController, _RecordingViz())
    errors = []
    monkeypatch.setattr(ctrl, "_handle_error", errors.append)

    worker = _FakeWorker()

    def _boom(_result):
        raise ValueError("paint blew up")

    ctrl._start_preview_worker(worker, _boom, status="x")
    worker.returned.emit("R")  # a throwing paint must be caught, not crash the thread

    assert errors and "paint blew up" in errors[0]
    # Unfreeze still runs on the terminal path.
    worker.finished.emit()
    assert ctrl.active_workers == []


def test_start_preview_worker_routes_compute_error(app, monkeypatch):
    from napariTFM.widgets.stress_widget import StressController

    ctrl = _make_controller(StressController, _RecordingViz())
    errors = []
    monkeypatch.setattr(ctrl, "_handle_error", errors.append)

    worker = _FakeWorker()
    ctrl._start_preview_worker(worker, lambda r: None, status="x")
    worker.errored.emit(RuntimeError("solve failed"))

    assert errors and "solve failed" in errors[0]


def test_start_preview_worker_paints_before_completion(app):
    from napariTFM.widgets.stress_widget import StressController

    ctrl = _make_controller(StressController, _RecordingViz())
    worker = _FakeWorker()
    calls = []

    ctrl._start_preview_worker(
        worker,
        lambda result: calls.append(("paint", result)),
        completion=lambda result: calls.append(("complete", result)),
    )
    worker.returned.emit("R")

    assert calls == [("paint", "R"), ("complete", "R")]


def test_start_preview_worker_routes_completion_error(app, monkeypatch):
    from napariTFM.widgets.stress_widget import StressController

    ctrl = _make_controller(StressController, _RecordingViz())
    errors = []
    monkeypatch.setattr(ctrl, "_handle_error", errors.append)
    worker = _FakeWorker()

    def _completion_boom(_result):
        raise ValueError("completion blew up")

    ctrl._start_preview_worker(worker, lambda _result: None,
                               completion=_completion_boom)
    worker.returned.emit("R")

    assert errors == ["completion blew up"]


def test_cancelled_preview_ignores_late_result_and_completion(app):
    from napariTFM.widgets.stress_widget import StressController

    ctrl = _make_controller(StressController, _RecordingViz())
    calls = []
    worker = _FakeWorker()
    ctrl._start_preview_worker(
        worker,
        lambda result: calls.append(("paint", result)),
        completion=lambda result: calls.append(("complete", result)),
    )

    ctrl.cancel()
    worker.returned.emit("LATE")

    assert calls == []


def test_start_preview_worker_skips_completion_after_paint_error(app, monkeypatch):
    from napariTFM.widgets.stress_widget import StressController

    ctrl = _make_controller(StressController, _RecordingViz())
    errors = []
    completions = []
    monkeypatch.setattr(ctrl, "_handle_error", errors.append)

    worker = _FakeWorker()

    def _boom(_result):
        raise ValueError("paint blew up")

    ctrl._start_preview_worker(worker, _boom, completion=completions.append)
    worker.returned.emit("R")

    assert errors == ["paint blew up"]
    assert completions == []


def test_force_preview_accepts_transient_displacement_without_storing_it(app, monkeypatch):
    from napariTFM.widgets.fttc_widget import FTTCController

    displacement = _Result(displacement_field=np.ones((1, 4, 4, 2)))
    data_manager = _Result(displacement_results=None, mask_stack=None)
    params = _Result(fwd_mask_strength=0)
    ctrl = FTTCController(
        viewer=object(), data_manager=data_manager,
        parameter_manager=_Result(get_fttc_parameters=lambda: params),
        visualization_manager=_RecordingViz(),
    )
    monkeypatch.setattr(ctrl, "_current_frame", lambda: 1)
    captured = []
    worker = _FakeWorker()
    monkeypatch.setattr(
        ctrl, "_preview_worker",
        lambda field, actual_params, frame: captured.append(
            (field, actual_params, frame)
        ) or worker,
    )

    ctrl.preview_force(displacement_result=displacement)

    assert np.array_equal(captured[0][0], displacement.displacement_field[0])
    assert captured[0][1] is params
    assert captured[0][2] == 1
    assert data_manager.displacement_results is None


def test_stress_preview_accepts_transient_force_without_storing_it(app, monkeypatch):
    from napariTFM.widgets.stress_widget import StressController

    force = _Result(
        force_field=np.ones((1, 4, 4, 2)),
        parameters=_Result(downscale_factor=3),
    )
    data_manager = _Result(mask_stack=np.ones((2, 4, 4)), force_results=None)
    params = _Result(max_stress=100.0)
    ctrl = StressController(
        viewer=object(), data_manager=data_manager,
        parameter_manager=_Result(get_stress_parameters=lambda: params),
        visualization_manager=_RecordingViz(),
    )
    monkeypatch.setattr(ctrl, "_current_frame", lambda: 1)
    captured = []
    worker = _FakeWorker()
    monkeypatch.setattr(
        ctrl, "_preview_worker",
        lambda field, mask, actual_params: captured.append(
            (field, mask, actual_params)
        ) or worker,
    )

    ctrl.preview_current_frame(force_result=force)

    assert np.array_equal(captured[0][0], force.force_field[0])
    assert np.array_equal(captured[0][1], data_manager.mask_stack[1])
    assert captured[0][2] is params
    assert data_manager.force_results is None


# --- the paint halves (regression guard on the refactor) ------------------

def test_show_displacement_preview_paints(app):
    from napariTFM.widgets.displacement_analysis_widget import DisplacementController

    viz = _RecordingViz()
    ctrl = _make_controller(DisplacementController, viz)
    params = _Result(d_max=5.0, disp_vector_stride=8, disp_arrow_scale=1.0,
                     downscale_factor=1)
    field = np.ones((1, 4, 4, 2), dtype=np.float32)
    progress = []
    ctrl.progress_updated.connect(lambda *a: progress.append(a))

    ctrl._show_displacement_preview(_Result(displacement_field=field), params)

    kinds = [c[0] for c in viz.calls]
    assert kinds == ["displacement", "front"]
    assert viz.calls[0][1] == (4, 4, 2)  # single frame handed to the viz
    assert viz.calls[1][1] == (('Displacement Magnitude', True),
                               ('Displacement Vectors', True))
    assert progress[-1][0] == 100


def test_show_force_preview_paints(app):
    from napariTFM.widgets.fttc_widget import FTTCController

    viz = _RecordingViz()
    ctrl = _make_controller(FTTCController, viz)
    params = _Result(f_max=20.0, force_vector_stride=8, force_arrow_scale=1.0,
                     downscale_factor=1)
    field = np.ones((1, 4, 4, 2), dtype=np.float32)
    progress = []
    ctrl.progress_updated.connect(lambda *a: progress.append(a))

    ctrl._show_force_preview(_Result(force_field=field, parameters=params))

    kinds = [c[0] for c in viz.calls]
    assert kinds == ["force", "front"]
    assert viz.calls[0][1] == (4, 4, 2)
    assert viz.calls[1][1] == (('Force Magnitude', True), ('Force Vectors', True))
    assert progress[-1][0] == 100


def test_show_stress_preview_paints(app):
    from napariTFM.widgets.stress_widget import StressController

    viz = _RecordingViz()
    ctrl = _make_controller(StressController, viz)
    params = _Result(max_stress=100.0)
    tensor = np.ones((4, 4), dtype=np.float32)
    progress = []
    ctrl.progress_updated.connect(lambda *a: progress.append(a))

    ctrl._show_stress_preview(_Result(stress_tensor=tensor, r2_traction=0.9),
                              params, current_frame=0, downscale=1)

    kinds = [c[0] for c in viz.calls]
    assert kinds == ["stress", "front"]
    assert viz.calls[1][1] == (('Normal Stress XX', False),
                               ('Normal Stress YY', False),
                               ('Average Normal Stress', True))
    assert progress[-1][0] == 100
