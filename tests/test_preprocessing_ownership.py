import pytest
from qtpy.QtWidgets import QApplication

import napariTFM.widgets.preprocessing_widget as pw


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeDims:
    class _Events:
        class _Step:
            def connect(self, *_):
                return None
        current_step = _Step()
    events = _Events()
    current_step = (0,)


class _FakeLayersSelection:
    active = None

    class _Events:
        class _Active:
            def connect(self, *_):
                return None
        active = _Active()
    events = _Events()


class _FakeLayers:
    selection = _FakeLayersSelection()

    def __iter__(self):
        return iter(())


class _FakeViewer:
    dims = _FakeDims()
    layers = _FakeLayers()


class _FakeDataManager:
    bead_stack = None
    reference = None
    cell_stack = None
    preprocessed_bead_stack = None
    preprocessed_reference = None
    preprocessed_cell_stack = None


class _FakeParameterManager:
    def __init__(self):
        from qtpy.QtCore import QObject, Signal

        class _PM(QObject):
            parameter_changed = Signal(str, object)
            parameters_reset = Signal(object)

        self._pm = _PM()
        self.parameter_changed = self._pm.parameter_changed
        self.parameters_reset = self._pm.parameters_reset

    def get_preprocessing_parameters(self):
        return object()


@pytest.fixture
def preprocessing_widget(app):
    return pw.PreprocessingWidget(
        _FakeViewer(),
        _FakeDataManager(),
        _FakeParameterManager(),
        object(),
    )


def test_preprocessing_exposes_action_contract(app, preprocessing_widget):
    w = preprocessing_widget
    assert hasattr(w, "action_states_changed")
    states = w.action_states()
    assert set(states) >= {"run", "preview", "cancel"}
    assert callable(w.run_action)
    assert callable(w.preview_action)
    assert callable(w.cancel_action)


def test_load_active_layer_delegates_to_controller(app, preprocessing_widget):
    # The file-status dot row (shell) calls widget.load_active_layer(role) to
    # assign the active napari layer; it must delegate to the controller (which
    # owns the real implementation), like the displacement widget does.
    calls = []
    preprocessing_widget.controller.load_active_layer = lambda role: calls.append(role)

    preprocessing_widget.load_active_layer("reference")

    assert calls == ["reference"]


def test_no_per_stage_status_label(app, preprocessing_widget):
    # P2: the shell's one global status label replaces per-stage labels.
    assert not hasattr(preprocessing_widget, "status_label")


def test_parameter_panel_class_is_removed():
    assert not hasattr(pw, "PreprocessingParameterPanel")


def test_data_panel_class_is_removed():
    assert not hasattr(pw, "PreprocessingDataPanel")


def test_controller_has_no_panel_attributes(app):
    controller = pw.PreprocessingController(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )
    assert not hasattr(controller, "parameter_panel")
    assert not hasattr(controller, "data_panel")
    assert not hasattr(controller, "set_panels")


def test_controller_freeze_emits_signal_without_panels(app):
    controller = pw.PreprocessingController(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )
    seen = []
    controller.ui_frozen.connect(seen.append)
    controller.freeze_ui()
    controller.unfreeze_ui()
    assert seen == [True, False]


class _RecordingDataManager:
    """Captures what the controller stores, so disk-loading is observable."""

    def __init__(self):
        self.bead_stack = None
        self.reference = None
        self.cell_stack = None
        self.preprocessed_bead_stack = None
        self.preprocessed_reference = None
        self.preprocessed_cell_stack = None

    def set_bead_stack(self, data, path=None, source=""):
        self.bead_stack = data

    def set_reference(self, data, path=None, source=""):
        self.reference = data

    def set_cell_stack(self, data, path=None, source=""):
        self.cell_stack = data


def _controller_with(dm):
    return pw.PreprocessingController(
        viewer=_FakeViewer(),
        data_manager=dm,
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )


def test_load_input_files_reads_discovery_files_into_memory(app, tmp_path):
    # Selecting an experiment must put its raw inputs in memory — that is what
    # enables Preview and Run, which both require the loaded arrays. The
    # discovery config names the files; the controller reads them from disk.
    import numpy as np
    import tifffile

    tifffile.imwrite(str(tmp_path / "r.tif"), np.zeros((4, 4), dtype=np.float32))
    tifffile.imwrite(str(tmp_path / "b.tif"), np.zeros((3, 4, 4), dtype=np.float32))
    dm = _RecordingDataManager()
    controller = _controller_with(dm)
    seen = []
    controller.data_updated.connect(seen.append)

    controller.load_input_files(str(tmp_path), {"beads": "b.tif", "reference": "r.tif"})

    assert dm.reference is not None and dm.reference.shape == (4, 4)
    assert dm.bead_stack is not None and dm.bead_stack.shape == (3, 4, 4)
    # data_updated fires per input so the shell re-enables Run/Preview.
    assert set(seen) >= {"beads", "reference"}


def test_load_input_files_promotes_2d_beads_to_a_stack(app, tmp_path):
    # A single-frame bead image on disk is 2D; the stack contract is 3D.
    import numpy as np
    import tifffile

    tifffile.imwrite(str(tmp_path / "b.tif"), np.zeros((4, 4), dtype=np.float32))
    dm = _RecordingDataManager()
    controller = _controller_with(dm)

    controller.load_input_files(str(tmp_path), {"beads": "b.tif"})

    assert dm.bead_stack is not None and dm.bead_stack.shape == (1, 4, 4)


def test_load_input_files_skips_missing_and_unnamed_inputs(app, tmp_path):
    # Only the reference is on disk; the un-named cells slot and the missing
    # bead file are simply skipped — no crash, no partial garbage.
    import numpy as np
    import tifffile

    tifffile.imwrite(str(tmp_path / "r.tif"), np.zeros((4, 4), dtype=np.float32))
    dm = _RecordingDataManager()
    controller = _controller_with(dm)

    controller.load_input_files(
        str(tmp_path), {"beads": "b.tif", "reference": "r.tif", "cells": ""}
    )

    assert dm.reference is not None
    assert dm.bead_stack is None
    assert dm.cell_stack is None


def test_load_input_files_noop_without_folder(app):
    dm = _RecordingDataManager()
    controller = _controller_with(dm)
    controller.load_input_files(None, {"beads": "b.tif"})  # must not raise
    assert dm.bead_stack is None


def test_parameter_changes_debounce_the_preview(app):
    # Sliders emit valueChanged continuously while dragging. Recomputing the
    # (expensive rolling-ball) preview synchronously on each event freezes the
    # UI. A burst of changes must coalesce into a single deferred recompute.
    controller = _controller_with(_FakeDataManager())
    controller.preview_enabled = True
    calls = []
    controller._update_preview = lambda: calls.append(1)

    for _ in range(5):
        controller._on_parameter_changed("gaussian_sigma", 1.0)

    # Nothing recomputes synchronously during the burst...
    assert calls == []
    assert controller._preview_timer.isActive()

    # ...and when the debounce settles, exactly one recompute fires.
    controller._preview_timer.timeout.emit()
    assert calls == [1]


def test_parameter_changes_skip_preview_when_disabled(app):
    controller = _controller_with(_FakeDataManager())
    controller.preview_enabled = False

    controller._on_parameter_changed("gaussian_sigma", 1.0)

    assert not controller._preview_timer.isActive()
