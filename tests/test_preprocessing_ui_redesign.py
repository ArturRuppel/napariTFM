import sys
import types

import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QPushButton, QScrollArea, QWidget

for module_name in [
    "napariTFM.utilities.data_manager",
    "napariTFM.utilities.parameter_manager",
    "napariTFM.utilities.visualization_manager",
    "napariTFM.widgets.preprocessing_widget",
]:
    sys.modules.pop(module_name, None)

widgets_package = sys.modules.get("napariTFM.widgets")
if widgets_package is not None and hasattr(widgets_package, "preprocessing_widget"):
    delattr(widgets_package, "preprocessing_widget")


class DataManager:
    def __init__(self):
        self.bead_stack = None
        self.reference = None
        self.cell_stack = None
        self.preprocessed_bead_stack = None
        self.preprocessed_reference = None
        self.preprocessed_cell_stack = None

    def set_bead_stack(self, data):
        self.bead_stack = data

    def set_reference(self, data):
        self.reference = data

    def set_cell_stack(self, data):
        self.cell_stack = data

    def set_preprocessed_bead_stack(self, data):
        self.preprocessed_bead_stack = data

    def set_preprocessed_reference(self, data):
        self.preprocessed_reference = data

    def set_preprocessed_cell_stack(self, data):
        self.preprocessed_cell_stack = data


data_manager_module = types.ModuleType("napariTFM.utilities.data_manager")
data_manager_module.DataManager = DataManager
sys.modules["napariTFM.utilities.data_manager"] = data_manager_module


class _RangeSlider(QWidget):
    valueChanged = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._value = (0, 1000)

    def setMinimum(self, value):
        pass

    def setMaximum(self, value):
        pass

    def setRange(self, minimum, maximum):
        pass

    def setSingleStep(self, value):
        pass

    def setPageStep(self, value):
        pass

    def setValue(self, value):
        self._value = value
        self.valueChanged.emit(value)

    def value(self):
        return self._value


qtrangeslider_module = types.ModuleType("qtrangeslider")
qtrangeslider_module.QRangeSlider = _RangeSlider
sys.modules["qtrangeslider"] = qtrangeslider_module

for module_name in [
    "gmsh",
    "solidspy",
    "solidspy.assemutil",
    "solidspy.postprocesor",
    "solidspy.solutil",
]:
    sys.modules[module_name] = types.ModuleType(module_name)

from napariTFM.utilities.visualization_manager import VisualizationManager
import napariTFM.widgets.preprocessing_widget as preprocessing_widget
from napariTFM.widgets.preprocessing_widget import PreprocessingWidget


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeSelectionEvents:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _FakeSelection:
    def __init__(self):
        self.active = None
        self.events = type("Events", (), {"active": _FakeSelectionEvents()})()


class _FakeLayer:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.visible = True
        self.colormap = None
        self.blending = None


class _FakeLayers(list):
    def __init__(self):
        super().__init__()
        self.selection = _FakeSelection()
        self.events = type("Events", (), {"removed": _FakeSelectionEvents()})()

    def __contains__(self, item):
        if isinstance(item, str):
            return any(layer.name == item for layer in self)
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)

    def remove(self, item):
        if isinstance(item, str):
            item = self[item]
        super().remove(item)


class _FakeDimsEvents:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        if callback in self.callbacks:
            self.callbacks.remove(callback)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayers()
        self.dims = type(
            "Dims",
            (),
            {"current_step": (0,), "events": type("Events", (), {"current_step": _FakeDimsEvents()})()},
        )()

    def add_image(self, data, **kwargs):
        layer = _FakeLayer(kwargs["name"], data)
        layer.colormap = kwargs.get("colormap")
        layer.blending = kwargs.get("blending")
        layer.visible = kwargs.get("visible", True)
        self.layers.append(layer)
        return layer


class _ParameterManager(QObject):
    parameter_changed = Signal(str, object)
    parameters_reset = Signal(object)

    def __init__(self):
        super().__init__()
        from napariTFM.backend.parameter_dataclasses import PreprocessingParameters

        self._params = PreprocessingParameters(
            rolling_ball_radius=0,
            min_intensity_percentile=0,
            max_intensity_percentile=100,
            gaussian_sigma=0,
            cell_min_intensity_percentile=0,
            cell_max_intensity_percentile=100,
            cell_gaussian_sigma=0,
            registration_mode="no registration",
        )
        self._values = self._params.__dict__.copy()
        self._values.update({"pixel_size": 1.0, "frame_interval": 1.0})

    def get_parameter(self, name):
        return self._values[name]

    def set_parameter(self, name, value):
        self._values[name] = value
        if hasattr(self._params, name):
            setattr(self._params, name, value)
        self.parameter_changed.emit(name, value)

    def get_preprocessing_parameters(self):
        return self._params

    def reset_preprocessing_parameters(self):
        self.parameters_reset.emit(object())


class _FakeVisualizationManager:
    def __init__(self):
        self.preview_calls = []

    def handle_preprocessing_preview(self, frames, enable=True):
        self.preview_calls.append((frames, enable))

    def update_preprocessing_visualization(self):
        pass

    def cleanup(self):
        pass


def test_preprocessing_widget_does_not_mount_stage_local_data_panel(app):
    widget = PreprocessingWidget(
        _FakeViewer(),
        DataManager(),
        _ParameterManager(),
        _FakeVisualizationManager(),
    )
    widget.show()
    app.processEvents()

    assert not hasattr(widget, "data_panel")
    assert widget.findChild(QPushButton, "preprocessing_reference_assign_button") is None
    assert widget.findChild(QPushButton, "preprocessing_beads_assign_button") is None
    assert widget.findChild(QPushButton, "preprocessing_cells_assign_button") is None


def test_preprocessing_widget_hides_large_body_actions_and_radio_buttons(app):
    widget = PreprocessingWidget(
        _FakeViewer(),
        DataManager(),
        _ParameterManager(),
        _FakeVisualizationManager(),
    )
    widget.show()
    app.processEvents()

    assert not widget.action_frame.isVisibleTo(widget)
    assert not widget.preview_frame.isVisibleTo(widget)
    assert not hasattr(widget, "bead_radio")
    assert not hasattr(widget, "reference_radio")
    assert not hasattr(widget, "cell_radio")


def test_preprocessing_widget_has_no_inner_scroll_area(app):
    widget = PreprocessingWidget(
        _FakeViewer(),
        DataManager(),
        _ParameterManager(),
        _FakeVisualizationManager(),
    )
    widget.resize(360, 220)
    widget.show()
    app.processEvents()

    # The stage widget no longer owns a scroll area or a fixed width; the
    # shell's single scroll area owns layout, so the body reflows to the dock.
    assert widget.findChild(QScrollArea) is None
    assert not hasattr(widget, "data_panel")


def test_preprocessing_preview_renders_all_loaded_inputs(monkeypatch, app):
    data_manager = DataManager()
    data_manager.set_bead_stack(np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32))
    data_manager.set_reference(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    data_manager.set_cell_stack(np.array([[[4.0, 3.0], [2.0, 1.0]]], dtype=np.float32))
    visualization = _FakeVisualizationManager()
    widget = PreprocessingWidget(
        _FakeViewer(),
        data_manager,
        _ParameterManager(),
        visualization,
    )

    widget.preview_check.setChecked(True)

    frames, enabled = visualization.preview_calls[-1]
    assert enabled is True
    assert set(frames) == {"beads", "reference", "cells"}


def test_preprocessing_run_enablement_tracks_required_inputs(app):
    data_manager = DataManager()
    widget = PreprocessingWidget(
        _FakeViewer(),
        data_manager,
        _ParameterManager(),
        _FakeVisualizationManager(),
    )

    assert not widget.process_btn.isEnabled()
    assert not hasattr(widget, "save_btn")

    data_manager.set_bead_stack(np.ones((1, 2, 2), dtype=np.float32))
    widget._update_ui_state()
    assert not widget.process_btn.isEnabled()

    data_manager.set_reference(np.ones((2, 2), dtype=np.float32))
    widget._update_ui_state()
    assert widget.process_btn.isEnabled()

    data_manager.set_preprocessed_reference(np.ones((2, 2), dtype=np.float32))
    widget._update_ui_state()
    assert widget.process_btn.isEnabled()


def test_preprocessing_preview_error_unchecks_widget_preview_control(monkeypatch, app):
    data_manager = DataManager()
    data_manager.set_reference(np.ones((2, 2), dtype=np.float32))
    widget = PreprocessingWidget(
        _FakeViewer(),
        data_manager,
        _ParameterManager(),
        _FakeVisualizationManager(),
    )
    monkeypatch.setattr(
        preprocessing_widget,
        "preprocess_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        preprocessing_widget.QMessageBox,
        "warning",
        lambda *args, **kwargs: None,
    )

    widget.preview_check.setChecked(True)

    assert not widget.preview_check.isChecked()
    assert widget.controller.preview_enabled is False


def test_visualization_manager_uses_additive_preprocessing_layers_without_overlay(monkeypatch):
    viewer = _FakeViewer()
    data_manager = DataManager()
    data_manager.set_preprocessed_bead_stack(np.ones((1, 2, 2), dtype=np.float32))
    data_manager.set_preprocessed_reference(np.ones((2, 2), dtype=np.float32))
    data_manager.set_preprocessed_cell_stack(np.ones((1, 2, 2), dtype=np.float32))
    manager = VisualizationManager(viewer, data_manager)
    monkeypatch.setattr(manager.colorbar_manager, "clear", lambda: None)

    manager.update_preprocessing_visualization()

    names = [layer.name for layer in viewer.layers]
    assert "Bead Overlay" not in names
    assert names == ["Preprocessed Cells", "Preprocessed Reference", "Preprocessed Beads"]
    assert all(layer.blending == "additive" for layer in viewer.layers)


def test_visualization_manager_preprocessing_preview_layers_are_separate():
    viewer = _FakeViewer()
    manager = VisualizationManager(viewer, DataManager())

    manager.handle_preprocessing_preview(
        {
            "beads": np.ones((2, 2), dtype=np.float32),
            "reference": np.ones((2, 2), dtype=np.float32),
            "cells": np.ones((2, 2), dtype=np.float32),
        },
        enable=True,
    )

    names = [layer.name for layer in viewer.layers]
    assert names == ["Preview Cells", "Preview Reference", "Preview Beads"]
    assert all(layer.blending == "additive" for layer in viewer.layers)

    manager.handle_preprocessing_preview({}, enable=False)

    assert viewer.layers == []
