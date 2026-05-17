import sys
import types

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QTabWidget, QWidget


class _StubParameterManager(QObject):
    parameter_changed = Signal(str, object)

    def __init__(self):
        super().__init__()
        self._values = {
            "pixel_size": 1.0,
            "frame_interval": 1.0,
        }
        self._callbacks = {}

    def register_callback(self, name, callback):
        self._callbacks[name] = callback

    def get_parameter(self, name):
        return self._values[name]

    def set_parameter(self, name, value):
        self._values[name] = value
        self.parameter_changed.emit(name, value)


class _StubDataManager:
    pass


class _StubVisualizationManager:
    def __init__(self, viewer, data_manager):
        self.viewer = viewer
        self.data_manager = data_manager


class _StubStageWidget(QWidget):
    preprocessing_completed = Signal(object)
    displacement_calculated = Signal(object)
    force_calculated = Signal(object)
    stress_calculated = Signal(object)

    def __init__(self, *args):
        super().__init__()


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module


_stub_module(
    "napariTFM.utilities.parameter_manager",
    ParameterManager=_StubParameterManager,
)
_stub_module("napariTFM.utilities.data_manager", DataManager=_StubDataManager)
_stub_module(
    "napariTFM.utilities.visualization_manager",
    VisualizationManager=_StubVisualizationManager,
)
_stub_module("napariTFM.widgets.preprocessing_widget", PreprocessingWidget=_StubStageWidget)
_stub_module(
    "napariTFM.widgets.displacement_analysis_widget",
    DisplacementAnalysisWidget=_StubStageWidget,
)
_stub_module("napariTFM.widgets.fttc_widget", FTTCWidget=_StubStageWidget)
_stub_module("napariTFM.widgets.msm_widget", MSMWidget=_StubStageWidget)
_stub_module(
    "napariTFM.widgets.batch_analysis_widget",
    BatchAnalysisWidget=_StubStageWidget,
)

from napariTFM.widgets import _widget


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_stage_section_toggles_child_without_destroying_it(app):
    child = QWidget()

    section = _widget._StageSection("Preprocessing", child, expanded=True)
    section.show()
    app.processEvents()

    assert child.isVisible()
    assert section._content.isVisible()

    section._toggle_button.setChecked(False)
    app.processEvents()

    assert child.parent() is section._content
    assert not child.isVisible()
    assert not section._content.isVisible()

    section._toggle_button.setChecked(True)
    app.processEvents()

    assert child.parent() is section._content
    assert child.isVisible()
    assert section._content.isVisible()


def test_main_widget_uses_stage_sections_instead_of_tabs(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.show()
    app.processEvents()

    assert widget.findChildren(QTabWidget) == []
    assert widget.preprocessing_widget.isVisible()
    assert not widget.displacement_widget.isVisible()
    assert not widget.force_widget.isVisible()
    assert not widget.msm_widget.isVisible()
    assert not widget.batch_widget.isVisible()
