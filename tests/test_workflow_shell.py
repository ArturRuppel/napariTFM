import sys
import types

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QPushButton, QTabWidget, QWidget


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
        self.preview_btn = QPushButton("Preview")
        self.process_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")


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


def test_stage_section_exposes_header_actions_with_stable_names(app):
    child = _StubStageWidget()

    section = _widget._StageSection(
        "Preprocessing",
        child,
        action_targets={
            "run": child.process_btn,
            "preview": child.preview_btn,
            "cancel": child.cancel_btn,
        },
        expanded=False,
    )

    assert section.run_button.objectName() == "stage_preprocessing_run_button"
    assert section.preview_button.objectName() == "stage_preprocessing_preview_button"
    assert section.cancel_button.objectName() == "stage_preprocessing_cancel_button"
    assert section.config_button.objectName() == "stage_preprocessing_config_button"

    assert section.run_button.toolTip() == "Run Preprocessing"
    assert section.preview_button.toolTip() == "Preview Preprocessing"
    assert section.cancel_button.toolTip() == "Cancel Preprocessing"
    assert section.config_button.toolTip() == "Configure Preprocessing"


def test_stage_section_header_actions_proxy_child_buttons(app):
    child = _StubStageWidget()
    clicks = {"run": 0, "preview": 0, "cancel": 0}
    child.process_btn.clicked.connect(lambda: clicks.__setitem__("run", clicks["run"] + 1))
    child.preview_btn.clicked.connect(lambda: clicks.__setitem__("preview", clicks["preview"] + 1))
    child.cancel_btn.clicked.connect(lambda: clicks.__setitem__("cancel", clicks["cancel"] + 1))

    section = _widget._StageSection(
        "Preprocessing",
        child,
        action_targets={
            "run": child.process_btn,
            "preview": child.preview_btn,
            "cancel": child.cancel_btn,
        },
        expanded=False,
    )

    section.run_button.click()
    section.preview_button.click()
    section.cancel_button.click()

    assert clicks == {"run": 1, "preview": 1, "cancel": 1}


def test_stage_section_disables_unsupported_actions_and_config_toggles(app):
    child = QWidget()

    section = _widget._StageSection("Batch Analysis", child, expanded=False)
    section.show()
    app.processEvents()

    assert not section.run_button.isEnabled()
    assert not section.preview_button.isEnabled()
    assert not section.cancel_button.isEnabled()

    section.config_button.click()
    app.processEvents()

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


def test_main_widget_stage_headers_wire_existing_stage_actions(monkeypatch, app):
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

    displacement_section = widget._stage_sections_by_key["displacement"]
    clicks = {"run": 0}
    widget.displacement_widget.process_btn.clicked.connect(
        lambda: clicks.__setitem__("run", clicks["run"] + 1)
    )

    displacement_section.run_button.click()

    assert clicks["run"] == 1
