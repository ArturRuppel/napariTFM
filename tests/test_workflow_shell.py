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
            "rolling_ball_radius": 0,
            "min_intensity_percentile": 0.0,
            "max_intensity_percentile": 100.0,
            "gaussian_sigma": 0.0,
            "cell_min_intensity_percentile": 0.0,
            "cell_max_intensity_percentile": 100.0,
            "cell_gaussian_sigma": 0.0,
            "registration_mode": "translation",
            "nscales": 3,
            "inner_iterations": 15,
            "outer_iterations": 5,
            "median_filtering": 9,
            "downscale_factor": 4,
            "disp_vector_stride": 20,
            "disp_arrow_scale": 1.0,
            "d_max": 1.0,
            "young_modulus": 5.0,
            "poisson_ratio_substrate": 0.5,
            "gel_height": 0.0,
            "lanczos_exp": 1,
            "regularization": -4.0,
            "auto_gcv": False,
            "force_vector_stride": 20,
            "force_arrow_scale": 1.0,
            "f_max": 500.0,
            "threshold": 0.0,
            "dilation": 10,
            "smoothing_sigma": 10.0,
            "density_factor": 0.01,
            "mesh_algorithm": "Frontal-Del.",
            "use_optimization": True,
            "poisson_ratio_cells": 0.5,
            "max_stress": 1.0,
        }
        self._callbacks = {}
        self.ui_writes = []

    def register_callback(self, name, callback):
        self._callbacks[name] = callback

    def get_parameter(self, name):
        return self._values[name]

    def set_parameter(self, name, value):
        self._values[name] = value
        self.parameter_changed.emit(name, value)

    def get_ui_parameter(self, name):
        return self.get_parameter(name)

    def set_ui_parameter(self, name, value):
        self.ui_writes.append((name, value))
        self.set_parameter(name, value)


class _StubDataManager:
    def __init__(self):
        self.bead_stack = None
        self.reference = None
        self.cell_stack = None
        self.preprocessed_bead_stack = None
        self.preprocessed_reference = None
        self.preprocessed_cell_stack = None
        self.mask_stack = None
        self.displacement_results = None
        self.force_results = None
        self.stress_results = None


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
        self.parameter_panel = QWidget()
        self.preview_btn = QPushButton("Preview")
        self.process_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save")


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
            "save": child.save_btn,
        },
        expanded=False,
    )

    assert section.run_button.objectName() == "stage_preprocessing_run_button"
    assert section.preview_button.objectName() == "stage_preprocessing_preview_button"
    assert section.cancel_button.objectName() == "stage_preprocessing_cancel_button"
    assert section.save_button.objectName() == "stage_preprocessing_save_button"
    assert section.config_button.objectName() == "stage_preprocessing_config_button"

    assert section.run_button.toolTip() == "Run Preprocessing"
    assert section.preview_button.toolTip() == "Preview Preprocessing"
    assert section.cancel_button.toolTip() == "Cancel Preprocessing"
    assert section.save_button.toolTip() == "Save Preprocessing"
    assert section.config_button.toolTip() == "Configure Preprocessing"


def test_stage_section_exposes_status_indicator_with_stable_name(app):
    child = QWidget()

    section = _widget._StageSection("Preprocessing", child, status="ready")

    assert section.status_indicator.objectName() == "stage_preprocessing_status_indicator"
    assert section.status_indicator.toolTip() == "Preprocessing status: ready"

    section.set_status("done")

    assert section.status_indicator.toolTip() == "Preprocessing status: done"


def test_stage_section_applies_stage_accent_to_header(app):
    child = QWidget()

    section = _widget._StageSection("Traction / FTTC", child, accent="#2a9d8f")

    assert "#2a9d8f" in section.header_label.styleSheet()


def test_stage_section_header_action_state_follows_child_button(app):
    child = _StubStageWidget()
    child.process_btn.setEnabled(False)

    section = _widget._StageSection(
        "Preprocessing",
        child,
        action_targets={"run": child.process_btn},
    )

    assert not section.run_button.isEnabled()

    child.process_btn.setEnabled(True)
    app.processEvents()

    assert section.run_button.isEnabled()


def test_stage_section_status_indicator_remains_visible_when_collapsed(app):
    child = QWidget()

    section = _widget._StageSection("Preprocessing", child, expanded=True, status="ready")
    section.show()
    app.processEvents()

    section.config_button.click()
    app.processEvents()

    assert not section._content.isVisible()
    assert section.status_indicator.isVisible()


def test_stage_section_header_actions_proxy_child_buttons(app):
    child = _StubStageWidget()
    clicks = {"run": 0, "preview": 0, "cancel": 0, "save": 0}
    child.process_btn.clicked.connect(lambda: clicks.__setitem__("run", clicks["run"] + 1))
    child.preview_btn.clicked.connect(lambda: clicks.__setitem__("preview", clicks["preview"] + 1))
    child.cancel_btn.clicked.connect(lambda: clicks.__setitem__("cancel", clicks["cancel"] + 1))
    child.save_btn.clicked.connect(lambda: clicks.__setitem__("save", clicks["save"] + 1))

    section = _widget._StageSection(
        "Preprocessing",
        child,
        action_targets={
            "run": child.process_btn,
            "preview": child.preview_btn,
            "cancel": child.cancel_btn,
            "save": child.save_btn,
        },
        expanded=False,
    )

    section.run_button.click()
    section.preview_button.click()
    section.cancel_button.click()
    section.save_button.click()

    assert clicks == {"run": 1, "preview": 1, "cancel": 1, "save": 1}


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


def test_stage_section_config_toggles_inline_parameter_panel_when_provided(app):
    child = QWidget()
    parameter_panel = QWidget()

    section = _widget._StageSection(
        "Displacement",
        child,
        parameter_panel=parameter_panel,
        expanded=False,
    )
    section.show()
    app.processEvents()

    assert not child.isVisible()
    assert not section._content.isVisible()
    assert not parameter_panel.isVisible()
    assert not section._parameter_content.isVisible()

    section.config_button.click()
    app.processEvents()

    assert not child.isVisible()
    assert not section._content.isVisible()
    assert parameter_panel.isVisible()
    assert section._parameter_content.isVisible()


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


def test_workflow_parameter_panel_exposes_one_control_per_managed_parameter(app):
    manager = _StubParameterManager()

    panel = _widget.WorkflowParameterPanel(manager)

    for name in [
        "pixel_size",
        "rolling_ball_radius",
        "nscales",
        "young_modulus",
        "auto_gcv",
        "mesh_algorithm",
    ]:
        assert name in panel.parameter_controls
        assert panel.parameter_controls[name].objectName() == f"workflow_parameter_{name}"

    assert "outer_iterations" not in panel.parameter_controls


def test_workflow_parameter_panel_writes_through_ui_parameter_api(app):
    manager = _StubParameterManager()
    panel = _widget.WorkflowParameterPanel(manager)

    panel.parameter_controls["young_modulus"].setValue(8.5)

    assert manager.ui_writes[-1] == ("young_modulus", 8.5)


def test_workflow_parameter_panel_syncs_from_parameter_manager(app):
    manager = _StubParameterManager()
    panel = _widget.WorkflowParameterPanel(manager)

    manager.set_parameter("nscales", 6)

    assert panel.parameter_controls["nscales"].value() == 6


def test_main_widget_hides_stage_parameter_panels(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    assert isinstance(widget.parameter_panel, _widget.WorkflowParameterPanel)
    assert not widget.parameter_panel.isVisibleTo(widget)
    assert not widget.displacement_widget.parameter_panel.isVisibleTo(widget)


def test_main_widget_groups_parameters_inline_per_stage(monkeypatch, app):
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

    assert set(widget._stage_parameter_panels_by_key) == {
        "preprocessing",
        "displacement",
        "force",
        "stress",
    }
    assert "batch" not in widget._stage_parameter_panels_by_key

    preprocessing_panel = widget._stage_parameter_panels_by_key["preprocessing"]
    displacement_panel = widget._stage_parameter_panels_by_key["displacement"]
    force_panel = widget._stage_parameter_panels_by_key["force"]
    stress_panel = widget._stage_parameter_panels_by_key["stress"]

    assert {"pixel_size", "rolling_ball_radius"}.issubset(preprocessing_panel.parameter_controls)
    assert "nscales" not in preprocessing_panel.parameter_controls
    assert {"nscales", "inner_iterations"}.issubset(displacement_panel.parameter_controls)
    assert "young_modulus" not in displacement_panel.parameter_controls
    assert {"young_modulus", "auto_gcv"}.issubset(force_panel.parameter_controls)
    assert {"threshold", "mesh_algorithm"}.issubset(stress_panel.parameter_controls)

    displacement_section = widget._stage_sections_by_key["displacement"]
    assert not displacement_panel.isVisibleTo(widget)
    displacement_section.config_button.click()
    app.processEvents()
    assert displacement_panel.isVisibleTo(widget)
    assert not widget.displacement_widget.isVisibleTo(widget)


def test_main_widget_exposes_collapsed_stage_data_status_panels(monkeypatch, app):
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

    displacement_panel = widget._stage_status_panels_by_key["displacement"]

    assert displacement_panel.objectName() == "stage_displacement_data_status_panel"
    assert displacement_panel.isVisibleTo(widget)
    assert not widget.displacement_widget.isVisible()


def test_stage_data_status_refreshes_from_data_manager(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    section = widget._stage_sections_by_key["preprocessing"]
    panel = widget._stage_status_panels_by_key["preprocessing"]

    assert section.status_indicator.toolTip() == "Preprocessing status: not_started"
    assert panel.artifact_labels["reference"].text() == "Reference image: missing"

    widget.data_manager.reference = object()
    widget.data_manager.bead_stack = object()
    widget.refresh_stage_statuses()

    assert section.status_indicator.toolTip() == "Preprocessing status: ready"
    assert panel.artifact_labels["reference"].text() == "Reference image: available"

    widget.data_manager.preprocessed_reference = object()
    widget.data_manager.preprocessed_bead_stack = object()
    widget.refresh_stage_statuses()

    assert section.status_indicator.toolTip() == "Preprocessing status: done"
    assert panel.artifact_labels["preprocessed_bead_stack"].text() == "Preprocessed beads: available"


def test_workflow_parameter_panel_labels_farneback_controls(app):
    manager = _StubParameterManager()
    panel = _widget.WorkflowParameterPanel(manager)

    labels = {
        panel.layout().itemAt(i).widget().layout().labelForField(control).text()
        for i in range(panel.layout().count())
        if panel.layout().itemAt(i).widget().title() == "Displacement"
        for control in panel.parameter_controls.values()
        if panel.layout().itemAt(i).widget().layout().labelForField(control) is not None
    }

    assert "Farneback Levels" in labels
    assert "Farneback Iterations" in labels
    assert "Window Size" in labels
    assert "Refinement Iterations" not in labels
