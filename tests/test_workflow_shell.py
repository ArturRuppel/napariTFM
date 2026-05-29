import sys
import types
from pathlib import Path

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QCheckBox, QPushButton, QTabWidget, QWidget


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
        self._callbacks = []
        self.output_dir = None
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
        self.auto_save_calls = []
        self.artifact_errors = []
        self.raise_on_save = None

    def add_change_callback(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_change_callback(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def notify_changed(self):
        for callback in list(self._callbacks):
            callback()

    def set_output_dir(self, path):
        self.output_dir = Path(path).expanduser() if path else None
        self.notify_changed()

    def auto_save_artifact(self, key, pixel_size=None, frame_interval=None):
        if self.raise_on_save is not None:
            raise self.raise_on_save
        self.auto_save_calls.append(
            {"key": key, "pixel_size": pixel_size, "frame_interval": frame_interval}
        )
        return None

    def mark_artifact_error(self, key, error):
        self.artifact_errors.append((key, error))


class _StubVisualizationManager:
    def __init__(self, viewer, data_manager):
        self.viewer = viewer
        self.data_manager = data_manager


class _StubStageWidget(QWidget):
    preprocessing_completed = Signal(object)
    displacement_calculated = Signal(object)
    force_calculated = Signal(object)
    stress_calculated = Signal(object)
    action_states_changed = Signal()

    def __init__(self, *args):
        super().__init__()
        self.parameter_panel = QWidget()
        self.preview_btn = QPushButton("Preview")
        self.process_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save")
        self.preview_check = QCheckBox("Show Preview")
        self.analyze_btn = QPushButton("Analyze")
        self.preview_frame_btn = QPushButton("Preview Frame")
        self.preview_mesh_btn = QPushButton("Preview Mesh")
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.data_panel = QWidget()
        self.action_panel = QWidget()
        self.data_panel.setVisible(True)
        self.action_panel.setVisible(True)
        self.loaded_active_layers = []
        self.loaded_files = []
        self.update_count = 0
        self.action_calls = {"run": 0, "preview": 0, "cancel": 0}
        self._action_states = {"run": False, "preview": False}

    def _update_ui_state(self):
        self.update_count += 1

    def run_action(self):
        self.action_calls["run"] += 1

    def preview_action(self):
        self.action_calls["preview"] += 1

    def cancel_action(self):
        self.action_calls["cancel"] += 1

    def action_states(self):
        return dict(self._action_states)

    def set_action_states(self, **states):
        self._action_states.update(states)
        self.action_states_changed.emit()

    def load_active_layer(self, role):
        self.loaded_active_layers.append(role)

    def load_result_artifact(self, key):
        self.loaded_files.append(key)


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
    panel = QWidget()

    section = _widget._StageSection("Preprocessing", child, parameter_panel=panel)
    section.show()
    app.processEvents()

    # Body is always visible and parented to _content.
    assert child.parent() is section._content
    assert child.isVisible()
    assert section._content.isVisible()

    # Toggling the params panel never reparents or hides the body.
    section.params_btn.setChecked(True)
    app.processEvents()

    assert child.parent() is section._content
    assert child.isVisible()
    assert section._content.isVisible()

    section.params_btn.setChecked(False)
    app.processEvents()

    assert child.parent() is section._content
    assert child.isVisible()
    assert section._content.isVisible()


def test_stage_section_exposes_header_actions_with_stable_names(app):
    child = _StubStageWidget()

    section = _widget._StageSection(
        "Preprocessing",
        child,
        actions={
            "run": child.run_action,
            "preview": child.preview_action,
            "cancel": child.cancel_action,
        },
        action_states=child.action_states,
        action_states_changed=child.action_states_changed,
    )

    assert section.params_btn.objectName() == "stage_preprocessing_params_button"
    assert section.run_cancel_btn.objectName() == "stage_preprocessing_run_cancel_button"
    assert section.preview_button.objectName() == "stage_preprocessing_preview_button"

    assert "Run" in section.run_cancel_btn.toolTip()
    assert section.preview_button.toolTip() == "Preview Preprocessing"
    assert "Toggle" in section.params_btn.toolTip()


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


def test_stage_section_header_action_state_follows_action_states(app):
    child = _StubStageWidget()

    section = _widget._StageSection(
        "Preprocessing",
        child,
        actions={"run": child.run_action},
        action_states=child.action_states,
        action_states_changed=child.action_states_changed,
    )

    # Run starts disabled (action_states reports run=False).
    assert not section.run_cancel_btn.isEnabled()

    # Emitting the contract signal with run enabled lights up the header button.
    child.set_action_states(run=True)
    app.processEvents()

    assert section.run_cancel_btn.isEnabled()


def test_stage_section_status_indicator_remains_visible_when_collapsed(app):
    child = QWidget()
    panel = QWidget()

    section = _widget._StageSection(
        "Preprocessing", child, parameter_panel=panel, status="ready"
    )
    section.show()
    app.processEvents()

    # Params panel collapsed by default; status indicator stays visible.
    assert not panel.isVisible()
    assert section.status_indicator.isVisible()


def test_stage_section_header_actions_invoke_contract_handlers(app):
    child = _StubStageWidget()

    section = _widget._StageSection(
        "Preprocessing",
        child,
        actions={
            "run": child.run_action,
            "preview": child.preview_action,
            "cancel": child.cancel_action,
        },
        action_states=child.action_states,
        action_states_changed=child.action_states_changed,
    )

    # Enable run + preview so the header buttons accept clicks.
    child.set_action_states(run=True, preview=True)
    app.processEvents()

    section.run_cancel_btn.click()
    section.preview_button.click()
    # While running, the run/cancel button invokes the cancel handler.
    section.set_status("running")
    section.run_cancel_btn.click()

    assert child.action_calls == {"run": 1, "preview": 1, "cancel": 1}


def test_stage_section_disables_unsupported_actions_and_params_toggles(app):
    child = QWidget()

    section = _widget._StageSection("Batch Analysis", child)
    section.show()
    app.processEvents()

    assert not section.run_cancel_btn.isEnabled()
    assert not section.preview_button.isEnabled()
    # No parameter panel -> no params affordance; body is always visible.
    assert not section.params_btn.isVisible()
    assert child.isVisible()
    assert section._content.isVisible()


def test_stage_section_params_toggles_inline_parameter_panel_when_provided(app):
    child = QWidget()
    parameter_panel = QWidget()

    section = _widget._StageSection(
        "Displacement",
        child,
        parameter_panel=parameter_panel,
    )
    section.show()
    app.processEvents()

    # Body always visible; parameter panel collapsed by default.
    assert child.isVisible()
    assert section._content.isVisible()
    assert not parameter_panel.isVisible()
    assert not section._param_section.is_expanded

    section.params_btn.click()
    app.processEvents()

    # Toggling params reveals only the parameter panel; body stays visible.
    assert child.isVisible()
    assert section._content.isVisible()
    assert parameter_panel.isVisible()
    assert section._param_section.is_expanded


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
    # Every stage body is always visible; only parameter panels collapse.
    assert widget.preprocessing_widget.isVisible()
    assert widget.displacement_widget.isVisible()
    assert widget.force_widget.isVisible()
    assert widget.msm_widget.isVisible()
    assert widget.batch_widget.isVisible()


def test_main_widget_lets_dock_determine_width(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    assert widget.maximumWidth() > 500


def test_data_manager_change_callback_refreshes_stage_widgets(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    widget.data_manager.notify_changed()

    assert widget.preprocessing_widget.update_count == 1
    assert widget.displacement_widget.update_count == 1
    assert widget.force_widget.update_count == 1
    assert widget.msm_widget.update_count == 1
    assert widget.batch_widget.update_count == 1


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
    # Header run drives the widget's run_action via the signal contract.
    widget.displacement_widget.set_action_states(run=True)
    app.processEvents()

    displacement_section.run_cancel_btn.click()

    assert widget.displacement_widget.action_calls["run"] == 1


def test_main_widget_stage_headers_wire_stage_specific_run_buttons(monkeypatch, app):
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

    # Contract stages: header run invokes the widget's run_action handler.
    contract_cases = [
        ("preprocessing", widget.preprocessing_widget),
        ("force", widget.force_widget),
        ("stress", widget.msm_widget),
    ]
    for key, stage_widget in contract_cases:
        stage_widget.set_action_states(run=True)
        app.processEvents()
        widget._stage_sections_by_key[key].run_cancel_btn.click()
        assert stage_widget.action_calls["run"] == 1, (
            f"{key} header run button did not trigger its widget run_action"
        )

    # Batch keeps its own run button; header run clicks it (always enabled).
    batch_clicks = {"n": 0}
    widget.batch_widget.run_analysis_btn.clicked.connect(
        lambda *_: batch_clicks.__setitem__("n", batch_clicks["n"] + 1)
    )
    widget._stage_sections_by_key["batch"].run_cancel_btn.click()
    assert batch_clicks["n"] == 1, "batch header run button did not click its run button"


def test_main_widget_stress_header_preview_wires_to_frame_preview(monkeypatch, app):
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

    # Header preview invokes the stress widget's preview_action via the contract.
    widget.msm_widget.set_action_states(preview=True)
    app.processEvents()
    widget._stage_sections_by_key["stress"].preview_button.click()

    assert widget.msm_widget.action_calls["preview"] == 1


def test_main_widget_does_not_use_action_target_reflection(app):
    assert not hasattr(_widget.napariTFMWidget, "_find_stage_action_targets")
    assert not hasattr(_widget.napariTFMWidget, "_first_existing_widget")


def test_stage_sections_use_signal_action_contract(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    main_widget = _widget.napariTFMWidget(object())

    sec = main_widget._stage_sections_by_key["displacement"]
    assert "run" in sec._actions and "preview" in sec._actions
    assert sec._action_states is not None
    assert not hasattr(sec, "_action_state_syncs")


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


def test_main_widget_does_not_expose_legacy_parameter_panel(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    assert not hasattr(widget, "parameter_panel")
    assert widget.project_section is not None


def test_main_widget_project_section_tracks_output_directory(monkeypatch, app, tmp_path):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)

    assert widget.project_section.output_dir_label.text() == str(tmp_path)


def test_preprocessing_data_rows_route_assignment_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    rows = widget._stage_status_panels_by_key["preprocessing"].artifact_rows

    rows["reference"].action_btn.click()
    rows["bead_stack"].action_btn.click()
    rows["cell_stack"].action_btn.click()

    assert widget.preprocessing_widget.loaded_active_layers == [
        "reference",
        "beads",
        "cells",
    ]


def test_generated_output_row_save_calls_data_manager_with_calibration(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.preprocessed_bead_stack = object()
    widget.refresh_stage_statuses()

    row = widget._stage_status_panels_by_key["preprocessing"].artifact_rows["preprocessed_bead_stack"]
    row.action_btn.click()

    assert widget.data_manager.auto_save_calls[-1] == {
        "key": "preprocessed_bead_stack",
        "pixel_size": 1.0,
        "frame_interval": 1.0,
    }


def test_failed_generated_output_save_marks_artifact_error(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget.QMessageBox, "warning", lambda *args: None)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.raise_on_save = RuntimeError("disk full")
    widget.data_manager.preprocessed_reference = object()
    widget.refresh_stage_statuses()

    row = widget._stage_status_panels_by_key["preprocessing"].artifact_rows["preprocessed_reference"]
    row.action_btn.click()

    assert widget.data_manager.artifact_errors[-1] == ("preprocessed_reference", "disk full")


def test_displacement_data_rows_route_assignment_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    rows = widget._stage_status_panels_by_key["displacement"].artifact_rows

    rows["preprocessed_reference"].action_btn.click()
    rows["preprocessed_bead_stack"].action_btn.click()

    assert widget.displacement_widget.loaded_active_layers == ["reference", "beads"]


def test_force_and_stress_input_rows_route_load_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    widget._stage_status_panels_by_key["force"].artifact_rows["displacement_results"].action_btn.click()
    widget._stage_status_panels_by_key["stress"].artifact_rows["force_results"].action_btn.click()
    widget._stage_status_panels_by_key["stress"].artifact_rows["mask_stack"].action_btn.click()

    assert widget.force_widget.loaded_files == ["displacement_results"]
    assert widget.msm_widget.loaded_files == ["force_results", "mask_stack"]



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

    assert "rolling_ball_radius" in preprocessing_panel.parameter_controls
    # Calibration lives only in the Project section, not the preprocessing panel.
    assert "pixel_size" not in preprocessing_panel.parameter_controls
    assert "nscales" not in preprocessing_panel.parameter_controls
    assert {"nscales", "inner_iterations"}.issubset(displacement_panel.parameter_controls)
    assert "young_modulus" not in displacement_panel.parameter_controls
    assert {"young_modulus", "auto_gcv"}.issubset(force_panel.parameter_controls)
    assert {"density_factor", "mesh_algorithm"}.issubset(stress_panel.parameter_controls)
    assert "threshold" not in stress_panel.parameter_controls

    displacement_section = widget._stage_sections_by_key["displacement"]
    assert displacement_section.parameter_panel is displacement_panel
    assert not displacement_panel.isVisibleTo(widget)
    displacement_section.params_btn.click()
    app.processEvents()
    assert displacement_panel.isVisibleTo(widget)


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
    # Stage body is always visible alongside its status panel.
    assert widget.displacement_widget.isVisible()


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
    assert panel.artifact_labels["reference"].text() == "Missing"

    widget.data_manager.reference = object()
    widget.data_manager.bead_stack = object()
    widget.refresh_stage_statuses()

    assert section.status_indicator.toolTip() == "Preprocessing status: ready"
    assert (
        "×" in panel.artifact_labels["reference"].text()
        or panel.artifact_labels["reference"].text() == "Loaded"
    )

    widget.data_manager.preprocessed_reference = object()
    widget.data_manager.preprocessed_bead_stack = object()
    widget.refresh_stage_statuses()

    assert section.status_indicator.toolTip() == "Preprocessing status: done"
    assert (
        "×" in panel.artifact_labels["preprocessed_bead_stack"].text()
        or panel.artifact_labels["preprocessed_bead_stack"].text() == "Loaded"
    )


def test_stage_status_is_done_when_output_files_exist_on_disk(monkeypatch, app, tmp_path):
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.data_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "data_manager.py",
    )
    _dm_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_dm_mod)
    DataManager = _dm_mod.DataManager

    monkeypatch.setattr(_widget, "DataManager", DataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)
    section = widget._stage_sections_by_key["preprocessing"]

    widget.refresh_stage_statuses()
    assert section.status_indicator.toolTip() != "Preprocessing status: done"

    (tmp_path / "preprocessed_beads.tif").write_bytes(b"x")
    (tmp_path / "preprocessed_reference.tif").write_bytes(b"x")
    widget.refresh_stage_statuses()

    assert section.status_indicator.toolTip() == "Preprocessing status: done"


def test_main_widget_constructs_when_stage_widget_lacks_parameter_panel(monkeypatch, app):
    class _NoPanelStage(_StubStageWidget):
        def __init__(self, *args):
            super().__init__(*args)
            del self.parameter_panel  # simulate a fully-inverted stage widget

    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _NoPanelStage)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())  # must not raise

    assert not hasattr(widget.preprocessing_widget, "parameter_panel")


def test_each_stage_has_single_inline_parameter_editor(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    # Each stage with parameters mounts exactly one editor: the section's
    # first-class parameter_panel (no nested faux-stage duplication).
    for key in ("preprocessing", "displacement", "force", "stress"):
        section = widget._stage_sections_by_key[key]
        assert section.parameter_panel is widget._stage_parameter_panels_by_key[key]
    assert widget._stage_sections_by_key["batch"].parameter_panel is None
    assert "batch" not in widget._stage_parameter_panels_by_key
    assert not hasattr(widget, "_hide_embedded_parameter_panels")


def test_workflow_parameter_panel_labels_farneback_controls(app):
    from qtpy.QtWidgets import QLabel

    manager = _StubParameterManager()
    panel = _widget.WorkflowParameterPanel(manager)

    labels = {label.text() for label in panel.findChildren(QLabel)}

    assert "Farneback Levels" in labels
    assert "Farneback Iterations" in labels
    assert "Window Size" in labels
    assert "Refinement Iterations" not in labels


def test_refresh_updates_every_stage_widget_once(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    stage_widgets = widget._stage_widgets()
    assert len(stage_widgets) == 5

    before = [w.update_count for w in stage_widgets]
    widget.refresh()
    after = [w.update_count for w in stage_widgets]
    assert all(a == b + 1 for a, b in zip(after, before))


def test_completion_signal_triggers_single_refresh(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    calls = {"n": 0}
    original = widget.refresh
    widget.refresh = lambda: (calls.__setitem__("n", calls["n"] + 1), original())[1]

    widget.force_widget.force_calculated.emit(object())
    assert calls["n"] == 1


def test_get_set_state_round_trips_parameters(monkeypatch, app, tmp_path):
    import importlib.util as _ilu

    _pm_spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.parameter_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "parameter_manager.py",
    )
    _pm_mod = _ilu.module_from_spec(_pm_spec)
    _pm_spec.loader.exec_module(_pm_mod)
    ParameterManager = _pm_mod.ParameterManager

    _dm_spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.data_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "data_manager.py",
    )
    _dm_mod = _ilu.module_from_spec(_dm_spec)
    _dm_spec.loader.exec_module(_dm_mod)
    DataManager = _dm_mod.DataManager

    monkeypatch.setattr(_widget, "DataManager", DataManager)
    monkeypatch.setattr(_widget, "ParameterManager", ParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)

    widget.parameter_manager.set_parameter("rolling_ball_radius", 7)
    state = widget.get_state()
    assert state["parameters"]["rolling_ball_radius"] == 7
    assert state["output_dir"] == str(tmp_path)

    widget.parameter_manager.set_parameter("rolling_ball_radius", 0)
    widget.set_state(state)
    assert widget.parameter_manager.get_parameter("rolling_ball_radius") == 7


def test_reconcile_loads_existing_config_from_output_dir(monkeypatch, app, tmp_path):
    import importlib.util as _ilu
    import json

    _pm_spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.parameter_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "parameter_manager.py",
    )
    _pm_mod = _ilu.module_from_spec(_pm_spec)
    _pm_spec.loader.exec_module(_pm_mod)
    RealParameterManager = _pm_mod.ParameterManager

    _dm_spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.data_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "data_manager.py",
    )
    _dm_mod = _ilu.module_from_spec(_dm_spec)
    _dm_spec.loader.exec_module(_dm_mod)
    RealDataManager = _dm_mod.DataManager

    monkeypatch.setattr(_widget, "DataManager", RealDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", RealParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    (tmp_path / "napariTFM_config.json").write_text(
        json.dumps({"version": 1, "parameters": {"rolling_ball_radius": 9},
                    "output_dir": str(tmp_path)})
    )

    widget.data_manager.set_output_dir(tmp_path)
    widget.project_section.body.output_dir_changed.emit()  # simulate dir chosen

    assert widget.parameter_manager.get_parameter("rolling_ball_radius") == 9


def test_calibration_change_updates_all_stage_widgets(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    stage = widget._stage_widgets()
    before = [w.update_count for w in stage]
    widget.parameter_manager.set_parameter("pixel_size", 2.0)
    after = [w.update_count for w in stage]
    assert all(a == b + 1 for a, b in zip(after, before))


def test_force_param_change_updates_only_force_widget(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    before = {id(w): w.update_count for w in widget._stage_widgets()}
    widget.parameter_manager.set_parameter("force_vector_stride", 30)
    assert widget.force_widget.update_count == before[id(widget.force_widget)] + 1
    for w in widget._stage_widgets():
        if w is widget.force_widget:
            continue
        assert w.update_count == before[id(w)]


def test_unrouted_param_change_updates_no_stage_widget(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    before = [w.update_count for w in widget._stage_widgets()]
    widget.parameter_manager.set_parameter("rolling_ball_radius", 5)
    after = [w.update_count for w in widget._stage_widgets()]
    assert after == before


def test_reconcile_writes_config_when_absent(monkeypatch, app, tmp_path):
    import importlib.util as _ilu

    _pm_spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.parameter_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "parameter_manager.py",
    )
    _pm_mod = _ilu.module_from_spec(_pm_spec)
    _pm_spec.loader.exec_module(_pm_mod)
    RealParameterManager = _pm_mod.ParameterManager

    _dm_spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.data_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "data_manager.py",
    )
    _dm_mod = _ilu.module_from_spec(_dm_spec)
    _dm_spec.loader.exec_module(_dm_mod)
    RealDataManager = _dm_mod.DataManager

    monkeypatch.setattr(_widget, "DataManager", RealDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", RealParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget", "BatchAnalysisWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)
    widget.project_section.body.output_dir_changed.emit()

    assert (tmp_path / "napariTFM_config.json").exists()


def test_shell_theme_button_switches_palette(monkeypatch, app):
    from napariTFM.widgets import _widget
    from napariTFM.widgets import _ui_style
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    original = _ui_style.active_theme_name()
    try:
        widget = _widget.napariTFMWidget(object())
        assert hasattr(widget, "theme_btn")
        other = next(n for n in _ui_style.theme_names() if n != original)
        widget._on_theme_selected(other)
        assert _ui_style.active_theme_name() == other
        accent = _ui_style.stage_accent("preprocessing")
        assert accent in widget._stage_sections_by_key["preprocessing"].header_label.styleSheet()
    finally:
        _ui_style.set_active_theme(original)


def _real_parameter_manager():
    import importlib.util as _ilu

    _pm_spec = _ilu.spec_from_file_location(
        "napariTFM.utilities.parameter_manager_real",
        Path(__file__).parent.parent / "napariTFM" / "utilities" / "parameter_manager.py",
    )
    _pm_mod = _ilu.module_from_spec(_pm_spec)
    _pm_spec.loader.exec_module(_pm_mod)
    return _pm_mod.ParameterManager()


def test_workflow_parameter_panel_uses_sliders_for_numeric(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel

    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Displacement",))
    # nscales is an int param -> islider; disp_arrow_scale is float -> dslider.
    assert type(panel.parameter_controls["nscales"]).__name__ == "QLabeledSlider"
    assert type(panel.parameter_controls["disp_arrow_scale"]).__name__ == "QLabeledDoubleSlider"


def test_workflow_parameter_panel_slider_writes_through(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel

    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Displacement",))
    panel.parameter_controls["nscales"].setValue(7)
    assert pm.get_ui_parameter("nscales") == 7


def test_wheel_guard_consumes_scroll_on_unfocused_slider(app):
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent

    from napariTFM.widgets._param_controls import islider
    from napariTFM.widgets._widget import SpinBoxEventFilter

    slider = islider(0, 10, 5)
    filt = SpinBoxEventFilter()
    event = QWheelEvent(
        QPointF(0, 0), QPointF(0, 0), QPoint(0, 0), QPoint(0, -120),
        Qt.NoButton, Qt.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    )
    # Unfocused slider: the wheel event must be swallowed.
    assert filt.eventFilter(slider, event) is True


def test_shell_mounts_param_panels_as_section_parameter_panel(monkeypatch, app):
    from napariTFM.widgets import _widget
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    section = widget._stage_sections_by_key["displacement"]
    # The panel is the section's first-class parameter_panel, not a nested section.
    assert section.parameter_panel is widget._stage_parameter_panels_by_key["displacement"]
    assert not hasattr(section, "add_inner_section")


def test_preprocessing_panel_excludes_general_calibration(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel

    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Preprocessing",))
    assert "pixel_size" not in panel.parameter_controls
    assert "frame_interval" not in panel.parameter_controls
    # Preprocessing-specific params still present.
    assert "rolling_ball_radius" in panel.parameter_controls


def test_shell_preprocessing_panel_has_no_calibration_controls(monkeypatch, app):
    from napariTFM.widgets import _widget
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "BatchAnalysisWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    panel = widget._stage_parameter_panels_by_key["preprocessing"]
    assert "pixel_size" not in panel.parameter_controls
    assert "frame_interval" not in panel.parameter_controls
