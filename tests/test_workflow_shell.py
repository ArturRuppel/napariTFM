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

    def get_all_parameters(self):
        return dict(self._values)

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
        self.artifact_errors = []

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

    def mark_artifact_error(self, key, error):
        self.artifact_errors.append((key, error))


class _StubVisualizationManager:
    def __init__(self, viewer, data_manager):
        self.viewer = viewer
        self.data_manager = data_manager


class _StubController(QObject):
    progress_updated = Signal(int, str)


class _StubStageWidget(QWidget):
    preprocessing_completed = Signal(object)
    displacement_calculated = Signal(object)
    force_calculated = Signal(object)
    stress_calculated = Signal(object)
    action_states_changed = Signal()

    def __init__(self, *args):
        super().__init__()
        self.controller = _StubController()
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

    def set_experiment_records(self, records):
        # The batch widget consumes the config table (P0); the stub just records.
        self.experiment_records = list(records)

    def save_config_to_yaml(self, filepath):
        self.saved_config_path = filepath

    def load_config_from_yaml(self, filepath):
        self.loaded_config_path = filepath

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

from napariTFM.widgets import _widget


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_stage_section_toggles_child_without_destroying_it(app):
    child = QWidget()
    panel = QWidget()

    section = _widget.StageSection("Preprocessing", child, parameter_panel=panel)
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

    section = _widget.StageSection(
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


def test_stage_section_tracks_status(app):
    section = _widget.StageSection("Preprocessing", QWidget(), status="ready")
    assert section.status == "ready"
    section.set_status("done")
    assert section.status == "done"


def test_stage_section_applies_stage_accent_to_header(app):
    from napariTFM.widgets._ui_style import muted_accent
    child = QWidget()

    section = _widget.StageSection("Traction / FTTC", child, accent="#2a9d8f")

    # The header title is an accent pill whose color is the muted accent.
    assert muted_accent("#2a9d8f") in section.header_label.styleSheet()


def test_stage_section_header_action_state_follows_action_states(app):
    child = _StubStageWidget()

    section = _widget.StageSection(
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


def test_stage_section_header_actions_invoke_contract_handlers(app):
    child = _StubStageWidget()

    section = _widget.StageSection(
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

    section = _widget.StageSection("Batch Analysis", child)
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

    section = _widget.StageSection(
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
    widget = _widget.napariTFMWidget(object())
    widget.show()
    app.processEvents()

    assert widget.findChildren(QTabWidget) == []
    # Every stage body is always visible; only parameter panels collapse.
    assert widget.preprocessing_widget.isVisible()
    assert widget.displacement_widget.isVisible()
    assert widget.force_widget.isVisible()
    assert widget.msm_widget.isVisible()


def test_stage_sections_receive_ordered_neighbour_accents(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    widget = _widget.napariTFMWidget(object())

    sections = widget._stage_sections
    # each section's "below" accent equals the next section's accent — a
    # continuous ramp down the rail.
    for i, sec in enumerate(sections[:-1]):
        assert sec._accent_below == sections[i + 1]._accent


def _stub_main_widget(monkeypatch):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
    return _widget.napariTFMWidget(object())


def test_stage_progress_feeds_one_global_status_label(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)

    # One panel-level status label, prefixed with the reporting stage (P2).
    assert hasattr(widget, "status_label")

    widget.displacement_widget.controller.progress_updated.emit(40, "Calculating…")
    assert widget.status_label.text() == "Displacement — Calculating…"

    widget.preprocessing_widget.controller.progress_updated.emit(0, "Error: boom")
    assert widget.status_label.text() == "Preprocessing — Error: boom"

    widget.msm_widget.controller.progress_updated.emit(100, "Done")
    assert widget.status_label.text() == "Stress — Done"


def _write_stage_ntfm(folder, **arrays):
    # Write a real .ntfm into the experiment's TFM_data folder (P3 truth source).
    import numpy as np  # noqa: F401  (kept local; arrays passed by caller)

    from napariTFM.utilities import ntfm

    tfm = folder / "TFM_data"
    tfm.mkdir(parents=True, exist_ok=True)
    df = ntfm.arrays_to_tidy(grid_spacing=1.0, frame_interval=1.0, **arrays)
    ntfm.write_ntfm(tfm / f"{folder.name}.ntfm", df, ntfm.build_metadata(config={}))


def test_experiment_stage_status_inputs_only_is_ready_frontier(monkeypatch, app, tmp_path):
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")

    statuses = widget._experiment_stage_status(str(folder))
    assert statuses["preprocessing"] == "ready"
    assert statuses["displacement"] == "not_started"
    assert statuses["force"] == "not_started"
    assert statuses["stress"] == "not_started"


def test_experiment_stage_status_displacement_only(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")
    _write_stage_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))

    statuses = widget._experiment_stage_status(str(folder))
    # Displacement present implies preprocessing ran; force is the next frontier.
    assert statuses["preprocessing"] == "done"
    assert statuses["displacement"] == "done"
    assert statuses["force"] == "ready"
    assert statuses["stress"] == "not_started"


def test_experiment_stage_status_through_force(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "exp"
    folder.mkdir()
    _write_stage_ntfm(
        folder,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
    )

    statuses = widget._experiment_stage_status(str(folder))
    assert statuses["preprocessing"] == "done"
    assert statuses["displacement"] == "done"
    assert statuses["force"] == "done"
    assert statuses["stress"] == "ready"


def test_experiment_stage_status_full_pipeline_all_done(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "exp"
    folder.mkdir()
    _write_stage_ntfm(
        folder,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
        stress_tensor=np.ones((1, 2, 2, 2, 2)),
    )

    statuses = widget._experiment_stage_status(str(folder))
    assert all(statuses[s] == "done" for s in ("preprocessing", "displacement", "force", "stress"))


def test_experiment_stage_status_disabled_stress_reads_off(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(False)
    folder = tmp_path / "exp"
    folder.mkdir()
    _write_stage_ntfm(
        folder,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
    )

    statuses = widget._experiment_stage_status(str(folder))
    # MSM is exempt from auto-skip (D1); disabling it reads as off, not ready.
    assert statuses["stress"] == "off"
    assert statuses["force"] == "done"


class _FakeBatchAnalysis:
    """Records its config and replays per-folder lifecycle to the callback."""

    last_config = None
    last_instance = None

    def __init__(self, config, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        type(self).last_config = config
        type(self).last_instance = self

    def process_all_folders(self):
        for folder in self.config["root_folders"]:
            if self.progress_callback:
                self.progress_callback(folder, "running")
                self.progress_callback(folder, "done")


def test_run_all_builds_config_from_table_and_runs_batch(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(
        ["/data/exp_a", "/data/exp_b"],
        input_files={"beads": "beads.tif", "reference": "reference.tif"},
        columns={"condition": "soft"},
    )

    widget.experiments_list.run_all_requested.emit()

    cfg = _FakeBatchAnalysis.last_config
    assert cfg["root_folders"] == ["/data/exp_a", "/data/exp_b"]
    assert cfg["experiment_metadata"]["/data/exp_a"] == {"condition": "soft"}
    assert cfg["analysis_steps"]["stress"] is True


def test_run_all_honours_disabled_stress(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(False)
    widget.experiments_list.add_folders(["/data/exp_a"])

    widget.experiments_list.run_all_requested.emit()

    assert _FakeBatchAnalysis.last_config["analysis_steps"]["stress"] is False


def test_run_all_progress_marks_running_then_refreshes(monkeypatch, app, tmp_path):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    widget = _stub_main_widget(monkeypatch)

    seen = []
    monkeypatch.setattr(
        widget.experiments_list, "mark_running", lambda path: seen.append(("run", path))
    )
    monkeypatch.setattr(
        widget.experiments_list, "refresh_statuses", lambda: seen.append(("refresh",))
    )
    widget.experiments_list.add_folders(["/data/exp_a"])

    widget.experiments_list.run_all_requested.emit()

    # running -> mark_running; done -> refresh from disk.
    assert ("run", "/data/exp_a") in seen
    assert ("refresh",) in seen


def test_run_all_with_no_experiments_is_a_noop(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    _FakeBatchAnalysis.last_config = None
    widget = _stub_main_widget(monkeypatch)

    widget.experiments_list.run_all_requested.emit()

    assert _FakeBatchAnalysis.last_config is None


def test_only_stress_stage_is_optional(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    assert widget._stage_sections_by_key["stress"].enable_btn is not None
    for key in ("preprocessing", "displacement", "force"):
        assert widget._stage_sections_by_key[key].enable_btn is None


def test_disabling_stress_marks_it_off_and_persists_in_state(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    stress = widget._stage_sections_by_key["stress"]

    stress.set_enabled(False)
    assert stress.spine._status == "off"
    assert widget.get_state()["disabled_stages"] == ["stress"]


def test_set_state_restores_disabled_stages(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    stress = widget._stage_sections_by_key["stress"]

    widget.set_state({"parameters": {}, "disabled_stages": ["stress"]})
    assert stress.is_enabled is False
    assert stress.spine._status == "off"

    widget.set_state({"parameters": {}, "disabled_stages": []})
    assert stress.is_enabled is True


def test_main_widget_lets_dock_determine_width(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

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

    widget = _widget.napariTFMWidget(object())

    widget.data_manager.notify_changed()

    assert widget.preprocessing_widget.update_count == 1
    assert widget.displacement_widget.update_count == 1
    assert widget.force_widget.update_count == 1
    assert widget.msm_widget.update_count == 1


def test_main_widget_stage_headers_wire_existing_stage_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

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


def test_main_widget_stress_header_preview_wires_to_frame_preview(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

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


def test_generated_output_rows_have_no_save_button(monkeypatch, app):
    # Preview-only (ROADMAP §4): stage runs write nothing to disk, so output
    # rows expose no Save action.
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    panels = widget._stage_status_panels_by_key
    output_rows = [
        panels["preprocessing"].artifact_rows["preprocessed_bead_stack"],
        panels["preprocessing"].artifact_rows["preprocessed_reference"],
        panels["displacement"].artifact_rows["displacement_results"],
        panels["force"].artifact_rows["force_results"],
        panels["stress"].artifact_rows["stress_results"],
    ]
    assert all(row.action_btn is None for row in output_rows)


def test_displacement_data_rows_route_assignment_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    rows = widget._stage_status_panels_by_key["displacement"].artifact_rows

    rows["preprocessed_reference"].action_btn.click()
    rows["preprocessed_bead_stack"].action_btn.click()

    assert widget.displacement_widget.loaded_active_layers == ["reference", "beads"]


def test_only_mask_input_row_routes_a_load_action(monkeypatch, app):
    # Results chain in-memory (ROADMAP §4): the displacement/force input rows are
    # status-only. The mask is an external input (ROADMAP §2), so its row keeps a
    # Load action.
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    force_input = widget._stage_status_panels_by_key["force"].artifact_rows["displacement_results"]
    stress_force_input = widget._stage_status_panels_by_key["stress"].artifact_rows["force_results"]
    mask_row = widget._stage_status_panels_by_key["stress"].artifact_rows["mask_stack"]

    assert force_input.action_btn is None
    assert stress_force_input.action_btn is None

    mask_row.action_btn.click()
    assert widget.msm_widget.loaded_files == ["mask_stack"]



def test_main_widget_groups_parameters_inline_per_stage(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

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

    widget = _widget.napariTFMWidget(object())
    widget.show()
    app.processEvents()

    displacement_panel = widget._stage_status_panels_by_key["displacement"]
    section = widget._stage_sections_by_key["displacement"]

    assert displacement_panel.objectName() == "stage_displacement_data_status_panel"
    # Status panel starts collapsed behind the 🔍 toggle.
    assert section._status_section.is_expanded is False
    assert not displacement_panel.isVisibleTo(widget)
    # Stage body stays visible regardless.
    assert widget.displacement_widget.isVisible()

    # The 🔍 toggle reveals the status panel.
    section.files_btn.setChecked(True)
    app.processEvents()
    assert displacement_panel.isVisibleTo(widget)


def test_stage_data_status_refreshes_from_data_manager(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    section = widget._stage_sections_by_key["preprocessing"]
    panel = widget._stage_status_panels_by_key["preprocessing"]

    assert section.status == "not_started"
    assert panel.artifact_rows["reference"].info_label.text() == "Missing"

    widget.data_manager.reference = object()
    widget.data_manager.bead_stack = object()
    widget.refresh_stage_statuses()

    assert section.status == "ready"
    assert (
        "×" in panel.artifact_rows["reference"].info_label.text()
        or panel.artifact_rows["reference"].info_label.text() == "Loaded"
    )

    widget.data_manager.preprocessed_reference = object()
    widget.data_manager.preprocessed_bead_stack = object()
    widget.refresh_stage_statuses()

    assert section.status == "done"
    assert (
        "×" in panel.artifact_rows["preprocessed_bead_stack"].info_label.text()
        or panel.artifact_rows["preprocessed_bead_stack"].info_label.text() == "Loaded"
    )


def test_stage_status_is_done_when_results_are_in_memory(monkeypatch, app, tmp_path):
    # Preview-only (ROADMAP §4): "done" follows in-memory results, not files on
    # disk. Uses the real DataManager (not the stub) to exercise availability.
    import importlib.util as _ilu
    import numpy as np

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

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)
    section = widget._stage_sections_by_key["preprocessing"]

    widget.refresh_stage_statuses()
    assert section.status != "done"

    # Files on disk are irrelevant — only in-memory results count.
    (tmp_path / "preprocessed_beads.tif").write_bytes(b"x")
    (tmp_path / "preprocessed_reference.tif").write_bytes(b"x")
    widget.refresh_stage_statuses()
    assert section.status != "done"

    widget.data_manager.set_preprocessed_bead_stack(np.zeros((2, 4, 4), dtype=np.float32))
    widget.data_manager.set_preprocessed_reference(np.zeros((4, 4), dtype=np.float32))
    widget.refresh_stage_statuses()
    assert section.status == "done"


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

    widget = _widget.napariTFMWidget(object())  # must not raise

    assert not hasattr(widget.preprocessing_widget, "parameter_panel")


def test_each_stage_has_single_inline_parameter_editor(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "PreprocessingWidget", "DisplacementAnalysisWidget",
        "FTTCWidget", "MSMWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    # Each stage with parameters mounts exactly one editor: the section's
    # first-class parameter_panel (no nested faux-stage duplication).
    for key in ("preprocessing", "displacement", "force", "stress"):
        section = widget._stage_sections_by_key[key]
        assert section.parameter_panel is widget._stage_parameter_panels_by_key[key]
    assert "batch" not in widget._stage_parameter_panels_by_key
    assert "batch" not in widget._stage_sections_by_key
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
        "FTTCWidget", "MSMWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    stage_widgets = widget._stage_widgets()
    assert len(stage_widgets) == 4

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
        "FTTCWidget", "MSMWidget",
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
        "FTTCWidget", "MSMWidget",
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
        "FTTCWidget", "MSMWidget",
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
        "FTTCWidget", "MSMWidget",
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

    original = _ui_style.active_theme_name()
    try:
        widget = _widget.napariTFMWidget(object())
        assert hasattr(widget, "theme_btn")
        other = next(n for n in _ui_style.theme_names() if n != original)
        widget._on_theme_selected(other)
        assert _ui_style.active_theme_name() == other
        accent = _ui_style.stage_accent("preprocessing")
        assert _ui_style.muted_accent(accent) in widget._stage_sections_by_key["preprocessing"].header_label.styleSheet()
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

    widget = _widget.napariTFMWidget(object())
    panel = widget._stage_parameter_panels_by_key["preprocessing"]
    assert "pixel_size" not in panel.parameter_controls
    assert "frame_interval" not in panel.parameter_controls


def test_experiments_list_is_present_above_pipeline(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    assert hasattr(widget, "experiments_list")
    assert widget.experiments_list is not None


def test_selecting_experiment_updates_pipeline_context_label(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.set_experiments(["/data/Ctrl/pos_00"])
    widget.experiments_list.set_active("/data/Ctrl/pos_00")
    assert "pos_00" in widget._pipeline_context_label.text()


def test_disabling_stress_refreshes_experiment_minirails(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.set_experiments(["/data/Ctrl/pos_00"])
    section = widget._stage_sections_by_key["stress"]
    section.set_enabled(False)
    row = widget.experiments_list._rows[0]
    fill, ring = row.mini_rail.appearance("stress")
    assert fill is None  # stress dot now reads 'off'


def test_state_round_trips_experiments_and_active(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.set_experiments(["/data/a", "/data/b"])
    widget.experiments_list.set_active("/data/b")

    state = widget.get_state()
    assert state["experiments"] == ["/data/a", "/data/b"]
    assert state["active_experiment"] == "/data/b"

    fresh = _stub_main_widget(monkeypatch)
    fresh.set_state(state)
    assert fresh.experiments_list.experiments() == ["/data/a", "/data/b"]
    assert fresh.experiments_list.active() == "/data/b"


def test_single_config_button_label_and_no_params_only_handler(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    # One save lives at the top now; the params-only handler is gone (P0b).
    assert widget.save_config_btn.text() == "Save Config"
    assert not hasattr(widget, "_save_parameters")


def test_save_config_writes_table_driven_yaml(monkeypatch, app, tmp_path):
    import yaml

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(
        ["/data/a", "/data/b"],
        input_files={"beads": "beads.tif", "reference": "reference.tif"},
        columns={"day": "1"},
    )
    out = tmp_path / "run.yaml"
    monkeypatch.setattr(
        _widget.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._save_config()

    config = yaml.safe_load(out.read_text())
    assert config["root_folders"] == ["/data/a", "/data/b"]
    assert config["experiment_metadata"]["/data/a"] == {"day": "1"}
    assert config["input_files"]["beads"] == "beads.tif"
    assert "parameters" in config


def test_load_config_rebuilds_the_experiments_table(monkeypatch, app, tmp_path):
    import yaml

    config = {
        "root_folders": ["/data/x", "/data/y"],
        "input_files": {"beads": "beads.tif", "reference": "reference.tif"},
        "parameters": {"young_modulus": 9.0},
        "experiment_metadata": {"/data/x": {"day": "1"}, "/data/y": {"day": "2"}},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))

    widget = _stub_main_widget(monkeypatch)
    monkeypatch.setattr(
        _widget.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._load_config()

    assert widget.experiments_list.experiments() == ["/data/x", "/data/y"]
    records = widget.experiments_list.experiment_records()
    assert records[0]["columns"] == {"day": "1"}
    assert records[1]["columns"] == {"day": "2"}
    assert widget.parameter_manager.get_parameter("young_modulus") == 9.0
