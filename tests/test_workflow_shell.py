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
            "min_intensity_percentile": 0.0,
            "max_intensity_percentile": 100.0,
            "gaussian_sigma": 0.0,
            "cell_min_intensity_percentile": 0.0,
            "cell_max_intensity_percentile": 100.0,
            "cell_gaussian_sigma": 0.0,
            "registration_mode": "translation",
            "outer_iterations": 5,
            "disp_method": "PIV",
            "disp_device": "auto",
            "piv_window": 16,
            "piv_overlap": 0.75,
            "piv_passes": 8,
            "ilk_radius": 7,
            "ilk_num_warp": 10,
            "ffd_level_spacing": 12.0,
            "ffd_num_levels": 6,
            "ffd_metric": "lncc",
            "ffd_num_iters": 50,
            "ffd_elastic": 0.0,
            "ffd_downscale": 2.0,
            "ffd_min_size": 16,
            "ffd_interp": "bicubic",
            "ffd_warmstart": True,
            "ffd_early_stop": 0.0,
            "disp_mask_confine": False,
            "disp_mask_margin_um": 20.0,
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
            "fwd_mask_strength": 0.0,
            "fwd_smoothness": 0.05,
            "force_vector_stride": 20,
            "force_arrow_scale": 1.0,
            "f_max": 500.0,
            "threshold": 0.0,
            "dilation": 10,
            "smoothing_sigma": 10.0,
            "bism_regularization": -6.0,
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

    def reset_all_parameters(self):
        # Reset all values to the stub defaults (re-initialise from the class-level dict).
        defaults = _StubParameterManager()._values
        for name, value in defaults.items():
            self._values[name] = value
            self.parameter_changed.emit(name, value)


class _StubDataManager:
    def __init__(self):
        self._callbacks = []
        self.output_dir = None
        self.active_input_folder = None
        self.active_input_files = {}
        self.bead_stack = None
        self.reference = None
        self.cell_stack = None
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

    def set_active_inputs(self, folder, input_files):
        self.active_input_folder = Path(folder).expanduser() if folder else None
        self.active_input_files = dict(input_files or {})
        self.notify_changed()

    def clear_generated_results(self):
        # Mirror the real DataManager: drop derived results on experiment switch.
        changed = False
        for attr in (
            "displacement_results",
            "force_results",
            "stress_results",
            "mask_stack",
        ):
            if getattr(self, attr, None) is not None:
                changed = True
            setattr(self, attr, None)
        if changed:
            self.notify_changed()

    def mark_artifact_error(self, key, error):
        self.artifact_errors.append((key, error))

    def set_displacement_results(self, results, path=None, source="", dirty=False):
        self.displacement_results = results
        if self.force_results is not None:
            self.force_results = None
        if self.stress_results is not None:
            self.stress_results = None
        self.notify_changed()

    def set_force_results(self, results, path=None, source="", dirty=False):
        self.force_results = results
        if self.stress_results is not None:
            self.stress_results = None
        self.notify_changed()

    def set_stress_results(self, results, path=None, source="", dirty=False):
        self.stress_results = results
        self.notify_changed()


class _StubVisualizationManager:
    def __init__(self, viewer, data_manager):
        self.viewer = viewer
        self.data_manager = data_manager
        self.vector_stream_calls = []
        self.vector_stream_frames = []
        self.stress_stream_calls = []
        self.stress_stream_frames = []

    # Run-selected snapshots/restores layer visibility around the streaming
    # takeover (worklist §4); the stub has no real viewer, so these no-op.
    def capture_layer_visibility(self):
        return {}

    def restore_layer_visibility(self, snapshot):
        pass

    def begin_vector_field_stream(self, kind, num_frames, vis_params):
        self.vector_stream_calls.append((kind, num_frames, dict(vis_params)))

    def stream_vector_field_frame(self, kind, frame_index, field_frame):
        self.vector_stream_frames.append((kind, frame_index, field_frame))

    def begin_stress_stream(self, num_frames, max_stress, downscale_factor):
        self.stress_stream_calls.append((num_frames, max_stress, downscale_factor))

    def stream_stress_frame(self, frame_index, stress_tensor_frame):
        self.stress_stream_frames.append((frame_index, stress_tensor_frame))


class _StubController(QObject):
    progress_updated = Signal(int, str)
    ui_frozen = Signal(bool)


class _StubStageWidget(QWidget):
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
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.data_panel = QWidget()
        self.action_panel = QWidget()
        self.data_panel.setVisible(True)
        self.action_panel.setVisible(True)
        self.loaded_active_layers = []
        self.loaded_input_files = None
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

    def gcv_action(self):
        self.action_calls["gcv"] = self.action_calls.get("gcv", 0) + 1

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

    def load_input_files(self, folder, input_files):
        self.loaded_input_files = (folder, dict(input_files or {}))

    def load_result_artifact(self, key):
        self.loaded_files.append(key)


_ORIGINAL_MODULES = {}


def _stub_module(name, **attrs):
    if name not in _ORIGINAL_MODULES:
        _ORIGINAL_MODULES[name] = sys.modules.get(name)
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module


def _restore_stubbed_modules():
    """Undo the module-level stubs so they don't leak into other test files.

    ``_widget`` binds the stub classes at import time (eager top-level
    imports), so once it has been imported the real modules can be put back in
    ``sys.modules``. Without this, the bare stub modules — no ``__file__`` and
    missing the real symbols — shadow the genuine widgets for every test
    collected afterwards, so e.g. ``from ...displacement_analysis_widget import
    DisplacementController`` fails with "unknown location".
    """
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _ORIGINAL_MODULES.clear()


_stub_module(
    "napariTFM.utilities.parameter_manager",
    ParameterManager=_StubParameterManager,
)
_stub_module("napariTFM.utilities.data_manager", DataManager=_StubDataManager)
_stub_module(
    "napariTFM.utilities.visualization_manager",
    VisualizationManager=_StubVisualizationManager,
)
_stub_module(
    "napariTFM.widgets.displacement_analysis_widget",
    DisplacementAnalysisWidget=_StubStageWidget,
)
_stub_module("napariTFM.widgets.fttc_widget", FTTCWidget=_StubStageWidget)
_stub_module("napariTFM.widgets.stress_widget", StressWidget=_StubStageWidget)

from napariTFM.widgets import _widget

_restore_stubbed_modules()


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
        "Displacement",
        child,
        actions={
            "run": child.run_action,
            "preview": child.preview_action,
            "cancel": child.cancel_action,
        },
        action_states=child.action_states,
        action_states_changed=child.action_states_changed,
    )

    assert section.params_btn.objectName() == "stage_displacement_params_button"
    assert section.run_cancel_btn.objectName() == "stage_displacement_run_cancel_button"
    assert section.preview_button.objectName() == "stage_displacement_preview_button"

    assert "Run" in section.run_cancel_btn.toolTip()
    assert section.preview_button.toolTip() == "Preview Displacement"
    assert "Toggle" in section.params_btn.toolTip()


def test_stage_section_tracks_status(app):
    section = _widget.StageSection("Displacement", QWidget(), status="ready")
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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)
    widget = _widget.napariTFMWidget(object())
    widget.show()
    app.processEvents()
    _enter_tuning(widget)
    app.processEvents()

    assert widget.findChildren(QTabWidget) == []
    # Every stage body is always visible; only parameter panels collapse.
    assert widget.displacement_widget.isVisible()
    assert widget.force_widget.isVisible()
    assert widget.stress_widget.isVisible()


def test_stage_sections_receive_ordered_neighbour_accents(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)
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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)
    return _widget.napariTFMWidget(object())


def _enter_tuning(widget, path="/data/exp"):
    """Drive a stub shell into G2: project open + a selected experiment."""
    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_experiments([path])
    widget.experiments_list.set_active(path)
    return widget


def test_stage_sections_are_accented_by_pipeline_key_not_title(monkeypatch, app):
    """Each spine segment must take its stage key's ramp colour. Titles like
    "Force Analysis" slugify to "force_analysis" (not a ramp key), so deriving
    the accent from the title silently collapses Force/Stress to the fallback
    colour and the whole rail reads as one flat colour."""
    from napariTFM.widgets._ui_style import stage_accent

    widget = _stub_main_widget(monkeypatch)
    for key, section in widget._stage_sections_by_key.items():
        assert section._accent == stage_accent(key), key
    # And the four accents are actually distinct (a real gradient, not flat).
    accents = [s._accent for s in widget._stage_sections]
    assert len(set(accents)) == len(accents)


def test_stage_progress_feeds_one_global_status_label(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)

    # One panel-level status label, prefixed with the reporting stage (P2).
    assert hasattr(widget, "status_label")

    widget.displacement_widget.controller.progress_updated.emit(40, "Calculating…")
    assert widget.status_label.text() == "Displacement — Calculating…"

    widget.stress_widget.controller.progress_updated.emit(100, "Done")
    assert widget.status_label.text() == "Stress — Done"


def test_stage_progress_also_fills_the_matching_spine_node(monkeypatch, app):
    """Item #10: a running stage's pill rail node should fill frame by frame,
    not just sit on a flat dot until the run completes."""
    widget = _stub_main_widget(monkeypatch)
    section = widget._stage_sections_by_key["displacement"]
    section.set_status("running")

    widget.displacement_widget.controller.progress_updated.emit(40, "Processing frame 2/5")
    assert section.spine._progress == 0.4

    widget.displacement_widget.controller.progress_updated.emit(100, "Analysis completed successfully")
    assert section.spine._progress == 1.0


def test_run_selected_stage_progress_drives_status_and_fill(monkeypatch, app):
    """The run path (ViewerSink's on_stage_progress) drives the same spine
    nodes the live single-stage path does, including flipping running/done."""
    widget = _stub_main_widget(monkeypatch)
    section = widget._stage_sections_by_key["force"]
    assert section.status != "running"

    widget._on_run_selected_stage_progress("force", "running", 0.0)
    assert section.status == "running"
    assert section.spine._progress == 0.0

    widget._on_run_selected_stage_progress("force", "running", 0.5)
    assert section.spine._progress == 0.5

    widget._on_run_selected_stage_progress("force", "done", None)
    assert section.status == "done"
    # Finishing must not leave a stale partial fill behind on the next run.
    assert section.spine._progress is None


def test_run_selected_stage_progress_ignores_unknown_stage(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._on_run_selected_stage_progress("not_a_stage", "running", 0.5)  # must not raise


def _write_stage_ntfm(folder, **arrays):
    # Write a real .ntfm at the canonical resolve_output_plan location — exactly
    # where the batch writes and the status dots read (P3 truth source). In-place
    # mode (no processed_root) puts it in the experiment's processed/ bucket.
    import numpy as np  # noqa: F401  (kept local; arrays passed by caller)

    from napariTFM.utilities import ntfm
    from napariTFM.utilities.batch_output import experiment_ntfm_path

    ntfm_path = experiment_ntfm_path(str(folder), None)
    ntfm_path.parent.mkdir(parents=True, exist_ok=True)
    df = ntfm.arrays_to_tidy(grid_spacing=1.0, frame_interval=1.0, **arrays)
    ntfm.write_ntfm(ntfm_path, df, ntfm.build_metadata(config={}))


def test_experiment_stage_status_inputs_only_is_ready_frontier(monkeypatch, app, tmp_path):
    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(True)
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")

    statuses = widget._experiment_stage_status(str(folder))
    # Inputs present -> displacement (the first stage) is the ready frontier.
    assert statuses["displacement"] == "ready"
    assert statuses["force"] == "not_started"
    assert statuses["stress"] == "not_started"


def test_experiment_stage_status_displacement_only(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(True)
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")
    _write_stage_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))

    statuses = widget._experiment_stage_status(str(folder))
    # Displacement done; force is the next frontier.
    assert statuses["displacement"] == "done"
    assert statuses["force"] == "ready"
    assert statuses["stress"] == "not_started"


def test_experiment_stage_status_through_force(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(True)
    folder = tmp_path / "exp"
    folder.mkdir()
    _write_stage_ntfm(
        folder,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
    )

    statuses = widget._experiment_stage_status(str(folder))
    assert statuses["displacement"] == "done"
    assert statuses["force"] == "done"
    assert statuses["stress"] == "ready"


def test_experiment_stage_status_full_pipeline_all_done(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(True)
    folder = tmp_path / "exp"
    folder.mkdir()
    _write_stage_ntfm(
        folder,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
        stress_tensor=np.ones((1, 2, 2, 2, 2)),
    )

    statuses = widget._experiment_stage_status(str(folder))
    assert all(statuses[s] == "done" for s in ("displacement", "force", "stress"))


def test_stress_stage_is_disabled_by_default(monkeypatch, app):
    # Stress needs an external mask, so it stays off until the user opts in (D1).
    widget = _stub_main_widget(monkeypatch)
    assert widget._stage_sections_by_key["stress"].is_enabled is False
    assert widget._disabled_stages() == ["stress"]


def test_stress_mask_input_is_required(app):
    from napariTFM.widgets._widget import STAGE_DATA_ARTIFACTS

    mask_spec = next(
        spec for spec in STAGE_DATA_ARTIFACTS["stress"] if spec.key == "mask_stack"
    )
    assert mask_spec.required is True


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
    # Stress is exempt from auto-skip (D1); disabling it reads as off, not ready.
    assert statuses["stress"] == "off"
    assert statuses["force"] == "done"


def test_experiment_stage_status_reads_output_eagerly(monkeypatch, app, tmp_path):
    """Status is eager: a discovered row's dots reflect which measures the
    `.ntfm` actually carries, right away — no click needed. Reading that is a
    header-only walk, so it's fine to do for every row.
    """
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")
    _write_stage_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))

    statuses = widget._experiment_stage_status(str(folder))
    assert statuses["displacement"] == "done"      # its series is present on disk
    assert statuses["force"] == "ready"            # displacement done -> force ready
    assert statuses["stress"] == "off"             # disabled by default, takes priority


def test_stage_node_click_loads_only_that_stage_data(monkeypatch, app, tmp_path):
    """Clicking a stage circle decodes only that stage's series into the viewer
    (display-only); other stages' data stays unloaded. Selection alone loads no
    output, and the status dots are eager regardless of what's been clicked.
    """
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "exp"
    folder.mkdir()
    _write_stage_ntfm(
        folder,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
    )
    _select(widget, folder)
    try:
        # Selecting the experiment decodes no output series.
        assert widget.data_manager.displacement_results is None
        assert widget.data_manager.force_results is None

        widget._on_stage_node_clicked("force")

        # Only force's series was decoded; displacement stays unloaded.
        assert widget.data_manager.force_results is not None
        assert widget.data_manager.displacement_results is None
    finally:
        widget.close()
        widget.deleteLater()


def test_stage_node_click_narrates_load_in_status_label(monkeypatch, app, tmp_path):
    """Clicking a circle reads/decodes on the GUI thread, so it must tell the
    user what's happening: a 'loaded' confirmation for a stage with output, and
    a plain 'nothing to show' for a stage with none (so the click isn't silent).
    """
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "exp"
    folder.mkdir()
    _write_stage_ntfm(folder, force_field=np.ones((1, 2, 2, 2)) * 5.0)
    _select(widget, folder)
    try:
        widget._on_stage_node_clicked("force")
        assert "Force" in widget.status_label.text()
        assert "loaded" in widget.status_label.text().lower()

        # Displacement has no series in this container.
        widget._on_stage_node_clicked("displacement")
        assert "Displacement" in widget.status_label.text()
        assert "no" in widget.status_label.text().lower()
    finally:
        widget.close()
        widget.deleteLater()


class _StubResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _select(widget, folder):
    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_experiments([str(folder)])
    widget.experiments_list.set_active(str(folder))


def test_interactive_run_persists_ntfm_and_syncs_both_dot_rows(monkeypatch, app, tmp_path):
    import numpy as np

    from napariTFM.utilities.batch_output import experiment_ntfm_path

    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(True)
    folder = tmp_path / "pos_00"
    folder.mkdir()
    _select(widget, folder)
    try:
        # Displacement + force just ran interactively: results live in memory.
        scale = {"grid_spacing": 1.0, "time_interval": 1.0}
        widget.data_manager.displacement_results = _StubResult(
            displacement_field=np.ones((1, 3, 3, 2)), physical_scale=scale
        )
        widget.data_manager.force_results = _StubResult(
            force_field=np.ones((1, 3, 3, 2)) * 5.0, physical_scale=scale
        )

        # The force-finished hook auto-persists, then reconciles the dots.
        widget._on_stage_persisted("force")

        # Written at the same canonical path the batch uses (not preview-only).
        ntfm_path = experiment_ntfm_path(str(folder), None)
        assert ntfm_path.exists()

        # The on-disk truth now reports displacement + force done...
        statuses = widget._experiment_stage_status(str(folder))
        assert statuses["displacement"] == "done"
        assert statuses["force"] == "done"
        # ...and every section spine (below) reads that same per-stage verdict as
        # the experiment's row dots (above) — true top/bottom sync.
        for key, section in widget._stage_sections_by_key.items():
            assert section._effective_status() == statuses[key], key
    finally:
        widget.close()
        widget.deleteLater()


def test_all_nan_stage_reads_not_done_in_both_dot_rows(monkeypatch, app, tmp_path):
    import numpy as np

    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_00"
    folder.mkdir()
    _select(widget, folder)
    try:
        # Force came out all-NaN (e.g. a divide-by-zero run): displacement is real.
        scale = {"grid_spacing": 1.0, "time_interval": 1.0}
        widget.data_manager.displacement_results = _StubResult(
            displacement_field=np.ones((1, 3, 3, 2)), physical_scale=scale
        )
        widget.data_manager.force_results = _StubResult(
            force_field=np.full((1, 3, 3, 2), np.nan), physical_scale=scale
        )
        widget._on_stage_persisted("force")

        statuses = widget._experiment_stage_status(str(folder))
        # All-NaN force is honestly NOT done — the next-frontier "ready" instead.
        assert statuses["displacement"] == "done"
        assert statuses["force"] == "ready"
        # The section dot agrees (no green-below / grey-above split).
        assert widget._stage_sections_by_key["force"]._effective_status() == "ready"
    finally:
        widget.close()
        widget.deleteLater()


class _FakeBatchAnalysis:
    """Records its config and replays per-folder lifecycle to the callback."""

    last_config = None
    last_instance = None

    def __init__(self, config, progress_callback=None, sink=None):
        self.config = config
        self.progress_callback = progress_callback
        self.sink = sink
        type(self).last_config = config
        type(self).last_instance = self

    def process_all_folders(self):
        for folder in self.config["root_folders"]:
            if self.progress_callback:
                self.progress_callback(folder, "running")
                self.progress_callback(folder, "done")


def test_run_selected_builds_config_from_selection_and_runs_batch(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(True)
    widget.experiments_list.add_folders(
        ["/data/exp_a", "/data/exp_b"],
        input_files={"beads": "beads.tif", "reference": "reference.tif"},
        columns={"condition": "soft"},
    )
    widget.experiments_list.select_all()

    widget.experiments_list.run_selected_requested.emit()

    cfg = _FakeBatchAnalysis.last_config
    assert cfg["root_folders"] == ["/data/exp_a", "/data/exp_b"]
    assert cfg["experiment_metadata"]["/data/exp_a"] == {"condition": "soft"}
    assert cfg["analysis_steps"]["stress"] is True


def test_run_selected_runs_only_the_selected_rows(monkeypatch, app):
    """A partial selection (2 of 5) runs only those rows, in row order."""
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(
        ["/data/exp_a", "/data/exp_b", "/data/exp_c", "/data/exp_d", "/data/exp_e"]
    )
    # Select exp_b and exp_d only (out of insertion order to prove row-ordering).
    widget.experiments_list.set_active(
        "/data/exp_d", selection={"/data/exp_d", "/data/exp_b"}
    )

    widget.experiments_list.run_selected_requested.emit()

    assert _FakeBatchAnalysis.last_config["root_folders"] == [
        "/data/exp_b",
        "/data/exp_d",
    ]


def test_run_selected_honours_disabled_stress(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    widget = _stub_main_widget(monkeypatch)
    widget._stage_sections_by_key["stress"].set_enabled(False)
    widget.experiments_list.add_folders(["/data/exp_a"])
    widget.experiments_list.select_all()

    widget.experiments_list.run_selected_requested.emit()

    assert _FakeBatchAnalysis.last_config["analysis_steps"]["stress"] is False


def test_run_selected_progress_marks_running_then_refreshes(monkeypatch, app, tmp_path):
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
    widget.experiments_list.select_all()

    widget.experiments_list.run_selected_requested.emit()

    # running -> mark_running; done -> refresh from disk.
    assert ("run", "/data/exp_a") in seen
    assert ("refresh",) in seen


def test_stage_freeze_surfaces_cancel_button(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    section = widget._stage_sections_by_key["displacement"]

    # A frozen controller (run or preview in flight) pins the pill to 'running',
    # which is what flips the header button into the wired Cancel control.
    widget.displacement_widget.controller.ui_frozen.emit(True)
    assert section.status == "running"
    assert section.run_cancel_btn.isEnabled() is True
    assert "Cancel" in section.run_cancel_btn.toolTip()

    # Clicking the surfaced button while running invokes the cancel handler.
    section.run_cancel_btn.click()
    assert widget.displacement_widget.action_calls["cancel"] == 1


def test_stage_unfreeze_refreshes_statuses(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    refreshed = []
    monkeypatch.setattr(
        widget, "refresh_stage_statuses", lambda: refreshed.append(True)
    )

    widget.force_widget.controller.ui_frozen.emit(False)
    assert refreshed == [True]


def test_run_selected_toggles_cancel_button_and_cancels_batch(monkeypatch, app):
    cancelled = []

    class _CancellableBatch(_FakeBatchAnalysis):
        def process_all_folders(self):
            # Mid-run, the active button is a Cancel; clicking it must reach
            # the live batch's request_cancel.
            assert self.parent_list.run_selected_btn.text() == "Cancel"
            self.parent_list._on_run_selected_clicked()

        def request_cancel(self):
            cancelled.append(True)

    monkeypatch.setattr(_widget, "BatchAnalysis", _CancellableBatch)
    widget = _stub_main_widget(monkeypatch)
    _CancellableBatch.parent_list = widget.experiments_list
    widget.experiments_list.add_folders(["/data/exp_a"])
    widget.experiments_list.select_all()

    widget.experiments_list.run_selected_requested.emit()

    assert cancelled == [True]
    # The button is restored once the run returns.
    assert widget.experiments_list.run_selected_btn.text() == "Run selected"
    assert widget._active_batch is None


def test_run_selected_with_no_selection_is_a_noop(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    _FakeBatchAnalysis.last_config = None
    widget = _stub_main_widget(monkeypatch)
    # Rows exist but none are selected -> nothing to run.
    widget.experiments_list.add_folders(["/data/exp_a"])
    widget.experiments_list.set_active(None)

    widget.experiments_list.run_selected_requested.emit()

    assert _FakeBatchAnalysis.last_config is None


class _FakeParallelBatchAnalysis:
    """Stand-in for BatchAnalysis's non-blocking parallel API.

    Records ``start_parallel`` calls and lets tests drive
    ``poll_parallel_progress`` manually via a queue of canned
    ``(events, finished)`` results, instead of spinning up a real
    ProcessPoolExecutor.
    """

    last_instance = None

    def __init__(self, config, progress_callback=None, sink=None):
        self.config = config
        self.progress_callback = progress_callback
        self.sink = sink
        self.start_parallel_calls = []
        self.cancel_calls = 0
        self._poll_queue: list[tuple[list[tuple[str, str]], bool]] = []
        type(self).last_instance = self

    def start_parallel(self, plan, num_workers):
        self.start_parallel_calls.append((plan, num_workers))

    def request_cancel(self):
        self.cancel_calls += 1

    def queue_poll_result(self, events, finished, stage_events=None):
        self._poll_queue.append((events, finished, stage_events or []))

    def poll_parallel_progress(self):
        events, finished, stage_events = self._poll_queue.pop(0)
        for folder, status in events:
            if self.progress_callback:
                self.progress_callback(folder, status)
        return events, stage_events, finished


class _FakeTimer:
    """Stand-in for QTimer exposing the connected slot for manual driving.

    Avoids depending on a real Qt event loop / pytest-qt (not installed in
    this env) to exercise the polling loop in tests.
    """

    instances: list["_FakeTimer"] = []

    def __init__(self, *args, **kwargs):
        self._slot = None
        self.started = False
        self.stopped = False
        _FakeTimer.instances.append(self)

    def setInterval(self, _ms):
        pass

    @property
    def timeout(self):
        return self

    def connect(self, slot):
        self._slot = slot

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def fire(self):
        self._slot()


def test_run_selected_parallel_constructs_no_viewer_sink(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    def _boom(*a, **k):
        raise AssertionError("ViewerSink must not be constructed in parallel mode")

    monkeypatch.setattr(_widget, "ViewerSink", _boom)

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/exp_a", "/data/exp_b"])
    widget.experiments_list._num_workers_spinbox.setValue(4)

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    analyzer = _FakeParallelBatchAnalysis.last_instance
    assert analyzer is not None
    assert analyzer.sink is None
    assert analyzer.config["num_workers"] == 4
    assert len(analyzer.start_parallel_calls) == 1
    _, num_workers = analyzer.start_parallel_calls[0]
    assert num_workers == 4


def test_run_selected_parallel_follows_topmost_folder_when_none_selected(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    widget = _stub_main_widget(monkeypatch)
    assert widget._active_experiment is None
    widget.experiments_list.add_folders(["/data/exp_a", "/data/exp_b"])
    widget.experiments_list._num_workers_spinbox.setValue(2)

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    assert widget._active_experiment == "/data/exp_a"


def test_run_selected_parallel_leaves_existing_selection_alone(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/exp_a", "/data/exp_b"])
    widget._active_experiment = "/data/exp_b"
    widget.experiments_list._num_workers_spinbox.setValue(2)

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    assert widget._active_experiment == "/data/exp_b"


def test_run_selected_parallel_reloads_only_the_followed_folder_on_done(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/exp_a", "/data/exp_b"])
    widget._active_experiment = "/data/exp_b"
    widget.experiments_list._num_workers_spinbox.setValue(2)

    loaded = []
    monkeypatch.setattr(
        widget, "_load_stage_results", lambda path, stages: loaded.append(path)
    )
    refreshed = []
    monkeypatch.setattr(widget, "refresh_stage_statuses", lambda: refreshed.append(True))

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    analyzer = _FakeParallelBatchAnalysis.last_instance
    timer = _FakeTimer.instances[-1]
    assert timer.started is True

    # The non-followed folder finishes first: no reload should happen for it.
    refreshed.clear()
    analyzer.queue_poll_result([("/data/exp_a", "done")], False)
    timer.fire()
    assert loaded == []
    assert refreshed == []
    assert timer.stopped is False

    # The followed folder finishes: reload + refresh fire, and the run ends.
    # (refresh_stage_statuses fires twice here: once for the matched "done"
    # event, once more from teardown since this same poll also reports
    # finished=True.)
    analyzer.queue_poll_result([("/data/exp_b", "done")], True)
    timer.fire()
    assert loaded == ["/data/exp_b"]
    assert refreshed == [True, True]
    assert timer.stopped is True
    assert widget._active_batch is None
    assert widget.experiments_list.run_selected_btn.text() == "Run selected"


def test_run_selected_parallel_retargets_reload_when_followed_folder_changes_mid_run(monkeypatch, app):
    """Proves the poll loop compares against the LIVE ``self._active_experiment``
    on every tick, not a value captured once before the loop starts.

    A buggy "frozen variable" implementation (e.g. ``followed =
    self._active_experiment`` read once before entering the loop, then reused
    on every tick) would pass ``test_run_selected_parallel_reloads_only_the_followed_
    folder_on_done`` identically to the correct implementation, because that
    test never mutates ``self._active_experiment`` between polls. Here we do:
    folder A is followed and reloads once; the user then clicks row B mid-run
    (simulated by assigning ``widget._active_experiment`` directly, exactly as
    the existing, untouched ``_on_active_experiment_changed`` handler would); a
    later "done" event for the now-stale folder A must be ignored; a "done"
    event for the newly-followed folder B must still be handled. A
    frozen-variable implementation would keep reloading for A in the third
    step (it never re-reads the live attribute) and this test would fail
    there.
    """
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/exp_a", "/data/exp_b"])
    widget._active_experiment = "/data/exp_a"
    widget.experiments_list._num_workers_spinbox.setValue(2)

    loaded = []
    monkeypatch.setattr(
        widget, "_load_stage_results", lambda path, stages: loaded.append(path)
    )
    monkeypatch.setattr(widget, "refresh_stage_statuses", lambda: None)

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    analyzer = _FakeParallelBatchAnalysis.last_instance
    timer = _FakeTimer.instances[-1]

    # 1. A is the followed folder when it reports done -> reload happens.
    analyzer.queue_poll_result([("/data/exp_a", "done")], False)
    timer.fire()
    assert loaded == ["/data/exp_a"]

    # 2. The user clicks a different, already-finished/still-running row
    #    mid-run. In the real app this goes through active_changed ->
    #    _on_active_experiment_changed, which reassigns this same attribute;
    #    assigning it directly here is equivalent and avoids dragging that
    #    unrelated handler's side effects into this test.
    widget._active_experiment = "/data/exp_b"

    # 3. A later event for the now-stale, originally-followed folder A must be
    #    IGNORED -- this is the step a frozen-variable implementation would
    #    get wrong, since it would still be comparing against "/data/exp_a".
    analyzer.queue_poll_result([("/data/exp_a", "done")], False)
    timer.fire()
    assert loaded == ["/data/exp_a"]  # unchanged: no second reload for A

    # 4. The newly-followed folder B reporting done IS handled, proving the
    #    loop retargeted to wherever self._active_experiment currently points.
    analyzer.queue_poll_result([("/data/exp_b", "done")], True)
    timer.fire()
    assert loaded == ["/data/exp_a", "/data/exp_b"]


def test_run_selected_parallel_keeps_polling_after_cancel_until_finished(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/exp_a"])
    widget.experiments_list._num_workers_spinbox.setValue(2)

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    analyzer = _FakeParallelBatchAnalysis.last_instance
    timer = _FakeTimer.instances[-1]

    widget._cancel_run_selected()
    assert analyzer.cancel_calls == 1

    # An in-flight worker keeps reporting after cancel; the loop must keep
    # polling until "finished" actually flips True.
    analyzer.queue_poll_result([], False)
    timer.fire()
    assert timer.stopped is False
    assert widget._active_batch is analyzer

    analyzer.queue_poll_result([("/data/exp_a", "done")], True)
    timer.fire()
    assert timer.stopped is True
    assert widget._active_batch is None


def test_run_selected_parallel_routes_stage_events_to_row_progress(monkeypatch, app):
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeParallelBatchAnalysis)
    monkeypatch.setattr(_widget, "QTimer", _FakeTimer)
    _FakeTimer.instances = []

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/exp_a", "/data/exp_b"])
    widget.experiments_list._num_workers_spinbox.setValue(2)

    seen = []
    monkeypatch.setattr(
        widget.experiments_list, "set_row_stage_progress",
        lambda path, stage, status, fraction: seen.append((path, stage, status, fraction)),
    )

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    analyzer = _FakeParallelBatchAnalysis.last_instance
    timer = _FakeTimer.instances[-1]

    analyzer.queue_poll_result(
        [], False,
        stage_events=[
            ("/data/exp_a", "displacement", "running", 0.5),
            ("/data/exp_b", "force", "running", 0.25),
        ],
    )
    timer.fire()

    assert seen == [
        ("/data/exp_a", "displacement", "running", 0.5),
        ("/data/exp_b", "force", "running", 0.25),
    ]


def test_run_selected_sequential_path_is_unchanged_for_default_num_workers(monkeypatch, app):
    """num_workers <= 1 (including the spinbox default of 1) keeps the exact
    pre-existing sequential, ViewerSink-streaming path -- a regression guard
    against the new branch swallowing the default case."""
    monkeypatch.setattr(_widget, "BatchAnalysis", _FakeBatchAnalysis)
    sink_calls = []
    real_viewer_sink = _widget.ViewerSink

    def _spy_sink(*args, **kwargs):
        sink_calls.append((args, kwargs))
        return real_viewer_sink(*args, **kwargs)

    monkeypatch.setattr(_widget, "ViewerSink", _spy_sink)

    widget = _stub_main_widget(monkeypatch)
    assert widget.experiments_list.num_workers() == 1
    widget.experiments_list.add_folders(["/data/exp_a"])

    widget.experiments_list.select_all()
    widget.experiments_list.run_selected_requested.emit()

    assert len(sink_calls) == 1
    cfg = _FakeBatchAnalysis.last_config
    assert cfg["num_workers"] == 1
    assert _FakeBatchAnalysis.last_instance.sink is not None


def test_only_stress_stage_is_optional(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    assert widget._stage_sections_by_key["stress"].enable_btn is not None
    for key in ("displacement", "force"):
        assert widget._stage_sections_by_key[key].enable_btn is None


def test_disabling_stress_marks_it_off(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    stress = widget._stage_sections_by_key["stress"]

    stress.set_enabled(False)
    assert stress.spine._status == "off"


def test_main_widget_lets_dock_determine_width(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    assert widget.maximumWidth() > 500


def test_data_manager_change_callback_refreshes_stage_widgets(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    widget.data_manager.notify_changed()

    assert widget.displacement_widget.update_count == 1
    assert widget.force_widget.update_count == 1
    assert widget.stress_widget.update_count == 1


def test_main_widget_stage_headers_wire_existing_stage_actions(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.show()
    app.processEvents()
    # Stress is off by default; enable it so its header run button is live.
    widget._stage_sections_by_key["stress"].set_enabled(True)

    # Contract stages: header run invokes the widget's run_action handler.
    contract_cases = [
        ("displacement", widget.displacement_widget),
        ("force", widget.force_widget),
        ("stress", widget.stress_widget),
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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.show()
    app.processEvents()
    # Stress is off by default; enable it so its header preview button is live.
    widget._stage_sections_by_key["stress"].set_enabled(True)

    # Header preview invokes the stress widget's preview_action via the contract.
    widget.stress_widget.set_action_states(preview=True)
    app.processEvents()
    widget._stage_sections_by_key["stress"].preview_button.click()

    assert widget.stress_widget.action_calls["preview"] == 1


def test_main_widget_does_not_use_action_target_reflection(app):
    assert not hasattr(_widget.napariTFMWidget, "_find_stage_action_targets")
    assert not hasattr(_widget.napariTFMWidget, "_first_existing_widget")


def test_stage_sections_use_signal_action_contract(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

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
        "piv_passes",
        "piv_window",
        "young_modulus",
        "auto_gcv",
        "bism_regularization",
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

    manager.set_parameter("piv_window", 24)

    assert panel.parameter_controls["piv_window"].value() == 24


def test_main_widget_does_not_expose_legacy_parameter_panel(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    assert not hasattr(widget, "parameter_panel")
    # Project config moved into the experiments (aggregation) layer.
    assert "pixel_size" in widget.experiments_list.calibration_controls


def test_main_widget_experiments_layer_tracks_output_directory(monkeypatch, app, tmp_path):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)

    assert widget.experiments_list.output_dir_label.text() == str(tmp_path)


def test_main_widget_groups_parameters_inline_per_stage(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.show()
    app.processEvents()
    _enter_tuning(widget)
    app.processEvents()

    assert set(widget._stage_parameter_panels_by_key) == {
        "displacement",
        "force",
        "stress",
    }
    assert "batch" not in widget._stage_parameter_panels_by_key

    displacement_panel = widget._stage_parameter_panels_by_key["displacement"]
    force_panel = widget._stage_parameter_panels_by_key["force"]
    stress_panel = widget._stage_parameter_panels_by_key["stress"]

    # Calibration lives only in the Project section, not a stage panel.
    assert "pixel_size" not in displacement_panel.parameter_controls
    assert {"piv_window", "piv_passes"}.issubset(displacement_panel.parameter_controls)
    assert "young_modulus" not in displacement_panel.parameter_controls
    assert {"young_modulus", "auto_gcv"}.issubset(force_panel.parameter_controls)
    assert {"bism_regularization", "max_stress"}.issubset(stress_panel.parameter_controls)
    assert "threshold" not in stress_panel.parameter_controls

    displacement_section = widget._stage_sections_by_key["displacement"]
    assert displacement_section.parameter_panel is displacement_panel
    assert not displacement_panel.isVisibleTo(widget)
    displacement_section.params_btn.click()
    app.processEvents()
    assert displacement_panel.isVisibleTo(widget)


def test_stage_data_status_refreshes_from_data_manager(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    section = widget._stage_sections_by_key["displacement"]

    # With no experiment selected, the spine node's status is the in-memory
    # verdict derived from the data manager (compute_stage_status).
    assert section.status == "not_started"

    widget.data_manager.reference = object()
    widget.data_manager.bead_stack = object()
    widget.refresh_stage_statuses()

    assert section.status == "ready"

    widget.data_manager.displacement_results = object()
    widget.refresh_stage_statuses()

    assert section.status == "done"


def test_stage_status_is_done_when_results_are_in_memory(monkeypatch, app, tmp_path):
    # Preview-only: "done" follows in-memory results, not files on
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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    widget.data_manager.set_output_dir(tmp_path)
    section = widget._stage_sections_by_key["displacement"]

    widget.refresh_stage_statuses()
    assert section.status != "done"

    # Files on disk are irrelevant — only in-memory results count.
    (tmp_path / "TFM_data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "TFM_data" / "results.ome.tif").write_bytes(b"x")
    widget.refresh_stage_statuses()
    assert section.status != "done"

    widget.data_manager.set_displacement_results(
        _StubResult(displacement_field=np.ones((1, 2, 2, 2)),
                    physical_scale={"grid_spacing": 1.0, "time_interval": 1.0})
    )
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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _NoPanelStage)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())  # must not raise

    assert not hasattr(widget.displacement_widget, "parameter_panel")


def test_each_stage_has_single_inline_parameter_editor(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "DisplacementAnalysisWidget",
        "FTTCWidget", "StressWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())

    # Each stage with parameters mounts exactly one editor: the section's
    # first-class parameter_panel (no nested faux-stage duplication).
    for key in ("displacement", "force", "stress"):
        section = widget._stage_sections_by_key[key]
        assert section.parameter_panel is widget._stage_parameter_panels_by_key[key]
    assert "batch" not in widget._stage_parameter_panels_by_key
    assert "batch" not in widget._stage_sections_by_key
    assert not hasattr(widget, "_hide_embedded_parameter_panels")


def test_workflow_parameter_panel_labels_piv_controls(app):
    from qtpy.QtWidgets import QLabel

    manager = _StubParameterManager()
    panel = _widget.WorkflowParameterPanel(manager)

    labels = {label.text() for label in panel.findChildren(QLabel)}

    assert "Interrogation Window (px)" in labels
    assert "Passes" in labels
    # The Farneback controls are gone entirely.
    assert "Farneback Levels" not in labels
    assert "Farneback Iterations" not in labels


def test_refresh_updates_every_stage_widget_once(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "DisplacementAnalysisWidget",
        "FTTCWidget", "StressWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    stage_widgets = widget._stage_widgets()
    assert len(stage_widgets) == 3

    before = [w.update_count for w in stage_widgets]
    widget.refresh()
    after = [w.update_count for w in stage_widgets]
    assert all(a == b + 1 for a, b in zip(after, before))


def test_completion_signal_triggers_single_refresh(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    for name in (
        "DisplacementAnalysisWidget",
        "FTTCWidget", "StressWidget",
    ):
        monkeypatch.setattr(_widget, name, _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    calls = {"n": 0}
    original = widget.refresh
    widget.refresh = lambda: (calls.__setitem__("n", calls["n"] + 1), original())[1]

    widget.force_widget.force_calculated.emit(object())
    assert calls["n"] == 1


def test_calibration_change_updates_all_stage_widgets(monkeypatch, app):
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    before = [w.update_count for w in widget._stage_widgets()]
    widget.parameter_manager.set_parameter("gaussian_sigma", 5)
    after = [w.update_count for w in widget._stage_widgets()]
    assert after == before



def test_shell_theme_button_switches_palette(monkeypatch, app):
    from napariTFM.widgets import _widget
    from napariTFM.widgets import _ui_style
    monkeypatch.setattr(_widget, "DataManager", _StubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(_widget, "VisualizationManager", _StubVisualizationManager)
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    original = _ui_style.active_theme_name()
    try:
        widget = _widget.napariTFMWidget(object())
        assert hasattr(widget, "theme_btn")
        other = next(n for n in _ui_style.theme_names() if n != original)
        widget._on_theme_selected(other)
        assert _ui_style.active_theme_name() == other
        accent = _ui_style.stage_accent("displacement")
        assert _ui_style.muted_accent(accent) in widget._stage_sections_by_key["displacement"].header_label.styleSheet()
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
    # piv_window is an int param -> islider; disp_arrow_scale is float -> dslider.
    assert type(panel.parameter_controls["piv_window"]).__name__ == "QLabeledSlider"
    assert type(panel.parameter_controls["disp_arrow_scale"]).__name__ == "QLabeledDoubleSlider"


def test_displacement_panel_exposes_piv_controls(app):
    from qtpy.QtWidgets import QLabel

    from napariTFM.widgets._widget import WorkflowParameterPanel

    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Displacement",))

    # The PIV knobs are exposed as editable controls.
    for name in ("piv_window", "piv_passes"):
        assert name in panel.parameter_controls

    # The Farneback "Advanced" group is gone; only "Visualization" remains as a
    # sub-header inside the section.
    labels = {w.text() for w in panel.findChildren(QLabel)}
    assert "Visualization" in labels
    assert "Advanced" not in labels


def test_workflow_parameter_panel_slider_writes_through(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel

    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Displacement",))
    panel.parameter_controls["piv_window"].setValue(32)
    assert pm.get_ui_parameter("piv_window") == 32


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
    monkeypatch.setattr(_widget, "DisplacementAnalysisWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)

    widget = _widget.napariTFMWidget(object())
    section = widget._stage_sections_by_key["displacement"]
    # The panel is the section's first-class parameter_panel, not a nested section.
    assert section.parameter_panel is widget._stage_parameter_panels_by_key["displacement"]
    assert not hasattr(section, "add_inner_section")


def test_displacement_panel_excludes_general_calibration(app):
    from napariTFM.widgets._widget import WorkflowParameterPanel

    pm = _real_parameter_manager()
    panel = WorkflowParameterPanel(pm, section_titles=("Displacement",))
    # Calibration lives only in the Project/General section, not a stage panel.
    assert "pixel_size" not in panel.parameter_controls
    assert "frame_interval" not in panel.parameter_controls
    # Displacement-specific params are present.
    assert "piv_window" in panel.parameter_controls


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


def test_toolbar_exposes_project_and_parameter_buttons(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    # Project front-door buttons live on the brand row, icon-only now.
    assert widget.new_project_btn.toolTip() == "Start a new project"
    assert widget.load_project_btn.toolTip() == "Load a project"
    assert widget.save_project_btn.toolTip() == "Save project as…"
    # Parameter preset buttons, same row, grouped after a divider.
    assert widget.load_params_btn.toolTip() == "Load parameters preset"
    assert widget.save_params_btn.toolTip() == "Save parameters preset"
    assert widget.reset_params_btn.toolTip() == "Reset parameters"
    for button in (
        widget.new_project_btn, widget.load_project_btn, widget.save_project_btn,
        widget.load_params_btn, widget.save_params_btn, widget.reset_params_btn,
    ):
        assert not button.icon().isNull()
        assert button.text() == ""
    # The experiments list no longer owns its own series Open/Save.
    assert not hasattr(widget.experiments_list, "load_series_btn")
    assert not hasattr(widget.experiments_list, "save_series_btn")
    assert not hasattr(widget, "_save_config")
    assert not hasattr(widget, "_load_config")


def test_toolbar_buttons_share_the_title_row(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    assert not hasattr(widget, "toolbar_grid")


def _icon_image(button):
    # QIcon/QPixmap.cacheKey() is identity-based (a fresh render gets a new
    # key even for pixel-identical content), so it can't tell us whether two
    # *different* buttons carry the same icon — only whether one button's
    # icon was *replaced*. Render to a QImage and compare pixels instead.
    return button.icon().pixmap(18, 18).toImage()


def test_toolbar_buttons_use_correct_distinct_icons(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    # load_project_btn and load_params_btn intentionally share the "load" icon
    # (tooltip + grouping disambiguate); everything else should be distinct.
    assert _icon_image(widget.load_project_btn) == _icon_image(widget.load_params_btn)
    assert _icon_image(widget.save_project_btn) == _icon_image(widget.save_params_btn)
    distinct_images = [
        _icon_image(widget.new_project_btn),
        _icon_image(widget.load_project_btn),
        _icon_image(widget.save_project_btn),
        _icon_image(widget.reset_params_btn),
    ]
    for i, image_a in enumerate(distinct_images):
        for image_b in distinct_images[i + 1:]:
            assert image_a != image_b


def test_toolbar_icons_retint_on_theme_change(monkeypatch, app):
    from napariTFM.widgets._ui_style import theme_names, active_theme_name
    widget = _stub_main_widget(monkeypatch)
    before = widget.new_project_btn.icon().cacheKey()
    names = [n for n in theme_names() if n != active_theme_name()]
    assert names, "need at least one non-default theme to test retinting"
    widget._on_theme_selected(names[0])
    assert widget.new_project_btn.icon().cacheKey() != before


def test_save_params_writes_knobs_without_paths(monkeypatch, app, tmp_path):
    import yaml

    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(
        ["/data/a"], input_files={"beads": "beads.tif"}, columns={"day": "1"}
    )
    out = tmp_path / "tfm_params.yaml"
    monkeypatch.setattr(
        _widget.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._save_params()

    preset = yaml.safe_load(out.read_text())
    assert "parameters" in preset
    assert "young_modulus" in preset["parameters"]
    # The preset is a portable recipe: no dataset paths leak into it.
    assert "root_folders" not in preset
    assert "experiment_metadata" not in preset


def test_load_params_applies_knobs(monkeypatch, app, tmp_path):
    import yaml

    path = tmp_path / "tfm_params.yaml"
    path.write_text(yaml.safe_dump({"format_version": 1, "parameters": {"young_modulus": 9.0}}))

    widget = _stub_main_widget(monkeypatch)
    monkeypatch.setattr(
        _widget.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._load_params()

    assert widget.parameter_manager.get_parameter("young_modulus") == 9.0



def test_g0_hides_workspace_and_pipeline_until_project_open(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    # Default state is G0: no project open.
    assert widget._project_open is False
    assert not widget.experiments_list.isVisibleTo(widget)
    assert not widget._pipeline_context_label.isVisibleTo(widget)
    for section in widget._stage_sections:
        assert not section.isVisibleTo(widget)


def test_g1_reveals_workspace_but_not_stage_pills(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._project_open = True
    widget._update_disclosure()
    # G1: workspace + status visible; pipeline label + pills still hidden.
    assert widget.experiments_list.isVisibleTo(widget)
    assert widget.status_label.isVisibleTo(widget)
    assert not widget._pipeline_context_label.isVisibleTo(widget)
    for section in widget._stage_sections:
        assert not section.isVisibleTo(widget)


def test_g2_reveals_stage_pills_when_experiment_selected(monkeypatch, app):
    widget = _enter_tuning(_stub_main_widget(monkeypatch))
    # G2: a row is selected — pipeline label + every stage pill revealed.
    assert widget._pipeline_context_label.isVisibleTo(widget)
    for section in widget._stage_sections:
        assert section.isVisibleTo(widget)


def test_deselecting_experiment_drops_back_to_g1(monkeypatch, app):
    widget = _enter_tuning(_stub_main_widget(monkeypatch))
    widget.experiments_list.set_active(None)
    for section in widget._stage_sections:
        assert not section.isVisibleTo(widget)
    assert not widget._pipeline_context_label.isVisibleTo(widget)


def test_save_project_bundles_dataset_and_parameters(monkeypatch, app, tmp_path):
    import yaml

    widget = _stub_main_widget(monkeypatch)
    widget._project_open = True
    widget.experiments_list.add_folders(
        ["/data/a", "/data/b"],
        input_files={"beads": "beads.tif", "reference": "reference.tif"},
        columns={"day": "1"},
    )
    widget.parameter_manager.set_parameter("young_modulus", 9.0)
    out = tmp_path / "project.yaml"
    monkeypatch.setattr(
        _widget.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._save_project()

    config = yaml.safe_load(out.read_text())
    # Dataset side (reuses the series shape) ...
    assert config["root_folders"] == ["/data/a", "/data/b"]
    assert config["experiment_metadata"]["/data/a"] == {"day": "1"}
    assert config["input_files"]["beads"] == "beads.tif"
    assert "run_options" in config
    # ... plus the analysis recipe, all in one file.
    assert config["parameters"]["young_modulus"] == 9.0


def test_load_project_restores_dataset_and_parameters(monkeypatch, app, tmp_path):
    import yaml

    config = {
        "format_version": 2,
        "root_folders": ["/data/x", "/data/y"],
        "input_files": {"beads": "beads.tif", "reference": "reference.tif"},
        "experiment_metadata": {"/data/x": {"day": "1"}, "/data/y": {"day": "2"}},
        "run_options": {"disabled_stages": ["stress"], "processed_root": None},
        "parameters": {"young_modulus": 12.0},
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(config))

    widget = _stub_main_widget(monkeypatch)
    monkeypatch.setattr(
        _widget.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    monkeypatch.setattr(_widget.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    widget._load_project()

    assert widget._project_open is True
    assert widget.experiments_list.experiments() == ["/data/x", "/data/y"]
    assert widget.experiments_list.experiment_records()[1]["columns"] == {"day": "2"}
    assert widget.parameter_manager.get_parameter("young_modulus") == 12.0
    assert widget._stage_sections_by_key["stress"].is_enabled is False


def test_new_project_clears_to_empty_open_workspace(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.add_folders(["/data/a"])
    widget.parameter_manager.set_parameter("young_modulus", 9.0)

    widget._new_project()

    assert widget._project_open is True
    assert widget.experiments_list.experiments() == []
    # Stress returns to its default-off state on a clean slate.
    assert widget._stage_sections_by_key["stress"].is_enabled is False


def test_new_project_reexpands_setup_section_after_prior_collapse(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget.experiments_list.set_records([{"path": "/data/a", "input_files": {}, "columns": {}}])
    assert widget.experiments_list.setup_section.is_expanded is False
    widget._new_project()
    assert widget.experiments_list.setup_section.is_expanded is True


def test_autosave_path_is_gone(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    for attr in ("_write_config", "_read_config", "_reconcile_to_output_dir",
                 "_config_path", "_save_series", "_load_series"):
        assert not hasattr(widget, attr)


def test_experiment_rows_live_in_a_bounded_scroll_area(monkeypatch, app):
    from qtpy.QtWidgets import QScrollArea

    widget = _stub_main_widget(monkeypatch)
    scroll = widget.experiments_list._rows_scroll
    assert isinstance(scroll, QScrollArea)
    assert scroll.widgetResizable() is True
    # Capped so a long list scrolls instead of pushing the panel down.
    assert 0 < scroll.maximumHeight() <= 600


def test_column_header_fields_reflect_the_table_columns(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    el = widget.experiments_list
    el.add_folders(["/data/ctrl_a", "/data/ctrl_b"], columns={"condition": "Ctrl"})
    el.add_folders(["/data/ko_a"], columns={"condition": "KO"})

    # The shared, editable column header carries one field per column.
    assert el.column_names() == ["condition"]
    assert [f.text() for f in el._header_fields] == ["condition"]


def test_column_header_is_placeholder_without_columns(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    el = widget.experiments_list
    el.add_folders(["/data/a"])

    # No columns → no editable header fields, just the rows.
    assert el.column_names() == []
    assert el._header_fields == []


def test_selecting_experiment_points_disk_check_at_its_inputs(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._project_open = True
    widget.experiments_list.add_folders(
        ["/data/exp_a"],
        input_files={"beads": "b.tif", "reference": "r.tif"},
    )

    widget.experiments_list.set_active("/data/exp_a")

    dm = widget.data_manager
    assert dm.active_input_folder == Path("/data/exp_a")
    assert dm.active_input_files == {"beads": "b.tif", "reference": "r.tif"}


def test_selecting_experiment_loads_its_input_files(monkeypatch, app):
    # Pointing the disk check is not enough — Preview and Run need the arrays in
    # memory. Selecting an experiment must hand its folder + discovery filenames
    # to the displacement stage (the first stage) so it loads them from disk.
    widget = _stub_main_widget(monkeypatch)
    widget._project_open = True
    widget.experiments_list.add_folders(
        ["/data/exp_a"],
        input_files={"beads": "b.tif", "reference": "r.tif"},
    )

    widget.experiments_list.set_active("/data/exp_a")

    assert widget.displacement_widget.loaded_input_files == (
        "/data/exp_a",
        {"beads": "b.tif", "reference": "r.tif"},
    )


def test_deselecting_experiment_clears_disk_check(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._project_open = True
    widget.experiments_list.add_folders(
        ["/data/exp_a"], input_files={"beads": "b.tif"}
    )
    widget.experiments_list.set_active("/data/exp_a")

    widget.experiments_list.set_active(None)

    assert widget.data_manager.active_input_folder is None
    assert widget.data_manager.active_input_files == {}


def test_discover_tooltip_lists_only_filled_inputs(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    el = widget.experiments_list
    el.file_name_inputs["beads"].setText("beads.tif")
    el.file_name_inputs["reference"].setText("reference.tif")
    el.file_name_inputs["cells"].setText("")
    el.file_name_inputs["masks"].setText("")

    tip = el.add_btn.toolTip()
    assert "beads.tif" in tip and "reference.tif" in tip
    assert "and reference.tif" in tip  # two-item grammar
    assert "cells" not in tip and "masks" not in tip


def test_discover_tooltip_includes_present_optionals(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    el = widget.experiments_list
    el.file_name_inputs["beads"].setText("b.tif")
    el.file_name_inputs["reference"].setText("r.tif")
    el.file_name_inputs["cells"].setText("c.tif")
    el.file_name_inputs["masks"].setText("m.tif")

    tip = el.add_btn.toolTip()
    # Oxford-free list: "b.tif, r.tif, c.tif and m.tif"
    assert "b.tif, r.tif, c.tif and m.tif" in tip


def test_parameter_edit_marks_project_dirty(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._new_project()            # opens a clean (not dirty) project
    assert widget._dirty is False
    widget.parameter_manager.set_parameter("young_modulus", 7.0)
    assert widget._dirty is True


def test_new_project_on_dirty_workspace_asks_before_discarding(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._new_project()
    widget.experiments_list.add_folders(["/data/a"])  # marks dirty
    assert widget._dirty is True

    asked = {"n": 0}

    def _decline(*a, **k):
        asked["n"] += 1
        return _widget.QMessageBox.No

    monkeypatch.setattr(_widget.QMessageBox, "question", staticmethod(_decline))

    widget._new_project()
    # The user declined: the workspace is left intact.
    assert asked["n"] == 1
    assert widget.experiments_list.experiments() == ["/data/a"]


def test_new_project_proceeds_when_discard_confirmed(monkeypatch, app):
    widget = _stub_main_widget(monkeypatch)
    widget._new_project()
    widget.experiments_list.add_folders(["/data/a"])

    monkeypatch.setattr(
        _widget.QMessageBox, "question",
        staticmethod(lambda *a, **k: _widget.QMessageBox.Yes),
    )

    widget._new_project()
    assert widget.experiments_list.experiments() == []
