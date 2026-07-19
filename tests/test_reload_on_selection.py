"""Tests for decoding a processed experiment's output series into the viewer.

Selecting an experiment from the list loads its inputs only — it must NOT by
itself decode `TFMresults.ome.tif` into memory (that's display-only and waits
for a click). Loading is split into `_read_stage_arrays` (parse) +
`_apply_*_result` (apply) + `_load_stage_results` (orchestrate, only the
requested stages), driven on demand by `_on_stage_node_clicked` (a spine
circle) or `_on_row_stage_clicked` (a list dot). Status, by contrast, is eager.
"""
import sys
import types as _types
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QCheckBox, QPushButton, QWidget

# ---------------------------------------------------------------------------
# Stubs — mirroring test_workflow_shell.py but with extended DataManager
# ---------------------------------------------------------------------------


class _StubParameterManager(QObject):
    parameter_changed = Signal(str, object)
    parameters_reset = Signal(object)

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
            "ffd_early_stop": 0.0,
            "disp_mask_confine": False,
            "disp_mask_margin_um": 20.0,
            "downscale_factor": 4,
            "disp_downscale_before": False,
            "disp_vector_stride": 20,
            "disp_arrow_scale": 1.0,
            "d_max": 1.0,
            "young_modulus": 5.0,
            "poisson_ratio_substrate": 0.5,
            "gel_height": 0.0,
            "regularization": -4.0,
            "bayesian_l2": False,
            "l1_sparsity": 0.0,
            "fwd_mask_strength": 0.0,
            "fwd_mask_softness": 2.0,
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
        return self._values[name]

    def set_ui_parameter(self, name, value):
        self.ui_writes.append((name, value))
        self._values[name] = value
        self.parameter_changed.emit(name, value)

    def reset_all_parameters(self):
        defaults = _StubParameterManager()._values
        for name, value in defaults.items():
            self._values[name] = value
            self.parameter_changed.emit(name, value)


class _ExtendedStubDataManager:
    """DataManager stub that adds set_* result methods needed by the bug fix.

    Extends the minimal stub pattern used in test_workflow_shell.py to include
    set_displacement_results / set_force_results / set_stress_results so the
    real _load_stage_results method can call them without error.
    """

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
        for cb in list(self._callbacks):
            cb()

    # Alias used by the real DataManager
    def _notify_changed(self):
        self.notify_changed()

    def set_output_dir(self, path):
        self.output_dir = Path(path).expanduser() if path else None
        self.notify_changed()

    def set_active_inputs(self, folder, input_files):
        self.active_input_folder = Path(folder).expanduser() if folder else None
        self.active_input_files = dict(input_files or {})
        self.notify_changed()

    def clear_generated_results(self):
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

    # ---- result setters (key addition over the minimal stub) ----

    def set_displacement_results(self, results, path=None, source="", dirty=False):
        self.displacement_results = results
        # Mirror DataManager._invalidate_downstream behaviour.
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

    def mark_artifact_error(self, key, error):
        self.artifact_errors.append((key, error))

    def artifact_available(self, key):
        return False

    def has_valid_output_dir(self):
        return self.output_dir is not None and self.output_dir.exists()


class _StubVisualizationManager:
    def __init__(self, viewer, data_manager):
        self.viewer = viewer
        self.data_manager = data_manager
        self.vector_stream_calls = []
        self.vector_stream_frames = []
        self.stress_stream_calls = []
        self.stress_stream_frames = []
        self.display_reference_shape = None

    def set_display_reference_shape(self, shape):
        self.display_reference_shape = shape

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

    def set_displacement_loader(self, loader):
        self.displacement_loader = loader

    def set_force_loader(self, loader):
        self.force_loader = loader


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
        self.loaded_mask_paths = []
        self.update_count = 0
        self.action_calls = {"run": 0, "preview": 0, "cancel": 0}
        self._action_states = {"run": False, "preview": False}

    def load_mask_from_file(self, mask_path, beads_shape=None):
        self.loaded_mask_paths.append(str(mask_path))
        self.loaded_mask_beads_shape = beads_shape
        return True

    def peek_input_xy_shape(self, folder, input_files, slot="beads"):
        return None

    def _update_ui_state(self):
        self.update_count += 1

    def set_displacement_available_check(self, check):
        self.displacement_available_check = check

    def set_force_available_check(self, check):
        self.force_available_check = check

    def run_action(self):
        self.action_calls["run"] += 1

    def preview_action(self):
        self.action_calls["preview"] += 1

    def cancel_action(self):
        self.action_calls["cancel"] += 1

    def bayesian_action(self):
        self.action_calls["bayesian"] = self.action_calls.get("bayesian", 0) + 1

    def action_states(self):
        return dict(self._action_states)

    def set_experiment_records(self, records):
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
    module = _types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module


def _restore_stubbed_modules():
    """Undo the module-level stubs so they don't leak into other test files.

    ``_widget`` binds the stub classes at import time, so once it has been
    imported the real modules can be put back in ``sys.modules``. Without this,
    the bare stub modules shadow the genuine widgets for every test collected
    afterwards (e.g. the streaming controllers fail to import with "unknown
    location"). This supersedes the older per-symbol workaround of leaving
    individual modules unstubbed.
    """
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _ORIGINAL_MODULES.clear()


# Install stubs before importing _widget so it picks them up.
_stub_module(
    "napariTFM.utilities.parameter_manager",
    ParameterManager=_StubParameterManager,
)
_stub_module(
    "napariTFM.utilities.data_manager",
    DataManager=_ExtendedStubDataManager,
)
_stub_module(
    "napariTFM.utilities.visualization_manager",
    VisualizationManager=_StubVisualizationManager,
)
_stub_module(
    "napariTFM.widgets.preprocessing_widget",
    PreprocessingWidget=_StubStageWidget,
)
_stub_module(
    "napariTFM.widgets.displacement_analysis_widget",
    DisplacementAnalysisWidget=_StubStageWidget,
)
# stress_widget.py imports ParameterCategory which is not in our stub, so stub
# the entire module to avoid that import path.
_stub_module("napariTFM.widgets.stress_widget", StressWidget=_StubStageWidget)
# fttc_widget is NOT stubbed at module level so that test_force_ownership.py
# (collected after this file) sees the real module.
# _stub_main_widget patches FTTCWidget per-test via monkeypatch instead.

from napariTFM.widgets import _widget  # noqa: E402 — stubs must be in place first

_restore_stubbed_modules()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_main_widget(monkeypatch):
    """Create a napariTFMWidget with all heavy sub-widgets replaced by stubs."""
    monkeypatch.setattr(_widget, "DataManager", _ExtendedStubDataManager)
    monkeypatch.setattr(_widget, "ParameterManager", _StubParameterManager)
    monkeypatch.setattr(
        _widget, "VisualizationManager", _StubVisualizationManager
    )
    monkeypatch.setattr(
        _widget, "DisplacementAnalysisWidget", _StubStageWidget
    )
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "StressWidget", _StubStageWidget)
    return _widget.napariTFMWidget(object())


def _write_ntfm(folder, config=None, **arrays):
    """Write a .ntfm at the canonical in-place output path."""
    from napariTFM.utilities import ntfm
    from napariTFM.utilities.batch_output import experiment_ntfm_path

    cfg = config or {}
    pixel_size = float(cfg.get("pixel_size", 1.0))
    downscale_factor = float(cfg.get("downscale_factor", 1))
    frame_interval = float(cfg.get("frame_interval", 1.0))

    ntfm_path = experiment_ntfm_path(str(folder), None)
    ntfm_path.parent.mkdir(parents=True, exist_ok=True)
    df = ntfm.arrays_to_tidy(
        grid_spacing=pixel_size * downscale_factor,
        frame_interval=frame_interval,
        **arrays,
    )
    ntfm.write_ntfm(ntfm_path, df, ntfm.build_metadata(config=cfg))
    return ntfm_path


def _select(widget, folder):
    """Drive a stub shell widget into state G2 (project open + experiment selected)."""
    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_experiments([str(folder)])
    widget.experiments_list.set_active(str(folder))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# Tests: _load_stage_results directly
# ---------------------------------------------------------------------------


def test_load_displacement_sets_results(monkeypatch, app, tmp_path):
    """Selecting an experiment with a displacement .ntfm loads it into memory."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_00"
    folder.mkdir()

    disp = np.ones((1, 3, 4, 2)) * 0.5
    _write_ntfm(
        folder,
        config={"pixel_size": 0.1, "downscale_factor": 4, "frame_interval": 2.0},
        displacement_field=disp,
    )

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    assert widget.data_manager.displacement_results is not None


def test_loaded_displacement_field_matches_stored(monkeypatch, app, tmp_path):
    """The in-memory displacement_field equals the stored array within tolerance."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_01"
    folder.mkdir()

    disp = np.linspace(0.1, 2.0, num=1 * 2 * 3 * 2).reshape(1, 2, 3, 2)
    _write_ntfm(
        folder,
        config={"pixel_size": 0.1, "downscale_factor": 4, "frame_interval": 2.0},
        displacement_field=disp,
    )

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    loaded = widget.data_manager.displacement_results.displacement_field
    np.testing.assert_allclose(loaded, disp, rtol=1e-5, atol=1e-7)


def test_loaded_result_physical_scale_is_usable(monkeypatch, app, tmp_path):
    """The physical_scale dict carries grid_spacing and time_interval."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_02"
    folder.mkdir()

    _write_ntfm(
        folder,
        config={"pixel_size": 0.1, "downscale_factor": 4, "frame_interval": 2.0},
        displacement_field=np.ones((1, 2, 2, 2)),
    )

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    scale = widget.data_manager.displacement_results.physical_scale
    assert "grid_spacing" in scale
    assert "time_interval" in scale
    assert scale["grid_spacing"] == pytest.approx(0.4, rel=1e-4)
    assert scale["time_interval"] == pytest.approx(2.0, rel=1e-4)


def test_no_ntfm_is_noop_no_exception(monkeypatch, app, tmp_path):
    """When no .ntfm exists, the method returns silently; results stay None."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "empty_exp"
    folder.mkdir()

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    assert widget.data_manager.displacement_results is None
    assert widget.data_manager.force_results is None
    assert widget.data_manager.stress_results is None


def test_load_force_sets_force_results(monkeypatch, app, tmp_path):
    """A .ntfm with displacement + force populates both in-memory results."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_03"
    folder.mkdir()

    disp = np.ones((1, 2, 3, 2)) * 0.3
    force = np.ones((1, 2, 3, 2)) * 100.0
    _write_ntfm(folder, displacement_field=disp, force_field=force)

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    assert widget.data_manager.displacement_results is not None
    assert widget.data_manager.force_results is not None
    np.testing.assert_allclose(
        widget.data_manager.force_results.force_field,
        force,
        rtol=1e-5,
    )


def test_load_force_only_keeps_displacement_resident_without_streaming(monkeypatch, app, tmp_path):
    """Decoding only the Force series (a Force-circle click) must make its displacement
    INPUT resident in memory — so Force Preview/Run enable — WITHOUT streaming the
    displacement to the viewer (display stays lazy). Regression: loading ["force"] used
    to leave displacement_results None, greying Force even though it was on disk.
    """
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_force_only"
    folder.mkdir()

    disp = np.ones((1, 2, 3, 2)) * 0.3
    force = np.ones((1, 2, 3, 2)) * 100.0
    _write_ntfm(folder, displacement_field=disp, force_field=force)

    widget._load_stage_results(str(folder), ["force"])

    # Resident in memory (the enabler for Force Preview/Run)…
    assert widget.data_manager.displacement_results is not None
    np.testing.assert_allclose(
        widget.data_manager.displacement_results.displacement_field, disp, rtol=1e-5
    )
    # …but only force was streamed to the viewer — displacement display stays lazy.
    streamed = [kind for kind, *_ in widget.visualization_manager.vector_stream_calls]
    assert "force" in streamed
    assert "displacement" not in streamed


def test_displacement_available_reads_disk_without_decoding(monkeypatch, app, tmp_path):
    """Force enables from disk: displacement done on disk but NOT resident makes
    _displacement_available() True via a header-only check, while displacement_results
    stays None (no pixel decode on selection)."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_disk"
    folder.mkdir()
    _write_ntfm(folder, displacement_field=np.ones((1, 2, 3, 2)) * 0.3)

    _select(widget, folder)  # selects + loads inputs, but decodes no output

    assert widget.data_manager.displacement_results is None   # not resident
    assert widget._displacement_available() is True           # yet visible on disk


def test_displacement_available_false_when_absent(monkeypatch, app, tmp_path):
    """No displacement on disk and none resident ⇒ not available (Force stays greyed)."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_empty_avail"
    folder.mkdir()

    _select(widget, folder)

    assert widget._displacement_available() is False


def test_ensure_displacement_resident_loads_from_disk_data_only(monkeypatch, app, tmp_path):
    """The on-demand loader (fired from Force Preview/Run) pulls displacement off disk
    into memory without streaming it to the viewer."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_ondemand"
    folder.mkdir()
    disp = np.ones((1, 2, 3, 2)) * 0.42
    _write_ntfm(folder, displacement_field=disp)

    _select(widget, folder)
    assert widget.data_manager.displacement_results is None

    assert widget._ensure_displacement_resident() is True
    np.testing.assert_allclose(
        widget.data_manager.displacement_results.displacement_field, disp, rtol=1e-5
    )
    streamed = [kind for kind, *_ in widget.visualization_manager.vector_stream_calls]
    assert "displacement" not in streamed


def test_force_available_reads_disk_without_decoding(monkeypatch, app, tmp_path):
    """Stress enables from disk: force done on disk but NOT resident makes
    _force_available() True via a header-only check, while force_results
    stays None (no pixel decode on selection)."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_disk_force"
    folder.mkdir()
    _write_ntfm(folder, force_field=np.ones((1, 2, 3, 2)) * 100.0)

    _select(widget, folder)  # selects + loads inputs, but decodes no output

    assert widget.data_manager.force_results is None   # not resident
    assert widget._force_available() is True            # yet visible on disk


def test_force_available_false_when_absent(monkeypatch, app, tmp_path):
    """No force on disk and none resident ⇒ not available (Stress stays greyed)."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_empty_avail_force"
    folder.mkdir()

    _select(widget, folder)

    assert widget._force_available() is False


def test_ensure_force_resident_loads_from_disk_data_only(monkeypatch, app, tmp_path):
    """The on-demand loader (fired from Stress Preview/Run) pulls force off disk
    into memory without streaming it to the viewer."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_ondemand_force"
    folder.mkdir()
    force = np.ones((1, 2, 3, 2)) * 42.0
    _write_ntfm(folder, force_field=force)

    _select(widget, folder)
    assert widget.data_manager.force_results is None

    assert widget._ensure_force_resident() is True
    np.testing.assert_allclose(
        widget.data_manager.force_results.force_field, force, rtol=1e-5
    )
    streamed = [kind for kind, *_ in widget.visualization_manager.vector_stream_calls]
    assert "force" not in streamed


def test_loaded_results_carry_reconstructed_parameters(monkeypatch, app, tmp_path):
    """The viewer reads result.parameters.downscale_factor etc.; it must be a
    real dataclass reconstructed from the stored config, not None."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_params"
    folder.mkdir()

    cfg = {
        "pixel_size": 0.13,
        "downscale_factor": 8,
        "frame_interval": 3.0,
        "d_max": 2.5,
        "f_max": 750.0,
        "max_stress": 4.0,
    }
    _write_ntfm(
        folder,
        config=cfg,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 10.0,
        stress_tensor=np.ones((1, 2, 2, 2, 2)) * 0.1,
    )

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    disp_params = widget.data_manager.displacement_results.parameters
    force_params = widget.data_manager.force_results.parameters
    stress_params = widget.data_manager.stress_results.parameters

    # None of them may be None — the viewer dereferences these.
    assert disp_params is not None
    assert force_params is not None
    assert stress_params is not None

    # The stored config values round-trip into the stage parameters the viewer reads.
    assert disp_params.downscale_factor == 8
    assert disp_params.d_max == pytest.approx(2.5)
    assert force_params.downscale_factor == 8
    assert force_params.f_max == pytest.approx(750.0)
    assert stress_params.downscale_factor == 8
    assert stress_params.max_stress == pytest.approx(4.0)


def test_loaded_parameters_default_when_config_empty(monkeypatch, app, tmp_path):
    """An empty/absent config still yields usable (default) parameters, not None,
    so the viewer never AttributeErrors on result.parameters.downscale_factor."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_noconfig"
    folder.mkdir()

    # _write_ntfm defaults to config={} when none is passed.
    _write_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    params = widget.data_manager.displacement_results.parameters
    assert params is not None
    # downscale_factor is the attribute the viewer dereferences first.
    assert params.downscale_factor is not None


def test_load_stress_sets_stress_results(monkeypatch, app, tmp_path):
    """A .ntfm with all stages populates displacement, force, and stress."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_04"
    folder.mkdir()

    disp = np.ones((1, 2, 2, 2)) * 0.2
    force = np.ones((1, 2, 2, 2)) * 50.0
    stress = np.ones((1, 2, 2, 2, 2)) * 0.01
    _write_ntfm(folder, displacement_field=disp, force_field=force, stress_tensor=stress)

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    assert widget.data_manager.stress_results is not None
    np.testing.assert_allclose(
        widget.data_manager.stress_results.stress_tensor,
        stress,
        rtol=1e-5,
    )


def test_all_nan_displacement_stays_none(monkeypatch, app, tmp_path):
    """An all-NaN displacement stage is not restored (same as populated_measures logic).

    In the OME-TIFF container an all-NaN stage is simply never written as a series,
    so it can't exist *alone* (there'd be nothing to write). We pair it with a real
    force stage — a realistic force-only container — and assert the absent
    displacement still reloads as None while force loads.
    """
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "nan_exp"
    folder.mkdir()

    nan_disp = np.full((1, 2, 2, 2), np.nan)
    real_force = np.ones((1, 2, 2, 2))
    _write_ntfm(folder, displacement_field=nan_disp, force_field=real_force)

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    assert widget.data_manager.displacement_results is None
    assert widget.data_manager.force_results is not None


# ---------------------------------------------------------------------------
# Tests: restored results also stream into the viewer (not just DataManager)
# ---------------------------------------------------------------------------


def test_load_displacement_streams_to_viewer(monkeypatch, app, tmp_path):
    """Restoring displacement pushes every frame through the same
    begin/stream_vector_field_frame calls a live run uses, so the viewer
    isn't left empty after selecting an already-processed experiment."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_viz_disp"
    folder.mkdir()

    disp = np.ones((2, 2, 2, 2)) * 0.5
    _write_ntfm(
        folder,
        config={"pixel_size": 0.1, "downscale_factor": 4, "d_max": 1.5},
        displacement_field=disp,
    )

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    viz = widget.visualization_manager
    assert len(viz.vector_stream_calls) == 1
    kind, num_frames, vis_params = viz.vector_stream_calls[0]
    assert kind == "displacement"
    assert num_frames == 2
    assert vis_params["v_max"] == pytest.approx(1.5)
    assert vis_params["downscale_factor"] == 4

    assert [f[1] for f in viz.vector_stream_frames] == [0, 1]
    assert all(f[0] == "displacement" for f in viz.vector_stream_frames)


def test_load_force_streams_to_viewer(monkeypatch, app, tmp_path):
    """Restoring force pushes every frame into the 'force' vector-field stream."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_viz_force"
    folder.mkdir()

    disp = np.ones((1, 2, 2, 2)) * 0.3
    force = np.ones((1, 2, 2, 2)) * 100.0
    _write_ntfm(
        folder,
        config={"f_max": 250.0, "downscale_factor": 4},
        displacement_field=disp,
        force_field=force,
    )

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    viz = widget.visualization_manager
    force_calls = [c for c in viz.vector_stream_calls if c[0] == "force"]
    assert len(force_calls) == 1
    _, num_frames, vis_params = force_calls[0]
    assert num_frames == 1
    assert vis_params["v_max"] == pytest.approx(250.0)

    force_frames = [f for f in viz.vector_stream_frames if f[0] == "force"]
    assert len(force_frames) == 1


def test_load_stress_streams_to_viewer(monkeypatch, app, tmp_path):
    """Restoring stress pushes every frame into the stress stream, using the
    force grid's downscale_factor (matching stress_widget's live-run lookup)."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "pos_viz_stress"
    folder.mkdir()

    disp = np.ones((1, 2, 2, 2)) * 0.2
    force = np.ones((1, 2, 2, 2)) * 50.0
    stress = np.ones((1, 2, 2, 2, 2)) * 0.01
    _write_ntfm(
        folder,
        config={"max_stress": 3.0, "downscale_factor": 6},
        displacement_field=disp,
        force_field=force,
        stress_tensor=stress,
    )

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    viz = widget.visualization_manager
    assert len(viz.stress_stream_calls) == 1
    num_frames, max_stress, downscale_factor = viz.stress_stream_calls[0]
    assert num_frames == 1
    assert max_stress == pytest.approx(3.0)
    # Sourced from force_results.parameters.downscale_factor, not stress_params.
    assert downscale_factor == widget.data_manager.force_results.parameters.downscale_factor

    assert len(viz.stress_stream_frames) == 1


def test_no_ntfm_streams_nothing(monkeypatch, app, tmp_path):
    """No .ntfm on disk means no viewer streaming calls at all."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "empty_exp_viz"
    folder.mkdir()

    widget._load_stage_results(str(folder), widget._NTFM_STAGES)

    viz = widget.visualization_manager
    assert viz.vector_stream_calls == []
    assert viz.stress_stream_calls == []


# ---------------------------------------------------------------------------
# Tests: full selection path via _on_active_experiment_changed
# ---------------------------------------------------------------------------


def test_selecting_experiment_does_not_load_displacement(monkeypatch, app, tmp_path):
    """Selecting an experiment alone must not read its `.ntfm` (P3-lazy data)."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "sel_00"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")

    disp = np.ones((2, 3, 4, 2)) * 1.5
    _write_ntfm(folder, displacement_field=disp)

    _select(widget, folder)

    assert widget.data_manager.displacement_results is None


def test_clicking_displacement_node_loads_displacement(monkeypatch, app, tmp_path):
    """Clicking the displacement stage's dot is what loads its array into memory."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "sel_00b"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")

    disp = np.ones((2, 3, 4, 2)) * 1.5
    _write_ntfm(folder, displacement_field=disp)

    _select(widget, folder)
    widget._on_stage_node_clicked("displacement")

    assert widget.data_manager.displacement_results is not None
    np.testing.assert_allclose(
        widget.data_manager.displacement_results.displacement_field,
        disp,
        rtol=1e-5,
    )


def test_selecting_experiment_auto_loads_discovered_mask(monkeypatch, app, tmp_path):
    """Selecting an experiment whose folder has masks.tif loads it into memory.

    The mask is an external Stress input; auto-loading it on selection (mirroring
    beads/reference) is what enables the Stress Run/Preview buttons without a
    manual layer load.
    """
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "sel_mask"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")
    (folder / "masks.tif").write_bytes(b"x")
    _write_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))

    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_records([{
        "path": str(folder),
        "input_files": {"beads": "beads.tif", "reference": "reference.tif", "masks": "masks.tif"},
        "columns": {},
    }])
    widget.experiments_list.set_active(str(folder))

    assert widget.stress_widget.loaded_mask_paths == [str(folder / "masks.tif")]


def test_selecting_experiment_without_mask_does_not_load(monkeypatch, app, tmp_path):
    """No masks.tif discovered → no mask auto-load attempt (no-op, not an error)."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "sel_nomask"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")
    _write_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))

    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_records([{
        "path": str(folder),
        "input_files": {"beads": "beads.tif", "reference": "reference.tif"},
        "columns": {},
    }])
    widget.experiments_list.set_active(str(folder))

    assert widget.stress_widget.loaded_mask_paths == []


def test_deselection_clears_results(monkeypatch, app, tmp_path):
    """Clearing the active experiment drops the loaded in-memory results."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "sel_01"
    folder.mkdir()

    _write_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))
    _select(widget, folder)
    widget._on_stage_node_clicked("displacement")
    assert widget.data_manager.displacement_results is not None

    # Clearing active experiment should drop results.
    widget.experiments_list.set_active(None)
    assert widget.data_manager.displacement_results is None


def test_switching_experiments_does_not_auto_load(monkeypatch, app, tmp_path):
    """Switching experiments must not leak stale data or auto-load the new one.

    Only a circle click decodes a `.ntfm` into the viewer (display-only) —
    switching to a position that was never clicked into leaves its output
    unloaded, exactly like switching to it directly would.
    """
    widget = _stub_main_widget(monkeypatch)

    folder_a = tmp_path / "exp_a"
    folder_a.mkdir()
    disp_a = np.ones((1, 2, 2, 2)) * 1.0
    _write_ntfm(folder_a, displacement_field=disp_a)

    folder_b = tmp_path / "exp_b"
    folder_b.mkdir()
    disp_b = np.ones((1, 2, 2, 2)) * 9.9
    _write_ntfm(folder_b, displacement_field=disp_b)

    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_experiments([str(folder_a), str(folder_b)])
    widget.experiments_list.set_active(str(folder_a))
    widget._on_stage_node_clicked("displacement")

    np.testing.assert_allclose(
        widget.data_manager.displacement_results.displacement_field,
        disp_a,
        rtol=1e-5,
    )

    widget.experiments_list.set_active(str(folder_b))

    # Neither exp_a's stale data nor an eager read of exp_b's own .ntfm.
    assert widget.data_manager.displacement_results is None


def test_reselecting_an_experiment_does_not_replay_data_until_clicked_again(monkeypatch, app, tmp_path):
    """Loaded output does not survive a round trip: switching away drops it, and
    switching back loads inputs only. The data comes back when the circle is
    clicked again — display is always on demand, never auto-restored.
    """
    widget = _stub_main_widget(monkeypatch)

    folder_a = tmp_path / "exp_a2"
    folder_a.mkdir()
    disp_a = np.ones((1, 2, 2, 2)) * 1.0
    _write_ntfm(folder_a, displacement_field=disp_a)

    folder_b = tmp_path / "exp_b2"
    folder_b.mkdir()

    widget._project_open = True
    widget._update_disclosure()
    widget.experiments_list.set_experiments([str(folder_a), str(folder_b)])
    widget.experiments_list.set_active(str(folder_a))
    widget._on_stage_node_clicked("displacement")
    assert widget.data_manager.displacement_results is not None

    widget.experiments_list.set_active(str(folder_b))
    widget.experiments_list.set_active(str(folder_a))
    # Back on A, but nothing decoded until the circle is clicked again.
    assert widget.data_manager.displacement_results is None

    widget._on_stage_node_clicked("displacement")
    np.testing.assert_allclose(
        widget.data_manager.displacement_results.displacement_field,
        disp_a,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Test: Force widget Run button enabled after selection
# ---------------------------------------------------------------------------


def test_force_run_enabled_after_selection_with_displacement(app):
    """FTTCWidget._update_ui_state enables Run when displacement_results is not None.

    This is the contract the bug fix satisfies: after _load_stage_results
    stores the displacement result, FTTCWidget enables its Run button.

    The real FTTCWidget is loaded via importlib so the module-level stub for
    napariTFM.widgets.fttc_widget (installed above to allow _widget to be
    imported cheaply) does not interfere.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_real_fttc_widget",
        Path(__file__).parent.parent / "napariTFM" / "widgets" / "fttc_widget.py",
    )
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)

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

    class _FakeParameterManager:
        def __init__(self):
            from qtpy.QtCore import QObject, Signal

            class _PM(QObject):
                parameter_changed = Signal(str, object)
                parameters_reset = Signal(object)

            self._pm = _PM()
            self.parameter_changed = self._pm.parameter_changed
            self.parameters_reset = self._pm.parameters_reset

        def get_fttc_parameters(self):
            return object()

    class _TrackingDataManager:
        displacement_results = None
        force_results = None

    dm = _TrackingDataManager()
    force_widget = fw.FTTCWidget(
        viewer=_FakeViewer(),
        data_manager=dm,
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )

    # Before: no displacement → Run disabled.
    assert force_widget.action_states()["run"] is False

    # Simulate _load_stage_results storing a displacement result
    # (the bug fix path).  The widget observes data_manager via the data_updated
    # signal fired by set_displacement_results.  Here we skip the signal and
    # call _update_ui_state directly, which mirrors what the real DataManager
    # does via its change-callback chain.
    dm.displacement_results = _types.SimpleNamespace(
        displacement_field=np.ones((1, 2, 2, 2)),
        physical_scale={"grid_spacing": 1.0, "time_interval": 1.0},
    )
    force_widget._update_ui_state()

    # After: displacement present → Run enabled.
    assert force_widget.action_states()["run"] is True
