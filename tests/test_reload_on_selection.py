"""Tests for interactive bug B fix: _load_active_experiment_results.

When the user selects a previously-processed experiment from the list,
_on_active_experiment_changed now calls _load_active_experiment_results
which reads the experiment's .ntfm and restores the computed results into
memory so that downstream widgets (Force, Stress) see their prerequisites.
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
            "pyr_scale": 0.5,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "use_gaussian_window": False,
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
            "stress_method": "MSM",
            "density_factor": 0.01,
            "mesh_algorithm": "Frontal-Del.",
            "use_optimization": True,
            "poisson_ratio_cells": 0.5,
            "bism_regularization": -6.0,
            "bism_lambda_method": "Fixed",
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
    real _load_active_experiment_results method can call them without error.
    """

    def __init__(self):
        self._callbacks = []
        self.output_dir = None
        self.active_input_folder = None
        self.active_input_files = {}
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
            "preprocessed_bead_stack",
            "preprocessed_reference",
            "preprocessed_cell_stack",
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


class _StubController(QObject):
    progress_updated = Signal(int, str)
    ui_frozen = Signal(bool)


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

    def run_action(self):
        self.action_calls["run"] += 1

    def preview_action(self):
        self.action_calls["preview"] += 1

    def cancel_action(self):
        self.action_calls["cancel"] += 1

    def gcv_action(self):
        self.action_calls["gcv"] = self.action_calls.get("gcv", 0) + 1

    def mesh_action(self):
        self.action_calls["mesh"] = self.action_calls.get("mesh", 0) + 1

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
# msm_widget.py imports ParameterCategory which is not in our stub, so stub
# the entire module to avoid that import path.
_stub_module("napariTFM.widgets.msm_widget", MSMWidget=_StubStageWidget)
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
    monkeypatch.setattr(_widget, "PreprocessingWidget", _StubStageWidget)
    monkeypatch.setattr(
        _widget, "DisplacementAnalysisWidget", _StubStageWidget
    )
    monkeypatch.setattr(_widget, "FTTCWidget", _StubStageWidget)
    monkeypatch.setattr(_widget, "MSMWidget", _StubStageWidget)
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
# Tests: _load_active_experiment_results directly
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

    widget._load_active_experiment_results(str(folder))

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

    widget._load_active_experiment_results(str(folder))

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

    widget._load_active_experiment_results(str(folder))

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

    widget._load_active_experiment_results(str(folder))

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

    widget._load_active_experiment_results(str(folder))

    assert widget.data_manager.displacement_results is not None
    assert widget.data_manager.force_results is not None
    np.testing.assert_allclose(
        widget.data_manager.force_results.force_field,
        force,
        rtol=1e-5,
    )


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

    widget._load_active_experiment_results(str(folder))

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

    widget._load_active_experiment_results(str(folder))

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

    widget._load_active_experiment_results(str(folder))

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

    widget._load_active_experiment_results(str(folder))

    assert widget.data_manager.displacement_results is None
    assert widget.data_manager.force_results is not None


# ---------------------------------------------------------------------------
# Tests: full selection path via _on_active_experiment_changed
# ---------------------------------------------------------------------------


def test_on_active_experiment_changed_loads_displacement(monkeypatch, app, tmp_path):
    """Selecting an experiment via the list drives _load_active_experiment_results."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "sel_00"
    folder.mkdir()
    (folder / "beads.tif").write_bytes(b"x")
    (folder / "reference.tif").write_bytes(b"x")

    disp = np.ones((2, 3, 4, 2)) * 1.5
    _write_ntfm(folder, displacement_field=disp)

    _select(widget, folder)

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

    assert widget.msm_widget.loaded_mask_paths == [str(folder / "masks.tif")]


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

    assert widget.msm_widget.loaded_mask_paths == []


def test_deselection_clears_results(monkeypatch, app, tmp_path):
    """Clearing the active experiment drops the loaded in-memory results."""
    widget = _stub_main_widget(monkeypatch)
    folder = tmp_path / "sel_01"
    folder.mkdir()

    _write_ntfm(folder, displacement_field=np.ones((1, 2, 2, 2)))
    _select(widget, folder)
    assert widget.data_manager.displacement_results is not None

    # Clearing active experiment should drop results.
    widget.experiments_list.set_active(None)
    assert widget.data_manager.displacement_results is None


def test_switching_experiments_loads_new_results(monkeypatch, app, tmp_path):
    """Switching to a different experiment loads its own results, not the previous one's."""
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

    np.testing.assert_allclose(
        widget.data_manager.displacement_results.displacement_field,
        disp_a,
        rtol=1e-5,
    )

    widget.experiments_list.set_active(str(folder_b))

    np.testing.assert_allclose(
        widget.data_manager.displacement_results.displacement_field,
        disp_b,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Test: Force widget Run button enabled after selection
# ---------------------------------------------------------------------------


def test_force_run_enabled_after_selection_with_displacement(app):
    """FTTCWidget._update_ui_state enables Run when displacement_results is not None.

    This is the contract the bug fix satisfies: after _load_active_experiment_results
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

    # Simulate _load_active_experiment_results storing a displacement result
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
