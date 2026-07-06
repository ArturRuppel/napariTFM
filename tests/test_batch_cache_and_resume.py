"""Batch stage-resume reads from the ``.ntfm``.

The ``.ntfm`` is the sole persisted result for displacement/force/stress.
Preprocessed ``.tif`` images are now always written (unconditionally) so that
preprocessing has the same persistence guarantee as every downstream stage.
"""

from types import SimpleNamespace

import numpy as np

from napariTFM.backend import batch_analysis as ba
from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.utilities import ntfm


class _Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _analysis(**config):
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {"parameters": {"pixel_size": 0.1, "downscale_factor": 4}, **config}
    return analysis


def _write_ntfm(analysis, out_dir, folder, disp=None, force=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis._write_experiment_ntfm(out_dir, folder, disp, force, None, None)


# --- resume helper ---------------------------------------------------------

def test_resume_field_from_ntfm_round_trips(tmp_path):
    scale = {"grid_spacing": 0.4, "time_interval": 2.0}
    disp = _Result(displacement_field=np.arange(2 * 3 * 3 * 2, dtype=float).reshape(2, 3, 3, 2),
                   physical_scale=scale)
    analysis = _analysis()
    folder = tmp_path / "exp"
    _write_ntfm(analysis, tmp_path, folder, disp=disp)

    field = analysis._resume_field_from_ntfm(tmp_path, folder, "displacement_field")
    assert field is not None
    np.testing.assert_allclose(field, disp.displacement_field)


def test_resume_field_missing_ntfm_returns_none(tmp_path):
    analysis = _analysis()
    field = analysis._resume_field_from_ntfm(tmp_path, tmp_path / "nope", "force_field")
    assert field is None


def test_resume_field_absent_measure_returns_none(tmp_path):
    # A .ntfm with only displacement has no force_field to resume.
    disp = _Result(displacement_field=np.ones((1, 3, 3, 2)),
                   physical_scale={"grid_spacing": 1.0, "time_interval": 1.0})
    analysis = _analysis()
    folder = tmp_path / "exp"
    _write_ntfm(analysis, tmp_path, folder, disp=disp)

    assert analysis._resume_field_from_ntfm(tmp_path, folder, "force_field") is None


# --- handler resume wiring -------------------------------------------------

def test_force_handler_resumes_displacement_from_ntfm(tmp_path, monkeypatch):
    disp = _Result(displacement_field=np.ones((1, 3, 3, 2)) * 7.0,
                   physical_scale={"grid_spacing": 1.0, "time_interval": 1.0})
    analysis = _analysis(analysis_steps={"force": True})
    folder = tmp_path / "exp"
    _write_ntfm(analysis, tmp_path, folder, disp=disp)

    captured = {}

    def _fake_execute_force(self, tfm_folder, displacement_data, folder=None):
        captured["field"] = displacement_data.displacement_field
        return _Result(force_field=np.zeros((1, 3, 3, 2)))

    monkeypatch.setattr(BatchAnalysis, "_execute_force_analysis", _fake_execute_force)

    # displacement_data is None -> must resume from the .ntfm.
    result = analysis._handle_force_execution(tmp_path, folder, None)
    assert result is not None
    np.testing.assert_allclose(captured["field"], disp.displacement_field)


def test_stress_handler_skips_when_no_ntfm(tmp_path):
    analysis = _analysis(analysis_steps={"stress": True})
    # force_data None and no .ntfm to resume from -> gracefully returns None.
    result = analysis._handle_stress_execution(tmp_path, tmp_path / "exp", None, np.ones((3, 3)))
    assert result is None


