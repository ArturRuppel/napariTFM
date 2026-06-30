"""Batch writes one ``.ntfm`` per experiment into the TFM_data/ bucket (ROADMAP §4)."""

import numpy as np

from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.utilities import ntfm
from napariTFM.utilities.batch_output import RESULTS_FILENAME


class _Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _analysis(**config):
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {"parameters": {"pixel_size": 0.1, "downscale_factor": 4}, **config}
    return analysis


def test_write_experiment_ntfm_round_trips(tmp_path):
    scale = {"grid_spacing": 0.4, "time_interval": 2.0}
    disp = _Result(displacement_field=np.ones((2, 3, 3, 2)), physical_scale=scale)
    force = _Result(force_field=np.ones((2, 3, 3, 2)) * 50.0, physical_scale=scale)

    analysis = _analysis(labels={str(tmp_path / "exp_A"): {"condition": "stiff"}})
    folder = tmp_path / "exp_A"
    out = tmp_path / "out"
    out.mkdir()

    analysis._write_experiment_ntfm(out, folder, disp, force, None, None)

    ntfm_path = out / RESULTS_FILENAME
    assert ntfm_path.exists()

    df, metadata = ntfm.read_ntfm(ntfm_path)
    assert len(df) == 2 * 3 * 3
    assert metadata["config"]["downscale_factor"] == 4
    assert metadata["labels"] == {"condition": "stiff"}
    assert metadata["inputs"]["folder"] == str(folder)


def test_write_experiment_ntfm_aligns_mask_to_grid(tmp_path):
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    force = _Result(force_field=np.ones((1, 3, 3, 2)), physical_scale=scale)
    # Raw mask is coarser/finer than the analysis grid; must be resized to (3, 3).
    raw_mask = np.ones((6, 6), dtype=np.uint8)

    analysis = _analysis()
    analysis._write_experiment_ntfm(tmp_path, tmp_path / "exp", None, force, None, raw_mask)

    df, _ = ntfm.read_ntfm(tmp_path / RESULTS_FILENAME)
    assert (df["mask"] == 1).all()


def test_apply_mask_on_save_zeroes_background_pixels(tmp_path):
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp = _Result(displacement_field=np.full((1, 2, 2, 2), 5.0), physical_scale=scale)
    force = _Result(force_field=np.full((1, 2, 2, 2), 50.0), physical_scale=scale)
    raw_mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)  # (0, 0) is background

    analysis = _analysis(apply_mask_on_save=True)
    analysis._write_experiment_ntfm(tmp_path, tmp_path / "exp", disp, force, None, raw_mask)

    df, _ = ntfm.read_ntfm(tmp_path / RESULTS_FILENAME)
    background = df[(df["row"] == 0) & (df["col"] == 0)].iloc[0]
    on_cell = df[(df["row"] == 0) & (df["col"] == 1)].iloc[0]

    assert background["mask"] == 0
    assert background["u_x[µm]"] == 0.0 and background["F_x[Pa]"] == 0.0
    assert on_cell["mask"] == 1
    assert on_cell["u_x[µm]"] == 5.0 and on_cell["F_x[Pa]"] == 50.0


def test_apply_mask_on_save_off_by_default_leaves_background_untouched(tmp_path):
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp = _Result(displacement_field=np.full((1, 2, 2, 2), 5.0), physical_scale=scale)
    raw_mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)

    analysis = _analysis()  # apply_mask_on_save defaults to off
    analysis._write_experiment_ntfm(tmp_path, tmp_path / "exp", disp, None, None, raw_mask)

    df, _ = ntfm.read_ntfm(tmp_path / RESULTS_FILENAME)
    background = df[(df["row"] == 0) & (df["col"] == 0)].iloc[0]
    assert background["u_x[µm]"] == 5.0


def test_no_results_skips_ntfm(tmp_path):
    analysis = _analysis()
    analysis._write_experiment_ntfm(tmp_path, tmp_path / "exp", None, None, None, None)
    assert not (tmp_path / RESULTS_FILENAME).exists()


def test_write_failure_propagates(tmp_path, monkeypatch):
    """A failed .ntfm write must not be swallowed: the .ntfm is the sole
    persisted artifact, so the error has to surface (-> folder reported error)
    rather than leaving the folder looking 'done' with nothing on disk."""
    import pytest

    from napariTFM.utilities import ntfm

    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    disp = _Result(displacement_field=np.ones((1, 3, 3, 2)), physical_scale=scale)

    def _boom(*args, **kwargs):
        raise RuntimeError("pyarrow missing")

    monkeypatch.setattr(ntfm, "results_to_ntfm", _boom)

    analysis = _analysis()
    with pytest.raises(RuntimeError, match="pyarrow missing"):
        analysis._write_experiment_ntfm(tmp_path, tmp_path / "exp", disp, None, None, None)


def test_load_mask_returns_none_when_absent(tmp_path):
    analysis = _analysis(input_files={"beads": "b.tif"})
    assert analysis._load_mask(tmp_path) is None


def test_process_all_folders_routes_each_to_tfm_data_bucket(tmp_path):
    from pathlib import Path

    a, b = str(tmp_path / "exp_A"), str(tmp_path / "exp_B")
    analysis = _analysis(root_folders=[a, b])  # no processed_root -> in-place

    seen = {}
    analysis.process_folder = lambda folder, output_dir: seen.__setitem__(folder, output_dir)
    analysis.process_all_folders()

    assert seen[a] == tmp_path / "TFM_data" / "exp_A"
    assert seen[b] == tmp_path / "TFM_data" / "exp_B"
