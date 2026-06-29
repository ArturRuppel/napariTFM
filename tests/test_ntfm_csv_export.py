"""Streaming per-pixel CSV export from a ``.ntfm`` (TODO #2)."""

import numpy as np
import pandas as pd
import pytest

from napariTFM.utilities import ntfm


def _write_ntfm(path, *, with_force=True, with_stress=True, mask=None):
    nt, ny, nx = 2, 3, 4
    rng = np.random.default_rng(1)
    kwargs = dict(
        displacement_field=rng.standard_normal((nt, ny, nx, 2)),
        grid_spacing=0.5,
        frame_interval=1.0,
    )
    if with_force:
        kwargs["force_field"] = rng.standard_normal((nt, ny, nx, 2)) * 10.0
    if with_stress:
        stress = np.zeros((nt, ny, nx, 2, 2))
        stress[..., 0, 0] = rng.standard_normal((nt, ny, nx))
        stress[..., 1, 1] = rng.standard_normal((nt, ny, nx))
        shear = rng.standard_normal((nt, ny, nx))
        stress[..., 0, 1] = shear
        stress[..., 1, 0] = shear
        kwargs["stress_tensor"] = stress
    if mask is not None:
        kwargs["mask"] = mask
    df = ntfm.arrays_to_tidy(**kwargs)
    ntfm.write_ntfm(path, df, ntfm.build_metadata(config={}))
    return df, (nt, ny, nx)


def test_export_writes_full_grid_for_every_frame(tmp_path):
    src = tmp_path / "exp.ntfm"
    df, (nt, ny, nx) = _write_ntfm(src, mask=np.ones((3, 4), dtype=np.int64))
    out = ntfm.export_ntfm_to_csv(src, tmp_path / "exp.csv")

    got = pd.read_csv(out)
    assert len(got) == nt * ny * nx
    for col in ("u_x[µm]", "u_y[µm]", "F_x[Pa]", "F_y[Pa]"):
        assert col in got.columns
    # Values survive the round trip (sorted by the canonical id order).
    np.testing.assert_allclose(
        got.sort_values(["t[min]", "row", "col"])["u_x[µm]"].to_numpy(),
        df.sort_values(["t[min]", "row", "col"])["u_x[µm]"].to_numpy(),
    )


def test_export_omits_stages_that_never_ran(tmp_path):
    src = tmp_path / "disp_only.ntfm"
    _write_ntfm(src, with_force=False, with_stress=False)
    got = pd.read_csv(ntfm.export_ntfm_to_csv(src, tmp_path / "out.csv"))

    assert "u_x[µm]" in got.columns
    assert "F_x[Pa]" not in got.columns
    assert "sigma_xx[mN/m]" not in got.columns
    assert "mask" not in got.columns  # no mask supplied -> all zero -> dropped


def test_export_includes_mask_and_stress_when_present(tmp_path):
    src = tmp_path / "full.ntfm"
    mask = np.zeros((3, 4), dtype=np.int64)
    mask[1, 1:3] = 1
    _write_ntfm(src, mask=mask)
    got = pd.read_csv(ntfm.export_ntfm_to_csv(src, tmp_path / "out.csv"))

    assert "sigma_xx[mN/m]" in got.columns
    assert "sigma_shear[mN/m]" in got.columns
    assert "mask" in got.columns
    assert got["mask"].sum() > 0


def test_export_is_chunk_size_invariant(tmp_path):
    src = tmp_path / "exp.ntfm"
    _write_ntfm(src, mask=np.ones((3, 4), dtype=np.int64))
    whole = pd.read_csv(ntfm.export_ntfm_to_csv(src, tmp_path / "whole.csv"))
    chunked = pd.read_csv(
        ntfm.export_ntfm_to_csv(src, tmp_path / "chunked.csv", chunk_rows=3)
    )
    pd.testing.assert_frame_equal(whole, chunked)


def test_export_missing_container_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ntfm.export_ntfm_to_csv(tmp_path / "nope.ntfm", tmp_path / "out.csv")
