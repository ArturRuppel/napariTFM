"""Round-trip + container tests for the native ``.ntfm`` format (ROADMAP §1)."""

import numpy as np
import pandas as pd
import pytest

from napariTFM.utilities import ntfm


# ---------------------------------------------------------------------------
# Fixtures: synthetic co-registered grid arrays
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_arrays():
    rng = np.random.default_rng(0)
    nt, ny, nx = 3, 4, 5
    displacement = rng.standard_normal((nt, ny, nx, 2))
    force = rng.standard_normal((nt, ny, nx, 2)) * 100.0
    stress = np.zeros((nt, ny, nx, 2, 2))
    stress[..., 0, 0] = rng.standard_normal((nt, ny, nx))
    stress[..., 1, 1] = rng.standard_normal((nt, ny, nx))
    shear = rng.standard_normal((nt, ny, nx))
    stress[..., 0, 1] = shear
    stress[..., 1, 0] = shear
    mask = np.zeros((ny, nx), dtype=np.int64)
    mask[1:3, 1:4] = 1
    return dict(
        displacement_field=displacement,
        force_field=force,
        stress_tensor=stress,
        mask=mask,
        grid_spacing=0.4,
        frame_interval=2.0,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_columns_match_schema(grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    assert list(df.columns) == ntfm.COLUMNS


def test_row_count_is_full_grid(grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    assert len(df) == 3 * 4 * 5


def test_identifier_columns_are_physical(grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    # y = row * spacing, x = col * spacing, t = frame * interval
    assert np.allclose(df["y[µm]"], df["row"] * 0.4)
    assert np.allclose(df["x[µm]"], df["col"] * 0.4)
    assert set(df["t[min]"].unique()) == {0.0, 2.0, 4.0}


def test_no_redundant_sigma_yx_column(grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    assert "sigma_yx[mN/m]" not in df.columns
    assert "sigma_shear[mN/m]" in df.columns


# ---------------------------------------------------------------------------
# Round-trip: arrays -> tidy -> arrays
# ---------------------------------------------------------------------------

def test_round_trip_lossless(grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    out = ntfm.tidy_to_arrays(df)

    np.testing.assert_array_equal(out["displacement_field"], grid_arrays["displacement_field"])
    np.testing.assert_array_equal(out["force_field"], grid_arrays["force_field"])
    np.testing.assert_array_equal(out["stress_tensor"], grid_arrays["stress_tensor"])
    # 2D mask is broadcast across t on the way out.
    expected_mask = np.broadcast_to(grid_arrays["mask"], (3, 4, 5))
    np.testing.assert_array_equal(out["mask"], expected_mask)


def test_round_trip_is_order_independent(grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = ntfm.tidy_to_arrays(shuffled)
    np.testing.assert_array_equal(out["stress_tensor"], grid_arrays["stress_tensor"])


def test_reconstructed_stress_is_symmetric(grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    stress = ntfm.tidy_to_arrays(df)["stress_tensor"]
    np.testing.assert_array_equal(stress[..., 0, 1], stress[..., 1, 0])


# ---------------------------------------------------------------------------
# Partial / absent measures
# ---------------------------------------------------------------------------

def test_displacement_only_emits_nan_measures():
    disp = np.ones((2, 3, 3, 2))
    df = ntfm.arrays_to_tidy(displacement_field=disp, grid_spacing=1.0, frame_interval=1.0)
    assert df["F_x[Pa]"].isna().all()
    assert df["sigma_xx[mN/m]"].isna().all()
    assert (df["mask"] == 0).all()


def test_absent_columns_map_to_none_on_read():
    disp = np.ones((2, 3, 3, 2))
    df = ntfm.arrays_to_tidy(displacement_field=disp, grid_spacing=1.0, frame_interval=1.0)
    df = df.drop(columns=["F_x[Pa]", "F_y[Pa]"])
    out = ntfm.tidy_to_arrays(df)
    assert out["force_field"] is None
    assert out["displacement_field"] is not None


def test_requires_at_least_one_measure():
    with pytest.raises(ValueError):
        ntfm.arrays_to_tidy(grid_spacing=1.0, frame_interval=1.0)


def test_inconsistent_grid_shape_raises():
    disp = np.ones((2, 3, 3, 2))
    force = np.ones((2, 3, 4, 2))
    with pytest.raises(ValueError):
        ntfm.arrays_to_tidy(
            displacement_field=disp, force_field=force, grid_spacing=1.0, frame_interval=1.0
        )


def test_off_mask_stress_nan_survives_round_trip():
    disp = np.ones((1, 2, 2, 2))
    stress = np.full((1, 2, 2, 2, 2), np.nan)
    stress[0, 0, 0] = [[1.0, 0.5], [0.5, 2.0]]
    df = ntfm.arrays_to_tidy(
        displacement_field=disp, stress_tensor=stress, grid_spacing=1.0, frame_interval=1.0
    )
    out = ntfm.tidy_to_arrays(df)["stress_tensor"]
    assert np.isnan(out[0, 1, 1, 0, 0])
    assert out[0, 0, 0, 0, 0] == 1.0


# ---------------------------------------------------------------------------
# Container I/O
# ---------------------------------------------------------------------------

def test_ntfm_container_round_trip(tmp_path, grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    metadata = ntfm.build_metadata(config={"pixel_size": 0.1, "downscale_factor": 4})
    path = tmp_path / "experiment.ntfm"

    ntfm.write_ntfm(path, df, metadata)
    assert path.exists()

    df2, metadata2 = ntfm.read_ntfm(path)
    pd.testing.assert_frame_equal(df, df2)
    assert metadata2["config"]["pixel_size"] == 0.1
    assert metadata2["format_version"] == ntfm.FORMAT_VERSION


def test_container_is_ome_tiff_with_expected_series(tmp_path, grid_arrays):
    import tifffile

    df = ntfm.arrays_to_tidy(**grid_arrays)
    path = tmp_path / "experiment.ome.tif"
    ntfm.write_ntfm(path, df, ntfm.build_metadata(config={}))

    with tifffile.TiffFile(path) as tf:
        # Full pipeline + a real mask -> all four named series present.
        assert {s.name for s in tf.series} == {
            "displacement",
            "traction",
            "stress",
            "mask",
        }


def test_channel_names_carry_units(tmp_path, grid_arrays):
    import tifffile

    df = ntfm.arrays_to_tidy(**grid_arrays)
    path = tmp_path / "experiment.ome.tif"
    ntfm.write_ntfm(path, df, ntfm.build_metadata(config={}))

    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata
    # Unit-bearing channel names are written into the OME-XML.
    for name in ("u_x[µm]", "F_x[Pa]", "sigma_xx[mN/m]", "sigma_shear[mN/m]"):
        assert name in xml


def test_physical_scale_in_ome_xml(tmp_path, grid_arrays):
    df = ntfm.arrays_to_tidy(**grid_arrays)
    path = tmp_path / "experiment.ome.tif"
    ntfm.write_ntfm(path, df, ntfm.build_metadata(config={}))

    # grid_spacing=0.4 µm, frame_interval=2.0 min survive into the OME-XML and
    # are recovered on read to rebuild the physical id columns exactly.
    df2, _ = ntfm.read_ntfm(path)
    assert np.allclose(df2["y[µm]"], df2["row"] * 0.4)
    assert set(df2["t[min]"].unique()) == {0.0, 2.0, 4.0}


def test_measures_stored_float32_read_float64(tmp_path, grid_arrays):
    import tifffile

    df = ntfm.arrays_to_tidy(**grid_arrays)
    path = tmp_path / "experiment.ome.tif"
    ntfm.write_ntfm(path, df, ntfm.build_metadata(config={}))

    with tifffile.TiffFile(path) as tf:
        by_name = {s.name: s for s in tf.series}
        assert by_name["displacement"].dtype == np.float32
        assert by_name["mask"].dtype == np.uint16

    # The in-memory contract is float64 — measures are cast back up on read.
    df2, _ = ntfm.read_ntfm(path)
    assert df2["u_x[µm]"].dtype == np.float64


def test_absent_stage_writes_no_series(tmp_path):
    import tifffile

    disp = np.ones((1, 2, 2, 2))
    df = ntfm.arrays_to_tidy(displacement_field=disp, grid_spacing=1.0, frame_interval=1.0)
    path = tmp_path / "disp_only.ome.tif"
    ntfm.write_ntfm(path, df, ntfm.build_metadata(config={}))

    with tifffile.TiffFile(path) as tf:
        # No traction/stress/mask series for stages that never ran.
        assert {s.name for s in tf.series} == {"displacement"}
    assert ntfm.populated_measures(path) == {"displacement"}


def test_metadata_carries_provenance():
    metadata = ntfm.build_metadata(config={"a": 1})
    assert metadata["format_version"] == ntfm.FORMAT_VERSION
    assert "git_commit" in metadata
    assert "git_dirty" in metadata
    assert "package_version" in metadata
    assert metadata["config"] == {"a": 1}


def test_metadata_carries_labels():
    labels = {"condition": "stiff", "replicate": 2, "position": "p1"}
    metadata = ntfm.build_metadata(config={}, labels=labels)
    assert metadata["labels"] == labels


def test_metadata_labels_default_to_empty_dict():
    metadata = ntfm.build_metadata(config={})
    assert metadata["labels"] == {}


# ---------------------------------------------------------------------------
# Result-dataclass adapter
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_dataframe_from_results_infers_scale_from_physical_scale():
    disp = np.ones((2, 3, 3, 2))
    scale = {"grid_spacing": 0.8, "time_interval": 5.0}
    result = _FakeResult(displacement_field=disp, physical_scale=scale)
    df = ntfm.dataframe_from_results(displacement_result=result)
    assert np.allclose(df["x[µm]"], df["col"] * 0.8)
    assert set(df["t[min]"].unique()) == {0.0, 5.0}


def test_results_to_ntfm_writes_container(tmp_path):
    disp = np.ones((1, 2, 2, 2))
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    result = _FakeResult(displacement_field=disp, physical_scale=scale)
    path = tmp_path / "exp.ntfm"
    ntfm.results_to_ntfm(path, config={"downscale_factor": 4}, displacement_result=result)

    df, metadata = ntfm.read_ntfm(path)
    assert len(df) == 4
    assert metadata["config"]["downscale_factor"] == 4


def test_results_to_ntfm_persists_labels(tmp_path):
    disp = np.ones((1, 2, 2, 2))
    scale = {"grid_spacing": 1.0, "time_interval": 1.0}
    result = _FakeResult(displacement_field=disp, physical_scale=scale)
    path = tmp_path / "exp.ntfm"
    ntfm.results_to_ntfm(
        path,
        config={"downscale_factor": 4},
        displacement_result=result,
        labels={"condition": "soft"},
    )

    _, metadata = ntfm.read_ntfm(path)
    assert metadata["labels"] == {"condition": "soft"}


# ---------------------------------------------------------------------------
# populated_measures — per-output truth for stage-status (P3)
# ---------------------------------------------------------------------------

def _write_measure_ntfm(tmp_path, **arrays):
    df = ntfm.arrays_to_tidy(grid_spacing=1.0, frame_interval=1.0, **arrays)
    path = tmp_path / "exp.ntfm"
    ntfm.write_ntfm(path, df, ntfm.build_metadata(config={}))
    return path


def test_populated_measures_displacement_only(tmp_path):
    path = _write_measure_ntfm(tmp_path, displacement_field=np.ones((1, 2, 2, 2)))
    assert ntfm.populated_measures(path) == {"displacement"}


def test_populated_measures_through_force(tmp_path):
    path = _write_measure_ntfm(
        tmp_path,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
    )
    assert ntfm.populated_measures(path) == {"displacement", "force"}


def test_populated_measures_full_pipeline(tmp_path):
    stress = np.ones((1, 2, 2, 2, 2))
    path = _write_measure_ntfm(
        tmp_path,
        displacement_field=np.ones((1, 2, 2, 2)),
        force_field=np.ones((1, 2, 2, 2)) * 5.0,
        stress_tensor=stress,
    )
    assert ntfm.populated_measures(path) == {"displacement", "force", "stress"}


def test_populated_measures_ignores_all_nan_columns(tmp_path):
    # A force-less run still writes F_x/F_y as all-NaN; that is not "present".
    path = _write_measure_ntfm(tmp_path, displacement_field=np.ones((1, 2, 2, 2)))
    df, _ = ntfm.read_ntfm(path)
    assert df["F_x[Pa]"].isna().all()
    assert "force" not in ntfm.populated_measures(path)


def test_populated_measures_missing_file_is_empty(tmp_path):
    assert ntfm.populated_measures(tmp_path / "nope.ntfm") == set()
