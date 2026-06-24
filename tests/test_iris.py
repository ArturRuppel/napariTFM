"""Aggregator: reduce a ``.ntfm`` series to a summary table, export ``.iris``
(ROADMAP §5)."""

import numpy as np
import pytest

from napariTFM.backend import metrics_calculator as mc
from napariTFM.utilities import ntfm
from napariTFM.utilities import iris


# ---------------------------------------------------------------------------
# Fixtures: a small two-region, two-frame .ntfm on disk
# ---------------------------------------------------------------------------

GRID_SPACING_UM = 0.5
FRAME_INTERVAL_MIN = 2.0


def _make_ntfm(path, *, with_force=True):
    """Write a deterministic 2-frame, 6x6 .ntfm with mask labels {1, 2}."""
    rng = np.random.default_rng(7)
    nt, ny, nx = 2, 6, 6
    displacement = rng.standard_normal((nt, ny, nx, 2))  # µm
    force = rng.standard_normal((nt, ny, nx, 2)) * 200.0  # Pa
    mask = np.zeros((ny, nx), dtype=np.int64)
    mask[1:3, 1:3] = 1  # region 1 (a 2x2 block)
    mask[3:5, 3:6] = 2  # region 2 (a 2x3 block)

    df = ntfm.arrays_to_tidy(
        displacement_field=displacement,
        force_field=force if with_force else None,
        mask=mask,
        grid_spacing=GRID_SPACING_UM,
        frame_interval=FRAME_INTERVAL_MIN,
    )
    metadata = ntfm.build_metadata(config={"pixel_size": 0.25, "downscale_factor": 2})
    ntfm.write_ntfm(path, df, metadata)
    return displacement, force, mask


def _expected_region_metrics(displacement, force, region_mask):
    """Compute the four scalars directly, mirroring iris's documented convention."""
    spacing_m = GRID_SPACING_UM * 1e-6
    pixel_area_m2 = spacing_m ** 2
    disp_m = displacement * 1e-6

    sed = mc.calculate_strain_energy_density(disp_m, force)
    tse = mc.calculate_total_strain_energy(sed, region_mask, pixel_area_m2)

    ny, nx = region_mask.shape
    row_idx, col_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ys, xs = np.where(region_mask)
    cy, cx = ys.mean(), xs.mean()
    positions = np.zeros((ny, nx, 2))
    positions[..., 0] = (col_idx - cx) * spacing_m  # x
    positions[..., 1] = (row_idx - cy) * spacing_m  # y

    moment = mc.calculate_moment_tensor(force, region_mask, positions, pixel_area_m2)
    pi, l1, l2 = mc.calculate_polarization(moment)
    return tse, pi, l1, l2


# ---------------------------------------------------------------------------
# Slice 1 — per-.ntfm reduction
# ---------------------------------------------------------------------------

def test_summarize_ntfm_grain_columns(tmp_path):
    path = tmp_path / "exp_a.ntfm"
    _make_ntfm(path)

    summary = iris.summarize_ntfm(path)

    required = {
        "experiment_id",
        "region_id",
        "frame",
        "total_strain_energy",
        "polarization_index",
        "lambda1",
        "lambda2",
    }
    assert required.issubset(summary.columns)


def test_summarize_ntfm_one_row_per_region_frame(tmp_path):
    path = tmp_path / "exp_a.ntfm"
    _make_ntfm(path)

    summary = iris.summarize_ntfm(path)

    # 2 regions x 2 frames, each region present in every frame (static mask).
    assert len(summary) == 4
    assert sorted(summary["region_id"].unique()) == [1, 2]
    assert sorted(summary["frame"].unique()) == [0, 1]


def test_summarize_ntfm_experiment_id_is_file_stem(tmp_path):
    path = tmp_path / "exp_a.ntfm"
    _make_ntfm(path)

    summary = iris.summarize_ntfm(path)

    assert set(summary["experiment_id"]) == {"exp_a"}


def test_summarize_ntfm_metrics_match_direct_calc(tmp_path):
    path = tmp_path / "exp_a.ntfm"
    displacement, force, mask = _make_ntfm(path)

    summary = iris.summarize_ntfm(path).set_index(["region_id", "frame"])

    for region in (1, 2):
        region_mask = (mask == region).astype(float)
        for frame in (0, 1):
            tse, pi, l1, l2 = _expected_region_metrics(
                displacement[frame], force[frame], region_mask
            )
            row = summary.loc[(region, frame)]
            assert row["total_strain_energy"] == pytest.approx(tse)
            assert row["polarization_index"] == pytest.approx(pi)
            assert row["lambda1"] == pytest.approx(l1)
            assert row["lambda2"] == pytest.approx(l2)


def test_summarize_ntfm_force_absent_yields_nan_metrics(tmp_path):
    path = tmp_path / "exp_a.ntfm"
    _make_ntfm(path, with_force=False)

    summary = iris.summarize_ntfm(path)

    assert summary["total_strain_energy"].isna().all()
    assert summary["polarization_index"].isna().all()


# ---------------------------------------------------------------------------
# Slice 2 — series reduction: stack + promote aggregator-assigned labels
# ---------------------------------------------------------------------------

def test_build_summary_table_stacks_experiments(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    b = tmp_path / "exp_b.ntfm"
    _make_ntfm(a)
    _make_ntfm(b)

    table = iris.build_summary_table([a, b])

    assert set(table["experiment_id"]) == {"exp_a", "exp_b"}
    assert len(table) == 8  # 2 experiments x 2 regions x 2 frames


def test_build_summary_table_promotes_label_columns(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    b = tmp_path / "exp_b.ntfm"
    _make_ntfm(a)
    _make_ntfm(b)

    table = iris.build_summary_table(
        [a, b],
        labels={
            "exp_a": {"condition": "ctrl", "replicate": "r1"},
            "exp_b": {"condition": "drug", "replicate": "r1"},
        },
    )

    assert "condition" in table.columns
    assert "replicate" in table.columns
    a_rows = table[table["experiment_id"] == "exp_a"]
    b_rows = table[table["experiment_id"] == "exp_b"]
    assert set(a_rows["condition"]) == {"ctrl"}
    assert set(b_rows["condition"]) == {"drug"}
    assert set(table["replicate"]) == {"r1"}


def test_build_summary_table_missing_label_is_none(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    b = tmp_path / "exp_b.ntfm"
    _make_ntfm(a)
    _make_ntfm(b)

    table = iris.build_summary_table(
        [a, b],
        labels={"exp_a": {"condition": "ctrl"}},  # exp_b has no label
    )

    b_rows = table[table["experiment_id"] == "exp_b"]
    assert b_rows["condition"].isna().all()


def test_build_summary_table_no_labels_has_only_grain_and_metrics(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    _make_ntfm(a)

    table = iris.build_summary_table([a])

    assert list(table.columns) == iris.ID_COLUMNS + iris.METRIC_COLUMNS


# ---------------------------------------------------------------------------
# Slice 3 — Iris schema typing (data/schema.json)
# ---------------------------------------------------------------------------

def _schema_by_name(table, **kw):
    return {entry["name"]: entry for entry in iris.build_schema(table, **kw)}


def test_schema_types_identifiers(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    _make_ntfm(a)
    schema = _schema_by_name(iris.build_summary_table([a]))

    for name in ("experiment_id", "region_id", "frame"):
        assert schema[name]["type"] == "identifier"


def test_schema_types_metrics_numeric_with_unit_and_label(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    _make_ntfm(a)
    schema = _schema_by_name(iris.build_summary_table([a]))

    energy = schema["total_strain_energy"]
    assert energy["type"] == "numeric"
    assert energy["unit"] == "J"
    assert energy["label"]  # human-readable, non-empty


def test_schema_types_labels_categorical_with_levels(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    b = tmp_path / "exp_b.ntfm"
    _make_ntfm(a)
    _make_ntfm(b)
    table = iris.build_summary_table(
        [a, b],
        labels={"exp_a": {"condition": "ctrl"}, "exp_b": {"condition": "drug"}},
    )
    schema = _schema_by_name(table)

    condition = schema["condition"]
    assert condition["type"] == "categorical"
    assert sorted(condition["levels"]) == ["ctrl", "drug"]


def test_schema_order_matches_table_columns(tmp_path):
    a = tmp_path / "exp_a.ntfm"
    _make_ntfm(a)
    table = iris.build_summary_table([a])
    schema = iris.build_schema(table)

    assert [entry["name"] for entry in schema] == list(table.columns)
