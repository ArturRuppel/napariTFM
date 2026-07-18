"""Cross-experiment aggregator: reduce a ``.ntfm`` series to a tidy summary.

Covers the revived per-experiment reduction plus the ITASC-ported improvements:
identity uniqueness, stable row id, ready/not-ready partition, and the
``provenance.json`` audit sidecar. Statistics/plotting are downstream (Option A),
so there are no premade-spec / ``.iris`` tests.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napariTFM.backend import aggregate
from napariTFM.backend import metrics_calculator as mc
from napariTFM.utilities import ntfm
from napariTFM.utilities.batch_output import RESULTS_FILENAME


GRID_SPACING_UM = 0.5
FRAME_INTERVAL_MIN = 2.0


def _make_ntfm(folder, *, with_force=True, with_mask=True, labels=None):
    """Write a deterministic 2-frame, 6x6 container into its own experiment folder.

    Mirrors a real batch output: ``<folder>/TFM_results.ome.tif`` with the source
    folder and design tags recorded in metadata. Returns
    ``(container_path, displacement, force, mask)``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
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
        mask=mask if with_mask else None,
        grid_spacing=GRID_SPACING_UM,
        frame_interval=FRAME_INTERVAL_MIN,
    )
    metadata = ntfm.build_metadata(
        config={"pixel_size": 0.25, "downscale_factor": 2},
        inputs={"folder": str(folder)},
        labels=labels or {},
    )
    path = folder / RESULTS_FILENAME
    ntfm.write_ntfm(path, df, metadata)
    return path, displacement, force, mask


def _expected_region_metrics(displacement, force, region_mask):
    """Compute the four scalars directly, mirroring the documented convention."""
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
# Per-.ntfm reduction
# ---------------------------------------------------------------------------

def test_summarize_grain_columns(tmp_path):
    path, *_ = _make_ntfm(tmp_path / "exp_a")
    summary = aggregate.summarize_ntfm(path)
    required = set(aggregate.ID_COLUMNS + aggregate.METRIC_COLUMNS)
    assert required.issubset(summary.columns)


def test_summarize_one_row_per_region_frame(tmp_path):
    path, *_ = _make_ntfm(tmp_path / "exp_a")
    summary = aggregate.summarize_ntfm(path)
    assert len(summary) == 4  # 2 regions x 2 frames (static mask)
    assert sorted(summary["region_id"].unique()) == [1, 2]
    assert sorted(summary["frame"].unique()) == [0, 1]


def test_experiment_id_is_source_folder_name(tmp_path):
    # Every container has the SAME filename (TFM_results.ome.tif) — the id must
    # come from the source folder, not the useless stem.
    path, *_ = _make_ntfm(tmp_path / "exp_a")
    summary = aggregate.summarize_ntfm(path)
    assert set(summary["experiment_id"]) == {"exp_a"}


def test_summarize_metrics_match_direct_calc(tmp_path):
    path, displacement, force, mask = _make_ntfm(tmp_path / "exp_a")
    summary = aggregate.summarize_ntfm(path).set_index(["region_id", "frame"])
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


def test_summarize_force_absent_yields_nan_metrics(tmp_path):
    path, *_ = _make_ntfm(tmp_path / "exp_a", with_force=False)
    summary = aggregate.summarize_ntfm(path)
    assert summary["total_strain_energy"].isna().all()
    assert summary["polarization_index"].isna().all()


def test_summarize_no_mask_is_empty(tmp_path):
    path, *_ = _make_ntfm(tmp_path / "exp_a", with_mask=False)
    summary = aggregate.summarize_ntfm(path)
    assert summary.empty
    assert list(summary.columns) == aggregate.ID_COLUMNS + aggregate.METRIC_COLUMNS


# ---------------------------------------------------------------------------
# Series reduction: stack + promote labels + stable id
# ---------------------------------------------------------------------------

def test_build_summary_table_stacks_experiments(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a")
    b, *_ = _make_ntfm(tmp_path / "exp_b")
    table = aggregate.build_summary_table([a, b])
    assert set(table["experiment_id"]) == {"exp_a", "exp_b"}
    assert len(table) == 8  # 2 experiments x 2 regions x 2 frames


def test_build_summary_table_reads_labels_off_disk(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a", labels={"condition": "ctrl", "replicate": "r1"})
    b, *_ = _make_ntfm(tmp_path / "exp_b", labels={"condition": "drug", "replicate": "r1"})
    table = aggregate.build_summary_table([a, b])  # no explicit labels -> read from containers
    assert set(table[table["experiment_id"] == "exp_a"]["condition"]) == {"ctrl"}
    assert set(table[table["experiment_id"] == "exp_b"]["condition"]) == {"drug"}
    assert set(table["replicate"]) == {"r1"}


def test_build_summary_table_explicit_labels_override(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a", labels={"condition": "on_disk"})
    table = aggregate.build_summary_table([a], labels={"exp_a": {"condition": "override"}})
    assert set(table["condition"]) == {"override"}


def test_build_summary_table_missing_label_is_none(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a", labels={"condition": "ctrl"})
    b, *_ = _make_ntfm(tmp_path / "exp_b")  # no labels
    table = aggregate.build_summary_table([a, b])
    b_rows = table[table["experiment_id"] == "exp_b"]
    assert b_rows["condition"].isna().all()


def test_stable_row_id_leads_and_is_deterministic(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a")
    b, *_ = _make_ntfm(tmp_path / "exp_b")
    t1 = aggregate.build_summary_table([a, b])
    t2 = aggregate.build_summary_table([b, a])  # different order
    assert t1.columns[0] == aggregate.ROW_ID_COLUMN
    assert set(t1[aggregate.ROW_ID_COLUMN]) == set(t2[aggregate.ROW_ID_COLUMN])
    assert "exp_a|1|0" in set(t1[aggregate.ROW_ID_COLUMN])


def test_duplicate_experiment_id_refuses_to_pool(tmp_path):
    # Two containers whose recorded source folder shares a basename collide.
    a, *_ = _make_ntfm(tmp_path / "day1" / "exp_a")
    b, *_ = _make_ntfm(tmp_path / "day2" / "exp_a")
    with pytest.raises(ValueError, match="share an identifier"):
        aggregate.build_summary_table([a, b])


# ---------------------------------------------------------------------------
# Ready / not-ready partition (header-only)
# ---------------------------------------------------------------------------

def test_partition_ready_splits_on_mask_and_force(tmp_path):
    full, *_ = _make_ntfm(tmp_path / "full")
    no_force, *_ = _make_ntfm(tmp_path / "no_force", with_force=False)
    no_mask, *_ = _make_ntfm(tmp_path / "no_mask", with_mask=False)
    ready, skipped = aggregate.partition_ready([full, no_force, no_mask])
    assert ready == [full]
    reasons = {p: r for p, r in skipped}
    assert reasons[no_force] == "no force"
    assert reasons[no_mask] == "no mask"


# ---------------------------------------------------------------------------
# The pool: materialized CSV + provenance.json + schema.json
# ---------------------------------------------------------------------------

def test_pool_writes_summary_schema_provenance(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a", labels={"condition": "ctrl"})
    b, *_ = _make_ntfm(tmp_path / "exp_b", labels={"condition": "drug"})
    out = tmp_path / "pool"

    result = aggregate.pool_experiments([a, b], out)

    assert result.summary_path == out / aggregate.SUMMARY_FILENAME
    assert result.summary_path.exists()
    assert (out / aggregate.SCHEMA_FILENAME).exists()
    assert (out / aggregate.PROVENANCE_FILENAME).exists()

    loaded = pd.read_csv(result.summary_path)
    assert set(loaded["experiment_id"]) == {"exp_a", "exp_b"}
    assert set(loaded["condition"]) == {"ctrl", "drug"}
    assert len(loaded) == 8
    # No leading unnamed pandas index column.
    first_field = result.summary_path.read_text(encoding="utf-8").splitlines()[0].split(",")[0]
    assert first_field == aggregate.ROW_ID_COLUMN


def test_pool_skips_not_ready_and_reports_them(tmp_path):
    full, *_ = _make_ntfm(tmp_path / "full", labels={"condition": "ctrl"})
    no_force, *_ = _make_ntfm(tmp_path / "no_force", with_force=False)
    out = tmp_path / "pool"

    result = aggregate.pool_experiments([full, no_force], out)

    assert result.ready == [full]
    assert [p for p, _ in result.skipped] == [no_force]
    prov = json.loads((out / aggregate.PROVENANCE_FILENAME).read_text())
    assert {e["experiment_id"] for e in prov["experiments"]} == {"full"}
    assert prov["skipped"][0]["reason"] == "no force"
    assert prov["skipped"][0]["source"] == str(no_force)


def test_pool_is_materialized_view_no_stale_rows(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a")
    b, *_ = _make_ntfm(tmp_path / "exp_b")
    out = tmp_path / "pool"

    aggregate.pool_experiments([a, b], out)
    # Re-pool with b removed: the CSV must be rewritten whole, not appended.
    aggregate.pool_experiments([a], out)

    loaded = pd.read_csv(out / aggregate.SUMMARY_FILENAME)
    assert set(loaded["experiment_id"]) == {"exp_a"}
    assert len(loaded) == 4


def test_pool_provenance_records_counts_and_version(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a", labels={"condition": "ctrl"})
    out = tmp_path / "pool"
    aggregate.pool_experiments([a], out)
    prov = json.loads((out / aggregate.PROVENANCE_FILENAME).read_text())
    assert prov["tool"] == "napariTFM"
    assert prov["grain"] == aggregate.ID_COLUMNS
    assert prov["table"]["n_rows"] == 4
    exp = prov["experiments"][0]
    assert exp["experiment_id"] == "exp_a"
    assert exp["n_rows"] == 4
    assert exp["labels"] == {"condition": "ctrl"}


def test_pool_no_ready_leaves_provenance_only(tmp_path):
    no_mask, *_ = _make_ntfm(tmp_path / "no_mask", with_mask=False)
    out = tmp_path / "pool"
    result = aggregate.pool_experiments([no_mask], out)
    assert result.summary_path is None
    assert result.n_rows == 0
    assert result.provenance_path.exists()
    prov = json.loads(result.provenance_path.read_text())
    assert prov["experiments"] == []
    assert prov["skipped"][0]["reason"] == "no mask"


# ---------------------------------------------------------------------------
# Schema typing (schema.json)
# ---------------------------------------------------------------------------

def test_schema_types_identifiers_metrics_and_labels(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a", labels={"condition": "ctrl"})
    b, *_ = _make_ntfm(tmp_path / "exp_b", labels={"condition": "drug"})
    table = aggregate.build_summary_table([a, b])
    schema = {e["name"]: e for e in aggregate.build_schema(table)}

    assert schema["experiment_id"]["type"] == "identifier"
    energy = schema["total_strain_energy"]
    assert energy["type"] == "numeric" and energy["unit"] == "J"
    condition = schema["condition"]
    assert condition["type"] == "categorical"
    assert sorted(condition["levels"]) == ["ctrl", "drug"]


def test_schema_order_matches_table_columns(tmp_path):
    a, *_ = _make_ntfm(tmp_path / "exp_a")
    table = aggregate.build_summary_table([a])
    schema = aggregate.build_schema(table)
    assert [e["name"] for e in schema] == list(table.columns)


# ---------------------------------------------------------------------------
# Pool output location
# ---------------------------------------------------------------------------

def test_aggregate_output_dir_processed_root():
    from napariTFM.utilities.batch_output import aggregate_output_dir

    out = aggregate_output_dir(["/data/wt/e1", "/data/ko/e2"], "/proc")
    assert out == Path("/proc/TFM_aggregate")


def test_aggregate_output_dir_in_place_common_parent():
    from napariTFM.utilities.batch_output import aggregate_output_dir

    # In-place: sits at the longest common parent, peer to the TFM_data buckets.
    out = aggregate_output_dir(["/data/wt/e1", "/data/wt/e2"])
    assert out == Path("/data/wt/TFM_aggregate")


def test_aggregate_output_dir_single_folder_falls_back_to_parent():
    from napariTFM.utilities.batch_output import aggregate_output_dir

    out = aggregate_output_dir(["/data/wt/e1"])
    assert out == Path("/data/wt/TFM_aggregate")
