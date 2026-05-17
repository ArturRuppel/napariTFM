from pathlib import Path

import numpy as np
import pandas as pd

from _dev.optical_flow_comparison.reporting import (
    write_per_algorithm_png,
    write_combined_scenario_png,
    write_summary_csv,
    write_summary_bar_chart,
    write_tidy_displacements_csv,
)


def _result_rows():
    rng = np.random.default_rng(0)
    beads = rng.uniform(10, 90, size=(15, 2)).astype(np.float32)
    pred = np.full((15, 2), 2.0, dtype=np.float32)
    gt = pred + rng.normal(scale=0.2, size=(15, 2)).astype(np.float32)
    valid = np.ones(15, dtype=bool)
    return beads, pred, gt, valid


def test_write_per_algorithm_png_creates_file(tmp_path: Path):
    beads, pred, gt, valid = _result_rows()
    img = np.zeros((100, 100), dtype=np.float32)
    out = tmp_path / "low" / "DIS.png"

    write_per_algorithm_png(
        out_path=out,
        reference_image=img,
        bead_positions=beads,
        predicted=pred,
        ground_truth=gt,
        valid=valid,
        title="low / DIS",
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_write_combined_scenario_png_handles_multiple_algorithms(tmp_path: Path):
    beads, pred, gt, valid = _result_rows()
    img = np.zeros((100, 100), dtype=np.float32)
    per_algo = {
        "DIS": (pred, gt, valid),
        "Farneback": (pred, gt, valid),
        "Lucas-Kanade": (pred, gt, valid),
    }
    out = tmp_path / "low" / "combined.png"

    write_combined_scenario_png(
        out_path=out,
        reference_image=img,
        bead_positions=beads,
        per_algorithm=per_algo,
        scenario_name="low",
    )

    assert out.exists()


def test_write_summary_csv_round_trips(tmp_path: Path):
    rows = [
        {"scenario": "low", "algorithm": "DIS", "rmse_px": 0.5, "n_beads": 10,
         "coverage": 1.0, "median_px": 0.4, "p95_px": 0.9,
         "bias_low": 0.0, "bias_mid": -0.1, "bias_high": -0.2},
    ]
    out = tmp_path / "summary.csv"

    write_summary_csv(out, rows)

    df = pd.read_csv(out)
    assert list(df.columns) == [
        "scenario", "algorithm", "n_beads", "coverage",
        "rmse_px", "median_px", "p95_px",
        "bias_low", "bias_mid", "bias_high",
    ]
    assert df.iloc[0]["rmse_px"] == 0.5


def test_write_summary_bar_chart_creates_file(tmp_path: Path):
    rows = [
        {"scenario": "low", "algorithm": "DIS", "rmse_px": 0.5},
        {"scenario": "low", "algorithm": "Lucas-Kanade", "rmse_px": 0.3},
        {"scenario": "mid", "algorithm": "DIS", "rmse_px": 0.8},
        {"scenario": "mid", "algorithm": "Lucas-Kanade", "rmse_px": 0.6},
    ]
    out = tmp_path / "summary.png"

    write_summary_bar_chart(out, rows)

    assert out.exists()


def test_write_tidy_displacements_csv_columns(tmp_path: Path):
    beads, pred, gt, valid = _result_rows()
    out = tmp_path / "displacements.csv"

    write_tidy_displacements_csv(
        out,
        scenario="low",
        algorithm="DIS",
        bead_positions=beads,
        predicted=pred,
        ground_truth=gt,
        valid=valid,
        append=False,
    )

    df = pd.read_csv(out)
    expected = [
        "scenario", "algorithm", "bead_id",
        "ref_x", "ref_y",
        "pred_dx", "pred_dy",
        "gt_dx", "gt_dy",
        "err_x", "err_y", "err_mag", "valid",
    ]
    assert list(df.columns) == expected
    assert len(df) == 15
