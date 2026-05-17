from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from _dev.optical_flow_comparison.runner import run


def _make_synthetic_scenario(scenario_dir: Path, shift_px=(2.0, 1.0)):
    """Create a fake benchmark scenario directory with reference + deformed
    TIFs and matching ground-truth .npy fields (in microns; pixel_size_um=0.1)."""
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    h, w = 128, 128

    positions = rng.uniform(15, min(h, w) - 15, size=(30, 2)).astype(np.float32)

    def render(centers):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w), dtype=np.float32)
        sigma = 1.5
        for (x, y) in centers:
            img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        img = img / max(img.max(), 1e-6) * 4000.0
        return img.astype(np.uint16)

    tifffile.imwrite(scenario_dir / "reference.tif", render(positions))
    tifffile.imwrite(scenario_dir / "deformed.tif",
                     render(positions + np.array(shift_px, dtype=np.float32)))

    # GT fields in microns. pixel_size_um=0.1 → 2 px shift = 0.2 µm.
    dx_um = np.full((h, w), shift_px[0] * 0.1, dtype=np.float32)
    dy_um = np.full((h, w), shift_px[1] * 0.1, dtype=np.float32)
    np.save(scenario_dir / "displacement_x.npy", dx_um)
    np.save(scenario_dir / "displacement_y.npy", dy_um)


def test_runner_smoke_end_to_end(tmp_path: Path):
    bench_root = tmp_path / "bench"
    _make_synthetic_scenario(bench_root / "tiny", shift_px=(2.0, 1.0))

    out_dir = tmp_path / "out"

    run(
        benchmark_root=bench_root,
        scenarios=["tiny"],
        algorithms=["DIS", "Farneback", "Lucas-Kanade"],
        output_dir=out_dir,
    )

    # Summary artifacts
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "summary.png").exists()
    assert (out_dir / "displacements.csv").exists()

    # Per-algorithm and combined plots
    for algo in ("DIS", "Farneback", "Lucas-Kanade"):
        assert (out_dir / "tiny" / f"{algo}.png").exists()
    assert (out_dir / "tiny" / "combined.png").exists()

    # Summary CSV sanity
    df = pd.read_csv(out_dir / "summary.csv")
    assert set(df["algorithm"]) == {"DIS", "Farneback", "Lucas-Kanade"}
    assert (df["rmse_px"] >= 0).all()
