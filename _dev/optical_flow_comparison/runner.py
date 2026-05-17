"""Optical flow comparison runner.

Usage:
    python _dev/optical_flow_comparison/runner.py \
        --scenarios low mid high \
        --algorithms DIS Farneback Lucas-Kanade \
        --benchmark-root _validation/benchmark_TFM \
        --output-dir _dev/optical_flow_comparison/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _dev.optical_flow_comparison.adapters.base import FlowAdapter, sample_dense_at_points
from _dev.optical_flow_comparison.adapters.dis import DISAdapter
from _dev.optical_flow_comparison.adapters.farneback import FarnebackAdapter
from _dev.optical_flow_comparison.adapters.lucas_kanade import LucasKanadeAdapter
from _dev.optical_flow_comparison.adapters.lucas_kanade_fb import LucasKanadeFBAdapter
from _dev.optical_flow_comparison.adapters.tvl1 import TVL1Adapter
from _dev.optical_flow_comparison.detection import detect_beads
from _dev.optical_flow_comparison.metrics import compute_metrics
from _dev.optical_flow_comparison.preprocessing import preprocess
from _dev.optical_flow_comparison.reporting import (
    write_combined_scenario_png,
    write_per_algorithm_png,
    write_summary_bar_chart,
    write_summary_csv,
    write_tidy_displacements_csv,
)

PIXEL_SIZE_UM = 0.1

ADAPTERS: dict[str, type[FlowAdapter]] = {
    "DIS": DISAdapter,
    "Farneback": FarnebackAdapter,
    "Lucas-Kanade": LucasKanadeAdapter,
    "Lucas-Kanade-FB": LucasKanadeFBAdapter,
    "TV-L1": TVL1Adapter,
}


def _load_scenario(scenario_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load + preprocess ref and deformed; load GT as a (H, W, 2) field in px."""
    ref = preprocess(tifffile.imread(scenario_dir / "reference.tif").astype(np.float32))
    deformed = preprocess(tifffile.imread(scenario_dir / "deformed.tif").astype(np.float32))
    gt_dx_um = np.load(scenario_dir / "displacement_x.npy")
    gt_dy_um = np.load(scenario_dir / "displacement_y.npy")
    gt_px = np.stack([gt_dx_um / PIXEL_SIZE_UM, gt_dy_um / PIXEL_SIZE_UM], axis=-1).astype(np.float32)
    return ref, deformed, gt_px


def run(
    benchmark_root: Path,
    scenarios: Sequence[str],
    algorithms: Sequence[str],
    output_dir: Path,
) -> None:
    """Run every (scenario, algorithm) pair and write all artifacts."""
    benchmark_root = Path(benchmark_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy_csv = output_dir / "displacements.csv"
    if tidy_csv.exists():
        tidy_csv.unlink()  # fresh run each invocation

    summary_rows: list[dict] = []

    for scenario in scenarios:
        scenario_dir = benchmark_root / scenario
        print(f"[{scenario}] loading...")
        ref, deformed, gt_field = _load_scenario(scenario_dir)
        beads = detect_beads(ref)
        gt_at_beads = sample_dense_at_points(gt_field, beads)
        print(f"[{scenario}] detected {len(beads)} beads")

        per_algo: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for algo_name in algorithms:
            adapter_cls = ADAPTERS[algo_name]
            adapter = adapter_cls()
            print(f"[{scenario}/{algo_name}] running...")
            pred, valid = adapter.displacements_at(ref, deformed, beads)
            metrics = compute_metrics(pred, gt_at_beads, valid)
            print(f"[{scenario}/{algo_name}] RMSE={metrics['rmse_px']:.3f} px  "
                  f"median={metrics['median_px']:.3f} px  coverage={metrics['coverage']:.2%}")

            write_per_algorithm_png(
                out_path=output_dir / scenario / f"{algo_name}.png",
                reference_image=ref,
                bead_positions=beads,
                predicted=pred,
                ground_truth=gt_at_beads,
                valid=valid,
                title=f"{scenario} / {algo_name}",
            )
            write_tidy_displacements_csv(
                out_path=tidy_csv,
                scenario=scenario,
                algorithm=algo_name,
                bead_positions=beads,
                predicted=pred,
                ground_truth=gt_at_beads,
                valid=valid,
                append=tidy_csv.exists(),
            )

            summary_rows.append({"scenario": scenario, "algorithm": algo_name, **metrics})
            per_algo[algo_name] = (pred, gt_at_beads, valid)

        write_combined_scenario_png(
            out_path=output_dir / scenario / "combined.png",
            reference_image=ref,
            bead_positions=beads,
            per_algorithm=per_algo,
            scenario_name=scenario,
        )

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_bar_chart(output_dir / "summary.png", summary_rows)
    print(f"\nDone. Artifacts in {output_dir}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare optical flow algorithms on TFM benchmarks.")
    p.add_argument("--scenarios", nargs="+", default=["low", "mid", "high"])
    p.add_argument("--algorithms", nargs="+", default=list(ADAPTERS.keys()),
                   choices=list(ADAPTERS.keys()))
    p.add_argument("--benchmark-root", type=Path,
                   default=_REPO_ROOT / "_validation" / "benchmark_TFM")
    p.add_argument("--output-dir", type=Path,
                   default=_HERE / "output")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        benchmark_root=args.benchmark_root,
        scenarios=args.scenarios,
        algorithms=args.algorithms,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
