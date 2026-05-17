import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_COLUMNS: tuple[str, ...] = (
    "scenario", "algorithm", "n_beads", "coverage",
    "rmse_px", "median_px", "p95_px",
    "bias_low", "bias_mid", "bias_high",
)

TIDY_COLUMNS: tuple[str, ...] = (
    "scenario", "algorithm", "bead_id",
    "ref_x", "ref_y",
    "pred_dx", "pred_dy",
    "gt_dx", "gt_dy",
    "err_x", "err_y", "err_mag", "valid",
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _render_panels(
    ax_quiver,
    ax_err,
    reference_image: np.ndarray,
    bead_positions: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    err_vmax: float | None = None,
) -> None:
    ax_quiver.imshow(reference_image, cmap="gray", origin="upper")
    pts = bead_positions[valid]
    ax_quiver.quiver(
        pts[:, 0], pts[:, 1],
        ground_truth[valid, 0], ground_truth[valid, 1],
        color="lime", angles="xy", scale_units="xy", scale=1,
        width=0.003, label="GT",
    )
    ax_quiver.quiver(
        pts[:, 0], pts[:, 1],
        predicted[valid, 0], predicted[valid, 1],
        color="red", angles="xy", scale_units="xy", scale=1,
        width=0.003, label="pred",
    )
    ax_quiver.axis("off")
    ax_quiver.legend(loc="upper right", fontsize=7)

    err = predicted - ground_truth
    err_mag = np.hypot(err[:, 0], err[:, 1])
    vmax = err_vmax if err_vmax is not None else (float(np.nanpercentile(err_mag[valid], 95)) if valid.any() else 1.0)
    sc = ax_err.scatter(
        bead_positions[valid, 0], bead_positions[valid, 1],
        c=err_mag[valid], cmap="hot_r", s=8, vmin=0, vmax=max(vmax, 1e-6),
    )
    ax_err.set_xlim(0, reference_image.shape[1])
    ax_err.set_ylim(reference_image.shape[0], 0)
    ax_err.set_aspect("equal")
    ax_err.axis("off")
    plt.colorbar(sc, ax=ax_err, fraction=0.046, pad=0.02, label="error (px)")


def write_per_algorithm_png(
    out_path: Path,
    reference_image: np.ndarray,
    bead_positions: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    title: str,
) -> None:
    """Two-panel PNG: quiver overlay + per-bead error heatmap."""
    _ensure_parent(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    _render_panels(ax1, ax2, reference_image, bead_positions, predicted, ground_truth, valid)
    ax1.set_title(f"{title} — {int(valid.sum())} / {len(valid)} beads")
    ax2.set_title("error vs GT")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_combined_scenario_png(
    out_path: Path,
    reference_image: np.ndarray,
    bead_positions: np.ndarray,
    per_algorithm: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scenario_name: str,
) -> None:
    """One row per algorithm; shared error color scale across rows."""
    _ensure_parent(out_path)
    algos = list(per_algorithm.keys())
    n_rows = len(algos)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    # Shared color scale: max of 95th-percentile error across algorithms.
    vmax = 0.0
    for pred, gt, valid in per_algorithm.values():
        if valid.any():
            err_mag = np.hypot((pred - gt)[:, 0], (pred - gt)[:, 1])
            vmax = max(vmax, float(np.nanpercentile(err_mag[valid], 95)))
    vmax = max(vmax, 1e-6)

    for row, algo in enumerate(algos):
        pred, gt, valid = per_algorithm[algo]
        _render_panels(axes[row, 0], axes[row, 1], reference_image, bead_positions,
                       pred, gt, valid, err_vmax=vmax)
        axes[row, 0].set_title(f"{algo} — quivers")
        axes[row, 1].set_title(f"{algo} — error (shared scale, vmax={vmax:.2f} px)")

    fig.suptitle(f"Scenario: {scenario_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(out_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write the summary CSV with a fixed column order."""
    _ensure_parent(out_path)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})


def write_summary_bar_chart(out_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Grouped bar chart of RMSE per (scenario, algorithm)."""
    _ensure_parent(out_path)
    df = pd.DataFrame(list(rows))
    pivot = df.pivot(index="scenario", columns="algorithm", values="rmse_px")

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("RMSE (px)")
    ax.set_title("Displacement RMSE by scenario x algorithm")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_tidy_displacements_csv(
    out_path: Path,
    scenario: str,
    algorithm: str,
    bead_positions: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    append: bool,
) -> None:
    """Tidy long-format per-bead rows. `append=False` writes a header."""
    _ensure_parent(out_path)
    err = predicted - ground_truth
    err_mag = np.hypot(err[:, 0], err[:, 1])
    n = len(bead_positions)
    rows = [
        {
            "scenario": scenario,
            "algorithm": algorithm,
            "bead_id": i,
            "ref_x": float(bead_positions[i, 0]),
            "ref_y": float(bead_positions[i, 1]),
            "pred_dx": float(predicted[i, 0]),
            "pred_dy": float(predicted[i, 1]),
            "gt_dx": float(ground_truth[i, 0]),
            "gt_dy": float(ground_truth[i, 1]),
            "err_x": float(err[i, 0]),
            "err_y": float(err[i, 1]),
            "err_mag": float(err_mag[i]),
            "valid": bool(valid[i]),
        }
        for i in range(n)
    ]
    mode = "a" if append else "w"
    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TIDY_COLUMNS))
        if not append:
            writer.writeheader()
        writer.writerows(rows)
