#!/usr/bin/env python3
"""Illustrative best-nRMSE traction recovery per method on a few diffuse-cell
scenes -- the face on cell_aggregate.py's competence numbers.

Companion to compare_methods.py (dipoles). Each method panel is that method's own
best-nRMSE operating point for the scene; the winner is boxed green. GT is the
stored benchmarkTFM fitted-fibre traction (gt_traction.npy). Ranking is whole-field
nRMSE (the Sabass J is undefined on a diffuse centripetal field).

Usage:  python cell_examples.py [--stage $STAGE] [--outdir figures]
"""
from __future__ import annotations
import argparse, csv, glob, os, tomllib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import sweep_config as C
from napariTFM.backend.forward_l1 import l1_traction_frame
from napariTFM.backend.parameter_dataclasses import FTTCParameters

HERE = os.path.dirname(os.path.abspath(__file__))
COND = "cell_s6j1"
N = C.GT_REFERENCE_SIZE
METHODS = ["PIV", "ILK", "FFD"]
MNAME = {"PIV": "PIV (x-corr)", "ILK": "iLK (optical flow)", "FFD": "FFD (B-spline)"}
# (scene, the lesson it illustrates)
SCENES = [
    ("synth00_u3.155", "82-fibre diffuse cell\nmid strength"),
    ("synth02_u3.155", "sweet spot\n(best recovery)"),
    ("synth03_u19.905", "high strength\nFFD holds up"),
]


def best_row(stage, scene, method):
    """min-nRMSE row for one method from the scene's shard."""
    best = None
    with open(os.path.join(stage, "results", f"sweep_{COND}_{scene}.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["method"] != method:
                continue
            if best is None or float(r["nrmse"]) < float(best["nrmse"]):
                best = r
    return best


def invert_best(stage, scene, row):
    key = row["method"]; res_val = float(row["res_val"])
    field = None
    for cf in glob.glob(os.path.join(stage, "cache", COND, scene, f"disp_{key}_res*.npz")):
        d = np.load(cf)
        if abs(float(d["res_val"]) - res_val) < 1e-6:
            field = d["field"]; break
    h = field.shape[0]
    eff_ps = C.PIXEL_SIZE_UM * (N / h)
    p = FTTCParameters(young_modulus=C.YOUNG_MODULUS, poisson_ratio_substrate=C.POISSON,
                       pixel_size=eff_ps, downscale_factor=1, fwd_device="cpu",
                       fwd_dtype="float32", fwd_mask_strength=0.0, l1_max_iter=C.L1_MAX_ITER,
                       l1_sparsity=float(row["l1"]), l2_ridge=float(row["l2"]))
    t = np.asarray(l1_traction_frame(field, p, mask=None))
    return zoom(t, (1, N / h, N / h), order=1)


def quiver(ax, vec, step, color, scale):
    H = vec.shape[1]; s = N / H
    yy, xx = np.mgrid[0:H:step, 0:H:step]
    ax.quiver(xx * s, yy * s, vec[0, ::step, ::step], -vec[1, ::step, ::step],
              color=color, scale=scale, width=0.005, headwidth=4, alpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="sweep stage dir (default: $STAGE)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    stage = a.stage or os.environ.get("STAGE")
    if not stage:
        raise SystemExit("set --stage or `source env.sh` to export STAGE")
    os.makedirs(a.outdir, exist_ok=True)

    fig, axes = plt.subplots(len(SCENES), 4, figsize=(16, 4 * len(SCENES)))
    for i, (scene, lesson) in enumerate(SCENES):
        gt = np.load(os.path.join(stage, "scenes", COND, scene, "gt_traction.npy"))
        rows = {m: best_row(stage, scene, m) for m in METHODS}
        recon = {m: invert_best(stage, scene, rows[m]) for m in METHODS}
        winner = min(METHODS, key=lambda m: float(rows[m]["nrmse"]))
        tmag_gt = np.hypot(gt[0], gt[1])
        tvmax = max([tmag_gt.max()] + [np.hypot(recon[m][0], recon[m][1]).max() for m in METHODS]) or 1.0

        ax = axes[i, 0]
        ax.imshow(tmag_gt, cmap="magma", vmin=0, vmax=tvmax)
        quiver(ax, gt, 42, "cyan", tvmax * 22)
        ax.set_ylabel(f"{scene}\n{lesson}", fontsize=9, fontweight="bold", labelpad=8)
        if i == 0:
            ax.set_title("GT traction |t| (fitted fibres)", fontsize=12, fontweight="bold")

        for k, m in enumerate(METHODS):
            ax = axes[i, k + 1]; tm = recon[m]; r = rows[m]
            ax.imshow(np.hypot(tm[0], tm[1]), cmap="magma", vmin=0, vmax=tvmax)
            quiver(ax, tm, 42, "cyan", tvmax * 22)
            won = (m == winner)
            ax.set_xlabel(f"nRMSE={float(r['nrmse']):.3f}  ang={float(r['ang_field']):.0f}°  "
                          f"l1={float(r['l1']):.3f} l2={float(r['l2']):g}",
                          fontsize=9, fontweight="bold" if won else "normal",
                          color="green" if won else "black")
            if i == 0:
                ax.set_title(MNAME[m], fontsize=12, fontweight="bold")
            if won:
                for sp in ax.spines.values():
                    sp.set_edgecolor("green"); sp.set_linewidth(3.5)

        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    fig.suptitle("Best-nRMSE traction recovery per method, diffuse cells — winner boxed green",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(a.outdir, "heuristic-sweep-cells-examples.png")
    fig.savefig(out, dpi=115, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
