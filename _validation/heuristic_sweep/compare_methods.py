#!/usr/bin/env python3
"""Side-by-side best-J traction recovery, PIV vs iLK vs FFD, for a few telling
scenes -- puts a face on the competence numbers. Each method panel is that
method's own best-J operating point for the scene; the winner is boxed green.
"""
from __future__ import annotations
import os, sys, glob, csv, tomllib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import sweep_config as C
from sweep_forces import rasterize_gt
from napariTFM.backend.forward_l1 import l1_traction_frame
from napariTFM.backend.parameter_dataclasses import FTTCParameters

_HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", default=None, help="sweep stage dir (default: $STAGE)")
_ap.add_argument("--outdir", default=os.path.join(_HERE, "figures"),
                 help="where to write the figure (use ../../docs/images to refresh the report)")
_a = _ap.parse_args()
STAGE = _a.stage or os.environ.get("STAGE")
if not STAGE:
    raise SystemExit("set --stage or `source env.sh` to export STAGE")
os.makedirs(_a.outdir, exist_ok=True)
CONDS = [f"s{s}_j{j}" for s in range(8) for j in (1, 3)]
N = C.GT_REFERENCE_SIZE
METHODS = ["PIV", "ILK", "FFD"]
MNAME = {"PIV": "PIV (x-corr)", "ILK": "iLK (optical flow)", "FFD": "FFD (B-spline)"}
# (scene, method the median analysis says should win, the lesson it illustrates)
SCENES = [
    ("f1.88_u3.155",  "FFD", "broad source, mid disp\nFFD smoothness wins"),
    ("f5_u1.256",     "ILK", "large + sub-px disp\niLK wins (thin margin)"),
    ("f0.707_u3.155", "PIV", "compact source\nPIV window wins"),
    ("f0.266_u1.256", "PIV", "small footprint\nall mediocre, PIV least-bad"),
]
OUT = os.path.join(_a.outdir, "heuristic-sweep-examples.png")


def best_row_for(cond, scene, method):
    """min-J row for one method from the scene's shard in a given condition."""
    best = None
    with open(os.path.join(STAGE, "results", f"sweep_{cond}_{scene}.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["method"] != method:
                continue
            if best is None or float(r["J"]) < float(best["J"]):
                best = r
    return best


def pick_condition(scene, want):
    """Among imaging conditions where `want` genuinely wins, choose the one where
    it recovers cleanest (lowest own J) -- shows the winner at its representative
    best while still being a real win, not a win-by-others'-collapse."""
    best_cond, best_J = None, np.inf
    for cond in CONDS:
        rows = {m: best_row_for(cond, scene, m) for m in METHODS}
        if any(rows[m] is None for m in METHODS):
            continue
        Js = {m: float(rows[m]["J"]) for m in METHODS}
        if min(Js, key=Js.get) != want:
            continue
        if Js[want] < best_J:
            best_cond, best_J = cond, Js[want]
    return best_cond


def invert_best(cond, scene, row):
    key = row["method"]; res_val = float(row["res_val"])
    field = None
    for cf in glob.glob(os.path.join(STAGE, "cache", cond, scene, f"disp_{key}_res*.npz")):
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


fig, axes = plt.subplots(len(SCENES), 4, figsize=(16, 4 * len(SCENES)))
for i, (scene, want, lesson) in enumerate(SCENES):
    cond = pick_condition(scene, want)
    with open(os.path.join(STAGE, "scenes", cond, scene, "scene.toml"), "rb") as fh:
        sc = tomllib.load(fh)
    gt_t, _ = rasterize_gt(sc, N)
    rows = {m: best_row_for(cond, scene, m) for m in METHODS}
    recon = {m: invert_best(cond, scene, rows[m]) for m in METHODS}
    tmag_gt = np.hypot(gt_t[0], gt_t[1])
    tvmax = max([tmag_gt.max()] + [np.hypot(recon[m][0], recon[m][1]).max() for m in METHODS]) or 1.0
    winner = min(METHODS, key=lambda m: float(rows[m]["J"]))

    ax = axes[i, 0]
    ax.imshow(tmag_gt, cmap="magma", vmin=0, vmax=tvmax)
    quiver(ax, gt_t, 42, "cyan", tvmax * 22)
    ax.set_ylabel(f"{scene} [{cond}]\n{lesson}", fontsize=9, fontweight="bold", labelpad=8)
    if i == 0: ax.set_title("GT traction |t|", fontsize=12, fontweight="bold")

    for k, m in enumerate(METHODS):
        ax = axes[i, k + 1]
        tm = recon[m]; r = rows[m]
        ax.imshow(np.hypot(tm[0], tm[1]), cmap="magma", vmin=0, vmax=tvmax)
        quiver(ax, tm, 42, "cyan", tvmax * 22)
        won = (m == winner)
        ax.set_xlabel(f"J={float(r['J']):.2f}  res={float(r['res_val']):g}  "
                      f"l1={float(r['l1']):.3f} l2={float(r['l2']):g}",
                      fontsize=9, fontweight="bold" if won else "normal",
                      color="green" if won else "black")
        if i == 0: ax.set_title(MNAME[m], fontsize=12, fontweight="bold")
        if won:
            for sp in ax.spines.values():
                sp.set_edgecolor("green"); sp.set_linewidth(3.5)

    for j in range(4):
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

fig.suptitle("Best-J traction recovery per method — winner boxed green (imaging condition in row label)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=115, bbox_inches="tight")
print("wrote", OUT)
