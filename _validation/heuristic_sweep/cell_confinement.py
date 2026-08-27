#!/usr/bin/env python3
"""Does mask confinement earn its keep on diffuse cells?

The main sweep ran with NO mask (mask=None, fwd_mask_strength=0) everywhere, on
purpose: it picked a regularization heuristic that ships safe for the user who
hasn't drawn a segmentation. So the sweep structurally could not answer whether
*adding* a cell mask helps. This script asks that, on the one field structure
where it is a fair question -- the diffuse cell scenes (a dipole's "mask" is just
its two blobs; confining to them merely restates the GT).

The honest prior is the cell OUTLINE (cell_mask.npy, from make_cells), NOT the GT
traction support: a real user segments the cell in brightfield, and the outline is
genuinely looser than the traction (which concentrates at the periphery/adhesions).
Confining to the GT support would be an oracle; we include it only as an upper
bound on what confinement could ever buy.

Everything else is held fixed so the mask is the ONLY variable: same cached PIV
window-24 displacement the cell sweep used, shipped-default regularization
(l1_sparsity=0.05, l2=0). We vary fwd_mask_strength along the 0..100 dial with the
cell-outline mask, and score:
  * nRMSE            -- the ranking metric (whole-field; background garbage, where
                        GT=0, adds straight to it, so confinement CAN lower it)
  * bg_leak / ext    -- spurious energy off the source / outside the cell
  * in-cell nRMSE    -- error inside the outline

Is the in-solver penalty worth its complexity, or would a post-hoc "zero everything
outside the mask" do? We answer it head-on with a "hard" variant: take the no-mask
solve and zero the exterior. Its interior is bit-identical to the baseline, so it
isolates *background removal only*. If the soft solver beats it INSIDE the cell, the
penalty is doing real work the simple version can't -- and it does: the non-local
Green's operator lets the unconstrained solve park real interior signal in exterior
pixels, which post-hoc zeroing deletes (leaving the interior under-fit) but the
in-objective penalty re-explains with interior force (better peaks). Measured here:
the soft solver lowers in-cell nRMSE below baseline in every useful-window scene,
while the post-hoc zero leaves it exactly at baseline.

Cells rank on whole-field nRMSE (the Sabass J is
undefined on a diffuse centripetal field). GT is the stored fitted-fibre traction.

Usage:  python cell_confinement.py [--stage $STAGE] [--outdir figures]
                                   [--scenarios-dir <benchmarkTFM scenarios>]  # backfill cell_mask.npy
"""
from __future__ import annotations
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import tifffile
import sweep_config as C
from scoring import metrics
from napariTFM.backend.forward_l1 import l1_traction_frame
from napariTFM.backend.parameter_dataclasses import FTTCParameters

HERE = os.path.dirname(os.path.abspath(__file__))
COND = "cell_s6j1"
N = C.GT_REFERENCE_SIZE
RES = 24.0                      # PIV window held fixed (the sweep-recommended displacement)
L1_FIXED = 0.05                 # shipped-default sparsity -- "with the defaults, does a mask help?"
DIALS = [0, 20, 40, 60, 80, 100]        # fwd_mask_strength; 0 == no confinement (baseline)
GT_FRAC = 0.05                  # GT-support oracle mask: |t_gt| > frac * max
# strength bands: noise floor / useful window / breakdown (px)
BANDS = [("noisy (|u|≤1.2)", lambda p: p <= 1.2),
         ("useful (1.2–8)",  lambda p: 1.2 < p <= 8.0),
         ("strong (|u|>8)",  lambda p: p > 8.0)]


def scene_list(stage):
    root = os.path.join(stage, "scenes", COND)
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_field(stage, scene):
    """Cached PIV window-24 displacement field (h,w,2) µm."""
    for cf in glob.glob(os.path.join(stage, "cache", COND, scene, "disp_PIV_res*.npz")):
        d = np.load(cf)
        if abs(float(d["res_val"]) - RES) < 1e-6:
            return d["field"]
    raise SystemExit(f"no PIV res={RES:g} cache for {scene}")


def load_cell_mask(stage, scene, scenarios_dir):
    """Cell-outline mask (512²) from the scene dir; backfill from benchmarkTFM if absent."""
    p = os.path.join(stage, "scenes", COND, scene, "cell_mask.npy")
    if os.path.exists(p):
        return np.load(p) > 0
    if scenarios_dir:
        cell = scene.rsplit("_", 1)[0]
        ome = os.path.join(scenarios_dir, cell, f"{cell}.ome.tif")
        with tifffile.TiffFile(ome) as t:
            m = {s.name: s.asarray() for s in t.series}["mask"] > 0
        np.save(p, m)                                  # backfill so re-runs are stage-only
        return m
    raise SystemExit(f"no cell_mask.npy for {scene}; re-run make_cells or pass --scenarios-dir to backfill")


def to_grid(mask512, h):
    """Nearest-neighbour resample a 512² mask onto the h×h force grid."""
    return zoom(mask512.astype(np.float32), (h / mask512.shape[0], h / mask512.shape[1]), order=0) > 0.5


def invert(field, mask_h, dial):
    """FISTA group-L1 at the shipped default sparsity, confined to mask_h at strength `dial`."""
    h = field.shape[0]
    eff_ps = C.PIXEL_SIZE_UM * (N / h)
    p = FTTCParameters(young_modulus=C.YOUNG_MODULUS, poisson_ratio_substrate=C.POISSON,
                       pixel_size=eff_ps, downscale_factor=1, fwd_device="cpu", fwd_dtype="float32",
                       fwd_mask_strength=float(dial), l1_max_iter=C.L1_MAX_ITER,
                       l1_sparsity=L1_FIXED, l2_ridge=0.0)
    t = np.asarray(l1_traction_frame(field, p, mask=mask_h))
    return zoom(t, (1, N / h, N / h), order=1)                # -> (2,N,N)


def score(t_up, gt, cell512):
    """nRMSE + bg_leak (from metrics), plus mask-defined exterior fraction and in-cell nRMSE."""
    m = metrics(t_up, gt, do_sabass=False)
    magt = np.hypot(t_up[0], t_up[1])
    ext = float(magt[~cell512].sum() / (magt.sum() or 1.0))         # recovered energy OUTSIDE the cell
    gnorm = float(np.sqrt((gt ** 2).sum())) or 1.0
    din = ((t_up - gt) ** 2).sum(axis=0)
    in_nrmse = float(np.sqrt(din[cell512].sum()) / gnorm)          # error INSIDE the outline
    return dict(nrmse=m["nrmse"], bg_leak=m["bg_leak"], ext_frac=ext, in_nrmse=in_nrmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="sweep stage dir (default: $STAGE)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"),
                    help="figure + CSV dir (use ../../docs/images to refresh the report)")
    ap.add_argument("--scenarios-dir", default=None,
                    help="benchmarkTFM scenarios dir; only needed to backfill cell_mask.npy "
                         "into a stage generated before make_cells saved it")
    a = ap.parse_args()
    stage = a.stage or os.environ.get("STAGE")
    if not stage:
        raise SystemExit("set --stage or `source env.sh` to export STAGE")
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for scene in scene_list(stage):
        P = float(scene.split("_u")[1]); cell = scene.rsplit("_", 1)[0]
        field = load_field(stage, scene)
        h = field.shape[0]
        gt = np.load(os.path.join(stage, "scenes", COND, scene, "gt_traction.npy")).astype(np.float32)
        cell512 = load_cell_mask(stage, scene, a.scenarios_dir)
        cell_h = to_grid(cell512, h)
        gt_support = np.hypot(gt[0], gt[1]) > GT_FRAC * np.hypot(gt[0], gt[1]).max()
        gt_h = to_grid(gt_support, h)

        # baseline: no mask (identical to dial 0)
        t_base = invert(field, None, 0.0)
        base = score(t_base, gt, cell512)
        rows.append(dict(scene=scene, cell=cell, P=P, mask="none", dial=0, **base))
        # the SIMPLE alternative: take the baseline solve and post-hoc zero everything
        # outside the outline. Interior is bit-identical to base by construction, so this
        # isolates "background removal only" -- the solver's in-cell refit must beat it.
        t_hard = t_base.copy(); t_hard[:, ~cell512] = 0.0
        rows.append(dict(scene=scene, cell=cell, P=P, mask="hard", dial=0,
                         **score(t_hard, gt, cell512)))
        # honest prior: cell outline, up the dial (the in-solver soft penalty)
        for dial in DIALS[1:]:
            rows.append(dict(scene=scene, cell=cell, P=P, mask="cell", dial=dial,
                             **score(invert(field, cell_h, dial), gt, cell512)))
        # oracle ceiling: GT support, up the dial
        for dial in DIALS[1:]:
            rows.append(dict(scene=scene, cell=cell, P=P, mask="gt_oracle", dial=dial,
                             **score(invert(field, gt_h, dial), gt, cell512)))
        b = min((r for r in rows if r["scene"] == scene and r["mask"] == "cell"), key=lambda r: r["nrmse"])
        hard = next(r for r in rows if r["scene"] == scene and r["mask"] == "hard")
        print(f"  {scene}: base in-cell={base['in_nrmse']:.3f}  post-hoc-zero in-cell={hard['in_nrmse']:.3f} "
              f"(nRMSE {hard['nrmse']:.3f})  soft in-cell={b['in_nrmse']:.3f} (nRMSE {b['nrmse']:.3f} @dial {b['dial']})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.outdir, "cell_confinement.csv"), index=False)
    summarize(df)
    figure(df, a.outdir)


def summarize(df):
    base = df[df["mask"] == "none"].set_index("scene")
    hard = df[df["mask"] == "hard"].set_index("scene")
    print(f"\n=== Does mask confinement earn its keep? ({df.scene.nunique()} scenes, "
          f"l1={L1_FIXED}, PIV win-{RES:g}) ===")
    print("  whole-field nRMSE  |  in-cell nRMSE   (post-hoc zero = baseline solve, exterior deleted)")
    for name, sel in BANDS:
        sc = [s for s in df.scene.unique() if sel(float(s.split("_u")[1]))]
        if not sc:
            continue
        b0 = base.loc[sc]
        cellrows = df[(df["mask"] == "cell") & df.scene.isin(sc)]
        soft = cellrows.loc[cellrows.groupby("scene").nrmse.idxmin()].set_index("scene")
        orc = df[(df["mask"] == "gt_oracle") & df.scene.isin(sc)].groupby("scene").nrmse.min()
        refit = (1 - soft.in_nrmse / b0.in_nrmse)               # in-cell improvement, solver vs baseline
        print(f"  {name:16s} n={len(sc):2d} | no-mask {b0.nrmse.median():.3f}/{b0.in_nrmse.median():.3f}"
              f"  post-hoc-zero {hard.loc[sc].nrmse.median():.3f}/{hard.loc[sc].in_nrmse.median():.3f}"
              f"  soft {soft.nrmse.median():.3f}/{soft.in_nrmse.median():.3f}"
              f"  | solver refits in-cell {refit.median() * 100:+.0f}% in {int((refit > 0.005).mean() * 100)}% of scenes"
              f"  | oracle {orc.median():.3f}")
    print("  Post-hoc zero leaves in-cell == baseline by construction; the soft solver's win over it is all interior.")


def figure(df, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    # A: nRMSE vs dial, normalized to baseline, per strength band + oracle ceiling
    a = ax[0]
    base = df[df["mask"] == "none"].set_index("scene").nrmse
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for (name, sel), col in zip(BANDS, colors):
        sc = [s for s in df.scene.unique() if sel(float(s.split("_u")[1]))]
        if not sc:
            continue
        sub = df[(df["mask"] == "cell") & df.scene.isin(sc)].copy()
        sub["rel"] = sub.nrmse.values / base.loc[sub.scene].values
        curve = sub.groupby("dial").rel.median()
        curve = pd.concat([pd.Series({0: 1.0}), curve])       # dial 0 == baseline == 1.0
        a.plot(curve.index, curve.values, "o-", color=col, lw=2, label=name)
        orc = df[(df["mask"] == "gt_oracle") & df.scene.isin(sc)].copy()
        orc["rel"] = orc.nrmse.values / base.loc[orc.scene].values
        a.axhline(orc.groupby("scene").rel.min().median(), ls=":", color=col, alpha=0.6)
    a.axhline(1.0, color="k", lw=1)
    a.text(2, 1.002, "baseline (no mask)", fontsize=8, va="bottom")
    a.set_xlabel("fwd_mask_strength dial"); a.set_ylabel("nRMSE / baseline  (<1 = confinement helps)")
    a.set_title("Does the cell mask earn its keep?\n(dotted = GT-support oracle ceiling per band)",
                fontsize=12, fontweight="bold")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    # B: is the in-solver penalty worth it over a post-hoc zero? Whole-field nRMSE
    # relative to no-mask, per band, for post-hoc-zero vs the soft solver, with the
    # solver's in-cell nRMSE marked -- the gap between the bars is the interior refit
    # that post-hoc zeroing (interior == baseline) cannot get.
    b = ax[1]
    base_i = df[df["mask"] == "none"].set_index("scene")
    x = np.arange(len(BANDS)); wbar = 0.36
    hard_rel, soft_rel, soft_in = [], [], []
    for name, sel in BANDS:
        sc = [s for s in df.scene.unique() if sel(float(s.split("_u")[1]))]
        b0 = base_i.loc[sc]
        hard = df[(df["mask"] == "hard") & df.scene.isin(sc)].set_index("scene")
        cr = df[(df["mask"] == "cell") & df.scene.isin(sc)]
        soft = cr.loc[cr.groupby("scene").nrmse.idxmin()].set_index("scene")
        hard_rel.append((hard.nrmse / b0.nrmse).median())
        soft_rel.append((soft.nrmse / b0.nrmse).median())
        soft_in.append((soft.in_nrmse / b0.in_nrmse).median())
    b.bar(x - wbar / 2, hard_rel, wbar, color="#9467bd", label="post-hoc zero (whole-field)")
    b.bar(x + wbar / 2, soft_rel, wbar, color="#2ca02c", label="soft solver (whole-field)")
    b.plot(x + wbar / 2, soft_in, "D", color="#1f77b4", ms=9, label="soft solver, in-cell only")
    ui = 1                                                     # annotate the useful (operating) band only
    b.annotate("solver's\ninterior refit", xy=(ui + wbar / 2, soft_rel[ui]),
               xytext=(ui - 0.42, 0.72), fontsize=8.5, ha="center",
               arrowprops=dict(arrowstyle="->", color="k", lw=1.3))
    b.axhline(1.0, color="k", lw=1); b.text(len(BANDS) - 0.55, 1.005, "no-mask baseline", fontsize=8, ha="right", va="bottom")
    b.set_xticks(x); b.set_xticklabels([n for n, _ in BANDS], fontsize=9)
    b.set_ylabel("nRMSE / no-mask baseline  (<1 = better)")
    b.set_ylim(0, 1.2)
    b.set_title("Post-hoc zero vs the solver: the gap is interior\n(post-hoc zero's in-cell == baseline by construction)",
                fontsize=12, fontweight="bold")
    b.legend(fontsize=8.5, loc="upper left"); b.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(outdir, "heuristic-sweep-cells-confinement.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("\nwrote", out)


if __name__ == "__main__":
    main()
