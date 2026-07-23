#!/usr/bin/env python3
"""L2-path regularization comparison: FTTC+GCV vs Bayesian-L2 vs GT-tuned FTTC λ.

Motivating symptom (real data): the parameter-free auto methods disagree badly --
FTTC+GCV *under*-regularizes, Bayesian-L2 *over*-regularizes. This script quantifies
that on the synthetic ladder, where a GT traction exists, by driving the SAME force
code the app runs:

  * GT-tuned FTTC (oracle) -- plain Fourier Tikhonov, λ swept over a grid, the λ that
    minimizes the ranking objective vs GT kept. This is the reference the two auto
    methods are judged against, and the whole FTTC J(λ) curve is retained so GCV's λ
    can be placed on the same axis (both are the Fourier operator's λ).
  * FTTC + GCV -- λ chosen per frame by Generalized Cross-Validation
    (`find_gcv_regularization`). Same Fourier operator as the oracle, so λ_gcv is
    directly comparable to λ* (ratio < 1 => under-regularizes).
  * Bayesian-L2 (ABL2) -- `reconstruct_bl2_frame`, evidence-max λ. A DIFFERENT
    (real-space, standardized) operator, so its λ is NOT numerically comparable to the
    FTTC λ; it is judged by OUTCOME (objective, magnitude bias, background leak) only.

Objective: Sabass J for dipoles, whole-field nRMSE for cells (per the sweep's rule).
Runs across every cached (resolution, piv_smooth) displacement input so the effect of
the displacement-side smoother on the force-stage auto-λ is visible.

Usage:  python compare_l2_reg.py --stage "$STAGE" --condition cell_s6j1 [--outdir ...]
"""
from __future__ import annotations
import os, sys, glob, argparse, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import sweep_config as C
from sweep_forces import rasterize_gt, metrics
from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend.fttc import FTTC, find_gcv_regularization
from napariTFM.backend.bayesian_l2 import reconstruct_bl2_frame, estimate_bayesian_lambda

N = C.GT_REFERENCE_SIZE
LAMBDA_GRID = np.geomspace(1e-9, 1e-1, 33)   # FTTC Fourier λ sweep for the oracle


def load_gt(sdir, scene):
    meta = scene["meta"]
    if meta.get("kind") == "cell":
        return np.load(os.path.join(sdir, "gt_traction.npy")).astype(np.float32), "cell"
    gt, _ = rasterize_gt(scene, N)
    return gt, "dipole"


def objective(t_up, gt, kind):
    """Ranking scalar: Sabass J for dipoles, whole-field nRMSE for cells (+ full dict)."""
    m = metrics(t_up, gt, do_sabass=(kind != "cell"))
    m["obj"] = m["nrmse"] if kind == "cell" else m["J"]
    return m


def fttc_traction(field, params, lam):
    (_, _), forces = FTTC(params).calculate_traction(
        field, pixel_size=params.pixel_size, downscale_factor=1, regularization=lam)
    return np.asarray(forces)                        # (2, h, w)


def up(t, h):
    return zoom(t, (1, N / h, N / h), order=1)


def analyze_field(field, gt, kind, eff_ps):
    h = field.shape[0]
    params = FTTCParameters(young_modulus=C.YOUNG_MODULUS, poisson_ratio_substrate=C.POISSON,
                            pixel_size=eff_ps, downscale_factor=1, fwd_device="auto",
                            fwd_dtype="float32", fwd_mask_strength=0.0)
    # --- oracle FTTC λ sweep ---
    curve = []
    for lam in LAMBDA_GRID:
        m = objective(up(fttc_traction(field, params, lam), h), gt, kind)
        curve.append(m["obj"])
    curve = np.array(curve)
    i_star = int(np.nanargmin(curve))
    lam_star, J_oracle = float(LAMBDA_GRID[i_star]), float(curve[i_star])
    # --- FTTC + GCV (same operator) ---
    lam_gcv = float(find_gcv_regularization(field, params))
    m_gcv = objective(up(fttc_traction(field, params, lam_gcv), h), gt, kind)
    # --- Bayesian-L2 ABL2 (no mask; β inferred from residual) ---
    t_bl2 = np.asarray(reconstruct_bl2_frame(field, C.YOUNG_MODULUS, C.POISSON, eff_ps, mask=None))
    m_bl2 = objective(up(t_bl2, h), gt, kind)
    lam_bl2 = float(estimate_bayesian_lambda(field, C.YOUNG_MODULUS, C.POISSON, eff_ps, mask=None))
    # --- Bayesian-L2 BL2 (masked; β pinned from the cell-free exterior) ---
    # Proxy the user's segmentation mask by the significant-GT footprint at this grid.
    magg = np.hypot(gt[0], gt[1])
    sig_full = magg > 0.1 * magg.max()
    mask = zoom(sig_full.astype(np.float32), (h / N, h / N), order=1) > 0.5
    t_blm = np.asarray(reconstruct_bl2_frame(field, C.YOUNG_MODULUS, C.POISSON, eff_ps, mask=mask))
    m_blm = objective(up(t_blm, h), gt, kind)
    lam_blm = float(estimate_bayesian_lambda(field, C.YOUNG_MODULUS, C.POISSON, eff_ps, mask=mask))
    m_star = objective(up(fttc_traction(field, params, lam_star), h), gt, kind)
    return dict(lam_star=lam_star, J_oracle=J_oracle, magbias_oracle=m_star["mag_bias"], bgleak_oracle=m_star["bg_leak"],
                lam_gcv=lam_gcv, J_gcv=m_gcv["obj"], magbias_gcv=m_gcv["mag_bias"], bgleak_gcv=m_gcv["bg_leak"],
                lam_bl2=lam_bl2, J_bl2=m_bl2["obj"], magbias_bl2=m_bl2["mag_bias"], bgleak_bl2=m_bl2["bg_leak"],
                lam_blm=lam_blm, J_blm=m_blm["obj"], magbias_blm=m_blm["mag_bias"], bgleak_blm=m_blm["bg_leak"],
                curve=curve.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=os.environ.get("STAGE"))
    ap.add_argument("--condition", default="cell_s6j1")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    ap.add_argument("--scene", default=None, help="restrict to one scene (debug)")
    ap.add_argument("--res", type=float, default=None, help="restrict to one piv_window (e.g. 24)")
    a = ap.parse_args()
    if not a.stage:
        raise SystemExit("set --stage or source env.sh")
    os.makedirs(a.outdir, exist_ok=True)
    root = os.path.join(a.stage, "scenes", a.condition)
    scenes = [a.scene] if a.scene else sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    rows, curves = [], []
    for scene in scenes:
        sdir = os.path.join(root, scene)
        with open(os.path.join(sdir, "scene.toml"), "rb") as fh:
            sc = tomllib.load(fh)
        gt, kind = load_gt(sdir, sc)
        for cf in sorted(glob.glob(os.path.join(a.stage, "cache", a.condition, scene, "disp_PIV_*_sm*.npz"))):
            d = np.load(cf)
            if a.res is not None and abs(float(d["res_val"]) - a.res) > 1e-6:
                continue
            field = d["field"].astype(np.float32); h = field.shape[0]
            eff_ps = C.PIXEL_SIZE_UM * (N / h)
            r = analyze_field(field, gt, kind, eff_ps)
            curve = r.pop("curve")
            strength = float(scene.split("_u")[1]) if "_u" in scene else float("nan")
            rows.append(dict(scene=scene, kind=kind, strength=strength, res_val=float(d["res_val"]),
                             smooth_val=float(d["smooth_val"]), **r))
            curves.append((scene, float(d["res_val"]), float(d["smooth_val"]), curve))
        print(f"  {scene} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.outdir, f"l2_reg_compare_{a.condition}.csv"), index=False)
    _summarize(df)
    _figure(df, curves, a.outdir, a.condition)


def _summarize(df):
    print(f"\n=== L2-path regularization ({len(df)} scene×input rows) ===")
    # strength-resolved (real signal lives above the jitter floor ~u1)
    if "strength" in df and df.strength.notna().any():
        print("\n-- by signal strength (peak |u| px), piv_smooth pooled, ratios to oracle --")
        print(f"  {'strength':>9} {'GCV':>7} {'ABL2':>7} {'BL2m':>7}   {'λgcv/λ*':>9}")
        for s, g in df.groupby("strength"):
            print(f"  {s:>9.3g} {(g.J_gcv/g.J_oracle).median():>7.2f} "
                  f"{(g.J_bl2/g.J_oracle).median():>7.2f} {(g.J_blm/g.J_oracle).median():>7.2f}   "
                  f"{np.nanmedian(g.lam_gcv/g.lam_star):>9.3g}")
    for sm, g in df.groupby("smooth_val"):
        print(f"\n-- piv_smooth = {sm} ({len(g)} rows) --")
        print(f"  objective (lower=better)   oracle-λ FTTC : {g.J_oracle.median():.3f}")
        print(f"                             FTTC+GCV      : {g.J_gcv.median():.3f}   "
              f"({(g.J_gcv/g.J_oracle).median():.2f}× oracle)")
        print(f"                             Bayesian ABL2 : {g.J_bl2.median():.3f}   "
              f"({(g.J_bl2/g.J_oracle).median():.2f}× oracle)  [no mask]")
        print(f"                             Bayesian BL2  : {g.J_blm.median():.3f}   "
              f"({(g.J_blm/g.J_oracle).median():.2f}× oracle)  [masked]")
        print(f"  λ_gcv / λ_oracle (FTTC units): median {np.nanmedian(g.lam_gcv/g.lam_star):.3g}  "
              f"(<1 = GCV under-regularizes)")
        print(f"  magnitude bias  oracle {g.magbias_oracle.median():+.2f}   GCV {g.magbias_gcv.median():+.2f}   "
              f"ABL2 {g.magbias_bl2.median():+.2f}   BL2 {g.magbias_blm.median():+.2f}  "
              f"(negative = suppressed / over-regularized)")
        print(f"  background leak oracle {g.bgleak_oracle.median():.3f}   GCV {g.bgleak_gcv.median():.3f}   "
              f"ABL2 {g.bgleak_bl2.median():.3f}   BL2 {g.bgleak_blm.median():.3f}  "
              f"(high = noisy / under-regularized)")


def _figure(df, curves, outdir, cond):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    # (a) example FTTC J(λ) curves at one res, both smooths, with GCV & oracle marked
    a0 = ax[0]
    ex = [c for c in curves if abs(c[1] - 24.0) < 1e-6][:12]
    for scene, res, sm, curve in ex:
        col = "#1f77b4" if sm == 0.0 else "#d62728"
        a0.plot(LAMBDA_GRID, curve, color=col, alpha=0.5, lw=1)
    a0.set_xscale("log"); a0.set_xlabel("FTTC Fourier λ"); a0.set_ylabel("objective (J or nRMSE)")
    a0.set_title("FTTC J(λ) curves (win=24)\nblue=piv_smooth 0, red=1", fontsize=11, fontweight="bold")
    a0.grid(alpha=0.3, which="both")
    # (b) objective by method, split by smooth
    b = ax[1]
    labels, data, colors = [], [], []
    for sm in sorted(df.smooth_val.unique()):
        g = df[df.smooth_val == sm]
        for name, col in [("oracle", "#2ca02c"), ("GCV", "#1f77b4"), ("BL2", "#d62728")]:
            key = {"oracle": "J_oracle", "GCV": "J_gcv", "BL2": "J_bl2"}[name]
            labels.append(f"{name}\nsm{sm:g}"); data.append(g[key].values); colors.append(col)
    bp = b.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for p, c in zip(bp["boxes"], colors):
        p.set_facecolor(c); p.set_alpha(0.6)
    b.set_ylabel("objective (lower=better)")
    b.set_title(f"Recovery error by L2 strategy\n({cond})", fontsize=11, fontweight="bold")
    b.grid(alpha=0.3, axis="y")
    # (c) λ_gcv / λ_oracle distribution (under/over on FTTC axis)
    c2 = ax[2]
    for sm in sorted(df.smooth_val.unique()):
        g = df[df.smooth_val == sm]
        ratio = np.log10((g.lam_gcv / g.lam_star).replace([np.inf, -np.inf], np.nan).dropna())
        c2.hist(ratio, bins=20, alpha=0.5, label=f"piv_smooth {sm:g}")
    c2.axvline(0, color="k", ls="--"); c2.set_xlabel("log10(λ_gcv / λ_oracle)")
    c2.set_ylabel("count"); c2.legend()
    c2.set_title("GCV λ vs GT-optimal λ (FTTC)\nleft of 0 = GCV under-regularizes", fontsize=11, fontweight="bold")
    fig.suptitle("L2-path regularization: FTTC+GCV vs Bayesian-L2 vs GT-tuned FTTC λ", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(outdir, f"l2_reg_compare_{cond}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("\nwrote", out)


if __name__ == "__main__":
    main()
