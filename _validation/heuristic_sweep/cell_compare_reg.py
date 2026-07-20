#!/usr/bin/env python3
"""Does smoothness close the gap on DIFFUSE fields? The dipole run found L1
(sparsity) beats parameter-free Bayesian-L2 (smoothness) by ~2.5-3x, because a
compact source is L1's home turf. A whole cell is the case where smoothness might
win. This re-runs that comparison on the diffuse-cell scenes.

Companion to compare_reg.py (dipoles). Holds the displacement input fixed (PIV
window 24) so only the regularization strategy varies. For every cell scene:
  * pull nRMSE(L1) at L2=0 and nRMSE(L2) at L1=0 straight from the sweep shard,
  * run parameter-free Bayesian-L2 (ABL2) on the same field.
Cells are scored on whole-field nRMSE (the Sabass J is undefined on a diffuse
centripetal field). GT is the stored fitted-fibre traction.

Usage:  python cell_compare_reg.py [--stage $STAGE] [--outdir figures]
"""
from __future__ import annotations
import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import sweep_config as C
from sweep_forces import metrics
from napariTFM.backend.bayesian_l2 import reconstruct_bl2_frame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", default=None, help="sweep stage dir (default: $STAGE)")
_ap.add_argument("--outdir", default=os.path.join(_HERE, "figures"),
                 help="where to write figure + CSV (use ../../docs/images to refresh the report)")
_a = _ap.parse_args()
S = _a.stage or os.environ.get("STAGE")
if not S:
    raise SystemExit("set --stage or `source env.sh` to export STAGE")
OUT = _a.outdir
os.makedirs(OUT, exist_ok=True)
COND = "cell_s6j1"
N = C.GT_REFERENCE_SIZE
L1S = [round(float(x), 4) for x in C.FRAC1]
RES = 24.0   # PIV window held fixed
# useful window: mid strengths, away from the noise floor and gross decorrelation
USEFUL = (1.2, 8.0)


def scene_list():
    root = os.path.join(S, "scenes", COND)
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def l1_profile_from_csv(scene):
    """nRMSE vs L1 at L2=0, and vs L2 at L1=floor, PIV res=24, from the shard."""
    prof, l2_prof = {}, {}
    with open(os.path.join(S, "results", f"sweep_{COND}_{scene}.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["method"] != "PIV" or abs(float(r["res_val"]) - RES) > 1e-6:
                continue
            l1, l2, v = round(float(r["l1"]), 4), float(r["l2"]), float(r["nrmse"])
            if l2 == 0.0:
                prof[l1] = v
            if round(float(r["l1"]), 4) == L1S[0]:
                l2_prof[l2] = v
    return prof, l2_prof


def bl2_nrmse(scene):
    """Parameter-free Bayesian-L2 on the same PIV-win24 field -> whole-field nRMSE."""
    field = np.load(os.path.join(S, "cache", COND, scene, "disp_PIV_res3.npz"))["field"]  # (h,w,2) µm
    h = field.shape[0]
    eff_ps = C.PIXEL_SIZE_UM * (N / h)
    t = reconstruct_bl2_frame(field, C.YOUNG_MODULUS, C.POISSON, eff_ps, mask=None)  # (2,h,w)
    t_up = zoom(t, (1, N / h, N / h), order=1)
    gt = np.load(os.path.join(S, "scenes", COND, scene, "gt_traction.npy")).astype(np.float32)
    return metrics(t_up, gt, do_sabass=False)["nrmse"]


rows = []
for scene in scene_list():
    disp = float(scene.split("_u")[1]); cell = scene.rsplit("_", 1)[0]
    l1p, l2p = l1_profile_from_csv(scene)
    if len(l1p) < len(L1S):
        print("  incomplete L1 profile, skip", scene); continue
    vbl2 = bl2_nrmse(scene)
    vl1 = {l1: l1p[l1] for l1 in L1S}
    l1_opt = min(vl1, key=vl1.get)
    rows.append(dict(cell=cell, scene=scene, disp=disp,
                     l1_opt=l1_opt, nrmse_l1_opt=vl1[l1_opt],
                     nrmse_l2_opt=min(l2p.values()) if l2p else np.nan,
                     nrmse_bl2=vbl2, **{f"nrmse_l1_{l1}": vl1[l1] for l1 in L1S}))
    print(f"  {scene}: L1*={l1_opt} nRMSE_L1={vl1[l1_opt]:.3f}  BL2={vbl2:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "cell_reg_compare.csv"), index=False)

res = df[df.disp.between(*USEFUL)].copy()
print(f"\n{len(df)} scenes total, {len(res)} in useful window {USEFUL}")

norm = np.array([[res[f"nrmse_l1_{l1}"].values[k] / res["nrmse_l1_opt"].values[k] for l1 in L1S]
                 for k in range(len(res))])
med_basin = np.median(norm, axis=0)
fixed_pen = {l1: np.median(res[f"nrmse_l1_{l1}"] / res["nrmse_l1_opt"]) for l1 in L1S}
l1_fixed = min(fixed_pen, key=fixed_pen.get)
print("\n=== L1 sensitivity (useful window) ===")
print(f"  best single fixed L1 = {l1_fixed} (median penalty {fixed_pen[l1_fixed]:.3f}x)")
print(f"  optimal L1 range: {res.l1_opt.min()}..{res.l1_opt.max()} (median {res.l1_opt.median()})")

print("\n=== Bayesian-L2 vs L1 (useful window, median nRMSE) ===")
print(f"  oracle per-scene L1 : {res.nrmse_l1_opt.median():.3f}")
print(f"  best fixed L1 ({l1_fixed}) : {res[f'nrmse_l1_{l1_fixed}'].median():.3f}")
print(f"  oracle grid L2      : {res.nrmse_l2_opt.median():.3f}")
print(f"  Bayesian-L2 (no tune): {res.nrmse_bl2.median():.3f}")
print(f"  --> BL2 / oracle-L1 = {(res.nrmse_bl2/res.nrmse_l1_opt).median():.2f}x")
print(f"  --> BL2 / fixed-L1  = {(res.nrmse_bl2/res[f'nrmse_l1_{l1_fixed}']).median():.2f}x")
print(f"  --> BL2 beats fixed-L1 in {int((res.nrmse_bl2 < res[f'nrmse_l1_{l1_fixed}']).mean()*100)}% of scenes")

# ================= FIGURE =================
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
a = ax[0]
for k in range(len(norm)):
    a.plot(L1S, norm[k], color="0.8", lw=0.6, alpha=0.5, zorder=1)
a.plot(L1S, med_basin, color="#d62728", lw=3, marker="o", label="median across cells", zorder=3)
a.axhline(1.05, ls=":", color="k", alpha=0.6); a.text(L1S[0], 1.055, " +5% band", fontsize=8, va="bottom")
a.axvline(l1_fixed, ls="--", color="#1f77b4", label=f"best fixed L1 = {l1_fixed}")
bl2_pen = (res.nrmse_bl2 / res.nrmse_l1_opt).median()
a.axhline(bl2_pen, ls="-", color="#d62728", alpha=0.4, lw=2)
a.text(L1S[-1], bl2_pen, f" Bayesian-L2 = {bl2_pen:.2f}x", fontsize=9, va="bottom", ha="right", color="#d62728")
a.set_xscale("log"); a.set_xlabel("L1 sparsity"); a.set_ylabel("nRMSE / per-scene best  (1.0 = optimal)")
a.set_title("What a wrong regularizer costs on cells\n(useful window, displacement fixed at PIV win-24)",
            fontsize=12, fontweight="bold")
a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3, which="both")

b = ax[1]
strategies = [("oracle\nper-scene L1", res.nrmse_l1_opt, "#2ca02c"),
              (f"fixed L1\n={l1_fixed}", res[f"nrmse_l1_{l1_fixed}"], "#1f77b4"),
              ("oracle\ngrid L2", res.nrmse_l2_opt, "#7f7f7f"),
              ("Bayesian-L2\n(no tuning)", res.nrmse_bl2, "#d62728")]
data = [s[1].values for s in strategies]
bp = b.boxplot(data, tick_labels=[s[0] for s in strategies], showfliers=False, patch_artist=True)
for patch, s in zip(bp["boxes"], strategies):
    patch.set_facecolor(s[2]); patch.set_alpha(0.6)
for i, s in enumerate(strategies, 1):
    b.text(i, s[1].median(), f"{s[1].median():.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
b.set_ylabel("whole-field nRMSE  (lower=better)")
b.set_title(f"Recovery error by strategy\n(useful window, {len(res)} cell scenes)", fontsize=12, fontweight="bold")
b.grid(alpha=0.3, axis="y")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT, "heuristic-sweep-cells-regularization.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)
