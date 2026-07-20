#!/usr/bin/env python3
"""How sensitive is the L1 sparsity choice, and how much does a wrong L1 cost
versus parameter-free Bayesian-L2?

Holds the displacement input fixed (PIV window 24, the recommended default) so
only the regularization strategy varies. For every scene:
  * pull J(L1) at L2=0 and J(L2) at L1=0 for that field straight from the sweep,
  * run parameter-free Bayesian-L2 (ABL2) on the same field.
Then quantify: the L1 basin shape, the penalty of a fixed/mistuned L1, and where
BL2 lands with no tuning at all.
"""
from __future__ import annotations
import os, sys, glob, csv, tomllib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import sweep_config as C
import sabass
from sweep_forces import rasterize_gt, metrics
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
CONDS = [f"s{s}_j{j}" for s in range(8) for j in (1, 3)]
N = C.GT_REFERENCE_SIZE
L1S = [round(float(x), 4) for x in C.FRAC1]
RES = 24.0   # PIV window held fixed


def scene_list(cond):
    root = os.path.join(S, "scenes", cond)
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def l1_profile_from_csv(cond, scene):
    """J vs L1 at L2=0, PIV res=24, from the sweep shard. Returns {l1: J}."""
    prof = {}
    l2_prof = {}
    with open(os.path.join(S, "results", f"sweep_{cond}_{scene}.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["method"] != "PIV" or abs(float(r["res_val"]) - RES) > 1e-6:
                continue
            l1, l2, J = round(float(r["l1"]), 4), float(r["l2"]), float(r["J"])
            if l2 == 0.0:
                prof[l1] = J
            if round(float(r["l1"]), 4) == L1S[0]:   # L1 at its floor -> ~pure L2 sweep
                l2_prof[l2] = J
    return prof, l2_prof


def bl2_J(cond, scene):
    """Parameter-free Bayesian-L2 on the same PIV-win24 field -> Sabass J."""
    cf = os.path.join(S, "cache", cond, scene, "disp_PIV_res3.npz")
    field = np.load(cf)["field"]                      # (h,w,2) µm
    h = field.shape[0]
    eff_ps = C.PIXEL_SIZE_UM * (N / h)
    t = reconstruct_bl2_frame(field, C.YOUNG_MODULUS, C.POISSON, eff_ps, mask=None)  # (2,h,w)
    t_up = zoom(t, (1, N / h, N / h), order=1)
    with open(os.path.join(S, "scenes", cond, scene, "scene.toml"), "rb") as fh:
        sc = tomllib.load(fh)
    gt, _ = rasterize_gt(sc, N)
    return metrics(t_up, gt)["J"]


rows = []
for cond in CONDS:
    for scene in scene_list(cond):
        foot = float(scene.split("_")[0][1:])
        disp = float(scene.split("_u")[1])
        l1p, l2p = l1_profile_from_csv(cond, scene)
        if len(l1p) < len(L1S):
            continue
        Jbl2 = bl2_J(cond, scene)
        Jl1 = {l1: l1p[l1] for l1 in L1S}
        l1_opt = min(Jl1, key=Jl1.get)
        rows.append(dict(cond=cond, scene=scene, foot=foot, disp=disp,
                         l1_opt=l1_opt, J_l1_opt=Jl1[l1_opt],
                         J_l2_opt=min(l2p.values()) if l2p else np.nan,
                         J_bl2=Jbl2, **{f"J_l1_{l1}": Jl1[l1] for l1 in L1S}))
    print(f"  {cond} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "reg_compare.csv"), index=False)

# resolvable regime: where the choice actually matters
res = df[(df.foot >= 0.707) & (df.disp >= 3.155) & (df.disp <= 19.905)].copy()
print(f"\n{len(df)} scenes total, {len(res)} in resolvable regime")

# ---- L1 sensitivity ----
# normalized basin J(l1)/J_opt per scene
norm = np.array([[res[f"J_l1_{l1}"].values[k] / res["J_l1_opt"].values[k] for l1 in L1S]
                 for k in range(len(res))])
med_basin = np.median(norm, axis=0)
# best single fixed L1: minimizes median penalty across resolvable scenes
fixed_pen = {l1: np.median(res[f"J_l1_{l1}"] / res["J_l1_opt"]) for l1 in L1S}
l1_fixed = min(fixed_pen, key=fixed_pen.get)
print("\n=== L1 sensitivity (resolvable) ===")
print(f"  best single fixed L1 = {l1_fixed}  (median penalty {fixed_pen[l1_fixed]:.3f}x vs per-scene oracle)")
print(f"  optimal L1 range across scenes: {res.l1_opt.min()}..{res.l1_opt.max()} (median {res.l1_opt.median()})")
# penalty of a 2x / 4x error from the fixed point
idx_fixed = L1S.index(l1_fixed)
for step, lbl in [(1, "±1 grid step (~1.5x)"), (2, "±2 steps (~2.4x)")]:
    up = L1S[min(idx_fixed + step, len(L1S) - 1)]
    dn = L1S[max(idx_fixed - step, 0)]
    pu = np.median(res[f"J_l1_{up}"] / res["J_l1_opt"])
    pd_ = np.median(res[f"J_l1_{dn}"] / res["J_l1_opt"])
    print(f"  {lbl}: up->{pu:.3f}x  down->{pd_:.3f}x")

# ---- vs Bayesian-L2 ----
print("\n=== Bayesian-L2 vs L1 (resolvable, median J) ===")
print(f"  oracle per-scene L1 : {res.J_l1_opt.median():.3f}")
print(f"  best fixed L1 ({l1_fixed}) : {res[f'J_l1_{l1_fixed}'].median():.3f}")
print(f"  oracle grid L2      : {res.J_l2_opt.median():.3f}")
print(f"  Bayesian-L2 (no tuning): {res.J_bl2.median():.3f}")
print(f"  --> BL2 / oracle-L1 = {(res.J_bl2/res.J_l1_opt).median():.2f}x")
print(f"  --> BL2 / fixed-L1  = {(res.J_bl2/res[f'J_l1_{l1_fixed}']).median():.2f}x")
print(f"  --> BL2 beats fixed-L1 in {int((res.J_bl2 < res[f'J_l1_{l1_fixed}']).mean()*100)}% of scenes")

# ================= FIGURES =================
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

# (a) L1 basin, with the Bayesian-L2 penalty drawn in for scale
a = ax[0]
for k in range(len(norm)):
    a.plot(L1S, norm[k], color="0.8", lw=0.6, alpha=0.5, zorder=1)
a.plot(L1S, med_basin, color="#d62728", lw=3, marker="o", label="median across scenes", zorder=3)
a.axhline(1.1, ls=":", color="k", alpha=0.6)
a.text(L1S[0], 1.115, " +10% band", fontsize=8, va="bottom")
a.axvline(l1_fixed, ls="--", color="#1f77b4", label=f"best fixed L1 = {l1_fixed}")
bl2_pen = (res.J_bl2 / res.J_l1_opt).median()
a.axhline(bl2_pen, ls="-", color="#d62728", alpha=0.4, lw=2)
a.text(L1S[-1], bl2_pen, f" Bayesian-L2 = {bl2_pen:.1f}x", fontsize=9, va="bottom", ha="right", color="#d62728")
a.set_xscale("log"); a.set_ylim(0.95, 3.6)
a.set_xlabel("L1 sparsity"); a.set_ylabel("J / per-scene best  (1.0 = optimal)")
a.set_title("How much a wrong L1 costs\n(resolvable regime, displacement fixed at PIV win-24)",
            fontsize=12, fontweight="bold")
a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3, which="both")

# (b) summary ladder: J distribution per strategy
b = ax[1]
strategies = [("oracle\nper-scene L1", res.J_l1_opt, "#2ca02c"),
              (f"fixed L1\n={l1_fixed}", res[f"J_l1_{l1_fixed}"], "#1f77b4"),
              ("oracle\ngrid L2", res.J_l2_opt, "#7f7f7f"),
              ("Bayesian-L2\n(no tuning)", res.J_bl2, "#d62728")]
data = [s[1].values for s in strategies]
bp = b.boxplot(data, tick_labels=[s[0] for s in strategies], showfliers=False, patch_artist=True)
for patch, s in zip(bp["boxes"], strategies):
    patch.set_facecolor(s[2]); patch.set_alpha(0.6)
for i, s in enumerate(strategies, 1):
    b.text(i, s[1].median(), f"{s[1].median():.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
b.set_ylabel("Sabass J  (lower=better)")
b.set_title("Recovery error by regularization strategy\n(resolvable regime, 144 scenes)",
            fontsize=12, fontweight="bold")
b.grid(alpha=0.3, axis="y")

fig.suptitle("L1 is forgiving if you err low; over-regularizing costs as much as dropping to parameter-free Bayesian-L2",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(OUT, "heuristic-sweep-regularization-sensitivity.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)
