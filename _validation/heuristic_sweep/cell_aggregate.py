#!/usr/bin/env python3
"""Aggregate the diffuse-cell sweep (condition cell_s6j1) into the competence +
heuristic-transfer figure.

Companion to aggregate.py (the dipole grid). The cells are the benchmarkTFM synth
cells (16-82 fitted stress fibres) rewarped onto the best-imaging bead stack at a
ladder of strengths; see make_cells.py. Because the Sabass composite J degenerates
on a diffuse centripetal field (the per-adhesion mean vector cancels), cells are
RANKED ON whole-field nRMSE instead -- recorded by sweep_forces alongside the
whole-field magnitude bias / angular error / background leak.

Four panels:
  A  winner (best nRMSE) per cell x strength     -- method competence is regime-structured
  B  best nRMSE vs strength, per method          -- the U-shaped useful window into breakdown
  C  L1 sensitivity basin (mid strengths)        -- the safe plateau vs the dipole plateau
  D  median optimal L1 / L2 vs strength          -- regularization tracks SNR

Usage:  python cell_aggregate.py [--stage $STAGE] [--outdir figures]
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
COND = "cell_s6j1"
MC = {"PIV": "#1f77b4", "ILK": "#ff7f0e", "FFD": "#2ca02c"}
ORDER = ["PIV", "FFD", "ILK"]
DIPOLE_PLATEAU = (0.07, 0.11)     # from aggregate.py / the dipole report


def load(stage):
    fns = sorted(glob.glob(os.path.join(stage, "results", f"sweep_{COND}_*.csv")))
    frames = []
    for f in fns:
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print("skip empty shard:", os.path.basename(f))
    if not frames:
        raise SystemExit(f"no {COND} shards under {stage}/results")
    df = pd.concat(frames, ignore_index=True)
    df["cell"] = df.scene_id.str.rsplit("_", n=1).str[0]
    df["P"] = df.peak_disp_px
    print(f"{len(frames)} shards, {len(df)} rows, {df.scene_id.nunique()} scenes")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="sweep stage dir (default: $STAGE)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    stage = a.stage or os.environ.get("STAGE")
    if not stage:
        raise SystemExit("set --stage or `source env.sh` to export STAGE")
    os.makedirs(a.outdir, exist_ok=True)
    df = load(stage)
    cells = sorted(df.cell.unique()); Ps = sorted(df.P.unique())
    best = df.loc[df.groupby(["cell", "P", "method"]).nrmse.idxmin()]

    fig = plt.figure(figsize=(15, 10)); gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.24)

    # A: winner map
    axA = fig.add_subplot(gs[0, 0])
    wm = np.zeros((len(cells), len(Ps)), int)
    for i, c in enumerate(cells):
        for j, P in enumerate(Ps):
            sub = best[(best.cell == c) & (best.P == P)]
            wm[i, j] = ORDER.index(sub.loc[sub.nrmse.idxmin(), "method"])
            axA.text(j, i, f"{sub.nrmse.min():.2f}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
    axA.imshow(wm, cmap=ListedColormap([MC[m] for m in ORDER]), aspect="auto", vmin=0, vmax=2)
    axA.set_xticks(range(len(Ps))); axA.set_xticklabels([f"{p:g}" for p in Ps])
    axA.set_xlabel("peak |u| (px)")
    axA.set_yticks(range(len(cells)))
    axA.set_yticklabels([f"{c}\n{int(best[best.cell==c].n_fibers.iloc[0])} fib" for c in cells])
    axA.set_title("A  Winner (best nRMSE) per cell × strength", fontweight="bold", loc="left")
    axA.legend(handles=[Patch(color=MC[m], label=m) for m in ORDER],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.13))

    # B: nRMSE vs strength per method (mean, min-max band over cells)
    axB = fig.add_subplot(gs[0, 1])
    for m in ["PIV", "ILK", "FFD"]:
        bs = best[best.method == m]
        stat = bs.groupby("P").nrmse.agg(["mean", "min", "max"])
        axB.plot(stat.index, stat["mean"], "o-", color=MC[m], label=m, lw=2)
        axB.fill_between(stat.index, stat["min"], stat["max"], color=MC[m], alpha=0.12)
    axB.set_xscale("log"); axB.set_xlabel("peak |u| (px)"); axB.set_ylabel("best nRMSE")
    axB.axvspan(0.4, 0.9, color="gray", alpha=0.12); axB.axvspan(22, 55, color="gray", alpha=0.12)
    axB.text(0.6, axB.get_ylim()[1] * 0.98, "noise\nfloor", fontsize=8, ha="center", va="top", color="gray")
    axB.text(33, axB.get_ylim()[1] * 0.98, "break\ndown", fontsize=8, ha="center", va="top", color="gray")
    axB.set_title("B  Recovery vs strength: U-shaped useful window", fontweight="bold", loc="left")
    axB.legend()

    # C: L1 sensitivity basin over the useful mid-strengths (PIV)
    axC = fig.add_subplot(gs[1, 0])
    mid = df[(df.method == "PIV") & (df.P.between(1.2, 8.0))]
    marg = mid.groupby(["cell", "P", "l1"]).nrmse.min().reset_index()
    tab = marg.groupby("l1").nrmse.mean(); rel = tab / tab.min()
    axC.plot(tab.index, rel, "o-", color=MC["PIV"], lw=2)
    plateau = tab.index[(rel <= 1.02).values]
    axC.axhspan(1.0, 1.02, color="green", alpha=0.10)
    if len(plateau):
        axC.axvspan(plateau.min(), plateau.max(), color="green", alpha=0.10)
        axC.text(np.sqrt(plateau.min() * plateau.max()), 1.055,
                 f"cell safe plateau\n[{plateau.min():.2f}, {plateau.max():.2f}]",
                 fontsize=8, color="green", ha="center")
    axC.axvspan(*DIPOLE_PLATEAU, color="purple", alpha=0.12)
    axC.text(np.sqrt(DIPOLE_PLATEAU[0] * DIPOLE_PLATEAU[1]), rel.max() * 0.9,
             "dipole\nplateau", fontsize=7, color="purple", ha="center")
    axC.set_xscale("log"); axC.set_xticks(list(tab.index))
    axC.set_xticklabels([f"{x:.2f}" for x in tab.index], fontsize=8)
    axC.set_xlabel("L1 sparsity"); axC.set_ylabel("nRMSE / best")
    axC.set_title(f"C  L1 is forgiving: ≤2% cost across the plateau", fontweight="bold", loc="left")

    # D: median optimal L1 / L2 vs strength (PIV)
    axD = fig.add_subplot(gs[1, 1])
    opt = df[df.method == "PIV"].loc[df[df.method == "PIV"].groupby(["cell", "P"]).nrmse.idxmin()]
    o = opt.groupby("P")[["l1", "l2"]].median()
    axD.plot(o.index, o["l1"], "s-", color=MC["PIV"], lw=2, label="median optimal L1")
    axD.set_xscale("log"); axD.set_xlabel("peak |u| (px)")
    axD.set_ylabel("optimal L1", color=MC["PIV"])
    axD2 = axD.twinx()
    axD2.plot(o.index, o["l2"], "^--", color="#d62728", lw=2, label="median optimal L2")
    axD2.set_yscale("symlog", linthresh=0.25); axD2.set_ylabel("optimal L2", color="#d62728")
    axD.axvspan(0.4, 0.9, color="gray", alpha=0.12)
    axD.set_title("D  Regularization tracks SNR: ramps at the noise floor", fontweight="bold", loc="left")
    h1, l1_ = axD.get_legend_handles_labels(); h2, l2_ = axD2.get_legend_handles_labels()
    axD.legend(h1 + h2, l1_ + l2_, loc="upper right", fontsize=8)

    fig.suptitle("Diffuse cell fields (benchmarkTFM synth cells, best imaging): "
                 "method competence + L1 heuristic transfer", fontsize=13, fontweight="bold")
    out = os.path.join(a.outdir, "heuristic-sweep-cells-competence.png")
    fig.savefig(out, dpi=115, bbox_inches="tight"); print("wrote", out)

    # console summary for the report text
    tally = best.loc[best.groupby(["cell", "P"]).nrmse.idxmin()].method.value_counts().to_dict()
    print("winner tally:", tally)
    print("best-on-average L1 (mid):", float(tab.idxmin()), "| plateau:",
          f"[{plateau.min():.3f}, {plateau.max():.3f}]" if len(plateau) else "n/a")


if __name__ == "__main__":
    main()
