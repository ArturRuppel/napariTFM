#!/usr/bin/env python3
"""Aggregate the cross-method heuristic sweep into method-competence surfaces,
per-regime winners, and parameter heuristics.

Reads every results/sweep_*.csv shard (16 imaging conditions x 30 scenes x
{PIV,ILK,FFD} x resolution x l1 x l2), reduces to each method's best-J operating
point per (condition, scene), then aggregates over the 16 imaging conditions to
ask, for every (footprint, displacement) regime:
  * how competent is each method (median best-J),
  * which method wins and by how much (margin to runner-up),
  * how robust that verdict is (fraction of conditions the winner also wins),
  * what parameters the winner uses (modal resolution, median l1/l2).

Writes two figures + prints the tables. "Best-J" is the Sabass composite the
sweep ranks on; lower is better.

Usage:  python aggregate.py [--stage $STAGE] [--outdir DIR]
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

METHODS = ["PIV", "ILK", "FFD"]
MCOLOR = {"PIV": "#1f77b4", "ILK": "#2ca02c", "FFD": "#d62728"}
MNAME = {"PIV": "PIV (x-corr)", "ILK": "iLK (optical flow)", "FFD": "FFD (B-spline)"}


def load(stage):
    files = sorted(glob.glob(os.path.join(stage, "results", "sweep_s*.csv")))
    def _r(f):
        try:
            d = pd.read_csv(f)
            return d if len(d) else None
        except pd.errors.EmptyDataError:
            return None
    df = pd.concat([d for d in map(_r, files) if d is not None], ignore_index=True)
    df["disp_px"] = df["scene_id"].str.extract(r"_u([0-9.]+)$").astype(float)
    df["foot_um"] = df["footprint"].astype(float)
    return df


def best_points(df):
    """Each method's best-J row per (condition, scene, method)."""
    idx = df.groupby(["condition", "scene_id", "method"])["J"].idxmin()
    return df.loc[idx].copy()


def surfaces(best):
    """Pivot to (foot x disp) grids: per-method median best-J, winner, margin,
    win-fraction, and winner parameters."""
    foots = sorted(best.foot_um.unique())
    disps = sorted(best.disp_px.unique())
    shape = (len(foots), len(disps))
    medJ = {m: np.full(shape, np.nan) for m in METHODS}
    winner = np.empty(shape, object)
    margin = np.full(shape, np.nan)      # 2nd-best median - best median (how decisive)
    winfrac = np.full(shape, np.nan)     # fraction of conditions the median-winner also wins
    params = np.empty(shape, object)

    for i, foot in enumerate(foots):
        for j, disp in enumerate(disps):
            cell = best[(best.foot_um == foot) & (best.disp_px == disp)]
            med = {m: cell[cell.method == m]["J"].median() for m in METHODS}
            for m in METHODS:
                medJ[m][i, j] = med[m]
            order = sorted(METHODS, key=lambda m: med[m])
            win = order[0]
            winner[i, j] = win
            margin[i, j] = med[order[1]] - med[order[0]]
            # per-condition winner -> robustness of the verdict
            per_cond_win = cell.loc[cell.groupby("condition")["J"].idxmin(), "method"]
            winfrac[i, j] = (per_cond_win == win).mean()
            # winner's parameters (its own best-J rows across conditions)
            w = cell[cell.method == win]
            params[i, j] = dict(res=w["res_val"].round(1).mode().iloc[0],
                                l1=w["l1"].median(), l2=w["l2"].median())
    return dict(foots=foots, disps=disps, medJ=medJ, winner=winner,
                margin=margin, winfrac=winfrac, params=params)


def fig_competence(S, out):
    foots, disps = S["foots"], S["disps"]
    fl = [f"{f:g}" for f in foots]; dl = [f"{d:g}" for d in disps]
    allJ = np.concatenate([S["medJ"][m].ravel() for m in METHODS])
    vmin, vmax = np.nanmin(allJ), np.nanmax(allJ)
    norm = LogNorm(vmin=max(vmin, 0.03), vmax=vmax)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    for k, m in enumerate(METHODS):
        ax = axes[k]
        im = ax.imshow(S["medJ"][m], cmap="viridis_r", norm=norm, aspect="auto", origin="lower")
        ax.set_title(f"{MNAME[m]}\nmedian best-J", fontsize=12, fontweight="bold")
        for i in range(len(foots)):
            for j in range(len(disps)):
                ax.text(j, i, f"{S['medJ'][m][i,j]:.2f}", ha="center", va="center",
                        color="white" if S['medJ'][m][i,j] > 0.6 else "black", fontsize=8)
        ax.set_xticks(range(len(disps))); ax.set_xticklabels(dl, rotation=45)
        ax.set_yticks(range(len(foots))); ax.set_yticklabels(fl)
        ax.set_xlabel("peak displacement (px)")
        if k == 0: ax.set_ylabel("footprint (µm)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="J (lower=better)")

    # winner map
    ax = axes[3]
    widx = np.vectorize(lambda w: METHODS.index(w))(S["winner"]).astype(float)
    cmap = ListedColormap([MCOLOR[m] for m in METHODS])
    ax.imshow(widx, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N),
              aspect="auto", origin="lower")
    for i in range(len(foots)):
        for j in range(len(disps)):
            w = S["winner"][i, j]; mg = S["margin"][i, j]; wf = S["winfrac"][i, j]
            ax.text(j, i, f"{w}\nΔ{mg:.02f}\n{wf*100:.0f}%", ha="center", va="center",
                    color="white", fontsize=7.5, fontweight="bold")
    ax.set_title("winner per regime\n(method / margin to 2nd / win-frac)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(disps))); ax.set_xticklabels(dl, rotation=45)
    ax.set_yticks(range(len(foots))); ax.set_yticklabels(fl)
    ax.set_xlabel("peak displacement (px)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MCOLOR[m], label=MNAME[m]) for m in METHODS],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    fig.suptitle("Method competence & regime winners  (median over 16 imaging conditions)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


def fig_params(S, out):
    """The heuristic table: winner + its parameters per regime, plus per-method
    winning-resolution and l1 tendencies."""
    foots, disps = S["foots"], S["disps"]
    fl = [f"{f:g}" for f in foots]; dl = [f"{d:g}" for d in disps]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # panel 1: chosen operating point (method + res + l1) as text over winner colour
    ax = axes[0]
    widx = np.vectorize(lambda w: METHODS.index(w))(S["winner"]).astype(float)
    cmap = ListedColormap([MCOLOR[m] for m in METHODS])
    ax.imshow(widx, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N),
              aspect="auto", origin="lower", alpha=0.35)
    for i in range(len(foots)):
        for j in range(len(disps)):
            w = S["winner"][i, j]; p = S["params"][i, j]
            ax.text(j, i, f"{w}\nres {p['res']:g}\nl1 {p['l1']:.02f}", ha="center", va="center",
                    color="black", fontsize=7.5, fontweight="bold")
    ax.set_title("recommended operating point\n(winner + resolution + L1)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(disps))); ax.set_xticklabels(dl, rotation=45)
    ax.set_yticks(range(len(foots))); ax.set_yticklabels(fl)
    ax.set_xlabel("peak displacement (px)"); ax.set_ylabel("footprint (µm)")

    # panel 2: winning L1 (median) as heatmap over the winner
    ax = axes[1]
    l1grid = np.array([[S["params"][i][j]["l1"] for j in range(len(disps))]
                       for i in range(len(foots))])
    im = ax.imshow(l1grid, cmap="magma", aspect="auto", origin="lower")
    for i in range(len(foots)):
        for j in range(len(disps)):
            ax.text(j, i, f"{l1grid[i,j]:.02f}", ha="center", va="center",
                    color="white" if l1grid[i, j] < l1grid.max() * 0.6 else "black", fontsize=8)
    ax.set_title("winner's L1 sparsity\n(regularization strength)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(disps))); ax.set_xticklabels(dl, rotation=45)
    ax.set_yticks(range(len(foots))); ax.set_yticklabels(fl)
    ax.set_xlabel("peak displacement (px)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="L1")

    # panel 3: winning L2 (median)
    ax = axes[2]
    l2grid = np.array([[S["params"][i][j]["l2"] for j in range(len(disps))]
                       for i in range(len(foots))])
    im = ax.imshow(l2grid, cmap="cividis", aspect="auto", origin="lower")
    for i in range(len(foots)):
        for j in range(len(disps)):
            ax.text(j, i, f"{l2grid[i,j]:g}", ha="center", va="center",
                    color="white" if l2grid[i, j] < l2grid.max() * 0.5 else "black", fontsize=8)
    ax.set_title("winner's L2 ridge", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(disps))); ax.set_xticklabels(dl, rotation=45)
    ax.set_yticks(range(len(foots))); ax.set_yticklabels(fl)
    ax.set_xlabel("peak displacement (px)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="L2")

    fig.suptitle("Parameter heuristics for the winning method", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


def print_tables(df, best, S):
    print(f"\nloaded {len(df):,} rows | {df.condition.nunique()} conditions | "
          f"{best.shape[0]} best-points")
    # overall method robustness: mean rank + within-10%-of-best frequency
    print("\n=== overall method standing (across all 480 scenes) ===")
    piv = best.pivot_table(index=["condition", "scene_id"], columns="method", values="J")
    ranks = piv.rank(axis=1)
    within = piv.div(piv.min(axis=1), axis=0)  # J / best-of-scene
    for m in METHODS:
        print(f"  {MNAME[m]:20s}  wins {int((ranks[m]==1).sum()):3d}/480  "
              f"mean-rank {ranks[m].mean():.2f}  median J/best {within[m].median():.2f}  "
              f"within-10%-of-best {int((within[m]<=1.1).mean()*100):3d}%")
    # winner tally
    print("\n=== regime-winner tally (30 cells) ===")
    vals, cnts = np.unique(S["winner"].ravel(), return_counts=True)
    for v, c in sorted(zip(vals, cnts), key=lambda x: -x[1]):
        print(f"  {v}: {c} cells")


HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_stage(arg):
    """Stage dir from --stage or $STAGE (see env.sh). No path is hardcoded: the
    real data lives in a private location the public repo must not name."""
    stage = arg or os.environ.get("STAGE")
    if not stage:
        raise SystemExit("set --stage or `source env.sh` to export STAGE")
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="sweep stage dir (default: $STAGE)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"),
                    help="where to write figures (use ../../docs/images to refresh the report)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = load(resolve_stage(a.stage))
    best = best_points(df)
    S = surfaces(best)
    print_tables(df, best, S)
    fig_competence(S, os.path.join(a.outdir, "heuristic-sweep-competence.png"))
    fig_params(S, os.path.join(a.outdir, "heuristic-sweep-parameters.png"))


if __name__ == "__main__":
    main()
