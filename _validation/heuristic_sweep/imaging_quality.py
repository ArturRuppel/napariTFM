#!/usr/bin/env python3
"""How do imaging parameters (density / NA / exposure / jitter) impact TFM
recovery quality, and how does the method picture change on the best images?

Attaches the scenario manifest (density, NA, exposure, jitter, ncc-vs-frame0) to
every swept row, then asks:
  * which imaging parameter drives achievable J (marginal Spearman),
  * does quality lower the J floor, or extend the recoverable envelope,
  * does the best method change between high- and low-exposure images.
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

HERE = os.path.dirname(os.path.abspath(__file__))
S = None    # sweep stage dir; set in main() from --stage or $STAGE (see env.sh)
METHODS = ["PIV", "ILK", "FFD"]
MCOLOR = {"PIV": "#1f77b4", "ILK": "#2ca02c", "FFD": "#d62728"}
EXPO_COLOR = {4.0: "#2ca02c", 1.0: "#ff7f0e", 0.25: "#d62728"}


def load():
    man = pd.read_csv(os.path.join(S, "images", "tif_stacks", "manifest.csv"))
    man = man[man.frame.isin([1, 3])].copy()
    man["condition"] = "s" + man.scenario.astype(str) + "_j" + man.frame.astype(str)
    meta = man.set_index("condition")[["density", "NA", "expo", "jitter_px", "ncc_vs_frame0"]]

    def _r(f):
        try:
            d = pd.read_csv(f); return d if len(d) else None
        except pd.errors.EmptyDataError:
            return None
    df = pd.concat([d for d in map(_r, sorted(glob.glob(os.path.join(S, "results", "sweep_s*.csv")))) if d is not None],
                   ignore_index=True)
    df["disp_px"] = df["scene_id"].str.extract(r"_u([0-9.]+)$").astype(float)
    df["foot_um"] = df["footprint"].astype(float)
    df = df.join(meta, on="condition")
    return df, meta


def fig_drivers(df, meta, out):
    best = df.loc[df.groupby(["condition", "scene_id", "method"])["J"].idxmin()]
    scene_best = best.loc[best.groupby(["condition", "scene_id"])["J"].idxmin()]
    res = scene_best[(scene_best.foot_um >= 0.707) & (scene_best.disp_px >= 3.155) &
                     (scene_best.disp_px <= 19.905)]
    tab = res.groupby("condition")["J"].median().to_frame("medJ").join(meta).sort_values("medJ")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) condition ranking, colored by exposure
    a = ax[0]
    y = np.arange(len(tab))
    a.barh(y, tab["medJ"], color=[EXPO_COLOR[e] for e in tab["expo"]])
    a.set_yticks(y)
    a.set_yticklabels([f"{c}  (NA{r.NA:g} d{r.density:g})" for c, r in tab.iterrows()], fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("median best-J, resolvable regime (lower=better)")
    a.set_title("achievable quality per imaging condition\n(color = exposure)", fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=EXPO_COLOR[e], label=f"expo {e:g}") for e in [4.0, 1.0, 0.25]],
             fontsize=9, loc="lower right")

    # (b) master quality curve: J vs ncc
    b = ax[1]
    for e in [4.0, 1.0, 0.25]:
        sub = tab[tab.expo == e]
        b.scatter(sub["ncc_vs_frame0"], sub["medJ"], s=90, color=EXPO_COLOR[e],
                  edgecolor="k", label=f"expo {e:g}", zorder=3)
    b.set_xlabel("bead-image quality  (NCC deformed-vs-reference)")
    b.set_ylabel("median best-J (resolvable)")
    b.set_title("recovery error vs image quality", fontsize=12, fontweight="bold")
    b.legend(fontsize=9); b.grid(alpha=0.3)

    # (c) marginal driver strength
    c = ax[2]
    params = ["expo", "ncc_vs_frame0", "jitter_px", "density", "NA"]
    rhos = [tab["medJ"].corr(tab[p], method="spearman") for p in params]
    labels = ["exposure", "image NCC", "jitter", "density", "NA"]
    order = np.argsort(np.abs(rhos))
    c.barh([labels[i] for i in order], [abs(rhos[i]) for i in order],
           color=["#333" if abs(rhos[i]) > 0.4 else "#bbb" for i in order])
    for k, i in enumerate(order):
        c.text(abs(rhos[i]) + 0.01, k, f"{rhos[i]:+.2f}", va="center", fontsize=9)
    c.set_xlim(0, 0.8)
    c.set_xlabel("|Spearman ρ|  (driver strength on J)")
    c.set_title("what drives recovery quality", fontsize=12, fontweight="bold")

    fig.suptitle("Imaging parameters vs TFM recovery quality  (resolvable regime)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)


def fig_envelope(df, out):
    """foot x disp median-J maps for high vs low exposure, + winner shift."""
    best = df.loc[df.groupby(["condition", "scene_id", "method"])["J"].idxmin()]
    scene_best = best.loc[best.groupby(["condition", "scene_id"])["J"].idxmin()]
    foots = sorted(df.foot_um.unique()); disps = sorted(df.disp_px.unique())
    fl = [f"{f:g}" for f in foots]; dl = [f"{d:g}" for d in disps]

    def grid(sub, valfn):
        g = np.full((len(foots), len(disps)), np.nan)
        for i, f in enumerate(foots):
            for j, d in enumerate(disps):
                cell = sub[(sub.foot_um == f) & (sub.disp_px == d)]
                if len(cell):
                    g[i, j] = valfn(cell)
        return g

    hi = scene_best[scene_best.expo == 4.0]     # best images
    lo = scene_best[scene_best.expo == 0.25]    # worst images
    ghi = grid(hi, lambda c: c["J"].median())
    glo = grid(lo, lambda c: c["J"].median())
    norm = LogNorm(vmin=0.05, vmax=np.nanmax([ghi, glo]))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for k, (g, ttl) in enumerate([(ghi, "HIGH exposure (4.0)\nbest images"),
                                  (glo, "LOW exposure (0.25)\nworst images")]):
        im = ax[k].imshow(g, cmap="viridis_r", norm=norm, aspect="auto", origin="lower")
        for i in range(len(foots)):
            for j in range(len(disps)):
                ax[k].text(j, i, f"{g[i,j]:.2f}", ha="center", va="center",
                           color="white" if g[i, j] > 0.6 else "black", fontsize=8)
        ax[k].set_title(ttl, fontsize=12, fontweight="bold")
        ax[k].set_xticks(range(len(disps))); ax[k].set_xticklabels(dl, rotation=45)
        ax[k].set_yticks(range(len(foots))); ax[k].set_yticklabels(fl)
        ax[k].set_xlabel("peak displacement (px)")
        if k == 0: ax[k].set_ylabel("footprint (µm)")
        plt.colorbar(im, ax=ax[k], fraction=0.046, label="median best-J")

    # difference: how much low exposure costs, per regime
    d = glo - ghi
    im = ax[2].imshow(d, cmap="magma", aspect="auto", origin="lower", vmin=0)
    for i in range(len(foots)):
        for j in range(len(disps)):
            ax[2].text(j, i, f"+{d[i,j]:.2f}", ha="center", va="center",
                       color="white" if d[i, j] < np.nanmax(d) * 0.6 else "black", fontsize=8)
    ax[2].set_title("J penalty of low exposure\n(low − high)", fontsize=12, fontweight="bold")
    ax[2].set_xticks(range(len(disps))); ax[2].set_xticklabels(dl, rotation=45)
    ax[2].set_yticks(range(len(foots))); ax[2].set_yticklabels(fl)
    ax[2].set_xlabel("peak displacement (px)")
    plt.colorbar(im, ax=ax[2], fraction=0.046, label="ΔJ")

    fig.suptitle("Where imaging quality matters: the recoverable envelope, high vs low exposure",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)

    # winner shift, printed
    print("\n=== winner tally by exposure (all 30 regimes x conditions) ===")
    for e, name in [(4.0, "HIGH expo"), (1.0, "MID expo"), (0.25, "LOW expo")]:
        sub = best[best.expo == e]
        w = sub.loc[sub.groupby(["condition", "scene_id"])["J"].idxmin(), "method"]
        vc = w.value_counts()
        n = vc.sum()
        print(f"  {name:9s}: " + "  ".join(f"{m} {vc.get(m,0)*100//n:2d}%" for m in METHODS))


def main():
    global S
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="sweep stage dir (default: $STAGE)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"),
                    help="where to write figures (use ../../docs/images to refresh the report)")
    a = ap.parse_args()
    S = a.stage or os.environ.get("STAGE")
    if not S:
        raise SystemExit("set --stage or `source env.sh` to export STAGE")
    os.makedirs(a.outdir, exist_ok=True)
    df, meta = load()
    fig_drivers(df, meta, os.path.join(a.outdir, "heuristic-sweep-imaging-drivers.png"))
    fig_envelope(df, os.path.join(a.outdir, "heuristic-sweep-imaging-envelope.png"))


if __name__ == "__main__":
    main()
