#!/usr/bin/env python3
"""Consolidated analysis of the TFM heuristic sweep.

Reads the two artifacts the pipeline produces and nothing else:

  * `$STAGE/renders/index.csv`  -- one row per (condition, scene, displacement
    input): the achievable force ceiling of each regularizer path on that input
    (`fttc_obj` / `l1_obj`) and the regularization each path chose.
  * `$STAGE/cache/<cond>/<scene>/force_*.npz` -- the same rows' full objective
    CURVES (`l1_obj_curve` over FRAC1, `fttc_obj_curve` over LAMBDA_GRID), read
    lazily one zip member at a time so the whole pass costs seconds.

Both are GT-tuned: every number here is a CEILING -- the best that regularizer
could do on that displacement field -- so the comparisons are about what the
*method* can reach, never about who guessed a parameter luckily.

Objective is the Sabass composite J for isolated dipoles and whole-field nRMSE
for diffuse cells (a centripetal field's per-adhesion terms degenerate); the
`objective` column carries which. Lower is better in both.

Usage:  python analyze.py [--stage $STAGE] [--outdir DIR] [--figs 1,2,5]
Omit --outdir and figures land in ./figures (gitignored) instead of docs/images.
"""
from __future__ import annotations
import argparse, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm, TwoSlopeNorm

import sweep_config as C
from make_stacks import SCENARIOS, JITTERS

# --- house style ------------------------------------------------------------ #
# Categorical hues are Okabe-Ito (colour-vision-deficiency safe; the tab10
# green/red the first report used is the classic confusable pair). Fixed order,
# never cycled, and every method is also direct-labelled so identity is never
# carried by colour alone.
METHODS = ["PIV", "ILK", "FFD"]
MCOLOR = {"PIV": "#0072B2", "ILK": "#E69F00", "FFD": "#009E73"}
MNAME = {"PIV": "PIV (x-corr)", "ILK": "iLK (optical flow)", "FFD": "FFD (B-spline)"}
PATHS = ["FISTA+L1", "FTTC+L2"]
PCOLOR = {"FISTA+L1": "#0072B2", "FTTC+L2": "#D55E00"}
SEQ = "viridis_r"          # magnitude: perceptually uniform; bright = low objective = good
DIV = "RdBu_r"             # polarity: two poles, neutral midpoint
PLATEAU_TOL = 1.05         # "safe" = within 5% of the best achievable median
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_index(stage):
    """renders/index.csv + the scene/condition axes parsed out of the names."""
    df = pd.read_csv(os.path.join(stage, "renders", "index.csv"))
    df["best_obj"] = df[["fttc_obj", "l1_obj"]].min(axis=1)
    df["is_cell"] = df["kind"].eq("cell")

    # dipole scene "f0.266_u1.256" -> footprint; cell scene "synth00_u3.155" -> cell id
    df["foot_um"] = df["scene"].str.extract(r"^f([0-9.]+)_u").astype(float)
    df["cell_id"] = df["scene"].str.extract(r"^(synth\d+)_u")
    # condition "s3_j1" -> scenario 3, jitter frame 1; cells are all "cell_s6j1"
    sj = df["cond"].str.extract(r"^s(\d+)_j(\d+)$")
    df["scenario"] = pd.to_numeric(sj[0], errors="coerce")
    df["jframe"] = pd.to_numeric(sj[1], errors="coerce")
    df.loc[df["is_cell"], ["scenario", "jframe"]] = [C.CELL_STACK_SCENARIO, C.CELL_DEFORM_FRAME]
    scen = pd.DataFrame(SCENARIOS, columns=["n_beads", "NA", "expo"])
    scen["density"] = scen["n_beads"] / (C.GT_REFERENCE_SIZE ** 2)
    df = df.join(scen, on="scenario")
    df["jitter"] = df["jframe"].map(lambda f: JITTERS[int(f)] if np.isfinite(f) else np.nan)
    # J blows up (>1e3) when a solve degenerates; keep it out of medians as NaN, and
    # say how often that happened rather than silently dropping it.
    bad = ~np.isfinite(df["best_obj"]) | (df["best_obj"] > 1e3)
    if bad.any():
        print(f"note: {bad.sum()}/{len(df)} rows ({100 * bad.mean():.1f}%) have a "
              f"degenerate objective (>1e3 or non-finite) -> excluded from medians")
    df.loc[bad, ["best_obj", "fttc_obj", "l1_obj"]] = np.nan
    return df


def load_curves(stage, df):
    """The per-row objective curves, from the force npzs (lazy per-member reads).

    Returns a tidy frame (cond, scene, input, path, reg, obj) with one row per
    regularization point, plus each row's own best, so a basin can be plotted as
    obj / best without re-deriving the optimum."""
    rows, mets = [], []
    for r in df.itertuples():
        fp = os.path.join(stage, "cache", r.cond, r.scene, f"force_{r.input}.npz")
        if not os.path.exists(fp):
            continue
        try:
            d = np.load(fp)
            # scalar metrics recorded at each path's own optimum -- these are what let
            # us ask whether the ranked objective is smuggling in a preference
            g = lambda k: float(d[k]) if k in d.files else float("nan")
            mets.append(dict(cond=r.cond, scene=r.scene, input=r.input, method=r.method,
                             is_cell=r.is_cell, foot_um=r.foot_um,
                             peak_disp_px=r.peak_disp_px,
                             l2_nrmse=g("fttc_nrmse"), l1_nrmse=g("l1_nrmse"),
                             l2_ang=g("fttc_ang_field"), l1_ang=g("l1_ang_field"),
                             l2_bg=g("fttc_bg_leak"), l1_bg=g("l1_bg_leak"),
                             l2_J=g("fttc_J"), l1_J=g("l1_J")))
            for path, ckey, gkey in (("FISTA+L1", "l1_obj_curve", "frac1_grid"),
                                     ("FTTC+L2", "fttc_obj_curve", "lambda_grid")):
                obj, grid = np.asarray(d[ckey], float), np.asarray(d[gkey], float)
                ok = np.isfinite(obj) & (obj < 1e3)
                if not ok.any():
                    continue
                best = obj[ok].min()
                for reg, o, good in zip(grid, obj, ok):
                    rows.append((r.cond, r.scene, r.input, r.method, r.is_cell,
                                 path, float(reg), float(o) if good else np.nan, best))
        except Exception as exc:                          # noqa: BLE001 -- log & skip
            print(f"  curve read failed {fp}: {type(exc).__name__}: {exc}")
    cur = pd.DataFrame(rows, columns=["cond", "scene", "input", "method", "is_cell",
                                      "path", "reg", "obj", "row_best"])
    cur["rel"] = cur["obj"] / cur["row_best"]
    return cur, pd.DataFrame(mets)


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #
def per_method_best(df):
    """Each method's ceiling per (condition, scene): the best it reaches over its
    own resolution ladder and both regularizer paths."""
    g = (df.groupby(["cond", "scene", "method"], as_index=False)
           .agg(best_obj=("best_obj", "min")))
    keys = df.drop_duplicates(["cond", "scene"]).set_index(["cond", "scene"])
    for c in ["foot_um", "peak_disp_px", "is_cell", "cell_id", "expo", "NA",
              "density", "jitter", "scenario"]:
        g[c] = g.set_index(["cond", "scene"]).index.map(keys[c])
    return g


def winners(pm):
    """Per (condition, scene): winning method, margin to runner-up."""
    p = pm.pivot_table(index=["cond", "scene"], columns="method", values="best_obj")
    p = p.dropna(how="all")
    out = pd.DataFrame(index=p.index)
    out["winner"] = p.idxmin(axis=1)
    srt = np.sort(p.values, axis=1)
    out["best"] = srt[:, 0]
    out["margin"] = srt[:, 1] - srt[:, 0]          # absolute gap to runner-up
    return out.reset_index()


def _grid(d, rows, cols, val, aggfunc="median"):
    return d.pivot_table(index=rows, columns=cols, values=val, aggfunc=aggfunc)


def _hm(ax, M, title, cbar_label, log=True, cmap=SEQ, vmin=None, vmax=None, fmt="{:.2f}",
        cbar=True):
    v = M.values.astype(float)
    norm = (LogNorm(vmin=vmin or np.nanmin(v[v > 0]), vmax=vmax or np.nanmax(v))
            if log else None)
    im = ax.imshow(v, cmap=cmap, norm=norm, vmin=None if log else vmin,
                   vmax=None if log else vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(M.shape[1]), [f"{c:g}" for c in M.columns], rotation=45)
    ax.set_yticks(range(M.shape[0]), [f"{r:g}" for r in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=6,
                        color=_ink(im.cmap(im.norm(v[i, j]))))
    ax.set_title(title, fontsize=8)
    ax.grid(False)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, fontsize=7)
    return im


def _ink(rgba):
    """Black or white label, whichever is legible on this cell (relative luminance)."""
    r, g, b = rgba[:3]
    return "w" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "k"


def _log_ticks(ax, values, fmt="{:g}", max_labels=9):
    """Label a log axis at the sampled points only. Matplotlib's default decade
    minor ticks collide illegibly when the grid spans well under a decade — and a
    33-point grid spanning eight decades collides just as badly if every point is
    labelled, so thin to at most `max_labels` evenly-spaced samples."""
    v = list(values)
    step = max(1, int(np.ceil(len(v) / max_labels)))
    keep = v[::step]
    if v[-1] not in keep:
        keep.append(v[-1])
    ax.set_xticks(keep, [fmt.format(x) for x in keep], rotation=45, fontsize=6.5)
    ax.minorticks_off()


# --------------------------------------------------------------------------- #
# Figure 1 -- method competence
# --------------------------------------------------------------------------- #
def fig_competence(pm, win, outdir):
    d = pm[~pm["is_cell"]]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    vals = d["best_obj"].dropna()
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    im = None
    for ax, m in zip(axes[:3], METHODS):
        M = _grid(d[d["method"] == m], "foot_um", "peak_disp_px", "best_obj")
        # one shared colourbar: the three panels share a scale, so three identical
        # bars would just be furniture between them
        im = _hm(ax, M, MNAME[m], "median best J", vmin=vmin, vmax=vmax, cbar=False)
        ax.set_xlabel("peak displacement (px)")
        if m == METHODS[0]:
            ax.set_ylabel("footprint σ (µm)")
    cb = fig.colorbar(im, ax=axes[:3].tolist(), fraction=0.02, pad=0.01)
    cb.set_label("median best J (lower is better)", fontsize=7)

    # winner panel: modal winner per regime cell, annotated with win-fraction
    w = win.merge(pm.drop_duplicates(["cond", "scene"])[["cond", "scene", "foot_um",
                                                         "peak_disp_px", "is_cell"]],
                  on=["cond", "scene"])
    w = w[~w["is_cell"]]
    tab = (w.groupby(["foot_um", "peak_disp_px"])["winner"]
             .agg(mode=lambda s: s.value_counts().idxmax(),
                  frac=lambda s: s.value_counts(normalize=True).max()))
    idx = sorted(w["foot_um"].unique()); col = sorted(w["peak_disp_px"].unique())
    codes = np.full((len(idx), len(col)), np.nan)
    ax = axes[3]
    for i, f in enumerate(idx):
        for j, u in enumerate(col):
            if (f, u) in tab.index:
                codes[i, j] = METHODS.index(tab.loc[(f, u), "mode"])
    cmap = ListedColormap([MCOLOR[m] for m in METHODS])
    ax.imshow(codes, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 3.5), 3),
              origin="lower", aspect="auto")
    for i, f in enumerate(idx):
        for j, u in enumerate(col):
            if (f, u) in tab.index:
                ax.text(j, i, f"{tab.loc[(f, u), 'mode']}\n{tab.loc[(f, u), 'frac']:.0%}",
                        ha="center", va="center", fontsize=6,
                        color=_ink(cmap(int(codes[i, j]))))
    ax.set_xticks(range(len(col)), [f"{c:g}" for c in col], rotation=45)
    ax.set_yticks(range(len(idx)), [f"{r:g}" for r in idx])
    ax.set_title("winner per regime (share of conditions)", fontsize=8)
    ax.set_xlabel("peak displacement (px)")
    ax.grid(False)
    fig.suptitle("Method competence: achievable force ceiling over the regime grid "
                 "(dipoles, lower J is better)", fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "heuristic-sweep-competence.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figure 2 -- what the winner uses
# --------------------------------------------------------------------------- #
def fig_parameters(df, win, outdir):
    d = df[~df["is_cell"]]
    # the winning ROW per (cond, scene): the single best (input, path) combination
    best_row = d.loc[d.groupby(["cond", "scene"])["best_obj"].idxmin().dropna()]
    best_row = best_row.assign(
        path=np.where(best_row["l1_obj"] <= best_row["fttc_obj"], "FISTA+L1", "FTTC+L2"))

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    # (a) modal resolution knob of the winner
    res = _grid(best_row, "foot_um", "peak_disp_px", "res_val",
                aggfunc=lambda s: s.value_counts().idxmax())
    _hm(axes[0], res, "winner's resolution knob\n(PIV window / iLK radius / FFD spacing)",
        "modal knob value", log=False, cmap="cividis", fmt="{:.0f}")
    axes[0].set_ylabel("footprint σ (µm)")

    # (b) which regularizer path wins, as a share
    shr = (best_row.assign(is_l1=best_row["path"].eq("FISTA+L1"))
                   .pivot_table(index="foot_um", columns="peak_disp_px", values="is_l1"))
    _hm(axes[1], shr, "sparsity vs smoothness\n(share of conditions won by FISTA+L1)",
        "L1 win share", log=False, cmap=DIV, vmin=0, vmax=1, fmt="{:.0%}")

    # (c,d) the regularization each path actually chose
    l1 = _grid(best_row[best_row["path"] == "FISTA+L1"], "foot_um", "peak_disp_px", "l1_frac1")
    _hm(axes[2], l1, "oracle L1 sparsity of the winner", "median l1_sparsity",
        log=False, cmap="magma_r", fmt="{:.3f}")
    lam = _grid(d, "foot_um", "peak_disp_px", "fttc_lambda")
    _hm(axes[3], lam, "oracle FTTC λ (L2 path)", "median λ", cmap="magma_r", fmt="{:.0e}")
    for ax in axes:
        ax.set_xlabel("peak displacement (px)")
    fig.suptitle("Parameter heuristics: what the winning configuration uses", fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "heuristic-sweep-parameters.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, best_row


# --------------------------------------------------------------------------- #
# Figure 3 -- regularization basins
# --------------------------------------------------------------------------- #
def _basin(ax, cur, path, xlabel, logx=True):
    """Median penalty curve with an IQR ribbon.

    Plotting one faint line per field (there are thousands) buries the median under
    a grey wall three decades tall, because a handful of degenerate solves ring up
    ratios of 100+. The quartile ribbon says the same thing about spread and leaves
    the basin readable."""
    c = cur[(cur["path"] == path) & np.isfinite(cur["rel"])]
    q = c.groupby("reg")["rel"].quantile([0.25, 0.75]).unstack()
    ax.fill_between(q.index, q[0.25], q[0.75], color=PCOLOR[path], alpha=0.16,
                    lw=0, zorder=1, label="interquartile range")
    med = c.groupby("reg")["rel"].median()
    ax.plot(med.index, med.values, color=PCOLOR[path], lw=2.2, zorder=3, label="median")
    # The plateau is defined against the BEST ACHIEVABLE MEDIAN, not against 1.0. The
    # median of per-field ratios never touches 1.0 (different fields peak at different
    # regularizations), so an absolute threshold would report "no safe range" even
    # where the curve is dead flat. What the operator cares about is the spread of
    # values that cost no more than a few percent over the best single choice.
    lo = float(med.min())
    ax.axhline(PLATEAU_TOL * lo, color="0.35", ls=":", lw=1, zorder=2)
    safe = med[med <= PLATEAU_TOL * lo]
    if len(safe):
        ax.axvspan(safe.index.min(), safe.index.max(), color=PCOLOR[path], alpha=0.10,
                   zorder=0,
                   label=f"plateau ≤{PLATEAU_TOL - 1:.0%} over best: "
                         f"{safe.index.min():.3g}–{safe.index.max():.3g}")
    if logx:
        ax.set_xscale("log")
        _log_ticks(ax, med.index, "{:.3g}")
    ax.set_ylim(1.0, max(3.0, float(q[0.75].max()) * 1.05))
    ax.set_xlabel(xlabel); ax.set_ylabel("objective / this field's best")
    ax.set_title(f"{path}: cost of a wrong regularization", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center")
    return med


def fig_regularization(cur, met, df, outdir):
    c = cur[~cur["is_cell"]]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.8))
    med_l1 = _basin(axes[0], c, "FISTA+L1", "L1 sparsity")
    med_l2 = _basin(axes[1], c, "FTTC+L2", "FTTC λ")

    # strategy ladder: per-field oracle vs the best single fixed value, per path.
    # Restricted to the CORE regime -- in the dead corners every strategy scores ~1
    # (nothing is recoverable), which compresses the comparison toward "no difference"
    # for reasons that have nothing to do with regularization.
    ax = axes[2]
    axes_key = df[~df["is_cell"]].drop_duplicates(["cond", "scene"])
    core_keys = set(map(tuple, axes_key[(axes_key["foot_um"] >= 0.266)
                                        & axes_key["peak_disp_px"].between(1.2, 20)]
                        [["cond", "scene"]].values))
    ccore = c[[(a, b) in core_keys for a, b in zip(c["cond"], c["scene"])]]
    data, labels, colors = [], [], []
    for path in PATHS:
        cc = ccore[ccore["path"] == path]
        oracle = cc.groupby(["cond", "scene", "input"])["row_best"].first().dropna()
        med = cc.groupby("reg")["rel"].median()
        fixed_reg = med.idxmin()
        fixed = cc[cc["reg"] == fixed_reg].groupby(["cond", "scene", "input"])["obj"].first().dropna()
        data += [oracle.values, fixed.values]
        labels += [f"{path}\noracle per field", f"{path}\nbest fixed ({fixed_reg:.3g})"]
        colors += [PCOLOR[path], PCOLOR[path]]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6)
    for patch, col, alpha in zip(bp["boxes"], colors, [0.85, 0.35, 0.85, 0.35]):
        patch.set_facecolor(col); patch.set_alpha(alpha)
    for med_line in bp["medians"]:
        med_line.set_color("k")
    ax.set_yscale("log"); ax.set_ylabel("Sabass J")
    ax.set_title(f"Tuned vs one-size-fits-all\n(core regime, {len(core_keys)} scenes)",
                 fontsize=8)
    ax.tick_params(axis="x", labelsize=6.5)
    # (d) THE CAVEAT PANEL. J contains DTMS, a direct reward for a zero background,
    # which group-L1 supplies by construction. Re-rank the same configurations on
    # metrics that carry no background term and the verdict changes character: L1's
    # accuracy edge is marginal while L2's directional fidelity is large.
    ax = axes[3]
    mc = met[[(a, b) in core_keys for a, b in zip(met["cond"], met["scene"])]]
    crit = [("Sabass J\n(contains DTMS)", "l2_J", "l1_J", True),
            ("whole-field\nnRMSE", "l2_nrmse", "l1_nrmse", True),
            ("angular\nerror", "l2_ang", "l1_ang", True)]
    xs = np.arange(len(crit))
    shares = []
    for _, a_, b_, lower_better in crit:
        ok = np.isfinite(mc[a_]) & np.isfinite(mc[b_])
        shares.append(100 * float((mc[b_][ok] < mc[a_][ok]).mean()))
    ax.bar(xs, shares, color=PCOLOR["FISTA+L1"], width=0.55, label="FISTA+L1 better")
    ax.bar(xs, [100 - v for v in shares], bottom=shares, color=PCOLOR["FTTC+L2"],
           width=0.55, label="FTTC+L2 better")
    for x, v in zip(xs, shares):
        ax.text(x, v / 2, f"{v:.0f}%", ha="center", va="center", color="w", fontsize=7.5)
        ax.text(x, v + (100 - v) / 2, f"{100 - v:.0f}%", ha="center", va="center",
                color="w", fontsize=7.5)
    ax.set_xticks(xs, [c[0] for c in crit], fontsize=6.5)
    ax.set_ylim(0, 100); ax.set_ylabel("share of configurations")
    ax.set_title("What the ranked objective hides\n(same configurations, three criteria)",
                 fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.grid(False)

    fig.suptitle("Regularization: the safe plateau, what tuning buys, and what the "
                 "objective is quietly rewarding", fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "heuristic-sweep-regularization.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, med_l1, med_l2


# --------------------------------------------------------------------------- #
# Figure 4 -- imaging drivers and the recoverable envelope
# --------------------------------------------------------------------------- #
def fig_imaging(pm, outdir):
    d = pm[~pm["is_cell"]].dropna(subset=["best_obj"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    # (a) per-condition ceiling, coloured by exposure
    per = (d.groupby(["cond", "expo", "NA", "density", "jitter"], as_index=False)
             ["best_obj"].median().sort_values("best_obj"))
    expos = sorted(per["expo"].unique())
    ecol = dict(zip(expos, plt.get_cmap(SEQ)(np.linspace(0.15, 0.85, len(expos)))))
    axes[0].barh(range(len(per)), per["best_obj"],
                 color=[ecol[e] for e in per["expo"]], height=0.75)
    axes[0].set_yticks(range(len(per)), per["cond"], fontsize=6)
    axes[0].set_xlabel("median best J over all regimes")
    axes[0].set_xlim(0, float(per["best_obj"].max()) * 1.08)
    axes[0].set_title("Imaging condition ranking", fontsize=8)
    for e in expos:
        axes[0].bar(0, 0, color=ecol[e], label=f"exposure {e:g}")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")

    # (b) driver strength (Spearman rho of each imaging parameter against the ceiling).
    # Correlated on the PER-CONDITION medians, not on raw rows: within one condition the
    # regime grid spans J from 0.02 to >1, so a row-level correlation measures footprint
    # and displacement (the axes that dominate the variance) and reports every imaging
    # parameter as ~0. One row per imaging condition is the question actually being asked.
    drivers = ["expo", "density", "NA", "jitter"]
    rho = {k: per[k].corr(per["best_obj"], method="spearman") for k in drivers}
    rho = dict(sorted(rho.items(), key=lambda kv: -abs(kv[1])))
    axes[1].barh(list(rho), list(rho.values()),
                 color=["#D55E00" if v > 0 else "#0072B2" for v in rho.values()], height=0.6)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Spearman ρ with achievable J   (negative = better images, lower J)")
    axes[1].set_title("Rank correlation per imaging parameter\n"
                      "(8 scenarios sample the axes non-orthogonally — describes, "
                      "does not attribute)", fontsize=7.5)
    lim = max(abs(v) for v in rho.values()) * 1.45
    axes[1].set_xlim(-lim, lim)
    for i, (k, v) in enumerate(rho.items()):
        axes[1].text(v + (0.03 * lim if v > 0 else -0.03 * lim), i, f"{v:+.2f}",
                     va="center", ha="left" if v > 0 else "right", fontsize=7)

    # (c) envelope: where good images pay off, best exposure minus worst
    hi = _grid(d[d["expo"] == max(expos)], "foot_um", "peak_disp_px", "best_obj")
    lo = _grid(d[d["expo"] == min(expos)], "foot_um", "peak_disp_px", "best_obj")
    diff = lo - hi
    v = np.nanmax(np.abs(diff.values))
    _hm(axes[2], diff, f"cost of low exposure\n(J at {min(expos):g} − J at {max(expos):g})",
        "ΔJ", log=False, cmap=DIV, vmin=-v, vmax=v)
    axes[2].set_xlabel("peak displacement (px)"); axes[2].set_ylabel("footprint σ (µm)")
    fig.suptitle("Imaging quality: every quality axis points the same way, and the "
                 "gain lands at the envelope edge", fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "heuristic-sweep-imaging.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, rho, per


# --------------------------------------------------------------------------- #
# Figure 5 -- diffuse cells
# --------------------------------------------------------------------------- #
def fig_cells(pm, df, cur, outdir):
    d = pm[pm["is_cell"]].dropna(subset=["best_obj"])
    df_cells = df[df["is_cell"]].dropna(subset=["best_obj"])
    if d.empty:
        return None, None
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    # (a) winner per (cell, strength)
    p = d.pivot_table(index="cell_id", columns="peak_disp_px", values="best_obj",
                      aggfunc="min")
    wm = (d.loc[d.groupby(["cell_id", "peak_disp_px"])["best_obj"].idxmin()]
            .pivot(index="cell_id", columns="peak_disp_px", values="method"))
    codes = wm.map(lambda m: METHODS.index(m) if isinstance(m, str) else np.nan).values
    ax = axes[0, 0]
    _mc = ListedColormap([MCOLOR[m] for m in METHODS])
    ax.imshow(codes.astype(float), cmap=_mc,
              norm=BoundaryNorm(np.arange(-0.5, 3.5), 3), origin="lower", aspect="auto")
    for i in range(wm.shape[0]):
        for j in range(wm.shape[1]):
            if isinstance(wm.values[i, j], str):
                ax.text(j, i, f"{wm.values[i, j]}\n{p.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=6,
                        color=_ink(_mc(int(codes[i, j]))))
    ax.set_xticks(range(wm.shape[1]), [f"{c:g}" for c in wm.columns], rotation=45)
    ax.set_yticks(range(wm.shape[0]), list(wm.index))
    ax.set_xlabel("peak displacement (px)"); ax.set_title("A  winner per cell & strength", fontsize=8)
    ax.grid(False)

    # (b) the U in strength
    ax = axes[0, 1]
    for m in METHODS:
        s = d[d["method"] == m].groupby("peak_disp_px")["best_obj"].median()
        ax.plot(s.index, s.values, "o-", color=MCOLOR[m], label=MNAME[m], ms=4, lw=1.8)
    ax.set_xscale("log"); ax.set_xlabel("peak displacement (px)")
    ax.set_ylabel("best whole-field nRMSE"); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("B  recovery is a U in strength", fontsize=8)

    # (c) does the dipole L1 plateau transfer?
    ax = axes[1, 0]
    cc = cur[cur["is_cell"] & (cur["path"] == "FISTA+L1") & np.isfinite(cur["rel"])]
    dd = cur[~cur["is_cell"] & (cur["path"] == "FISTA+L1") & np.isfinite(cur["rel"])]
    for src, col, lab in ((dd, "0.55", "dipoles"), (cc, PCOLOR["FISTA+L1"], "cells")):
        med = src.groupby("reg")["rel"].median()
        ax.plot(med.index, med.values, "o-", color=col, lw=2, ms=4, label=lab)
        # plateau against each curve's OWN best, as in fig 3: the level of these two
        # curves is not comparable (it reflects how far the optimum roams across
        # scenes, not how costly tuning is), but their SHAPE is.
        safe = med[med <= PLATEAU_TOL * float(med.min())]
        if len(safe):
            ax.axvspan(safe.index.min(), safe.index.max(), color=col, alpha=0.12,
                       label=f"{lab} plateau {safe.index.min():.3g}–{safe.index.max():.3g}")
    ax.set_xscale("log")
    _log_ticks(ax, sorted(cc["reg"].unique()), "{:.3g}")
    ax.set_xlabel("L1 sparsity"); ax.set_ylabel("nRMSE / best")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("C  the L1 plateau transfers from dipole to cell", fontsize=8)

    # (d) the optimum tracks SNR. Both knobs on one axis, each normalised to its own
    # maximum (they have incommensurate units and a second y-scale would be a lie);
    # the endpoints carry the real values so the shape is readable without one.
    ax = axes[1, 1]
    dc = df_cells.loc[df_cells.groupby(["cell_id", "peak_disp_px"])["best_obj"].idxmin()]
    for col, key, lab, fmt in ((PCOLOR["FISTA+L1"], "l1_frac1", "oracle L1 sparsity", "{:.3g}"),
                               (PCOLOR["FTTC+L2"], "fttc_lambda", "oracle FTTC λ", "{:.0e}")):
        s = dc.groupby("peak_disp_px")[key].median().dropna()
        if s.empty:
            continue
        ax.plot(s.index, s.values / s.values.max(), "o-", color=col, lw=1.8, ms=4, label=lab)
        for x, dy in ((s.index[0], 7 if key == "l1_frac1" else -11),
                      (s.index[-1], 7 if key == "l1_frac1" else -11)):
            ax.annotate(fmt.format(s.loc[x]), (x, s.loc[x] / s.values.max()),
                        textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=6, color=col)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("peak displacement (px)"); ax.set_ylabel("optimum (normalised to its max)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("D  regularization relaxes as SNR improves", fontsize=8)

    fig.suptitle("Diffuse fields: benchmarkTFM synth cells (ranked on whole-field nRMSE)",
                 fontsize=9)
    fig.tight_layout()
    out = os.path.join(outdir, "heuristic-sweep-cells.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out, d


# --------------------------------------------------------------------------- #
# Figure 6 -- what winning looks like
# --------------------------------------------------------------------------- #
def _gt_of(stage, cond, scene):
    import tomllib
    sdir = os.path.join(stage, "scenes", cond, scene)
    with open(os.path.join(sdir, "scene.toml"), "rb") as fh:
        sc = tomllib.load(fh)
    if sc["meta"].get("kind") == "cell":
        return np.load(os.path.join(sdir, "gt_traction.npy")).astype(np.float32)
    from scoring import rasterize_gt
    return rasterize_gt(sc, C.GT_REFERENCE_SIZE)[0]


def fig_examples(stage, df, outdir, picks=None):
    """GT vs each path's oracle recovery, for a few representative scenes."""
    d = df.dropna(subset=["best_obj"])
    if picks is None:
        # one clean dipole, one hard dipole, one mid-strength cell -- chosen by rank,
        # not by hand, so the illustration cannot flatter the result.
        dip = d[~d["is_cell"]]
        rank = dip.groupby(["cond", "scene"])["best_obj"].min().sort_values()
        picks = [rank.index[0], rank.index[len(rank) // 2]]
        cell = d[d["is_cell"]]
        if len(cell):
            cr = cell.groupby(["cond", "scene"])["best_obj"].min().sort_values()
            picks.append(cr.index[0])
    fig, axes = plt.subplots(len(picks), 3, figsize=(9.6, 3.15 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, (cond, scene) in enumerate(picks):
        gt = _gt_of(stage, cond, scene)
        row = d[(d["cond"] == cond) & (d["scene"] == scene)].nsmallest(1, "best_obj").iloc[0]
        fp = os.path.join(stage, "cache", cond, scene, f"force_{row['input']}.npz")
        z = np.load(fp)
        vmax = float(np.hypot(gt[0], gt[1]).max())
        panels = [("ground truth", gt, ""),
                  ("FTTC+L2 oracle", np.asarray(z["fttc_map"], np.float32),
                   f"λ={float(z['fttc_lambda']):.1e}  obj={float(row['fttc_obj']):.3f}"),
                  ("FISTA+L1 oracle", np.asarray(z["l1_map"], np.float32),
                   f"l1={float(z['l1_frac1']):.3g}  obj={float(row['l1_obj']):.3f}")]
        for j, (name, t, sub) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(np.hypot(t[0], t[1]), cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"{name}\n{sub}" if sub else name, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{cond}/{scene}\n{row['input']}", fontsize=6.5)
            if j == 2:
                cb = fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.03, pad=0.02)
                cb.set_label("|t| (Pa)", fontsize=6.5)
        # box the winner
        wj = 1 if row["fttc_obj"] <= row["l1_obj"] else 2
        wpath = "FTTC+L2" if row["fttc_obj"] <= row["l1_obj"] else "FISTA+L1"
        for sp in axes[i, wj].spines.values():
            sp.set(visible=True, color=PCOLOR[wpath], linewidth=2.5)
    fig.suptitle("What the ceiling looks like: ground truth vs each path's oracle "
                 "recovery (winner boxed)", fontsize=9, y=0.995)
    p = os.path.join(outdir, "heuristic-sweep-examples.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Edge artifact -- does the sparsity prior manufacture a boundary?
# --------------------------------------------------------------------------- #
def fig_edge(stage, df, outdir):
    """Radial profile out of a pole, GT vs both paths, plus zoomed crops.

    The GT is a Gaussian: it decays smoothly and has no boundary. Group-L1 makes a
    per-pixel keep-or-zero decision, so it terminates the blob at a level set of the
    field -- a step that exists in neither the ground truth nor the displacement.
    Shown on the sweep's single best L1 recovery, so it cannot be dismissed as a
    badly-tuned case."""
    import tomllib
    d = df[(~df["is_cell"]) & (df["foot_um"] >= 0.266)
           & df["peak_disp_px"].between(1.2, 20)].dropna(subset=["best_obj"])
    r = d.nsmallest(1, "l1_obj").iloc[0]
    cond, scene = r["cond"], r["scene"]
    sdir = os.path.join(stage, "scenes", cond, scene)
    with open(os.path.join(sdir, "scene.toml"), "rb") as fh:
        sc = tomllib.load(fh)
    from scoring import rasterize_gt
    N = C.GT_REFERENCE_SIZE
    gt, _ = rasterize_gt(sc, N)
    z = np.load(os.path.join(stage, "cache", cond, scene, f"force_{r['input']}.npz"))
    fields = [("ground truth", np.hypot(*gt)),
              ("FTTC+L2", np.hypot(*np.asarray(z["fttc_map"], np.float32))),
              ("FISTA+L1", np.hypot(*np.asarray(z["l1_map"], np.float32)))]
    ps = sc["meta"]["pixel_size"] * (sc["meta"]["image_size"] / N)
    axd = np.radians(sc["pair"].get("axis_deg", 0.0))
    sep = sc["pair"]["separation"] / ps
    cy, cx = N / 2 + (sep / 2) * np.sin(axd), N / 2 + (sep / 2) * np.cos(axd)

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    half = int(sep * 0.55)
    y0, y1 = max(0, int(cy) - half), min(N, int(cy) + half)
    x0, x1 = max(0, int(cx) - half), min(N, int(cx) + half)
    vmax = float(fields[0][1].max())
    for ax, (name, mag) in zip(axes[:3], fields):
        im = ax.imshow(mag[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{name}\npeak {mag.max():.0f} Pa", fontsize=8)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    cb.set_label("|t| (Pa)", fontsize=7)

    ax = axes[3]
    # Ray outward from the pole, perpendicular to the dipole axis. Samples that leave
    # the frame are DROPPED, not clamped: clamping repeats the border pixel and paints
    # a flat tail that is an artifact of the indexing, not of the reconstruction.
    t = np.linspace(0, sep * 0.75, 120)
    py_f, px_f = cy + t * np.cos(axd), cx - t * np.sin(axd)
    inside = (py_f >= 0) & (py_f < N) & (px_f >= 0) & (px_f < N)
    t, py, px = t[inside], py_f[inside].astype(int), px_f[inside].astype(int)
    for (name, mag), col in zip(fields, ["0.35", PCOLOR["FTTC+L2"], PCOLOR["FISTA+L1"]]):
        prof = 100 * mag[py, px] / (mag.max() or 1.0)
        ax.plot(t, prof, color=col, lw=2, label=name)
        if name == "FISTA+L1":
            # The cached maps are float16, so "zero" is a quantization floor at ~3e-3 %
            # of peak, not a literal 0. Cut at 0.5% of peak -- three orders above that
            # floor and still far below anything the reconstruction means.
            nz = np.flatnonzero(prof > 0.5)
            if len(nz) and nz[-1] + 1 < len(t):
                xcut = t[nz[-1] + 1]
                gt_here = 100 * fields[0][1][py[nz[-1] + 1], px[nz[-1] + 1]] / fields[0][1].max()
                ax.axvline(xcut, color=col, ls=":", lw=1.2)
                ax.annotate(f"L1 support ends\nGT still at {gt_here:.0f}% of peak",
                            (xcut, 8), fontsize=7, color=col, ha="left",
                            xytext=(8, 0), textcoords="offset points",
                            bbox=dict(fc="w", ec=col, lw=0.6, alpha=0.9, pad=1.5))
    ax.set_xlabel("distance from pole centre (px)")
    ax.set_ylabel("|t| as % of that field's peak")
    # Log axis: on a linear one the whole point -- a fall of three orders of magnitude
    # in three pixels -- is compressed into the thickness of the axis line.
    ax.set_yscale("log"); ax.set_ylim(1e-2, 200)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("The sparsity prior manufactures an edge", fontsize=8)
    fig.suptitle(f"Edge artifact, on the sweep's best L1 recovery "
                 f"({cond}/{scene}, J = {r['l1_obj']:.3f})", fontsize=9)
    fig.tight_layout()
    out = os.path.join(outdir, "heuristic-sweep-edge-artifact.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Clip test -- does post-hoc mask clipping change the L1-vs-L2 verdict?
# --------------------------------------------------------------------------- #
def clip_test(stage, df):
    """Re-score both paths' cached maps with and without the shipped post-hoc mask.

    Cells only: a dipole's mask is its own two blobs, so clipping to it would just
    restate the ground truth. The cached maps were selected on nRMSE (the cell
    objective) and the mask is applied AFTER selection, which is exactly what the
    Force panel's "Clip Outside Mask" does."""
    from scoring import metrics
    cells, rows, gt_cache = df[df["is_cell"]], [], {}
    for r in cells.itertuples():
        sdir = os.path.join(stage, "scenes", r.cond, r.scene)
        mp = os.path.join(sdir, "cell_mask.npy")
        if not os.path.exists(mp):
            print(f"  no cell_mask.npy for {r.scene}; re-run make_cells or pass "
                  f"--scenarios-dir to cell_confinement.py to backfill")
            return None
        if r.scene not in gt_cache:
            gt_cache[r.scene] = (np.load(os.path.join(sdir, "gt_traction.npy")).astype(np.float32),
                                 np.load(mp).astype(bool))
        gt, mask = gt_cache[r.scene]
        fp = os.path.join(stage, "cache", r.cond, r.scene, f"force_{r.input}.npz")
        if not os.path.exists(fp):
            continue
        d = np.load(fp)
        out = dict(scene=r.scene, u=r.peak_disp_px)
        for tag, key in (("l2", "fttc_map"), ("l1", "l1_map")):
            t = np.asarray(d[key], np.float32)
            for how, tt in (("raw", t), ("clip", t * mask[None])):
                m = metrics(tt, gt, do_sabass=False)
                out[f"{tag}_{how}_nr"] = m["nrmse"]
                out[f"{tag}_{how}_ang"] = m["ang_field"]
        rows.append(out)
    m = pd.DataFrame(rows)
    m["band"] = np.where(m["u"] <= 1.2, "noise (<=1.2)",
                np.where(m["u"] <= 8, "useful (1.2-8)", "strong (>8)"))
    print(f"\n=== clip test: {len(m)} cell configurations ===")
    print(f"{'band':16s} {'L2 raw':>8s} {'L2 clip':>8s} {'L1 raw':>8s} {'L1 clip':>8s}   (nRMSE)")
    for b in ["noise (<=1.2)", "useful (1.2-8)", "strong (>8)", "ALL"]:
        s_ = m if b == "ALL" else m[m.band == b]
        print(f"{b:16s} {s_.l2_raw_nr.median():8.3f} {s_.l2_clip_nr.median():8.3f} "
              f"{s_.l1_raw_nr.median():8.3f} {s_.l1_clip_nr.median():8.3f}")
    ok = np.isfinite(m.l2_clip_nr) & np.isfinite(m.l1_clip_nr)
    print(f"  clipped L2 beats clipped L1 on nRMSE in "
          f"{100 * (m.l2_clip_nr[ok] < m.l1_clip_nr[ok]).mean():.1f}%")
    ok2 = np.isfinite(m.l2_clip_ang) & np.isfinite(m.l1_clip_ang)
    print(f"  clipped L2 beats clipped L1 on ANGLE in "
          f"{100 * (m.l2_clip_ang[ok2] < m.l1_clip_ang[ok2]).mean():.1f}%   "
          f"(clipping cannot move this: it only deletes exterior pixels)")
    return m


# --------------------------------------------------------------------------- #
def report(df, pm, win, best_row, med_l1, med_l2, rho, per, cells, met, outdir):
    """Print the numbers the write-up quotes, and save them as a tidy CSV."""
    dip = pm[~pm["is_cell"]]
    w = win.merge(pm.drop_duplicates(["cond", "scene"])[["cond", "scene", "is_cell"]],
                  on=["cond", "scene"])
    wd = w[~w["is_cell"]]
    print("\n=== method competence (dipoles) ===")
    tbl = []
    for m in METHODS:
        sub = dip[dip["method"] == m]
        rank = (dip.pivot_table(index=["cond", "scene"], columns="method",
                                values="best_obj").rank(axis=1))
        near = (dip.pivot_table(index=["cond", "scene"], columns="method", values="best_obj"))
        within = (near[m] <= 1.10 * near.min(axis=1)).mean()
        tbl.append(dict(method=m, wins=int((wd["winner"] == m).sum()), n=len(wd),
                        mean_rank=float(rank[m].mean()), within_10pct=float(within),
                        median_J=float(sub["best_obj"].median())))
    t = pd.DataFrame(tbl)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== regularizer path (dipoles, winning configuration) ===")
    print(best_row["path"].value_counts(normalize=True).to_string())

    print("\n=== L1 basin (median obj / best) ===")
    print(med_l1.to_string(float_format=lambda x: f"{x:.3f}"))
    safe = med_l1[med_l1 <= 1.02]
    if len(safe):
        print(f"safe plateau (<=2% penalty): {safe.index.min():.4g} .. {safe.index.max():.4g}")
    print("\n=== core regime only (footprint >= 0.266 um, 1.2 <= u <= 20 px) ===")
    core = dip[(dip["foot_um"] >= 0.266) & dip["peak_disp_px"].between(1.2, 20)]
    cw = win.merge(pm.drop_duplicates(["cond", "scene"])[
        ["cond", "scene", "foot_um", "peak_disp_px", "is_cell"]], on=["cond", "scene"])
    cw = cw[(~cw["is_cell"]) & (cw["foot_um"] >= 0.266) & cw["peak_disp_px"].between(1.2, 20)]
    for m in METHODS:
        print(f"  {m:4s} wins {int((cw['winner'] == m).sum()):3d}/{len(cw)}  "
              f"median J {core[core['method'] == m]['best_obj'].median():.3f}")

    print("\n=== L1 vs L2 by criterion (core regime; metrics at each path's own optimum) ===")
    mc = met[(~met["is_cell"]) & (met["foot_um"] >= 0.266)
             & met["peak_disp_px"].between(1.2, 20)]
    for lab, a_, b_ in [("Sabass J (has DTMS)", "l2_J", "l1_J"),
                        ("whole-field nRMSE", "l2_nrmse", "l1_nrmse"),
                        ("angular error", "l2_ang", "l1_ang"),
                        ("background leak", "l2_bg", "l1_bg")]:
        ok = np.isfinite(mc[a_]) & np.isfinite(mc[b_])
        print(f"  {lab:22s} L1 better in {100 * (mc[b_][ok] < mc[a_][ok]).mean():5.1f}%   "
              f"median L2 {mc[a_][ok].median():8.3f}  L1 {mc[b_][ok].median():8.3f}")
    mk = met[met["is_cell"]]
    print("  cells (objective IS nRMSE, so both paths are nRMSE-optimal there):")
    for lab, a_, b_ in [("whole-field nRMSE", "l2_nrmse", "l1_nrmse"),
                        ("angular error", "l2_ang", "l1_ang")]:
        ok = np.isfinite(mk[a_]) & np.isfinite(mk[b_])
        print(f"  {lab:22s} L1 better in {100 * (mk[b_][ok] < mk[a_][ok]).mean():5.1f}%   "
              f"median L2 {mk[a_][ok].median():8.3f}  L1 {mk[b_][ok].median():8.3f}")

    print("\n=== imaging drivers ===")
    print(f"Spearman rho on the {len(per)} per-condition medians:")
    for k, v in rho.items():
        print(f"  {k:9s} {v:+.3f}")
    # The 8 scenarios sample (density x NA x exposure) sparsely and NOT orthogonally,
    # so a rank correlation cannot cleanly attribute credit between them. Marginal
    # medians per level are the honest reading of the same data -- report both.
    print("marginal median J per level (confounded design -- read as description, "
          "not attribution):")
    for k in ["expo", "density", "NA", "jitter"]:
        s = per.groupby(k)["best_obj"].median()
        print(f"  {k:9s} " + "  ".join(f"{lv:g}:{v:.3f}" for lv, v in s.items()))

    if cells is not None and len(cells):
        print("\n=== diffuse cells ===")
        cw = (cells.loc[cells.groupby(["cell_id", "peak_disp_px"])["best_obj"].idxmin()])
        print(cw["method"].value_counts().to_string())
        print(cells.groupby("peak_disp_px")["best_obj"].min().to_string(
            float_format=lambda x: f"{x:.3f}"))

    # the headline table goes next to the script, never into --outdir: that points at
    # docs/images/ in the canonical invocation and a CSV has no business living there.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures",
                       "sweep_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t.to_csv(out, index=False)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=os.environ.get("STAGE"))
    ap.add_argument("--outdir", default=None,
                    help="where figures go; default ./figures (gitignored)")
    ap.add_argument("--clip-test", action="store_true",
                    help="also re-score both regularizer paths with and without the "
                         "shipped post-hoc mask clip (cells only; ~1 min)")
    a = ap.parse_args()
    if not a.stage:
        ap.error("--stage or $STAGE required")
    outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)

    print(f"stage  {a.stage}\noutdir {outdir}")
    df = load_index(a.stage)
    print(f"index  {len(df)} rows · {df['cond'].nunique()} conditions · "
          f"{df.groupby(['cond', 'scene']).ngroups} scenes")
    pm = per_method_best(df)
    win = winners(pm)
    cur, met = load_curves(a.stage, df)
    print(f"curves {len(cur)} regularization points")

    paths = [fig_competence(pm, win, outdir)]
    p2, best_row = fig_parameters(df, win, outdir); paths.append(p2)
    p3, med_l1, med_l2 = fig_regularization(cur, met, df, outdir); paths.append(p3)
    p4, rho, per = fig_imaging(pm, outdir); paths.append(p4)
    p5, cells = fig_cells(pm, df, cur, outdir); paths.append(p5)
    paths.append(fig_examples(a.stage, df, outdir))
    paths.append(fig_edge(a.stage, df, outdir))

    report(df, pm, win, best_row, med_l1, med_l2, rho, per, cells, met, outdir)
    if a.clip_test:
        clip_test(a.stage, df)
    print("\nfigures:")
    for p in paths:
        if p:
            print(" ", p)


if __name__ == "__main__":
    main()
