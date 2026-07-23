#!/usr/bin/env python3
"""Pre-render the oracle force cache into browsable PNGs.

The browser (browse.py) used to read the displacement cache and RECOMPUTE traction
on every interaction. Now the GT-tuned oracle force maps are cached
(build_force_cache.py: FTTC+L2 and FISTA+L1, each tuned to the scene objective),
so there is nothing left to solve at browse time -- we render one self-contained
comparison card per (scene, displacement input) here, once, and the browser just
displays the PNG.

Per (cond, scene, disp input) the card is a 3x3:

    input beads      | GT |u|            | measured |u| (quiver)
    GT |t|           | FTTC+L2 oracle    | FISTA+L1 oracle          (shared Pa scale)
    disp error       | FTTC force error  | FISTA force error        (errors)

Force maps come straight from the cache (float16). The GT displacement is the same
Green's-function field the beads were warped by, so measured-vs-GT displacement is
scored like-for-like. Every scalar (nRMSE / Sabass J, lambda*, frac1*, the winner)
goes in the suptitle, and one row per input is appended to renders/index.csv so the
browser can show a ranking table without touching a single npz.

Usage:
    python render_cache.py [--stage DIR] [--condition C] [--scene S] [--out DIR]
Defaults: stage=/home/aruppel/Data/tfm_heuristic, out=$stage/renders, all conditions.
"""
from __future__ import annotations
import argparse, glob, os, re, csv, tomllib
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
from functools import lru_cache

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import sweep_config as C
from sweep_forces import rasterize_gt
from make_scenes import greens_displacement

N = C.GT_REFERENCE_SIZE
SEQ_DISP, SEQ_TRAC, SEQ_ERR = "viridis", "inferno", "magma"
# index.csv columns: enough to rank configs by force ceiling without reading a map.
INDEX_COLS = ["cond", "scene", "kind", "objective", "input", "method", "res_knob",
              "res_val", "smooth", "peak_disp_px", "disp_nrmse",
              "fttc_obj", "fttc_lambda", "l1_obj", "l1_frac1", "winner", "png"]


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def load_scene(stage, cond, scene):
    """Input beads (ref, dfm), GT traction (2,N,N) Pa, GT displacement (N,N,2) um, kind.

    GT displacement = greens_displacement(GT traction): exactly the field the beads
    were warped by in generation, so the measured field can be scored against it."""
    sdir = os.path.join(stage, "scenes", cond, scene)
    with open(os.path.join(sdir, "scene.toml"), "rb") as fh:
        sc = tomllib.load(fh)
    ref = np.asarray(Image.open(os.path.join(sdir, "reference.tif")), np.float32)
    dfm = np.asarray(Image.open(os.path.join(sdir, "deformed.tif")), np.float32)
    if sc["meta"].get("kind") == "cell":
        gt = np.load(os.path.join(sdir, "gt_traction.npy")).astype(np.float32)
        kind = "cell"
    else:
        gt, _ = rasterize_gt(sc, N)
        kind = "dipole"
    u_gt = np.moveaxis(greens_displacement(gt, N), 0, -1).astype(np.float32)  # (N,N,2)
    return ref, dfm, gt, u_gt, kind


def _gt_at(a512, h, is_vec_last):
    """Block-mean a 512 GT map to grid h. is_vec_last: (512,512,2) vs (2,512,512)."""
    f = N // h
    if f <= 1:
        return a512
    if is_vec_last:
        return a512[:h * f, :h * f].reshape(h, f, h, f, 2).mean(axis=(1, 3))
    return a512[:, :h * f, :h * f].reshape(2, h, f, h, f).mean(axis=(2, 4))


def _disp_nrmse(field, u_gt_h):
    denom = np.sqrt((u_gt_h ** 2).sum()) or 1.0
    return float(np.sqrt(((field - u_gt_h) ** 2).sum()) / denom)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _heat(ax, mag, cmap, title, vmax=None, quiver=None):
    im = ax.imshow(mag, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if quiver is not None:
        u, v, step = quiver
        yy, xx = np.mgrid[0:u.shape[0]:step, 0:u.shape[1]:step]
        # origin="upper" flips y; negate the row component so downward disp draws down.
        ax.quiver(xx, yy, u[::step, ::step], -v[::step, ::step], color="white",
                  scale_units="xy", width=0.004, alpha=0.7)


def _overlay(ax, ref, dfm, title):
    lo, hi = np.percentile(np.concatenate([ref.ravel(), dfm.ravel()]), [1, 99.5])
    r = np.clip((ref - lo) / (hi - lo + 1e-9), 0, 1)
    d = np.clip((dfm - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = (np.stack([d, r, d], -1) * 255).astype(np.uint8)   # R=dfm, G=ref
    ax.imshow(rgb, origin="upper"); ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])


def render_card(stage, cond, scene, disp_path, force_path, out_png):
    """Render one (scene, disp input) comparison card; return the index row dict."""
    ref, dfm, gt, u_gt, kind = load_scene(stage, cond, scene)
    d = np.load(disp_path)
    field = np.asarray(d["field"], np.float32)          # (h,h,2) um; cache is float16
    h = field.shape[0]
    fz = np.load(force_path)
    fttc = np.asarray(fz["fttc_map"], np.float32)        # (2,512,512) Pa
    l1 = np.asarray(fz["l1_map"], np.float32)
    objective = str(fz["objective"])                     # "nrmse" (cell) | "J" (dipole)
    fttc_obj = float(fz[f"fttc_{objective}"]); l1_obj = float(fz[f"l1_{objective}"])
    fttc_lam = float(fz["fttc_lambda"]); l1_f1 = float(fz["l1_frac1"])
    peak_disp = float(fz["peak_disp_px"]) if "peak_disp_px" in fz.files else float("nan")

    # force maps are 512 (native oracle grid); block-mean GT and error at 512 too.
    gt_f = gt                                            # (2,512,512)
    fttc_err = np.hypot(fttc[0] - gt_f[0], fttc[1] - gt_f[1])
    l1_err = np.hypot(l1[0] - gt_f[0], l1[1] - gt_f[1])
    gmag = np.hypot(gt_f[0], gt_f[1])
    tvmax = float(gmag.max()) or None
    evmax = float(max(fttc_err.max(), l1_err.max())) or None   # shared error scale

    # displacement (grid h): measured vs GT@h
    u_gt_h = _gt_at(u_gt, h, is_vec_last=True)
    umag = np.hypot(field[..., 0], field[..., 1])
    ugmag = np.hypot(u_gt_h[..., 0], u_gt_h[..., 1])
    uerr = np.hypot(field[..., 0] - u_gt_h[..., 0], field[..., 1] - u_gt_h[..., 1])
    uvmax = float(max(ugmag.max(), umag.max())) or None
    qstep = max(h // 24, 1)
    disp_nrmse = _disp_nrmse(field, u_gt_h)

    lo = "nRMSE" if objective == "nrmse" else "J"
    better = "FTTC+L2" if fttc_obj < l1_obj else "FISTA+L1"    # lower objective wins

    # row0 = displacement (GT | measured | error); row1 = FTTC+L2 force stage,
    # row2 = FISTA+L1 force stage -- each force row is (GT/context | recovered | error).
    # Columns thus read GT/reference | measured/recovered | error.
    fig = Figure(figsize=(11.5, 11.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.subplots(3, 3)
    _heat(ax[0, 0], ugmag, SEQ_DISP, f"GT |u|  peak {ugmag.max():.2f} um", vmax=uvmax,
          quiver=(u_gt_h[..., 0], u_gt_h[..., 1], qstep))
    _heat(ax[0, 1], umag, SEQ_DISP, f"measured |u|  peak {umag.max():.2f} um  grid {h}",
          vmax=uvmax, quiver=(field[..., 0], field[..., 1], qstep))
    _heat(ax[0, 2], uerr, SEQ_ERR, f"disp error |u-GT|  peak {uerr.max():.2f} um", vmax=uvmax)
    _heat(ax[1, 0], gmag, SEQ_TRAC, f"GT |t|  peak {gmag.max():.0f} Pa", vmax=tvmax)
    _heat(ax[1, 1], np.hypot(fttc[0], fttc[1]), SEQ_TRAC,
          f"FTTC+L2 oracle |t|  {lo}={fttc_obj:.3f}  lam={fttc_lam:.2g}", vmax=tvmax)
    _heat(ax[1, 2], fttc_err, SEQ_ERR, f"FTTC force error |t-GT|  peak {fttc_err.max():.0f} Pa",
          vmax=evmax)
    _overlay(ax[2, 0], ref, dfm, f"input  ref(G)/deformed(M)  {ref.shape[0]}px")
    _heat(ax[2, 1], np.hypot(l1[0], l1[1]), SEQ_TRAC,
          f"FISTA+L1 oracle |t|  {lo}={l1_obj:.3f}  f1={l1_f1:.3g}", vmax=tvmax)
    _heat(ax[2, 2], l1_err, SEQ_ERR, f"FISTA force error |t-GT|  peak {l1_err.max():.0f} Pa",
          vmax=evmax)

    sup = (f"{cond}/{scene} ({kind})  ·  {re.sub(r'^disp_|.npz$', '', os.path.basename(disp_path))}"
           f"  ·  FTTC+L2 {lo}={fttc_obj:.3f}@lam={fttc_lam:.2g}"
           f"  ·  FISTA+L1 {lo}={l1_obj:.3f}@f1={l1_f1:.3g}  ·  winner {better}")
    fig.suptitle(sup, fontsize=9.5, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.0, w_pad=0.8)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=100)

    return {
        "cond": cond, "scene": scene, "kind": kind, "objective": objective,
        "input": re.sub(r"^disp_|\.npz$", "", os.path.basename(disp_path)),
        "method": str(d["method"]), "res_knob": str(d["res_knob"]),
        "res_val": int(d["res_val"]),
        "smooth": (float(d["smooth_val"]) if "smooth_val" in d.files else float("nan")),
        "peak_disp_px": peak_disp, "disp_nrmse": round(disp_nrmse, 5),
        "fttc_obj": round(fttc_obj, 5), "fttc_lambda": fttc_lam,
        "l1_obj": round(l1_obj, 5), "l1_frac1": l1_f1, "winner": better,
        "png": os.path.relpath(out_png, os.path.join(stage, "renders")),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_scene(stage, out_root, cond, scene, rows):
    cdir = os.path.join(stage, "cache", cond, scene)
    disps = sorted(f for f in glob.glob(os.path.join(cdir, "disp_*_res*.npz"))
                   if ".tmp." not in os.path.basename(f))
    n = 0
    for dp in disps:
        suffix = re.sub(r"^disp_", "", os.path.basename(dp))         # <method>_res..npz
        fp = os.path.join(cdir, "force_" + suffix)
        if not os.path.exists(fp):
            continue
        out_png = os.path.join(out_root, cond, scene, re.sub(r"\.npz$", ".png", suffix))
        row = render_card(stage, cond, scene, dp, fp, out_png)
        rows.append(row)
        n += 1
    print(f"  [{cond}/{scene}] {n} cards", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="/home/aruppel/Data/tfm_heuristic")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_root = a.out or os.path.join(a.stage, "renders")
    os.makedirs(out_root, exist_ok=True)

    conds = [a.condition] if a.condition else sorted(
        d for d in os.listdir(os.path.join(a.stage, "cache"))
        if os.path.isdir(os.path.join(a.stage, "cache", d)))
    rows = []
    for cond in conds:
        scenes = [a.scene] if a.scene else sorted(
            d for d in os.listdir(os.path.join(a.stage, "cache", cond))
            if os.path.isdir(os.path.join(a.stage, "cache", cond, d)))
        for scene in scenes:
            process_scene(a.stage, out_root, cond, scene, rows)

    # merge into index.csv (replace rows for the cond/scene we just rendered)
    idx = os.path.join(out_root, "index.csv")
    existing = []
    if os.path.exists(idx):
        with open(idx) as fh:
            existing = [r for r in csv.DictReader(fh)]
    touched = {(r["cond"], r["scene"]) for r in rows}
    keep = [r for r in existing if (r["cond"], r["scene"]) not in touched]
    allrows = keep + [{k: r[k] for k in INDEX_COLS} for r in rows]
    allrows.sort(key=lambda r: (r["cond"], r["scene"], str(r["input"])))
    with open(idx, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS); w.writeheader(); w.writerows(allrows)
    print(f"wrote {len(rows)} cards, index.csv now {len(allrows)} rows -> {idx}", flush=True)


if __name__ == "__main__":
    main()
