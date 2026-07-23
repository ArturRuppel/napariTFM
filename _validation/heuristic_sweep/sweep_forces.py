#!/usr/bin/env python3
"""Stage 2: elastic-net force sweep over the displacement cache.

For each cached (scene, resolution) displacement field, invert over the full
FRAC1 x FRAC2 grid, upsample each recovery to the common GT_REFERENCE_SIZE grid,
and score it against the analytic GT traction rasterized from scene.toml.
Resolution stays IN the search. Writes one tidy CSV row per
(condition, scene, resolution, l1, l2).

The objective is the Sabass (2008) composite J = |DTM| + DTMS + DTA/45 -- it splits
recovery error into magnitude-on-adhesions / spurious-background / direction, the three
modes the L1+L2 knobs trade off. nRMSE and corr are recorded alongside as cross-checks,
never as the ranked objective (a single blended nRMSE hides the DTM<->DTMS tradeoff the
heuristic exists to characterize).

Usage:
    python sweep_forces.py --stage "$STAGE" [--condition realistic]
                           [--scene f30_m400] [--out results/sweep.csv]
"""
from __future__ import annotations
import argparse, csv, glob, os, sys, tomllib
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.ndimage import zoom

import sweep_config as C
import sabass
from napariTFM.backend.forward_l1 import l1_traction_frame
from napariTFM.backend.parameter_dataclasses import FTTCParameters


def _add_pair(tx, ty, xx, yy, cx, cy, sep, foot, axis_deg, mag, profile):
    """Add one balanced contractile pair (two equal-and-opposite poles along the
    axis, each pulling toward the pair centre) to the traction accumulators, in px
    units. Net force of the pair is zero, so any sum of pairs is DC-free too."""
    ax = np.radians(axis_deg)
    ux, uy = np.cos(ax), np.sin(ax)
    for sgn in (+1.0, -1.0):
        px = cx + sgn * (sep / 2) * ux
        py = cy + sgn * (sep / 2) * uy
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        if profile == "gaussian":
            w = np.exp(-r2 / (2 * foot ** 2))
        else:  # tophat
            w = (r2 < foot ** 2).astype(np.float32)
        tx += -sgn * mag * ux * w      # contractile: pole pulls inward (-sgn·axis)
        ty += -sgn * mag * uy * w


def rasterize_gt(scene, N):
    """Analytic GT traction of the balanced dipole pair on an N x N grid ->
    ((2, N, N) Pa, significant mask). Two equal-and-opposite contractile poles,
    net force zero (required: the Green's operator zeroes the DC mode). Generation
    and scoring both call this, so the dipole GT never drifts between them.

    Cell scenes do NOT go through here: their GT traction is a stored field
    (gt_traction.npy, the benchmarkTFM fitted-fibre geometry), loaded directly by
    sweep_scene. See make_cells.py."""
    meta = scene["meta"]
    ps = meta["pixel_size"] * (meta["image_size"] / N)         # µm per GT px
    pair = scene["pair"]
    yy, xx = np.mgrid[0:N, 0:N]
    tx = np.zeros((N, N), np.float32)
    ty = np.zeros((N, N), np.float32)
    cx = N / 2 + pair.get("center", [0.0, 0.0])[0] / ps
    cy = N / 2 + pair.get("center", [0.0, 0.0])[1] / ps
    _add_pair(tx, ty, xx, yy, cx, cy, pair["separation"] / ps, pair["footprint"] / ps,
              pair.get("axis_deg", 0.0), pair["magnitude"],
              pair.get("profile", C.GT_TRACTION_PROFILE))
    return np.stack([tx, ty], 0), (tx ** 2 + ty ** 2) > 0


def field_metrics(t, gt, frac=0.05):
    """Whole-field recovery metrics that stay well-defined on a DIFFUSE field,
    where the per-adhesion Sabass terms degrade (one merged component, a mean
    traction vector that cancels). Scored over the significant-GT region.

    - mag_bias  : magnitude-weighted scale error, sum‖t‖/sum‖t_gt‖ − 1 (signed).
    - ang_field : magnitude-weighted mean angular error in degrees (per-pixel
                  cosine, so it survives cancellation that kills a mean vector).
    - bg_leak   : spurious energy off the source, sum‖t‖(bg) / sum‖t‖(sig).
    """
    magt, magg = np.hypot(t[0], t[1]), np.hypot(gt[0], gt[1])
    if magg.max() <= 0:
        return dict(mag_bias=float("nan"), ang_field=float("nan"), bg_leak=float("nan"))
    sig = magg > frac * magg.max()
    bg = ~sig
    sum_g = float(magg[sig].sum()) or 1.0
    mag_bias = float(magt[sig].sum() / sum_g - 1.0)
    dot = (t[0] * gt[0] + t[1] * gt[1])
    denom = np.where((magt > 0) & (magg > 0), magt * magg, 1.0)
    cospix = np.clip(dot / denom, -1.0, 1.0)
    w = magg[sig]
    cos_w = float((w * cospix[sig]).sum() / (w.sum() or 1.0))
    ang_field = float(np.degrees(np.arccos(np.clip(cos_w, -1.0, 1.0))))
    on = float(magt[sig].sum()) or 1.0
    bg_leak = float(magt[bg].sum() / on)
    return dict(mag_bias=mag_bias, ang_field=ang_field, bg_leak=bg_leak)


def metrics(t, gt, do_sabass=True):
    """Whole-field metrics (well-suited to diffuse cells) + nRMSE/corr, plus the
    Sabass composite J and its components when `do_sabass` (isolated dipoles).

    The RANKING metric is chosen at analysis time per scene kind: J for dipoles,
    the whole-field terms for cells. Sabass is skipped for cells: its per-adhesion
    loop is O(n_adh · pixels) and blows up on an 82-fibre field, and DTA/DTM are
    ill-defined once the significant region merges -- exactly why the whole-field
    terms exist. Skipped fields are recorded as NaN so the schema stays uniform."""
    magt, magg = np.hypot(t[0], t[1]), np.hypot(gt[0], gt[1])
    gnorm = float(np.sqrt((gt ** 2).sum())) or 1.0
    out = dict(
        J=float("nan"), dtm=float("nan"), dtms=float("nan"), dta=float("nan"), n_adh=0,
        nrmse=float(np.sqrt(((t - gt) ** 2).sum()) / gnorm),
        corr=float(np.corrcoef(magt.ravel(), magg.ravel())[0, 1]),
        **field_metrics(t, gt),
    )
    if do_sabass:
        fa = sabass.significant_mask(gt, frac=0.1)
        s = sabass.sabass_metrics(t, gt, fa)
        out.update(J=float(sabass.objective(s["dtm"], s["dtms"], s["dta"])),
                   dtm=s["dtm"], dtms=s["dtms"], dta=s["dta"], n_adh=s["n_adh"])
    return out


def invert(field, gt_h):
    """Traction from one cached displacement field, upsampled to the GT grid."""
    h, w = field.shape[:2]
    eff_ps = C.PIXEL_SIZE_UM * (C.GT_REFERENCE_SIZE / h)   # physical node spacing, µm
    base = dict(young_modulus=C.YOUNG_MODULUS, poisson_ratio_substrate=C.POISSON,
                pixel_size=eff_ps, downscale_factor=1, fwd_device="auto",
                fwd_dtype="float32", fwd_mask_strength=0.0, l1_max_iter=C.L1_MAX_ITER)
    for f1 in C.FRAC1:
        for f2 in C.FRAC2:
            p = FTTCParameters(l1_sparsity=float(f1), l2_ridge=float(f2), **base)
            t = np.asarray(l1_traction_frame(field, p, mask=None))     # (2, h, w)
            t_up = zoom(t, (1, gt_h / h, gt_h / h), order=1)           # -> GT grid
            yield f1, f2, t_up


def sweep_scene(stage, condition, scene_id, writer):
    sdir = os.path.join(stage, "scenes", condition, scene_id)
    with open(os.path.join(sdir, "scene.toml"), "rb") as fh:
        scene = tomllib.load(fh)
    N = C.GT_REFERENCE_SIZE
    meta = scene["meta"]
    if meta.get("kind") == "cell":                            # cell: stored GT field
        kind = "cell"
        gt = np.load(os.path.join(sdir, "gt_traction.npy")).astype(np.float32)
        footprint = float("nan")
        magnitude = float(meta.get("rms_traction_pa", float("nan")))
    else:                                                      # dipole: analytic GT
        kind = "dipole"
        gt, _ = rasterize_gt(scene, N)
        footprint = scene["pair"]["footprint"]
        magnitude = scene["pair"]["magnitude"]
    scene_cols = dict(
        kind=kind, footprint=footprint, magnitude=magnitude,
        peak_disp_px=float(meta.get("peak_disp_px", float("nan"))),
        n_fibers=int(meta.get("n_fibers", 0)))
    caches = sorted(glob.glob(os.path.join(stage, "cache", condition, scene_id, "disp_*_res*.npz")))
    if not caches:
        print(f"  [{scene_id}] no cache -- run build_cache.py first", flush=True)
        return
    for cf in caches:
        d = np.load(cf)
        field = d["field"].astype(np.float32)   # cache is float16; compute in float32
        method = str(d["method"])
        smooth_val = float(d["smooth_val"]) if "smooth_val" in d.files else float("nan")
        for f1, f2, t_up in invert(field, N):
            m = metrics(t_up, gt, do_sabass=(kind != "cell"))
            writer.writerow(dict(
                condition=condition, scene_id=scene_id, method=method, **scene_cols,
                res_knob=str(d["res_knob"]), res_val=float(d["res_val"]),
                conv_val=float(d["conv_val"]), smooth_val=smooth_val, grid=field.shape[0],
                l1=f1, l2=f2, **m))
        print(f"  [{scene_id}] {os.path.basename(cf)} swept "
              f"({len(C.FRAC1)}x{len(C.FRAC2)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--condition", default=C.CONDITIONS[0])
    ap.add_argument("--scene", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = os.path.join(a.stage, "scenes", a.condition)
    scenes = [a.scene] if a.scene else sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    out = a.out or os.path.join(a.stage, "results", f"sweep_{a.condition}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["condition", "scene_id", "method", "kind", "footprint", "magnitude",
            "peak_disp_px", "n_fibers", "res_knob", "res_val", "conv_val", "smooth_val",
            "grid", "l1", "l2", "J", "dtm", "dtms", "dta", "n_adh", "nrmse", "corr",
            "mag_bias", "ang_field", "bg_leak"]
    print(f"sweeping {len(scenes)} scene(s) -> {out}")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for s in scenes:
            sweep_scene(a.stage, a.condition, s, writer)
    print("done ->", out)


if __name__ == "__main__":
    main()
