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


def rasterize_gt(scene, N):
    """Analytic GT traction of the balanced pair on an N x N grid -> (2, N, N) Pa.

    Two equal-and-opposite uniform-traction discs; net force zero (required, the
    Green's operator zeroes DC). Direction along the pair axis.
    """
    pair = scene["pair"]
    ps = scene["meta"]["pixel_size"] * (scene["meta"]["image_size"] / N)  # µm per GT px
    foot = pair["footprint"] / ps
    sep = pair["separation"] / ps
    ax = np.radians(pair.get("axis_deg", 0.0))
    ux, uy = np.cos(ax), np.sin(ax)
    cx = N / 2 + pair.get("center", [0.0, 0.0])[0] / ps
    cy = N / 2 + pair.get("center", [0.0, 0.0])[1] / ps
    mag = pair["magnitude"]
    profile = pair.get("profile", C.GT_TRACTION_PROFILE)

    yy, xx = np.mgrid[0:N, 0:N]
    tx = np.zeros((N, N), np.float32)
    ty = np.zeros((N, N), np.float32)
    for sgn in (+1.0, -1.0):
        px = cx + sgn * (sep / 2) * ux
        py = cy + sgn * (sep / 2) * uy
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        if profile == "gaussian":
            w = np.exp(-r2 / (2 * foot ** 2))
        else:  # tophat
            w = (r2 < foot ** 2).astype(np.float32)
        # contractile: each pole pulls toward the pair centre (inward = -sgn·axis)
        tx += -sgn * mag * ux * w
        ty += -sgn * mag * uy * w
    return np.stack([tx, ty], 0), (tx ** 2 + ty ** 2) > 0


def metrics(t, gt):
    """Sabass composite J (PRIMARY) + its three components, with nRMSE/corr as cross-checks.
    The adhesion mask is the significant-GT region (|t_gt| > 0.1·peak) -- the two poles."""
    fa = sabass.significant_mask(gt, frac=0.1)
    s = sabass.sabass_metrics(t, gt, fa)
    J = sabass.objective(s["dtm"], s["dtms"], s["dta"])   # |DTM| + DTMS + DTA/45
    magt, magg = np.hypot(t[0], t[1]), np.hypot(gt[0], gt[1])
    gnorm = float(np.sqrt((gt ** 2).sum())) or 1.0
    return dict(
        J=float(J),                                        # PRIMARY objective
        dtm=s["dtm"], dtms=s["dtms"], dta=s["dta"], n_adh=s["n_adh"],
        nrmse=float(np.sqrt(((t - gt) ** 2).sum()) / gnorm),   # cross-check, not ranked
        corr=float(np.corrcoef(magt.ravel(), magg.ravel())[0, 1]),
    )


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
    gt, _ = rasterize_gt(scene, N)
    pair = scene["pair"]
    caches = sorted(glob.glob(os.path.join(stage, "cache", condition, scene_id, "disp_res*.npz")))
    if not caches:
        print(f"  [{scene_id}] no cache -- run build_cache.py first", flush=True)
        return
    for cf in caches:
        d = np.load(cf)
        field = d["field"]
        for f1, f2, t_up in invert(field, N):
            m = metrics(t_up, gt)
            writer.writerow(dict(
                condition=condition, scene_id=scene_id,
                footprint=pair["footprint"], magnitude=pair["magnitude"],
                res_knob=str(d["res_knob"]), res_val=float(d["res_val"]),
                conv_val=float(d["conv_val"]), grid=field.shape[0],
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
    cols = ["condition", "scene_id", "footprint", "magnitude", "res_knob",
            "res_val", "conv_val", "grid", "l1", "l2",
            "J", "dtm", "dtms", "dta", "n_adh", "nrmse", "corr"]
    print(f"sweeping {len(scenes)} scene(s) -> {out}")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for s in scenes:
            sweep_scene(a.stage, a.condition, s, writer)
    print("done ->", out)


if __name__ == "__main__":
    main()
