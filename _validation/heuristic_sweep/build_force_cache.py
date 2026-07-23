#!/usr/bin/env python3
"""Stage 2b: GT-tuned oracle FORCE cache over the displacement cache.

For every cached displacement field, compute two *ground-truth-tuned* (oracle)
traction maps and cache them:

  * FTTC + L2   -- Fourier Tikhonov, lambda swept over LAMBDA_GRID, the lambda
                   minimizing the scene objective kept ("what the best pure-L2
                   FTTC could do on this displacement").
  * FISTA + L1  -- real-space group-L1 (l2_ridge = 0), l1_sparsity swept over
                   sweep_config.FRAC1, the sparsity minimizing the scene
                   objective kept ("best pure-L1 could do on this displacement").

Both are GT-TUNED: the regularization is chosen by the *scored* objective against
the analytic / stored GT traction, so each cached map is the achievable force
CEILING for that (displacement field, regularizer). Objective matches the sweep's
convention: Sabass J for isolated dipoles, whole-field nRMSE for diffuse cells.

Grid is native (downscale_factor = 1 -> field is already GT_REFERENCE_SIZE), so
there is no upsample: recovery and GT are compared like-for-like.

Storage: the cached force maps are float16 (the disp cache is float16 too). Selection
of the optimal lambda / l1 is done on full-precision float32 maps -- the float16
rounding (~1e-3 rel) never flips the argmin -- and only the winning map is downcast
at save time. savez_compressed applies lossless DEFLATE on top; the L1 maps are
genuinely sparse (group-L1 zeroes whole regions) so those cost almost nothing.

One npz per displacement field: cache/<cond>/<scene>/force_<method>_res<k>[_sm<s>].npz
mirroring the disp_ naming so the browser can pair them.

Usage:
    python build_force_cache.py --stage "$STAGE" --condition <c> --scene <s>
"""
from __future__ import annotations
import argparse, glob, os, re, tomllib
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

import sweep_config as C
from sweep_forces import rasterize_gt, metrics
from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend.fttc import FTTC
from napariTFM.backend.forward_l1 import l1_traction_frame

N = C.GT_REFERENCE_SIZE
# Backend device for the force solves. Default 'auto' (CuPy if importable, else CPU)
# for local/dev runs; the cluster sbatch sets FWD_DEVICE=cuda so a missing CuPy wheel
# fails LOUD instead of silently falling back to CPU (which is how a GPU job once ran
# the whole FISTA solve at CPU speed with GPUEff 0%).
FWD_DEVICE = os.environ.get("FWD_DEVICE", "auto")
LAMBDA_GRID = np.geomspace(1e-9, 1e-1, 33)          # FTTC Fourier lambda (== browse/compare_l2_reg)
FRAC1_GRID = np.asarray(C.FRAC1, float)             # FISTA L1 sparsity fractions
# metrics recorded at each solver's optimum (flattened into the npz with a prefix)
MET_KEYS = ["nrmse", "corr", "J", "dtm", "dtms", "dta", "n_adh",
            "mag_bias", "ang_field", "bg_leak"]


def _params(eff_ps, **extra):
    return FTTCParameters(young_modulus=C.YOUNG_MODULUS, poisson_ratio_substrate=C.POISSON,
                          pixel_size=eff_ps, downscale_factor=1, fwd_device=FWD_DEVICE,
                          fwd_dtype="float32", fwd_mask_strength=0.0, **extra)


def _fttc(field, params, lam):
    (_, _), forces = FTTC(params).calculate_traction(
        field, pixel_size=params.pixel_size, downscale_factor=1, regularization=lam)
    return np.asarray(forces, np.float32)               # (2, h, w)


def _obj_of(m, kind):
    """Scene objective: Sabass J for dipoles, whole-field nRMSE for cells (the
    sweep/browse convention). NaN when the metric is undefined so the argmin
    guard below can fall back."""
    return m["nrmse"] if kind == "cell" else m["J"]


def _pick(maps, gt, kind):
    """Score each candidate map against GT, return (best_map, best_index, obj_curve,
    metrics_at_best). argmin over the objective with a NaN guard (fall back to
    nRMSE, then to the middle of the grid) so a degenerate J never aborts a field."""
    do_sabass = (kind != "cell")
    mets = [metrics(t, gt, do_sabass=do_sabass) for t in maps]
    obj = np.array([_obj_of(m, kind) for m in mets], float)
    if np.all(np.isnan(obj)):                            # J degenerate -> rank by nRMSE
        obj = np.array([m["nrmse"] for m in mets], float)
    i = int(np.nanargmin(obj)) if not np.all(np.isnan(obj)) else len(maps) // 2
    return maps[i], i, obj, mets[i]


def oracle_fttc(field, gt, eff_ps, kind):
    p = _params(eff_ps)
    maps = [_fttc(field, p, lam) for lam in LAMBDA_GRID]
    best, i, obj, m = _pick(maps, gt, kind)
    return dict(map=best, reg=float(LAMBDA_GRID[i]), obj_curve=obj, met=m)


def oracle_l1(field, gt, eff_ps, kind):
    maps = []
    for f1 in FRAC1_GRID:
        p = _params(eff_ps, l1_max_iter=C.L1_MAX_ITER, l1_sparsity=float(f1), l2_ridge=0.0)
        maps.append(np.asarray(l1_traction_frame(field, p, mask=None), np.float32))
    best, i, obj, m = _pick(maps, gt, kind)
    return dict(map=best, reg=float(FRAC1_GRID[i]), obj_curve=obj, met=m)


def load_gt(sdir, scene_toml):
    meta = scene_toml["meta"]
    if meta.get("kind") == "cell":
        gt = np.load(os.path.join(sdir, "gt_traction.npy")).astype(np.float32)
        return gt, "cell", dict(footprint=float("nan"),
                                magnitude=float(meta.get("rms_traction_pa", float("nan"))),
                                n_fibers=int(meta.get("n_fibers", 0)))
    gt, _ = rasterize_gt(scene_toml, N)
    return gt, "dipole", dict(footprint=float(scene_toml["pair"]["footprint"]),
                              magnitude=float(scene_toml["pair"]["magnitude"]), n_fibers=0)


def process_scene(stage, condition, scene_id, overwrite=False):
    sdir = os.path.join(stage, "scenes", condition, scene_id)
    with open(os.path.join(sdir, "scene.toml"), "rb") as fh:
        sc = tomllib.load(fh)
    gt, kind, scene_cols = load_gt(sdir, sc)
    peak_disp_px = float(sc["meta"].get("peak_disp_px", float("nan")))
    objective = "nrmse" if kind == "cell" else "J"
    cdir = os.path.join(stage, "cache", condition, scene_id)
    caches = sorted(f for f in glob.glob(os.path.join(cdir, "disp_*_res*.npz"))
                    if ".tmp." not in os.path.basename(f))   # skip in-flight float16 temps
    if not caches:
        print(f"  [{scene_id}] no displacement cache -- run build_cache.py first", flush=True)
        return
    for cf in caches:
        suffix = re.sub(r"^disp_", "", os.path.basename(cf))          # <method>_res<k>[_sm<s>].npz
        out = os.path.join(cdir, "force_" + suffix)
        if os.path.exists(out) and not overwrite:
            print(f"  [{scene_id}] {os.path.basename(out)} exists, skip", flush=True)
            continue
        d = np.load(cf)
        field = np.asarray(d["field"], np.float32)      # disp cache is float16; solve in float32
        h = field.shape[0]
        eff_ps = C.PIXEL_SIZE_UM * (N / h)                            # native -> == PIXEL_SIZE_UM
        fttc = oracle_fttc(field, gt, eff_ps, kind)
        l1 = oracle_l1(field, gt, eff_ps, kind)
        payload = dict(
            # provenance mirrored from the displacement field
            method=str(d["method"]), res_knob=str(d["res_knob"]), res_val=int(d["res_val"]),
            conv_val=int(d["conv_val"]),
            smooth_val=float(d["smooth_val"]) if "smooth_val" in d.files else float("nan"),
            grid=int(h), pixel_size=float(eff_ps), kind=kind, objective=objective,
            peak_disp_px=peak_disp_px, lambda_grid=LAMBDA_GRID, frac1_grid=FRAC1_GRID,
            **scene_cols,
            # FTTC + L2 oracle (map float16; DEFLATE lossless on top)
            fttc_map=fttc["map"].astype(np.float16), fttc_lambda=fttc["reg"],
            fttc_obj_curve=fttc["obj_curve"],
            # FISTA + L1 oracle (L2 = 0); sparse -> compresses hard
            l1_map=l1["map"].astype(np.float16), l1_frac1=l1["reg"], l1_obj_curve=l1["obj_curve"],
        )
        for pre, r in (("fttc", fttc), ("l1", l1)):
            for k in MET_KEYS:
                payload[f"{pre}_{k}"] = r["met"].get(k, float("nan"))
        np.savez_compressed(out, **payload)
        print(f"  [{scene_id}] {os.path.basename(out)}  "
              f"FTTC J*={_obj_of(fttc['met'], kind):.4g}@lam={fttc['reg']:.2g}  "
              f"L1 J*={_obj_of(l1['met'], kind):.4g}@f1={l1['reg']:.3g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    process_scene(a.stage, a.condition, a.scene, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
