#!/usr/bin/env python3
"""Stage 1: displacement caching.

For each generated scene, sweep the resolution knob; at each resolution raise the
convergence knob until the recovered displacement field stops changing, and cache
that converged field. Convergence is a self-consistency criterion, never a force
score -- see README.md. One cached field per resolution; the force sweep decides
which resolution wins.

Usage (paths come from $STAGE; see env.sh):
    python build_cache.py --stage "$STAGE" [--condition realistic]
                          [--scene f30_m400]      # single scene, e.g. SLURM array
                          [--method PIV]
"""
from __future__ import annotations
import argparse, os, sys, tomllib
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

import sweep_config as C
from scoring import rasterize_gt
from make_scenes import greens_displacement
from napariTFM.backend.displacement_analysis import calculate_displacement_field
from napariTFM.backend.parameter_dataclasses import DisplacementParameters


def load(p):
    return np.asarray(Image.open(p), np.float32)


def prep(im):
    """Match the pipeline's displacement preprocessing (as in the bridge)."""
    im = np.asarray(im, np.float32)
    lo, hi = np.percentile(im, (80, 99.9))
    if hi > lo:
        im = np.clip((im - lo) / (hi - lo), 0.0, 1.0)
    return gaussian_filter(im, 1)


def run_disp(ref, dfm, method, res_val, conv_val, smooth_val=None):
    """One displacement solve at a given resolution + convergence + smoothing setting."""
    kw = {"disp_method": C.METHOD_LABEL[method], "disp_device": "auto",
          "pixel_size": C.PIXEL_SIZE_UM, "downscale_factor": C.DOWNSCALE_FACTOR}
    kw[C.RES_KNOB[method]] = res_val      # for PIV this IS piv_window (downscale_factor stays fixed)
    kw[C.CONV_KNOB[method]] = conv_val
    sk = C.SMOOTH_KNOB.get(method)        # displacement-side smoother (PIV only); None = method default
    if sk is not None and smooth_val is not None:
        kw[sk] = smooth_val
    p = DisplacementParameters(**kw)
    g = calculate_displacement_field(ref, dfm, p)
    try:
        while True:
            next(g)
    except StopIteration as e:
        return np.asarray(e.value.displacement_field)[0]   # (y, x, 2), µm


def converge(ref, dfm, method, res_val, smooth_val=None):
    """Raise the convergence knob until the field settles; return converged field."""
    prev = None
    for conv_val in C.CONV_LADDER[method]:
        field = run_disp(ref, dfm, method, res_val, conv_val, smooth_val)
        if prev is not None and prev.shape == field.shape:
            denom = float(np.sqrt((prev ** 2).sum())) or 1.0
            rel = float(np.sqrt(((field - prev) ** 2).sum())) / denom
            if rel < C.CONV_TOL:
                return field, conv_val, rel
        prev = field
    return prev, C.CONV_LADDER[method][-1], float("nan")   # never settled -> last


# --------------------------------------------------------------------------- #
# Discrepancy smoothing selection. For each (method, resolution) we solve at every
# candidate smoothing and keep the LARGEST smoothing whose photometric warp residual
# still sits within tol* the noise floor. The floor is the residual of the TRUE (GT)
# displacement field -- the irreducible image mismatch from jitter+photon noise -- so
# "residual back up to the floor" means "stop smoothing once you're only fitting
# noise." Validated to reproduce the disp-nRMSE-vs-GT optimum across regimes (an
# L-curve corner on the same residual/roughness does NOT: the residual has no sharp
# corner). The floor uses GT for one scalar per scene only; it is fully decoupled from
# the downstream force objective (not circular). One field cached per (method, res).
# --------------------------------------------------------------------------- #
DISCREPANCY_TOL = 1.05


def _to_full_px(field, N):
    """(h,h,2) um displacement -> (2,N,N) px ([0]=x,[1]=y) for warping N-sized images."""
    h = field.shape[0]
    up = field if h == N else np.stack(
        [zoom(field[..., c], N / h, order=1) for c in range(2)], axis=-1)
    return np.stack([up[..., 0], up[..., 1]], axis=0) / C.PIXEL_SIZE_UM


def _warp(img, upx):
    N = img.shape[0]
    yy, xx = np.mgrid[0:N, 0:N]
    return map_coordinates(img, [yy + upx[1], xx + upx[0]], order=1, mode="nearest")


def _resid(ref, dfm, field):
    """Symmetric photometric warp residual (matches the PIV objective): warp ref by
    -u/2 and dfm by +u/2, RMS of the leftover mismatch, normalized by image std.
    Lower = the field better explains the observed motion."""
    upx = _to_full_px(field, ref.shape[0])
    a = _warp(ref, -0.5 * upx); b = _warp(dfm, 0.5 * upx)
    return float(np.sqrt(((a - b) ** 2).mean()) / (ref.std() + 1e-9))


def _roughness(field):
    gx = np.gradient(field[..., 0]); gy = np.gradient(field[..., 1])
    return float((gx[0] ** 2 + gx[1] ** 2 + gy[0] ** 2 + gy[1] ** 2).mean())


def gt_floor(stage, condition, scene_id, ref, dfm):
    """Noise floor = photometric residual of the analytic GT displacement field.
    Dipoles only (GT is analytic via rasterize_gt -> greens_displacement)."""
    N = C.GT_REFERENCE_SIZE
    with open(os.path.join(stage, "scenes", condition, scene_id, "scene.toml"), "rb") as fh:
        sc = tomllib.load(fh)
    gt, _ = rasterize_gt(sc, N)
    u_gt = np.moveaxis(greens_displacement(gt, N), 0, -1).astype(np.float32)   # (N,N,2) um
    return _resid(ref, dfm, u_gt)


def _discrepancy_pick(resids, floor):
    """Largest-smoothing index whose residual is within DISCREPANCY_TOL*floor;
    fall back to the min-residual candidate if none qualify."""
    ok = [i for i, r in enumerate(resids) if r <= DISCREPANCY_TOL * floor]
    return max(ok) if ok else int(np.argmin(resids))


def cache_scene(stage, condition, scene_id, methods):
    """Cache every method x resolution for one scene. Returns the number of
    fields written. A method that raises (e.g. FFD on a node without CUDA) is
    logged and skipped so the others still cache -- the caller exits 0 as long as
    at least one field landed, keeping the scene's aftercorr sweep alive."""
    sdir = os.path.join(stage, "scenes", condition, scene_id)
    ref = prep(load(os.path.join(sdir, "reference.tif")))
    dfm = prep(load(os.path.join(sdir, "deformed.tif")))
    outdir = os.path.join(stage, "cache", condition, scene_id)
    os.makedirs(outdir, exist_ok=True)
    floor = gt_floor(stage, condition, scene_id, ref, dfm)   # noise floor, one per scene
    n_written = 0
    for method in methods:
        cands = C.SMOOTH_VALUES.get(method, [None])
        multi = len(cands) > 1
        for k, res_val in enumerate(C.RES_VALUES[method]):
            # Solve at every candidate smoothing, score each by its warp residual, and
            # keep the largest smoothing still within tol*floor (discrepancy). One
            # field cached per (method, resolution).
            solved = []
            for smooth_val in cands:
                try:
                    field, conv_val, rel = converge(ref, dfm, method, res_val, smooth_val)
                except Exception as exc:                   # noqa: BLE001 -- log & skip
                    print(f"  [{scene_id}] {method} res#{k}={res_val} sm={smooth_val} "
                          f"FAILED: {type(exc).__name__}: {exc}", flush=True)
                    continue
                rs = _resid(ref, dfm, field) if multi else float("nan")
                rg = _roughness(field) if multi else float("nan")
                solved.append((smooth_val, field, conv_val, rel, rs, rg))
            if not solved:
                continue
            pick = (_discrepancy_pick([s[4] for s in solved], floor)
                    if len(solved) > 1 else 0)
            smooth_val, field, conv_val, rel, rs, rg = solved[pick]
            out = os.path.join(outdir, f"disp_{method}_res{k}_sm0.npz")
            # Stored as float16: displacement fields are good to ~2 sig figs, float16's
            # 10-bit mantissa (~0.05% rel) is well below solver noise, and it halves the
            # cache (savez DEFLATE is ~useless on float32 mantissa entropy). Readers
            # promote back to float32 on load. smooth_val = the L-curve-chosen knob.
            np.savez_compressed(
                out, field=field.astype(np.float16),
                method=method, res_knob=C.RES_KNOB[method], res_val=res_val,
                conv_knob=C.CONV_KNOB[method], conv_val=conv_val, conv_rel=rel,
                smooth_knob=str(C.SMOOTH_KNOB.get(method)),
                smooth_val=(np.nan if smooth_val is None else float(smooth_val)),
                smooth_resid=rs, smooth_rough=rg, smooth_floor=floor,
                smooth_candidates=np.asarray(
                    [np.nan if c is None else float(c) for c in cands], np.float32),
                downscale=C.DOWNSCALE_FACTOR, pixel_size=C.PIXEL_SIZE_UM,
            )
            n_written += 1
            print(f"  [{scene_id}] {method} res {C.RES_KNOB[method]}={res_val:<5} "
                  f"discrepancy sm={smooth_val} (of {len(solved)}/{len(cands)}; "
                  f"resid {rs:.4f} vs floor {floor:.4f}) conv {C.CONV_KNOB[method]}={conv_val} "
                  f"grid={field.shape[:2]} -> {os.path.basename(out)}", flush=True)
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--condition", default=C.CONDITIONS[0])
    ap.add_argument("--scene", default=None, help="one scene id; default = all")
    ap.add_argument("--methods", default=",".join(C.METHODS),
                    help="comma-sep method keys to cache (default all: PIV,ILK,FFD)")
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in C.METHODS]
    if bad:
        sys.exit(f"unknown method(s) {bad}; known: {C.METHODS}")

    root = os.path.join(a.stage, "scenes", a.condition)
    if not os.path.isdir(root):
        sys.exit(f"no scenes at {root} -- has the generator run?")
    scenes = [a.scene] if a.scene else sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"caching {len(scenes)} scene(s), methods={methods}, condition={a.condition}")
    total = 0
    for s in scenes:
        total += cache_scene(a.stage, a.condition, s, methods)
    if total == 0:
        sys.exit("no displacement fields written -- every method failed")


if __name__ == "__main__":
    main()
