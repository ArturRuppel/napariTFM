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
import argparse, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import sweep_config as C
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


def run_disp(ref, dfm, method, res_val, conv_val):
    """One displacement solve at a given resolution + convergence setting."""
    kw = {"disp_method": method, "disp_device": "auto",
          "pixel_size": C.PIXEL_SIZE_UM, "downscale_factor": C.DOWNSCALE_FACTOR}
    kw[C.RES_KNOB[method]] = res_val      # for PIV this IS downscale_factor (overrides default)
    kw[C.CONV_KNOB[method]] = conv_val
    p = DisplacementParameters(**kw)
    g = calculate_displacement_field(ref, dfm, p)
    try:
        while True:
            next(g)
    except StopIteration as e:
        return np.asarray(e.value.displacement_field)[0]   # (y, x, 2), µm


def converge(ref, dfm, method, res_val):
    """Raise the convergence knob until the field settles; return converged field."""
    prev = None
    for conv_val in C.CONV_LADDER[method]:
        field = run_disp(ref, dfm, method, res_val, conv_val)
        if prev is not None and prev.shape == field.shape:
            denom = float(np.sqrt((prev ** 2).sum())) or 1.0
            rel = float(np.sqrt(((field - prev) ** 2).sum())) / denom
            if rel < C.CONV_TOL:
                return field, conv_val, rel
        prev = field
    return prev, C.CONV_LADDER[method][-1], float("nan")   # never settled -> last


def cache_scene(stage, condition, scene_id, method):
    sdir = os.path.join(stage, "scenes", condition, scene_id)
    ref = prep(load(os.path.join(sdir, "reference.tif")))
    dfm = prep(load(os.path.join(sdir, "deformed.tif")))
    outdir = os.path.join(stage, "cache", condition, scene_id)
    os.makedirs(outdir, exist_ok=True)
    for k, res_val in enumerate(C.RES_VALUES[method]):
        field, conv_val, rel = converge(ref, dfm, method, res_val)
        out = os.path.join(outdir, f"disp_res{k}.npz")
        np.savez_compressed(
            out, field=field.astype(np.float32),
            method=method, res_knob=C.RES_KNOB[method], res_val=res_val,
            conv_knob=C.CONV_KNOB[method], conv_val=conv_val, conv_rel=rel,
            downscale=C.DOWNSCALE_FACTOR, pixel_size=C.PIXEL_SIZE_UM,
        )
        print(f"  [{scene_id}] res {C.RES_KNOB[method]}={res_val:<5} "
              f"conv {C.CONV_KNOB[method]}={conv_val} (rel={rel:.4f}) "
              f"grid={field.shape[:2]} -> {os.path.basename(out)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--condition", default=C.CONDITIONS[0])
    ap.add_argument("--scene", default=None, help="one scene id; default = all")
    ap.add_argument("--method", default=C.DISP_METHOD)
    a = ap.parse_args()

    root = os.path.join(a.stage, "scenes", a.condition)
    if not os.path.isdir(root):
        sys.exit(f"no scenes at {root} -- has the generator run?")
    scenes = [a.scene] if a.scene else sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"caching {len(scenes)} scene(s), method={a.method}, condition={a.condition}")
    for s in scenes:
        cache_scene(a.stage, a.condition, s, a.method)


if __name__ == "__main__":
    main()
