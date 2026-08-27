#!/usr/bin/env python3
"""Stage 0 (local): synthesize the dipole scenario grid from the synthetic stacks.

For each imaging condition (one scenario stack x one jitter frame) and each
(footprint, magnitude) balanced dipole:
  * rasterize the analytic GT traction (the SAME rasterize_gt the scorer uses,
    so generation and scoring share one GT definition -- no drift between them),
  * forward-project it to a GT displacement field u via the Green's operator,
  * deformed = warp(frame_k, u); reference = frame 0 (the zero-jitter reference).
The pair is cross-frame on purpose: frame 0 and frame k are the same synthetic
field a known registration jitter apart (0.067-0.2 px) plus independent photon
noise, so that real reference-vs-deformed noise rides along while GT stays u.
Each (scenario, jitter frame) is one condition "s<idx>_j<k>" -> one full sweep.

Writes $STAGE/scenes/<condition>/<scene_id>/{reference.tif,deformed.tif,scene.toml}
per the generator contract in README.md.

Usage:
    python make_scenes.py --images-dir "$STAGE/images/tif_stacks" --stage "$STAGE"
                          [--scenarios 0,1 --jitter-frames 1,3 --n-foot 2 --n-disp 2]
"""
from __future__ import annotations
import argparse, os
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

import sweep_config as C
from scoring import rasterize_gt
from napariTFM.backend.forward_tfm import _greens_operator
from napariTFM.backend.parameter_dataclasses import FTTCParameters


def centre_crop(a, n):
    h, w = a.shape[-2:]
    y0, x0 = (h - n) // 2, (w - n) // 2
    return a[..., y0:y0 + n, x0:x0 + n]


def greens_displacement(traction, N):
    """GT displacement (µm) from GT traction (Pa) via the Boussinesq Green's op."""
    p = FTTCParameters(young_modulus=C.YOUNG_MODULUS, poisson_ratio_substrate=C.POISSON,
                       pixel_size=C.PIXEL_SIZE_UM, downscale_factor=1,
                       fwd_device="cpu", fwd_dtype="float64", fwd_mask_strength=0.0,
                       l1_max_iter=1, l1_sparsity=0.1, l2_ridge=0.0)
    G = _greens_operator(N, N, p).astype(np.complex128)        # (2,2,N,N), û=G·t̂
    tk = np.fft.fft2(traction.astype(np.complex128), axes=(-2, -1))
    uk = np.einsum("ijhw,jhw->ihw", G, tk)
    return np.fft.ifft2(uk, axes=(-2, -1)).real                # (2,N,N) µm


def warp(ref, u, ps):
    """deformed[p] = ref[p - u]; optical flow ref->deformed then recovers +u."""
    N = ref.shape[0]
    Y, X = np.mgrid[0:N, 0:N].astype(np.float64)
    rows = Y - u[1] / ps          # u[1] = y-component
    cols = X - u[0] / ps          # u[0] = x-component
    return map_coordinates(ref.astype(np.float64), [rows, cols], order=1, mode="reflect")


def scene_dict(cond, sid, foot, mag, disp_px=None):
    return {
        "meta": {"condition": cond, "scene_id": sid, "image_size": C.CROP_SIZE,
                 "pixel_size": C.PIXEL_SIZE_UM, "peak_disp_px": disp_px},
        "substrate": {"young_modulus": C.YOUNG_MODULUS, "poisson": C.POISSON},
        "pair": {"profile": C.GT_TRACTION_PROFILE, "footprint": foot, "magnitude": mag,
                 "axis_deg": C.AXIS_DEG, "separation": C.SEPARATION_UM, "center": [0.0, 0.0]},
    }


def write_toml(path, s):
    m, sub, p = s["meta"], s["substrate"], s["pair"]
    disp = "" if m.get("peak_disp_px") is None else f'peak_disp_px = {m["peak_disp_px"]}\n'
    with open(path, "w") as f:
        f.write(f'[meta]\ncondition = "{m["condition"]}"\nscene_id = "{m["scene_id"]}"\n'
                f'image_size = {m["image_size"]}\npixel_size = {m["pixel_size"]}\n{disp}\n'
                f'[substrate]\nyoung_modulus = {sub["young_modulus"]}\npoisson = {sub["poisson"]}\n\n'
                f'[pair]\nprofile = "{p["profile"]}"\nfootprint = {p["footprint"]}\n'
                f'magnitude = {p["magnitude"]}\naxis_deg = {p["axis_deg"]}\n'
                f'separation = {p["separation"]}\ncenter = [{p["center"][0]}, {p["center"][1]}]\n')


def main():
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="dir of synthetic scenario*.tif (TYX) stacks")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--n-foot", type=int, default=len(C.FOOTPRINTS_UM))
    ap.add_argument("--n-disp", type=int, default=len(C.PEAK_DISP_PX))
    ap.add_argument("--scenarios", default=None, help="comma-sep scenario indices (default all)")
    ap.add_argument("--jitter-frames", default=None, help="comma-sep frames (default C.JITTER_FRAMES)")
    a = ap.parse_args()

    stacks = sorted(glob.glob(os.path.join(a.images_dir, "scenario*.tif")))
    def sidx(p): return int(os.path.basename(p).split("scenario")[1].split("_")[0])
    if a.scenarios:
        want = {int(x) for x in a.scenarios.split(",")}
        stacks = [s for s in stacks if sidx(s) in want]
    jframes = [int(x) for x in a.jitter_frames.split(",")] if a.jitter_frames else list(C.JITTER_FRAMES)

    foots = C.FOOTPRINTS_UM[:a.n_foot]
    disps = C.PEAK_DISP_PX[:a.n_disp]
    N = C.CROP_SIZE
    nsc = len(foots) * len(disps)
    print(f"{len(stacks)} scenarios x {len(jframes)} jitter frames = {len(stacks)*len(jframes)} conditions; "
          f"{len(foots)}x{len(disps)}={nsc} scenes each -> {len(stacks)*len(jframes)*nsc} scenes total")

    # Unit displacement peak is scenario-independent (geometry + pixel only) -> once per footprint.
    unit_peak = {}
    for foot in foots:
        ut, _ = rasterize_gt(scene_dict("x", "unit", foot, 1.0), N)
        unit_peak[foot] = float(np.hypot(*greens_displacement(ut, N)).max())   # µm per Pa

    for sp in stacks:
        idx = sidx(sp)
        stack = tifffile.imread(sp)                       # (T, H, W) single-channel TYX
        ref = centre_crop(stack[C.REF_FRAME], N).astype(np.float64)   # frame 0, zero-jitter reference
        for k in jframes:
            cond = f"s{idx}_j{k}"
            src = centre_crop(stack[k], N).astype(np.float64)         # deform-source (jittered frame)
            root = os.path.join(a.stage, "scenes", cond)
            for foot in foots:
                for dpx in disps:
                    mag = (dpx * C.PIXEL_SIZE_UM) / unit_peak[foot]   # derived peak traction, Pa
                    base = f"f{foot:g}_u{dpx:g}"
                    gt_traction, _ = rasterize_gt(scene_dict(cond, base, foot, mag, dpx), N)
                    u = greens_displacement(gt_traction, N)
                    deformed = warp(src, u, C.PIXEL_SIZE_UM)          # deform the jittered frame by u
                    d = os.path.join(root, base)
                    os.makedirs(d, exist_ok=True)
                    tifffile.imwrite(os.path.join(d, "reference.tif"),
                                     np.clip(ref, 0, 65535).astype(np.uint16))       # frame 0
                    tifffile.imwrite(os.path.join(d, "deformed.tif"),
                                     np.clip(deformed, 0, 65535).astype(np.uint16))
                    write_toml(os.path.join(d, "scene.toml"), scene_dict(cond, base, foot, mag, dpx))
            print(f"  {cond}: {nsc} scenes (ref=frame{C.REF_FRAME}, deform=frame{k})", flush=True)


if __name__ == "__main__":
    main()
