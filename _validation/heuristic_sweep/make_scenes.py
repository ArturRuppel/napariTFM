#!/usr/bin/env python3
"""Stage 0 (local): synthesize the dipole scenario grid from a real bead stack.

For each (footprint, magnitude) balanced dipole:
  * rasterize the analytic GT traction (the SAME rasterize_gt the scorer uses,
    so generation and scoring share one GT definition -- no drift between them),
  * forward-project it to a GT displacement field via the Green's operator,
  * for each real bead frame k: reference = frame_k, deformed = warp(frame_k),
    a self-consistent pair -> one seed each.
The synthetic warp is ground truth; the real per-frame bead texture (density, PSF,
photon noise) is the realism, and varying it across frames gives the seeds. The
8 timepoints are 60 s apart and fully decorrelate (hi-freq corr ~0.08), so they
are NOT cross-referenced -- each frame is paired only with its own warp.

Writes $STAGE/scenes/<condition>/<scene_id>/{reference.tif,deformed.tif,scene.toml}
per the generator contract in README.md. The bead stack is PRIVATE: pass it with
--stack; it is never written into the repo or committed.

Usage:
    python make_scenes.py --stack /path/to/reference.ome.tiff --stage "$STAGE"
                          [--n-foot 2 --n-mag 2 --n-seed 2]   # pilot subset
"""
from __future__ import annotations
import argparse, os
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

import sweep_config as C
from sweep_forces import rasterize_gt
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, help="private bead stack (8,2,H,W)")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--condition", default=C.CONDITIONS[0])
    ap.add_argument("--n-foot", type=int, default=len(C.FOOTPRINTS_UM))
    ap.add_argument("--n-disp", type=int, default=len(C.PEAK_DISP_PX))
    ap.add_argument("--n-seed", type=int, default=C.N_SEEDS)
    a = ap.parse_args()

    stack = tifffile.imread(a.stack)                      # (T, C, H, W)
    beads = centre_crop(stack[:, C.BEAD_CHANNEL], C.CROP_SIZE).astype(np.float64)
    frames = list(enumerate(beads[:min(a.n_seed, beads.shape[0])]))  # (k, frame_k)

    foots = C.FOOTPRINTS_UM[:a.n_foot]
    disps = C.PEAK_DISP_PX[:a.n_disp]
    N = C.CROP_SIZE
    root = os.path.join(a.stage, "scenes", a.condition)
    print(f"generating {len(foots)}x{len(disps)}x{len(frames)} = "
          f"{len(foots)*len(disps)*len(frames)} scenes -> {root}")
    for foot in foots:
        # Green's op is linear: displacement peak for unit peak-traction sets the
        # magnitude scale, so we derive mag to hit each target peak displacement.
        unit_t, _ = rasterize_gt(scene_dict(a.condition, "unit", foot, 1.0), N)
        unit_peak = float(np.hypot(*greens_displacement(unit_t, N)).max())   # µm per Pa
        for dpx in disps:
            target_um = dpx * C.PIXEL_SIZE_UM
            mag = target_um / unit_peak                   # derived peak traction, Pa
            base = f"f{foot:g}_u{dpx:g}"
            gt_traction, _ = rasterize_gt(scene_dict(a.condition, base, foot, mag, dpx), N)
            u = greens_displacement(gt_traction, N)       # GT displacement, shared across seeds
            umax = float(np.hypot(u[0], u[1]).max())
            for k, frame in frames:                       # each real frame -> its own self-warped seed
                deformed = warp(frame, u, C.PIXEL_SIZE_UM)
                sid = f"{base}__seed{k}"
                d = os.path.join(root, sid)
                os.makedirs(d, exist_ok=True)
                tifffile.imwrite(os.path.join(d, "reference.tif"),
                                 np.clip(frame, 0, 65535).astype(np.uint16))
                tifffile.imwrite(os.path.join(d, "deformed.tif"),
                                 np.clip(deformed, 0, 65535).astype(np.uint16))
                write_toml(os.path.join(d, "scene.toml"), scene_dict(a.condition, sid, foot, mag, dpx))
            print(f"  {base}: mag={mag:8.1f}Pa |u|max={umax:.3f}µm (target {target_um:.3f}) "
                  f"x{len(frames)} seeds", flush=True)


if __name__ == "__main__":
    main()
