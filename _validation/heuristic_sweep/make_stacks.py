"""Stage 0 of the sweep: synthesize the 8 imaging-condition bead stacks.

One 4-frame TIFF stack per scenario. The 4 frames share the same scene (bead
positions, NA, SNR) and differ only in bead-position JITTER:
    frame 0 = 0.0 px     (un-jittered reference)
    frame 1 = 0.0667 px
    frame 2 = 0.1333 px
    frame 3 = 0.2 px     (max)
Each frame also gets an independent camera-noise draw. 8 scenarios -> 8 stacks,
sampling (bead density x NA x exposure/SNR). Output: 512x512 uint16 camera counts,
plus manifest.csv (the per-frame density/NA/expo/jitter/ncc table the analysis reads).

Everything is deterministic (fixed per-scenario/-frame seeds): bit-identical under the
same numpy + psfmodels build, and within a couple of camera counts across psfmodels
versions (the PSF kernel shifts slightly; manifest NCCs are unchanged). The exact stack
bytes are frozen in the data deposit. The physics/constants live in calib_psf.py.
Needs `psfmodels` (validation-only): pip install psfmodels.

    python make_stacks.py                    # writes to $STAGE/images/tif_stacks
    python make_stacks.py --outdir <dir>     # or an explicit target
"""
from __future__ import annotations
import os
import csv
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile

import calib_psf as C

PED, READ, GAIN, N = C.PED, C.READ, C.GAIN, C.N
Z = 3.0
FBEAD_REF = C.flux_per_px * N * N / 12000.0        # per-bead flux at expo=1
JITTERS = [0.0, 0.0667, 0.1333, 0.2]               # px, per frame; max = 0.2 px

# the 8 scenarios: (n_beads, NA, expo) -- density x NA x SNR
SCENARIOS = [
    (6000, 0.8, 4.0), (6000, 1.2, 0.25), (12000, 0.8, 1.0), (12000, 1.4, 4.0),
    (24000, 0.6, 0.25), (24000, 1.0, 1.0), (48000, 0.8, 4.0), (48000, 1.2, 0.25),
]


def scene(n_beads, seed):
    r = np.random.default_rng(seed)
    return r.uniform(0, N, n_beads), r.uniform(0, N, n_beads), r.uniform(-Z, Z, n_beads)


def splat(sub, xs, ys, F):
    """Bilinear (area-weighted) sub-pixel splatting: preserves flux AND true sub-pixel
    position, so a fractional-pixel shift moves a fractional amount of light (no integer
    quantization)."""
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = xs - x0; fy = ys - y0
    for dx, dy, w in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
        np.add.at(sub, (np.clip(y0 + dy, 0, N - 1), np.clip(x0 + dx, 0, N - 1)), F * w)


def signal(xs, ys, d, NA, expo):
    zs, st = C.psf_stack(round(NA, 3), Z)
    F = expo * FBEAD_REF
    bi = np.clip(np.round((d + Z) / C.DZ_STACK).astype(int), 0, len(zs) - 1)
    img = np.zeros((N, N), np.float32)
    for b in range(len(zs)):
        sel = bi == b
        if not sel.any():
            continue
        sub = np.zeros((N, N), np.float32)
        splat(sub, xs[sel], ys[sel], F)
        img += ndimage.convolve(sub, st[b], mode="wrap")
    return img


def camera(sig, seed):
    r = np.random.default_rng(seed)
    im = PED + GAIN * r.poisson(np.clip(sig / GAIN, 0, None)) + r.normal(0, READ, (N, N))
    return np.clip(np.round(im), 0, 65535).astype(np.uint16)


def ncc(a, b):
    return float(np.corrcoef(a.astype(float).ravel(), b.astype(float).ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default=os.environ.get("STAGE"),
                    help="sweep stage dir (default: $STAGE); stacks go to <stage>/images/tif_stacks")
    ap.add_argument("--outdir", default=None,
                    help="explicit output dir (overrides --stage/$STAGE)")
    a = ap.parse_args()
    if a.outdir:
        out = Path(a.outdir)
    elif a.stage:
        out = Path(a.stage) / "images" / "tif_stacks"
    else:
        raise SystemExit("set --outdir, or --stage / `source env.sh` to export STAGE")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for si, (nb, NA, expo) in enumerate(SCENARIOS):
        xs, ys, d = scene(nb, seed=si)
        frames = []
        for fi, jit in enumerate(JITTERS):
            if jit == 0:
                xj, yj = xs, ys
            else:
                jr = np.random.default_rng(1000 * si + fi)
                xj, yj = xs + jr.normal(0, jit, nb), ys + jr.normal(0, jit, nb)
            frames.append(camera(signal(xj, yj, d, NA, expo), seed=5000 * si + fi))
        stack = np.stack(frames)                        # (4, 512, 512)
        fn = f"scenario{si}_dens{nb / (N * N):.3f}_NA{NA}_expo{expo}.tif"
        tifffile.imwrite(out / fn, stack, photometric="minisblack",
                         metadata={"axes": "TYX", "jitter_px": JITTERS, "pitch_um": C.DXY,
                                   "frame0": "reference (jitter 0)"})
        for fi, jit in enumerate(JITTERS):
            rows.append(dict(scenario=si, file=fn, N=nb, density=round(nb / (N * N), 4), NA=NA, expo=expo,
                             frame=fi, jitter_px=jit, ncc_vs_frame0=round(ncc(frames[0], frames[fi]), 3)))
        print(fn, " NCC/frame:", [r["ncc_vs_frame0"] for r in rows[-4:]])

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(SCENARIOS)} four-frame stacks -> {out}")


if __name__ == "__main__":
    main()
