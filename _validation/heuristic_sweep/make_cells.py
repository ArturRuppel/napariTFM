#!/usr/bin/env python3
"""Stage 0 (local): synthesize the realistic-cell scene grid from benchmarkTFM.

Same pipeline as the dipole grid (make_scenes.py), one thing swapped: the GT is
no longer a single analytic dipole but a whole cell -- a benchmarkTFM synth cell,
a real cell outline with 16-82 fitted contractile stress fibres. We take THAT
cell's traction field as the GT shape and:
  * forward-project it to a displacement field u with THIS pipeline's Green's
    operator (validated to reproduce benchmarkTFM's own u to cos > 0.999), so u
    and t_gt are a consistent forward/inverse pair -- no cross-operator bias floor,
  * scale (t_gt, u) so peak |u| hits each strength target (px),
  * warp the SAME best-imaging bead stack the dipole run used (scenario 6): frame 0
    is the reference, deformed = warp(frame 1) so mild registration jitter + photon
    noise ride along exactly as in the dipole run.
The cell's scaled traction is stored as gt_traction.npy (the authoritative GT the
scorer reads); the cell OUTLINE mask is stored as cell_mask.npy (the honest,
user-available segmentation prior -- looser than the traction support, which
concentrates at the periphery -- so cell_confinement.py can test whether mask
confinement earns its keep without cheating with the GT support); scene.toml
carries only metadata (kind="cell", strength, provenance).

Same E and pixel size as the dipole run (sweep_config), so the only variable
versus that run is the field structure: one localized dipole -> a diffuse cell.

Writes $STAGE/scenes/<condition>/<scene_id>/{reference.tif,deformed.tif,
gt_traction.npy,cell_mask.npy,scene.toml}, scene_id = "<cell>_u<peakpx>".

Usage:
    python make_cells.py --scenarios-dir ~/Projects/benchmarkTFM/benchmarks/scenarios \
                         --images-dir "$STAGE/images/tif_stacks" --stage "$STAGE"
                         [--cells synth00,synth01 --strengths 3.155,7.924]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import tifffile

import sweep_config as C
from make_scenes import greens_displacement, warp, centre_crop


def load_cell_geometry(scen_dir, cell):
    """(2,N,N) GT traction (Pa), (N,N) cell-outline mask, and fibre count for one
    benchmarkTFM synth cell. traction and mask share the same 512² grid (co-registered)."""
    ome = os.path.join(scen_dir, cell, f"{cell}.ome.tif")
    with tifffile.TiffFile(ome) as t:
        series = {s.name: s.asarray() for s in t.series}
    t_gt = np.asarray(series["traction"], np.float64)          # (2,H,W) Pa
    mask = np.asarray(series["mask"]) > 0                       # (H,W) cell outline (not the traction support)
    cfg = json.load(open(os.path.join(scen_dir, cell, "config.json")))
    return t_gt, mask, len(cfg["fibres"])


def find_stack(images_dir, scenario):
    hits = sorted(glob.glob(os.path.join(images_dir, f"scenario{scenario}_*.tif")))
    if not hits:
        raise SystemExit(f"no scenario{scenario}_*.tif in {images_dir}")
    return hits[0]


def write_toml(path, meta, sub):
    with open(path, "w") as f:
        f.write(f'[meta]\ncondition = "{meta["condition"]}"\nscene_id = "{meta["scene_id"]}"\n'
                f'image_size = {meta["image_size"]}\npixel_size = {meta["pixel_size"]}\n'
                f'kind = "cell"\nsource_cell = "{meta["source_cell"]}"\n'
                f'n_fibers = {meta["n_fibers"]}\npeak_disp_px = {meta["peak_disp_px"]}\n'
                f'rms_traction_pa = {meta["rms_traction_pa"]}\n\n'
                f'[substrate]\nyoung_modulus = {sub["young_modulus"]}\npoisson = {sub["poisson"]}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-dir", required=True, help="benchmarkTFM benchmarks/scenarios dir")
    ap.add_argument("--images-dir", required=True, help="dir of synthetic scenario*.tif stacks")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cells", default=None, help="comma-sep cell ids (default sweep_config)")
    ap.add_argument("--strengths", default=None, help="comma-sep target peak |u| px (default sweep_config)")
    a = ap.parse_args()

    cells = a.cells.split(",") if a.cells else list(C.CELL_SOURCE_CELLS)
    strengths = [float(x) for x in a.strengths.split(",")] if a.strengths else list(C.CELL_STRENGTHS_PX)
    N = C.GT_REFERENCE_SIZE
    ps = C.PIXEL_SIZE_UM
    cond = C.CELL_CONDITION

    stack = tifffile.imread(find_stack(a.images_dir, C.CELL_STACK_SCENARIO))   # (T,H,W)
    ref = centre_crop(stack[C.CELL_REF_FRAME], N).astype(np.float64)           # zero-jitter reference
    src = centre_crop(stack[C.CELL_DEFORM_FRAME], N).astype(np.float64)        # jittered deform-source
    print(f"{len(cells)} cells x {len(strengths)} strengths = {len(cells)*len(strengths)} scenes "
          f"-> condition {cond} (scenario {C.CELL_STACK_SCENARIO}, frames "
          f"{C.CELL_REF_FRAME}->{C.CELL_DEFORM_FRAME})")

    for cell in cells:
        t_gt, cell_mask, n_fib = load_cell_geometry(a.scenarios_dir, cell)
        u_unit = greens_displacement(t_gt, N)                  # µm, this pipeline's operator
        peak_unit = float(np.hypot(u_unit[0], u_unit[1]).max())   # µm at unit (native) scale
        for P in strengths:
            scale = (P * ps) / peak_unit                       # hit peak |u| = P px
            t_s = (t_gt * scale).astype(np.float32)
            u = u_unit * scale                                 # µm
            deformed = warp(src, u, ps)
            sid = f"{cell}_u{P:g}"
            d = os.path.join(a.stage, "scenes", cond, sid)
            os.makedirs(d, exist_ok=True)
            mag = np.hypot(t_s[0], t_s[1])
            rms = float(np.sqrt((mag[mag > 0.05 * mag.max()] ** 2).mean()))
            tifffile.imwrite(os.path.join(d, "reference.tif"),
                             np.clip(ref, 0, 65535).astype(np.uint16))
            tifffile.imwrite(os.path.join(d, "deformed.tif"),
                             np.clip(deformed, 0, 65535).astype(np.uint16))
            np.save(os.path.join(d, "gt_traction.npy"), t_s)
            np.save(os.path.join(d, "cell_mask.npy"), cell_mask)   # cell outline (strength-independent)
            write_toml(os.path.join(d, "scene.toml"),
                       dict(condition=cond, scene_id=sid, image_size=N, pixel_size=ps,
                            source_cell=cell, n_fibers=n_fib, peak_disp_px=P,
                            rms_traction_pa=round(rms, 3)),
                       dict(young_modulus=C.YOUNG_MODULUS, poisson=C.POISSON))
            print(f"  {sid}: peak|u|={P:g}px  rms_t={rms:.0f}Pa  ({n_fib} fibres)", flush=True)


if __name__ == "__main__":
    main()
