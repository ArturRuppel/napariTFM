#!/usr/bin/env python3
"""
Bayesian Inversion Stress Microscopy (BISM) — experimental Python port.

This is a single-file, dependency-light port of the MATLAB reference
implementation (Nier et al., Biophys. J. 110(7):1625-1635, 2016; original
BISM.m by Vincent Nier). It is intended as an EXPERIMENTAL script that the
napariTFM project can later import properly as an alternative stress-inference
backend alongside the FEM-based MSM in ``napariTFM/backend/msm.py``.

Unlike the FEM-based MSM, BISM:
  * needs NO material parameters (no Young's modulus / Poisson ratio),
  * works on a regular rectangular grid (no meshing),
  * is a single sparse linear solve of a Bayesian inverse problem, and
  * yields per-pixel posterior stress uncertainty.

Method (forward model A*sigma = T, the discretized divergence operator):
    sigma_MAP = (lambda*B + l^2 A^T A)^{-1} (l^2 A^T T)
where B is the prior covariance (stress-norm regularization + shear-symmetry
term + optional free-stress boundary conditions).

The public entry point is :func:`compute_bism_stress`. Running the module as a
script executes three validation stages:
  1. Reproduce the MATLAB reference result on its bundled data
     (Traction_field.mat -> Stress_field.mat ground truth).
  2. Run BISM through this project's two MSM benchmarks (file-based + square
     plate) and compare against the FEM-based napariTFM MSM.

License note: BISM is GPLv3, matching napariTFM (pyTFM-derived).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Single source of truth: the validated BISM core now lives in the package.
# This script remains a standalone, manually-run validation harness around it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from napariTFM.backend.bism import (  # noqa: E402
    BISMResult,
    compute_bism_stress,
)



# --------------------------------------------------------------------------- #
# Validation harness (run as a script)
# --------------------------------------------------------------------------- #
def _metrics(calc, gt, mask=None):
    """Pearson r, best-fit slope (calc ~ slope*gt) and signed mean rel. error."""
    if mask is None:
        mask = np.ones_like(calc, dtype=bool)
    m = mask & ~np.isnan(calc) & ~np.isnan(gt)
    c, g = calc[m], gt[m]
    if c.size < 2 or np.std(c) == 0 or np.std(g) == 0:
        return dict(r=np.nan, slope=np.nan, mre=np.nan, n=int(c.size))
    r = np.corrcoef(c, g)[0, 1]
    slope = np.dot(g, c) / np.dot(g, g)          # least-squares through origin
    thr = 0.01 * np.max(np.abs(g))
    sig = np.abs(g) > thr
    mre = np.mean((c[sig] - g[sig]) / np.abs(g[sig])) if np.any(sig) else np.nan
    return dict(r=float(r), slope=float(slope), mre=float(mre), n=int(c.size))


def _print_metrics(title, m):
    print(f"  {title:<10}  r={m['r']:+.3f}  slope={m['slope']:+.3f}  "
          f"MRE={m['mre']:+.1%}  (n={m['n']})")


def stage1_matlab_reference(bism_dir: Path, out_dir: Path):
    """Reproduce the MATLAB BISM validation on its bundled data."""
    import scipy.io as sio
    print("=" * 64)
    print("STAGE 1 — MATLAB reference data (Traction_field.mat)")
    print("=" * 64)

    tr = sio.loadmat(bism_dir / "Traction_field.mat")["traction"]["frame1"][0, 0]
    tx, ty = tr["tx"][0, 0], tr["ty"][0, 0]
    R, C = tx.shape
    l = 98.0 / (C - 1)        # x = 1..99 over C points, as in BISM.m (coeff=1)

    print(f"  Grid: {R}x{C}, l={l:.4f}  (BC=free, lambda=1e-6, noise=1e-3)")
    res = compute_bism_stress(tx, ty, l=l, lam=1e-6, free_bc=True,
                              noise_value=1e-3, return_uncertainty=True)

    st = sio.loadmat(bism_dir / "Stress_field.mat")["stress"]["frame1"][0, 0]
    gt_xx, gt_yy, gt_xy = st["sxx"][0, 0], st["syy"][0, 0], st["sxy"][0, 0]

    print(f"  Traction-reconstruction R^2 = {res.r2_traction:.4f}  (want ~1)")
    print("  Inferred-vs-true stress (paper-style validation):")
    _print_metrics("sigma_xx", _metrics(res.sxx, gt_xx))
    _print_metrics("sigma_yy", _metrics(res.syy, gt_yy))
    _print_metrics("sigma_xy", _metrics(res.sxy, gt_xy))
    if res.error_sxx is not None:
        print(f"  Posterior std (mean):  sxx={res.error_sxx.mean():.3g}  "
              f"syy={res.error_syy.mean():.3g}  sxy={res.error_sxy.mean():.3g}")

    _save_scatter(out_dir / "bism_stage1_matlab_scatter.png",
                  [(gt_xx, res.sxx, "sigma_xx"),
                   (gt_yy, res.syy, "sigma_yy"),
                   (gt_xy, res.sxy, "sigma_xy")],
                  "BISM (Python) vs ground truth — MATLAB reference data")


def _run_msm(tx, ty, mask, params):
    """Run the FEM-based napariTFM MSM with an explicit mask; return (sxx, syy) in mN/m."""
    import validate_MSM as V
    from napariTFM.backend.msm import calculate_stresses
    force_field = np.stack([np.nan_to_num(tx), np.nan_to_num(ty)], axis=-1)
    gen = calculate_stresses(force_field, mask[np.newaxis, ...], params)
    result = None
    try:
        for intermediate, _f, _t in gen:
            result = intermediate
    except StopIteration as e:
        result = e.value
    st = result.stress_tensor[0]
    return st[:, :, 0, 0], st[:, :, 1, 1]


def stage2_project_benchmarks(out_dir: Path):
    """Run BISM through the project's two MSM benchmarks and compare with MSM."""
    import validate_MSM as V
    print("\n" + "=" * 64)
    print("STAGE 2 — napariTFM MSM benchmarks (BISM vs MSM vs ground truth)")
    print("=" * 64)

    # ---- 2a: file-based benchmark -------------------------------------- #
    print("\n[2a] File-based benchmark")
    bdir = Path(V.__file__).parent
    params = V.get_msm_parameters()                      # pixel_size = 0.3 um
    tx, ty = V.load_ground_truth_traction(str(bdir))
    gt_xx, gt_yy = V.load_ground_truth_stress(str(bdir))
    gt_xx, gt_yy = gt_xx * 1000.0, gt_yy * 1000.0        # N/m -> mN/m (as in validate_MSM)
    mask = ~np.isnan(gt_xx) & ~np.isnan(gt_yy)

    # Masked BISM: l = grid spacing (um); convert Pa*um -> mN/m with *1e-3
    res = compute_bism_stress(tx, ty, l=params.pixel_size, mask=mask)
    conv = 1e-3
    b_xx, b_yy = res.sxx * conv, res.syy * conv
    m_xx, m_yy, _, cond, resid = V.calculate_msm_stress(tx, ty, params)

    print(f"  BISM traction-reconstruction R^2 = {res.r2_traction:.4f}")
    print("  BISM vs ground truth:")
    _print_metrics("sigma_xx", _metrics(b_xx, gt_xx, mask))
    _print_metrics("sigma_yy", _metrics(b_yy, gt_yy, mask))
    print("  MSM  vs ground truth:")
    _print_metrics("sigma_xx", _metrics(m_xx, gt_xx, mask))
    _print_metrics("sigma_yy", _metrics(m_yy, gt_yy, mask))
    print("  BISM vs MSM (agreement):")
    _print_metrics("sigma_xx", _metrics(b_xx, m_xx, mask))
    _print_metrics("sigma_yy", _metrics(b_yy, m_yy, mask))

    # ---- 2b: square-plate benchmark ----------------------------------- #
    print("\n[2b] Square-plate benchmark (uniform analytical stress)")
    from napariTFM.backend.parameter_dataclasses import MSMParameters
    sp_params = MSMParameters(
        density_factor=0.01, mesh_algorithm="Frontal-Del.", use_optimization=False,
        poisson_ratio_cells=0.5, young_modulus=1000.0,
        pixel_size=1.0, downscale_factor=1,
    )
    tx, ty, pmask, tscale = V.create_square_plate_problem(
        size=50, edge_traction=1000, buffer=5)
    gt_xx, gt_yy, _ = V.calculate_square_plate_analytical_stress(
        tscale, pmask, sp_params)

    res = compute_bism_stress(tx, ty, l=sp_params.pixel_size, mask=pmask)
    # Masked BISM returns NaN outside the plate; zero it so the full-domain
    # correlation captures the 0 -> uniform border transition (MSM does this too).
    b_xx = np.nan_to_num(res.sxx) * 1e-3 * pmask
    b_yy = np.nan_to_num(res.syy) * 1e-3 * pmask
    m_xx, m_yy = _run_msm(tx, ty, pmask, sp_params)

    print(f"  Analytical uniform stress = {gt_xx[pmask].mean():.4f} mN/m")
    print(f"  BISM mean inside plate:  sxx={b_xx[pmask].mean():.4f}  "
          f"syy={b_yy[pmask].mean():.4f} mN/m")
    print(f"  MSM  mean inside plate:  sxx={m_xx[pmask].mean():.4f}  "
          f"syy={m_yy[pmask].mean():.4f} mN/m")
    print("  BISM vs analytical (full domain, incl. border):")
    _print_metrics("sigma_xx", _metrics(b_xx, gt_xx))
    _print_metrics("sigma_yy", _metrics(b_yy, gt_yy))
    print("  MSM  vs analytical (full domain, incl. border):")
    _print_metrics("sigma_xx", _metrics(m_xx, gt_xx))
    _print_metrics("sigma_yy", _metrics(m_yy, gt_yy))


def _save_scatter(path, triples, suptitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(triples), figsize=(4 * len(triples), 4))
    for ax, (gt, calc, name) in zip(np.atleast_1d(axes), triples):
        g, c = gt.ravel(), calc.ravel()
        ax.plot(g, c, "b+", ms=2, alpha=0.4)
        lo, hi = np.nanmin(g), np.nanmax(g)
        ax.plot([lo, hi], [lo, hi], "r", lw=1)
        ax.set_xlabel(f"{name} (true)")
        ax.set_ylabel(f"{name} (BISM)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved: {path}")


def main():
    here = Path(__file__).resolve().parent
    # Make napariTFM importable and allow `import validate_MSM`.
    import sys
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent.parent))
    bism_dir = Path("/home/aruppel/Projects/BISM")

    stage1_matlab_reference(bism_dir, here)
    stage2_project_benchmarks(here)
    print("\n" + "=" * 64)
    print("BISM VALIDATION COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
