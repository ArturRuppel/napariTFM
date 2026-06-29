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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu


# --------------------------------------------------------------------------- #
# Core BISM algorithm
# --------------------------------------------------------------------------- #
@dataclass
class BISMResult:
    """Inferred stress field and diagnostics from BISM.

    Stress components are (R, C) arrays in units of [traction] * [l].
    With traction in Pa and ``l`` in micrometres, that is Pa*um (= 1e-3 mN/m).
    """
    sxx: np.ndarray
    syy: np.ndarray
    sxy: np.ndarray
    lam: float
    r2_traction: float
    error_sxx: Optional[np.ndarray] = None
    error_syy: Optional[np.ndarray] = None
    error_sxy: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


def _build_A(R: int, C: int, l: float) -> sp.csr_matrix:
    """Discretized divergence operator A such that A @ sigma = T (2N x Ninf)."""
    # Forward-difference stencils on the staggered grid.
    Dx = sp.diags([-np.ones(C), np.ones(C)], [0, 1], shape=(C, C + 1))   # C x (C+1)
    Dy = sp.diags([-np.ones(R), np.ones(R)], [0, 1], shape=(R, R + 1))   # R x (R+1)
    Ax = sp.kron(sp.eye(R), Dx)            # N x (N+R), d/dx on a R x (C+1) grid
    Ay = sp.kron(Dy, sp.eye(C))            # N x (N+C), d/dy on a (R+1) x C grid

    # Component order in sigma: [sxx(N+R), syy(N+C), sxy(N+C), syx(N+R)]
    # Tx eqn: d(sxx)/dx + d(sxy)/dy = Tx
    # Ty eqn: d(syy)/dy + d(syx)/dx = Ty
    A = sp.bmat([
        [Ax,   None, Ay,   None],
        [None, Ay,   None, Ax],
    ], format="csr") / l
    return A


def _build_B(R: int, C: int, free_bc: bool,
             alpha_xy: float, alpha_bc: float) -> sp.csr_matrix:
    """Prior covariance B (Ninf x Ninf): identity + shear-symmetry + free-BC."""
    N = R * C
    Ninf = 4 * N + 2 * (C + R)

    # (1) stress-norm (L2) regularization
    B = sp.eye(Ninf, format="csr")

    # (2) shear-symmetry: enforce interpolated sxy == syx
    Sy = sp.diags([np.ones(R), np.ones(R)], [0, 1], shape=(R, R + 1))    # sum stencil
    Sx = sp.diags([-np.ones(C), -np.ones(C)], [0, 1], shape=(C, C + 1))  # neg sum stencil
    bd1 = sp.kron(Sy, sp.eye(C))            # N x (N+C), averages sxy
    bd2 = sp.kron(sp.eye(R), Sx)            # N x (N+R), -averages syx
    # Leading zero columns cover the sxx (N+R) and syy (N+C) blocks.
    pad = sp.csr_matrix((N, (N + R) + (N + C)))
    Bdiff = sp.hstack([pad, bd1, bd2], format="csr")   # N x Ninf
    B = B + (alpha_xy ** 2) * (Bdiff.T @ Bdiff)

    # (3) free-stress boundary conditions: normal stress = 0 on the domain edge
    if free_bc:
        # Bbcy: picks left & right edge values on a R x (C+1) grid  (sxx, syx)
        rows_y, cols_y = [], []
        for i in range(R):
            rows_y.append(i);     cols_y.append(i * (C + 1) + 0)      # left edge
            rows_y.append(R + i); cols_y.append(i * (C + 1) + C)      # right edge
        Bbcy = sp.csr_matrix((np.ones(len(rows_y)), (rows_y, cols_y)),
                             shape=(2 * R, N + R))
        # Bbcx: picks top & bottom edge values on a (R+1) x C grid   (syy, sxy)
        rows_x, cols_x = [], []
        for j in range(C):
            rows_x.append(j);     cols_x.append(0 * C + j)            # top edge
            rows_x.append(C + j); cols_x.append(R * C + j)            # bottom edge
        Bbcx = sp.csr_matrix((np.ones(len(rows_x)), (rows_x, cols_x)),
                             shape=(2 * C, N + C))
        # Block-diagonal over [sxx, syy, sxy, syx]
        Bbc = sp.block_diag([Bbcy, Bbcx, Bbcx, Bbcy], format="csr")
        B = B + (alpha_bc ** 2) * (Bbc.T @ Bbc)

    return B.tocsr()


def _interp_to_grid(sigma: np.ndarray, R: int, C: int):
    """Average the four staggered stress fields onto the R x C data grid."""
    N = R * C
    sxx_v = sigma[0:N + R]
    syy_v = sigma[N + R:2 * N + R + C]
    sxy_v = sigma[2 * N + R + C:3 * N + R + 2 * C]
    syx_v = sigma[3 * N + R + 2 * C:4 * N + 2 * R + 2 * C]

    g_xx = sxx_v.reshape(R, C + 1)
    g_yy = syy_v.reshape(R + 1, C)
    g_xy = sxy_v.reshape(R + 1, C)
    g_yx = syx_v.reshape(R, C + 1)

    sxx = 0.5 * (g_xx[:, :-1] + g_xx[:, 1:])      # horizontal average
    syy = 0.5 * (g_yy[:-1, :] + g_yy[1:, :])      # vertical average
    sxy = 0.5 * (g_xy[:-1, :] + g_xy[1:, :])
    syx = 0.5 * (g_yx[:, :-1] + g_yx[:, 1:])
    sxy = 0.5 * (sxy + syx)                        # symmetrize
    return sxx, syy, sxy


def compute_bism_stress(
    tx: np.ndarray,
    ty: np.ndarray,
    l: float = 1.0,
    lam: float = 1e-6,
    alpha_xy: float = 1e3,
    alpha_bc: float = 1e3,
    free_bc: bool = True,
    noise_value: float = 0.0,
    return_uncertainty: bool = False,
    mask: Optional[np.ndarray] = None,
) -> BISMResult:
    """Infer the internal stress tensor from a traction field via BISM.

    Args:
        tx, ty: (R, C) traction components (Pa). NaNs are treated as 0.
        l:      grid spacing (micrometres). Sets the output stress units.
        lam:    regularization hyperparameter Lambda (fixed-value method).
        alpha_xy:  shear-symmetry hyperparameter (large).
        alpha_bc:  free-BC hyperparameter (large).
        free_bc:   impose free-stress boundary conditions in the prior.
        noise_value: traction noise amplitude (sets posterior variance scale).
        return_uncertainty: also compute per-pixel stress std (small grids only).
        mask:   optional boolean (R, C). When given, the divergence operator and
                free-stress boundary conditions are restricted to the masked
                region and its actual contour (masked BISM) — the correct
                formulation for a monolayer that does not fill the field. The
                full rectangular formulation is used when mask is None.

    Returns:
        BISMResult with sxx, syy, sxy in units of [traction]*[l] (Pa*um).
        With a mask, pixels outside the mask are NaN.
    """
    if mask is not None:
        return _compute_bism_masked(tx, ty, mask, l, lam, alpha_xy, alpha_bc)

    tx = np.nan_to_num(np.asarray(tx, dtype=float), nan=0.0)
    ty = np.nan_to_num(np.asarray(ty, dtype=float), nan=0.0)
    R, C = tx.shape
    N = R * C
    Ninf = 4 * N + 2 * (C + R)

    A = _build_A(R, C, l)
    B = _build_B(R, C, free_bc, alpha_xy, alpha_bc)

    vTx = tx.ravel()          # row-major == MATLAB reshape(mTx', N, 1)
    vTy = ty.ravel()
    T = np.concatenate([vTx, vTy])

    AtA = (A.T @ A).tocsc()
    M = (lam * B + (l ** 2) * AtA).tocsc()
    rhs = (l ** 2) * (A.T @ T)
    sigma = spsolve(M, rhs)

    sxx, syy, sxy = _interp_to_grid(sigma, R, C)

    # Traction-reconstruction R^2 (should be close to 1)
    T_inf = A @ sigma
    Tx_inf, Ty_inf = T_inf[:N], T_inf[N:]
    def _r2(obs, pred):
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2 = 0.5 * (_r2(vTx, Tx_inf) + _r2(vTy, Ty_inf))

    result = BISMResult(
        sxx=sxx, syy=syy, sxy=sxy, lam=lam, r2_traction=r2,
        meta={"R": R, "C": C, "l": l, "free_bc": free_bc,
              "alpha_xy": alpha_xy, "alpha_bc": alpha_bc, "Ninf": Ninf},
    )

    if return_uncertainty and noise_value > 0:
        lu = splu(M)
        diagS = np.empty(Ninf)
        chunk = 512
        for start in range(0, Ninf, chunk):
            cols = np.arange(start, min(start + chunk, Ninf))
            E = np.zeros((Ninf, len(cols)))
            E[cols, np.arange(len(cols))] = 1.0
            X = lu.solve(E)
            diagS[cols] = X[cols, np.arange(len(cols))]
        Spost = (noise_value ** 2) * (l ** 2) * diagS

        S_xx = Spost[0:N + R].reshape(R, C + 1)
        S_yy = Spost[N + R:2 * N + R + C].reshape(R + 1, C)
        S_xy = Spost[2 * N + R + C:3 * N + R + 2 * C].reshape(R + 1, C)
        S_yx = Spost[3 * N + R + 2 * C:4 * N + 2 * R + 2 * C].reshape(R, C + 1)
        e_xx = 0.25 * (S_xx[:, :-1] + S_xx[:, 1:])
        e_yy = 0.25 * (S_yy[:-1, :] + S_yy[1:, :])
        e_xy = 0.25 * (S_xy[:-1, :] + S_xy[1:, :])
        e_yx = 0.25 * (S_yx[:, :-1] + S_yx[:, 1:])
        result.error_sxx = np.sqrt(e_xx)
        result.error_syy = np.sqrt(e_yy)
        result.error_sxy = np.sqrt(0.25 * (e_xy + e_yx))

    return result


def _compute_bism_masked(tx, ty, mask, l, lam, alpha_xy, alpha_bc) -> BISMResult:
    """Masked BISM: divergence operator and free-BC restricted to the mask.

    Stress lives on cell faces (a staggered grid):
      * x-faces (vertical, between horizontal neighbours) carry sigma_xx, sigma_yx
      * y-faces (horizontal, between vertical neighbours) carry sigma_yy, sigma_xy
    Only faces touching an in-mask cell are unknowns; a face touching exactly one
    in-mask cell is a free boundary (its normal traction is penalized to zero).
    """
    tx = np.nan_to_num(np.asarray(tx, dtype=float), nan=0.0)
    ty = np.nan_to_num(np.asarray(ty, dtype=float), nan=0.0)
    mask = np.asarray(mask, dtype=bool)
    R, C = mask.shape

    # ---- enumerate cells and faces ------------------------------------- #
    ii, jj = np.where(mask)                       # in-mask cell coordinates
    ncell = ii.size

    # x-faces: shape (R, C+1). Face (i,k) borders cells (i,k-1) and (i,k).
    x_active = np.zeros((R, C + 1), bool)
    x_active[:, :C] |= mask                       # right neighbour = cell (i,k)
    x_active[:, 1:] |= mask                       # left  neighbour = cell (i,k-1)
    x_left = np.zeros((R, C + 1), bool); x_left[:, 1:] = mask
    x_right = np.zeros((R, C + 1), bool); x_right[:, :C] = mask
    x_bound = x_active & (x_left ^ x_right)        # exactly one neighbour in mask
    x_id = np.full((R, C + 1), -1, int)
    x_id[x_active] = np.arange(int(x_active.sum()))
    nx = int(x_active.sum())

    # y-faces: shape (R+1, C). Face (m,j) borders cells (m-1,j) and (m,j).
    y_active = np.zeros((R + 1, C), bool)
    y_active[:R, :] |= mask                        # bottom neighbour = cell (m,j)
    y_active[1:, :] |= mask                        # top    neighbour = cell (m-1,j)
    y_top = np.zeros((R + 1, C), bool); y_top[1:, :] = mask
    y_bot = np.zeros((R + 1, C), bool); y_bot[:R, :] = mask
    y_bound = y_active & (y_top ^ y_bot)
    y_id = np.full((R + 1, C), -1, int)
    y_id[y_active] = np.arange(int(y_active.sum()))
    ny = int(y_active.sum())

    # unknown layout: [sigma_xx(nx), sigma_yy(ny), sigma_xy(ny), sigma_yx(nx)]
    o_xx, o_yy, o_xy, o_yx = 0, nx, nx + ny, nx + 2 * ny
    ninf = 2 * nx + 2 * ny

    # face ids for each in-mask cell
    lx = x_id[ii, jj]            # left  x-face
    rx = x_id[ii, jj + 1]        # right x-face
    ty_ = y_id[ii, jj]           # top   y-face
    by = y_id[ii + 1, jj]        # bottom y-face

    # ---- divergence operator A (A @ sigma = T) ------------------------- #
    c = np.arange(ncell)
    rows = np.concatenate([
        c, c, c, c,                               # Tx eqn
        ncell + c, ncell + c, ncell + c, ncell + c,   # Ty eqn
    ])
    cols = np.concatenate([
        o_xx + rx, o_xx + lx, o_xy + by, o_xy + ty_,
        o_yy + by, o_yy + ty_, o_yx + rx, o_yx + lx,
    ])
    vals = np.concatenate([
        np.ones(ncell), -np.ones(ncell), np.ones(ncell), -np.ones(ncell),
        np.ones(ncell), -np.ones(ncell), np.ones(ncell), -np.ones(ncell),
    ])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * ncell, ninf)) / l
    T = np.concatenate([tx[mask], ty[mask]])

    # ---- prior B --------------------------------------------------------- #
    B = sp.eye(ninf, format="csr")

    # shear symmetry: interp(sigma_xy) == interp(sigma_yx) at each cell
    drow = np.concatenate([c, c, c, c])
    dcol = np.concatenate([o_xy + ty_, o_xy + by, o_yx + lx, o_yx + rx])
    dval = np.concatenate([0.5 * np.ones(ncell), 0.5 * np.ones(ncell),
                           -0.5 * np.ones(ncell), -0.5 * np.ones(ncell)])
    Bdiff = sp.csr_matrix((dval, (drow, dcol)), shape=(ncell, ninf))
    B = B + (alpha_xy ** 2) * (Bdiff.T @ Bdiff)

    # free-stress boundary conditions on the mask contour
    xb = x_id[x_bound]          # boundary x-faces -> sigma_xx = sigma_yx = 0
    yb = y_id[y_bound]          # boundary y-faces -> sigma_yy = sigma_xy = 0
    bc_cols = np.concatenate([o_xx + xb, o_yx + xb, o_yy + yb, o_xy + yb])
    nb = bc_cols.size
    Bbc = sp.csr_matrix((np.ones(nb), (np.arange(nb), bc_cols)), shape=(nb, ninf))
    B = B + (alpha_bc ** 2) * (Bbc.T @ Bbc)

    # ---- solve ----------------------------------------------------------- #
    M = (lam * B + (l ** 2) * (A.T @ A)).tocsc()
    rhs = (l ** 2) * (A.T @ T)
    sigma = spsolve(M, rhs)

    # ---- interpolate faces -> cells ------------------------------------- #
    sxx = np.full((R, C), np.nan)
    syy = np.full((R, C), np.nan)
    sxy = np.full((R, C), np.nan)
    sxx[ii, jj] = 0.5 * (sigma[o_xx + lx] + sigma[o_xx + rx])
    syy[ii, jj] = 0.5 * (sigma[o_yy + ty_] + sigma[o_yy + by])
    s_xy_c = 0.5 * (sigma[o_xy + ty_] + sigma[o_xy + by])
    s_yx_c = 0.5 * (sigma[o_yx + lx] + sigma[o_yx + rx])
    sxy[ii, jj] = 0.5 * (s_xy_c + s_yx_c)

    T_inf = A @ sigma
    obs_x, obs_y = T[:ncell], T[ncell:]
    def _r2(obs, pred):
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        return 1.0 - np.sum((obs - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan
    r2 = 0.5 * (_r2(obs_x, T_inf[:ncell]) + _r2(obs_y, T_inf[ncell:]))

    return BISMResult(
        sxx=sxx, syy=syy, sxy=sxy, lam=lam, r2_traction=r2,
        meta={"R": R, "C": C, "l": l, "masked": True, "ncell": ncell,
              "nx": nx, "ny": ny, "Ninf": ninf},
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
