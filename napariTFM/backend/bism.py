"""Bayesian Inversion Stress Microscopy (BISM) stress engine.

A dependency-light port of the MATLAB reference implementation (Nier et al.,
Biophys. J. 110(7):1625-1635, 2016; original BISM.m by Vincent Nier).

BISM:
  * needs NO material parameters (no Young's modulus / Poisson ratio),
  * works on a regular rectangular grid (no meshing),
  * is a single sparse linear solve of a Bayesian inverse problem, and
  * yields per-pixel posterior stress uncertainty.

Forward model ``A @ sigma = T`` (the discretized divergence operator):
    sigma_MAP = (lambda*B + l^2 A^T A)^{-1} (l^2 A^T T)
where B is the prior covariance (stress-norm regularization + shear-symmetry
term + optional free-stress boundary conditions). Lambda is a fixed,
user-supplied regularization hyperparameter.

:func:`compute_bism_stress` is the per-frame entry point;
:func:`calculate_bism_stresses` is the stage generator.

License note: BISM is GPLv3, matching napariTFM (pyTFM-derived).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from skimage.transform import resize

from napariTFM.backend.parameter_dataclasses import StressParameters
from napariTFM.backend.stress import StressResult


# --------------------------------------------------------------------------- #
# Core BISM algorithm
# --------------------------------------------------------------------------- #
@dataclass
class BISMResult:
    """Inferred stress field and diagnostics from a single BISM solve.

    Stress components are (R, C) arrays in units of [traction] * [l].
    With traction in Pa and ``l`` in micrometres, that is Pa*um (= 1e-3 mN/m).
    """
    sxx: np.ndarray
    syy: np.ndarray
    sxy: np.ndarray
    lam: float
    r2_traction: float
    meta: dict = field(default_factory=dict)


def compute_bism_stress(
    tx: np.ndarray,
    ty: np.ndarray,
    l: float = 1.0,
    lam: float = 1e-6,
    alpha_xy: float = 1e3,
    alpha_bc: float = 1e3,
    mask: Optional[np.ndarray] = None,
) -> BISMResult:
    """Infer the internal stress tensor from a traction field via masked BISM.

    Args:
        tx, ty: (R, C) traction components (Pa). NaNs are treated as 0.
        l:      grid spacing (micrometres). Sets the output stress units.
        lam:    regularization hyperparameter Lambda (fixed value).
        alpha_xy:  shear-symmetry hyperparameter (large).
        alpha_bc:  free-BC hyperparameter (large).
        mask:   boolean (R, C). The divergence operator and free-stress boundary
                conditions are restricted to the masked region and its actual
                contour — the correct formulation for a monolayer that does not
                fill the field. A mask is required (a TFM stress solve is always
                over an externally supplied cell/monolayer mask, ROADMAP §2).

    Returns:
        BISMResult with sxx, syy, sxy in units of [traction]*[l] (Pa*um).
        Pixels outside the mask are NaN. ``result.lam`` is the Lambda used.
    """
    if mask is None:
        raise ValueError(
            "compute_bism_stress requires a mask — the rectangular (unmasked) "
            "formulation was removed as unused (napariTFM always solves over an "
            "external mask)."
        )
    return _compute_bism_masked(tx, ty, mask, l, lam, alpha_xy, alpha_bc)


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
    AtA = (A.T @ A).tocsc()
    M = (lam * B + (l ** 2) * AtA).tocsc()
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
# Stage generator
# --------------------------------------------------------------------------- #
# BISM stress is returned in Pa*um (= 1e-3 mN/m); the rest of the pipeline works
# in mN/m, so every solved frame is scaled by this factor.
_BISM_TO_MN_PER_M = 1e-3


def _create_bism_physical_scale(params: StressParameters) -> dict:
    """Physical-scale dict for the stress stage."""
    return {
        'pixel_size': params.pixel_size,
        'grid_spacing': params.pixel_size * params.downscale_factor,
        'time_interval': params.frame_interval,
        'stress_units': 'mN/m',
        'grid_spacing_units': 'µm',
        'time_interval_units': 'min',
    }


def calculate_bism_stresses(
    force_field: np.ndarray,
    masks: np.ndarray,
    params: StressParameters,
) -> Generator[Tuple[StressResult, int, int], None, StressResult]:
    """Infer stress per frame with BISM.

    Yields ``(StressResult, frame_1based, total)`` per frame and returns the
    final cumulative :class:`StressResult`. No mesh is generated — BISM solves
    on the regular grid.
    """
    if force_field.ndim == 3:
        force_field = force_field[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    total_frames = force_field.shape[0]
    stress_shape = (*force_field.shape[1:3], 2, 2)
    stress_stack = np.zeros((total_frames, *stress_shape), dtype=np.float32)
    physical_scale = _create_bism_physical_scale(params)
    grid_spacing = params.pixel_size * params.downscale_factor  # µm per grid step
    r2_values = []

    def process_frame(frame_idx):
        tx = force_field[frame_idx, ..., 0]
        ty = force_field[frame_idx, ..., 1]
        current_mask = masks[frame_idx]

        if current_mask.shape != tx.shape:
            current_mask = resize(
                current_mask.astype(float), tx.shape,
                order=0, preserve_range=True, anti_aliasing=False,
            ) > 0.5
        current_mask = current_mask.astype(bool)

        # An empty mask has nothing to solve; leave the frame at zero stress.
        if not current_mask.any():
            zeros = np.zeros(tx.shape, dtype=np.float32)
            return zeros, zeros, zeros, np.nan

        res = compute_bism_stress(
            tx, ty, l=grid_spacing, lam=params.bism_regularization,
            mask=current_mask,
        )
        # Pa*um -> mN/m. Off-mask pixels are NaN from the masked solve; zero
        # them out so downstream maps are clean.
        sxx = np.nan_to_num(res.sxx) * _BISM_TO_MN_PER_M
        syy = np.nan_to_num(res.syy) * _BISM_TO_MN_PER_M
        sxy = np.nan_to_num(res.sxy) * _BISM_TO_MN_PER_M
        return sxx, syy, sxy, res.r2_traction

    for frame in range(total_frames):
        sxx, syy, sxy, r2 = process_frame(frame)
        stress_stack[frame, ..., 0, 0] = sxx
        stress_stack[frame, ..., 1, 1] = syy
        stress_stack[frame, ..., 0, 1] = sxy
        stress_stack[frame, ..., 1, 0] = sxy   # symmetric tensor
        r2_values.append(r2)

        yield StressResult(
            stress_tensor=stress_stack[:frame + 1],
            parameters=params,
            physical_scale=physical_scale,
            original_shape=force_field.shape[1:3],
            stress_shape=stress_stack.shape[1:3],
            method="BISM",
            r2_traction=float(r2) if np.isfinite(r2) else None,
        ), frame + 1, total_frames

    finite_r2 = [v for v in r2_values if np.isfinite(v)]
    return StressResult(
        stress_tensor=stress_stack,
        parameters=params,
        physical_scale=physical_scale,
        original_shape=force_field.shape[1:3],
        stress_shape=stress_stack.shape[1:3],
        method="BISM",
        r2_traction=float(np.mean(finite_r2)) if finite_r2 else None,
    )
