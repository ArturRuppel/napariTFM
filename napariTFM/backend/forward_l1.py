"""Sparse traction inversion by group-L1 (the sparsity prior on the traction field).

FTTC regularizes with L2 (a Tikhonov ridge). L2 *spreads*: it cannot place a force at an
adhesion without raising the floor around it, so displacement noise survives as
low-amplitude spurious traction across the cell interior. This solver regularizes
with group-L1 instead, minimizing over the surface traction ``t`` (Pa)

    ½‖W·(G·t − u)‖² / denom  +  λ₁ · Σ_pixels ‖t[:, pixel]‖₂

The group-L1 term (a sum over pixels of the per-pixel traction *vector* norm) drives
whole pixels to exactly zero, so the recovered field is sparse: it concentrates on a
few spots without being told where they are. That matches the real support of an
adhesion-bearing cell, where traction genuinely is sparse. On the force benchmark
(benchmarkTFM force_measurements) this beats both FTTC and the L2 cell-confinement on
in-cell accuracy while needing *no* mask, and preserves the traction peak best of any
method (it thresholds rather than spreads).

- ``G`` is the *same* Boussinesq / finite-thickness Green's operator FTTC and the
  confined solver use (reused verbatim from :mod:`napariTFM.backend.forward_tfm`; folds
  in E, ν, gel_height, pixel_size). ``û = G·t̂`` maps traction → displacement per mode.
- ``λ₁`` (``l1_sparsity``) is set as a *fraction* of ``λ₁_max``, the value above which
  every pixel thresholds to zero. The fraction is scene-independent (it transfers
  across fields), so the dial means the same thing everywhere: ~0 dense, ~1 empty,
  useful band ~0.05–0.2, scaling up with noise. 0 disables this solver.
- ``mask`` (optional) is only a data fit-region weight ``W`` when callers opt into
  that lower-level behavior directly. The normal force pipeline passes no mask into
  this solver; Force-panel mask clipping is applied post-hoc in
  :mod:`napariTFM.backend.fttc`.

The problem is convex but non-smooth (the L1 term), so it is solved by FISTA
(accelerated proximal gradient): a gradient step on the smooth data term followed by
the group-L1 proximal operator (per-pixel soft-thresholding). The gradient uses the
same Fourier-diagonal ``P``/``Pᵀ`` operators as the confined solver, so like it this
path is one array-module-agnostic (numpy | cupy) routine: torch-free CPU, CuPy GPU.

Output contract matches FTTC / the confined solver: traction ``(2, H, W)`` float32 in
Pa, ``[0] = t_x``.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend.forward_tfm import (
    _greens_operator, _fit_weight, _resolve_backend, _asnumpy)

logger = logging.getLogger(__name__)

def _exterior_penalty(mask, valid, l_data, params, xp, dtype):
    """Compatibility shim for the removed soft mask-support penalty."""
    return xp.zeros((1,) + valid.shape, dtype=dtype)


def l1_traction_frame(displacement_frame: np.ndarray,
                      params: FTTCParameters,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Invert one displacement frame to a sparse traction field via FISTA group-L1.

    Args:
        displacement_frame: ``(H, W, 2)`` displacement in µm (``[...,0] = u_x``).
        params: FTTC/force parameters; ``l1_sparsity`` (fraction of λ₁_max) sets the
            sparsity, ``l1_max_iter`` the FISTA iteration budget, and ``fwd_device`` /
            ``fwd_dtype`` select the backend.
        mask: optional ``(H, W)`` data fit-region for direct low-level callers. The
            normal force pipeline passes ``None`` and applies any mask clipping after
            the solve.

    Returns:
        ``(2, H, W)`` float32 traction in Pa (``[0] = t_x``, ``[1] = t_y``).
    """
    u = np.stack([displacement_frame[..., 0], displacement_frame[..., 1]]).astype(np.float64)
    frac = float(params.l1_sparsity)
    height, width = u.shape[1:]

    xp, fft, on_gpu = _resolve_backend(params.fwd_device)
    dtype = xp.float32 if str(params.fwd_dtype) == "float32" else xp.float64
    cdtype = xp.complex64 if dtype == xp.float32 else xp.complex128

    G = _greens_operator(height, width, params)               # (2,2,H,W) real, û=G·t̂, DC=0
    GE = xp.asarray(G, dtype=dtype)
    GEc = GE.astype(cdtype)
    GtG = xp.real(xp.einsum("ikhw,kjhw->ijhw", GE, GE))        # per-mode 2×2 GᵀG

    valid = np.isfinite(u).all(axis=0)
    w_fit = _fit_weight(mask, valid, params)                   # (H,W) data-term weight W
    wf = xp.asarray(w_fit, dtype=dtype)
    u_t = xp.asarray(np.nan_to_num(u, nan=0.0), dtype=dtype)
    denom = float((wf * u_t ** 2).sum())
    denom = denom if denom > 1e-12 else 1e-12

    def P(t):                                                  # traction → u_pred (µm)
        tk = fft.fft2(t.astype(cdtype), axes=(-2, -1))
        uk = xp.einsum("ijhw,jhw->ihw", GEc, tk)
        return fft.ifft2(uk, axes=(-2, -1)).real

    def Pt(r):                                                 # adjoint (G symmetric real)
        rk = fft.fft2(r.astype(cdtype), axes=(-2, -1))
        sk = xp.einsum("ijhw,jhw->ihw", GEc, rk)
        return fft.ifft2(sk, axes=(-2, -1)).real

    # Lipschitz constant of the smooth part's gradient: ‖GᵀWG‖/denom ≤ max_mode λmax(GᵀG)
    # / denom (W ≤ 1). λmax of a symmetric 2×2 is ½(tr + √(tr²−4det)).
    tr = GtG[0, 0] + GtG[1, 1]
    det = GtG[0, 0] * GtG[1, 1] - GtG[0, 1] * GtG[1, 0]
    lam_mode = 0.5 * (tr + xp.sqrt(xp.clip(tr * tr - 4 * det, 0.0, None)))
    lmax = float(lam_mode.max())
    l_data = lmax / denom

    pen = _exterior_penalty(mask, valid, l_data, params, xp, dtype)   # (1,H,W) ≥ 0

    # Elastic Net ridge: a *global* L2 shrinkage ½ λ₂‖t‖² (gradient λ₂·t) added to the
    # smooth objective, turning the pure-L1 solve into an elastic net. λ₂ is a
    # scene-independent fraction of the *median* per-mode curvature — NOT l_data (the
    # max): the Boussinesq spectrum decays steeply, so scaling to the median keeps the
    # dial gentle and in a useful band (~0.1..1). Constant curvature ⇒ folds into the
    # Lipschitz constant / step. l2_ridge = 0 ⇒ pure group-L1 (bit-for-bit).
    pos_curv = lam_mode[lam_mode > 0]
    l_ridge = (float(xp.median(pos_curv)) / denom) if pos_curv.size else l_data
    lam2 = float(getattr(params, "l2_ridge", 0.0)) * l_ridge
    L = l_data + float(pen.max()) + lam2
    step = 1.0 / max(L, 1e-30)

    # λ₁_max: the per-pixel gradient magnitude at t=0. Above it every pixel thresholds
    # to zero, so parametrizing λ₁ = frac · λ₁_max makes the dial scene-independent.
    # (pen·t vanishes at t=0, so it leaves λ₁_max — and thus the sparsity dial — intact.)
    grad0 = Pt(wf * u_t) / denom
    lam1 = frac * float(xp.sqrt((grad0 * grad0).sum(axis=0)).max())
    tau = lam1 * step

    def prox(z):                                               # group soft-threshold
        n = xp.sqrt((z * z).sum(axis=0, keepdims=True) + 1e-30)
        return z * xp.clip(1.0 - tau / n, 0.0, None)

    def gradf(t):
        return Pt(wf * (P(t) - u_t)) / denom + pen * t + lam2 * t

    t = xp.zeros((2, height, width), dtype=dtype)
    z = t.copy()
    s = 1.0
    for _ in range(int(params.l1_max_iter)):
        t_new = prox(z - step * gradf(z))
        s_new = 0.5 * (1.0 + (1.0 + 4.0 * s * s) ** 0.5)
        z = t_new + ((s - 1.0) / s_new) * (t_new - t)
        t, s = t_new, s_new

    # Pin the net force to zero. G has no DC mode, so the mean traction is unobservable
    # from the displacement and FTTC sets it to zero (that is why FTTC's net force is
    # exactly 0). Subtracting each component's global mean is that same DC-nullspace
    # projection: the data fit is unchanged, Σt → 0 exactly, and the offset it leaves on
    # the (sparse) background is negligible. Matches the equilibrium of an isolated cell.
    t = t - t.mean(axis=(1, 2), keepdims=True)

    nnz = float((xp.sqrt((t * t).sum(axis=0)) > 1e-6).mean())
    logger.info("forward L1: frac=%.3f lam1=%.3g nnz=%.3f iters=%d backend=%s",
                frac, lam1, nnz, int(params.l1_max_iter), "cupy" if on_gpu else "numpy")
    return _asnumpy(t).astype(np.float32)
