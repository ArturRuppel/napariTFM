"""Bayesian L2 traction reconstruction (Huang et al., *Sci. Rep.* 9:539, 2019).

Automatic, evidence-maximizing Tikhonov regularization -- no manual λ, chosen per frame from
the data itself. This is the paper's actual method: a **real-space, column-standardized,
over-determined** forward operator ``M`` (traction on a coarse mesh → displacement on a finer
grid), inverted by Tikhonov with the regularization set by maximizing the marginal likelihood
(the *evidence*) of the displacement.

**Why not the Fourier/FTTC shortcut.** An earlier version bolted evidence maximization onto
FTTC's per-Fourier-mode 2×2 SVD blocks. That is *not* Huang et al.'s method and it is
mathematically degenerate: the raw Boussinesq spectrum spans ~5 decades and the square,
unstandardized pixel-resolution problem (~5×10⁵ modes) has no interior evidence optimum, so it
pinned λ to the spectrum floor and under-regularized by ~10⁴×. Two ingredients from the paper
fix it, and both are absent in the Fourier bolt-on:

* **Column standardization** -- normalize every column of ``M`` to unit variance before the SVD
  (``F = F./sd`` undoes it on the solution). This conditions the operator; without it the
  evidence is degenerate. Standardization has no clean per-Fourier-mode analogue, which is why
  the method must live in real space.
* **An over-determined system** -- more displacement samples than traction nodes. This is what
  creates the interior evidence maximum; the square Fourier problem cannot.

**Model.** Gaussian prior on traction (precision ``α``) and Gaussian displacement noise
(precision ``β``) make the MAP traction the Tikhonov solution with ``λ = α/β``. The evidence
``p(u | α, β)`` integrates in closed form (Huang et al. Eq. 8; MacKay, *Neural Comput.* 4:415,
1992). We recover (α, β) by the MacKay fixed point on the standardized SVD:

    γ = Σ sᵢ²/(sᵢ²+λ)            # effective number of resolved parameters
    α = γ / ‖F‖²,   β = (n_data − γ) / ‖M F − u‖²,   λ = α/β

iterated to convergence. Two variants (both the paper's):

* **ABL2** (default, parameter-free) -- infer both α and β from the evidence. Robust, and it
  needs no noise input, so it is immune to the fact that a regularized displacement front-end
  (e.g. FFD) has already smoothed away the pixel noise a direct estimate would need.
* **BL2** -- pin β from a measured displacement-noise variance (``estimate_noise_variance`` over
  the cell-free exterior) and infer only α. Used when a reliable exterior noise level is
  available; the paper reports it slightly more robust at high noise.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- noise estimation
def _highpass_noise_variance(field: np.ndarray,
                             region: Optional[np.ndarray] = None) -> Optional[float]:
    """Robust (MAD) high-pass estimate of white-noise variance in a smooth 2D field.

    Convolving with the Laplacian mask ``N = [[1,-2,1],[-2,4,-2],[1,-2,1]]`` (Immerkær 1996)
    annihilates smooth content and passes noise; for white noise of variance σ² the response has
    std ``6σ``. We take σ from the **median absolute deviation** of the response (``σ ≈
    median(|conv|) / (0.6745·6)``) so a few sharp signal features don't inflate it. ``region``
    (truthy = use) restricts the estimate to those pixels (e.g. the cell exterior); ``None`` uses
    the whole field. Returns σ² (or ``None`` if too few samples).
    """
    from scipy import ndimage
    a = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
    h, w = a.shape
    if h < 3 or w < 3:
        return None
    kernel = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    conv = ndimage.convolve(a, kernel, mode="reflect")
    if region is not None:
        sel = np.abs(conv)[np.asarray(region) > 0]
        if sel.size < 16:
            sel = np.abs(conv).ravel()
    else:
        sel = np.abs(conv).ravel()
    sigma = float(np.median(sel)) / (0.6745 * 6.0)
    var = sigma * sigma
    return var if np.isfinite(var) and var > 0.0 else None


def estimate_noise_variance(displacement_frame: np.ndarray,
                            mask: Optional[np.ndarray]) -> Optional[float]:
    """Per-component displacement noise variance ``σ_r²`` -- BL2's measured ``1/β``.

    Restricted to the cell exterior when a foreground ``mask`` is given (truthy = cell), per the
    paper's "noise recorded far from cells". Because it is a high-pass it tolerates the smooth
    near-cell displacement halo that would corrupt a raw far-field variance. Returns the mean of
    the two components' variances, or ``None`` if it cannot be estimated (caller then uses ABL2).
    """
    region = None
    if mask is not None and np.asarray(mask).shape == displacement_frame.shape[:2]:
        region = ~(np.asarray(mask) > 0)
    vx = _highpass_noise_variance(displacement_frame[..., 0], region)
    vy = _highpass_noise_variance(displacement_frame[..., 1], region)
    if vx is None or vy is None:
        return None
    return 0.5 * (vx + vy)


# ------------------------------------------------------------------ real-space forward operator
def _boussinesq_M(disp_xy: np.ndarray, force_xy: np.ndarray, E: float, nu: float,
                  node_area: float) -> np.ndarray:
    """Boussinesq forward operator ``M`` (2·N_d × 2·N_f): ``u_µm = M @ t_Pa``.

    Surface Green's function of a semi-infinite elastic half-space for a tangential point force
    (Landau–Lifshitz / Boussinesq): ``G_ij(r) = (1+ν)/(πE r)·[(1-ν)δ_ij + ν r_i r_j / r²]``,
    units 1/(Pa·µm). A far entry is ``G(r)·node_area``; the self/near term (``r`` below a fraction
    of the node spacing) is the analytic patch integral ``∫_patch G`` (finite: ``G ~ 1/r`` is
    integrable in 2D). ``disp_xy``/``force_xy`` are (N,2) positions in µm.
    """
    dx = disp_xy[:, 0][:, None] - force_xy[:, 0][None, :]
    dy = disp_xy[:, 1][:, None] - force_xy[:, 1][None, :]
    r = np.hypot(dx, dy)
    node = float(np.sqrt(node_area))
    r_floor = 0.5 * node
    near = r < r_floor
    r_eval = np.where(near, 1.0, r)          # avoid 1/0; near entries overwritten below
    r2 = r_eval * r_eval
    pref = (1.0 + nu) / (np.pi * E * r_eval)
    Gxx = pref * ((1.0 - nu) + nu * dx * dx / r2) * node_area
    Gyy = pref * ((1.0 - nu) + nu * dy * dy / r2) * node_area
    Gxy = pref * (nu * dx * dy / r2) * node_area
    # self/near patch integral over one node's w×w cell (cell-centred subsample, no r=0)
    sub = (np.arange(20) + 0.5) / 20.0 - 0.5
    sx, sy = np.meshgrid(sub * node, sub * node)
    sr = np.hypot(sx, sy)
    sa = (node / 20.0) ** 2
    pf = (1.0 + nu) / (np.pi * E * sr)
    sxx = float(np.sum(pf * ((1.0 - nu) + nu * sx * sx / sr ** 2)) * sa)
    syy = float(np.sum(pf * ((1.0 - nu) + nu * sy * sy / sr ** 2)) * sa)
    Gxx[near] = sxx
    Gyy[near] = syy
    Gxy[near] = 0.0
    return np.block([[Gxx, Gxy], [Gxy, Gyy]])


def _solve_evidence(Xs: np.ndarray, uc: np.ndarray, beta_fixed: Optional[float] = None,
                    lam_fixed: Optional[float] = None,
                    n_iter: int = 200, tol: float = 1e-5) -> Tuple[np.ndarray, dict]:
    """MacKay evidence fixed point on the standardized operator ``Xs`` and centred data ``uc``.

    Returns ``(F_std, info)``: the traction in standardized coordinates (undo with ``/sd``) and
    an info dict with ``alpha``/``beta``/``lam``/``method``/``iters``. ``beta_fixed`` pins the
    noise precision (**BL2**); ``None`` infers it (**ABL2**). ``lam_fixed`` pins the ridge itself
    (skip inference entirely and just apply it) -- how a λ estimated on one frame is *reused* on
    another: the operator ``Xs`` is identical across frames, so the same λ transfers exactly.

    Solved through the eigendecomposition of the small ``n_param × n_param`` normal matrix
    ``XᵀX`` rather than the full SVD of ``Xs``: the left singular vectors are never needed, and
    every evidence quantity (‖F‖², the residual, the trace γ) is closed-form in the eigenvalues
    (= s²) and the projected data. Much cheaper than an ``n_data × n_param`` SVD.
    """
    n_data, n_param = Xs.shape
    G = Xs.T @ Xs
    w, V = np.linalg.eigh(G)                       # eigenvalues w = s² (ascending), V orthonormal
    w = np.clip(w, 0.0, None)
    p = V.T @ (Xs.T @ uc)                          # data projected onto the eigenbasis
    uu = float(uc @ uc)
    pos = w[w > 0]
    beta = beta_fixed if beta_fixed is not None else 1.0
    method = "BL2" if beta_fixed is not None else "ABL2"
    alpha = 1.0
    it = 0
    if lam_fixed is not None:                      # frozen ridge: apply, no inference
        lam = float(lam_fixed)
        method = "fixed-λ"
    else:
        lam = float(np.median(pos)) if pos.size else 1.0
        for it in range(1, n_iter + 1):
            denom = w + lam
            coef = p / denom
            Ff = float(coef @ coef)
            gamma = float((w / denom).sum())
            alpha = gamma / max(Ff, 1e-30)
            if beta_fixed is None:
                resid2 = uu - 2.0 * float(p @ coef) + float((w * coef) @ coef)
                beta = (n_data - gamma) / max(resid2, 1e-30)
            lam_new = alpha / max(beta, 1e-30)
            if abs(lam_new - lam) <= tol * lam:
                lam = lam_new
                break
            lam = lam_new
    F = V @ (p / (w + lam))                        # traction in standardized coordinates
    info = dict(alpha=alpha, beta=beta, lam=lam, method=method, iters=it,
                n_data=n_data, n_param=n_param)
    return F, info


def _standardized_operator(displacement_field: np.ndarray, E: float, nu: float, pixel_size: float,
                           n_force: int, overdetermine: float):
    """Build the column-standardized Boussinesq operator and centred data for one frame.

    Returns ``(Xs, sd, uc, nfx, nfy)``: the standardized operator ``Xs`` (n_data × n_param), the
    per-column scales ``sd`` (undo with ``/sd`` on the solution), the mean-centred stacked
    displacement ``uc``, and the traction-mesh dimensions. Geometry depends only on the grid /
    material / mesh -- so for a fixed experiment ``Xs`` and ``sd`` are identical across frames and
    only ``uc`` changes, which is exactly what lets a λ estimated on one frame transfer to others.
    """
    from scipy import ndimage

    H, W = displacement_field.shape[:2]
    ps = float(pixel_size)
    fs_px = max(H, W) / float(n_force)
    nfx = max(4, int(round(W / fs_px)))
    nfy = max(4, int(round(H / fs_px)))
    fx = (np.arange(nfx) + 0.5) * (W / nfx)
    fy = (np.arange(nfy) + 0.5) * (H / nfy)
    FX, FY = np.meshgrid(fx, fy)
    force_xy = np.column_stack([FX.ravel() * ps, FY.ravel() * ps])
    node_area = (W / nfx * ps) * (H / nfy * ps)
    ndx = max(nfx + 1, int(round(nfx * overdetermine)))
    ndy = max(nfy + 1, int(round(nfy * overdetermine)))
    dxp = (np.arange(ndx) + 0.5) * (W / ndx)
    dyp = (np.arange(ndy) + 0.5) * (H / ndy)
    DX, DY = np.meshgrid(dxp, dyp)
    disp_xy = np.column_stack([DX.ravel() * ps, DY.ravel() * ps])
    ux = ndimage.map_coordinates(displacement_field[..., 0], [DY.ravel(), DX.ravel()], order=1)
    uy = ndimage.map_coordinates(displacement_field[..., 1], [DY.ravel(), DX.ravel()], order=1)
    u = np.concatenate([ux, uy]).astype(np.float64)

    M = _boussinesq_M(disp_xy, force_xy, E, nu, node_area)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (M - M.mean(axis=0)) / sd                   # column-standardize (Huang et al.)
    uc = u - u.mean()
    return Xs, sd, uc, nfx, nfy


def _resolve_beta(displacement_field, mask, noise_var):
    """Noise precision β for BL2: pinned from ``noise_var`` or a masked exterior estimate, else
    ``None`` (ABL2 infers it)."""
    if noise_var is None and mask is not None:
        noise_var = estimate_noise_variance(displacement_field, mask)
    return (1.0 / noise_var) if (noise_var and noise_var > 0.0) else None


def estimate_bayesian_lambda(displacement_field: np.ndarray, E: float, nu: float,
                             pixel_size: float, mask: Optional[np.ndarray] = None,
                             n_force: int = 32, overdetermine: float = 1.6,
                             noise_var: Optional[float] = None) -> float:
    """Evidence-optimal ridge ``λ = α/β`` inferred on one frame, to *freeze* and reuse across the
    experiment (the auto-λ button). Same signature/knobs as :func:`reconstruct_bl2_frame`; returns
    the scalar λ to pass back in as ``lam``."""
    Xs, _, uc, _, _ = _standardized_operator(displacement_field, E, nu, pixel_size,
                                             n_force, overdetermine)
    beta_fixed = _resolve_beta(displacement_field, mask, noise_var)
    _, info = _solve_evidence(Xs, uc, beta_fixed=beta_fixed)
    return float(info["lam"])


def reconstruct_bl2_frame(displacement_field: np.ndarray, E: float, nu: float,
                          pixel_size: float, mask: Optional[np.ndarray] = None,
                          n_force: int = 32, overdetermine: float = 1.6,
                          noise_var: Optional[float] = None,
                          lam: Optional[float] = None) -> np.ndarray:
    """Bayesian-L2 traction for one displacement frame -> ``(2, H, W)`` traction in Pa.

    Args:
        displacement_field: ``(H, W, 2)`` displacement in **µm** (``[...,0]=u_x``).
        E, nu: substrate Young's modulus (Pa) and Poisson ratio.
        pixel_size: grid spacing in **µm** (already ``pixel_size × downscale_factor``).
        mask: ``(H, W)`` foreground mask (truthy = cell) for the BL2 exterior noise estimate;
            ``None`` uses ABL2 (both α and β inferred).
        n_force: traction-mesh nodes along the larger axis (coarse; keeps the eigensolve small and
            makes the system over-determined against the finer displacement sampling).
        overdetermine: displacement samples per traction node along each axis (>1).
        noise_var: per-component displacement-noise variance to pin β (**BL2**). ``None`` and no
            usable ``mask`` estimate -> **ABL2**.
        lam: frozen ridge to apply instead of inferring one -- pass the value from
            :func:`estimate_bayesian_lambda` to reuse one frame's regularization across the whole
            experiment (keeps the smoothing constant, hence comparable, frame to frame).

    Returns:
        ``(2, H, W)`` traction (Pa), the coarse-mesh solution bilinearly resampled to the input
        grid; ``[0]=t_x``, ``[1]=t_y``.
    """
    from scipy import ndimage

    H, W = displacement_field.shape[:2]
    Xs, sd, uc, nfx, nfy = _standardized_operator(displacement_field, E, nu, pixel_size,
                                                  n_force, overdetermine)
    beta_fixed = _resolve_beta(displacement_field, mask, noise_var)
    F_std, info = _solve_evidence(Xs, uc, beta_fixed=beta_fixed, lam_fixed=lam)
    t = F_std / sd                                   # undo standardization -> traction (Pa)
    logger.info("Bayesian L2 (%s): lam=%.4g alpha=%.3g beta=%.3g nodes=%dx%d iters=%d",
                info["method"], info["lam"], info["alpha"], info["beta"], nfx, nfy, info["iters"])

    nnode = nfx * nfy
    tx = t[:nnode].reshape(nfy, nfx)
    ty = t[nnode:].reshape(nfy, nfx)
    out = np.empty((2, H, W), dtype=np.float32)
    out[0] = ndimage.zoom(tx, (H / nfy, W / nfx), order=1).astype(np.float32)
    out[1] = ndimage.zoom(ty, (H / nfy, W / nfx), order=1).astype(np.float32)
    return out
