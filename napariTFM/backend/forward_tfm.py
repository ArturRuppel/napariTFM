"""Forward-model traction inversion (displacement input) with a soft support prior.

This ports the validated *log-soft mask confinement* from the photometric one-shot
prototype (napariTFM2.5D ``oneshot.py``) onto the **displacement field** as input,
so the confinement rides the validated PIV/FFD front-end instead of the fragile
image-formation model (PSF, bleaching, out-of-plane motion). The photometric
variant underperformed on real data precisely because it was hostage to that model;
taking the displacement field as input drops bead texture and warp modelling out
entirely.

For each frame we solve for the surface traction ``t`` (Pa) by minimizing

    J(t) = ‖ W · (G·t − u) ‖²  +  λ²‖t‖²  +  β‖ t·(1−p) ‖²

- ``G`` is the *same* Boussinesq / finite-thickness Green's operator FTTC inverts
  (reused verbatim from :mod:`napariTFM.backend.fttc`; folds in E, ν, gel_height,
  pixel_size). ``û = G·t̂`` maps traction → displacement per Fourier mode.
- ``λ`` is the Tikhonov amplitude ridge — the *same* ``regularization`` dial FTTC
  uses, entering as the identical physical ``λ²‖t‖²`` penalty on both the β=0
  (closed-form) and β>0 (iterative) branches so the dial means one thing throughout.
  It is the sole traction-field regularizer on both branches (there is no separate
  gradient-smoothness prior).
- ``β`` is the soft support prior — the off-mask penalty, applied to ``t·(1−p)`` where
  ``p`` is the support-probability map (:func:`support_probability`). There is
  deliberately no hard gate: the one-shot benchmark found gating clips genuine
  near-edge forces (|t| r 0.95 vs 0.99 for strong-soft), so "maximum confinement" is
  strong soft.
- ``R`` (``fwd_mask_reach``) sets *where* the boundary is: the free region is the mask
  grown outward by R px (``p ≡ 1`` there, a zero-penalty apron), and beyond it the
  penalty ramps up over a small fixed anti-ring edge (:func:`support_probability`). R is
  orthogonal to β *in effect*: because the apron is genuinely zero-penalty, β cannot
  shrink it, so R still moves the boundary at maximum strength (β sets *how hard* the
  exterior is pushed, R sets *how far out* forces are still allowed). Same map feeds the
  L1 route's soft support.
- ``W`` is an optional *fit-region weight*: it trusts the displacement only inside
  ``mask`` dilated by ``fwd_fit_margin_um``. The photometric MVP had this as
  ``fit_margin_um`` and it matters here too — without it, a neighbour cell's
  displacement sitting in the crop *demands* forces to explain it while β *forbids*
  off-mask forces, and the solver dumps residual stress onto the mask boundary.

The problem is a convex quadratic in ``t``, so:

- **β = 0 → closed form.** With no confinement, λ is diagonal in Fourier and the
  per-mode 2×2 Tikhonov solve ``t̂ = (GᴴG + λ²I)⁻¹ Gᴴ û`` reuses FTTC's exact
  machinery. Pure numpy/FFT; no torch required.
- **β > 0 → iterative.** The support term couples Fourier modes, so
  we solve the (still convex) QP as its normal equations ``A t = b`` by preconditioned
  Conjugate Gradient, with the Fourier-diagonal β=0 operator as the preconditioner
  (see docs/specs/forward-solver-pcg.md). ``A`` is SPD, so CG is exact in exact
  arithmetic. The operator is one array-module-agnostic (numpy | cupy) function — no
  autograd, no torch — so the CPU path is torch-free and the GPU path is CuPy.

Output contract matches FTTC: traction ``(2, H, W)`` float32 in Pa, ``[0] = t_x``.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from napariTFM.backend.fttc import FTTC
from napariTFM.backend.fttc_numba_functions import calculate_traction_2d
from napariTFM.backend.parameter_dataclasses import FTTCParameters

logger = logging.getLogger(__name__)

# The 0..100 "Mask confinement" dial is mapped LOGARITHMICALLY onto the soft
# penalty weight β, exactly as the photometric one-shot widget does — linear β is
# the wrong axis (off-mask force energy is driven toward zero, so equal *dial*
# steps must be equal *log-β* steps to each do visible work). These bounds are
# PROVISIONAL for the displacement operator: the one-shot's β~500 was tuned
# against a bounded ZNCC image loss, whereas here the data term is a *relative*
# displacement residual (O(1)), so the useful band differs. Recalibrate on real
# data — finding that band by playing with the dial is the whole point of the UI.
MASK_BETA_MIN = 1e-3
MASK_BETA_MAX = 1e1

# Fixed Gaussian width (force-grid px) that rounds the confinement boundary — an
# anti-ring edge so the (otherwise sharp) dilated-support edge does not Gibbs-ring.
# Not a user knob: reach sets *where* the boundary is, strength *how hard*, and this
# just keeps the transition C1. Small enough that the boundary still reads as crisp.
_RING_SIGMA_PX = 1.5


def confinement_to_beta(strength: float) -> float:
    """Map the 0..100 confinement dial onto a log-spaced soft-penalty β (0 → off)."""
    s = float(strength)
    if s <= 0.0:
        return 0.0
    f = min(max(s / 100.0, 0.0), 1.0)
    return MASK_BETA_MIN * (MASK_BETA_MAX / MASK_BETA_MIN) ** f


def support_probability(mask, params) -> np.ndarray:
    """Probability map ``p(x) ∈ [0, 1]`` that traction is *allowed* at ``x``.

    The free region is the mask grown outward by ``fwd_mask_reach`` (R, force-grid px):
    ``p ≡ 1`` on the mask *and* the R-px apron around it — a genuinely **zero-penalty**
    region, so the confinement strength β cannot shrink it (``0 · β = 0``). Beyond the
    apron ``p`` decays to 0 over a small *fixed* Gaussian (:data:`_RING_SIGMA_PX`), an
    anti-ring edge that keeps the boundary from Gibbs-ringing. With ``d`` the exterior
    distance from the mask (0 on the mask, growing outward),

        ``p = exp(−max(d − R, 0)² / 2·σ_ring²)``.

    This makes the two mask knobs orthogonal *in effect*: **R sets where** the boundary
    sits — purely spatial, so even at maximum strength it still moves the boundary — and
    **β sets how hard** the exterior beyond it is pushed. The apron only ever reaches
    outward, so real forces on the cell rim are never clipped. This is the shared shape
    of the exterior penalty on *both* mask-consuming routes (the L2 confined solve and
    the L1 soft support), each penalizing traction ∝ ``(1 − p)``. ``mask is None`` or a
    full mask ⇒ ``p ≡ 1`` (nothing to confine). Returns ``(H, W)`` float64.
    """
    support = np.asarray(mask) > 0
    from scipy import ndimage
    d = ndimage.distance_transform_edt(~support)            # 0 on the mask, grows outward
    reach = max(float(getattr(params, "fwd_mask_reach", 0.0)), 0.0)
    d_ext = np.maximum(d - reach, 0.0)                       # 0 within mask+R, grows beyond
    return np.exp(-(d_ext * d_ext) / (2.0 * _RING_SIGMA_PX * _RING_SIGMA_PX))


def _greens_operator(height: int, width: int, params: FTTCParameters) -> np.ndarray:
    """The Boussinesq/finite-thickness Green's operator on the force grid.

    Returns the real ``(2, 2, H, W)`` Fourier-space tensor ``G`` with ``û = G·t̂``,
    reusing FTTC's kernel verbatim (same E, ν, gel_height, and effective pixel
    size ``pixel_size · downscale_factor``). DC (k=0) is zeroed by FTTC, i.e. the
    mean traction lives in the operator's null space — as in FTTC, it is fixed by
    the priors, not the data.
    """
    calc = FTTC(params)
    pixelsize = params.pixel_size * params.downscale_factor
    kx, ky = calc._calculate_fourier_modes(width, height, pixelsize)
    return calc._calculate_greens_function(kx, ky)


def _fit_weight(mask: Optional[np.ndarray], valid: np.ndarray,
                params: FTTCParameters) -> np.ndarray:
    """Per-pixel weight on the data term: 1 inside mask+margin (and finite u), else 0.

    ``valid`` is the finite-displacement mask (NaNs get zero weight). With no
    support mask, or an effectively infinite margin, the whole (finite) field is
    trusted.
    """
    if mask is None:
        return valid.astype(np.float64)
    margin_px = float(params.fwd_fit_margin_um) / max(1e-9, params.pixel_size * params.downscale_factor)
    support = np.asarray(mask) > 0
    if not np.isfinite(margin_px) or margin_px > max(mask.shape):
        region = np.ones_like(support)
    else:
        from scipy import ndimage
        region = ndimage.binary_dilation(support, iterations=int(round(margin_px)))
    return (region & valid).astype(np.float64)


def _solve_closed_form(u: np.ndarray, params: FTTCParameters) -> np.ndarray:
    """β=0 path: FTTC's per-mode 2×2 Tikhonov inversion.

    ``u`` is ``(2, H, W)`` in µm. Returns traction ``(2, H, W)`` in Pa (float32).
    """
    height, width = u.shape[1:]
    G = _greens_operator(height, width, params)
    lam = float(params.regularization)
    Ginv = calculate_traction_2d(G, lam ** 2)  # (2,2,H,W): (GᵀG + λ²I)⁻¹ Gᵀ
    Ftux = np.fft.fft2(u[0])
    Ftuy = np.fft.fft2(u[1])
    Ftfx = Ginv[0, 0] * Ftux + Ginv[0, 1] * Ftuy
    Ftfy = Ginv[1, 0] * Ftux + Ginv[1, 1] * Ftuy
    tx = np.fft.ifft2(Ftfx).real
    ty = np.fft.ifft2(Ftfy).real
    return np.stack([tx, ty]).astype(np.float32)


def _resolve_backend(request: str):
    """Return ``(xp, fft, on_gpu)`` for 'auto' | 'cuda' | 'cpu'.

    GPU is **CuPy** (torch is intentionally not a backend on this path — see
    docs/specs/forward-solver-pcg.md). 'cuda' requires CuPy + a visible device;
    'auto' uses CuPy when importable with a device, else falls back to numpy/scipy.
    CuPy supplies only the array module + FFTs; the CG loop is hand-rolled
    (:func:`_pcg`) so it is the same algorithm on both backends.
    """
    req = str(request).lower()

    def _numpy_backend():
        import scipy.fft as fft
        return np, fft, False

    if req == "cpu":
        return _numpy_backend()
    try:
        import cupy as cp
        import cupyx.scipy.fft as fft
        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp, fft, True
    except Exception:
        pass
    if req == "cuda":
        raise RuntimeError(
            "fwd_device='cuda' needs CuPy with a visible CUDA device (the forward "
            "solver's GPU backend is CuPy, not torch). Install a matching cupy-cuda* "
            "wheel, or use fwd_device='auto'/'cpu'."
        )
    return _numpy_backend()


def _asnumpy(a) -> np.ndarray:
    """Bring an xp array back to host numpy (no-op for numpy, ``.get()`` for cupy)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _build_normal_equations(u: np.ndarray, mask: np.ndarray, beta: float,
                            params: FTTCParameters, xp, fft):
    """Assemble the confined-QP normal equations ``A w = b`` for one frame.

    Returns ``(apply_A, apply_Minv, b, (H, W), meta)``. ``apply_A`` is the SPD system
    operator; ``apply_Minv`` the Fourier-diagonal preconditioner (the β=0, W=I
    operator, invertible per mode); ``b`` the right-hand side. All act on the
    non-dimensional traction ``w`` (``t = E·T0·w``) as ``(2, H, W)`` real ``xp``
    arrays. Data term normalized by ``denom``; the λ (Tikhonov) term is FTTC's physical
    ``λ²‖t‖²`` (coefficient ``λ²·(E·T0)²/denom``, so it reproduces ``_solve_closed_form``
    at β=0); the β term keeps the forward-native ``1/N`` normalization
    (``N = 2·H·W``). See docs/specs/forward-solver-pcg.md.
    """
    height, width = u.shape[1:]
    E = float(params.young_modulus)
    T0 = float(params.fwd_traction_scale)
    lam = float(params.regularization)
    N = 2.0 * height * width

    G = _greens_operator(height, width, params)           # (2,2,H,W) real, û=G·t̂, DC=0
    GE = xp.asarray(E * G)                                 # non-dim forward map is O(1)
    GEc = GE.astype(xp.complex128)
    GtG = xp.real(xp.einsum("ikhw,kjhw->ijhw", GE, GE))   # (2,2,H,W) = GᴴG per mode

    valid = np.isfinite(u).all(axis=0)
    w_fit = _fit_weight(mask, valid, params)
    u_clean = np.nan_to_num(u, nan=0.0)
    wf = xp.asarray(w_fit)                                 # (H,W) data-term weight W
    u_t = xp.asarray(u_clean)                              # (2,H,W) µm
    off = xp.asarray(1.0 - support_probability(mask, params))  # (H,W) graded off-support (1−p)
    denom = float((wf * u_t ** 2).sum())
    denom = denom if denom > 1e-12 else 1e-12

    # λ (`regularization`) is the SHARED dial with FTTC/`_solve_closed_form`, so its
    # Tikhonov penalty must be the identical *physical* λ²‖t‖² on this path too — not a
    # differently-scaled λ, or the same dial means different things on the two sides of
    # the confinement switch (the β=0/β>0 branches would disagree on both the power of λ
    # AND the normalization). In the non-dim w (t = E·T0·w) with the data term carrying
    # the 1/denom factor, the coefficient that reproduces FTTC's absolute λ²‖t‖² is
    # λ²·(E·T0)²/denom: the denom cancels against the data term, so at β=0,γ=0,W=I this
    # solve == `_solve_closed_form` at the same λ, on every frame. (β/γ keep their
    # forward-native 1/N normalization — they have no FTTC counterpart to match.)
    lam_coef = lam ** 2 * (E * T0) ** 2 / denom

    def _P(w):                                            # w:(2,H,W) real → u_pred (µm)
        wk = fft.fft2(w.astype(xp.complex128), axes=(-2, -1))
        uk = xp.einsum("ijhw,jhw->ihw", GEc, wk)
        return T0 * fft.ifft2(uk, axes=(-2, -1)).real

    def _Pt(r):                                           # adjoint (GE Hermitian ⇒ same form)
        rk = fft.fft2(r.astype(xp.complex128), axes=(-2, -1))
        sk = xp.einsum("ijhw,jhw->ihw", GEc, rk)
        return T0 * fft.ifft2(sk, axes=(-2, -1)).real

    def apply_A(w):
        # Confine to the zero-mean subspace on BOTH sides (P0·A·P0): apply_Minv
        # annihilates the DC mode, so A must map into and out of the same subspace or
        # the DC it injects (β·(off·w) carries a mean; FFT round-off adds more) has no
        # preconditioned correction and CG stalls. Projecting both sides also keeps A
        # self-adjoint (P0·A·P0), which CG requires — projecting only the output does
        # not. Iterates are already zero-mean, so the input projection is a no-op in
        # the solve; it is there for symmetry (and the tests that check it).
        w = w - w.mean(axis=(1, 2), keepdims=True)
        data = _Pt(wf * _P(w)) / denom
        out = data + lam_coef * w + (beta / N) * (off * w)
        return out - out.mean(axis=(1, 2), keepdims=True)

    # preconditioner: M̂ = (T0²/denom)·GᴴG + lam_coef·I  (W=I, β dropped).
    # lam_coef = λ²·(E·T0)²/denom is FTTC's physical Tikhonov (see above); the same
    # coefficient must sit in A and M or one-step exactness silently fails.
    diag = lam_coef
    s = (T0 * T0) / denom
    M00 = s * GtG[0, 0] + diag
    M01 = s * GtG[0, 1]
    M10 = s * GtG[1, 0]
    M11 = s * GtG[1, 1] + diag
    det = M00 * M11 - M01 * M10
    det = xp.where(det == 0, 1.0, det)
    Mi00, Mi01, Mi10, Mi11 = M11 / det, -M01 / det, -M10 / det, M00 / det

    def apply_Minv(r):
        rk = fft.fft2(r.astype(xp.complex128), axes=(-2, -1))
        s0 = Mi00 * rk[0] + Mi01 * rk[1]
        s1 = Mi10 * rk[0] + Mi11 * rk[1]
        s0[0, 0] = 0.0                                    # project the DC/nullspace out (spec)
        s1[0, 0] = 0.0
        sk = xp.stack([s0, s1])
        return fft.ifft2(sk, axes=(-2, -1)).real

    b = _Pt(wf * u_t) / denom
    b = b - b.mean(axis=(1, 2), keepdims=True)            # zero DC of b (spec)
    meta = {"E": E, "T0": T0, "denom": denom,
            "rho": float((np.asarray(mask) > 0).mean())}
    return apply_A, apply_Minv, b, (height, width), meta


def _pcg(apply_A, apply_Minv, b, xp, tol, maxiter):
    """Preconditioned CG on the zero-mean subspace, backend-agnostic (numpy | cupy).

    Hand-rolled rather than scipy/cupyx ``cg`` on purpose: those libraries apply an
    ``M`` (LinearOperator) preconditioner *inconsistently* — the same M that
    converges under scipy stalls under cupyx 14.x — so depending on both would break
    the single-source guarantee. A loop over ``xp`` primitives (dot, axpy, and the
    ``apply_*`` operators) is the identical algorithm on both backends. Operates on
    ``(2, H, W)`` arrays directly. Returns ``(x, iters, converged)``.
    """
    def dot(a, c):
        return float(xp.sum(a * c))

    x = xp.zeros_like(b)
    r = b - apply_A(x)
    bnorm = float(xp.sqrt(xp.sum(b * b)))
    if bnorm == 0.0:
        return x, 0, True
    z = apply_Minv(r)
    p = z
    rz = dot(r, z)
    for it in range(1, int(maxiter) + 1):
        Ap = apply_A(p)
        alpha = rz / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if float(xp.sqrt(xp.sum(r * r))) <= tol * bnorm:
            return x, it, True
        z = apply_Minv(r)
        rz_new = dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, int(maxiter), False


def _solve_iterative(u: np.ndarray, mask: np.ndarray, beta: float,
                     params: FTTCParameters) -> np.ndarray:
    """β>0 path: preconditioned-CG solve of the confined QP's normal equations.

    ``u`` is ``(2, H, W)`` µm, ``mask`` is ``(H, W)`` truthy on support. Returns
    traction ``(2, H, W)`` Pa (float32). The SPD system ``A t = b`` (convex QP) is
    solved by a hand-rolled preconditioned CG (:func:`_pcg`) with the Fourier-diagonal
    β=0 operator as preconditioner — no autograd, no torch. Runs on numpy (CPU) or
    cupy (GPU); float64 internally. See the module docstring and
    docs/specs/forward-solver-pcg.md.
    """
    xp, fft, on_gpu = _resolve_backend(params.fwd_device)
    apply_A, apply_Minv, b, (height, width), meta = _build_normal_equations(
        u, mask, beta, params, xp, fft)
    tol = float(getattr(params, "fwd_cg_tol", 1e-8))
    w, iters, converged = _pcg(apply_A, apply_Minv, b, xp, tol, params.fwd_max_iter)
    t = _asnumpy(meta["E"] * meta["T0"] * w).astype(np.float32)
    logger.info("forward PCG: beta=%.3g masked_frac=%.3f iters=%d converged=%s backend=%s",
                float(beta), meta["rho"], iters, converged, "cupy" if on_gpu else "numpy")
    return t


def forward_traction_frame(displacement_frame: np.ndarray,
                           params: FTTCParameters,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Invert one displacement frame to traction via the forward method.

    Args:
        displacement_frame: ``(H, W, 2)`` displacement in µm (``[...,0]=u_x``).
        params: FTTC/force parameters; the ``fwd_*`` fields select behaviour.
        mask: optional ``(H, W)`` support mask (truthy where traction may act).
            Ignored when the confinement dial (``fwd_mask_strength``) is 0.

    Returns:
        ``(2, H, W)`` float32 traction in Pa (``[0]=t_x``, ``[1]=t_y``).
    """
    u = np.stack([displacement_frame[..., 0], displacement_frame[..., 1]]).astype(np.float64)

    beta = confinement_to_beta(params.fwd_mask_strength) if mask is not None else 0.0
    if beta <= 0.0:
        return _solve_closed_form(u, params)
    return _solve_iterative(u, np.asarray(mask), beta, params)
