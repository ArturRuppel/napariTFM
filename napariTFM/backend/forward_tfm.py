"""Forward-model traction inversion (displacement input) with a soft support prior.

This ports the validated *log-soft mask confinement* from the photometric one-shot
prototype (napariTFM2.5D ``oneshot.py``) onto the **displacement field** as input,
so the confinement rides the validated PIV/FFD front-end instead of the fragile
image-formation model (PSF, bleaching, out-of-plane motion). The photometric
variant underperformed on real data precisely because it was hostage to that model;
taking the displacement field as input drops bead texture and warp modelling out
entirely.

For each frame we solve for the surface traction ``t`` (Pa) by minimizing

    J(t) = ‖ W · (G·t − u) ‖²  +  λ‖t‖²  +  γ‖∇t‖²  +  β‖ t·(1−mask) ‖²

- ``G`` is the *same* Boussinesq / finite-thickness Green's operator FTTC inverts
  (reused verbatim from :mod:`napariTFM.backend.fttc`; folds in E, ν, gel_height,
  pixel_size). ``û = G·t̂`` maps traction → displacement per Fourier mode.
- ``λ`` is a Tikhonov amplitude ridge (conditioning only).
- ``γ`` (``fwd_smoothness``) is the **gradient-smoothness prior and the primary
  regularizer of the confined solve.** The photometric one-shot parametrized
  traction on a coarse cubic-B-spline / Gaussian-per-bead basis — a few hundred
  DOF, smooth by construction — and *that basis was its real smoother*. This
  displacement-input port solves on a free per-pixel grid instead, so the basis
  smoothness has to come back as an explicit ‖∇t‖² term. Without it, confining
  forces to the mask removes the solver's off-mask escape valve and the in-mask
  field overfits the (delocalized) displacement into high-frequency garbage —
  confinement then *hurts* (recovered error worse than zero). With it, confinement
  beats unconfined FTTC. See the ``_dev`` why-artifacts probe.
- ``β`` is the soft support prior — the off-mask penalty. There is deliberately no
  hard gate: the one-shot benchmark found gating clips genuine near-edge forces
  (|t| r 0.95 vs 0.99 for strong-soft), so "maximum confinement" is strong soft.
- ``W`` is an optional *fit-region weight*: it trusts the displacement only inside
  ``mask`` dilated by ``fwd_fit_margin_um``. The photometric MVP had this as
  ``fit_margin_um`` and it matters here too — without it, a neighbour cell's
  displacement sitting in the crop *demands* forces to explain it while β *forbids*
  off-mask forces, and the solver dumps residual stress onto the mask boundary.

The problem is a convex quadratic in ``t``, so:

- **β = 0 → closed form.** With no confinement there is no escape valve to abuse,
  so the γ prior is unnecessary: λ is diagonal in Fourier and the per-mode 2×2
  Tikhonov solve ``t̂ = (GᴴG + λ²I)⁻¹ Gᴴ û`` reuses FTTC's exact machinery *minus*
  the Lanczos low-pass. Pure numpy/FFT; no torch required. (γ is iterative-only.)
- **β > 0 → iterative.** The support and smoothness terms couple Fourier modes, so
  we solve the (still convex) QP with L-BFGS on the non-dimensionalized traction
  (torch, autograd through the same FFT operator). L-BFGS converges to the global
  optimum; no pyramid needed (the loss is convex, unlike the photometric ZNCC).

Output contract matches FTTC: traction ``(2, H, W)`` float32 in Pa, ``[0] = t_x``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from napariTFM.backend.fttc import FTTC
from napariTFM.backend.fttc_numba_functions import calculate_traction_2d
from napariTFM.backend.parameter_dataclasses import FTTCParameters

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


def confinement_to_beta(strength: float) -> float:
    """Map the 0..100 confinement dial onto a log-spaced soft-penalty β (0 → off)."""
    s = float(strength)
    if s <= 0.0:
        return 0.0
    f = min(max(s / 100.0, 0.0), 1.0)
    return MASK_BETA_MIN * (MASK_BETA_MAX / MASK_BETA_MIN) ** f


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
    kx, ky, _, _ = calc._calculate_fourier_modes(width, height, pixelsize)
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
    """β=0 path: FTTC's per-mode 2×2 Tikhonov inversion, *without* the Lanczos filter.

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


def _resolve_torch_device(request: str):
    """Return a torch.device for 'auto' | 'cuda' | 'cpu', mirroring the PIV/FFD backend."""
    import torch
    req = str(request).lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("fwd_device='cuda' but no CUDA device is available; "
                               "use 'auto' or 'cpu'.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _solve_iterative(u: np.ndarray, mask: np.ndarray, beta: float,
                     params: FTTCParameters) -> np.ndarray:
    """β>0 path: convex-QP solve with the soft support prior, via L-BFGS (torch).

    ``u`` is ``(2, H, W)`` µm, ``mask`` is ``(H, W)`` truthy on support. Returns
    traction ``(2, H, W)`` Pa (float32). Non-dimensionalized as ``t = E·T0·w`` with
    ``w = O(1)`` for conditioning; the data term is a *relative* residual so λ and β
    are operator-scale-independent.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "The forward method's support prior (β > 0) needs PyTorch. Install the "
            "torch extra, or set the mask confinement to 0 to use the torch-free "
            "closed-form path."
        ) from e

    device = _resolve_torch_device(params.fwd_device)
    dtype = torch.float64 if str(params.fwd_dtype) == "float64" else torch.float32
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    height, width = u.shape[1:]
    E = float(params.young_modulus)
    T0 = float(params.fwd_traction_scale)
    lam = float(params.regularization)

    G = _greens_operator(height, width, params)          # (2,2,H,W) real, ∝ 1/E
    valid = np.isfinite(u).all(axis=0)                   # (H,W)
    w_fit = _fit_weight(mask, valid, params)             # (H,W) in {0,1}
    u_clean = np.nan_to_num(u, nan=0.0)

    # E·G removes the 1/E scaling → the forward map on w is O(1) and well-conditioned.
    GE = torch.as_tensor(E * G, device=device, dtype=cdtype)          # (2,2,H,W)
    u_t = torch.as_tensor(u_clean, device=device, dtype=dtype)        # (2,H,W)
    wf = torch.as_tensor(w_fit, device=device, dtype=dtype)           # (H,W)
    off = torch.as_tensor((~(np.asarray(mask) > 0)).astype(np.float64),
                          device=device, dtype=dtype)                 # (H,W) off-support
    w = torch.zeros(2, height, width, device=device, dtype=dtype, requires_grad=True)

    denom = (wf * u_t.pow(2)).sum().clamp_min(1e-12)     # relative-residual normalizer

    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=int(params.fwd_max_iter),
                            history_size=25, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)

    smooth_w = float(params.fwd_smoothness)

    def closure():
        opt.zero_grad()
        wc = w.to(cdtype)
        wk = torch.fft.fft2(wc)                                       # (2,H,W)
        uk = torch.einsum("ijhw,jhw->ihw", GE, wk)                    # G·(E·w) in Fourier
        u_pred = T0 * torch.fft.ifft2(uk).real                        # (2,H,W) µm
        resid = wf * (u_pred - u_t).pow(2)
        data = resid.sum() / denom
        white = lam * w.pow(2).mean()
        soft = beta * (w * off).pow(2).mean()
        # Gradient-smoothness prior: the PRIMARY regularizer here. Confinement
        # removes the off-mask escape valve, so without this the free per-pixel
        # in-mask field overfits the delocalized displacement (see forward_tfm
        # module docstring / the _dev why-artifacts probe). Periodic ∇ via roll,
        # matching the one-shot solver's TV term.
        if smooth_w > 0.0:
            gx = torch.roll(w, -1, 2) - w
            gy = torch.roll(w, -1, 1) - w
            smooth = smooth_w * (gx.pow(2) + gy.pow(2)).mean()
        else:
            smooth = 0.0
        loss = data + white + soft + smooth
        loss.backward()
        return loss

    opt.step(closure)
    t = (E * T0 * w).detach().cpu().numpy()
    return t.astype(np.float32)


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
