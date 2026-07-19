"""Bayesian L2 regularization: automatic, evidence-maximizing choice of the FTTC
Tikhonov parameter (Huang et al., *Sci. Rep.* 9:539, 2019).

The plain FTTC path needs a Tikhonov λ. FTTC ships two ways to pick it: a manual value
and Generalized Cross-Validation (GCV). Huang et al. show that the classical automatic
selectors — the L-curve criterion and GCV — disagree and behave unreliably as the noise
grows (their Fig. 2 / Fig. S1), which biases any cross-condition comparison. Their
recommended replacement is to treat regularization as Bayesian inference and pick λ by
*maximizing the evidence* (the marginal likelihood of the displacement data), which needs
no manual tuning and adapts per frame.

**The model.** With a Gaussian prior on traction (variance ``1/α``) and Gaussian
displacement noise (variance ``1/β``), the maximum-a-posteriori traction is exactly the
L2 (Tikhonov) solution with ``λ = α/β`` — so choosing λ *is* choosing α and β. The
evidence ``p(u | α, β)`` is Gaussian and integrates in closed form (Huang et al. Eq. 8;
MacKay, *Neural Comput.* 4:415, 1992). Maximizing it gives the fixed-point updates

    γ      = Σ_i s_i² / (s_i² + λ)          # effective number of resolved parameters
    α      = γ / (2 E_f),   2 E_f = Σ_i f_i²                (‖traction‖²)
    β      = (m − γ) / (2 E_u),   2 E_u = Σ_i (λ d_i /(s_i²+λ))²   (‖residual‖²)
    λ      = α / β

iterated to convergence, where ``s_i`` are the singular values of the forward operator,
``d_i`` the data projected onto its left singular vectors, and ``f_i = s_i d_i/(s_i²+λ)``
the traction in the SVD basis. This is diagonal in the SVD basis, and FTTC's Fourier
inversion already produces exactly that decomposition per 2×2 mode block
(:meth:`FTTC._svd_block`), so the whole iteration is a handful of vector ops over the
same ``s``/``d`` GCV consumes — no new linear algebra.

**Two variants (both from the paper):**

* **BL2** (Bayesian L2) — the noise level ``β`` is measured directly from the data
  (displacement variance far from any cell) and held fixed; only ``α`` is inferred. The
  paper finds this the more robust of the two (a one-parameter search) and superior to
  classical L2 at high noise. Used here when a mask is available to define the cell
  exterior (see :func:`estimate_noise_variance`).
* **ABL2** (advanced Bayesian L2) — both ``α`` and ``β`` are inferred from the evidence,
  requiring no extra input. Used as the maskless fallback. The paper reports its inferred
  β lands very close to the true noise level.

Only the *resolvable* subspace enters the sums: the forward operator's DC/null modes
(where ``s_i = 0``, e.g. the zeroed k=0 Green's-function block) carry no traction
information, so including them would count the unexplainable mean displacement as noise
and deflate β. Excluding them is equivalent to the mean-subtraction ("standardization")
the paper applies before inference.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _highpass_noise_variance(field: np.ndarray,
                             region: Optional[np.ndarray] = None) -> Optional[float]:
    """Robust (MAD) high-pass estimate of white-noise variance in a smooth 2D field.

    Convolving with the Laplacian mask ``N = [[1,-2,1],[-2,4,-2],[1,-2,1]]`` (Immerkær
    1996) annihilates smooth content and passes noise. For white noise of variance σ²
    the response has variance ``ΣN² · σ² = 36 σ²`` (std ``6σ``). We take the noise std
    from the **median absolute deviation** of the response — ``σ ≈ median(|conv|) /
    (0.6745 · 6)`` — rather than Immerkær's mean, so sparse sharp signal features (a few
    concentrated adhesions) don't inflate the estimate.

    Using a high-pass (rather than the raw variance) is what lets the estimate use the
    cell's *near* exterior: a tight foreground mask leaves the substrate-deformation
    *halo* just outside it, which is real signal that would wreck a raw far-field
    variance — but the halo is smooth, so the Laplacian annihilates it and only the noise
    survives. ``region`` (optional, truthy = use) restricts the MAD to those pixels (e.g.
    the mask exterior); when ``None`` the whole field is used. Returns ``σ²`` (or ``None``
    if too few samples).
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
            sel = np.abs(conv).ravel()   # exterior too small: fall back to the full field
    else:
        sel = np.abs(conv).ravel()
    # std(conv) = 6σ for white noise; recover it robustly from the MAD (0.6745 = the
    # MAD-to-std factor for a Gaussian), so sparse signal outliers don't dominate.
    sigma = float(np.median(sel)) / (0.6745 * 6.0)
    var = sigma * sigma
    return var if np.isfinite(var) and var > 0.0 else None


def estimate_noise_variance(displacement_frame: np.ndarray,
                            mask: Optional[np.ndarray]) -> Optional[float]:
    """Per-component displacement noise variance ``σ_r²`` — BL2's measured ``1/β``.

    Huang et al. estimate the noise from displacement in regions far from any cell. We
    realize that with the MAD high-pass estimator (:func:`_highpass_noise_variance`):
    when a foreground ``mask`` is present it is restricted to the cell exterior (so it
    reads the noise where the cell isn't, as the paper prescribes), and because the
    estimator is a high-pass it tolerates the near-cell displacement *halo* that would
    corrupt a raw far-field variance. Without a mask it runs over the whole field — TFM
    displacement is smooth, so its high-frequency content is noise. Either way BL2 (the
    paper's more robust method) stays usable instead of the fragile parameter-free ABL2.
    Returns the mean of the two components' variances (the per-component ``σ_r²`` the
    isotropic Gaussian noise model expects).

    Args:
        displacement_frame: ``(H, W, 2)`` displacement (``[...,0] = u_x``), any unit.
        mask: ``(H, W)`` foreground mask (truthy = cell) restricting the estimate to the
            exterior, or ``None`` to use the whole field.

    Returns:
        Per-component real-space noise variance ``σ_r²`` in the displacement's units²,
        or ``None`` if it cannot be estimated (caller then infers the noise via ABL2).
    """
    region = None
    if mask is not None and np.asarray(mask).shape == displacement_frame.shape[:2]:
        region = ~(np.asarray(mask) > 0)     # cell exterior
    vx = _highpass_noise_variance(displacement_frame[..., 0], region)
    vy = _highpass_noise_variance(displacement_frame[..., 1], region)
    if vx is None or vy is None:
        return None
    return 0.5 * (vx + vy)


def evidence_optimal_lambda(s: np.ndarray, data_coef: np.ndarray, *,
                            noise_var_fourier: Optional[float] = None,
                            n_grid: int = 120) -> Tuple[float, dict]:
    """Evidence-maximizing ridge ``λ = α/β`` for the SVD-diagonalized problem.

    Operates on the singular values ``s`` and the data projected onto the left singular
    vectors ``data_coef`` (``= Uᴴ · û``), both aligned per SVD mode. Returns the additive
    ridge ``λ`` that goes on ``s²`` — i.e. the square of FTTC's stored ``regularization``
    dial, since the force path applies ``regularization**2`` as the Tikhonov term.

    The log-evidence (Huang et al. Eq. 8) is maximized *directly in 1-D over* ``λ`` rather
    than by the coupled α/β fixed-point of the MacKay updates: with the operator's many
    near-null noise modes the fixed-point iteration is unstable (it can run λ off to
    ∞), whereas the 1-D objective below is smooth and bracketed. Working in the resolved
    subspace (``s > 0``) with ``m`` modes, the log-evidence reduces (up to constants) to

        BL2  (β pinned):  ℓ(λ) = −½ β λ Σ dᵢ²/(sᵢ²+λ)  +  (m/2) log λ  −  ½ Σ log(λ+sᵢ²)
        ABL2 (β profiled): ℓ(λ) = −(m/2) log Σ dᵢ²/(sᵢ²+λ)  −  ½ Σ log(λ+sᵢ²)

    (the ABL2 form follows by maximizing ℓ over β in closed form, ``β* = m /
    (λ Σ dᵢ²/(sᵢ²+λ))``, and substituting back). Both are maximized on a log-λ grid
    bracketed by the singular-value spectrum, then refined by a bounded scalar optimizer.

    Args:
        s: singular values of the forward operator (real, ``≥ 0``), any length.
        data_coef: data projected onto the left singular vectors, same length as ``s``
            (complex allowed; only ``|·|²`` is used).
        noise_var_fourier: per-coefficient noise variance in the SVD/Fourier basis. When
            given, ``β = 1/noise_var_fourier`` is pinned (**BL2**) and only ``α`` (⇔ λ)
            is inferred; when ``None``, ``β`` is profiled out and both are inferred from
            the evidence (**ABL2**).
        n_grid: number of log-spaced λ samples used to bracket the maximum before refining.

    Returns:
        ``(lam, info)`` where ``lam`` is the ridge on ``s²`` and ``info`` records the
        inferred ``alpha``/``beta``/``lam``, the method label (``"BL2"`` / ``"ABL2"``),
        and the number of resolved modes ``m_eff``.
    """
    from scipy import optimize

    s2 = np.asarray(s, dtype=np.float64) ** 2
    d2 = np.abs(np.asarray(data_coef)) ** 2

    # Resolvable subspace only: null/DC modes (s == 0) carry no traction information, so
    # counting their residual as noise would deflate β. Dropping them == the paper's
    # mean-subtraction standardization for the DC mode.
    resolvable = s2 > 0.0
    s2 = s2[resolvable]
    d2 = d2[resolvable]
    m_eff = int(s2.size)

    method = "BL2" if (noise_var_fourier and noise_var_fourier > 0.0) else "ABL2"
    if m_eff == 0 or d2.sum() <= 0.0:
        # Degenerate input (no signal / no resolvable modes): hand back a harmless
        # small ridge rather than a NaN.
        return 1e-12, {"alpha": float("nan"), "beta": float("nan"), "lam": 1e-12,
                       "converged": False, "m_eff": m_eff, "method": method}

    beta_fixed = (1.0 / float(noise_var_fourier)) if method == "BL2" else None

    def neg_log_evidence(log_lam: float) -> float:
        lam = float(np.exp(log_lam))
        A = float((d2 / (s2 + lam)).sum())            # Σ dᵢ²/(sᵢ²+λ)
        logdet = float(np.log(lam + s2).sum())        # Σ log(λ+sᵢ²)
        if beta_fixed is not None:
            ll = -0.5 * beta_fixed * lam * A + 0.5 * m_eff * log_lam - 0.5 * logdet
        else:
            ll = -0.5 * m_eff * np.log(max(A, 1e-300)) - 0.5 * logdet
        return -ll

    # Bracket λ by the singular-value spectrum: from well below the smallest resolved
    # s² (weak regularization) to well above the largest (strong). Grid-scan for the
    # global basin, then refine within the winning cell.
    lo = np.log(max(s2.min(), 1e-300)) - 6.0 * np.log(10.0)
    hi = np.log(s2.max()) + 6.0 * np.log(10.0)
    grid = np.linspace(lo, hi, int(n_grid))
    vals = np.array([neg_log_evidence(g) for g in grid])
    k = int(vals.argmin())
    a = grid[max(k - 1, 0)]
    b = grid[min(k + 1, grid.size - 1)]
    res = optimize.minimize_scalar(neg_log_evidence, bounds=(a, b), method="bounded")
    log_lam = float(res.x) if res.success else float(grid[k])
    lam = float(np.exp(log_lam))

    # Recover α, β for reporting (BL2: β pinned; ABL2: β from the profiled optimum).
    A = float((d2 / (s2 + lam)).sum())
    beta = beta_fixed if beta_fixed is not None else m_eff / max(lam * A, 1e-300)
    alpha = lam * beta
    info = {"alpha": alpha, "beta": beta, "lam": lam, "converged": bool(res.success),
            "m_eff": m_eff, "method": method}
    logger.info("Bayesian L2 (%s): lam=%.6g (reg=%.6g) alpha=%.3g beta=%.3g m_eff=%d",
                method, lam, float(np.sqrt(lam)), alpha, beta, m_eff)
    return lam, info
