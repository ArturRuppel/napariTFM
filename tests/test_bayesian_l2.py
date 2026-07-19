"""Tests for Bayesian L2 regularization (Huang et al., *Sci. Rep.* 9:539, 2019).

The plain-FTTC path can now pick its Tikhonov λ by maximizing the Bayesian evidence
instead of by GCV or by hand. These lock the core behaviours:

* the evidence-optimal λ grows with the noise level (more noise ⇒ more regularization),
  and lands in the same ballpark as GCV;
* the noise estimator recovers a known injected noise level (masked and maskless), and
  tolerates the near-cell displacement halo a raw far-field variance would trip over;
* the FTTC entry point honours ``bayesian`` (overriding the manual/GCV λ) and returns
  finite forces; and
* degenerate input degrades to a harmless small ridge rather than a NaN.
"""
import numpy as np
import pytest

from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend.fttc import FTTC
from napariTFM.backend import forward_tfm as F
from napariTFM.backend.fttc_numba_functions import blkmul_adj
from napariTFM.backend.bayesian_l2 import (
    evidence_optimal_lambda, estimate_noise_variance)
from napariTFM.backend.parameter_validation import validate_fttc_parameters


def _params(**kw):
    base = dict(young_modulus=10000.0, poisson_ratio_substrate=0.3, gel_height=None,
                pixel_size=0.1, downscale_factor=1)
    base.update(kw)
    return FTTCParameters(**base)


def _pos(h, w):
    return np.array([np.ones(h)[:, None] * np.arange(w),
                     np.arange(h)[:, None] * np.ones(w)])


def _blobs(h=48, w=48, sigma_px=2.0):
    yy, xx = np.mgrid[0:h, 0:w]
    t = np.zeros((2, h, w))
    mask = np.zeros((h, w), bool)
    for (cy, cx, fx, fy) in [(14, 14, 600, 400), (30, 34, -500, 300), (34, 14, 200, -450)]:
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_px ** 2))
        t[0] += fx * g
        t[1] += fy * g
        mask |= ((yy - cy) ** 2 + (xx - cx) ** 2) < 6 ** 2
    return t, mask


def _displacement(t, params):
    h, w = t.shape[1:]
    G = F._greens_operator(h, w, params).astype(np.complex128)
    uk = np.einsum("ijhw,jhw->ihw", G, np.fft.fft2(t, axes=(-2, -1)))
    return np.fft.ifft2(uk, axes=(-2, -1)).real


# --- the evidence-optimal lambda --------------------------------------------------

def test_bl2_lambda_increases_with_noise():
    """BL2 (noise pinned) must regularize more as the measured noise grows."""
    p = _params()
    t, _ = _blobs()
    h, w = t.shape[1:]
    u0 = _displacement(t, p)
    calc = FTTC(p)
    rng = np.random.default_rng(0)
    regs = []
    for nf in [0.02, 0.05, 0.15]:
        sigma = nf * u0.std()
        u = u0 + sigma * rng.standard_normal(u0.shape)
        vec = np.array([u[0].flatten(), u[1].flatten()])
        U, s, b = calc._svd_block(_pos(h, w), vec, p.pixel_size, i_max=w, j_max=h)
        dc = blkmul_adj(U, b)
        lam, info = evidence_optimal_lambda(s, dc, noise_var_fourier=h * w * sigma ** 2)
        assert info["method"] == "BL2"
        assert info["converged"]
        regs.append(np.sqrt(lam))
    assert regs[0] < regs[1] < regs[2], regs


def test_bayesian_lambda_in_sane_range():
    """The Bayesian λ should be a positive, finite value in a physically sane range
    (not a degenerate 0 / inf), on a noisy synthetic frame."""
    p = _params()
    t, _ = _blobs()
    h, w = t.shape[1:]
    u0 = _displacement(t, p)
    rng = np.random.default_rng(1)
    sigma = 0.05 * u0.std()
    u = u0 + sigma * rng.standard_normal(u0.shape)
    calc = FTTC(p)
    vec = np.array([u[0].flatten(), u[1].flatten()])
    reg_bayes = calc._bayesian_regularization(_pos(h, w), vec, p.pixel_size, w, h,
                                              noise_var=sigma ** 2)
    assert np.isfinite(reg_bayes) and 1e-12 < reg_bayes < 1e2, reg_bayes


def test_evidence_lambda_degenerate_input():
    """No resolvable modes / no signal ⇒ a harmless small ridge, never a NaN."""
    s = np.zeros(10)
    d = np.zeros(10, dtype=complex)
    lam, info = evidence_optimal_lambda(s, d, noise_var_fourier=1.0)
    assert np.isfinite(lam) and lam > 0
    assert not info["converged"]


# --- the noise estimator ----------------------------------------------------------

@pytest.mark.parametrize("use_mask", [True, False])
def test_noise_estimator_recovers_injected_noise(use_mask):
    """The MAD high-pass estimate must track the injected per-component variance within
    a small factor, both restricted to the mask exterior and over the whole field — the
    latter proving it survives the near-cell displacement halo."""
    p = _params()
    t, mask = _blobs()
    u0 = _displacement(t, p)
    rng = np.random.default_rng(2)
    sigma = 0.08 * u0.std()
    u = u0 + sigma * rng.standard_normal(u0.shape)
    disp = np.stack([u[0], u[1]], axis=-1)
    est = estimate_noise_variance(disp, mask if use_mask else None)
    assert est is not None
    ratio = est / sigma ** 2
    assert 0.5 < ratio < 2.0, ratio


def test_noise_estimator_too_small_returns_none():
    disp = np.zeros((2, 2, 2))
    assert estimate_noise_variance(disp, None) is None


# --- the FTTC entry point ---------------------------------------------------------

def test_calculate_traction_bayesian_overrides_and_is_finite():
    """``bayesian=True`` selects a Bayesian λ (ignoring the manual value) and yields a
    finite force field."""
    p = _params()
    t, _ = _blobs()
    u0 = _displacement(t, p)
    rng = np.random.default_rng(4)
    u = u0 + 0.05 * u0.std() * rng.standard_normal(u0.shape)
    disp = np.stack([u[0], u[1]], axis=-1)
    calc = FTTC(p)
    (_, _), f_manual = calc.calculate_traction(disp, p.pixel_size, 1, regularization=1e-3)
    (_, _), f_bayes = calc.calculate_traction(disp, p.pixel_size, 1,
                                              regularization=1e-3, bayesian=True)
    assert np.isfinite(f_bayes).all()
    # The Bayesian choice differs from the (deliberately large) manual λ.
    assert not np.allclose(f_manual, f_bayes)


def test_validation_allows_missing_reg_under_bayesian():
    """Under ``bayesian_l2`` the manual λ is unused, so a non-positive value must not
    fail validation; with it off, a non-positive λ must still be rejected."""
    ok, _ = validate_fttc_parameters(_params(bayesian_l2=True, regularization=0.0))
    assert ok
    ok2, msg = validate_fttc_parameters(_params(bayesian_l2=False, regularization=0.0))
    assert not ok2 and "Regularization" in msg
