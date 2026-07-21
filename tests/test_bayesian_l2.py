"""Tests for Bayesian-L2 traction reconstruction (Huang et al., *Sci. Rep.* 9:539, 2019).

``bayesian_l2`` is a dedicated real-space, column-standardized, over-determined Tikhonov
reconstruction whose regularization is chosen automatically by evidence maximization -- not a λ
fed to FTTC. These lock the core behaviours:

* the evidence-inferred λ grows with the noise level (more noise ⇒ more regularization);
* the reconstruction recovers a known synthetic traction (right location, sign, magnitude scale),
  both parameter-free (ABL2) and with a measured-noise β (BL2);
* the noise estimator recovers a known injected level (masked and maskless), tolerating the
  near-cell displacement halo a raw far-field variance would trip over;
* the FTTC dispatch runs the Bayesian path and returns finite forces; and
* the GCV auto-λ suggestion for the manual path is a sane positive value.
"""
import numpy as np
import pytest

from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend.fttc import calculate_force_field, find_bayesian_regularization
from napariTFM.backend import forward_tfm as F
from napariTFM.backend.bayesian_l2 import (
    reconstruct_bl2_frame, estimate_bayesian_lambda, _boussinesq_M, _solve_evidence,
    estimate_noise_variance)
from napariTFM.backend.parameter_validation import validate_fttc_parameters


def _params(**kw):
    # l1_sparsity=0 so dispatch reaches the FTTC/Bayesian paths this module exercises; the
    # shipped default (0.05) would route every call to the group-L1 solver instead.
    base = dict(young_modulus=10000.0, poisson_ratio_substrate=0.3, gel_height=None,
                pixel_size=0.1, downscale_factor=1, l1_sparsity=0.0)
    base.update(kw)
    return FTTCParameters(**base)


def _blobs(h=64, w=64, sigma_px=4.0):
    yy, xx = np.mgrid[0:h, 0:w]
    t = np.zeros((2, h, w))
    mask = np.zeros((h, w), bool)
    centres = [(20, 20, 600, 400), (42, 46, -500, 300), (46, 20, 200, -450)]
    for (cy, cx, fx, fy) in centres:
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_px ** 2))
        t[0] += fx * g
        t[1] += fy * g
        mask |= ((yy - cy) ** 2 + (xx - cx) ** 2) < (2 * sigma_px) ** 2
    return t, mask, centres


def _displacement(t, params):
    h, w = t.shape[1:]
    G = F._greens_operator(h, w, params).astype(np.complex128)
    uk = np.einsum("ijhw,jhw->ihw", G, np.fft.fft2(t, axes=(-2, -1)))
    return np.fft.ifft2(uk, axes=(-2, -1)).real


# --- the evidence solver ----------------------------------------------------------

def test_solve_evidence_lambda_increases_with_noise():
    """ABL2's inferred λ must grow as the noise level grows (more noise ⇒ more smoothing)."""
    rng = np.random.default_rng(0)
    # small controlled real-space problem via the same Boussinesq operator the solver uses
    c = (np.arange(20) - 10) * 0.6
    fx, fy = np.meshgrid(c, c)
    force_xy = np.column_stack([fx.ravel(), fy.ravel()])
    cd = (np.arange(28) - 14) * (0.6 * 20 / 28)
    dx, dy = np.meshgrid(cd, cd)
    disp_xy = np.column_stack([dx.ravel(), dy.ravel()])
    M = _boussinesq_M(disp_xy, force_xy, 10000.0, 0.5, 0.6 ** 2)
    ftrue = rng.normal(0, 200, M.shape[1])
    u0 = M @ ftrue
    sd = M.std(0); sd[sd == 0] = 1.0
    Xs = (M - M.mean(0)) / sd
    lams = []
    for frac in [0.02, 0.10]:
        u = u0 + rng.normal(0, frac * u0.std(), u0.size)
        _, info = _solve_evidence(Xs, u - u.mean())
        assert info["method"] == "ABL2"
        lams.append(info["lam"])
    assert lams[1] > lams[0], lams


# --- the reconstruction -----------------------------------------------------------

@pytest.mark.parametrize("use_mask", [False, True])
def test_reconstruct_bl2_recovers_traction(use_mask):
    """ABL2 (no mask) and BL2 (mask) both recover a known blob traction: finite, correctly
    located main peak, correct sign there, and correlated with the ground truth."""
    p = _params()
    t, mask, centres = _blobs()
    u = _displacement(t, p)
    disp = np.stack([u[0], u[1]], axis=-1)
    that = reconstruct_bl2_frame(disp, p.young_modulus, p.poisson_ratio_substrate,
                                 p.pixel_size, mask=(mask if use_mask else None), n_force=16)
    assert that.shape == (2, *t.shape[1:])
    assert np.isfinite(that).all()
    mag_hat = np.hypot(that[0], that[1])
    mag_gt = np.hypot(t[0], t[1])
    # main peak recovered near a true blob centre
    py, px = np.unravel_index(int(mag_hat.argmax()), mag_hat.shape)
    assert min((py - cy) ** 2 + (px - cx) ** 2 for cy, cx, *_ in centres) < 8 ** 2
    # x-traction sign at the strongest (positive-fx) blob matches
    cy, cx = centres[0][:2]
    assert that[0, cy, cx] > 0
    # correlated with ground truth
    r = np.corrcoef(mag_hat.ravel(), mag_gt.ravel())[0, 1]
    assert r > 0.4, r


def test_reconstruct_bl2_magnitude_scale():
    """Recovered peak traction is within a factor of a few of ground truth (no gross bias)."""
    p = _params()
    t, _, _ = _blobs()
    disp = np.moveaxis(_displacement(t, p), 0, -1)
    that = reconstruct_bl2_frame(disp, p.young_modulus, p.poisson_ratio_substrate,
                                 p.pixel_size, mask=None, n_force=16)
    ratio = np.hypot(that[0], that[1]).max() / np.hypot(t[0], t[1]).max()
    assert 0.2 < ratio < 3.0, ratio


# --- the noise estimator ----------------------------------------------------------

@pytest.mark.parametrize("use_mask", [True, False])
def test_noise_estimator_recovers_injected_noise(use_mask):
    p = _params()
    t, mask, _ = _blobs()
    u0 = _displacement(t, p)
    rng = np.random.default_rng(2)
    sigma = 0.08 * u0.std()
    u = u0 + sigma * rng.standard_normal(u0.shape)
    disp = np.stack([u[0], u[1]], axis=-1)
    est = estimate_noise_variance(disp, mask if use_mask else None)
    assert est is not None
    assert 0.5 < est / sigma ** 2 < 2.0, est / sigma ** 2


def test_noise_estimator_too_small_returns_none():
    assert estimate_noise_variance(np.zeros((2, 2, 2)), None) is None


# --- the FTTC dispatch + GCV button -----------------------------------------------

def test_calculate_force_field_bayesian_dispatch_is_finite():
    """``bayesian_l2=True`` runs the Bayesian reconstruction and yields finite forces of the
    input shape, distinct from a plain-FTTC result at a fixed λ."""
    p_bayes = _params(bayesian_l2=True)
    p_manual = _params(regularization=1e-3)
    t, _, _ = _blobs()
    disp = np.moveaxis(_displacement(t, p_bayes), 0, -1)[np.newaxis]
    fb = _run(calculate_force_field(disp, p_bayes))
    fm = _run(calculate_force_field(disp, p_manual))
    assert fb.force_field.shape == (1, *t.shape[1:], 2)
    assert np.isfinite(fb.force_field).all()
    assert not np.allclose(fb.force_field, fm.force_field)


def test_find_bayesian_regularization_sane():
    """The auto-λ button estimates a positive, finite Bayesian ridge on a real frame."""
    p = _params()
    t, _, _ = _blobs()
    disp = np.moveaxis(_displacement(t, p), 0, -1)
    lam = find_bayesian_regularization(disp, p)
    assert np.isfinite(lam) and lam > 0, lam


def test_frozen_lambda_reused_across_frames():
    """A λ estimated on one frame, frozen, and applied to a *different* frame yields a finite
    reconstruction — and the same λ on the same frame reproduces the free-inference result
    (the operator is shared, so reuse is exact)."""
    p = _params()
    t1, _, _ = _blobs()
    t2, _, _ = _blobs(sigma_px=5.0)          # a different frame (different traction)
    d1 = np.moveaxis(_displacement(t1, p), 0, -1)
    d2 = np.moveaxis(_displacement(t2, p), 0, -1)
    E, nu, ps = p.young_modulus, p.poisson_ratio_substrate, p.pixel_size
    lam = estimate_bayesian_lambda(d1, E, nu, ps, n_force=16)
    # reuse the same λ on a different frame -> finite traction, same shape
    t2_hat = reconstruct_bl2_frame(d2, E, nu, ps, n_force=16, lam=lam)
    assert t2_hat.shape == (2, *t2.shape[1:]) and np.isfinite(t2_hat).all()
    # freezing λ on frame 1 reproduces frame 1's free inference (identical operator + data)
    free = reconstruct_bl2_frame(d1, E, nu, ps, n_force=16)
    frozen = reconstruct_bl2_frame(d1, E, nu, ps, n_force=16, lam=lam)
    assert np.allclose(free, frozen, atol=1e-4 * np.abs(free).max())


def test_validation_allows_missing_reg_under_bayesian():
    ok, _ = validate_fttc_parameters(_params(bayesian_l2=True, regularization=0.0))
    assert ok
    ok2, msg = validate_fttc_parameters(_params(bayesian_l2=False, regularization=0.0))
    assert not ok2 and "Regularization" in msg


def _run(gen):
    """Drive the calculate_force_field generator to its FTTCResult return value."""
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        return exc.value
