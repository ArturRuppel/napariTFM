"""Tests for the elastic-net (``l2_ridge``) extension of the sparse group-L1 solver.

Huang et al. (*Sci. Rep.* 9:539, 2019) find the elastic net — L1 sparsity plus an L2
ridge — the most accurate TFM regularizer: the L1 term keeps a clean background while
the L2 term reins in the peak-traction overshoot pure L1 leaves. Here ``l2_ridge`` adds
that ridge to the FISTA solve as ``½ λ₂‖t‖²``, with λ₂ a scene-independent fraction of
the median per-mode data curvature.

These lock the defining behaviour: ``l2_ridge = 0`` is exactly the pure-L1 solve, the
ridge shrinks the recovered traction monotonically (peak and total energy), and the
sparsity dial keeps its meaning (λ₁_max is unchanged, so ``l1_sparsity = 1`` still
empties the field regardless of the ridge).
"""
import dataclasses

import numpy as np
import pytest

from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend import forward_l1 as L1
from napariTFM.backend import forward_tfm as F


def _params(**kw):
    base = dict(l1_sparsity=0.05, l1_max_iter=800, young_modulus=5000.0,
                poisson_ratio_substrate=0.5, gel_height=None, pixel_size=0.1,
                downscale_factor=1, fwd_fit_margin_um=1.0, fwd_mask_strength=0.0,
                fwd_device="cpu", fwd_dtype="float64")
    base.update(kw)
    return FTTCParameters(**base)


def _forward_displacement(t, params):
    """u = P(t): synthesize a displacement field from a known traction via the same
    Green's operator the L1 solver inverts. (2,H,W) Pa → (H,W,2) µm."""
    h, w = t.shape[1:]
    G = F._greens_operator(h, w, params).astype(np.complex128)
    uk = np.einsum("ijhw,jhw->ihw", G, np.fft.fft2(t, axes=(-2, -1)))
    u = np.fft.ifft2(uk, axes=(-2, -1)).real
    return np.stack([u[0], u[1]], axis=-1)


def _scene(seed=0, noise=0.04):
    p = _params()
    h = w = 32
    rng = np.random.default_rng(seed)
    t = np.zeros((2, h, w))
    for (yy, xx, fx, fy) in [(10, 10, 800, 600), (20, 22, -500, 400)]:
        t[0, yy, xx] = fx
        t[1, yy, xx] = fy
    disp = _forward_displacement(t, p)
    disp = disp + noise * disp.std() * rng.standard_normal(disp.shape)
    return p, t, disp


def _peak(a):
    return float(np.sqrt(a[0] ** 2 + a[1] ** 2).max())


def _energy(a):
    return float((a[0] ** 2 + a[1] ** 2).sum())


def test_zero_ridge_is_pure_l1():
    """``l2_ridge = 0`` must reproduce the pure group-L1 solve bit-for-bit (the ridge is
    a strict add-on that vanishes at the dial's zero)."""
    p, _, disp = _scene()
    base = L1.l1_traction_frame(disp, p)
    zero_ridge = L1.l1_traction_frame(disp, dataclasses.replace(p, l2_ridge=0.0))
    np.testing.assert_array_equal(base, zero_ridge)


def test_ridge_shrinks_peak_and_energy_monotonically():
    """Raising ``l2_ridge`` monotonically shrinks the recovered traction — the elastic
    net's peak-overshoot control (Huang et al.). Peak magnitude and total energy both
    decrease as the ridge grows."""
    p, _, disp = _scene()
    peaks, energies = [], []
    for frac in [0.0, 0.1, 0.3, 0.7]:
        t = L1.l1_traction_frame(disp, dataclasses.replace(p, l2_ridge=frac))
        peaks.append(_peak(t))
        energies.append(_energy(t))
    # Strictly non-increasing, and the strongest ridge is well below pure L1.
    assert all(b <= a + 1e-6 for a, b in zip(peaks, peaks[1:])), peaks
    assert all(b <= a + 1e-6 for a, b in zip(energies, energies[1:])), energies
    assert peaks[-1] < peaks[0]
    assert energies[-1] < 0.9 * energies[0]


def test_ridge_leaves_lambda1_max_intact():
    """The ridge vanishes at ``t = 0``, so it must not change λ₁_max: at ``l1_sparsity =
    1`` the field is emptied whether or not a ridge is present."""
    p, _, disp = _scene()
    for frac in [0.0, 0.5]:
        t = L1.l1_traction_frame(
            disp, dataclasses.replace(p, l1_sparsity=1.0, l2_ridge=frac))
        assert _peak(t) < 1e-6, (frac, _peak(t))


def test_ridge_solution_is_finite():
    p, _, disp = _scene(noise=0.1)
    t = L1.l1_traction_frame(disp, dataclasses.replace(p, l2_ridge=0.5))
    assert np.isfinite(t).all()
