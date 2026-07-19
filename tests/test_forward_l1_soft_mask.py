"""Tests for the sparse (group-L1) solver's *soft* mask support.

The mask used to be a hard support (traction projected to zero outside it every
FISTA iteration), and the resulting step at the mask edge rang (Gibbs). It is now a
soft support: an off-mask L2 penalty ``½ Σ c(x)·|t|²`` added to the objective, with
``c`` zero inside the mask and ramping up to ``confinement_to_beta(fwd_mask_strength)``
outside over a smoothstep collar. This is the same mechanism the L2 confined solver
uses, ported to FISTA.

Why a penalty in the objective and not a per-iteration nudge: the exterior traction
lives in the fit's near-nullspace (off-mask force explains in-mask displacement via
the non-local Green's operator), so scaling the L1 threshold or multiplying the
iterate by a <1 window is *compensated away* by the solver and does not confine —
only a penalty the minimizer must trade against actually suppresses it. These tests
lock: the collar is graded (no cliff), the dial confines monotonically, and the dial
at 0 is exactly pure sparsity.
"""
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
    """u = P(t): the Green's operator the L1 solver inverts, used to synthesize a
    displacement field from a known traction. (2,H,W) Pa → (2,H,W) µm."""
    h, w = t.shape[1:]
    G = F._greens_operator(h, w, params).astype(np.complex128)      # û = G·t̂
    uk = np.einsum("ijhw,jhw->ihw", G, np.fft.fft2(t, axes=(-2, -1)))
    return np.fft.ifft2(uk, axes=(-2, -1)).real


# --- the exterior penalty coefficient: graded, not a cliff ----------------------

def test_exterior_penalty_off_is_zero():
    """Dial at 0 (or no mask) ⇒ no penalty: pure sparsity, unchanged."""
    h = w = 40
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= 10 ** 2).astype(np.uint8)
    valid = np.ones((h, w), dtype=bool)
    p = _params(fwd_mask_strength=0.0)
    np.testing.assert_array_equal(L1._exterior_penalty(mask, valid, 1.0, p, np, np.float64), 0.0)
    np.testing.assert_array_equal(L1._exterior_penalty(None, valid, 1.0, p, np, np.float64), 0.0)


def test_exterior_penalty_graded_without_a_cliff():
    """c(x) is exactly 0 inside the mask (force is free there), rises to β·l_data far
    outside, and the transition is a graded smoothstep collar — the largest jump
    between neighbouring pixels is a small fraction of the full rise. A hard support
    would be a full-height single-pixel step (the Gibbs source)."""
    h = w = 64
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= 16 ** 2).astype(np.uint8)
    valid = np.ones((h, w), dtype=bool)
    l_data = 3.0
    p = _params(fwd_mask_strength=60.0)
    c = np.asarray(L1._exterior_penalty(mask, valid, l_data, p, np, np.float64))[0]

    full = F.confinement_to_beta(60.0) * l_data
    assert c[mask > 0].max() == 0.0                          # no penalty anywhere inside
    assert c.max() == pytest.approx(full, rel=1e-6)          # full weight far outside
    assert c[0, 0] == pytest.approx(full, rel=1e-6)
    step = np.abs(np.diff(c, axis=0)).max()
    assert step < 0.5 * full                                 # graded, not a full-height cliff


# --- the solver: soft confinement, monotone in the dial -------------------------

def test_soft_mask_suppresses_excluded_force_monotonically():
    """A net-zero contractile dipole whose mask covers only one lobe. The other lobe
    is genuine exterior force that unconfined L1 recovers; raising the dial must
    monotonically suppress it (the mask says 'no force here'), while the solve stays
    valid. Net-zero force avoids the net-force DC offset the solver spreads uniformly."""
    h = w = 64
    yy, xx = np.mgrid[0:h, 0:w]
    g = lambda cx, cy: np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2)))
    t_true = np.stack([150.0 * (g(24, h / 2) - g(40, h / 2)), np.zeros((h, w))])  # +x | −x
    mask = (((xx - 24) ** 2 + (yy - h / 2) ** 2) <= 8 ** 2).astype(np.uint8)      # left lobe only
    excluded = xx > 32                                                            # the right lobe's region

    u = _forward_displacement(t_true, _params())
    u = u + 0.03 * u.std() * np.random.default_rng(1).standard_normal(u.shape)
    frame = np.moveaxis(u, 0, -1)

    energies = []
    for s in (0.0, 25.0, 50.0, 75.0, 100.0):
        t = L1.l1_traction_frame(frame, _params(fwd_mask_strength=s), mask=mask)
        mag = np.sqrt((t ** 2).sum(axis=0))
        energies.append(float((mag[excluded] ** 2).sum()))
        assert np.isfinite(t).all()

    # Monotone non-increasing, and driven down by well over an order of magnitude.
    assert all(a >= b for a, b in zip(energies, energies[1:])), energies
    assert energies[-1] < 0.05 * energies[0], energies


def test_zero_strength_with_mask_equals_pure_sparsity():
    """Dial at 0 must be indistinguishable from no mask — the mask enters the penalty
    only through fwd_mask_strength (with margin ≫ image the data-fit region is the
    whole field), so a masked-but-off solve == an unmasked solve."""
    h = w = 48
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= 12 ** 2).astype(np.uint8)
    p = _params(fwd_mask_strength=0.0, fwd_fit_margin_um=1e6)
    frame = 0.05 * np.random.default_rng(1).standard_normal((h, w, 2))
    t_masked = L1.l1_traction_frame(frame, p, mask=mask)
    t_free = L1.l1_traction_frame(frame, p, mask=None)
    np.testing.assert_allclose(t_masked, t_free, atol=1e-7)
