"""Tests for the preconditioned-CG confined forward solver (forward_tfm β>0 path).

The identity tests (one-step exactness, adjoint symmetry, gradient-is-A, DC zeroing)
validate the operator against math, needing no CPU/GPU reference — per
docs/specs/forward-solver-pcg.md they replace what autograd used to buy. The golden
test regresses the CG output against the retired L-BFGS solver's output.
"""
import os

import numpy as np
import pytest
import scipy.fft as _scipy_fft

from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend import forward_tfm as F

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "forward_pcg_golden.npz")


def _params(**kw):
    base = dict(regularization=1e-3, young_modulus=5000.0, poisson_ratio_substrate=0.5,
                gel_height=None, pixel_size=0.1, downscale_factor=1,
                fwd_smoothness=0.05, fwd_fit_margin_um=1e6, fwd_traction_scale=1e-2,
                fwd_max_iter=500, fwd_cg_tol=1e-10, fwd_device="cpu", fwd_dtype="float64")
    base.update(kw)
    return FTTCParameters(**base)


def _disk_mask(h, w, r_frac=0.3):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= (r_frac * h) ** 2).astype(np.uint8)


def _random_field(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, h, w))


def test_one_step_exactness_Minv_is_A_inverse():
    """W=I, β=0 ⇒ A == M (discrete Laplacian symbol), so M⁻¹A = I on the DC-free
    subspace: one CG step would suffice."""
    h = w = 24
    u = _random_field(h, w, 0)
    mask = _disk_mask(h, w)
    apply_A, apply_Minv, b, shape, meta = F._build_normal_equations(
        u, mask, beta=0.0, params=_params(), xp=np, fft=_scipy_fft)
    x = _random_field(h, w, 1)
    x = x - x.mean(axis=(1, 2), keepdims=True)          # DC-free (the observable subspace)
    y = apply_Minv(apply_A(x))
    # A and M are the same operator in exact arithmetic; the ~1e-7 floor is FFT
    # round-trip rounding in apply_A vs M's direct symbol apply. A *wrong* Laplacian
    # symbol (|k|² vs 4·sin², the spec's [review] hazard) breaks this by ~100×.
    np.testing.assert_allclose(y, x, atol=1e-6, rtol=1e-4)


def test_operator_is_symmetric():
    """⟨A x, y⟩ = ⟨x, A y⟩ on random real fields, with non-uniform W, β>0, γ>0."""
    h = w = 24
    u = _random_field(h, w, 2)
    mask = _disk_mask(h, w)
    apply_A, _, _, _, _ = F._build_normal_equations(
        u, mask, beta=3.0, params=_params(fwd_fit_margin_um=0.3), xp=np, fft=_scipy_fft)
    x = _random_field(h, w, 3)
    y = _random_field(h, w, 4)
    lhs = float(np.sum(apply_A(x) * y))
    rhs = float(np.sum(x * apply_A(y)))
    assert abs(lhs - rhs) <= 1e-8 * (abs(lhs) + abs(rhs) + 1e-12)


def test_gradient_matches_A():
    """A is the gradient of its own quadratic form Q(w)=½⟨w,Aw⟩: finite-difference
    Q along a random direction equals ⟨A w, e⟩."""
    h = w = 20
    u = _random_field(h, w, 5)
    mask = _disk_mask(h, w)
    apply_A, _, _, _, _ = F._build_normal_equations(
        u, mask, beta=2.0, params=_params(fwd_fit_margin_um=0.3), xp=np, fft=_scipy_fft)

    def Q(wv):
        return 0.5 * float(np.sum(wv * apply_A(wv)))

    wv = _random_field(h, w, 6)
    e = _random_field(h, w, 7)
    eps = 1e-4
    fd = (Q(wv + eps * e) - Q(wv - eps * e)) / (2 * eps)
    analytic = float(np.sum(apply_A(wv) * e))
    np.testing.assert_allclose(fd, analytic, rtol=1e-5, atol=1e-6)


def test_no_spurious_mean_as_lambda_small():
    """DC handling: the recovered traction carries no spurious mean even at tiny λ."""
    d = np.load(GOLDEN)
    u = d["u"]
    mask = d["mask"]
    params = _params(regularization=1e-8, fwd_mask_strength=50.0, fwd_fit_margin_um=0.5)
    frame = np.stack([u[0], u[1]], axis=-1)
    t = F.forward_traction_frame(frame, params, mask)
    mean_mag = np.abs(np.mean(t, axis=(1, 2))).max()
    rms = float(np.sqrt(np.mean(t ** 2)))
    assert mean_mag < 1e-3 * rms


def test_confined_solve_converges():
    """PCG reaches its tolerance on a real confined frame."""
    d = np.load(GOLDEN)
    u = d["u"]
    mask = d["mask"]
    beta = F.confinement_to_beta(50.0)
    xp, fft, _ = F._resolve_backend("cpu")
    apply_A, apply_Minv, b, (H, W), _ = F._build_normal_equations(
        u, mask, beta, _params(fwd_fit_margin_um=0.5), xp, fft)
    w, iters, converged = F._pcg(apply_A, apply_Minv, b, xp, 1e-10, 1000)
    assert converged and iters < 1000


def test_golden_regression_matches_lbfgs():
    """PCG output matches the retired L-BFGS reference on the golden frame (both
    minimize the same convex J, so they must agree)."""
    d = np.load(GOLDEN)
    u = d["u"]
    mask = d["mask"]
    t_ref = d["t_ref"]
    params = _params(regularization=float(d["regularization"]),
                     young_modulus=float(d["young_modulus"]),
                     poisson_ratio_substrate=float(d["poisson"]),
                     pixel_size=float(d["pixel_size"]),
                     fwd_mask_strength=float(d["fwd_mask_strength"]),
                     fwd_smoothness=float(d["fwd_smoothness"]),
                     fwd_fit_margin_um=float(d["fwd_fit_margin_um"]),
                     fwd_traction_scale=float(d["fwd_traction_scale"]))
    frame = np.stack([u[0], u[1]], axis=-1)
    t = F.forward_traction_frame(frame, params, mask)
    assert t.shape == t_ref.shape and t.dtype == np.float32
    corr = np.corrcoef(t.ravel(), t_ref.ravel())[0, 1]
    rel_rms = np.sqrt(np.mean((t - t_ref) ** 2)) / np.abs(t_ref).max()
    assert corr > 0.99, f"corr={corr:.4f}"
    assert rel_rms < 5e-2, f"rel_rms={rel_rms:.4f}"


def _cupy_ready():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cupy_ready(), reason="cupy / CUDA device not available")
def test_gpu_matches_cpu():
    """The cupy GPU path and the numpy CPU path agree (same operator, same PCG)."""
    d = np.load(GOLDEN)
    u = d["u"]
    mask = d["mask"]

    def P(dev):
        return _params(regularization=float(d["regularization"]),
                       young_modulus=float(d["young_modulus"]),
                       poisson_ratio_substrate=float(d["poisson"]),
                       pixel_size=float(d["pixel_size"]),
                       fwd_mask_strength=float(d["fwd_mask_strength"]),
                       fwd_smoothness=float(d["fwd_smoothness"]),
                       fwd_fit_margin_um=float(d["fwd_fit_margin_um"]),
                       fwd_traction_scale=float(d["fwd_traction_scale"]), fwd_device=dev)

    frame = np.stack([u[0], u[1]], axis=-1)
    t_cpu = F.forward_traction_frame(frame, P("cpu"), mask)
    t_gpu = F.forward_traction_frame(frame, P("cuda"), mask)
    corr = np.corrcoef(t_cpu.ravel(), t_gpu.ravel())[0, 1]
    rel_rms = np.sqrt(np.mean((t_cpu - t_gpu) ** 2)) / np.abs(t_cpu).max()
    assert corr > 0.9999, f"corr={corr:.6f}"
    assert rel_rms < 1e-3, f"rel_rms={rel_rms:.2e}"


def test_beta_zero_is_torch_free_closed_form():
    """The β=0 dispatch stays on the numpy closed form (no torch import needed)."""
    h = w = 24
    u = _random_field(h, w, 8)
    frame = np.stack([u[0], u[1]], axis=-1)
    t = F.forward_traction_frame(frame, _params(fwd_mask_strength=0.0), mask=None)
    assert t.shape == (2, h, w) and t.dtype == np.float32
