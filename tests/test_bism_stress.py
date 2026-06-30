"""Tests for the packaged BISM stress engine and its frame generator."""

import numpy as np
from scipy.sparse.linalg import spsolve

from napariTFM.backend import bism
from napariTFM.backend.parameter_dataclasses import MSMParameters
from napariTFM.backend.stress import StressResult


def _biaxial_traction(R=12, C=12, edge=1000.0):
    """A square in-mask plate pulled inward on all four edges (biaxial).

    Tractions live on the plate edges in both x and y, so both traction
    components carry structure (a finite reconstruction R²); the masked region
    is the full grid. Mirrors the square-plate idea used in the MSM benchmark.
    """
    tx = np.zeros((R, C), dtype=np.float32)
    tx[:, 0] = edge        # left edge,  +x (inward)
    tx[:, -1] = -edge      # right edge, -x (inward)
    ty = np.zeros((R, C), dtype=np.float32)
    ty[0, :] = edge        # top edge,    +y (inward)
    ty[-1, :] = -edge      # bottom edge, -y (inward)
    mask = np.ones((R, C), dtype=bool)
    return tx, ty, mask


def test_compute_bism_stress_runs_masked():
    tx, ty, mask = _biaxial_traction()
    res = bism.compute_bism_stress(tx, ty, l=1.0, mask=mask)
    assert res.sxx.shape == (12, 12)
    # A real, non-trivial stress field comes back (not all zero / NaN).
    assert np.isfinite(res.sxx[mask]).all()
    assert np.nanmax(np.abs(res.sxx[mask])) > 0
    assert 0.0 <= res.r2_traction <= 1.0 + 1e-6


def _noisy_biaxial(noise=50.0, seed=0):
    """Biaxial plate plus seeded white noise on the tractions.

    MAP needs a real noise floor: on a perfectly-fittable (noise-free) field
    there is no genuine noise to estimate and the MAP optimum is degenerate, so
    the estimator finds no stable fixed point and falls back to the fixed
    Lambda. Adding noise gives the residual a floor and a well-posed stable
    fixed point — the realistic regime the method is built for.
    """
    tx, ty, mask = _biaxial_traction()
    rng = np.random.default_rng(seed)
    tx = tx + rng.normal(0.0, noise, tx.shape)
    ty = ty + rng.normal(0.0, noise, ty.shape)
    return tx, ty, mask


def _smooth_fittable_field(R=30, C=30, l=0.3, seed=4):
    """A smooth traction field that is exactly the divergence of a stress field.

    Built as ``T = A @ sigma0`` where ``sigma0`` solves the BISM system at a
    tiny Lambda for a few low-frequency Fourier tractions — so a near-perfect
    fit exists (like the real benchmark monolayers) *and* the underlying stress
    is smooth, hence its prior norm is small. That combination places the
    natural Lambda far below BISM.m's hardcoded start of 1e-3, which is exactly
    the regime where the bare fixed-point iteration diverges *upward*
    (Lambda -> ~1e25) and over-regularizes the stress to zero. The robust
    estimator must not do that. Returns full-grid ``(tx, ty)``.
    """
    A = bism._build_A(R, C, l)
    B = bism._build_B(R, C, True, 1e3, 1e3)
    yy, xx = np.mgrid[0:R, 0:C].astype(float)
    rng = np.random.default_rng(seed)
    tx = np.zeros((R, C)); ty = np.zeros((R, C))
    for kx, ky in [(1, 0), (0, 1), (1, 1)]:
        tx += rng.normal() * np.sin(np.pi * (kx + 1) * xx / C) * np.cos(np.pi * ky * yy / R)
        ty += rng.normal() * np.cos(np.pi * kx * xx / C) * np.sin(np.pi * (ky + 1) * yy / R)
    tx *= 1000.0 / np.abs(tx).max(); ty *= 1000.0 / np.abs(ty).max()
    T = np.concatenate([tx.ravel(), ty.ravel()])
    AtA = (A.T @ A).tocsc()
    sigma0 = spsolve((1e-8 * B + (l ** 2) * AtA).tocsc(), (l ** 2) * (A.T @ T))
    T2 = A @ sigma0
    N = R * C
    return T2[:N].reshape(R, C), T2[N:].reshape(R, C), l


def test_map_lambda_estimation_masked():
    # MAP picks a Lambda from the data (ignoring the passed-in lam) and reports a
    # noise estimate; on noisy data the fixed point converges (does not run away).
    tx, ty, mask = _noisy_biaxial()
    fixed = bism.compute_bism_stress(tx, ty, l=1.0, mask=mask, lam=1e-6)
    mapped = bism.compute_bism_stress(
        tx, ty, l=1.0, mask=mask, lam=1e-6, lam_method="MAP",
    )

    # MAP overrode the supplied lam with its own (finite, positive) estimate.
    assert mapped.lam != fixed.lam
    assert np.isfinite(mapped.lam) and mapped.lam > 0
    # Noise amplitude estimate falls out of the same iteration, in the right
    # ballpark of the injected ~50 Pa (order of magnitude, not exact).
    nv = mapped.meta["noise_value_map"]
    assert nv is not None and np.isfinite(nv)
    assert 5.0 < nv < 500.0
    # Still a sane, finite stress field.
    assert np.isfinite(mapped.sxx[mask]).all()
    assert np.nanmax(np.abs(mapped.sxx[mask])) > 0
    assert 0.0 <= mapped.r2_traction <= 1.0 + 1e-6


def test_map_lambda_estimation_full_grid():
    # The full (unmasked) rectangular path also supports MAP and converges.
    tx, ty, _ = _noisy_biaxial()
    fixed = bism.compute_bism_stress(tx, ty, l=1.0, lam=1e-6)
    mapped = bism.compute_bism_stress(tx, ty, l=1.0, lam=1e-6, lam_method="map")

    assert mapped.lam != fixed.lam
    assert np.isfinite(mapped.lam) and mapped.lam > 0
    assert mapped.meta["noise_value_map"] > 0
    assert np.isfinite(mapped.sxx).all()


def test_map_lambda_runaway_stays_finite():
    # Degenerate case: on a perfectly-fittable (noise-free) field MAP is
    # ill-posed (no genuine noise to estimate, no stable fixed point). The
    # estimator must fall back to the supplied fixed Lambda rather than emit a
    # runaway Lambda or a NaN stress field.
    tx, ty, mask = _biaxial_traction()
    mapped = bism.compute_bism_stress(
        tx, ty, l=1.0, mask=mask, lam=1e-6, lam_method="MAP",
    )
    assert mapped.lam == 1e-6                       # fell back to the fixed lam
    assert mapped.meta["noise_value_map"] is None   # no MAP noise estimate
    assert np.isfinite(mapped.sxx[mask]).all()


def test_map_does_not_collapse_to_zero_field():
    # Regression for the "0 everywhere" bug: BISM.m's bare fixed-point iteration
    # starts from a hardcoded Lambda=1e-3 and, on a smooth real-scale field whose
    # natural Lambda is orders of magnitude smaller, diverges *upward* and
    # over-regularizes the stress to zero (R^2 -> 0). The robust estimator must
    # instead return a stress field that actually reconstructs the tractions.
    tx, ty, l = _smooth_fittable_field()
    fixed = bism.compute_bism_stress(tx, ty, l=l, lam=1e-6)
    mapped = bism.compute_bism_stress(tx, ty, l=l, lam=1e-6, lam_method="MAP")

    # The fixed-lam solve fits this field well; MAP must not do dramatically
    # worse (the bug produced R^2 ~ 0 and an all-but-zero field).
    assert fixed.r2_traction > 0.5
    assert mapped.r2_traction > 0.5
    # Field magnitude is preserved, not collapsed toward zero.
    fixed_rms = float(np.sqrt(np.mean(fixed.sxx ** 2)))
    mapped_rms = float(np.sqrt(np.mean(mapped.sxx ** 2)))
    assert mapped_rms > 0.3 * fixed_rms
    # And of course finite throughout.
    assert np.isfinite(mapped.sxx).all()


def test_calculate_bism_stresses_honors_lambda_method(monkeypatch):
    # The frame generator threads params.bism_lambda_method into the solver.
    tx, ty, mask = _biaxial_traction()
    force_field = np.stack([tx, ty], axis=-1)[np.newaxis, ...]
    masks = mask[np.newaxis, ...]
    params = MSMParameters(stress_method="BISM", bism_lambda_method="MAP")

    seen = {}
    real = bism.compute_bism_stress

    def _spy(*args, **kwargs):
        seen["lam_method"] = kwargs.get("lam_method")
        return real(*args, **kwargs)

    monkeypatch.setattr(bism, "compute_bism_stress", _spy)

    gen = bism.calculate_bism_stresses(force_field, masks, params)
    try:
        while True:
            next(gen)
    except StopIteration:
        pass

    assert seen["lam_method"] == "MAP"


def test_calculate_bism_stresses_generator_contract():
    tx, ty, mask = _biaxial_traction()
    force_field = np.stack([tx, ty], axis=-1)[np.newaxis, ...]   # (1, R, C, 2)
    force_field = np.concatenate([force_field, force_field], axis=0)  # 2 frames
    masks = np.stack([mask, mask])
    params = MSMParameters(pixel_size=0.2, downscale_factor=5, stress_method="BISM")

    gen = bism.calculate_bism_stresses(force_field, masks, params)
    progress = []
    try:
        while True:
            progress.append(next(gen))
    except StopIteration as exc:
        result = exc.value

    assert [(f, t) for _, f, t in progress] == [(1, 2), (2, 2)]
    assert isinstance(result, StressResult)
    assert result.method == "BISM"
    assert result.stress_tensor.shape == (2, 12, 12, 2, 2)
    assert result.stress_tensor.dtype == np.float32
    # Symmetric tensor: [0,1] == [1,0].
    assert np.allclose(result.stress_tensor[..., 0, 1], result.stress_tensor[..., 1, 0])
    # No FEM mesh; BISM carries the traction-reconstruction R² instead.
    assert result.nodes is None and result.elements is None
    assert result.r2_traction is not None
    # grid_spacing = pixel_size * downscale_factor, units preserved from MSM.
    assert result.physical_scale["grid_spacing"] == 1.0
    assert result.physical_scale["stress_units"] == "mN/m"


def test_calculate_bism_stresses_empty_mask_is_zero():
    R = C = 8
    force_field = np.zeros((1, R, C, 2), dtype=np.float32)
    masks = np.zeros((1, R, C), dtype=bool)
    params = MSMParameters(stress_method="BISM")

    gen = bism.calculate_bism_stresses(force_field, masks, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        result = exc.value

    assert np.all(result.stress_tensor == 0)
