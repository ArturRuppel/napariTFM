"""Registration tests: CPU reference recovers a known shift, and the GPU port's
drift estimate is bit-identical to the scikit-image reference.

CPU tests run on a plain install; GPU-parity tests are CUDA-gated (like the
displacement-backend parity tests).
"""
import numpy as np
import pytest
from scipy import ndimage

from napariTFM.backend.registration import (
    apply_drift,
    drift_fingerprint,
    estimate_drift,
    load_drift_csv,
    save_drift_csv,
    valid_region,
)

_has_cuda = False
try:
    import torch as _torch
    _has_cuda = _torch.cuda.is_available()
except Exception:
    pass
requires_cuda = pytest.mark.skipif(not _has_cuda, reason="no CUDA device / torch")


def _speckle(H=256, W=256, seed=0):
    """A bead-like speckle field: sparse bright dots on noise, smoothed."""
    rng = np.random.default_rng(seed)
    img = rng.normal(20, 3, (H, W)).astype(np.float32)
    ys = rng.integers(10, H - 10, 400)
    xs = rng.integers(10, W - 10, 400)
    img[ys, xs] += rng.uniform(80, 200, 400)
    return ndimage.gaussian_filter(img, 1.2)


@pytest.mark.parametrize("u_x,u_y", [(3.0, -2.0), (0.5, 1.5), (-4.25, 0.0)])
def test_estimate_drift_recovers_known_shift(u_x, u_y):
    """estimate_drift recovers a synthetic bulk translation (subpixel)."""
    anchor = _speckle()
    # Shift content by (+u_y,+u_x): drift should read (u_x, u_y).
    moved = ndimage.shift(anchor, shift=(u_y, u_x), order=3, mode="nearest")
    drift = estimate_drift(anchor, moved)
    assert np.allclose(drift, [u_x, u_y], atol=0.1), f"got {drift}, want ({u_x},{u_y})"


def test_apply_drift_deadzone_is_identity():
    """A near-zero drift returns the input untouched (no needless blur)."""
    img = _speckle()
    out = apply_drift(img, np.array([1e-3, -1e-3]))
    assert out is img or np.array_equal(out, img)


def test_apply_drift_roundtrips_estimate():
    """apply_drift(drift) undoes the shift estimate_drift measured."""
    anchor = _speckle()
    moved = ndimage.shift(anchor, shift=(1.5, -3.0), order=3, mode="nearest")
    drift = estimate_drift(anchor, moved)
    back = apply_drift(moved, drift)
    m = 20  # ignore resampled border
    assert np.corrcoef(anchor[m:-m, m:-m].ravel(), back[m:-m, m:-m].ravel())[0, 1] > 0.99


def test_valid_region_crops_only_drifted_sides():
    """No drift -> full frame; a positive u_x crops the right edge by ~ceil+margin."""
    r0, r1, c0, c1 = valid_region(np.zeros(2), np.zeros((3, 2)), (100, 100))
    assert (r0, r1, c0, c1) == (0, 100, 0, 100)
    r0, r1, c0, c1 = valid_region(np.zeros(2), np.array([[3.2, 0.0]]), (100, 100))
    assert c1 < 100 and c0 == 0 and r0 == 0 and r1 == 100


def test_drift_cache_roundtrips(tmp_path):
    """save_drift_csv -> load_drift_csv recovers the reference + per-frame drift."""
    path = tmp_path / "registration_drift.csv"
    ref_drift = np.array([1.25, -3.5], dtype=np.float32)
    drift_pixels = np.array([[0.0, 0.0], [2.1, -1.4], [-0.75, 3.2]], dtype=np.float32)
    fp = "deadbeef"
    save_drift_csv(path, ref_drift, drift_pixels, fp)

    loaded = load_drift_csv(path, expected_frames=3, fingerprint=fp)
    assert loaded is not None
    got_ref, got_frames = loaded
    assert np.allclose(got_ref, ref_drift, atol=1e-6)
    assert np.allclose(got_frames, drift_pixels, atol=1e-6)


def test_drift_cache_invalidates_on_fingerprint_or_count(tmp_path):
    """A changed fingerprint or a frame-count mismatch fails safe to None."""
    path = tmp_path / "registration_drift.csv"
    save_drift_csv(path, np.zeros(2, np.float32),
                   np.zeros((3, 2), np.float32), "fp-original")
    # Right fingerprint, wrong frame count -> None (partial/foreign cache).
    assert load_drift_csv(path, expected_frames=5, fingerprint="fp-original") is None
    # Right count, stale fingerprint (inputs changed) -> None.
    assert load_drift_csv(path, expected_frames=3, fingerprint="fp-other") is None
    # Missing file -> None.
    assert load_drift_csv(tmp_path / "nope.csv", 3, "fp-original") is None


def test_drift_fingerprint_tracks_input_changes():
    """The fingerprint is stable for identical inputs and moves when any frame changes."""
    ref = _speckle(seed=10)
    target = np.stack([_speckle(seed=11), _speckle(seed=12), _speckle(seed=13)])
    fp = drift_fingerprint(ref, target)
    assert drift_fingerprint(ref, target) == fp          # deterministic
    # The fingerprint is a strided subsample (a swapped/edited dataset heuristic,
    # not a single-pixel checksum): perturb a region, as a real edit would.
    changed = target.copy()
    changed[1, 16:48, 16:48] += 50.0
    assert drift_fingerprint(ref, changed) != fp
    assert drift_fingerprint(ref + 1.0, target) != fp    # perturb the reference


class _StubAnalyzer:
    """Minimal analyzer so the cache path is exercised without a real method."""

    def calculate_flow(self, reference, moving):
        return np.zeros((*reference.shape, 2), dtype=np.float32)

    def downscale_flow(self, field, factor):
        return field


def test_calculate_displacement_field_writes_then_reuses_cache(tmp_path):
    """First run writes the sidecar; a hit is read verbatim (bypasses estimation)."""
    from napariTFM.backend.displacement_analysis import calculate_displacement_field
    from napariTFM.backend.parameter_dataclasses import DisplacementParameters

    anchor = _speckle(H=96, W=96, seed=20)
    target = np.stack([
        anchor,
        ndimage.shift(anchor, shift=(1.0, -1.5), order=3, mode="nearest"),
    ]).astype(np.float32)
    reference = _speckle(H=96, W=96, seed=21)
    params = DisplacementParameters(disp_device="cpu", downscale_factor=1)
    path = tmp_path / "registration_drift.csv"

    def _run():
        gen = calculate_displacement_field(reference, target, params,
                                           analyzer=_StubAnalyzer(), drift_cache=path)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    first = _run()
    assert path.exists(), "first run should write the sidecar"
    real_drift = first.drift_pixels.copy()

    # Overwrite the cache with sentinel drifts under the SAME (still-valid)
    # fingerprint: a second run that reuses the cache must surface the sentinels,
    # proving it read the file instead of re-estimating.
    fp = drift_fingerprint(reference, target)
    sentinel = np.array([[0.5, -0.5], [1.5, -2.0]], dtype=np.float32)
    save_drift_csv(path, np.array([0.25, 0.75], np.float32), sentinel, fp)

    second = _run()
    assert np.allclose(second.drift_pixels, sentinel, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("u_x,u_y", [(3.0, -2.0), (0.5, 1.5), (-4.25, 6.75)])
def test_gpu_drift_is_bit_identical_to_skimage(u_x, u_y):
    """The GPU phase-correlation drift matches the scikit-image reference exactly."""
    from napariTFM.backend._registration_torch import TorchDriftEstimator

    anchor = _speckle(seed=1)
    moved = ndimage.shift(anchor, shift=(u_y, u_x), order=3, mode="nearest")
    cpu = estimate_drift(anchor, moved)
    gpu = TorchDriftEstimator(anchor, device="cuda")(moved)
    assert np.array_equal(cpu, gpu), f"cpu={cpu} gpu={gpu}"


@requires_cuda
def test_gpu_apply_drift_matches_scipy_within_tolerance():
    """grid_sample resample tracks scipy cubic well below the displacement noise floor."""
    from napariTFM.backend._registration_torch import apply_drift_torch

    img = _speckle(seed=2)
    drift = np.array([2.3, -1.7], dtype=np.float32)
    cpu = apply_drift(img, drift, order=3)
    gpu = apply_drift_torch(img, drift, device="cuda", order=3)
    m = 30
    rel = np.abs(cpu - gpu)[m:-m, m:-m] / (img.max() - img.min())
    assert rel.mean() < 0.02 and np.percentile(rel, 99) < 0.1


@requires_cuda
def test_registration_ops_uses_gpu_on_cuda_else_cpu():
    """_registration_ops picks the torch estimator on cuda and the CPU refs on cpu."""
    from napariTFM.backend._registration_torch import TorchDriftEstimator
    from napariTFM.backend.displacement_analysis import _registration_ops
    from napariTFM.backend.parameter_dataclasses import DisplacementParameters

    anchor = _speckle(seed=3)
    est_gpu, _ = _registration_ops(anchor, DisplacementParameters(disp_device="cuda"))
    assert isinstance(est_gpu, TorchDriftEstimator)
    est_cpu, apply_cpu = _registration_ops(anchor, DisplacementParameters(disp_device="cpu"))
    assert est_cpu(anchor) is not None and apply_cpu is apply_drift
