"""Tests for the displacement backend (multi-pass FFT cross-correlation PIV).

The core tests are NOT gated behind torch: PIV has a torch-free numpy core, so
it must work on a plain install. A separate test checks the numpy path still
runs with torch forcibly absent, and the GPU-equivalence test is gated behind
torch + CUDA.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from napariTFM.backend.displacement_analysis import (
    calculate_displacement_field,
    validate_displacement_image,
)
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.piv_displacement import PIVDisplacementAnalyzer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _params(**overrides):
    """DisplacementParameters pinned to a fast, deterministic CPU (numpy) PIV config."""
    base = dict(piv_device="cpu", piv_window=16, piv_passes=4)
    base.update(overrides)
    return DisplacementParameters(**base)


def _textured_image(seed=0, size=96):
    """A smooth, well-textured image so window correlation is well-posed."""
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((size, size))
    return ndimage.gaussian_filter(img, sigma=2.0).astype(np.float64)


def test_analyzer_reports_piv_algorithm():
    analyzer = PIVDisplacementAnalyzer(_params())
    assert analyzer.algorithm_name == "PIV"
    assert analyzer._backend == "numpy"  # piv_device="cpu" forces the numpy core


def test_piv_recovers_known_translation():
    ref = _textured_image(seed=1, size=96)
    dx, dy = 1.5, -1.0                       # columns (x), rows (y)
    moving = ndimage.shift(ref, shift=(dy, dx), order=3, mode="reflect")

    flow = PIVDisplacementAnalyzer(_params()).calculate_flow(ref, moving)

    m = slice(24, -24)                       # interior, avoid border
    ux = np.median(flow[m, m, 0])
    uy = np.median(flow[m, m, 1])
    assert abs(ux - dx) < 0.2, f"u_x {ux:.3f} vs {dx}"
    assert abs(uy - dy) < 0.2, f"u_y {uy:.3f} vs {dy}"


def test_piv_flow_contract():
    ref = _textured_image(seed=2, size=64)
    moving = ndimage.shift(ref, shift=(0.0, 1.0), order=3, mode="reflect")

    flow = PIVDisplacementAnalyzer(_params()).calculate_flow(ref, moving)

    assert flow.shape == (64, 64, 2)          # full native resolution, (H,W,2)
    assert flow.dtype == np.float32
    assert np.isfinite(flow).all()
    # component 0 is u_x: a +1px column shift must show up positive there
    assert np.median(flow[20:-20, 20:-20, 0]) > 0.3


def test_piv_numpy_core_is_deterministic():
    ref = _textured_image(seed=3, size=64)
    moving = ndimage.shift(ref, shift=(0.7, -0.4), order=3, mode="reflect")

    p = _params()
    f1 = PIVDisplacementAnalyzer(p).calculate_flow(ref, moving)
    f2 = PIVDisplacementAnalyzer(p).calculate_flow(ref, moving)
    np.testing.assert_array_equal(f1, f2)


def test_piv_numpy_core_needs_no_torch(monkeypatch):
    """With torch import forced to fail, PIV still works on the numpy core (device
    'cpu' and 'auto' both). This is the torch-free contract of the backend."""
    monkeypatch.setitem(sys.modules, "torch", None)

    ref = _textured_image(seed=5, size=48)
    moving = ndimage.shift(ref, shift=(0.0, 1.0), order=3, mode="reflect")

    for device in ("cpu", "auto"):
        analyzer = PIVDisplacementAnalyzer(_params(piv_device=device))
        assert analyzer._backend == "numpy"
        flow = analyzer.calculate_flow(ref, moving)
        assert flow.shape == (48, 48, 2)
        assert np.isfinite(flow).all()


def test_piv_cuda_without_torch_raises_actionable_error(monkeypatch):
    """piv_device='cuda' with torch absent errors clearly; 'auto'/'cpu' do not."""
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match=r"napariTFM\[piv\]"):
        PIVDisplacementAnalyzer(_params(piv_device="cuda"))


def test_piv_gpu_matches_numpy_on_dense_data():
    """Where PIV is well-posed (dense texture, every window populated), the GPU
    backend is numerically equivalent to the numpy core. (Skipped without CUDA.)"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    ref = _textured_image(seed=7, size=128)
    moving = ndimage.shift(ref, shift=(0.6, -0.9), order=3, mode="reflect")

    f_np = PIVDisplacementAnalyzer(_params(piv_device="cpu")).calculate_flow(ref, moving)
    f_gpu = PIVDisplacementAnalyzer(_params(piv_device="cuda")).calculate_flow(ref, moving)

    m = slice(24, -24)
    assert np.median(np.abs(f_np[m, m] - f_gpu[m, m])) < 0.05


def test_backend_validates_displacement_images():
    assert validate_displacement_image(None) == (False, "No image data provided")
    assert validate_displacement_image([[1, 2], [3, 4]]) == (False, "Image must be a numpy array")
    assert validate_displacement_image(np.zeros((2, 2, 2, 2))) == (
        False,
        "Image must be 2D or 3D (time series)",
    )
    assert validate_displacement_image(np.full((2, 2), np.nan)) == (
        False,
        "Image contains only NaN values",
    )
    assert validate_displacement_image(np.zeros((2, 2), dtype=np.float32)) == (True, "")


def test_backend_calculates_displacement_result_with_progress():
    ref = _textured_image(seed=4, size=48)
    moving = np.stack([
        ndimage.shift(ref, shift=(0.0, 1.0), order=3, mode="reflect"),
        ndimage.shift(ref, shift=(0.5, -0.5), order=3, mode="reflect"),
    ]).astype(np.float64)
    params = _params(pixel_size=0.2, downscale_factor=2, piv_passes=3)

    generator = calculate_displacement_field(ref, moving, params)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        result = exc.value

    assert [(frame, total) for _, frame, total in progress] == [(1, 2), (2, 2)]
    assert result.displacement_field.shape == (2, 24, 24, 2)   # downscaled by 2
    assert result.displacement_field.dtype == np.float32
    assert result.original_shape == (48, 48)
    assert result.displacement_field_shape == (24, 24)
    assert result.parameters == params
    assert result.physical_scale == {
        "pixel_size": 0.2,
        "grid_spacing": 0.4,
        "time_interval": 1,
        "displacement_units": "µm",
        "grid_spacing_units": "µm",
        "time_interval_units": "min",
    }
    assert np.isfinite(result.displacement_field).all()


def test_calculate_field_folds_global_translation_into_drift():
    """A spatially-uniform shift is stage drift, not deformation. PIV recovers it,
    then calculate_displacement_field subtracts it: the reported field is ~0 and
    drift_pixels captures the shift (pixels, ordered [u_x, u_y])."""
    ref = _textured_image(seed=11, size=96)
    dx, dy = 1.5, -1.0                       # columns (x), rows (y)
    moving = ndimage.shift(ref, shift=(dy, dx), order=3, mode="reflect")

    params = _params(pixel_size=1.0, downscale_factor=1)  # native grid; µm == px
    gen = calculate_displacement_field(ref, moving, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        result = exc.value

    # The bulk translation is captured as drift, ~ (dx, dy).
    assert abs(result.drift_pixels[0, 0] - dx) < 0.2
    assert abs(result.drift_pixels[0, 1] - dy) < 0.2

    # ...and removed from the reported field, which is ~0 (pure drift, no strain).
    m = slice(24, -24)
    field = result.displacement_field[0]
    assert abs(np.median(field[m, m, 0])) < 0.2
    assert abs(np.median(field[m, m, 1])) < 0.2


def test_calculate_field_keeps_localized_deformation_over_drift():
    """Drift removal must not eat real signal. A localized (zero-median) bump of
    deformation superimposed on a uniform drift: the drift is subtracted, the bump
    survives. This is the TFM regime — at-rest background dominates the median."""
    size = 96
    ref = _textured_image(seed=12, size=size)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cy, cx = size / 2, size / 2
    bump = 3.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0 ** 2))
    drift_x = 1.0
    # Sample ref shifted right by (bump + drift) in x → moving frame.
    moving = ndimage.map_coordinates(
        ref, [yy, xx - (bump + drift_x)], order=3, mode="reflect"
    )

    params = _params(pixel_size=1.0, downscale_factor=1)
    gen = calculate_displacement_field(ref, moving, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        result = exc.value

    # Background median is pure drift → captured, no spurious y drift.
    assert abs(result.drift_pixels[0, 0] - drift_x) < 0.4
    assert abs(result.drift_pixels[0, 1]) < 0.4

    field = result.displacement_field[0]
    center = field[44:52, 44:52, 0]          # blob (true u_x ~3 after drift removal)
    background = field[20:28, 20:28, 0]       # at-rest gel (true u_x ~0)
    assert np.median(center) > 1.5
    assert abs(np.median(background)) < 0.6


def _downscale_flow_reference(flow, factor):
    """Independent block-mean reference: the original O(H*W) double loop.

    Kept as the oracle the vectorized ``downscale_flow`` must match exactly, so
    the optimization can never silently change values.
    """
    if factor <= 1:
        return flow
    h, w = flow.shape[:2]
    new_h, new_w = h // factor, w // factor
    out = np.zeros((new_h, new_w, 2))
    for i in range(new_h):
        for j in range(new_w):
            block = flow[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
            out[i, j] = np.mean(block, axis=(0, 1))
    return out


def test_downscale_flow_matches_block_mean_reference():
    analyzer = PIVDisplacementAnalyzer(_params())
    rng = np.random.default_rng(0)
    for factor in (2, 3, 4, 5):
        flow = rng.standard_normal((37, 41, 2)).astype(np.float32)  # non-divisible dims
        got = analyzer.downscale_flow(flow, factor)
        expected = _downscale_flow_reference(flow, factor)
        assert got.shape == expected.shape == (37 // factor, 41 // factor, 2)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_downscale_flow_factor_one_returns_input_unchanged():
    analyzer = PIVDisplacementAnalyzer(_params())
    flow = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    assert analyzer.downscale_flow(flow, 1) is flow


def test_downscale_flow_exact_block_average():
    analyzer = PIVDisplacementAnalyzer(_params())
    # A 2x2 grid of constant 2x2 blocks: each output cell is that block's value.
    flow = np.zeros((4, 4, 2), dtype=np.float32)
    flow[0:2, 0:2] = [1.0, -1.0]
    flow[0:2, 2:4] = [2.0, 0.0]
    flow[2:4, 0:2] = [0.0, 3.0]
    flow[2:4, 2:4] = [-4.0, 5.0]
    out = analyzer.downscale_flow(flow, 2)
    np.testing.assert_allclose(out[0, 0], [1.0, -1.0])
    np.testing.assert_allclose(out[0, 1], [2.0, 0.0])
    np.testing.assert_allclose(out[1, 0], [0.0, 3.0])
    np.testing.assert_allclose(out[1, 1], [-4.0, 5.0])


def test_production_code_does_not_depend_on_displacement_service_layer():
    removed_module = ".".join(("services", "displacement_service"))
    removed_path = Path("napariTFM") / "services" / "displacement_service.py"

    assert not (REPO_ROOT / removed_path).exists()

    production_files = [
        path
        for root in ("napariTFM",)
        for path in (REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in production_files
        if removed_module in path.read_text()
    ]

    assert offenders == []
