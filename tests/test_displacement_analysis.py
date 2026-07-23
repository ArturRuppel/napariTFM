"""Tests for the displacement backends (PIV, Lucas-Kanade, FFD).

PIV is a single torch implementation run on CPU or CUDA (torch is a core
dependency). Lucas-Kanade still has a torch-free scikit-image CPU path.
GPU-parity tests are gated behind CUDA. FFD is GPU-only, so its run test is
CUDA-gated; its unavailable path (no GPU) is checked directly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from napariTFM.backend.displacement_analysis import (
    build_analyzer,
    calculate_displacement_field,
    validate_displacement_image,
)
from napariTFM.backend.ffd_displacement import FFDDisplacementAnalyzer
from napariTFM.backend.ilk_displacement import ILKDisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.piv_displacement import PIVDisplacementAnalyzer


REPO_ROOT = Path(__file__).resolve().parents[1]

_has_cuda = False
try:  # torch is optional; CUDA-gated tests skip cleanly without it
    import torch as _torch
    _has_cuda = _torch.cuda.is_available()
except Exception:
    pass
requires_cuda = pytest.mark.skipif(not _has_cuda, reason="no CUDA device / torch")


def _params(**overrides):
    """DisplacementParameters pinned to a fast, deterministic CPU PIV config."""
    base = dict(disp_method="PIV", disp_device="cpu", piv_window=16, piv_passes=4)
    base.update(overrides)
    return DisplacementParameters(**base)


def _textured_image(seed=0, size=128):
    """A smooth, well-textured image so window correlation is well-posed."""
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((size, size))
    return ndimage.gaussian_filter(img, sigma=2.0).astype(np.float64)


# ------------------------------------------------------------- dispatch #
def test_build_analyzer_dispatches_by_method():
    assert isinstance(build_analyzer(_params(disp_method="PIV")), PIVDisplacementAnalyzer)
    assert isinstance(build_analyzer(_params(disp_method="Lucas-Kanade")), ILKDisplacementAnalyzer)


def test_build_analyzer_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown displacement method"):
        build_analyzer(_params(disp_method="nope"))


def test_analyzer_reports_backends():
    assert PIVDisplacementAnalyzer(_params()).algorithm_name == "PIV"
    assert PIVDisplacementAnalyzer(_params())._backend == "torch"    # PIV is always torch
    assert ILKDisplacementAnalyzer(_params())._backend == "skimage"  # cpu -> skimage


# ---------------------------------------------------- CPU translation #
@pytest.mark.parametrize("method,analyzer_cls", [
    ("PIV", PIVDisplacementAnalyzer),
    ("Lucas-Kanade", ILKDisplacementAnalyzer),
])
def test_cpu_recovers_known_translation(method, analyzer_cls):
    ref = _textured_image(seed=1, size=128)
    dx, dy = 1.5, -1.0                       # columns (x), rows (y)
    moving = ndimage.shift(ref, shift=(dy, dx), order=3, mode="reflect")

    flow = analyzer_cls(_params(disp_method=method)).calculate_flow(ref, moving)

    m = slice(32, -32)                       # interior, avoid border
    ux = np.median(flow[m, m, 0]); uy = np.median(flow[m, m, 1])
    assert abs(ux - dx) < 0.25, f"{method} u_x {ux:.3f} vs {dx}"
    assert abs(uy - dy) < 0.25, f"{method} u_y {uy:.3f} vs {dy}"


def test_flow_contract():
    ref = _textured_image(seed=2, size=96)
    moving = ndimage.shift(ref, shift=(0.0, 1.0), order=3, mode="reflect")

    flow = PIVDisplacementAnalyzer(_params()).calculate_flow(ref, moving)

    assert flow.shape == (96, 96, 2)          # full native resolution, (H,W,2)
    assert flow.dtype == np.float32
    assert np.isfinite(flow).all()
    # component 0 is u_x: a +1px column shift must show up positive there
    assert np.median(flow[20:-20, 20:-20, 0]) > 0.3


def test_cpu_recovers_localized_displacement_upright():
    """Flip-sensitive orientation guard. A UNIFORM translation is blind to a vertical
    flip of the field (flipud of a constant field is a no-op), so the uniform-shift
    tests above cannot catch a row-mirrored backend. Use a displacement LOCALIZED to a
    known off-centre spot (top quarter) and assert the recovered bump sits there, not at
    its flipud mirror in the bottom quarter. Guards the field orientation (row 0 = top,
    +uy = downward) against any future row-mirroring regression."""
    size = 128
    ref = _textured_image(seed=7, size=size)
    r0, c0, amp, sig = 32, 64, 3.0, 10.0          # bump in the TOP quarter, +down shift
    yy, xx = np.mgrid[0:size, 0:size]
    uy = amp * np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2 * sig ** 2))
    moving = ndimage.map_coordinates(ref, [yy - uy, xx], order=3, mode="reflect")

    flow = PIVDisplacementAnalyzer(_params(piv_window=24)).calculate_flow(ref, moving)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    pr, pc = np.unravel_index(mag.argmax(), mag.shape)
    assert pr < size // 2, f"recovered bump at row {pr} (mirrored to bottom half -> flipud bug)"
    assert abs(pr - r0) < 16 and abs(pc - c0) < 16, f"bump at ({pr},{pc}), expected ~({r0},{c0})"
    assert flow[r0, c0, 1] > 0.5, f"downward shift must recover +uy, got {flow[r0, c0, 1]:.2f}"


def test_cpu_backend_is_deterministic():
    ref = _textured_image(seed=3, size=96)
    moving = ndimage.shift(ref, shift=(0.7, -0.4), order=3, mode="reflect")

    p = _params()
    f1 = PIVDisplacementAnalyzer(p).calculate_flow(ref, moving)
    f2 = PIVDisplacementAnalyzer(p).calculate_flow(ref, moving)
    np.testing.assert_array_equal(f1, f2)


def test_ilk_cpu_needs_no_torch(monkeypatch):
    """With torch import forced to fail, iLK still runs on its scikit-image CPU path
    for device 'cpu' and 'auto' -- the torch-free contract that survives for iLK."""
    monkeypatch.setitem(sys.modules, "torch", None)

    ref = _textured_image(seed=5, size=96)
    moving = ndimage.shift(ref, shift=(0.0, 1.0), order=3, mode="reflect")

    for device in ("cpu", "auto"):
        analyzer = build_analyzer(_params(disp_method="Lucas-Kanade", disp_device=device))
        assert analyzer._backend == "skimage"
        flow = analyzer.calculate_flow(ref, moving)
        assert flow.shape == (96, 96, 2)
        assert np.isfinite(flow).all()


def test_piv_without_torch_raises_actionable_error(monkeypatch):
    """PIV is torch-only now (CPU path included); with torch absent, constructing the
    analyzer errors clearly, for every device."""
    monkeypatch.setitem(sys.modules, "torch", None)

    for device in ("cpu", "auto", "cuda"):
        with pytest.raises(ImportError, match=r"[Pp]y[Tt]orch"):
            PIVDisplacementAnalyzer(_params(disp_device=device))


# ------------------------------------------------------------------ FFD #
def test_ffd_is_gpu_only_without_cuda():
    """FFD refuses to construct on the CPU path with an actionable message."""
    with pytest.raises(RuntimeError, match="GPU-only"):
        FFDDisplacementAnalyzer(_params(disp_method="FFD", disp_device="cpu"))


@requires_cuda
def test_ffd_recovers_known_translation():
    ref = _textured_image(seed=6, size=128)
    dx, dy = 1.5, -1.0
    moving = ndimage.shift(ref, shift=(dy, dx), order=3, mode="reflect")

    flow = FFDDisplacementAnalyzer(
        _params(disp_method="FFD", disp_device="cuda")
    ).calculate_flow(ref, moving)

    m = slice(32, -32)
    assert abs(np.median(flow[m, m, 0]) - dx) < 0.3
    assert abs(np.median(flow[m, m, 1]) - dy) < 0.3


# --------------------------------------------------- FFD param exposure #
@requires_cuda
def test_ffd_threads_all_params_into_ffd_pyr(monkeypatch):
    """Every FFD knob (existing + newly exposed) reaches ffd_pyr, defensively coerced.

    Spies on ffd_pyr rather than running a fit: the point is the wiring, not the math.
    """
    import napariTFM.backend._ffd_torch as ffd_mod

    captured = {}

    def spy(ref, dfm, **kwargs):
        captured.update(kwargs)
        H, W = np.asarray(ref).shape
        u = np.zeros((2, H, W), dtype=np.float32)
        return (u, 0.0) if kwargs.get("return_loss") else u

    monkeypatch.setattr(ffd_mod, "ffd_pyr", spy)

    params = _params(
        disp_method="FFD", disp_device="cuda",
        ffd_level_spacing=10.0, ffd_metric="mse",
        ffd_num_iters=33, ffd_elastic=0.25,
        ffd_downscale=1.8, ffd_min_size=12, ffd_interp="bilinear",
        ffd_early_stop=0.0015,
    )
    ref = _textured_image(seed=30, size=64)
    FFDDisplacementAnalyzer(params).calculate_flow(ref, ref)

    assert captured["level_spacing"] == 10.0
    assert captured["num_iters"] == 33
    assert captured["metric"] == "mse"
    assert captured["elastic"] == 0.25
    assert captured["downscale"] == 1.8
    assert captured["min_size"] == 12
    assert captured["interp"] == "bilinear"
    assert captured["early_stop"] == 0.0015
    # Pyramid depth is derived from downscale + min_size, not threaded as a count;
    # the between-level tol and the warm-start init_field are gone entirely.
    assert "num_levels" not in captured
    assert "tol" not in captured
    assert "init_field" not in captured


# ------------------------------------------------------- GPU parity #
@requires_cuda
@pytest.mark.parametrize("method", ["PIV", "Lucas-Kanade"])
def test_gpu_matches_cpu_on_dense_data(method):
    """CPU and CUDA run the same code for both methods, so on dense, well-posed texture
    they agree to within device-level floating-point noise, not just 'parity'."""
    ref = _textured_image(seed=7, size=160)
    moving = ndimage.shift(ref, shift=(0.6, -0.9), order=3, mode="reflect")

    f_cpu = build_analyzer(_params(disp_method=method, disp_device="cpu")).calculate_flow(ref, moving)
    f_gpu = build_analyzer(_params(disp_method=method, disp_device="cuda")).calculate_flow(ref, moving)

    m = slice(32, -32)
    assert np.median(np.abs(f_cpu[m, m] - f_gpu[m, m])) < 0.02


# ----------------------------------------------------- validation/util #
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
    ref = _textured_image(seed=4, size=96)
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
    assert result.displacement_field.shape == (2, 48, 48, 2)   # downscaled by 2
    assert result.displacement_field.dtype == np.float32
    assert result.original_shape == (96, 96)
    assert result.displacement_field_shape == (48, 48)
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


def test_registration_folds_interframe_drift_into_drift_pixels():
    """Registration removes bulk stage drift before measurement: each frame is
    aligned to the anchor (first frame), so the reported field is ~0 for pure drift
    and drift_pixels captures the shift (pixels, ordered [u_x, u_y])."""
    base = _textured_image(seed=11, size=160)
    dx, dy = 6.0, -4.0                       # columns (x), rows (y): a bulk stage drift
    frame0 = base                            # anchor: relaxed, no deformation
    frame1 = ndimage.shift(base, shift=(dy, dx), order=3, mode="nearest")  # pure drift
    stack = np.stack([frame0, frame1])

    params = _params(pixel_size=1.0, downscale_factor=1)  # native grid; µm == px
    gen = calculate_displacement_field(base, stack, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        result = exc.value

    # Frame 1's bulk drift is captured (~ (dx, dy)); frame 0 is the anchor (~0).
    assert abs(result.drift_pixels[1, 0] - dx) < 0.3
    assert abs(result.drift_pixels[1, 1] - dy) < 0.3
    assert np.abs(result.drift_pixels[0]).max() < 0.3

    # ...and removed from every reported field, which is ~0 (pure drift, no strain).
    m = slice(48, -48)
    for f in (0, 1):
        field = result.displacement_field[f]
        assert abs(np.median(field[m, m, 0])) < 0.3
        assert abs(np.median(field[m, m, 1])) < 0.3


def test_registration_keeps_localized_deformation_over_drift():
    """Drift removal must not eat real signal. Frame 1 carries a localized bump plus
    a uniform drift: registration removes the bulk drift, the bump survives."""
    size = 160
    base = _textured_image(seed=12, size=size)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cy, cx = size / 2, size / 2
    bump = 3.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 12.0 ** 2))
    drift_x = 5.0
    frame0 = base
    frame1 = ndimage.map_coordinates(base, [yy, xx - (bump + drift_x)], order=3, mode="nearest")
    stack = np.stack([frame0, frame1])

    params = _params(pixel_size=1.0, downscale_factor=1)
    gen = calculate_displacement_field(base, stack, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        result = exc.value

    # Bulk drift is captured (~drift_x, ~0); the localized bump is not mistaken for it.
    assert abs(result.drift_pixels[1, 0] - drift_x) < 0.7
    assert abs(result.drift_pixels[1, 1]) < 0.7

    field = result.displacement_field[1]
    c = size // 2
    center = field[c - 4:c + 4, c - 4:c + 4, 0]   # blob (true u_x ~3 after drift removal)
    background = field[24:32, 24:32, 0]           # at-rest gel (true u_x ~0)
    assert np.median(center) > 1.5
    assert abs(np.median(background)) < 0.7


def _localized_deformation_stack(size=160, bump_amp=3.0, drift_x=5.0, seed=13):
    """A relaxed anchor + a frame with a central Gaussian bump plus a bulk drift."""
    base = _textured_image(seed=seed, size=size)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cy, cx = size / 2, size / 2
    bump = bump_amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 12.0 ** 2))
    frame1 = ndimage.map_coordinates(base, [yy, xx - (bump + drift_x)], order=3, mode="nearest")
    return base, np.stack([base, frame1])


def _run(base, stack, params):
    gen = calculate_displacement_field(base, stack, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        return exc.value


def test_downsample_before_matches_after_grid_and_units():
    """disp_downscale_before bins the images before measuring instead of binning the
    field after. Both must yield the SAME output grid and physical units, and both
    must recover the localized bump (the fast path measures real signal, not noise)."""
    size = 160
    base, stack = _localized_deformation_stack(size=size)

    after = _run(base, stack, _params(pixel_size=1.0, downscale_factor=2,
                                      disp_downscale_before=False))
    before = _run(base, stack, _params(pixel_size=1.0, downscale_factor=2,
                                       disp_downscale_before=True))

    # Same coarse output grid and units either way.
    assert after.displacement_field.shape == before.displacement_field.shape == (2, 80, 80, 2)
    assert after.physical_scale == before.physical_scale

    c = size // 2 // 2   # bump centre on the /2 grid
    for result in (after, before):
        field = result.displacement_field[1]
        center = field[c - 2:c + 2, c - 2:c + 2, 0]   # true u_x ~3 (px==µm here)
        background = field[12:16, 12:16, 0]           # at-rest gel, ~0
        assert np.median(center) > 1.5                # real signal survives binning
        assert abs(np.median(background)) < 0.7

    # The two paths agree closely (this is the accuracy the toggle trades for speed).
    epe = np.linalg.norm(after.displacement_field[1] - before.displacement_field[1], axis=-1)
    assert np.median(epe) < 0.5


def test_downsample_before_is_noop_without_factor():
    """With downscale_factor == 1 there is nothing to bin, so the flag changes nothing."""
    base, stack = _localized_deformation_stack(size=128)
    off = _run(base, stack, _params(pixel_size=1.0, downscale_factor=1,
                                    disp_downscale_before=False))
    on = _run(base, stack, _params(pixel_size=1.0, downscale_factor=1,
                                   disp_downscale_before=True))
    assert off.displacement_field.shape == on.displacement_field.shape
    np.testing.assert_allclose(off.displacement_field, on.displacement_field)


def test_registration_estimate_and_undo_roundtrip():
    """estimate_drift recovers a known (u_x, u_y) shift; apply_drift undoes it."""
    from napariTFM.backend.registration import apply_drift, estimate_drift

    base = _textured_image(seed=20, size=160)
    dx, dy = 4.0, -3.0
    moved = ndimage.shift(base, shift=(dy, dx), order=3, mode="nearest")

    drift = estimate_drift(base, moved)              # (u_x, u_y)
    assert abs(drift[0] - dx) < 0.3 and abs(drift[1] - dy) < 0.3

    recov = apply_drift(moved, drift)
    m = slice(40, -40)
    span = float(base.max() - base.min())
    assert np.median(np.abs(recov - base)[m, m]) < 0.05 * span


@pytest.mark.parametrize("method", ["PIV", "Lucas-Kanade"])
def test_registration_removes_large_drift_for_all_methods(method):
    """A drift large enough to strain a capture-limited method is removed by
    registration before the method runs, so every method returns a ~zero field. This
    is what folding drift into a single backend could not do for Lucas-Kanade/FFD."""
    base = _textured_image(seed=21, size=192)
    frame1 = ndimage.shift(base, shift=(-6.0, 8.0), order=3, mode="nearest")  # ~10 px drift
    stack = np.stack([base, frame1])

    params = _params(disp_method=method, disp_device="cpu", pixel_size=1.0, downscale_factor=1)
    gen = calculate_displacement_field(base, stack, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        result = exc.value

    assert abs(result.drift_pixels[1, 0] - 8.0) < 0.4
    assert abs(result.drift_pixels[1, 1] + 6.0) < 0.4
    m = slice(56, -56)
    assert np.abs(result.displacement_field[1][m, m]).max() < 0.6


def _downscale_flow_reference(flow, factor):
    """Independent block-mean reference: the original O(H*W) double loop."""
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
