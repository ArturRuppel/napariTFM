from pathlib import Path

import numpy as np

from napariTFM.backend import fttc
from napariTFM.backend.parameter_dataclasses import FTTCParameters


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_validates_fttc_displacement_fields():
    assert fttc.validate_displacement_field(None) == (False, "No displacement field data provided")
    assert fttc.validate_displacement_field([[1, 2], [3, 4]]) == (
        False,
        "Displacement field must be a numpy array",
    )
    assert fttc.validate_displacement_field(np.zeros((2, 2))) == (
        False,
        "Displacement field must be 3D (y,x,2) or 4D (t,y,x,2)",
    )
    assert fttc.validate_displacement_field(np.zeros((2, 2, 3))) == (
        False,
        "Last dimension must be 2 (x,y components), got 3",
    )
    assert fttc.validate_displacement_field(np.full((2, 2, 2), np.nan)) == (
        False,
        "Displacement field contains only NaN values",
    )
    assert fttc.validate_displacement_field(np.zeros((2, 2, 2), dtype=np.float32)) == (True, "")


def test_backend_calculates_fttc_result_with_progress(monkeypatch):
    # l1_sparsity=0 routes to plain FTTC (the FakeFTTC below); the shipped default (0.05)
    # would take the group-L1 path instead and never reach this branch.
    params = FTTCParameters(pixel_size=0.2, downscale_factor=3, frame_interval=2,
                            regularization=1e-5, l1_sparsity=0.0)
    displacement_field = np.zeros((2, 4, 5, 2), dtype=np.float32)
    displacement_field[0, ..., 0] = 1
    displacement_field[1, ..., 1] = 2

    class FakeFTTC:
        def __init__(self, received_params):
            assert received_params == params

        def calculate_traction(self, displacements, pixel_size, downscale_factor, regularization):
            assert displacements.shape == (4, 5, 2)
            assert pixel_size == params.pixel_size
            assert downscale_factor == params.downscale_factor
            assert regularization == params.regularization
            frame_value = float(displacements[..., 0].max() + displacements[..., 1].max())
            return None, np.stack(
                [
                    np.full((4, 5), frame_value, dtype=np.float32),
                    np.full((4, 5), frame_value + 10, dtype=np.float32),
                ]
            )

    monkeypatch.setattr(fttc, "FTTC", FakeFTTC)

    generator = fttc.calculate_force_field(displacement_field, params)
    progress = []
    try:
        while True:
            progress.append(next(generator))
    except StopIteration as exc:
        result = exc.value

    assert [(frame, total) for _, frame, total in progress] == [(1, 2), (2, 2)]
    assert result.force_field.shape == (2, 4, 5, 2)
    assert result.force_field.dtype == np.float32
    assert result.force_field[0, ..., 0].max() == 1
    assert result.force_field[1, ..., 1].max() == 12
    assert result.original_shape == (4, 5)
    assert result.force_shape == (4, 5)
    assert result.parameters == params
    assert result.physical_scale == {
        "pixel_size": 0.2,
        "grid_spacing": 0.6000000000000001,
        "time_interval": 2,
        "force_units": "Pa",
        "grid_spacing_units": "µm",
        "time_interval_units": "min",
    }


def test_gcv_picks_positive_lambda_and_auto_path_matches_it():
    """FTTC+GCV: the one-shot picker returns a positive Fourier λ, and the per-frame
    auto_gcv dispatch is identical to feeding that same λ manually (same operator)."""
    h = w = 40
    yy, xx = np.mgrid[0:h, 0:w]
    bump = np.exp(-(((xx - 20) ** 2 + (yy - 16) ** 2) / 36.0))
    disp = np.stack([0.2 * bump, -0.15 * bump], axis=-1).astype(np.float64)

    p = FTTCParameters(young_modulus=5000, poisson_ratio_substrate=0.5,
                       pixel_size=0.1, downscale_factor=4, l1_sparsity=0.0)

    lam = fttc.find_gcv_regularization(disp, p)
    assert np.isfinite(lam) and lam > 0

    def run(params):
        gen = fttc.calculate_force_field(disp[np.newaxis], params)
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            return exc.value.force_field

    from dataclasses import replace
    f_auto = run(replace(p, auto_gcv=True))
    f_manual = run(replace(p, auto_gcv=False, regularization=lam))
    assert np.isfinite(f_auto).all()
    assert np.allclose(f_auto, f_manual)


def test_force_method_auto_matches_legacy_inference_and_explicit_overrides():
    """force_method="auto" follows the current solver routing; an explicit method overrides
    the numeric flags (so l1_sparsity=0.05 no longer forces the L1 path when FTTC is chosen)."""
    from dataclasses import replace
    from napariTFM.backend.fttc import infer_force_method

    h = w = 40
    yy, xx = np.mgrid[0:h, 0:w]
    bump = np.exp(-(((xx - 20) ** 2 + (yy - 16) ** 2) / 36.0))
    disp = np.stack([0.2 * bump, -0.15 * bump], axis=-1).astype(np.float64)
    base = FTTCParameters(young_modulus=5000, poisson_ratio_substrate=0.5,
                          pixel_size=0.1, downscale_factor=4)

    assert infer_force_method(replace(base, l1_sparsity=0.05)) == "Elastic net"
    assert infer_force_method(replace(base, l1_sparsity=0.0, bayesian_l2=True)) == "Bayesian L2"
    assert infer_force_method(replace(base, l1_sparsity=0.0, fwd_mask_strength=10.0),
                              mask_present=True) == "FTTC + GCV"
    assert infer_force_method(replace(base, l1_sparsity=0.0)) == "FTTC + GCV"

    def run(p):
        gen = fttc.calculate_force_field(disp[np.newaxis], p)
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            return exc.value.force_field

    f_auto_l1 = run(replace(base, force_method="auto", l1_sparsity=0.05))
    f_l1 = run(replace(base, force_method="Elastic net", l1_sparsity=0.05))
    assert np.allclose(f_auto_l1, f_l1)  # auto reproduces legacy L1 routing

    f_fttc = run(replace(base, force_method="FTTC", l1_sparsity=0.05))
    f_plain = run(replace(base, force_method="auto", l1_sparsity=0.0))
    assert not np.allclose(f_fttc, f_l1)   # explicit FTTC ignores the nonzero l1_sparsity
    assert np.allclose(f_fttc, f_plain)    # ...and matches a plain-FTTC solve


def test_force_mask_clip_is_posthoc_hard_zero_with_radius():
    from dataclasses import replace

    h = w = 32
    rng = np.random.default_rng(4)
    disp = rng.standard_normal((1, h, w, 2)).astype(np.float64) * 0.02
    base = FTTCParameters(young_modulus=5000, poisson_ratio_substrate=0.5,
                          pixel_size=0.1, downscale_factor=1, force_method="FTTC + GCV")
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[12:20, 12:20] = 1

    def run(p, m=None):
        gen = fttc.calculate_force_field(disp, p, mask=m)
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            return exc.value.force_field[0]

    unclipped = run(replace(base, fwd_mask_strength=0.0), mask)
    clipped = run(replace(base, fwd_mask_strength=1.0, fwd_mask_reach=2.0), mask)

    from scipy import ndimage
    keep = ndimage.distance_transform_edt(~(mask > 0)) <= 2.0
    assert np.count_nonzero(np.linalg.norm(unclipped[~keep], axis=-1) > 0) > 0
    np.testing.assert_array_equal(clipped[~keep], 0.0)
    assert np.count_nonzero(np.linalg.norm(clipped[keep], axis=-1) > 0) > 0


def test_production_code_does_not_depend_on_fttc_service_layer():
    removed_module = ".".join(("services", "fttc_service"))
    removed_path = Path("napariTFM") / "services" / "fttc_service.py"

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
