"""Tests for the packaged BISM stress engine and its frame generator."""

import numpy as np

from napariTFM.backend import bism
from napariTFM.backend.parameter_dataclasses import StressParameters
from napariTFM.backend.stress import StressResult, process_mask_data


def _biaxial_traction(R=12, C=12, edge=1000.0):
    """A square in-mask plate pulled inward on all four edges (biaxial).

    Tractions live on the plate edges in both x and y, so both traction
    components carry structure (a finite reconstruction R²); the masked region
    is the full grid.
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


def test_process_mask_data_resizes_to_force_field():
    mask_data = np.array([[0, 2], [3, 0]], dtype=np.uint8)
    force_field = np.zeros((1, 4, 4, 2), dtype=np.float32)

    processed, warnings = process_mask_data(mask_data, force_field)

    assert processed.shape == (1, 4, 4)
    assert processed.dtype == bool
    assert warnings == ["Multiple non-zero values detected in mask. Converting to binary (0 and 1)."]


def test_calculate_bism_stresses_generator_contract():
    tx, ty, mask = _biaxial_traction()
    force_field = np.stack([tx, ty], axis=-1)[np.newaxis, ...]   # (1, R, C, 2)
    force_field = np.concatenate([force_field, force_field], axis=0)  # 2 frames
    masks = np.stack([mask, mask])
    params = StressParameters(pixel_size=0.2, downscale_factor=5)

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
    assert result.r2_traction is not None
    assert result.physical_scale["grid_spacing"] == 1.0
    assert result.physical_scale["stress_units"] == "mN/m"


def test_calculate_bism_stresses_empty_mask_is_zero():
    R = C = 8
    force_field = np.zeros((1, R, C, 2), dtype=np.float32)
    masks = np.zeros((1, R, C), dtype=bool)
    params = StressParameters()

    gen = bism.calculate_bism_stresses(force_field, masks, params)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        result = exc.value

    assert np.all(result.stress_tensor == 0)
