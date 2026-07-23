"""Tests for removal of the old sparse-solver soft mask support.

Mask controls in the Force panel now clip completed force maps after the selected
solver has run. They no longer add a smooth off-mask penalty inside the L1 objective.
"""
import numpy as np

from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend import forward_l1 as L1
from napariTFM.backend import fttc


def _params(**kw):
    base = dict(l1_sparsity=0.05, l1_max_iter=80, young_modulus=5000.0,
                poisson_ratio_substrate=0.5, gel_height=None, pixel_size=0.1,
                downscale_factor=1, fwd_fit_margin_um=1e6, fwd_mask_strength=0.0,
                fwd_device="cpu", fwd_dtype="float64")
    base.update(kw)
    return FTTCParameters(**base)


def test_exterior_penalty_is_removed():
    h = w = 24
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= 5 ** 2).astype(np.uint8)
    valid = np.ones((h, w), dtype=bool)
    params = _params(fwd_mask_strength=100.0)

    penalty = L1._exterior_penalty(mask, valid, 3.0, params, np, np.float64)

    np.testing.assert_array_equal(penalty, 0.0)


def test_l1_mask_clipping_is_posthoc():
    h = w = 28
    rng = np.random.default_rng(8)
    disp = rng.standard_normal((1, h, w, 2)).astype(np.float64) * 0.02
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:18, 10:18] = 1

    params = _params(force_method="Elastic net", fwd_mask_strength=1.0,
                     fwd_mask_reach=0.0)
    gen = fttc.calculate_force_field(disp, params, mask=mask)
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        force = exc.value.force_field[0]

    np.testing.assert_array_equal(force[mask == 0], 0.0)
    assert np.count_nonzero(np.linalg.norm(force[mask > 0], axis=-1) > 0) > 0
