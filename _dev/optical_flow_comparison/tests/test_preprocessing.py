import numpy as np

from _dev.optical_flow_comparison.preprocessing import preprocess


def test_preprocess_returns_float32_in_unit_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4096, size=(64, 64), dtype=np.uint16).astype(np.float32)

    out = preprocess(img)

    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_preprocess_is_deterministic_for_same_input():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 4096, size=(32, 32), dtype=np.uint16).astype(np.float32)

    a = preprocess(img)
    b = preprocess(img)

    np.testing.assert_array_equal(a, b)
