import numpy as np

from _dev.optical_flow_comparison.adapters.base import sample_dense_at_points


def test_sample_dense_at_points_returns_exact_values_at_integer_coords():
    # Flow field where dx = column index, dy = row index. Sampling at integer
    # points must return those indices exactly.
    h, w = 10, 12
    flow = np.zeros((h, w, 2), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    flow[..., 0] = xx
    flow[..., 1] = yy

    pts = np.array([[0.0, 0.0], [5.0, 3.0], [11.0, 9.0]], dtype=np.float32)
    out = sample_dense_at_points(flow, pts)

    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], [0.0, 5.0, 11.0], atol=1e-5)
    np.testing.assert_allclose(out[:, 1], [0.0, 3.0, 9.0], atol=1e-5)


def test_sample_dense_at_points_interpolates_between_pixels():
    h, w = 4, 4
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = np.arange(w)[None, :]
    flow[..., 1] = 0.0

    pts = np.array([[1.5, 2.0]], dtype=np.float32)  # x=1.5 → dx should be 1.5
    out = sample_dense_at_points(flow, pts)

    np.testing.assert_allclose(out[0, 0], 1.5, atol=1e-5)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-5)


def test_sample_dense_at_points_clamps_out_of_bounds_points():
    h, w = 4, 4
    flow = np.ones((h, w, 2), dtype=np.float32)
    pts = np.array([[-1.0, -1.0], [100.0, 100.0]], dtype=np.float32)
    out = sample_dense_at_points(flow, pts)

    # Clamped to the edge; edge value of the constant field is 1.0.
    np.testing.assert_allclose(out, np.ones((2, 2)), atol=1e-5)


def test_sample_dense_at_points_bilinear_in_both_axes():
    # dx varies with x, dy varies with y. Sampling at (1.5, 2.5) should give
    # dx=1.5, dy=2.5 — true 2D bilinear interpolation.
    h, w = 5, 5
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = np.arange(w)[None, :]
    flow[..., 1] = np.arange(h)[:, None]

    pts = np.array([[1.5, 2.5]], dtype=np.float32)
    out = sample_dense_at_points(flow, pts)

    np.testing.assert_allclose(out[0, 0], 1.5, atol=1e-5)
    np.testing.assert_allclose(out[0, 1], 2.5, atol=1e-5)


def test_sample_dense_at_points_handles_empty_points_array():
    flow = np.zeros((4, 4, 2), dtype=np.float32)
    out = sample_dense_at_points(flow, np.empty((0, 2), dtype=np.float32))
    assert out.shape == (0, 2)
    assert out.dtype == np.float32
