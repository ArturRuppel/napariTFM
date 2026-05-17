import numpy as np
import pytest

from _dev.optical_flow_comparison.detection import detect_beads


def _synthetic_beads(shape=(128, 128), positions=((20, 30), (70, 80), (100, 50))):
    """Float32 [0, 1] image with Gaussian bead-like peaks at the given (x, y).

    Beads have heterogeneous amplitudes so the auto-minmass percentile cut
    behaves the way it does on real data (some "weak" peaks below the cut,
    some "strong" peaks above). All `positions` here are strong peaks.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w), dtype=np.float32)
    sigma = 1.5
    # Real peaks at the given positions: amplitude 1.0 (strong).
    for (x, y) in positions:
        img += 1.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    # Extra weak peaks scattered around to populate the lower mass tail.
    weak_positions = [(15, 100), (110, 110), (50, 20), (90, 25), (40, 100)]
    for (x, y) in weak_positions:
        img += 0.25 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    img = img / img.max()
    return img


def test_detect_beads_finds_all_synthetic_peaks():
    positions = [(20, 30), (70, 80), (100, 50)]
    img = _synthetic_beads(positions=positions)

    points = detect_beads(img)

    assert points.shape[1] == 2
    assert len(points) >= len(positions)
    # Every synthetic peak should have a detected point within 1 px.
    for (x, y) in positions:
        d = np.hypot(points[:, 0] - x, points[:, 1] - y)
        assert d.min() < 1.0, f"no detection near ({x}, {y}); nearest = {d.min()} px"


def test_detect_beads_returns_float32_xy_columns():
    img = _synthetic_beads()
    points = detect_beads(img)
    assert points.dtype == np.float32
    assert points.shape[1] == 2
