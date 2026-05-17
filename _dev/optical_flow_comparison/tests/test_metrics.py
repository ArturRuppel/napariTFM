import numpy as np

from _dev.optical_flow_comparison.metrics import compute_metrics, MAGNITUDE_BINS


def test_compute_metrics_zero_error_case():
    gt = np.array([[1.0, 0.5], [2.0, 0.0], [0.1, 0.1]], dtype=np.float32)
    pred = gt.copy()
    valid = np.array([True, True, True])

    m = compute_metrics(pred, gt, valid)

    assert m["n_beads"] == 3
    assert m["coverage"] == 1.0
    assert m["rmse_px"] == 0.0
    assert m["median_px"] == 0.0
    assert m["p95_px"] == 0.0


def test_compute_metrics_ignores_invalid_beads():
    gt = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    pred = np.array([[1.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    valid = np.array([True, False, False])

    m = compute_metrics(pred, gt, valid)

    assert m["n_beads"] == 3
    assert m["coverage"] == 1 / 3
    assert m["rmse_px"] == 0.0  # only the valid bead is scored, and it is exact


def test_compute_metrics_magnitude_binned_bias_keys_present():
    gt = np.array([[0.2, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    pred = np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]], dtype=np.float32)  # all undershoot
    valid = np.array([True, True, True])

    m = compute_metrics(pred, gt, valid)

    # MAGNITUDE_BINS = [0, 1, 3, inf] → keys bias_low/mid/high
    for key in ("bias_low", "bias_mid", "bias_high"):
        assert key in m
    # The "high" bin (>3 px GT) should have a clearly negative signed error along x.
    assert m["bias_high"] < 0
