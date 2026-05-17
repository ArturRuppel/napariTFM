from typing import Dict

import numpy as np

# Edges in pixels: [0, 1, 3, inf] → three bins: low, mid, high.
MAGNITUDE_BINS: tuple[float, ...] = (0.0, 1.0, 3.0, float("inf"))
BIN_LABELS: tuple[str, ...] = ("low", "mid", "high")


def compute_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    """Compute per-(scenario, algorithm) summary metrics.

    Errors are evaluated only over beads where `valid` is True.

    Args:
        predicted:    shape (N, 2), columns (dx, dy) in pixels.
        ground_truth: shape (N, 2), columns (dx, dy) in pixels.
        valid:        shape (N,) bool.

    Returns:
        dict with keys: n_beads, coverage, rmse_px, median_px, p95_px,
                        bias_low, bias_mid, bias_high.
        Bias is the mean signed error magnitude (|pred| - |gt|) in pixels
        within each GT-magnitude bin. Negative = undershoot, positive = overshoot.
        Bins with no beads return np.nan.
    """
    n = len(predicted)
    coverage = float(valid.sum()) / n if n > 0 else 0.0

    err = predicted - ground_truth
    err_mag = np.hypot(err[:, 0], err[:, 1])
    valid_mag = err_mag[valid]

    if len(valid_mag) == 0:
        rmse = median = p95 = float("nan")
    else:
        rmse = float(np.sqrt(np.mean(valid_mag ** 2)))
        median = float(np.median(valid_mag))
        p95 = float(np.percentile(valid_mag, 95))

    gt_mag = np.hypot(ground_truth[:, 0], ground_truth[:, 1])
    pred_mag = np.hypot(predicted[:, 0], predicted[:, 1])
    signed = pred_mag - gt_mag

    bin_idx = np.digitize(gt_mag, MAGNITUDE_BINS) - 1  # 0, 1, 2
    bias: dict[str, float] = {}
    for i, label in enumerate(BIN_LABELS):
        in_bin = (bin_idx == i) & valid
        bias[f"bias_{label}"] = float(np.mean(signed[in_bin])) if in_bin.any() else float("nan")

    return {
        "n_beads": n,
        "coverage": coverage,
        "rmse_px": rmse,
        "median_px": median,
        "p95_px": p95,
        **bias,
    }
