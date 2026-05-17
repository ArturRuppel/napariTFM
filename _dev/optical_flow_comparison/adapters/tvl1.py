import cv2
import numpy as np

from _dev.optical_flow_comparison.adapters.base import sample_dense_at_points


def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32, copy=False)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo <= 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - lo) / (hi - lo) * 255.0
    return np.ascontiguousarray(scaled.astype(np.uint8))


class TVL1Adapter:
    """OpenCV Dual TV-L1 optical flow (opencv-contrib).

    What v1.0 of napariTFM used. Edge-preserving total-variation
    regularization + L1 data term — designed to preserve discontinuities
    where Farneback's quadratic smoothing would blur them.

    Parameters match _validation/benchmark_TFM/validate_TFM.py for the "high"
    scenario (lambda_=0.5 chosen because that's where the algorithm
    differences actually matter — low/mid are easy regardless).
    """

    name = "TV-L1"

    # Per the v1 validation script (see validate_TFM.py lines 88-122).
    TAU = 0.1
    LAMBDA = 0.5
    THETA = 0.15
    NSCALES = 50
    WARPS = 10
    EPSILON = 0.01
    INNER_ITERATIONS = 25
    OUTER_ITERATIONS = 15
    SCALE_STEP = 0.95
    GAMMA = 0.0
    MEDIAN_FILTERING = 5
    USE_INITIAL_FLOW = False

    def __init__(self) -> None:
        self._flow = cv2.optflow.DualTVL1OpticalFlow_create(
            self.TAU, self.LAMBDA, self.THETA,
            self.NSCALES, self.WARPS, self.EPSILON,
            self.INNER_ITERATIONS, self.OUTER_ITERATIONS,
            self.SCALE_STEP, self.GAMMA,
            self.MEDIAN_FILTERING, self.USE_INITIAL_FLOW,
        )

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_u8 = _to_uint8(reference)
        def_u8 = _to_uint8(deformed)
        flow = self._flow.calc(ref_u8, def_u8, None).astype(np.float32, copy=False)
        displacements = sample_dense_at_points(flow, query_points)
        valid = np.ones(len(query_points), dtype=bool)
        return displacements, valid
