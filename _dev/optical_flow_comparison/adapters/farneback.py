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


class FarnebackAdapter:
    """OpenCV Farneback dense optical flow, sampled at query points."""

    name = "Farneback"

    # Deep pyramid + small window. winsize=9 is the main smoothness knob:
    # smaller window = less local averaging = less squashing of large
    # displacements. Same logic as zeroing DIS's variational refinement.
    PYR_SCALE = 0.5
    LEVELS = 10
    WIN_SIZE = 9
    ITERATIONS = 10
    POLY_N = 5
    POLY_SIGMA = 1.2
    FLAGS = 0

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_u8 = _to_uint8(reference)
        def_u8 = _to_uint8(deformed)
        flow = cv2.calcOpticalFlowFarneback(
            ref_u8, def_u8, None,
            self.PYR_SCALE, self.LEVELS, self.WIN_SIZE,
            self.ITERATIONS, self.POLY_N, self.POLY_SIGMA, self.FLAGS,
        ).astype(np.float32, copy=False)
        displacements = sample_dense_at_points(flow, query_points)
        valid = np.ones(len(query_points), dtype=bool)
        return displacements, valid
