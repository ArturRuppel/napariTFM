import cv2
import numpy as np


def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32, copy=False)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo <= 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - lo) / (hi - lo) * 255.0
    return np.ascontiguousarray(scaled.astype(np.uint8))


class LucasKanadeAdapter:
    """OpenCV pyramidal Lucas-Kanade sparse optical flow."""

    name = "Lucas-Kanade"

    # Tuned: deep pyramid for displacement reach, modest window to keep
    # sub-pixel accuracy. Same principle as the validation script's DIS
    # config: lots of pyramid levels + lots of iterations.
    WIN_SIZE = (15, 15)
    MAX_LEVEL = 7
    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_u8 = _to_uint8(reference)
        def_u8 = _to_uint8(deformed)
        # OpenCV expects shape (N, 1, 2) and float32 for point arrays.
        pts = query_points.astype(np.float32).reshape(-1, 1, 2)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            ref_u8, def_u8, pts, None,
            winSize=self.WIN_SIZE,
            maxLevel=self.MAX_LEVEL,
            criteria=self.CRITERIA,
        )
        new_pts = new_pts.reshape(-1, 2)
        displacements = (new_pts - query_points.astype(np.float32)).astype(np.float32)
        valid = status.reshape(-1).astype(bool)
        # Zero-out displacements where LK failed so downstream consumers
        # cannot accidentally read stale numbers; the valid mask is the truth.
        displacements[~valid] = 0.0
        return displacements, valid
