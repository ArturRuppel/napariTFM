import cv2
import numpy as np

from _dev.optical_flow_comparison.adapters.lucas_kanade import _to_uint8


class LucasKanadeFBAdapter:
    """LK with forward-backward consistency filtering.

    Tracks forward (ref -> deformed) and then backward from the predicted
    positions (deformed -> ref). The round-trip distance |bwd - original_ref|
    is the per-bead trust score: real tracks round-trip to within ~0.1 px,
    spurious tracks (locked onto the wrong neighbor) do not. Beads whose
    round-trip exceeds FB_THRESHOLD are marked invalid.

    Uses the same LK parameters as `LucasKanadeAdapter` so the only difference
    in the comparison is the FB filter itself.
    """

    name = "Lucas-Kanade-FB"

    WIN_SIZE = (15, 15)
    MAX_LEVEL = 7
    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    FB_THRESHOLD = 1.0  # pixels

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_u8 = _to_uint8(reference)
        def_u8 = _to_uint8(deformed)
        qp = query_points.astype(np.float32)
        pts_fwd_in = qp.reshape(-1, 1, 2)

        fwd_pts, fwd_status, _ = cv2.calcOpticalFlowPyrLK(
            ref_u8, def_u8, pts_fwd_in, None,
            winSize=self.WIN_SIZE, maxLevel=self.MAX_LEVEL, criteria=self.CRITERIA,
        )
        bwd_pts, bwd_status, _ = cv2.calcOpticalFlowPyrLK(
            def_u8, ref_u8, fwd_pts, None,
            winSize=self.WIN_SIZE, maxLevel=self.MAX_LEVEL, criteria=self.CRITERIA,
        )

        fwd_pts_2d = fwd_pts.reshape(-1, 2)
        bwd_pts_2d = bwd_pts.reshape(-1, 2)
        round_trip = np.hypot(bwd_pts_2d[:, 0] - qp[:, 0], bwd_pts_2d[:, 1] - qp[:, 1])

        valid = (
            fwd_status.reshape(-1).astype(bool)
            & bwd_status.reshape(-1).astype(bool)
            & (round_trip < self.FB_THRESHOLD)
        )

        displacements = (fwd_pts_2d - qp).astype(np.float32)
        displacements[~valid] = 0.0
        return displacements, valid
