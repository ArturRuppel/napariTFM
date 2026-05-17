from typing import Protocol

import cv2
import numpy as np


class FlowAdapter(Protocol):
    """Uniform interface every algorithm wrapper exposes.

    Implementations are responsible for any algorithm-specific preprocessing
    (e.g., float→uint8 conversion). The runner passes float32 images in
    [0, 1] and the same set of query points to every adapter, and treats the
    return value as opaque.
    """

    name: str

    def displacements_at(
        self,
        reference: np.ndarray,
        deformed: np.ndarray,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict displacements at the given query points.

        Args:
            reference: float32, shape (H, W), range [0, 1].
            deformed:  float32, shape (H, W), range [0, 1].
            query_points: float32, shape (N, 2), columns (x, y) in pixels.

        Returns:
            displacements: float32, shape (N, 2), columns (dx, dy) in pixels.
            valid_mask:    bool,    shape (N,). True where the prediction is
                           trustworthy. Dense methods return all-True.
        """
        ...


def sample_dense_at_points(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear sample a dense flow field at scattered points.

    Args:
        flow:   shape (H, W, 2), dtype convertible to float32.
        points: shape (N, 2), columns (x, y).

    Returns:
        shape (N, 2), columns (dx, dy). Points outside the image are clamped
        to the nearest edge value.
    """
    h, w = flow.shape[:2]
    fx = np.clip(points[:, 0].astype(np.float32), 0.0, w - 1.0)
    fy = np.clip(points[:, 1].astype(np.float32), 0.0, h - 1.0)

    dx = cv2.remap(
        flow[..., 0].astype(np.float32),
        fx.reshape(-1, 1),
        fy.reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()
    dy = cv2.remap(
        flow[..., 1].astype(np.float32),
        fx.reshape(-1, 1),
        fy.reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()

    return np.column_stack([dx, dy]).astype(np.float32)
