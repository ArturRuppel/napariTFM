"""Pure vector-field helpers shared by the live viewer and batch export.

Both the interactive :class:`VisualizationManager` (napari Vectors overlay) and
the batch :class:`~napariTFM.backend.batch_visualizations.BatchVisualizationSaver`
(matplotlib quiver) sample a displacement/force field onto a regular grid and
colour each arrow by magnitude. Keeping that geometry in one place is what lets
the exported movies match the viewer (worklist §7) even though the two use
different renderers.

No napari/Qt imports here: just numpy, cv2 and matplotlib colormaps, so the
math stays importable and unit-testable without a display.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt


def upscale_field(field: np.ndarray, downscale_factor: int) -> np.ndarray:
    """Upscale a 2D vector field ``(H, W, 2)`` by an integer factor for display."""
    if downscale_factor <= 1:
        return field

    return cv2.resize(
        field,
        (field.shape[1] * downscale_factor, field.shape[0] * downscale_factor),
        interpolation=cv2.INTER_LINEAR,
    )


def build_frame_vectors(
    flow_scaled: np.ndarray,
    original_flow: np.ndarray,
    stride: int,
    v_max: Optional[float],
    colormap: str = "viridis",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one frame's field onto a regular grid as napari vectors + colours.

    Parameters
    ----------
    flow_scaled
        Field already scaled for *display* arrow length, ``(H, W, 2)``.
    original_flow
        The *unscaled* field, used only to colour each arrow by its true
        magnitude, ``(H, W, 2)``.
    stride
        Grid spacing between sampled arrows (in pixels of the display field).
    v_max
        Magnitude that maps to the top of *colormap*; ``None`` uses the frame's
        own max.
    colormap
        Matplotlib colormap name (``viridis`` for displacement, ``inferno`` for
        force).

    Returns
    -------
    vectors : np.ndarray
        ``(N, 2, 2)`` — ``[start(y, x), direction(dy, dx)]`` per arrow, the
        layout napari's 2D Vectors layer expects.
    colors : np.ndarray
        ``(N, 4)`` RGBA, one colour per arrow.
    """
    h, w = flow_scaled.shape[:2]
    stride = max(1, stride)

    y_points = np.arange(stride // 2, h - stride // 2, stride)
    x_points = np.arange(stride // 2, w - stride // 2, stride)
    Y, X = np.meshgrid(y_points, x_points, indexing="ij")

    U = flow_scaled[Y, X, 0]  # x-component
    V = flow_scaled[Y, X, 1]  # y-component

    orig_u = original_flow[Y, X, 0]
    orig_v = original_flow[Y, X, 1]
    magnitudes = np.sqrt(orig_u ** 2 + orig_v ** 2)

    Y_flat = Y.flatten()
    X_flat = X.flatten()
    U_flat = U.flatten()
    V_flat = V.flatten()

    N = len(Y_flat)
    vectors = np.zeros((N, 2, 2))
    vectors[:, 0, 1] = X_flat  # start x
    vectors[:, 0, 0] = Y_flat  # start y
    vectors[:, 1, 1] = U_flat  # direction x
    vectors[:, 1, 0] = V_flat  # direction y

    max_mag = v_max if v_max is not None else magnitudes.max()
    if max_mag > 0:
        normalized = np.clip(magnitudes.flatten() / max_mag, 0, 1)
        colors = plt.get_cmap(colormap)(normalized)
    else:
        colors = plt.get_cmap(colormap)(np.zeros(N))

    return vectors, colors
