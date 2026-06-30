"""Stress result for napariTFM's BISM (Bayesian Inversion Stress Microscopy) engine.

napariTFM infers the monolayer's internal stress from the traction field using
BISM (:mod:`napariTFM.backend.bism`) — a Bayesian inversion with no material
parameters and no mesh (a single sparse solve on the regular grid); carries a
traction-reconstruction R² and, optionally, per-pixel posterior uncertainty.

Every downstream consumer — the viewer stream, the ``.ntfm`` writer, the tidy-
table export, the stage-status dots — reads the common fields (``stress_tensor``
+ ``physical_scale``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from skimage.transform import resize


@dataclass
class StressResult:
    """Inferred internal stress field, from the BISM engine.

    Common fields (always populated, read by every downstream consumer):
        stress_tensor: ``(nt, ny, nx, 2, 2)`` Cauchy stress (mN/m). Components are
            ``[..., 0, 0] = sigma_xx``, ``[..., 1, 1] = sigma_yy`` and
            ``[..., 0, 1] = [..., 1, 0] = sigma_xy`` (symmetric).
        parameters: the :class:`StressParameters` the stage ran with.
        physical_scale: pixel size / grid spacing / units dict.
        original_shape / stress_shape: ``(ny, nx)`` of the force grid.
        method: ``"BISM"`` — which engine produced this result.

    BISM diagnostics:
        r2_traction: traction-reconstruction R² (≈1 for a good solve).
    """

    stress_tensor: np.ndarray
    parameters: object
    physical_scale: dict
    original_shape: tuple
    stress_shape: tuple
    method: str = "BISM"

    # BISM diagnostics
    r2_traction: Optional[float] = None


def process_mask_data(
        mask_data: np.ndarray,
        force_field: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[str]]:
    """Convert mask data to a validated boolean stack, resizing to force shape when needed."""
    warnings_list = []

    if mask_data is None:
        raise ValueError("No mask data provided")

    unique_values = np.unique(mask_data)
    unique_values = unique_values[unique_values != 0]
    if len(unique_values) > 1:
        warnings_list.append("Multiple non-zero values detected in mask. Converting to binary (0 and 1).")

    mask_data = mask_data > 0

    if mask_data.ndim == 2:
        mask_data = mask_data[np.newaxis, ...]

    if mask_data.ndim != 3:
        raise ValueError(f"Mask data must be 2D or 3D, got shape {mask_data.shape}")

    if force_field is not None:
        force_shape = force_field.shape[1:3]
        if mask_data.shape[1:] != force_shape:
            mask_data = np.stack([
                resize(
                    frame.astype(float),
                    force_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5
                for frame in mask_data
            ])

    return mask_data, warnings_list
