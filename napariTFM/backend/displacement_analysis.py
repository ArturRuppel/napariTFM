from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np

from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.piv_displacement import PIVDisplacementAnalyzer


@dataclass
class DisplacementResult:
    """Results from displacement field calculation."""

    displacement_field: np.ndarray  # Shape (t, y, x, 2) for time series, units in µm
    original_shape: tuple  # Original image shape (y, x)
    displacement_field_shape: tuple  # Displacement field shape (y, x)
    parameters: DisplacementParameters
    physical_scale: dict  # Dictionary containing physical scaling information
    # Per-frame bulk translation (stage drift) removed from the field, shape
    # (t, 2) in pixels, ordered [u_x, u_y] to match displacement_field's last
    # axis. This is the translation-correction PIV absorbed; retained so the
    # same shift can be applied to the cell channel.
    drift_pixels: Optional[np.ndarray] = None


def validate_displacement_image(image: np.ndarray) -> Tuple[bool, str]:
    """Validate input image data for displacement analysis."""
    if image is None:
        return False, "No image data provided"

    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"

    if image.ndim not in (2, 3):
        return False, "Image must be 2D or 3D (time series)"

    if np.all(np.isnan(image)):
        return False, "Image contains only NaN values"

    return True, ""


def calculate_displacement_field(
    reference: np.ndarray,
    target: np.ndarray,
    params: DisplacementParameters,
    analyzer: Optional[PIVDisplacementAnalyzer] = None,
) -> Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]:
    """Calculate the displacement field between a reference image and target image(s).

    Uses the multi-pass PIV backend (:class:`PIVDisplacementAnalyzer`). Yields
    per-frame displacement fields in physical units with 1-based frame progress,
    and returns a complete :class:`DisplacementResult` when exhausted.
    """
    is_valid, error_msg = validate_displacement_image(reference)
    if not is_valid:
        raise ValueError(f"Invalid reference image: {error_msg}")

    is_valid, error_msg = validate_displacement_image(target)
    if not is_valid:
        raise ValueError(f"Invalid target image: {error_msg}")

    if target.ndim == 2:
        target = target[np.newaxis, ...]

    analyzer = analyzer or PIVDisplacementAnalyzer(params)
    total_frames = target.shape[0]

    if params.downscale_factor > 1:
        displacement_field_shape = (
            total_frames,
            target.shape[1] // params.downscale_factor,
            target.shape[2] // params.downscale_factor,
            2,
        )
    else:
        displacement_field_shape = (total_frames, target.shape[1], target.shape[2], 2)

    displacement_field_stack = np.zeros(displacement_field_shape, dtype=np.float32)
    drift_pixels = np.zeros((total_frames, 2), dtype=np.float32)

    for frame in range(total_frames):
        displacement_field_pixels = analyzer.calculate_flow(reference, target[frame])

        # Fold translation-correction into PIV. Bulk stage drift between the
        # reference and this frame is a spatially-uniform offset that the
        # multi-pass coarse pass already captures; estimate it robustly as the
        # per-frame median vector and subtract it, so the reported field is the
        # cell-induced deformation only — the job image-level registration used
        # to do. FTTC nulls the DC mode, so this leaves traction unchanged; it
        # only cleans the displacement output and hands back the shift for the
        # cell channel.
        frame_drift = np.median(displacement_field_pixels.reshape(-1, 2), axis=0)
        displacement_field_pixels = displacement_field_pixels - frame_drift
        drift_pixels[frame] = frame_drift

        if params.downscale_factor > 1:
            displacement_field_pixels = analyzer.downscale_flow(
                displacement_field_pixels,
                params.downscale_factor,
            )

        displacement_field_stack[frame] = displacement_field_pixels * params.pixel_size

        yield displacement_field_stack[frame].copy(), frame + 1, total_frames

    physical_scale = {
        "pixel_size": params.pixel_size,
        "grid_spacing": params.pixel_size * params.downscale_factor,
        "time_interval": params.frame_interval,
        "displacement_units": "µm",
        "grid_spacing_units": "µm",
        "time_interval_units": "min",
    }

    return DisplacementResult(
        displacement_field=displacement_field_stack,
        original_shape=reference.shape,
        displacement_field_shape=displacement_field_stack.shape[1:3],
        parameters=params,
        physical_scale=physical_scale,
        drift_pixels=drift_pixels,
    )





