from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np

from napariTFM.backend._displacement_base import BaseDisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.ffd_displacement import FFDDisplacementAnalyzer
from napariTFM.backend.ilk_displacement import ILKDisplacementAnalyzer
from napariTFM.backend.piv_displacement import PIVDisplacementAnalyzer
from napariTFM.backend.registration import apply_drift, estimate_drift, valid_region

# Map the UI method label (stored verbatim in DisplacementParameters.disp_method) to
# its analyzer. Each analyzer honours DisplacementParameters.disp_device the same way:
# a trusted CPU reference by default, the torch GPU port when available/selected.
_ANALYZERS = {
    "PIV": PIVDisplacementAnalyzer,
    "Lucas-Kanade": ILKDisplacementAnalyzer,
    "FFD": FFDDisplacementAnalyzer,
}


def build_analyzer(params: DisplacementParameters):
    """Construct the displacement analyzer selected by ``params.disp_method``.

    Raises ``ValueError`` on an unknown method label. FFD raises ``RuntimeError`` at
    construction when no CUDA GPU is available (it is GPU-only).
    """
    try:
        cls = _ANALYZERS[params.disp_method]
    except KeyError:
        raise ValueError(
            f"Unknown displacement method {params.disp_method!r}; "
            f"expected one of {sorted(_ANALYZERS)}."
        )
    return cls(params)


@dataclass
class DisplacementResult:
    """Results from displacement field calculation."""

    displacement_field: np.ndarray  # Shape (t, y, x, 2) for time series, units in µm
    original_shape: tuple  # Original image shape (y, x)
    displacement_field_shape: tuple  # Displacement field shape (y, x)
    parameters: DisplacementParameters
    physical_scale: dict  # Dictionary containing physical scaling information
    # Per-frame bulk translation (stage drift) of each target frame relative to
    # the anchor (first frame), shape (t, 2) in pixels, ordered [u_x, u_y] to
    # match displacement_field's last axis. Estimated by the registration step
    # that removed it before measurement; retained so the same shift registers
    # the cell channel into the anchor frame for overlays.
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
    analyzer: Optional[BaseDisplacementAnalyzer] = None,
) -> Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]:
    """Calculate the displacement field between a reference image and target image(s).

    Dispatches to the backend selected by ``params.disp_method`` (PIV, Lucas-Kanade,
    or FFD) via :func:`build_analyzer`. Before measuring, it removes bulk stage drift
    by rigidly registering the reference and every target frame to a common anchor
    (the first target frame) with parameter-free phase cross-correlation, so the
    displacement method only ever sees cell-induced deformation, not drift. This is
    what keeps the residual within each method's capture range (critical for the
    capture-limited Lucas-Kanade and FFD). The per-frame drift is recorded in
    ``drift_pixels`` and reused to register the cell channel for overlays.

    Registering by a shift fabricates a border strip (edge replication), so the
    measurement is confined to the interior region every registered frame still
    fills with real, co-observed data (:func:`registration.valid_region`); the
    excluded border is returned as zero displacement. The output field keeps the
    full frame shape regardless.

    Yields per-frame displacement fields in physical units with 1-based frame
    progress, and returns a complete :class:`DisplacementResult` when exhausted.
    """
    is_valid, error_msg = validate_displacement_image(reference)
    if not is_valid:
        raise ValueError(f"Invalid reference image: {error_msg}")

    is_valid, error_msg = validate_displacement_image(target)
    if not is_valid:
        raise ValueError(f"Invalid target image: {error_msg}")

    if target.ndim == 2:
        target = target[np.newaxis, ...]

    analyzer = analyzer or build_analyzer(params)
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

    # Register everything to the first target frame (the anchor) so bulk stage drift
    # is gone before the displacement method runs. Estimate every bulk drift first
    # (reference + all frames): the whole set is needed up front to fix the common
    # measurement region. drift_pixels[frame] is that frame's bulk (u_x, u_y) drift
    # relative to the anchor, reused to move the cell channel into the same frame
    # downstream (batch_analysis._cells_for_overlay). FTTC nulls the DC mode, so the
    # shared anchor leaves traction unchanged.
    anchor = target[0]
    ref_drift = estimate_drift(anchor, reference)
    for frame in range(total_frames):
        drift_pixels[frame] = estimate_drift(anchor, target[frame])

    # Registering by a shift fabricates a border strip (edge replication). Crop every
    # registered frame to the interior box that ALL frames still fill with real,
    # co-observed data, so no method measures a fabricated border. The measurement
    # runs on the crop and is re-embedded into a full-size zero field, so the excluded
    # border reads as no motion and the output shape is unchanged.
    r0, r1, c0, c1 = valid_region(ref_drift, drift_pixels, (target.shape[1], target.shape[2]))
    reference_registered = apply_drift(reference, ref_drift)[r0:r1, c0:c1]

    for frame in range(total_frames):
        frame_registered = apply_drift(target[frame], drift_pixels[frame])[r0:r1, c0:c1]
        field_crop = analyzer.calculate_flow(reference_registered, frame_registered)

        displacement_field_pixels = np.zeros(
            (target.shape[1], target.shape[2], 2), dtype=np.float32
        )
        displacement_field_pixels[r0:r1, c0:c1] = field_crop

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





