from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from napariTFM.backend.parameter_dataclasses import DisplacementParameters


@dataclass
class DisplacementResult:
    """Results from displacement field calculation using optical flow."""

    displacement_field: np.ndarray  # Shape (t, y, x, 2) for time series, units in µm
    original_shape: tuple  # Original image shape (y, x)
    displacement_field_shape: tuple  # Displacement field shape (y, x)
    parameters: DisplacementParameters
    physical_scale: dict  # Dictionary containing physical scaling information


class DisplacementAnalyzer:
    """Analyzes displacements using dense optical flow.

    This class implements displacement analysis for bead tracking in microscopy
    using OpenCV's Farneback optical flow algorithm. It supports both
    full-resolution and downscaled analysis, with methods for calculating,
    manipulating, and applying flow fields.
    """

    def __init__(self, params: Optional[DisplacementParameters] = None):
        """Initialize Farneback optical flow analyzer.

        Args:
            params (DisplacementParameters, optional): Algorithm parameters including:
                - nscales: Number of pyramid levels
                - inner_iterations: Farneback iteration count
                - median_filtering: Farneback window size
                - pyr_scale / poly_n / poly_sigma: Farneback internals
                - use_gaussian_window: Gaussian vs. box windowing
                If None, uses default parameters.
        """
        self.params = params or DisplacementParameters()
        self.algorithm_name = "Farneback"

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Calculate optical flow between reference and moving image at full resolution.

        Computes the displacement field between two images using Farneback optical flow.
        Images are automatically normalized before processing.

        Args:
            reference (np.ndarray): Reference (fixed) image
            moving (np.ndarray): Moving (deformed) image
                Both images should be 2D arrays of the same shape.

        Returns:
            np.ndarray: Optical flow field with shape (H, W, 2) where:
                - H, W are the image dimensions
                - Last dimension contains (dx, dy) displacements in pixels
                Positive values indicate rightward/downward motion.

        Note:
            Images are normalized to 8-bit before processing to ensure
            consistent results regardless of input intensity range.
        """
        ref_image = self._normalize_for_optical_flow(reference)
        mov_image = self._normalize_for_optical_flow(moving)

        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if self.params.use_gaussian_window else 0
        return cv2.calcOpticalFlowFarneback(
            ref_image,
            mov_image,
            None,
            float(self.params.pyr_scale),
            max(1, self.params.nscales),
            self._farneback_window_size(),
            max(1, self.params.inner_iterations),
            max(1, int(self.params.poly_n)),
            float(self.params.poly_sigma),
            flags,
        ).astype(np.float32, copy=False)

    def _farneback_window_size(self) -> int:
        """Return an odd, positive Farneback window size from the legacy field."""
        window_size = max(1, int(self.params.median_filtering))
        if window_size % 2 == 0:
            window_size += 1
        return window_size

    @staticmethod
    def _normalize_for_optical_flow(image: np.ndarray) -> np.ndarray:
        """Convert microscopy intensity data to the 8-bit format expected by OpenCV."""
        image_float = image.astype(np.float32, copy=False)
        image_range = image_float.max() - image_float.min()

        if image_range <= 1e-8:
            normalized = np.zeros_like(image_float, dtype=np.uint8)
        else:
            normalized = ((image_float - image_float.min()) / image_range * 255).astype(np.uint8)

        return np.ascontiguousarray(normalized)

    def downscale_flow(self, flow: np.ndarray, factor: int) -> np.ndarray:
        """Downscale flow field using local averaging.

        Reduces flow field resolution while preserving vector information by
        averaging displacement vectors within local neighborhoods.

        Args:
            flow (np.ndarray): Input flow field of shape (H, W, 2)
            factor (int): Downscaling factor
                Output dimensions will be H/factor × W/factor

        Returns:
            np.ndarray: Downscaled flow field of shape (H/factor, W/factor, 2)
                Vector magnitudes are preserved (not scaled)

        Note:
            Uses simple averaging of vectors within each block. For factor=1,
            returns the input flow field unchanged.
        """
        if factor <= 1:
            return flow

        h, w = flow.shape[:2]
        new_h, new_w = h // factor, w // factor

        # Handle each component separately to preserve vector information
        downscaled = np.zeros((new_h, new_w, 2))

        for i in range(new_h):
            for j in range(new_w):
                # Extract block
                y_start = i * factor
                y_end = min((i + 1) * factor, h)
                x_start = j * factor
                x_end = min((j + 1) * factor, w)

                block = flow[y_start:y_end, x_start:x_end]
                # Average the x and y components separately
                downscaled[i, j] = np.mean(block, axis=(0, 1))

        return downscaled


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
    analyzer: Optional[DisplacementAnalyzer] = None,
) -> Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]:
    """Calculate optical flow between a reference image and target image(s).

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

    analyzer = analyzer or DisplacementAnalyzer(params)
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

    for frame in range(total_frames):
        displacement_field_pixels = analyzer.calculate_flow(reference, target[frame])

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
    )





