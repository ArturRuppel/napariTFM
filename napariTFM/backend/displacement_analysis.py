from typing import Optional

import cv2
import numpy as np

from napariTFM.backend.parameter_dataclasses import DisplacementParameters


class DisplacementAnalyzer:
    """Analyzes displacements using dense optical flow.

    This class implements displacement analysis for bead tracking in microscopy
    using OpenCV's DIS optical flow algorithm. It supports both full-resolution
    and downscaled analysis, with methods for calculating, manipulating, and
    applying flow fields.
    """

    def __init__(self, params: Optional[DisplacementParameters] = None):
        """Initialize DIS optical flow analyzer.

        Args:
            params (DisplacementParameters, optional): Algorithm parameters including:
                - nscales: Number of scales for pyramid
                - inner_iterations: Inner iteration count
                - outer_iterations: Outer iteration count
                If None, uses default parameters.
        """
        self.params = params or DisplacementParameters()
        self.flow_algorithm = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.flow_algorithm.setCoarsestScale(max(0, self.params.nscales - 1))
        self.flow_algorithm.setGradientDescentIterations(max(1, self.params.inner_iterations))
        self.flow_algorithm.setVariationalRefinementIterations(max(0, self.params.outer_iterations))

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Calculate optical flow between reference and moving image at full resolution.

        Computes the displacement field between two images using DIS optical flow.
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
            Images are normalized to [0, 1] range before processing to ensure
            consistent results regardless of input intensity range.
        """
        ref_image = self._normalize_for_optical_flow(reference)
        mov_image = self._normalize_for_optical_flow(moving)

        return self.flow_algorithm.calc(ref_image, mov_image, None).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_for_optical_flow(image: np.ndarray) -> np.ndarray:
        """Convert microscopy intensity data to the 8-bit format expected by DIS."""
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







