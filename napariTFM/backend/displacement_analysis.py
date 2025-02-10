from dataclasses import dataclass
from typing import Optional, Dict, Generator

import cv2
import numpy as np

from napariTFM.backend.parameter_dataclasses import DisplacementParameters


class DisplacementAnalyzer:
    """Analyzes displacements using TV-L1 optical flow.

    This class implements displacement analysis for bead tracking in microscopy
    using the TV-L1 optical flow algorithm. It supports both full-resolution
    and downscaled analysis, with methods for calculating, manipulating, and
    applying flow fields.

    The TV-L1 algorithm is particularly suitable for microscopy analysis as it:
    - Preserves discontinuities in the displacement field
    - Is robust to brightness changes
    - Provides sub-pixel accuracy
    """

    def __init__(self, params: Optional[DisplacementParameters] = None):
        """Initialize TV-L1 optical flow analyzer.

        Args:
            params (DisplacementParameters, optional): Algorithm parameters including:
                - tau: Time step for TV-L1 (default determined by params)
                - lambda_: Weight parameter for data term
                - theta: Weight parameter for gradient term
                - nscales: Number of scales for pyramid
                - warps: Number of warpings per scale
                - epsilon: Stopping criterion threshold
                - inner_iterations: Inner iteration count
                - outer_iterations: Outer iteration count
                - scale_step: Scale step for pyramid
                - median_filtering: Whether to apply median filtering
                If None, uses default parameters.
        """
        self.params = params or DisplacementParameters()
        self.flow_algorithm = cv2.optflow.DualTVL1OpticalFlow_create(
            self.params.tau, self.params.lambda_, self.params.theta,
            self.params.nscales, self.params.warps, self.params.epsilon,
            self.params.inner_iterations, self.params.outer_iterations,
            self.params.scale_step, 0.0,
            self.params.median_filtering, False
        )

        self.params = params or DisplacementParameters()
        self.flow_algorithm = cv2.optflow.DualTVL1OpticalFlow_create(
            self.params.tau, self.params.lambda_, self.params.theta,
            self.params.nscales, self.params.warps, self.params.epsilon,
            self.params.inner_iterations, self.params.outer_iterations,
            self.params.scale_step, 0.0,
            self.params.median_filtering, False
        )


    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
        """Calculate optical flow between reference and moving image at full resolution.

        Computes the displacement field between two images using TV-L1 optical flow.
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
        # Ensure images are float32 and normalized
        ref_float = (reference.astype(np.float32) - reference.min()) / (reference.max() - reference.min())
        mov_float = (moving.astype(np.float32) - moving.min()) / (moving.max() - moving.min())

        return self.flow_algorithm.calc(ref_float, mov_float, None)

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








