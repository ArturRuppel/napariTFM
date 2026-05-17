from typing import Tuple, Generator

import numpy as np

from napariTFM.backend.displacement_analysis import (
    DisplacementAnalyzer,
    DisplacementResult,
    calculate_displacement_field,
    validate_displacement_image,
)
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.parameter_validation import validate_displacement_parameters


class DisplacementService:
    """Service layer handling business logic for displacement analysis using optical flow.

    This class provides a high-level interface for calculating displacement fields
    between microscopy images using OpenCV DIS optical flow. It handles
    parameter validation, unit conversion, and supports both single-frame and
    time series analysis.
    """

    def __init__(self, params: DisplacementParameters):
        """Initialize displacement service with analysis parameters.

        Args:
            params (DisplacementParameters): Configuration including:
                - DIS algorithm parameters
                - Physical parameters (pixel size, frame interval)
                - Processing options (downscaling factor)
                - Visualization settings

        Raises:
            ValueError: If any parameters are invalid

        Example:
            >>> params = DisplacementParameters(
            ...     pixel_size=0.1,  # 0.1 μm per pixel
            ...     downscale_factor=4
            ... )
            >>> service = DisplacementService(params)
        """
        is_valid, error_msg = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(error_msg)

        self.analyzer = DisplacementAnalyzer(params)
        self.params = params

    def update_parameters(self, parameters: DisplacementParameters):
        """Update displacement analysis parameters.

        Creates a new analyzer instance with the updated parameters after
        validating them.

        Args:
            parameters (DisplacementParameters): New parameters to use

        Raises:
            ValueError: If any parameters are invalid
        """
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)

        self.analyzer = DisplacementAnalyzer(parameters)
        self.params = parameters

    @staticmethod
    def validate_parameters(params: DisplacementParameters) -> Tuple[bool, str]:
        """Validate displacement analysis parameters.

        Checks all parameters for physical and numerical validity including:
        - DIS optical flow parameters
        - Physical parameters (pixel size, frame interval)
        - Processing options (downscaling factor)
        - Visualization settings

        Args:
            params (DisplacementParameters): Parameters to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
                error_message is empty string if valid

        Note:
            Specific validation rules include:
            - DIS pyramid scales and iteration counts must be valid
            - Downscaling factors must be ≥ 1
            - Physical parameters must be positive
            - Visualization stride and scale must be positive
        """
        return validate_displacement_parameters(params)

    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """Validate input image data for displacement analysis."""
        return validate_displacement_image(image)

    def calculate_displacement_field(
            self,
            reference: np.ndarray,
            target: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]:
        """Calculate optical flow between reference and target image(s).

        Computes displacement fields using DIS optical flow, with optional
        downscaling for efficiency. For time series, yields intermediate results
        to allow progress tracking.

        Args:
            reference (np.ndarray): Reference image (2D array)
            target (np.ndarray): Target image(s)
                - 2D array for single frame
                - 3D array (t, y, x) for time series

        Yields:
            Tuple[np.ndarray, int, int]: Intermediate results containing:
                - Current frame's displacement field (y, x, 2) in μm
                - Frame index (1-based)
                - Total number of frames

        Returns:
            DisplacementResult: Complete calculation results including:
                - Full displacement field array
                - Physical scaling information
                - Calculation parameters
                Accessible via StopIteration.value when generator completes.

        Raises:
            ValueError: If input images are invalid

        Example:
            >>> # Get the generator
            >>> disp_generator = service.calculate_displacement_field(ref_img, target_imgs)
            >>>
            >>> # Process intermediate results
            >>> try:
            ...     while True:
            ...         # Get next frame result
            ...         disp_field, frame, total = next(disp_generator)
            ...         print(f"Processed frame {frame}/{total}")
            ... except StopIteration as e:
            ...     # Get final result from generator's return value
            ...     final_result = e.value
        """
        return calculate_displacement_field(
            reference,
            target,
            self.params,
            analyzer=self.analyzer,
        )
