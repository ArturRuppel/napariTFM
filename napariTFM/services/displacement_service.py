from dataclasses import dataclass
from typing import Tuple, Generator

import numpy as np

from napariTFM.backend.displacement_analysis import DisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.parameter_validation import validate_displacement_parameters


@dataclass
class DisplacementResult:
    """Results from displacement field calculation using optical flow.

    Attributes:
        displacement_field (np.ndarray): Calculated displacement field with shape (t, y, x, 2).
            t is time points (1 for single frame), units in micrometers (μm).
            Last dimension contains (dx, dy) displacement components.
        original_shape (tuple): Original image shape (y, x)
        displacement_field_shape (tuple): Shape of displacement field (y, x)
        parameters (DisplacementParameters): Parameters used for calculation
        physical_scale (dict): Physical scaling information including:
            - pixel_size: Size of each pixel
            - grid_spacing: Effective grid spacing after downsampling
            - time_interval: Time between frames
            - displacement_units: Displacement units (μm)
            - grid_spacing_units: Spatial units (μm)
            - time_interval_units: Time units (min)
    """
    displacement_field: np.ndarray  # Shape (t, y, x, 2) for time series, units in µm
    original_shape: tuple  # Original image shape (y, x)
    displacement_field_shape: tuple  # Displacement_field field shape (y, x)
    parameters: DisplacementParameters
    physical_scale: dict  # Dictionary containing physical scaling information


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
            - Iterations must be positive
            - Scaling factors must be ≥ 1
            - Physical parameters must be positive
            - Theta must be between 0 and 10
        """
        return validate_displacement_parameters(params)

    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """Validate input image data for displacement analysis.

        Checks that image data is suitable for optical flow calculation:
        - Correct data type (numpy array)
        - Valid dimensions (2D or 3D for time series)
        - Contains valid values (not all NaN)

        Args:
            image (np.ndarray): Image data to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
                error_message is empty string if valid
        """
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
        # Validate input images
        is_valid, error_msg = self.validate_image(reference)
        if not is_valid:
            raise ValueError(f"Invalid reference image: {error_msg}")

        is_valid, error_msg = self.validate_image(target)
        if not is_valid:
            raise ValueError(f"Invalid target image: {error_msg}")

        # Ensure target is 3D
        if target.ndim == 2:
            target = target[np.newaxis, ...]

        total_frames = target.shape[0]

        # Calculate output shape based on downscaling
        if self.params.downscale_factor > 1:
            displacement_field_shape = (
                total_frames,
                target.shape[1] // self.params.downscale_factor,
                target.shape[2] // self.params.downscale_factor,
                2
            )
        else:
            displacement_field_shape = (total_frames, target.shape[1], target.shape[2], 2)

        displacement_field_stack = np.zeros(displacement_field_shape, dtype=np.float32)

        # Calculate displacement_field for each frame
        for frame in range(total_frames):
            # Calculate displacement_field in pixels
            displacement_field_pixels = self.analyzer.calculate_flow(reference, target[frame])

            # Apply downscaling if needed
            if self.params.downscale_factor > 1:
                displacement_field_pixels = self.analyzer.downscale_flow(displacement_field_pixels, self.params.downscale_factor)

            # Convert to physical units (µm)
            displacement_field_stack[frame] = displacement_field_pixels * self.params.pixel_size

            # Yield intermediate result with progress info
            yield displacement_field_stack[frame].copy(), frame + 1, total_frames

        # Create physical scale information
        physical_scale = {
            'pixel_size': self.params.pixel_size,
            'grid_spacing': self.params.pixel_size * self.params.downscale_factor,
            'time_interval': self.params.frame_interval,
            'displacement_units': 'µm',
            'grid_spacing_units': 'µm',
            'time_interval_units': 'min',
        }

        return DisplacementResult(
            displacement_field=displacement_field_stack,
            original_shape=reference.shape,
            displacement_field_shape=displacement_field_stack.shape[1:3],
            parameters=self.params,
            physical_scale=physical_scale
        )
