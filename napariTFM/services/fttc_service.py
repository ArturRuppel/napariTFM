from dataclasses import dataclass
from typing import Tuple, Generator

import numpy as np

from napariTFM.backend.fttc import FTTC
from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend.parameter_validation import validate_fttc_parameters


@dataclass
class FTTCResult:
    """Results from FTTC force calculation.

    Attributes:
        force_field (np.ndarray): Calculated force field with shape (t, y, x, 2).
            t is time points (1 for single frame), units in Pascals.
        original_shape (tuple): Original displacement field shape (y, x)
        force_shape (tuple): Force field shape (y, x)
        parameters (FTTCParameters): Parameters used for calculation
        physical_scale (dict): Physical scaling information including:
            - pixel_size: Size of each pixel
            - grid_spacing: Effective grid spacing after downsampling
            - time_interval: Time between frames
            - force_units: Force units (Pa)
            - grid_spacing_units: Spatial units (μm)
            - time_interval_units: Time units (min)
    """
    force_field: np.ndarray  # Shape: (t, y, x, 2) for time series, units in Pa
    original_shape: tuple  # Original displacement field shape (y, x)
    force_shape: tuple  # Force field shape (y, x)
    parameters: FTTCParameters
    physical_scale: dict  # Dictionary containing physical scaling information



class FTTCService:
    """Service layer for handling FTTC force calculations.

    This class provides a high-level interface for calculating traction forces from
    displacement field measurements. It handles data validation, parameter management,
    and both single-frame and time series calculations.

    The service supports automatic regularization parameter selection using GCV and
    provides progress updates during calculation of multi-frame datasets.
    """

    def __init__(self, params: FTTCParameters):
        """Initialize FTTC service with calculation parameters.

        Args:
            params (FTTCParameters): Configuration for FTTC calculations including:
                - Material properties (Young's modulus, Poisson ratio)
                - Calculation parameters (regularization, downscaling)
                - Visualization settings

        Raises:
            ValueError: If any parameters are invalid
        """
        is_valid, error_msg = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(error_msg)

        self.calculator = FTTC(params)
        self.params = params

    def calculate_forces(
            self,
            displacement_field: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, FTTCResult]:
        """Calculate traction forces from displacement field data.

        Handles both single frames and time series data. For time series, yields
        intermediate results to allow progress tracking. The final result is returned
        through the generator's return value (accessible via StopIteration.value).

        Args:
            displacement_field (np.ndarray): Displacement measurements with shape:
                - (y, x, 2) for single frame
                - (t, y, x, 2) for time series
                where final dimension contains (dx, dy) displacements in μm

        Yields:
            Tuple[np.ndarray, int, int]: Intermediate results containing:
                - Current frame's force field with shape (y, x, 2)
                - Frame index (1-based)
                - Total number of frames

        Returns:
            FTTCResult: Complete calculation results including:
                - Full force field array
                - Physical scaling information
                - Calculation parameters
                Accessible via StopIteration.value when generator completes.

        Raises:
            ValueError: If displacement field has invalid shape or values

        Example:
            >>> service = FTTCService(params)
            >>> # Get the generator
            >>> force_generator = service.calculate_forces(displacements)
            >>>
            >>> # Process intermediate results
            >>> try:
            ...     while True:
            ...         # Get next frame result
            ...         force_field, frame, total = next(force_generator)
            ...         print(f"Processed frame {frame}/{total}")
            ... except StopIteration as e:
            ...     # Get final result from generator's return value
            ...     final_result = e.value
            ...
            >>> # Access results
            >>> print(f"Final force field shape: {final_result.force_field.shape}")
            >>> print(f"Max force: {np.max(final_result.force_field)} Pa")
        """
        # Ensure displacement field is 4D
        if displacement_field.ndim == 3:
            displacement_field = displacement_field[np.newaxis, ...]

        total_frames = displacement_field.shape[0]
        force_shape = displacement_field.shape[1:4]  # (y, x, 2)
        force_stack = np.zeros((total_frames, *force_shape), dtype=np.float32)

        for frame in range(total_frames):
            # Calculate forces for current frame
            result = self.calculator.calculate_traction(
                displacements=displacement_field[frame],
                pixel_size=self.params.pixel_size,
                downscale_factor=self.params.downscale_factor,
                regularization=None if self.params.auto_gcv else self.params.regularization
            )

            # Extract force components and store in stack
            force_stack[frame, ..., 0] = result[1][0]  # tx
            force_stack[frame, ..., 1] = result[1][1]  # ty

            # Yield intermediate result with progress info
            yield force_stack[frame].copy(), frame + 1, total_frames

        # Create physical scale information
        physical_scale = {
            'pixel_size': self.params.pixel_size,
            'grid_spacing': self.params.pixel_size * self.params.downscale_factor,
            'time_interval': self.params.frame_interval,
            'force_units': 'Pa',
            'grid_spacing_units': 'µm',
            'time_interval_units': 'min',
        }

        return FTTCResult(
            force_field=force_stack,
            original_shape=displacement_field.shape[1:3],
            force_shape=force_stack.shape[1:3],
            parameters=self.params,
            physical_scale=physical_scale,
        )

    def find_optimal_regularization(
            self,
            displacement_field: np.ndarray
    ) -> float:
        """Find optimal regularization parameter using Generalized Cross-Validation.

        Uses the GCV method to automatically determine the Tikhonov regularization
        parameter that best balances noise suppression and force resolution.

        Args:
            displacement_field (np.ndarray): Displacement field with shape (y, x, 2)
                containing (dx, dy) displacements in μm

        Returns:
            float: Optimal regularization parameter λ

        Note:
            This is computationally intensive and should be used sparingly,
            typically to determine a good parameter value for a dataset that
            can then be reused for similar experimental conditions.
        """
        shape = displacement_field.shape[:-1]
        pos = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy'))
        vec = np.array([displacement_field[..., 0], displacement_field[..., 1]])
        return self.calculator._find_regularization(
            pos, vec, self.params.pixel_size * self.params.downscale_factor, shape[1], shape[0])

    @staticmethod
    def validate_displacement_field(displacement_field: np.ndarray) -> Tuple[bool, str]:
        """Validate displacement field data format and values.

        Checks for:
        - Correct array type and shape
        - Presence of required displacement components
        - Valid data values (not all NaN)

        Args:
            displacement_field (np.ndarray): Displacement field to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
                error_message is empty string if valid
        """
        if displacement_field is None:
            return False, "No displacement field data provided"

        if not isinstance(displacement_field, np.ndarray):
            return False, "Displacement field must be a numpy array"

        if displacement_field.ndim not in (3, 4):
            return False, "Displacement field must be 3D (y,x,2) or 4D (t,y,x,2)"

        if displacement_field.shape[-1] != 2:
            return False, f"Last dimension must be 2 (x,y components), got {displacement_field.shape[-1]}"

        if np.all(np.isnan(displacement_field)):
            return False, "Displacement field contains only NaN values"

        return True, ""

    @staticmethod
    def validate_parameters(params: FTTCParameters) -> Tuple[bool, str]:
        """Validate FTTC calculation parameters.

        Checks all parameters for physical validity including:
        - Material properties (positive Young's modulus, valid Poisson ratio)
        - Calculation parameters (positive regularization, valid downscaling)
        - Visualization settings (valid scales and strides)

        Args:
            params (FTTCParameters): Parameters to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
                error_message is empty string if valid
        """
        return validate_fttc_parameters(params)

    def update_parameters(self, parameters: FTTCParameters):
        """Update FTTC calculation parameters.

        Creates a new calculator instance with the updated parameters after
        validating them.

        Args:
            parameters (FTTCParameters): New parameters to use

        Raises:
            ValueError: If any parameters are invalid
        """
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)
        self.calculator = FTTC(parameters)
        self.params = parameters
