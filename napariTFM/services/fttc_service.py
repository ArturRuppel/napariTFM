from dataclasses import dataclass
from typing import Tuple, Generator

import numpy as np

from napariTFM.backend.fttc import FTTC
from napariTFM.backend.parameter_dataclasses import FTTCParameters


@dataclass
class FTTCResult:
    """Results from force calculation"""
    force_field: np.ndarray  # Shape: (t, y, x, 2) for time series, units in Pa
    original_shape: tuple  # Original displacement field shape (y, x)
    force_shape: tuple  # Force field shape (y, x)
    parameters: FTTCParameters
    physical_scale: dict  # Dictionary containing physical scaling information
    condition_number: float
    residual: float


class FTTCService:
    """Service layer handling business logic for FTTC force calculations."""

    def __init__(self, params: FTTCParameters):
        """
        Initialize the FTTC service with calculation parameters.

        Parameters
        ----------
        params : FTTCParameters
            Parameters for the FTTC calculations
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
        """
        Calculate forces from displacement field data.
        Always returns force field with shape (t, y, x, 2) where t=1 for single frames.
        Yields intermediate results during calculation.

        Parameters
        ----------
        displacement_field : np.ndarray
            Displacement field data with shape (t, y, x, 2)

        Returns
        -------
        Generator[Tuple[np.ndarray, int, int], None, FTTCResult]
            Generator yielding (intermediate_forces, frame_index, total_frames) during calculation
            and returning final FTTCCalculationResult when exhausted
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
            condition_number=getattr(self.calculator, 'last_condition_number', 0.0),
            residual=getattr(self.calculator, 'last_residual', 0.0)
        )

    def find_optimal_regularization(
            self,
            displacement_field: np.ndarray
    ) -> float:
        """
        Find optimal regularization parameter using GCV.

        Parameters
        ----------
        displacement_field : np.ndarray
            2D array of displacement vectors (height, width, 2)

        Returns
        -------
        float
            Optimal regularization parameter
        """
        shape = displacement_field.shape[:-1]
        pos = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy'))
        vec = np.array([displacement_field[..., 0], displacement_field[..., 1]])
        return self.calculator._find_regularization(pos, vec, self.params.pixel_size * self.params.downscale_factor)

    @staticmethod
    def validate_displacement_field(displacement_field: np.ndarray) -> Tuple[bool, str]:
        """
        Validate displacement field data.

        Parameters
        ----------
        displacement_field : np.ndarray
            Displacement field data to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
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
        """
        Validate FTTC calculation parameters.

        Parameters
        ----------
        params : FTTCParameters
            Parameters to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        if params.young_modulus <= 0:
            return False, "Young's modulus must be positive"

        if not 0 <= params.poisson_ratio_substrate <= 0.5:
            return False, "Poisson ratio must be between 0 and 0.5"

        if params.gel_height is not None and params.gel_height < 0:
            return False, "Gel height must be non-negative or None (infinite)"

        if params.lanczos_exp < 0:
            return False, "Lanczos exponent must be non-negative"

        if params.regularization <= 0:
            return False, "Regularization parameter must be positive"

        if params.force_vector_stride < 1:
            return False, "Vector stride must be at least 1"

        if params.force_arrow_scale <= 0:
            return False, "Arrow scale must be positive"

        if params.f_max <= 0:
            return False, "Maximum force must be positive"

        if params.frame_interval <= 0:
            return False, "Frame interval must be positive"

        if params.pixel_size <= 0:
            return False, "Pixel size must be positive"

        if params.downscale_factor < 1:
            return False, "Downscale factor must be at least 1"

        return True, ""

    def update_parameters(self, parameters: FTTCParameters):
        """Update preprocessing parameters"""
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)
        self.analyzer = FTTC(parameters)
        self.params = parameters