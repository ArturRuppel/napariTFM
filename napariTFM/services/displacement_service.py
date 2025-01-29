from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator, Union
import numpy as np
from skimage.transform import resize

from napariTFM.backend.displacement_analysis import DisplacementAnalyzer, TVL1Parameters


@dataclass
class DisplacementParameters:
    """Parameters for displacement analysis"""
    # TV-L1 optical flow parameters
    tau: float
    lambda_: float
    theta: float
    nscales: int
    warps: int
    epsilon: float
    inner_iterations: int
    outer_iterations: int
    scale_step: float
    median_filtering: int

    # Analysis parameters
    downscale_factor: int
    pixel_size: float
    frame_interval: float

    # Visualization parameters
    d_max: float
    disp_vector_stride: int
    disp_arrow_scale: float


@dataclass
class DisplacementResult:
    """Results from displacement field calculation"""
    displacement_field: np.ndarray  # Shape (t, y, x, 2) for time series, units in µm
    original_shape: tuple  # Original image shape (y, x)
    displacement_field_shape: tuple  # Displacement_field field shape (y, x)
    parameters: DisplacementParameters
    physical_scale: dict  # Dictionary containing physical scaling information


class DisplacementService:
    """Service layer handling business logic for displacement analysis."""

    def __init__(self, params: DisplacementParameters):
        """
        Initialize the displacement service with analysis parameters.

        Parameters
        ----------
        params : DisplacementParameters
            Parameters for the displacement analysis

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        is_valid, error_msg = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(error_msg)

        tvl1_params = TVL1Parameters(
            tau=params.tau,
            lambda_=params.lambda_,
            theta=params.theta,
            nscales=params.nscales,
            warps=params.warps,
            epsilon=params.epsilon,
            inner_iterations=params.inner_iterations,
            outer_iterations=params.outer_iterations,
            scale_step=params.scale_step,
            median_filtering=params.median_filtering,
            downscale_factor=params.downscale_factor
        )

        self.analyzer = DisplacementAnalyzer(tvl1_params)
        self.params = params

    def update_parameters(self, parameters: DisplacementParameters):
        """
        Update displacement analysis parameters.

        Parameters
        ----------
        parameters : DisplacementParameters
            New parameters to use

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)

        tvl1_params = TVL1Parameters(
            tau=parameters.tau,
            lambda_=parameters.lambda_,
            theta=parameters.theta,
            nscales=parameters.nscales,
            warps=parameters.warps,
            epsilon=parameters.epsilon,
            inner_iterations=parameters.inner_iterations,
            outer_iterations=parameters.outer_iterations,
            scale_step=parameters.scale_step,
            median_filtering=parameters.median_filtering,
            downscale_factor=parameters.downscale_factor
        )

        self.analyzer = DisplacementAnalyzer(tvl1_params)
        self.params = parameters

    @staticmethod
    def validate_parameters(params: DisplacementParameters) -> Tuple[bool, str]:
        """
        Validate displacement analysis parameters.

        Parameters
        ----------
        params : DisplacementParameters
            Parameters to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        if params.tau <= 0:
            return False, "tau must be positive"

        if params.lambda_ <= 0:
            return False, "lambda must be positive"

        if not 0 < params.theta < 1:
            return False, "theta must be between 0 and 1"

        if params.nscales < 1:
            return False, "nscales must be at least 1"

        if params.warps < 1:
            return False, "warps must be at least 1"

        if params.epsilon <= 0:
            return False, "epsilon must be positive"

        if params.inner_iterations < 1:
            return False, "inner_iterations must be at least 1"

        if params.outer_iterations < 1:
            return False, "outer_iterations must be at least 1"

        if params.scale_step <= 0:
            return False, "scale_step must be positive"

        if params.median_filtering < 0:
            return False, "median_filtering must be non-negative"

        if params.downscale_factor < 1:
            return False, "downscale_factor must be at least 1"

        if params.pixel_size <= 0:
            return False, "pixel_size must be positive"

        if params.frame_interval <= 0:
            return False, "frame_interval must be positive"

        if params.d_max <= 0:
            return False, "d_max must be positive"

        if params.disp_vector_stride < 1:
            return False, "disp_vector_stride must be at least 1"

        if params.disp_arrow_scale <= 0:
            return False, "disp_arrow_scale must be positive"

        return True, ""

    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate input image data.

        Parameters
        ----------
        image : np.ndarray
            Image data to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
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
        """
        Calculate optical flow between reference and target image(s).
        Always returns displacement field with shape (t, y, x, 2) where t=1 for single frames.
        Yields intermediate results during calculation.

        Parameters
        ----------
        reference : np.ndarray
            Reference image (2D)
        target : np.ndarray
            Target image(s) - will be converted to 3D (t, y, x) if 2D

        Returns
        -------
        Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]
            Generator yielding (intermediate_displacement_field, frame_index, total_frames) during calculation
            and returning final DisplacementCalculationResult when exhausted

        Raises
        ------
        ValueError
            If input images are invalid
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