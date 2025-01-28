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
    flow: np.ndarray  # Shape (t, y, x, 2) for time series, units in µm
    original_shape: tuple  # Original image shape (y, x)
    flow_shape: tuple  # Flow field shape (y, x)
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
        """
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

    def calculate_flow(
            self,
            reference: np.ndarray,
            target: np.ndarray,
            yield_intermediates: bool = False
    ) -> Union[DisplacementResult, Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]]:
        """
        Calculate optical flow between reference and target image(s).
        Always returns flow with shape (t, y, x, 2) where t=1 for single frames.

        Parameters
        ----------
        reference : np.ndarray
            Reference image (2D)
        target : np.ndarray
            Target image(s) - will be converted to 3D (t, y, x) if 2D
        yield_intermediates : bool
            If True, yields (flow, current_frame, total_frames) tuples during calculation

        Returns
        -------
        Union[DisplacementResult, Generator]
            If yield_intermediates is False:
                DisplacementCalculationResult containing final flow field
            If yield_intermediates is True:
                Generator yielding (intermediate_flow, frame_index, total_frames) during calculation
                and returning final DisplacementCalculationResult when exhausted
        """
        # Ensure target is 3D
        if target.ndim == 2:
            target = target[np.newaxis, ...]

        total_frames = target.shape[0]

        # Calculate output shape based on downscaling
        if self.params.downscale_factor > 1:
            flow_shape = (
                total_frames,
                target.shape[1] // self.params.downscale_factor,
                target.shape[2] // self.params.downscale_factor,
                2
            )
        else:
            flow_shape = (total_frames, target.shape[1], target.shape[2], 2)

        flow_stack = np.zeros(flow_shape, dtype=np.float32)

        def calculate_with_intermediates():
            # Calculate flow for each frame
            for frame in range(total_frames):
                # Calculate flow in pixels
                flow_pixels = self.analyzer.calculate_flow(reference, target[frame])

                # Apply downscaling if needed
                if self.params.downscale_factor > 1:
                    flow_pixels = self.analyzer.downscale_flow(flow_pixels, self.params.downscale_factor)

                # Convert to physical units (µm)
                flow_stack[frame] = flow_pixels * self.params.pixel_size

                # Yield intermediate result with progress info
                yield flow_stack[frame].copy(), frame + 1, total_frames

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
                flow=flow_stack,
                original_shape=reference.shape,
                flow_shape=flow_stack.shape[1:3],
                parameters=self.params,
                physical_scale=physical_scale
            )

        if yield_intermediates:
            return calculate_with_intermediates()
        else:
            # Calculate without yielding intermediates
            for frame in range(total_frames):
                flow_pixels = self.analyzer.calculate_flow(reference, target[frame])
                if self.params.downscale_factor > 1:
                    flow_pixels = self.analyzer.downscale_flow(flow_pixels, self.params.downscale_factor)
                flow_stack[frame] = flow_pixels * self.params.pixel_size

            physical_scale = {
                'pixel_size': self.params.pixel_size,
                'grid_spacing': self.params.pixel_size * self.params.downscale_factor,
                'time_interval': self.params.frame_interval,
                'displacement_units': 'µm',
                'grid_spacing_units': 'µm',
                'time_interval_units': 'min',
            }

            return DisplacementResult(
                flow=flow_stack,
                original_shape=reference.shape,
                flow_shape=flow_stack.shape[1:3],
                parameters=self.params,
                physical_scale=physical_scale
            )