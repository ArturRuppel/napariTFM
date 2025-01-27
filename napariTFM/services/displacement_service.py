from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator
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
    vector_stride: int
    arrow_scale: float


@dataclass
class DisplacementCalculationResult:
    """Results from displacement field calculation"""
    flow: np.ndarray
    original_shape: tuple
    flow_shape: tuple
    parameters: DisplacementParameters


class DisplacementService:
    """Service layer handling business logic for displacement analysis."""

    def __init__(self):
        self.analyzer = None

    def _ensure_analyzer(self, params: DisplacementParameters):
        """Ensure analyzer exists with current parameters."""
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

    def process_image_data(self, image_data: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Process image data with validation and normalization.

        Args:
            image_data: Input image data array

        Returns:
            Tuple of (processed_image_data, warning_messages)
        """
        warnings = []

        if image_data is None:
            raise ValueError("No image data provided")

        # Convert to float and normalize if needed
        if image_data.dtype != np.float32:
            image_data = image_data.astype(np.float32)
            warnings.append("Image data converted to float32")

        # If single image, add time dimension
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
            warnings.append("Single image expanded to 3D array")

        # Ensure we have a 3D array (time, height, width)
        if image_data.ndim != 3:
            raise ValueError(f"Image data must be 2D or 3D, got shape {image_data.shape}")

        return image_data, warnings

    def calculate_flow(
            self,
            reference: np.ndarray,
            moving: np.ndarray,
            params: DisplacementParameters
    ) -> DisplacementCalculationResult:
        """Calculate optical flow between reference and moving image."""
        self._ensure_analyzer(params)

        # Calculate flow in pixels
        flow_pixels = self.analyzer.calculate_flow(reference, moving)

        # Apply downscaling if needed
        if params.downscale_factor > 1:
            flow_pixels = self.analyzer.downscale_flow(flow_pixels, params.downscale_factor)

        # Convert to physical units
        flow = flow_pixels * params.pixel_size

        return DisplacementCalculationResult(
            flow=flow,
            original_shape=reference.shape,
            flow_shape=flow.shape[:2],
            parameters=params
        )

    def calculate_flow_stack(
            self,
            reference: np.ndarray,
            bead_stack: np.ndarray,
            params: DisplacementParameters
    ) -> Generator[Tuple[DisplacementCalculationResult, int, int], None, List[DisplacementCalculationResult]]:
        """
        Calculate flow for all frames as a generator that yields intermediate results.

        Parameters
        ----------
        reference : np.ndarray
            Reference image
        bead_stack : np.ndarray
            Stack of bead images
        params : DisplacementParameters
            Analysis parameters

        Yields
        ------
        Tuple[DisplacementCalculationResult, int, int]
            (result, current_frame, total_frames)
            Yields each frame's flow calculation along with progress information

        Returns
        -------
        List[DisplacementCalculationResult]
            Complete list of results for all frames
        """
        self._ensure_analyzer(params)

        # Handle 2D input
        if bead_stack.ndim == 2:
            bead_stack = bead_stack[np.newaxis, ...]

        total_frames = bead_stack.shape[0]
        all_results = []

        for frame in range(total_frames):
            # Calculate flow for current frame
            result = self.calculate_flow(reference, bead_stack[frame], params)
            all_results.append(result)

            # Yield intermediate results
            yield result, frame, total_frames

        return all_results

    def validate_parameters(self, params: DisplacementParameters) -> Tuple[bool, str]:
        """Validate displacement analysis parameters."""
        if params.tau <= 0:
            return False, "Tau must be positive"

        if params.lambda_ <= 0:
            return False, "Lambda must be positive"

        if not 0 < params.theta <= 1:
            return False, "Theta must be between 0 and 1"

        if params.nscales < 1:
            return False, "Number of scales must be at least 1"

        if params.warps < 1:
            return False, "Number of warps must be at least 1"

        if params.epsilon <= 0:
            return False, "Epsilon must be positive"

        if params.inner_iterations < 1:
            return False, "Inner iterations must be at least 1"

        if params.outer_iterations < 1:
            return False, "Outer iterations must be at least 1"

        if not 0 < params.scale_step < 1:
            return False, "Scale step must be between 0 and 1"

        if params.median_filtering < 1 or params.median_filtering % 2 == 0:
            return False, "Median filtering size must be odd and at least 1"

        if params.downscale_factor < 1:
            return False, "Downscale factor must be at least 1"

        if params.pixel_size <= 0:
            return False, "Pixel size must be positive"

        if params.frame_interval <= 0:
            return False, "Frame interval must be positive"

        if params.d_max <= 0:
            return False, "Maximum displacement must be positive"

        if params.vector_stride < 1:
            return False, "Vector stride must be at least 1"

        if params.arrow_scale <= 0:
            return False, "Arrow scale must be positive"

        return True, ""

    def apply_flow_to_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Apply flow field to deform an image."""
        if self.analyzer is None:
            self.analyzer = DisplacementAnalyzer()
        return self.analyzer.apply_flow(image, flow)