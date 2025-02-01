from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator

import numpy as np

from napariTFM.backend.parameter_dataclasses import PreprocessingParameters
from napariTFM.backend.preprocessing import ImageProcessor


@dataclass
class PreprocessingIntermediateResult:
    """Results from preprocessing operations"""
    processed_image: np.ndarray
    transform_matrix: Optional[np.ndarray] = None
    info: Dict[str, Any] = None


class PreprocessingService:
    """Service layer for image preprocessing operations"""

    def __init__(self, params: PreprocessingParameters):
        """Initialize with required parameters"""
        is_valid, error_msg = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(error_msg)
        self.params = params
        self.transform_matrices = []
        self._processor = ImageProcessor()

    def update_parameters(self, parameters: PreprocessingParameters):
        """Update preprocessing parameters"""
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)
        self.params = parameters

    @staticmethod
    def validate_parameters(params: PreprocessingParameters) -> Tuple[bool, str]:
        """Validate preprocessing parameters."""
        if not 0 <= params.min_intensity_percentile < params.max_intensity_percentile <= 100:
            return False, "Invalid intensity percentile range"

        if not 0 <= params.cell_min_intensity_percentile < params.cell_max_intensity_percentile <= 100:
            return False, "Invalid cell intensity percentile range"

        if params.gaussian_sigma < 0:
            return False, "Gaussian sigma must be non-negative"

        if params.cell_gaussian_sigma < 0:
            return False, "Cell gaussian sigma must be non-negative"

        if params.registration_mode not in ['translation', 'rigid', 'no registration']:
            return False, f"Invalid registration mode: {params.registration_mode}"

        return True, ""

    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """Validate input image data."""
        if image is None:
            return False, "No image data provided"

        if not isinstance(image, np.ndarray):
            return False, "Image must be a numpy array"

        if image.ndim not in (2, 3):
            return False, "Image must be 2D or 3D (time series)"

        if np.all(np.isnan(image)):
            return False, "Image contains only NaN values"

        return True, ""

    def preprocess_frame(self, image: np.ndarray, is_cell: bool = False,
                         reference_image: Optional[np.ndarray] = None) -> PreprocessingIntermediateResult:
        """Preprocess a single image frame"""
        info = {
            'original_dtype': image.dtype,
            'original_range': (float(image.min()), float(image.max())),
            'original_mean': float(image.mean()),
            'original_std': float(image.std())
        }

        # Apply rolling ball background subtraction first
        processed = self._processor.apply_rolling_ball(image, self.params.rolling_ball_radius)

        # Select appropriate parameters
        if is_cell:
            params = (self.params.cell_min_intensity_percentile,
                      self.params.cell_max_intensity_percentile,
                      self.params.cell_gaussian_sigma)
        else:
            params = (self.params.min_intensity_percentile,
                      self.params.max_intensity_percentile,
                      self.params.gaussian_sigma)

        # Apply remaining processing steps
        processed = self._processor.apply_gaussian_filter(processed, params[2])
        processed, intensity_range = self._processor.apply_intensity_scaling(processed, params[0], params[1])

        # Handle registration if needed
        transform_matrix = None
        if reference_image is not None and self.params.registration_mode != 'no registration':
            processed, transform_matrix = self._processor.register_to_reference(
                processed, reference_image, self.params.registration_mode
            )

        # Update info
        info.update({
            'final_mean': float(processed.mean()),
            'final_std': float(processed.std()),
            'intensity_range': intensity_range,
            'gaussian_sigma': params[2],
            'rolling_ball_radius': self.params.rolling_ball_radius
        })

        return PreprocessingIntermediateResult(
            processed_image=processed,
            transform_matrix=transform_matrix,
            info=info
        )

    def preprocess_stack(
            self,
            image_stack: Optional[np.ndarray] = None,
            reference_image: Optional[np.ndarray] = None,
            is_cell: bool = False
    ) -> Generator[Tuple[PreprocessingIntermediateResult, int, int], None, List[PreprocessingIntermediateResult]]:
        """Process an image stack, yielding progress updates"""
        if image_stack is None:
            return []

        # Handle 2D input
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]

        results = []
        total_frames = image_stack.shape[0]

        # Process reference image first if provided
        processed_ref = None
        if reference_image is not None:
            ref_result = self.preprocess_frame(reference_image)
            processed_ref = ref_result.processed_image

        # Process each frame
        for frame in range(total_frames):
            result = self.preprocess_frame(
                image_stack[frame],
                is_cell=is_cell,
                reference_image=processed_ref
            )

            if result.transform_matrix is not None:
                self.transform_matrices.append(result.transform_matrix)

            results.append(result)
            yield result, frame, total_frames

        return results
