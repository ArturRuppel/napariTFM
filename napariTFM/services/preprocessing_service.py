from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator

import numpy as np

from napariTFM.backend.parameter_dataclasses import PreprocessingParameters
from napariTFM.backend.parameter_validation import validate_preprocessing_parameters
from napariTFM.backend.preprocessing import ImageProcessor


@dataclass
class PreprocessingIntermediateResult:
    """Results from preprocessing operations on microscopy images.

    Attributes:
        processed_image (np.ndarray): Preprocessed image data
        transform_matrix (np.ndarray, optional): Registration transformation matrix
            2x3 matrix describing the spatial transform if registration was performed
        info (Dict[str, Any]): Processing metadata including:
            - original_dtype: Original data type
            - original_range: (min, max) of original data
            - original_mean: Mean of original data
            - original_std: Standard deviation of original data
            - final_mean: Mean after processing
            - final_std: Standard deviation after processing
            - intensity_range: (min, max) used for scaling
            - gaussian_sigma: Applied Gaussian smoothing sigma
            - rolling_ball_radius: Applied background correction radius
    """
    processed_image: np.ndarray
    transform_matrix: Optional[np.ndarray] = None
    info: Dict[str, Any] = None


class PreprocessingService:
    """Service layer for microscopy image preprocessing operations.

    This class provides a high-level interface for preprocessing microscopy
    images, handling parameter validation, processing workflows, and result
    tracking. It supports both single-frame and time series processing,
    with separate parameters for cell and bead/reference images.

    The service implements a complete preprocessing pipeline including:
    - Background correction using rolling ball algorithm
    - Gaussian smoothing for noise reduction
    - Intensity normalization using percentile-based scaling
    - Image registration to a reference frame
    """


    def __init__(self, params: PreprocessingParameters):
        """Initialize preprocessing service with analysis parameters.

        Args:
            params (PreprocessingParameters): Configuration including:
                - Intensity scaling parameters (percentiles)
                - Gaussian smoothing parameters (sigma)
                - Rolling ball background correction radius
                - Registration mode and parameters
                Separate parameters are maintained for cell vs. bead images

        Raises:
            ValueError: If any parameters are invalid

        Example:
            >>> params = PreprocessingParameters(
            ...     min_intensity_percentile=1,
            ...     max_intensity_percentile=99,
            ...     gaussian_sigma=1.0,
            ...     rolling_ball_radius=50
            ... )
            >>> service = PreprocessingService(params)
        """
        is_valid, error_msg = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(error_msg)
        self.params = params
        self.transform_matrices = []
        self._processor = ImageProcessor()

    def update_parameters(self, parameters: PreprocessingParameters):
        """Update preprocessing parameters.

        Creates a new processor instance with the updated parameters after
        validating them.

        Args:
            parameters (PreprocessingParameters): New parameters to use

        Raises:
            ValueError: If any parameters are invalid
        """
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)
        self.params = parameters

    @staticmethod
    def validate_parameters(params: PreprocessingParameters) -> Tuple[bool, str]:
        """Validate preprocessing parameters.

        Checks all parameters for physical and numerical validity including:
        - Intensity scaling ranges
        - Filter parameters
        - Registration settings

        Args:
            params (PreprocessingParameters): Parameters to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
                error_message is empty string if valid

        Note:
            Specific validation rules include:
            - Percentiles must be in valid ranges
            - Sigmas must be non-negative
            - Registration mode must be supported
        """
        return validate_preprocessing_parameters(params)

    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """Validate input image data for preprocessing.

        Checks that image data is suitable for preprocessing:
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

    def preprocess_frame(self, image: np.ndarray, is_cell: bool = False,
                         reference_image: Optional[np.ndarray] = None) -> PreprocessingIntermediateResult:
        """Preprocess a single microscopy image frame.

        Applies the complete preprocessing pipeline to a single frame, with
        different parameter sets for cell vs. bead/reference images.

        Args:
            image (np.ndarray): Input image to process
            is_cell (bool): Whether the image contains cells (affects parameters)
            reference_image (np.ndarray, optional): Reference for registration

        Returns:
            PreprocessingIntermediateResult: Complete processing results including:
                - Processed image
                - Registration transform (if applicable)
                - Processing metadata

        Note:
            The processing pipeline includes:
            1. Rolling ball background correction (except for cell images)
            2. Gaussian smoothing
            3. Intensity scaling
            4. Registration (if reference provided)
        """
        info = {
            'original_dtype': image.dtype,
            'original_range': (float(image.min()), float(image.max())),
            'original_mean': float(image.mean()),
            'original_std': float(image.std())
        }

        # Select appropriate parameters based on image type
        if is_cell:
            params = (self.params.cell_min_intensity_percentile,
                      self.params.cell_max_intensity_percentile,
                      self.params.cell_gaussian_sigma)
            processed = image  # No rolling ball for cell images
        else:
            # Apply rolling ball only for bead/reference images
            params = (self.params.min_intensity_percentile,
                      self.params.max_intensity_percentile,
                      self.params.gaussian_sigma)
            processed = self._processor.apply_rolling_ball(image, self.params.rolling_ball_radius)

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
            'rolling_ball_radius': self.params.rolling_ball_radius if not is_cell else None
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
        """Process an image stack with progress tracking.

        Applies preprocessing pipeline to a stack of images (time series),
        yielding intermediate results for progress monitoring. Supports both
        cell and bead/reference image processing with appropriate parameters.

        Args:
            image_stack (np.ndarray, optional): Stack of images to process
                Shape should be (t, y, x) for time series or (y, x) for single frame
            reference_image (np.ndarray, optional): Reference for registration
            is_cell (bool): Whether images contain cells (affects parameters)

        Yields:
            Tuple[PreprocessingIntermediateResult, int, int]: Tuple containing:
                - Current frame's preprocessing results
                - Frame index (1-based)
                - Total number of frames
            Yielded after each frame is processed

        Returns:
            List[PreprocessingIntermediateResult]: Complete results for all frames
                Accessible via StopIteration.value when generator completes

        Example:
            >>> # Get the generator
            >>> prep_generator = service.preprocess_stack(image_stack)
            >>>
            >>> # Process intermediate results
            >>> try:
            ...     while True:
            ...         # Get next frame result
            ...         result, frame, total = next(prep_generator)
            ...         print(f"Processed frame {frame}/{total}")
            ... except StopIteration as e:
            ...     # Get final results from generator's return value
            ...     final_results = e.value
        """
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
