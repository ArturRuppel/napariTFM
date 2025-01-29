from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator
import numpy as np
from scipy.ndimage import gaussian_filter

@dataclass
class PreprocessingParameters:
    """Parameters for image preprocessing"""
    min_intensity_percentile: float = 0.0
    max_intensity_percentile: float = 100
    gaussian_sigma: float = 0.0
    cell_min_intensity_percentile: float = 0.0
    cell_max_intensity_percentile: float = 100
    cell_gaussian_sigma: float = 0.0
    registration_mode: str = 'translation'


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

    def update_parameters(self, parameters: PreprocessingParameters):
        """Update preprocessing parameters"""
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(error_msg)
        self.params = parameters

    @staticmethod
    def validate_parameters(params: PreprocessingParameters) -> Tuple[bool, str]:
        """
        Validate preprocessing parameters.

        Parameters
        ----------
        params : PreprocessingParameters
            Parameters to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
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

    def preprocess_frame(self, image: np.ndarray, is_cell: bool = False,
                         reference_image: Optional[np.ndarray] = None) -> PreprocessingIntermediateResult:
        """
        Preprocess a single image frame

        Parameters
        ----------
        image : np.ndarray
            Input image to process
        is_cell : bool
            Whether this is a cell image (uses cell-specific parameters)
        reference_image : np.ndarray, optional
            Reference image for registration

        Returns
        -------
        PreprocessingIntermediateResult
            Processed image and associated metadata
        """
        info = {
            'original_dtype': image.dtype,
            'original_range': (float(image.min()), float(image.max())),
            'original_mean': float(image.mean()),
            'original_std': float(image.std())
        }

        processed = image.copy()

        # Use appropriate parameters based on image type
        if is_cell:
            min_percentile = self.params.cell_min_intensity_percentile
            max_percentile = self.params.cell_max_intensity_percentile
            gaussian_sigma = self.params.cell_gaussian_sigma
        else:
            min_percentile = self.params.min_intensity_percentile
            max_percentile = self.params.max_intensity_percentile
            gaussian_sigma = self.params.gaussian_sigma

        # Apply gaussian filter if sigma is non-zero
        if gaussian_sigma > 0:
            processed = gaussian_filter(processed, sigma=gaussian_sigma)
            info['gaussian_sigma'] = gaussian_sigma
        else:
            info['gaussian_sigma'] = 0

        # Calculate intensity limits
        min_val = np.percentile(processed, min_percentile)
        max_val = np.percentile(processed, max_percentile)

        # Apply intensity scaling
        processed = np.clip(processed, min_val, max_val)
        processed = (processed - min_val) / (max_val - min_val)

        # Perform registration if reference image is provided
        transform_matrix = None
        if reference_image is not None and self.params.registration_mode != 'no registration':
            processed, transform_matrix = self.register_images(processed, reference_image)

        # Store final statistics
        info.update({
            'final_mean': float(processed.mean()),
            'final_std': float(processed.std()),
            'intensity_range': (float(min_val), float(max_val))
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
        """
        Process an image stack, yielding progress updates

        Parameters
        ----------
        image_stack : np.ndarray
            Stack of images to process
        reference_image : np.ndarray, optional
            Reference image for registration
        is_cell : bool
            Whether these are cell images

        Yields
        ------
        Tuple[PreprocessingResult, int, int]
            (result, current_frame, total_frames)

        Returns
        -------
        List[PreprocessingIntermediateResult]
            Complete list of preprocessing results
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

    def register_images(self, moving_image: np.ndarray, reference_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register a moving image to a reference image

        Parameters
        ----------
        moving_image : np.ndarray
            Image to be registered
        reference_image : np.ndarray
            Reference image

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (registered_image, transform_matrix)
        """
        import cv2

        # Convert images to 8-bit for registration
        moving_norm = ((moving_image - moving_image.min()) * 255 /
                       (moving_image.max() - moving_image.min())).astype(np.uint8)
        ref_norm = ((reference_image - reference_image.min()) * 255 /
                    (reference_image.max() - reference_image.min())).astype(np.uint8)

        # Define registration method
        warp_mode = cv2.MOTION_TRANSLATION if self.params.registration_mode == 'translation' else cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_norm,
                moving_norm,
                warp_matrix,
                warp_mode,
                criteria,
                inputMask=None,
                gaussFiltSize=1
            )
        except cv2.error:
            # Registration failed, use identity transform
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Apply transformation
        registered = cv2.warpAffine(
            moving_image,
            warp_matrix,
            (moving_image.shape[1], moving_image.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        return registered, warp_matrix
