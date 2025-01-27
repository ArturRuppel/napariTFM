from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator
import numpy as np
from scipy.ndimage import gaussian_filter

# TODO service has basically the same class, should just keep one
@dataclass
class PreprocessingParameters:
    """Parameters for image preprocessing"""
    min_intensity_percentile: float = 0.0
    max_intensity_percentile: float = 1.0
    enable_gaussian_filter: bool = False
    gaussian_sigma: float = 0.0
    cell_min_intensity_percentile: float = 0.0
    cell_max_intensity_percentile: float = 1.0
    enable_cell_gaussian_filter: bool = False
    cell_gaussian_sigma: float = 0.0
    registration_mode: str = 'translation'

    def validate(self):
        """Validate parameter values"""
        if not 0 <= self.min_intensity_percentile < self.max_intensity_percentile <= 1:
            raise ValueError("Invalid intensity percentile range")

        if not 0 <= self.cell_min_intensity_percentile < self.cell_max_intensity_percentile <= 1:
            raise ValueError("Invalid cell intensity percentile range")

        if self.enable_gaussian_filter and self.gaussian_sigma < 0:
            raise ValueError("Gaussian sigma must be non-negative")

        if self.enable_cell_gaussian_filter and self.cell_gaussian_sigma < 0:
            raise ValueError("Cell gaussian sigma must be non-negative")

        if self.registration_mode not in ['translation', 'rigid', 'no registration']:
            raise ValueError(f"Invalid registration mode: {self.registration_mode}")


@dataclass
class PreprocessingResult:
    """Results from preprocessing operations"""
    processed_image: np.ndarray
    transform_matrix: Optional[np.ndarray] = None
    info: Dict[str, Any] = None


class PreprocessingService:
    """Service layer for image preprocessing operations"""

    def __init__(self, parameters: Optional[PreprocessingParameters] = None):
        """Initialize with optional parameters"""
        self.params = parameters or PreprocessingParameters()
        self.transform_matrices = []

    def update_parameters(self, parameters: PreprocessingParameters):
        """Update preprocessing parameters"""
        parameters.validate()
        self.params = parameters

    def preprocess_frame(self, image: np.ndarray, is_cell: bool = False,
                         reference_image: Optional[np.ndarray] = None) -> PreprocessingResult:
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
        PreprocessingResult
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
            use_gaussian = self.params.enable_cell_gaussian_filter
            gaussian_sigma = self.params.cell_gaussian_sigma
        else:
            min_percentile = self.params.min_intensity_percentile
            max_percentile = self.params.max_intensity_percentile
            use_gaussian = self.params.enable_gaussian_filter
            gaussian_sigma = self.params.gaussian_sigma

        # Calculate intensity limits
        min_val = np.percentile(processed, min_percentile * 100)
        max_val = np.percentile(processed, max_percentile * 100)

        # Apply intensity scaling
        processed = np.clip(processed, min_val, max_val)
        processed = (processed - min_val) / (max_val - min_val)

        # Apply gaussian filter if enabled
        if use_gaussian:
            processed = gaussian_filter(processed, gaussian_sigma)

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

        return PreprocessingResult(
            processed_image=processed,
            transform_matrix=transform_matrix,
            info=info
        )

    def preprocess_stack(
            self,
            image_stack: Optional[np.ndarray] = None,
            reference_image: Optional[np.ndarray] = None,
            is_cell: bool = False
    ) -> Generator[Tuple[PreprocessingResult, int, int], None, List[PreprocessingResult]]:
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
        List[PreprocessingResult]
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