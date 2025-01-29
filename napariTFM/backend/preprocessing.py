import logging
from typing import Tuple, Optional, List, Dict, Generator

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from napariTFM.services.preprocessing_service import PreprocessingParameters

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing operations."""
    def __init__(self, parameters: Optional[PreprocessingParameters] = None):
        """Initialize with optional parameters."""
        self.params = parameters or PreprocessingParameters()
        self.registration_result = None

    def preprocess_frame(self, image: np.ndarray, is_cell: bool = False,
                         reference_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Preprocess a single image frame.
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

        # Calculate intensity limits based on percentiles
        min_val = np.percentile(processed, min_percentile * 100)
        max_val = np.percentile(processed, max_percentile * 100)

        # Apply intensity scaling
        processed = np.clip(processed, min_val, max_val)
        processed = (processed - min_val) / (max_val - min_val)

        # Apply gaussian filter if enabled
        if use_gaussian:
            processed = gaussian_filter(processed, gaussian_sigma)

        # Store final statistics
        info.update({
            'final_mean': float(processed.mean()),
            'final_std': float(processed.std()),
            'intensity_range': (float(min_val), float(max_val))
        })

        return processed, info

    def register_images(self, moving_image: np.ndarray, reference_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register a moving image to a reference image.
        Returns the registered image and the transformation matrix.
        """
        # Return early if registration is disabled
        if self.params.registration_mode == 'no registration':
            return moving_image, np.eye(2, 3, dtype=np.float32)

        # Convert images to 8-bit for registration
        moving_norm = ((moving_image - moving_image.min()) * 255 /
                       (moving_image.max() - moving_image.min())).astype(np.uint8)
        ref_norm = ((reference_image - reference_image.min()) * 255 /
                    (reference_image.max() - reference_image.min())).astype(np.uint8)

        # Define registration method based on mode
        warp_mode = cv2.MOTION_TRANSLATION if self.params.registration_mode == 'translation' else cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)

        # Run registration with error handling
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
        except cv2.error as e:
            logger.warning(f"Registration failed: {str(e)}. Using identity transform.")
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Apply transformation to original image with inverse map flag
        registered = cv2.warpAffine(
            moving_image,
            warp_matrix,
            (moving_image.shape[1], moving_image.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        return registered, warp_matrix


    def update_parameters(self, parameters: PreprocessingParameters):
        """Update preprocessing parameters."""
        parameters.validate()  # Validate parameters before updating
        self.params = parameters

