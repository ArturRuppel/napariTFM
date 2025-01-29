import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Core image processing operations without business logic"""

    @staticmethod
    def apply_gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian filter if sigma is non-zero"""
        return gaussian_filter(image, sigma=sigma) if sigma > 0 else image.copy()

    @staticmethod
    def apply_intensity_scaling(image: np.ndarray, min_percentile: float, max_percentile: float) -> tuple[np.ndarray, tuple[float, float]]:
        """Apply intensity scaling based on percentiles"""
        min_val = np.percentile(image, min_percentile)
        max_val = np.percentile(image, max_percentile)
        processed = np.clip(image, min_val, max_val)
        processed = (processed - min_val) / (max_val - min_val)
        return processed, (min_val, max_val)

    @staticmethod
    def register_to_reference(moving_image: np.ndarray, reference_image: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Register moving image to reference image"""
        # Convert images to 8-bit for registration
        moving_norm = ((moving_image - moving_image.min()) * 255 /
                       (moving_image.max() - moving_image.min())).astype(np.uint8)
        ref_norm = ((reference_image - reference_image.min()) * 255 /
                    (reference_image.max() - reference_image.min())).astype(np.uint8)

        # Define registration method
        warp_mode = cv2.MOTION_TRANSLATION if mode == 'translation' else cv2.MOTION_EUCLIDEAN
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
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        registered = cv2.warpAffine(
            moving_image,
            warp_matrix,
            (moving_image.shape[1], moving_image.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        return registered, warp_matrix
