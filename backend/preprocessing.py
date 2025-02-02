import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)
# TODO implement rolling ball background substraction

class ImageProcessor:
    """Core image processing operations without business logic"""

    @staticmethod
    def apply_rolling_ball(image: np.ndarray, radius: float) -> np.ndarray:
        """Apply rolling ball background subtraction if radius is non-zero"""
        if radius <= 0:
            return image.copy()

        # Create properly sized kernel for rolling ball
        kernel_size = int(2 * radius + 1)  # Ensure odd size
        if kernel_size < 3:  # Minimum size of 3x3
            kernel_size = 3

        # Create structuring element
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Store original dtype for later
        orig_dtype = image.dtype

        # Estimate background using morphological opening
        # OpenCV expects uint8/uint16 or float32, so we keep uint16/uint8 as is
        # but convert other types to float32
        if orig_dtype not in (np.uint8, np.uint16):
            image = image.astype(np.float32)

        bg = cv2.morphologyEx(image, cv2.MORPH_OPEN, se, iterations=1)
        bg = cv2.GaussianBlur(bg, (kernel_size, kernel_size), 0)

        # Subtract background - convert to float32 for subtraction to avoid underflow
        corrected = image.astype(np.float32) - bg.astype(np.float32)

        # Clip negative values
        corrected = np.clip(corrected, 0, None)

        # Convert back to original dtype
        if orig_dtype in (np.uint8, np.uint16):
            corrected = np.clip(corrected, 0, np.iinfo(orig_dtype).max).astype(orig_dtype)

        return corrected

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
