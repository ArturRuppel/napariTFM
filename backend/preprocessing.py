import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Core image processing operations for microscopy image analysis.

    This class implements fundamental image processing operations commonly used
    in microscopy analysis, including background subtraction, filtering, intensity
    scaling, and image registration. Each operation is implemented as a stateless
    method, making the class suitable for both single-image and batch processing.

    The processor is particularly optimized for microscopy data as it:
    - Preserves original data types and ranges where appropriate
    - Handles both 8-bit and 16-bit images correctly
    - Implements robust background correction
    - Provides accurate image registration
    """

    @staticmethod
    def apply_rolling_ball(image: np.ndarray, radius: float) -> np.ndarray:
        """Apply rolling ball background subtraction to microscopy images.

        Implements rolling ball background subtraction using morphological
        operations. This method is particularly effective for removing uneven
        background illumination in microscopy images while preserving local
        intensity variations.

        Args:
            image (np.ndarray): Input image to process
            radius (float): Radius of the rolling ball in pixels
                If radius <= 0, returns a copy of the input image
                Larger radius values remove larger-scale background variations

        Returns:
            np.ndarray: Background-corrected image in the same dtype as input
                Negative values are clipped to 0

        Note:
            The implementation uses a combination of morphological opening and
            Gaussian filtering to approximate the rolling ball algorithm. The
            kernel size is automatically calculated from the radius.
        """
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
        """Apply Gaussian smoothing filter to reduce noise.

        Performs Gaussian filtering using scipy's implementation, which handles
        edge effects appropriately. If sigma is 0, returns a copy of the input
        image without filtering.

        Args:
            image (np.ndarray): Input image to filter
            sigma (float): Standard deviation of Gaussian kernel
                If sigma <= 0, returns a copy of the input image
                Larger values produce stronger smoothing

        Returns:
            np.ndarray: Filtered image in same dtype as input

        Note:
            The filter size is automatically determined by scipy based on sigma.
            The implementation preserves the image edges by appropriate padding.
        """
        return gaussian_filter(image, sigma=sigma) if sigma > 0 else image.copy()

    @staticmethod
    def apply_intensity_scaling(image: np.ndarray, min_percentile: float, max_percentile: float) -> tuple[np.ndarray, tuple[float, float]]:
        """Scale image intensities using percentile-based normalization.

        Normalizes image intensities to [0, 1] range based on specified
        percentiles. This is particularly useful for standardizing microscopy
        images with varying intensity ranges or outliers.

        Args:
            image (np.ndarray): Input image to normalize
            min_percentile (float): Lower percentile for scaling (0-100)
            max_percentile (float): Upper percentile for scaling (0-100)
                Values should satisfy: 0 <= min_percentile < max_percentile <= 100

        Returns:
            tuple[np.ndarray, tuple[float, float]]: Tuple containing:
                - Normalized image with values in [0, 1]
                - Tuple of (min_val, max_val) used for scaling

        Note:
            Values below min_percentile or above max_percentile are clipped.
            This is useful for removing intensity outliers while preserving
            the relative intensities of the majority of pixels.
        """
        min_val = np.percentile(image, min_percentile)
        max_val = np.percentile(image, max_percentile)
        processed = np.clip(image, min_val, max_val)
        processed = (processed - min_val) / (max_val - min_val)
        return processed, (min_val, max_val)

    @staticmethod
    def register_to_reference(moving_image: np.ndarray, reference_image: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Register a moving image to a reference image using OpenCV.

        Performs image registration using Enhanced Correlation Coefficient (ECC)
        maximization. Supports both translation-only and rigid (translation +
        rotation) registration modes.

        Args:
            moving_image (np.ndarray): Image to be registered
            reference_image (np.ndarray): Reference image to register against
            mode (str): Registration mode, one of:
                - 'translation': Translation only (x, y shifts)
                - 'rigid': Translation and rotation
                Both images should be 2D arrays of the same shape

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Registered image in same dtype as input
                - 2x3 transformation matrix

        Note:
            The implementation automatically handles intensity normalization
            for registration. If registration fails, returns the original
            image and an identity transformation matrix.
        """
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
