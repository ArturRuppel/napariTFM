from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingParameters:
    """Parameters for image preprocessing with relative intensity values"""
    # Contrast enhancement (relative intensity values between 0 and 1)
    min_intensity_percentile: float = 0.0
    max_intensity_percentile: float = 1.0

    # Optional gaussian filter
    enable_gaussian_filter: bool = False
    gaussian_sigma: float = 1.0

    # Registration parameters
    enable_registration: bool = False
    registration_mode: str = 'translation'  # 'translation' or 'rigid'
    reference_frame: int = 0

    def validate(self):
        """Validate parameter values"""
        if not 0 <= self.min_intensity_percentile < self.max_intensity_percentile <= 1:
            raise ValueError("Intensity percentiles must be between 0 and 1")
        if self.gaussian_sigma <= 0:
            raise ValueError("Gaussian sigma must be positive")
        if self.registration_mode not in ['translation', 'rigid']:
            raise ValueError("Invalid registration mode")
        if self.reference_frame < 0:
            raise ValueError("Reference frame must be non-negative")

class RegistrationResult:
    """Stores registration transformation matrices"""

    def __init__(self, num_frames: int):
        self.matrices = [np.eye(3) for _ in range(num_frames)]
        self.reference_image = None

    def apply_to_stack(self, stack: np.ndarray) -> np.ndarray:
        """Apply stored transformations to a stack"""
        if len(self.matrices) != len(stack):
            raise ValueError("Number of frames doesn't match transformation matrices")

        registered_stack = np.zeros_like(stack)
        for i, (frame, matrix) in enumerate(zip(stack, self.matrices)):
            if matrix is not None:
                registered_stack[i] = cv2.warpAffine(
                    frame,
                    matrix[:2, :],  # Use only first two rows for 2D transformation
                    (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
            else:
                registered_stack[i] = frame

        return registered_stack

class ImagePreprocessor:
    """Handles image preprocessing with intensity scaling and registration"""


    def __init__(self, parameters: Optional[PreprocessingParameters] = None):
        self.params = parameters or PreprocessingParameters()
        self.registration_result = None

    def preprocess_all(
            self,
            bead_stack: Optional[np.ndarray] = None,
            reference_image: Optional[np.ndarray] = None,
            cell_stack: Optional[np.ndarray] = None
    ) -> Dict[str, Tuple[np.ndarray, List[dict]]]:
        """
        Preprocess all image data maintaining proper registration dependencies.

        Args:
            bead_stack: 3D numpy array of bead images
            reference_image: 2D numpy array of reference image
            cell_stack: 3D numpy array of cell images

        Returns:
            Dictionary containing processed data and preprocessing info for each type
        """
        results = {}

        # First process reference image if available
        if reference_image is not None:
            processed_ref, ref_info = self.preprocess_frame(reference_image)
            results['reference'] = (processed_ref, [ref_info])

            # Store processed reference for registration
            self.registration_result = None
            reference_for_registration = processed_ref
        else:
            reference_for_registration = None

        # Process and register bead stack if available
        if bead_stack is not None:
            # First apply basic preprocessing to each frame
            processed_frames = []
            preprocessing_info = []

            for frame in bead_stack:
                proc_frame, frame_info = self.preprocess_frame(frame)
                processed_frames.append(proc_frame)
                preprocessing_info.append(frame_info)

            processed_stack = np.stack(processed_frames)

            # Register against reference image if available and registration is enabled
            if self.params.enable_registration and reference_for_registration is not None:
                registered_stack, reg_result = self.register_stack_to_reference(
                    processed_stack,
                    reference_for_registration
                )
                self.registration_result = reg_result
                results['beads'] = (registered_stack, preprocessing_info)
            else:
                results['beads'] = (processed_stack, preprocessing_info)

        # Process and register cell stack if available
        if cell_stack is not None:
            processed_frames = []
            preprocessing_info = []

            for frame in cell_stack:
                proc_frame, frame_info = self.preprocess_frame(frame)
                processed_frames.append(proc_frame)
                preprocessing_info.append(frame_info)

            processed_stack = np.stack(processed_frames)

            # Apply bead registration transforms if available
            if self.registration_result is not None:
                registered_stack = self.registration_result.apply_to_stack(processed_stack)
                results['cells'] = (registered_stack, preprocessing_info)
            else:
                results['cells'] = (processed_stack, preprocessing_info)

        return results

    def register_stack_to_reference(
            self,
            stack: np.ndarray,
            reference: np.ndarray
    ) -> Tuple[np.ndarray, RegistrationResult]:
        """
        Register all frames in a stack to a reference image.

        Args:
            stack: 3D numpy array (frames, height, width)
            reference: 2D numpy array of reference image

        Returns:
            Tuple of (registered stack, registration result)
        """
        if not self.params.enable_registration:
            return stack, None

        num_frames = len(stack)
        registration_result = RegistrationResult(num_frames)
        registration_result.reference_image = reference

        registered_stack = np.zeros_like(stack)

        # Define registration method
        if self.params.registration_mode == 'translation':
            warp_mode = cv2.MOTION_TRANSLATION
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:  # rigid
            warp_mode = cv2.MOTION_EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-7)

        for i in range(num_frames):
            try:
                # Find transformation
                _, matrix = cv2.findTransformECC(
                    reference.astype(np.float32),
                    stack[i].astype(np.float32),
                    warp_matrix,
                    warp_mode,
                    criteria,
                    None,
                    5
                )

                # Store transformation
                full_matrix = np.eye(3)
                full_matrix[:2, :] = matrix
                registration_result.matrices[i] = full_matrix

                # Apply transformation
                registered_stack[i] = cv2.warpAffine(
                    stack[i],
                    matrix,
                    (stack[i].shape[1], stack[i].shape[0]),
                    flags=cv2.INTER_LINEAR
                )

            except cv2.error as e:
                logger.warning(f"Registration failed for frame {i}: {e}")
                registered_stack[i] = stack[i]
                registration_result.matrices[i] = None

        return registered_stack, registration_result

    def preprocess_frame(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Preprocess a single image frame.

        Args:
            image: 2D numpy array

        Returns:
            Tuple of (preprocessed image, preprocessing info dictionary)
        """
        # Store original statistics
        info = {
            'original_dtype': image.dtype,
            'original_range': (float(image.min()), float(image.max())),
            'original_mean': float(image.mean()),
            'original_std': float(image.std())
        }

        processed = image.copy()

        # Calculate intensity limits based on percentiles
        min_val = np.percentile(processed, self.params.min_intensity_percentile * 100)
        max_val = np.percentile(processed, self.params.max_intensity_percentile * 100)

        # Apply intensity scaling
        processed = np.clip(processed, min_val, max_val)
        processed = (processed - min_val) / (max_val - min_val)

        # Apply gaussian filter if enabled
        if self.params.enable_gaussian_filter:
            processed = gaussian_filter(processed, self.params.gaussian_sigma)

        # Store final statistics
        info.update({
            'final_mean': float(processed.mean()),
            'final_std': float(processed.std()),
            'intensity_range': (float(min_val), float(max_val))
        })

        return processed, info

    def register_stack(self, stack: np.ndarray) -> Tuple[np.ndarray, RegistrationResult]:
        """
        Register all frames in a stack to a reference frame.

        Args:
            stack: 3D numpy array (frames, height, width)

        Returns:
            Tuple of (registered stack, registration result)
        """
        if not self.params.enable_registration:
            return stack, None

        num_frames = len(stack)
        reference_frame = stack[self.params.reference_frame]
        registration_result = RegistrationResult(num_frames)
        registration_result.reference_frame = self.params.reference_frame

        registered_stack = np.zeros_like(stack)
        registered_stack[self.params.reference_frame] = reference_frame

        # Define registration method
        if self.params.registration_mode == 'translation':
            warp_mode = cv2.MOTION_TRANSLATION
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:  # rigid
            warp_mode = cv2.MOTION_EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-7)

        for i in range(num_frames):
            if i == self.params.reference_frame:
                continue

            try:
                # Find transformation
                _, matrix = cv2.findTransformECC(
                    reference_frame.astype(np.float32),
                    stack[i].astype(np.float32),
                    warp_matrix,
                    warp_mode,
                    criteria,
                    None,
                    5
                )

                # Store transformation
                full_matrix = np.eye(3)
                full_matrix[:2, :] = matrix
                registration_result.matrices[i] = full_matrix

                # Apply transformation
                registered_stack[i] = cv2.warpAffine(
                    stack[i],
                    matrix,
                    (stack[i].shape[1], stack[i].shape[0]),
                    flags=cv2.INTER_LINEAR
                )

            except cv2.error as e:
                logger.warning(f"Registration failed for frame {i}: {e}")
                registered_stack[i] = stack[i]
                registration_result.matrices[i] = None

        return registered_stack, registration_result

    def update_parameters(self, new_params: PreprocessingParameters) -> None:
        """Update preprocessing parameters"""
        new_params.validate()
        self.params = new_params
        logger.debug(f"Updated preprocessing parameters: {new_params}")