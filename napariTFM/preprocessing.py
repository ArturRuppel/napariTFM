import logging
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class PreprocessingParameters:
    """Parameters for image preprocessing"""

    def __init__(
            self,
            min_intensity_percentile: float = 0.0,
            max_intensity_percentile: float = 1.0,
            enable_gaussian_filter: bool = False,
            gaussian_sigma: float = 0.0,
            cell_min_intensity_percentile: float = 0.0,
            cell_max_intensity_percentile: float = 1.0,
            enable_cell_gaussian_filter: bool = False,
            cell_gaussian_sigma: float = 0.0,
            enable_registration: bool = True,
            registration_mode: str = 'translation'
    ):
        self.min_intensity_percentile = min_intensity_percentile
        self.max_intensity_percentile = max_intensity_percentile
        self.enable_gaussian_filter = enable_gaussian_filter
        self.gaussian_sigma = gaussian_sigma
        self.cell_min_intensity_percentile = cell_min_intensity_percentile
        self.cell_max_intensity_percentile = cell_max_intensity_percentile
        self.enable_cell_gaussian_filter = enable_cell_gaussian_filter
        self.cell_gaussian_sigma = cell_gaussian_sigma
        self.enable_registration = enable_registration
        self.registration_mode = registration_mode.lower()

    def validate(self):
        """Validate parameter values"""
        # Validate intensity percentiles
        if not 0 <= self.min_intensity_percentile < self.max_intensity_percentile <= 1:
            raise ValueError("Invalid intensity percentile range")

        if not 0 <= self.cell_min_intensity_percentile < self.cell_max_intensity_percentile <= 1:
            raise ValueError("Invalid cell intensity percentile range")

        # Validate gaussian sigma values - allow 0 for "no filter"
        if self.enable_gaussian_filter and self.gaussian_sigma < 0:
            raise ValueError("Gaussian sigma must be non-negative")

        if self.enable_cell_gaussian_filter and self.cell_gaussian_sigma < 0:
            raise ValueError("Cell gaussian sigma must be non-negative")

        # Validate registration mode
        if self.registration_mode not in ['translation', 'rigid', 'no registration']:
            raise ValueError(f"Invalid registration mode: {self.registration_mode}")

    def __str__(self):
        """String representation of parameters"""
        return (
            f"PreprocessingParameters(\n"
            f"  min_intensity_percentile={self.min_intensity_percentile},\n"
            f"  max_intensity_percentile={self.max_intensity_percentile},\n"
            f"  enable_gaussian_filter={self.enable_gaussian_filter},\n"
            f"  gaussian_sigma={self.gaussian_sigma},\n"
            f"  cell_min_intensity_percentile={self.cell_min_intensity_percentile},\n"
            f"  cell_max_intensity_percentile={self.cell_max_intensity_percentile},\n"
            f"  enable_cell_gaussian_filter={self.enable_cell_gaussian_filter},\n"
            f"  cell_gaussian_sigma={self.cell_gaussian_sigma},\n"
            f"  enable_registration={self.enable_registration},\n"
            f"  registration_mode={self.registration_mode}\n"
            f")"
        )

# class RegistrationResult:
#     """Stores registration transformation matrices"""
#
#     def __init__(self, num_frames: int):
#         self.matrices = [np.eye(3) for _ in range(num_frames)]
#         self.reference_image = None


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    def __init__(self, parameters: Optional[PreprocessingParameters] = None):
        """Initialize with optional parameters."""
        self.params = parameters or PreprocessingParameters()
        self.registration_result = None

    def register_images(self, moving_image: np.ndarray, reference_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register a moving image to a reference image.
        Returns the registered image and the transformation matrix.
        """
        # Convert images to 8-bit for registration
        moving_norm = ((moving_image - moving_image.min()) * 255 /
                       (moving_image.max() - moving_image.min())).astype(np.uint8)
        ref_norm = ((reference_image - reference_image.min()) * 255 /
                    (reference_image.max() - reference_image.min())).astype(np.uint8)

        # Define registration method based on mode
        if self.params.registration_mode == 'translation':
            warp_mode = cv2.MOTION_TRANSLATION
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:  # rigid
            warp_mode = cv2.MOTION_EUCLIDEAN
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

    def preprocess_frame(self, image: np.ndarray, is_cell: bool = False,
                         apply_registration: bool = True,
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

    def preprocess_all(
            self,
            bead_stack: Optional[np.ndarray] = None,
            reference_image: Optional[np.ndarray] = None,
            cell_stack: Optional[np.ndarray] = None,
            progress_callback: Optional[callable] = None
    ) -> Dict[str, Tuple[np.ndarray, List[Dict]]]:
        """Preprocess all available data."""
        results = {}
        self.transform_matrices = []

        # Calculate total steps for progress
        total_steps = 0
        current_step = 0
        if reference_image is not None:
            total_steps += 1
        if bead_stack is not None:
            total_steps += len(bead_stack)
        if cell_stack is not None:
            total_steps += len(cell_stack)

        # Process reference image first if provided
        processed_ref = None
        if reference_image is not None:
            if progress_callback:
                progress_callback(current_step / total_steps * 100, "Processing reference image...")
            processed_ref, ref_info = self.preprocess_frame(reference_image)
            results['reference'] = (processed_ref, [ref_info])
            current_step += 1

        # Process bead stack if provided
        if bead_stack is not None:
            processed_stack = np.zeros_like(bead_stack, dtype=float)
            info_list = []
            self.transform_matrices = []

            for i in range(bead_stack.shape[0]):
                print(f'Processing frame {i + 1}/{bead_stack.shape[0]}')
                if progress_callback:
                    progress_callback(
                        current_step / total_steps * 100,
                        f"Processing bead frame {i + 1}/{bead_stack.shape[0]}..."
                    )

                # Get preprocessed frame without registration first
                frame, frame_info = self.preprocess_frame(
                    bead_stack[i],
                    apply_registration=False
                )

                # Perform registration if enabled and reference is available
                if self.params.enable_registration and processed_ref is not None:
                    registered_frame, transform_matrix = self.register_images(frame, processed_ref)
                    self.transform_matrices.append(transform_matrix)
                    processed_stack[i] = registered_frame
                else:
                    processed_stack[i] = frame
                    if self.params.enable_registration:
                        self.transform_matrices.append(np.eye(2, 3, dtype=np.float32))

                info_list.append(frame_info)
                current_step += 1

            results['beads'] = (processed_stack, info_list)

        # Process cell stack if provided
        if cell_stack is not None:
            processed_cells = np.zeros_like(cell_stack, dtype=float)
            info_list = []

            for i in range(cell_stack.shape[0]):
                if progress_callback:
                    progress_callback(
                        current_step / total_steps * 100,
                        f"Processing cell frame {i + 1}/{cell_stack.shape[0]}..."
                    )

                # Preprocess cell frame
                processed_frame, frame_info = self.preprocess_frame(
                    cell_stack[i],
                    is_cell=True,
                    apply_registration=False
                )

                # Apply registration transform if available
                if self.params.enable_registration and len(self.transform_matrices) > i:
                    transform_matrix = self.transform_matrices[i]
                    processed_frame = cv2.warpAffine(
                        processed_frame,
                        transform_matrix,
                        (processed_frame.shape[1], processed_frame.shape[0]),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    )

                processed_cells[i] = processed_frame
                info_list.append(frame_info)
                current_step += 1

            results['cells'] = (processed_cells, info_list)

        if progress_callback:
            progress_callback(100, "Preprocessing complete")

        return results



    def update_parameters(self, parameters: PreprocessingParameters):
        """Update preprocessing parameters."""
        parameters.validate()  # Validate parameters before updating
        self.params = parameters

