from typing import Tuple, Optional, List, Dict, Any
import numpy as np


class DataManager:
    """Handles data management for TFM analysis."""


    def __init__(self):
        # Raw input data for preprocessing
        self._preprocessing_bead_stack: Optional[np.ndarray] = None
        self._preprocessing_reference_image: Optional[np.ndarray] = None
        self._preprocessing_cell_stack: Optional[np.ndarray] = None

        # Raw input data for displacement analysis
        self._displacement_bead_stack: Optional[np.ndarray] = None
        self._displacement_reference_image: Optional[np.ndarray] = None

        # Preprocessed data (output from preprocessing)
        self._preprocessed_bead_stack: Optional[np.ndarray] = None
        self._preprocessed_reference: Optional[np.ndarray] = None
        self._preprocessed_cell_stack: Optional[np.ndarray] = None

        # Add mask-related attributes
        self._mask_stack: Optional[np.ndarray] = None
        self._visualization_mask_stack: Optional[np.ndarray] = None
        self.mask_processing_info: Optional[Dict[str, Any]] = None

        self.bead_preprocessing_info: Optional[List[Dict[str, Any]]] = None
        self.reference_preprocessing_info: Optional[List[Dict[str, Any]]] = None
        self.cell_preprocessing_info: Optional[List[Dict[str, Any]]] = None
        self.registration_transforms: Optional[Dict[str, Any]] = None
        self.displacement_results: Optional[Dict[str, Any]] = None
        self._force_results: Optional[Dict[str, Any]] = None
        self._num_frames: Optional[int] = None
        self._image_shape: Optional[Tuple[int, ...]] = None



    @property
    def preprocessing_bead_stack(self) -> Optional[np.ndarray]:
        return self._preprocessing_bead_stack

    @property
    def preprocessing_reference_image(self) -> Optional[np.ndarray]:
        return self._preprocessing_reference_image

    @property
    def preprocessing_cell_stack(self) -> Optional[np.ndarray]:
        return self._preprocessing_cell_stack

    @property
    def preprocessed_bead_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_bead_stack

    @property
    def preprocessed_reference(self) -> Optional[np.ndarray]:
        return self._preprocessed_reference

    @property
    def preprocessed_cell_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_cell_stack

    @preprocessed_bead_stack.setter
    def preprocessed_bead_stack(self, data: Optional[np.ndarray]):
        if data is not None and data.ndim == 2:
            data = data[np.newaxis, ...]
        self._preprocessed_bead_stack = data

    @preprocessed_reference.setter
    def preprocessed_reference(self, data: Optional[np.ndarray]):
        if data is not None and data.ndim == 3:
            raise ValueError("Preprocessed reference should be 2D")
        self._preprocessed_reference = data

    @preprocessed_cell_stack.setter
    def preprocessed_cell_stack(self, data: Optional[np.ndarray]):
        if data is not None and data.ndim == 2:
            data = data[np.newaxis, ...]
        self._preprocessed_cell_stack = data

    @property
    def displacement_bead_stack(self) -> Optional[np.ndarray]:
        return self._displacement_bead_stack

    @property
    def displacement_reference_image(self) -> Optional[np.ndarray]:
        return self._displacement_reference_image


    @property
    def force_results(self) -> Optional[Dict[str, Any]]:
        """Get force calculation results"""
        return self._force_results

    @force_results.setter
    def force_results(self, results: Optional[Dict[str, Any]]):
        """Set force calculation results after validation"""
        if results is not None:
            # Check for required fields
            required_keys = ['tx', 'ty', 'parameters']
            if not all(key in results for key in required_keys):
                raise ValueError("Force results missing required fields")

            # Validate array shapes
            if not all(isinstance(results[key], np.ndarray) for key in ['tx', 'ty']):
                raise ValueError("Force results must contain numpy arrays")

            if results['tx'].shape != results['ty'].shape:
                raise ValueError("Force components tx and ty must have same shape")

        self._force_results = results

    @property
    def mask_stack(self) -> Optional[np.ndarray]:
        """Get the mask stack at analysis resolution."""
        return self._mask_stack

    @property
    def visualization_mask_stack(self) -> Optional[np.ndarray]:
        """Get the mask stack at visualization resolution."""
        return self._visualization_mask_stack

    def set_preprocessing_bead_stack(self, data: np.ndarray):
        """Set bead stack data for preprocessing after validation"""
        if data.ndim not in [2, 3]:
            raise ValueError("Bead stack must be 2D or 3D (frames, height, width)")

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._preprocessing_bead_stack = data
        self._num_frames = data.shape[0]
        self._image_shape = data.shape[1:]

    def set_preprocessing_reference_image(self, data: np.ndarray):
        """Set reference image data for preprocessing after validation"""
        if data.ndim != 2:
            raise ValueError("Reference image must be 2D (height, width)")

        if self._image_shape is not None and data.shape != self._image_shape:
            raise ValueError(f"Reference image shape {data.shape} doesn't match expected shape {self._image_shape}")

        self._preprocessing_reference_image = data
        if self._image_shape is None:
            self._image_shape = data.shape

    def set_preprocessing_cell_stack(self, data: np.ndarray):
        """Set cell stack data for preprocessing after validation"""
        if data.ndim not in [2, 3]:
            raise ValueError("Cell stack must be 2D or 3D (frames, height, width)")

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        if self._image_shape is not None and data.shape[1:] != self._image_shape:
            raise ValueError(f"Cell stack shape {data.shape[1:]} doesn't match expected shape {self._image_shape}")

        self._preprocessing_cell_stack = data
        if self._image_shape is None:
            self._image_shape = data.shape[1:]
            self._num_frames = data.shape[0]

    def set_displacement_bead_stack(self, data: np.ndarray):
        """Set bead stack data for displacement analysis after validation"""
        if data.ndim not in [2, 3]:
            raise ValueError("Bead stack must be 2D or 3D (frames, height, width)")

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._displacement_bead_stack = data

    def set_displacement_reference_image(self, data: np.ndarray):
        """Set reference image data for displacement analysis after validation"""
        if data.ndim != 2:
            raise ValueError("Reference image must be 2D (height, width)")

        self._displacement_reference_image = data


    def set_mask_stack(self, mask_stack: np.ndarray, visualization_mask_stack: Optional[np.ndarray] = None,
                       processing_info: Optional[Dict[str, Any]] = None):
        """Set the mask stack data after validation.

        Args:
            mask_stack: Mask data at analysis resolution (frames, height, width)
            visualization_mask_stack: Optional upscaled mask for visualization
            processing_info: Optional dictionary containing mask processing parameters
        """
        if mask_stack.ndim not in [2, 3]:
            raise ValueError("Mask stack must be 2D or 3D (frames, height, width)")

        if mask_stack.ndim == 2:
            mask_stack = mask_stack[np.newaxis, ...]

        self._mask_stack = mask_stack
        self._visualization_mask_stack = visualization_mask_stack
        self.mask_processing_info = processing_info

    def clear_all_data(self):
        """Clear all stored data, resetting the DataManager to its initial state."""
        # Clear raw input data for preprocessing
        self._preprocessing_bead_stack = None
        self._preprocessing_reference_image = None
        self._preprocessing_cell_stack = None

        # Clear raw input data for displacement analysis
        self._displacement_bead_stack = None
        self._displacement_reference_image = None

        # Clear preprocessed data
        self._preprocessed_bead_stack = None
        self._preprocessed_reference = None
        self._preprocessed_cell_stack = None

        # Clear preprocessing info
        self.bead_preprocessing_info = None
        self.reference_preprocessing_info = None
        self.cell_preprocessing_info = None

        # Clear analysis results and transforms
        self.registration_transforms = None
        self.displacement_results = None
        self._force_results = None

        # Clear metadata
        self._num_frames = None
        self._image_shape = None

        self._mask_stack = None
        self._visualization_mask_stack = None
        self.mask_processing_info = None

    def validate_mask_shape(self, shape: Tuple[int, ...]) -> bool:
        """Validate if mask shape matches expected dimensions.

        Args:
            shape: Shape to validate against current mask

        Returns:
            bool: True if shapes match, False otherwise
        """
        if self._mask_stack is None:
            return False

        return self._mask_stack.shape[1:] == shape

    # def clear_force_results(self):
    #     """Clear force calculation results"""
    #     self._force_results = None

    # def has_required_force_data(self) -> bool:
    #     """Check if required data for force calculation is available"""
    #     return self.displacement_results is not None and 'flows' in self.displacement_results