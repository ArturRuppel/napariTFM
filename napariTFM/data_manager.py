from typing import Tuple, Optional, List, Dict, Any

import numpy as np


class DataManager:
    """Handles data management for TFM analysis."""

    def __init__(self):
        # Raw input data
        self.bead_stack: Optional[np.ndarray] = None
        self.reference_image: Optional[np.ndarray] = None
        self.cell_stack: Optional[np.ndarray] = None

        # Preprocessed data
        self.preprocessed_bead_stack: Optional[np.ndarray] = None
        self.preprocessed_reference: Optional[np.ndarray] = None
        self.preprocessed_cell_stack: Optional[np.ndarray] = None

        # Preprocessing information
        self.bead_preprocessing_info: Optional[List[Dict[str, Any]]] = None
        self.reference_preprocessing_info: Optional[List[Dict[str, Any]]] = None
        self.cell_preprocessing_info: Optional[List[Dict[str, Any]]] = None

        # Registration information
        self.registration_transforms: Optional[Dict[str, Any]] = None

    def set_bead_stack(self, data: np.ndarray):
        """Set bead stack data after validation"""
        if data.ndim != 3:
            raise ValueError("Bead stack must be 3D (frames, height, width)")
        self.bead_stack = data
        # Clear preprocessed data when new raw data is set
        self.preprocessed_bead_stack = None
        self.bead_preprocessing_info = None

    def set_reference_image(self, data: np.ndarray):
        """Set reference image data after validation"""
        if data.ndim != 2:
            raise ValueError("Reference image must be 2D (height, width)")
        self.reference_image = data
        self.preprocessed_reference = None
        self.reference_preprocessing_info = None

    def set_cell_stack(self, data: np.ndarray):
        """Set cell stack data after validation"""
        if data.ndim != 3:
            raise ValueError("Cell stack must be 3D (frames, height, width)")
        self.cell_stack = data
        self.preprocessed_cell_stack = None
        self.cell_preprocessing_info = None
    def clear_data(self):
        """Clear all stored data"""
        self.__init__()

    def has_required_registration_data(self) -> bool:
        """Check if required data for registration is available"""
        return (self.bead_stack is not None and
                self.reference_image is not None)

    def has_any_data(self) -> bool:
        """Check if any data is loaded"""
        return any([
            self.bead_stack is not None,
            self.reference_image is not None,
            self.cell_stack is not None
        ])

    @property
    def bead_stack(self) -> Optional[np.ndarray]:
        return self._bead_stack

    @bead_stack.setter
    def bead_stack(self, data: np.ndarray):
        if data is not None:
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            self._num_frames = data.shape[0]
            self._image_shape = data.shape[1:]
        self._bead_stack = data

    @property
    def reference_image(self) -> Optional[np.ndarray]:
        return self._reference_image

    @reference_image.setter
    def reference_image(self, data: np.ndarray):
        if data is not None:
            if data.ndim == 3:
                raise ValueError("Reference image should be 2D")
            if self._image_shape and data.shape != self._image_shape:
                raise ValueError("Reference image shape doesn't match bead stack")
        self._reference_image = data

    @property
    def preprocessed_bead_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_bead_stack

    @preprocessed_bead_stack.setter
    def preprocessed_bead_stack(self, data: np.ndarray):
        if data is not None and data.ndim == 2:
            data = data[np.newaxis, ...]
        self._preprocessed_bead_stack = data

    @property
    def preprocessed_reference(self) -> Optional[np.ndarray]:
        return self._preprocessed_reference

    @preprocessed_reference.setter
    def preprocessed_reference(self, data: np.ndarray):
        if data is not None and data.ndim == 3:
            raise ValueError("Preprocessed reference should be 2D")
        self._preprocessed_reference = data

    def validate_data(self, data: np.ndarray, is_reference: bool = False) -> Tuple[bool, str]:
        """Validate input data format."""
        if data is None:
            return False, "No data provided"

        if not isinstance(data, np.ndarray):
            return False, "Data must be a numpy array"

        if is_reference:
            if data.ndim != 2:
                return False, "Reference image must be 2D"
            if self._image_shape and data.shape != self._image_shape:
                return False, "Reference image shape doesn't match bead stack"
        else:
            if not (2 <= data.ndim <= 3):
                return False, f"Invalid bead stack dimensions: {data.ndim}"

        return True, "Data valid"

    def clear(self):
        """Clear all data."""
        self._bead_stack = None
        self._reference_image = None
        self._preprocessed_bead_stack = None
        self._preprocessed_reference = None
        self._bead_preprocessing_info = None
        self._reference_preprocessing_info = None
        self._num_frames = 0
        self._image_shape = None