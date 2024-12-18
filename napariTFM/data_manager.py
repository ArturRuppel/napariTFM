from typing import Tuple, Optional, List, Dict, Any
import numpy as np


class DataManager:
    """Handles data management for TFM analysis."""

    def __init__(self):
        # Raw input data
        self._bead_stack: Optional[np.ndarray] = None
        self._reference_image: Optional[np.ndarray] = None
        self._cell_stack: Optional[np.ndarray] = None

        # Preprocessed data
        self._preprocessed_bead_stack: Optional[np.ndarray] = None
        self._preprocessed_reference: Optional[np.ndarray] = None
        self._preprocessed_cell_stack: Optional[np.ndarray] = None

        # Preprocessing information
        self.bead_preprocessing_info: Optional[List[Dict[str, Any]]] = None
        self.reference_preprocessing_info: Optional[List[Dict[str, Any]]] = None
        self.cell_preprocessing_info: Optional[List[Dict[str, Any]]] = None

        # Registration information
        self.registration_transforms: Optional[Dict[str, Any]] = None

        # Displacement results
        self.displacement_results: Optional[Dict[str, Any]] = None

        # Force calculation results
        self._force_results: Optional[Dict[str, Any]] = None

        # Internal state
        self._num_frames: Optional[int] = None
        self._image_shape: Optional[Tuple[int, ...]] = None
    def set_bead_stack(self, data: np.ndarray):
        """Set bead stack data after validation"""
        if data.ndim not in [2, 3]:
            raise ValueError("Bead stack must be 2D or 3D (frames, height, width)")

        # Convert 2D to 3D if necessary
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._bead_stack = data
        self._num_frames = data.shape[0]
        self._image_shape = data.shape[1:]

        # Clear preprocessed data when new raw data is set
        self._preprocessed_bead_stack = None
        self.bead_preprocessing_info = None

    def set_reference_image(self, data: np.ndarray):
        """Set reference image data after validation"""
        if data.ndim != 2:
            raise ValueError("Reference image must be 2D (height, width)")

        # Check shape compatibility if we have a bead stack
        if self._image_shape is not None:
            if data.shape != self._image_shape:
                raise ValueError(f"Reference image shape {data.shape} doesn't match bead stack shape {self._image_shape}")

        self._reference_image = data
        if self._image_shape is None:
            self._image_shape = data.shape

        self._preprocessed_reference = None
        self.reference_preprocessing_info = None

    def set_cell_stack(self, data: np.ndarray):
        """Set cell stack data after validation"""
        if data.ndim not in [2, 3]:
            raise ValueError("Cell stack must be 2D or 3D (frames, height, width)")

        # Convert 2D to 3D if necessary
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Check shape compatibility
        if self._image_shape is not None:
            if data.shape[1:] != self._image_shape:
                raise ValueError(f"Cell stack shape {data.shape[1:]} doesn't match expected shape {self._image_shape}")

        self._cell_stack = data
        if self._image_shape is None:
            self._image_shape = data.shape[1:]
            self._num_frames = data.shape[0]

        self._preprocessed_cell_stack = None
        self.cell_preprocessing_info = None

    def clear_data(self):
        """Clear all stored data"""
        self.__init__()

    def has_required_registration_data(self) -> bool:
        """Check if required data for registration is available"""
        has_beads = self._bead_stack is not None or self._preprocessed_bead_stack is not None
        has_reference = self._reference_image is not None or self._preprocessed_reference is not None
        return has_beads and has_reference

    def has_any_data(self) -> bool:
        """Check if any data is loaded"""
        return any([
            self._bead_stack is not None,
            self._reference_image is not None,
            self._cell_stack is not None,
            self._preprocessed_bead_stack is not None,
            self._preprocessed_reference is not None,
            self._preprocessed_cell_stack is not None
        ])

    @property
    def bead_stack(self) -> Optional[np.ndarray]:
        """Get bead stack data"""
        return self._bead_stack

    @bead_stack.setter
    def bead_stack(self, data: Optional[np.ndarray]):
        """Set bead stack data with validation"""
        if data is not None:
            self.set_bead_stack(data)
        else:
            self._bead_stack = None
            self._preprocessed_bead_stack = None
            self.bead_preprocessing_info = None

    @property
    def reference_image(self) -> Optional[np.ndarray]:
        """Get reference image data"""
        return self._reference_image

    @reference_image.setter
    def reference_image(self, data: Optional[np.ndarray]):
        """Set reference image data with validation"""
        if data is not None:
            self.set_reference_image(data)
        else:
            self._reference_image = None
            self._preprocessed_reference = None
            self.reference_preprocessing_info = None

    @property
    def cell_stack(self) -> Optional[np.ndarray]:
        """Get cell stack data"""
        return self._cell_stack

    @cell_stack.setter
    def cell_stack(self, data: Optional[np.ndarray]):
        """Set cell stack data with validation"""
        if data is not None:
            self.set_cell_stack(data)
        else:
            self._cell_stack = None
            self._preprocessed_cell_stack = None
            self.cell_preprocessing_info = None

    @property
    def preprocessed_bead_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_bead_stack

    @preprocessed_bead_stack.setter
    def preprocessed_bead_stack(self, data: Optional[np.ndarray]):
        if data is not None and data.ndim == 2:
            data = data[np.newaxis, ...]
        self._preprocessed_bead_stack = data

    @property
    def preprocessed_reference(self) -> Optional[np.ndarray]:
        return self._preprocessed_reference

    @preprocessed_reference.setter
    def preprocessed_reference(self, data: Optional[np.ndarray]):
        if data is not None and data.ndim == 3:
            raise ValueError("Preprocessed reference should be 2D")
        self._preprocessed_reference = data

    @property
    def preprocessed_cell_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_cell_stack

    @preprocessed_cell_stack.setter
    def preprocessed_cell_stack(self, data: Optional[np.ndarray]):
        if data is not None and data.ndim == 2:
            data = data[np.newaxis, ...]
        self._preprocessed_cell_stack = data

    def validate_data(self, data: np.ndarray, is_reference: bool = False) -> Tuple[bool, str]:
        """Validate input data format."""
        if data is None:
            return False, "No data provided"

        if not isinstance(data, np.ndarray):
            return False, "Data must be a numpy array"

        if is_reference:
            if data.ndim != 2:
                return False, "Reference image must be 2D"
            if self._image_shape is not None and data.shape != self._image_shape:
                return False, f"Reference image shape {data.shape} doesn't match expected shape {self._image_shape}"
        else:
            if data.ndim not in [2, 3]:
                return False, f"Invalid data dimensions: {data.ndim}"
            if data.ndim == 3 and self._image_shape is not None:
                if data.shape[1:] != self._image_shape:
                    return False, f"Data shape {data.shape[1:]} doesn't match expected shape {self._image_shape}"

        return True, "Data valid"

    @property
    def force_results(self) -> Optional[Dict[str, Any]]:
        """Get force calculation results"""
        return self._force_results

    @force_results.setter
    def force_results(self, results: Optional[Dict[str, Any]]):
        """Set force calculation results after validation"""
        if results is not None:
            required_keys = ['tx', 'ty', 'energy', 'contractile_force', 'parameters']
            if not all(key in results for key in required_keys):
                raise ValueError("Force results missing required fields")

            # Validate array shapes
            if not all(isinstance(results[key], np.ndarray) for key in ['tx', 'ty', 'energy']):
                raise ValueError("Force results must contain numpy arrays")

            if results['tx'].shape != results['ty'].shape:
                raise ValueError("Force components tx and ty must have same shape")

        self._force_results = results

    def clear_force_results(self):
        """Clear force calculation results"""
        self._force_results = None

    def has_required_force_data(self) -> bool:
        """Check if required data for force calculation is available"""
        return self.displacement_results is not None and 'flows' in self.displacement_results