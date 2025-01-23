from typing import Optional, Dict, Any

import numpy as np


class DataManager:
    """
    Manages data flow and storage for TFM analysis pipeline.

    The manager handles five main data categories:
    1. Raw input data for preprocessing
    2. Preprocessed output data
    3. Displacement analysis results
    4. Force calculation results
    5. Stress calculation results
    """

    def __init__(self):
        # 1. Raw input data
        self._input_bead_stack: Optional[np.ndarray] = None  # (t, x, y)
        self._input_reference: Optional[np.ndarray] = None  # (x, y)
        self._input_cell_stack: Optional[np.ndarray] = None  # (t, x, y)

        # 2. Preprocessed data
        self._preprocessed_bead_stack: Optional[np.ndarray] = None  # (t, x, y)
        self._preprocessed_reference: Optional[np.ndarray] = None  # (x, y)
        self._preprocessed_cell_stack: Optional[np.ndarray] = None  # (t, x, y)
        self._preprocessing_params: Optional[Dict[str, Any]] = None

        # 3. Displacement results
        self._displacement_field: Optional[np.ndarray] = None  # (t, x, y, 2)
        self._displacement_params: Optional[Dict[str, Any]] = None

        # 4. Force results
        self._force_field: Optional[np.ndarray] = None  # (t, x, y, 2)
        self._force_params: Optional[Dict[str, Any]] = None

        # 5. Stress results
        self._stress_tensor: Optional[np.ndarray] = None  # (t, x, y, 2, 2)
        self._mask_data: Optional[np.ndarray] = None  # (t, x, y)
        self._stress_params: Optional[Dict[str, Any]] = None

    # Input data setters with validation
    def set_input_bead_stack(self, data: np.ndarray) -> None:
        """Set and validate input bead stack."""
        self._validate_input_stack(data, "bead stack")
        self._input_bead_stack = data

    def set_input_reference(self, data: np.ndarray) -> None:
        """Set and validate input reference image."""
        self._validate_reference_image(data)
        self._input_reference = data

    def set_input_cell_stack(self, data: np.ndarray) -> None:
        """Set and validate input cell stack."""
        self._validate_input_stack(data, "cell stack")
        self._input_cell_stack = data

    # Properties for accessing input data
    @property
    def input_bead_stack(self) -> Optional[np.ndarray]:
        return self._input_bead_stack

    @property
    def input_reference(self) -> Optional[np.ndarray]:
        return self._input_reference

    @property
    def input_cell_stack(self) -> Optional[np.ndarray]:
        return self._input_cell_stack

    # Properties for accessing preprocessed data
    @property
    def preprocessed_bead_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_bead_stack

    @property
    def preprocessed_reference(self) -> Optional[np.ndarray]:
        return self._preprocessed_reference

    @property
    def preprocessed_cell_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_cell_stack

    # Methods for setting results with validation
    def set_preprocessing_results(self, bead_stack: Optional[np.ndarray] = None,
                                reference: Optional[np.ndarray] = None,
                                cell_stack: Optional[np.ndarray] = None,
                                params: Optional[Dict[str, Any]] = None) -> None:
        """Store preprocessing results with validation."""
        if bead_stack is not None:
            self._validate_input_stack(bead_stack, "preprocessed bead stack")
            self._preprocessed_bead_stack = bead_stack

        if reference is not None:
            self._validate_reference_image(reference)
            self._preprocessed_reference = reference

        if cell_stack is not None:
            self._validate_input_stack(cell_stack, "preprocessed cell stack")
            self._preprocessed_cell_stack = cell_stack

        if params is not None:
            self._preprocessing_params = params.copy()

    # Validation methods
    def _validate_input_stack(self, data: np.ndarray, name: str) -> None:
        """Validate dimensions and type of input stack data."""
        if not isinstance(data, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        if data.ndim not in [2, 3]:
            raise ValueError(f"{name} must be 2D or 3D (got {data.ndim}D)")

    def _validate_reference_image(self, data: np.ndarray) -> None:
        """Validate dimensions and type of reference image."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Reference image must be a numpy array")
        if data.ndim != 2:
            raise ValueError(f"Reference image must be 2D (got {data.ndim}D)")

    def _validate_field_data(self, data: np.ndarray, expected_dims: int) -> None:
        """Validate dimensions and type of field data."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Field data must be a numpy array")
        if data.ndim != expected_dims:
            raise ValueError(f"Field data must be {expected_dims}D (got {data.ndim}D)")

    # Utility methods
    def clear_all(self) -> None:
        """Clear all stored data."""
        for attr in vars(self):
            setattr(self, attr, None)

    def get_mask_data(self):
        """Get the mask data."""
        return self._mask_data

    def set_mask_data(self, analysis_mask, visualization_mask=None, **kwargs):
        """Set the mask data."""
        self._mask_data = analysis_mask

    def get_force_data(self):
        """Get the force data."""
        return self._force_data

    def set_force_data(self, force_data):
        """Set the force data."""
        self._force_data = force_data

    def get_stress_data(self):
        """Get the stress data."""
        return self._stress_data

    def set_stress_data(self, stress_data):
        """Set the stress data."""
        self._stress_data = stress_data