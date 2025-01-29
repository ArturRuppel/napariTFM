from typing import Optional
import numpy as np
from napariTFM.services.preprocessing_service import PreprocessingResult
from napariTFM.services.displacement_service import DisplacementResult
from napariTFM.services.fttc_service import FTTCResult
from napariTFM.services.msm_service import MSMResult


class DataManager:
    """
    DataManager storing complete Result objects from service layer.
    """
    def __init__(self):
        # Raw input data
        self._input_bead_stack: Optional[np.ndarray] = None
        self._input_reference: Optional[np.ndarray] = None
        self._input_cell_stack: Optional[np.ndarray] = None

        # Analysis results as service Result objects
        self._preprocessing_results: Optional[PreprocessingResult] = None
        self._displacement_results: Optional[DisplacementResult] = None
        self._force_results: Optional[FTTCResult] = None
        self._stress_results: Optional[MSMResult] = None

    def set_input_bead_stack(self, data: np.ndarray) -> None:
        """Set and validate input bead stack."""
        self._validate_input_stack(data, "bead stack")
        self._input_bead_stack = data
        self._invalidate_from_preprocessing()

    def set_input_reference(self, data: np.ndarray) -> None:
        """Set and validate input reference image."""
        self._validate_reference_image(data)
        self._input_reference = data
        self._invalidate_from_preprocessing()

    def set_input_cell_stack(self, data: np.ndarray) -> None:
        """Set and validate input cell stack."""
        self._validate_input_stack(data, "cell stack")
        self._input_cell_stack = data
        self._invalidate_from_preprocessing()

    def set_preprocessing_results(self, results: PreprocessingResult) -> None:
        """Store preprocessing results and invalidate dependent analyses."""
        self._preprocessing_results = results
        self._invalidate_from_displacement()

    def set_displacement_results(self, results: DisplacementResult) -> None:
        """Store displacement results and invalidate dependent analyses."""
        self._displacement_results = results
        self._invalidate_from_force()

    def set_force_results(self, results: FTTCResult) -> None:
        """Store force results and invalidate dependent analyses."""
        self._force_results = results
        self._invalidate_stress()

    def set_stress_results(self, results: MSMResult) -> None:
        """Store stress results."""
        self._stress_results = results

    def _invalidate_from_preprocessing(self):
        """Invalidate all analysis steps from preprocessing onwards."""
        self._preprocessing_results = None
        self._invalidate_from_displacement()

    def _invalidate_from_displacement(self):
        """Invalidate all analysis steps from displacement onwards."""
        self._displacement_results = None
        self._invalidate_from_force()

    def _invalidate_from_force(self):
        """Invalidate all analysis steps from force onwards."""
        self._force_results = None
        self._invalidate_stress()

    def _invalidate_stress(self):
        """Invalidate stress analysis."""
        self._stress_results = None

    # Input data properties
    @property
    def input_bead_stack(self) -> Optional[np.ndarray]:
        return self._input_bead_stack

    @property
    def input_reference(self) -> Optional[np.ndarray]:
        return self._input_reference

    @property
    def input_cell_stack(self) -> Optional[np.ndarray]:
        return self._input_cell_stack

    # Result properties
    @property
    def preprocessing_results(self) -> Optional[PreprocessingResult]:
        return self._preprocessing_results

    @property
    def displacement_results(self) -> Optional[DisplacementResult]:
        return self._displacement_results

    @property
    def force_results(self) -> Optional[FTTCResult]:
        return self._force_results

    @property
    def stress_results(self) -> Optional[MSMResult]:
        return self._stress_results

    # Validation methods
    def _validate_input_stack(self, data: np.ndarray, name: str) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        if data.ndim not in [2, 3]:
            raise ValueError(f"{name} must be 2D or 3D (got {data.ndim}D)")

    def _validate_reference_image(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError("Reference image must be a numpy array")
        if data.ndim != 2:
            raise ValueError(f"Reference image must be 2D (got {data.ndim}D)")