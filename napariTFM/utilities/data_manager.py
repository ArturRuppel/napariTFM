from typing import Optional
import numpy as np
from napariTFM.backend.displacement_analysis import DisplacementResult
from napariTFM.backend.fttc import FTTCResult
from napariTFM.services.msm_service import MSMResult


class DataManager:
    """
    DataManager storing complete analysis result objects.
    """
    def __init__(self):
        # Raw input data
        self._bead_stack: Optional[np.ndarray] = None
        self._reference: Optional[np.ndarray] = None
        self._cell_stack: Optional[np.ndarray] = None

        self._preprocessed_bead_stack: Optional[np.ndarray] = None
        self._preprocessed_cell_stack: Optional[np.ndarray] = None
        self._preprocessed_reference: Optional[np.ndarray] = None

        self._mask_stack: Optional[np.ndarray] = None

        # Analysis results
        self._displacement_results: Optional[DisplacementResult] = None
        self._force_results: Optional[FTTCResult] = None
        self._stress_results: Optional[MSMResult] = None

    def set_bead_stack(self, data: np.ndarray) -> None:
        """Set and validate input bead stack."""
        self._validate_input_stack(data, "bead stack")
        self._bead_stack = data

    def set_reference(self, data: np.ndarray) -> None:
        """Set and validate input reference image."""
        self._validate_reference_image(data)
        self._reference = data

    def set_cell_stack(self, data: np.ndarray) -> None:
        """Set and validate input cell stack."""
        self._validate_input_stack(data, "cell stack")
        self._cell_stack = data

    def set_preprocessed_bead_stack(self, data: np.ndarray) -> None:
        """Set and validate input bead stack."""
        self._validate_input_stack(data, "bead stack")
        self._preprocessed_bead_stack = data
    def set_preprocessed_cell_stack(self, data: np.ndarray) -> None:
        """Set and validate input cell stack."""
        self._validate_input_stack(data, "cell stack")
        self._preprocessed_cell_stack = data

    def set_preprocessed_reference(self, data: np.ndarray) -> None:
        """Set and validate input reference image."""
        self._validate_reference_image(data)
        self._preprocessed_reference = data

    def set_mask_stack(self, data: np.ndarray) -> None:
        """Set and validate input bead stack."""
        self._validate_input_stack(data, "mask stack")
        self._mask_stack = data

    def set_displacement_results(self, results: DisplacementResult) -> None:
        """Store displacement results and invalidate dependent analyses."""
        self._displacement_results = results

    def set_force_results(self, results: FTTCResult) -> None:
        """Store force results and invalidate dependent analyses."""
        self._force_results = results

    def set_stress_results(self, results: MSMResult) -> None:
        """Store stress results."""
        self._stress_results = results

    # Input data properties
    @property
    def bead_stack(self) -> Optional[np.ndarray]:
        return self._bead_stack

    @property
    def reference(self) -> Optional[np.ndarray]:
        return self._reference

    @property
    def cell_stack(self) -> Optional[np.ndarray]:
        return self._cell_stack

    # Result properties
    @property
    def preprocessed_bead_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_bead_stack

    @property
    def preprocessed_reference(self) -> Optional[np.ndarray]:
        return self._preprocessed_reference

    @property
    def preprocessed_cell_stack(self) -> Optional[np.ndarray]:
        return self._preprocessed_cell_stack

    @property
    def mask_stack(self) -> Optional[np.ndarray]:
        return self._mask_stack

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
