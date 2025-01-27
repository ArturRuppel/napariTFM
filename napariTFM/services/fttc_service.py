from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Generator
import numpy as np

from napariTFM.backend.fttc import FTTC


@dataclass
class FTTCParameters:
    """Parameters for FTTC calculations"""
    # Material parameters
    young_modulus: float  # Pa
    poisson_ratio_substrate: float
    gel_height: Optional[float]  # μm (None for infinite thickness)
    lanczos_exp: int

    # Regularization parameters
    regularization: float
    auto_gcv: bool

    # Visualization parameters
    force_vector_stride: int
    force_arrow_scale: float
    f_max: float

    # Time parameters
    frame_interval: float  # minutes


@dataclass
class FTTCCalculationResult:
    """Results from force calculation"""
    force_field: np.ndarray  # Shape: (frames, height, width, 2) for x,y components
    condition_number: float
    residual: float
    parameters: FTTCParameters


class FTTCService:
    """Service layer handling business logic for FTTC force calculations."""

    def __init__(self):
        self.calculator = None

    def initialize_calculator(self, params: FTTCParameters):
        """Initialize or update the FTTC calculator with given parameters."""
        self.calculator = FTTC(
            E=params.young_modulus,
            nu=params.poisson_ratio_substrate,
            lanczos_exp=params.lanczos_exp,
            gel_height=params.gel_height
        )

    def calculate_forces(
            self,
            displacement_field: np.ndarray,
            pixel_size: float,
            downscale_factor: int = 1,
            regularization: Optional[float] = None,
            use_gcv: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forces from displacement field for a single frame.

        Parameters
        ----------
        displacement_field : np.ndarray
            2D array of displacement vectors (height, width, 2)
        pixel_size : float
            Physical size of each pixel in micrometers
        downscale_factor : int
            Factor by which the displacement field was downscaled
        regularization : float, optional
            Regularization parameter (if None and use_gcv=False, will use current value)
        use_gcv : bool
            Whether to use GCV to determine regularization parameter

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Calculated force components (tx, ty)
        """
        if self.calculator is None:
            raise RuntimeError("Calculator not initialized. Call initialize_calculator first.")

        # Calculate forces using the FTTC calculator
        if use_gcv:
            regularization = None  # Force GCV calculation

        result = self.calculator.calculate_traction(
            displacements=displacement_field,
            pixel_size=pixel_size,
            downscale_factor=downscale_factor,
            regularization=regularization
        )

        return result[1][0], result[1][1]  # Return tx, ty components

    def calculate_force_stack(
            self,
            displacement_field: np.ndarray,
            pixel_size: float,
            downscale_factor: int = 1,
            regularization: Optional[float] = None,
            use_gcv: bool = False
    ) -> Generator[Tuple[Dict[str, Any], int, int], None, Dict[str, Any]]:
        """
        Calculate forces for all frames in the displacement stack.

        Yields progress updates and returns final results.

        Parameters
        ----------
        displacement_field : np.ndarray
            4D array of displacement vectors (frames, height, width, 2)
        pixel_size : float
            Physical size of each pixel in micrometers
        downscale_factor : int
            Factor by which the displacement field was downscaled
        regularization : float, optional
            Regularization parameter (if None and use_gcv=False, will use current value)
        use_gcv : bool
            Whether to use GCV to determine regularization parameter

        Yields
        ------
        Tuple[Dict[str, Any], int, int]
            (progress_info, current_frame, total_frames)

        Returns
        -------
        Dict[str, Any]
            Complete force calculation results
        """
        if self.calculator is None:
            raise RuntimeError("Calculator not initialized. Call initialize_calculator first.")

        total_frames = len(displacement_field)
        force_results = {'tx': [], 'ty': []}

        for frame_idx in range(total_frames):
            # Calculate forces for current frame
            tx, ty = self.calculate_forces(
                displacement_field=displacement_field[frame_idx],
                pixel_size=pixel_size,
                downscale_factor=downscale_factor,
                regularization=regularization,
                use_gcv=use_gcv
            )

            force_results['tx'].append(tx)
            force_results['ty'].append(ty)

            # Calculate magnitude for progress statistics
            magnitude = np.sqrt(tx ** 2 + ty ** 2)

            # Create progress info
            progress_info = {
                'frame': frame_idx,
                'mean_force': np.mean(magnitude),
                'max_force': np.max(magnitude),
                'median_force': np.median(magnitude)
            }

            yield progress_info, frame_idx + 1, total_frames

        # Convert lists to arrays for final results
        force_results['tx'] = np.stack(force_results['tx'])
        force_results['ty'] = np.stack(force_results['ty'])

        return force_results

    def find_optimal_regularization(
            self,
            displacement_field: np.ndarray,
            pixel_size: float,
            downscale_factor: int = 1
    ) -> float:
        """
        Find optimal regularization parameter using GCV for current frame.

        Parameters
        ----------
        displacement_field : np.ndarray
            2D array of displacement vectors (height, width, 2)
        pixel_size : float
            Physical size of each pixel in micrometers
        downscale_factor : int
            Factor by which the displacement field was downscaled

        Returns
        -------
        float
            Optimal regularization parameter
        """
        if self.calculator is None:
            raise RuntimeError("Calculator not initialized. Call initialize_calculator first.")

        # Prepare data for GCV calculation
        shape = displacement_field.shape[:-1]
        pos = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy'))
        vec = np.array([displacement_field[..., 0], displacement_field[..., 1]])

        # Calculate optimal regularization
        return self.calculator._find_regularization(pos, vec, pixel_size * downscale_factor)

    def validate_displacement_field(self, displacement_field: np.ndarray) -> Tuple[bool, str]:
        """
        Validate displacement field data.

        Parameters
        ----------
        displacement_field : np.ndarray
            Displacement field data to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        if displacement_field is None:
            return False, "No displacement field data provided"

        if not isinstance(displacement_field, np.ndarray):
            return False, "Displacement field must be a numpy array"

        if displacement_field.ndim != 4:
            return False, f"Displacement field must be 4D (frames, height, width, 2), got shape {displacement_field.shape}"

        if displacement_field.shape[-1] != 2:
            return False, f"Last dimension must be 2 (x,y components), got {displacement_field.shape[-1]}"

        if np.all(np.isnan(displacement_field)):
            return False, "Displacement field contains only NaN values"

        return True, ""

    def validate_parameters(self, params: FTTCParameters) -> Tuple[bool, str]:
        """
        Validate FTTC calculation parameters.

        Parameters
        ----------
        params : FTTCParameters
            Parameters to validate

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        if params.young_modulus <= 0:
            return False, "Young's modulus must be positive"

        if not 0 <= params.poisson_ratio_substrate <= 0.5:
            return False, "Poisson ratio must be between 0 and 0.5"

        if params.gel_height is not None and params.gel_height < 0:
            return False, "Gel height must be non-negative or None (infinite)"

        if params.lanczos_exp < 0:
            return False, "Lanczos exponent must be non-negative"

        if params.regularization <= 0:
            return False, "Regularization parameter must be positive"

        if params.force_vector_stride < 1:
            return False, "Vector stride must be at least 1"

        if params.force_arrow_scale <= 0:
            return False, "Arrow scale must be positive"

        if params.f_max <= 0:
            return False, "Maximum force must be positive"

        if params.frame_interval <= 0:
            return False, "Frame interval must be positive"

        return True, ""

    def process_force_data(
            self,
            force_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
        """
        Process loaded force data into standardized format.

        Parameters
        ----------
        force_data : Dict[str, Any]
            Raw force data dictionary from file

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any], List[str]]
            (force_field, parameters, warnings)
        """
        warnings = []

        # Validate basic structure
        if 'force_field' not in force_data or 'parameters' not in force_data:
            raise ValueError("Invalid force data format: missing required keys")

        force_field = force_data['force_field']
        parameters = force_data['parameters']

        # Convert to numpy array if needed
        if not isinstance(force_field, np.ndarray):
            force_field = np.array(force_field)
            warnings.append("Force field converted to numpy array")

        # Validate shape
        if force_field.ndim != 4 or force_field.shape[-1] != 2:
            raise ValueError(f"Invalid force field shape: {force_field.shape}")

        # Ensure parameters has required keys
        required_params = ['pixel_size', 'young_modulus', 'poisson_ratio_substrate']
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        return force_field, parameters, warnings