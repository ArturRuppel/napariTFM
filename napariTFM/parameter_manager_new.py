from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, Callable, Set, Optional, Tuple
from enum import Enum, auto
import yaml
from pathlib import Path
from qtpy.QtCore import QObject, Signal

from napariTFM.services.displacement_service import DisplacementService, DisplacementParameters
from napariTFM.services.fttc_service import FTTCService, FTTCParameters
from napariTFM.services.msm_service import MSMService, MSMParameters
from napariTFM.services.preprocessing_service import PreprocessingService, PreprocessingParameters


class ParameterCategory(Enum):
    """Enum defining different categories of parameters."""
    GENERAL = auto()
    PREPROCESSING = auto()
    DISPLACEMENT = auto()
    FORCE = auto()
    STRESS = auto()
    VISUALIZATION = auto()


@dataclass
class UnifiedParameters:
    """Single source of truth for all parameters"""
    # General parameters
    pixel_size: float = 1.0
    frame_interval: float = 1.0

    # Preprocessing parameters
    min_intensity_percentile: float = 0.0
    max_intensity_percentile: float = 1.0
    enable_gaussian_filter: bool = False
    gaussian_sigma: float = 0.0
    cell_min_intensity_percentile: float = 0.0
    cell_max_intensity_percentile: float = 1.0
    enable_cell_gaussian_filter: bool = False
    cell_gaussian_sigma: float = 0.0
    registration_mode: str = 'translation'

    # Displacement parameters
    tau: float = 0.25
    lambda_: float = 0.4
    theta: float = 0.3
    nscales: int = 3
    warps: int = 3
    epsilon: float = 0.01
    inner_iterations: int = 15
    outer_iterations: int = 5
    scale_step: float = 0.5
    median_filtering: int = 5
    downscale_factor: int = 1
    disp_vector_stride: int = 20
    disp_arrow_scale: float = 1.0
    d_max: float = 5.0

    # Force parameters
    young_modulus: float = 10000.0
    poisson_ratio_substrate: float = 0.49
    gel_height: Optional[float] = None
    lanczos_exp: int = 1
    regularization: float = 1e-4
    auto_gcv: bool = False
    force_vector_stride: int = 20
    force_arrow_scale: float = 1.0
    f_max: float = 1000.0

    # Stress parameters
    threshold: float = 0.0
    dilation: int = 10
    smoothing_sigma: float = 10.0
    density_factor: float = 0.025
    mesh_algorithm: str = 'Frontal-Del.'
    use_optimization: bool = True
    poisson_ratio_cells: float = 0.5
    max_stress: float = 1.0

    # Visualization parameters (you can add more as needed)
    show_vectors: bool = True
    show_colormap: bool = True

    def to_preprocessing_parameters(self) -> PreprocessingParameters:
        """Create PreprocessingParameters from unified parameters"""
        return PreprocessingParameters(
            min_intensity_percentile=self.min_intensity_percentile,
            max_intensity_percentile=self.max_intensity_percentile,
            enable_gaussian_filter=self.enable_gaussian_filter,
            gaussian_sigma=self.gaussian_sigma,
            cell_min_intensity_percentile=self.cell_min_intensity_percentile,
            cell_max_intensity_percentile=self.cell_max_intensity_percentile,
            enable_cell_gaussian_filter=self.enable_cell_gaussian_filter,
            cell_gaussian_sigma=self.cell_gaussian_sigma,
            registration_mode=self.registration_mode
        )

    def to_displacement_parameters(self) -> DisplacementParameters:
        """Create DisplacementParameters from unified parameters"""
        return DisplacementParameters(
            tau=self.tau,
            lambda_=self.lambda_,
            theta=self.theta,
            nscales=self.nscales,
            warps=self.warps,
            epsilon=self.epsilon,
            inner_iterations=self.inner_iterations,
            outer_iterations=self.outer_iterations,
            scale_step=self.scale_step,
            median_filtering=self.median_filtering,
            downscale_factor=self.downscale_factor,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval,
            d_max=self.d_max,
            disp_vector_stride=self.disp_vector_stride,
            disp_arrow_scale=self.disp_arrow_scale
        )

    def to_fttc_parameters(self) -> FTTCParameters:
        """Create FTTCParameters from unified parameters"""
        return FTTCParameters(
            young_modulus=self.young_modulus,
            poisson_ratio_substrate=self.poisson_ratio_substrate,
            gel_height=self.gel_height,
            lanczos_exp=self.lanczos_exp,
            regularization=self.regularization,
            auto_gcv=self.auto_gcv,
            downscale_factor=self.downscale_factor,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval,
            force_vector_stride=self.force_vector_stride,
            force_arrow_scale=self.force_arrow_scale,
            f_max=self.f_max
        )

    def to_msm_parameters(self) -> MSMParameters:
        """Create MSMParameters from unified parameters"""
        return MSMParameters(
            threshold=self.threshold,
            dilation=self.dilation,
            smoothing_sigma=self.smoothing_sigma,
            density_factor=self.density_factor,
            algorithm=self.mesh_algorithm,
            use_optimization=self.use_optimization,
            poisson_ratio_cells=self.poisson_ratio_cells,
            young_modulus=self.young_modulus,
            pixel_size=self.pixel_size,
            downscale_factor=self.downscale_factor,
            frame_interval=self.frame_interval,
            max_stress=self.max_stress
        )


class ParameterManager(QObject):
    """Manages all parameters for the TFM analysis pipeline"""

    parameter_changed = Signal(str, object)  # (parameter_name, new_value)
    parameters_reset = Signal(ParameterCategory)  # Emitted when parameters are reset

    def __init__(self):
        super().__init__()
        self._parameters = UnifiedParameters()
        self._callbacks: Dict[str, Set[Callable]] = {}
        self._initialize_callbacks()

    def _initialize_callbacks(self):
        """Initialize callback sets for all parameters"""
        for field in fields(self._parameters):
            self._callbacks[field.name] = set()

    def register_callback(self, param_name: str, callback: Callable) -> None:
        """Register a callback for parameter changes"""
        if not hasattr(self._parameters, param_name):
            raise ValueError(f"Unknown parameter: {param_name}")
        self._callbacks[param_name].add(callback)

    def unregister_callback(self, param_name: str, callback: Callable) -> None:
        """Unregister a callback for parameter changes"""
        if not hasattr(self._parameters, param_name):
            raise ValueError(f"Unknown parameter: {param_name}")
        self._callbacks[param_name].discard(callback)

    def get_parameter(self, name: str) -> Any:
        """Get a parameter value"""
        if not hasattr(self._parameters, name):
            raise ValueError(f"Unknown parameter: {name}")
        return getattr(self._parameters, name)

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value and trigger callbacks"""
        if not hasattr(self._parameters, name):
            raise ValueError(f"Unknown parameter: {name}")

        current_value = getattr(self._parameters, name)
        if current_value != value:
            setattr(self._parameters, name, value)

            # Trigger callbacks
            for callback in self._callbacks[name]:
                callback(value)

            # Emit signal
            self.parameter_changed.emit(name, value)

    def get_preprocessing_parameters(self) -> PreprocessingParameters:
        """Get parameters for preprocessing service"""
        return self._parameters.to_preprocessing_parameters()

    def get_displacement_parameters(self) -> DisplacementParameters:
        """Get parameters for displacement service"""
        return self._parameters.to_displacement_parameters()

    def get_fttc_parameters(self) -> FTTCParameters:
        """Get parameters for FTTC service"""
        return self._parameters.to_fttc_parameters()

    def get_msm_parameters(self) -> MSMParameters:
        """Get parameters for MSM service"""
        return self._parameters.to_msm_parameters()

    def reset_all_parameters(self) -> None:
        """Reset all parameters to default values"""
        new_params = UnifiedParameters()
        self._update_all_parameters(new_params)
        for category in ParameterCategory:
            self.parameters_reset.emit(category)

    def reset_preprocessing_parameters(self) -> None:
        """Reset preprocessing parameters to defaults"""
        defaults = UnifiedParameters()
        for field in fields(PreprocessingParameters):
            self.set_parameter(field.name, getattr(defaults, field.name))
        self.parameters_reset.emit(ParameterCategory.PREPROCESSING)

    def reset_displacement_parameters(self) -> None:
        """Reset displacement parameters to defaults"""
        defaults = UnifiedParameters()
        for field in fields(DisplacementParameters):
            self.set_parameter(field.name, getattr(defaults, field.name))
        self.parameters_reset.emit(ParameterCategory.DISPLACEMENT)

    def reset_force_parameters(self) -> None:
        """Reset force parameters to defaults"""
        defaults = UnifiedParameters()
        for field in fields(FTTCParameters):
            self.set_parameter(field.name, getattr(defaults, field.name))
        self.parameters_reset.emit(ParameterCategory.FORCE)

    def reset_stress_parameters(self) -> None:
        """Reset stress parameters to defaults"""
        defaults = UnifiedParameters()
        for field in fields(MSMParameters):
            self.set_parameter(field.name, getattr(defaults, field.name))
        self.parameters_reset.emit(ParameterCategory.STRESS)

    def load_from_file(self, filepath: Path) -> None:
        """Load parameters from file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Create new parameters instance with loaded values
        current_params = asdict(self._parameters)
        current_params.update(data)
        new_params = UnifiedParameters(**current_params)

        # Update all parameters
        self._update_all_parameters(new_params)

    def save_to_file(self, filepath: Path) -> None:
        """Save parameters to file"""
        data = asdict(self._parameters)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def _update_all_parameters(self, new_params: UnifiedParameters) -> None:
        """Update all parameters with validation"""
        old_values = asdict(self._parameters)
        new_values = asdict(new_params)

        for name, new_value in new_values.items():
            if old_values.get(name) != new_value:
                self.set_parameter(name, new_value)

    def validate_all_parameters(self) -> Tuple[bool, str]:
        """Validate all parameters using service validation methods"""
        # Check preprocessing parameters
        preproc_params = self.get_preprocessing_parameters()
        valid, msg = PreprocessingService.validate_parameters(preproc_params)
        if not valid:
            return False, f"Preprocessing parameters invalid: {msg}"

        # Check displacement parameters
        disp_params = self.get_displacement_parameters()
        valid, msg = DisplacementService.validate_parameters(disp_params)
        if not valid:
            return False, f"Displacement parameters invalid: {msg}"

        # Check force parameters
        force_params = self.get_fttc_parameters()
        valid, msg = FTTCService.validate_parameters(force_params)
        if not valid:
            return False, f"Force parameters invalid: {msg}"

        # Check stress parameters
        stress_params = self.get_msm_parameters()
        valid, msg = MSMService.validate_parameters(stress_params)
        if not valid:
            return False, f"Stress parameters invalid: {msg}"

        return True, ""