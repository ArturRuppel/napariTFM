from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, Callable, Set, Optional, Tuple
from enum import Enum, auto
import math
import yaml
from pathlib import Path
from qtpy.QtCore import QObject, Signal
from napariTFM.backend.parameter_dataclasses import PreprocessingParameters, DisplacementParameters, FTTCParameters, MSMParameters, UnifiedParameters
from napariTFM.backend.parameter_validation import (
    validate_displacement_parameters,
    validate_fttc_parameters,
    validate_msm_parameters,
    validate_preprocessing_parameters,
)


class ParameterCategory(Enum):
    """Enum defining different categories of parameters."""
    GENERAL = auto()
    PREPROCESSING = auto()
    DISPLACEMENT = auto()
    FORCE = auto()
    STRESS = auto()
    VISUALIZATION = auto()


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
        value = getattr(self._parameters, name)

        # Special handling for gel_height
        if name == 'gel_height':
            # Convert None or infinity to 0 for UI display
            if value is None or value == float('inf'):
                return 0
        return value

    def get_ui_parameter(self, name: str) -> Any:
        """Get a parameter value converted for display in UI controls."""
        value = self.get_parameter(name)
        if name == 'young_modulus':
            return value / 1000
        if name == 'regularization':
            return math.log10(value)
        if name == 'gel_height':
            return value
        return value

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value and trigger callbacks"""
        if not hasattr(self._parameters, name):
            raise ValueError(f"Unknown parameter: {name}")

        # Special handling for gel_height
        if name == 'gel_height':
            # Convert 0 to None for internal storage
            if value == 0 or value == float('inf'):
                value = None

        current_value = getattr(self._parameters, name)
        if current_value != value:
            setattr(self._parameters, name, value)

            # Trigger callbacks
            for callback in self._callbacks[name]:
                callback(value)

            # Emit signal
            self.parameter_changed.emit(name, value)

    def set_ui_parameter(self, name: str, value: Any) -> None:
        """Set a parameter from a UI control value, converting to internal units."""
        if name == 'young_modulus':
            value = value * 1000
        elif name == 'regularization':
            value = 10 ** value
        self.set_parameter(name, value)

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

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of all parameters with their current values
        """
        # Start with all parameters from the dataclass
        parameters = {}

        # Get all fields from UnifiedParameters
        for field in fields(self._parameters):
            value = getattr(self._parameters, field.name)

            # Handle special cases
            if field.name == 'young_modulus':
                # Store in Pa even though it's displayed in kPa
                parameters[field.name] = value
            elif field.name == 'gel_height' and value is None:
                # Convert None to 0 for infinity
                parameters[field.name] = 0
            elif field.name == 'regularization':
                # Store actual value, not log10
                parameters[field.name] = value
            else:
                parameters[field.name] = value

        return parameters

    def get_category_parameters(self, category: ParameterCategory) -> Dict[str, Any]:
        """
        Get parameters for a specific category.

        Args:
            category: ParameterCategory enum value

        Returns:
            Dict[str, Any]: Dictionary of parameters for the specified category
        """
        category_parameters = {}

        # Map categories to their parameter prefixes or names
        category_mappings = {
            ParameterCategory.GENERAL: ['pixel_size', 'frame_interval'],
            ParameterCategory.PREPROCESSING: [
                'rolling_ball_radius', 'min_intensity_percentile', 'max_intensity_percentile',
                'gaussian_sigma', 'cell_min_intensity_percentile',
                'cell_max_intensity_percentile', 'cell_gaussian_sigma',
                'registration_mode'
            ],
            ParameterCategory.DISPLACEMENT: [
                'nscales', 'inner_iterations',
                'median_filtering', 'downscale_factor',
                'disp_vector_stride', 'disp_arrow_scale', 'd_max'
            ],
            ParameterCategory.FORCE: [
                'young_modulus', 'poisson_ratio_substrate', 'gel_height',
                'lanczos_exp', 'regularization', 'auto_gcv',
                'force_vector_stride', 'force_arrow_scale', 'f_max'
            ],
            ParameterCategory.STRESS: [
                'density_factor', 'mesh_algorithm', 'use_optimization',
                'poisson_ratio_cells', 'max_stress'
            ],
            ParameterCategory.VISUALIZATION: [
                'save_bead_overlay', 'save_displacement_map', 'save_force_map',
                'save_force_cell_overlay', 'save_sigma_xx', 'save_sigma_yy',
                'save_normal_stress', 'save_mesh', 'show_vectors', 'show_colormap'
            ]
        }

        # Get parameters for the requested category
        if category in category_mappings:
            for param_name in category_mappings[category]:
                if hasattr(self._parameters, param_name):
                    value = getattr(self._parameters, param_name)
                    # Apply any necessary conversions
                    if param_name == 'young_modulus':
                        # Store in Pa even though it's displayed in kPa
                        category_parameters[param_name] = value
                    elif param_name == 'gel_height' and value is None:
                        # Convert None to 0 for infinity
                        category_parameters[param_name] = 0
                    elif param_name == 'regularization':
                        # Store actual value, not log10
                        category_parameters[param_name] = value
                    else:
                        category_parameters[param_name] = value

        return category_parameters

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
        valid_parameter_names = {field.name for field in fields(UnifiedParameters)}
        current_params.update({
            name: value
            for name, value in data.items()
            if name in valid_parameter_names
        })
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
        valid, msg = validate_preprocessing_parameters(preproc_params)
        if not valid:
            return False, f"Preprocessing parameters invalid: {msg}"

        # Check displacement parameters
        disp_params = self.get_displacement_parameters()
        valid, msg = validate_displacement_parameters(disp_params)
        if not valid:
            return False, f"Displacement parameters invalid: {msg}"

        # Check force parameters
        force_params = self.get_fttc_parameters()
        valid, msg = validate_fttc_parameters(force_params)
        if not valid:
            return False, f"Force parameters invalid: {msg}"

        # Check stress parameters
        stress_params = self.get_msm_parameters()
        valid, msg = validate_msm_parameters(stress_params)
        if not valid:
            return False, f"Stress parameters invalid: {msg}"

        return True, ""
