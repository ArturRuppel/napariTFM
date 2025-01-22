from typing import Dict, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import yaml
from pathlib import Path
from qtpy.QtCore import QObject, Signal


class ParameterCategory(Enum):
    """Enum defining different categories of parameters."""
    GENERAL = auto()
    PREPROCESSING = auto()
    DISPLACEMENT = auto()
    FORCE = auto()
    STRESS = auto()
    VISUALIZATION = auto()


@dataclass
class Parameter:
    """Class to store parameter metadata and value."""
    value: Any
    category: ParameterCategory
    callbacks: Set[Callable] = field(default_factory=set)


class ParameterManager(QObject):
    """Centralized manager for TFM analysis parameters."""

    # Signal emitted when any parameter changes
    parameter_changed = Signal(str, object)  # (parameter_name, new_value)

    def __init__(self):
        super().__init__()
        self._parameters: Dict[str, Parameter] = {}
        self._initialize_default_parameters()

    def _initialize_default_parameters(self):
        """Initialize all parameters with their default values."""
        # General parameters
        self._add_parameter('pixel_size', 1.0, ParameterCategory.GENERAL)
        self._add_parameter('frame_interval', 1.0, ParameterCategory.GENERAL)

        # Preprocessing parameters
        preproc_params = {
            'min_intensity': 0.0,
            'max_intensity': 100.0,
            'gaussian_sigma': 0.0,
            'cell_min_intensity': 0.0,
            'cell_max_intensity': 100.0,
            'cell_gaussian_sigma': 0.0,
            'registration_mode': 'none'
        }
        for name, value in preproc_params.items():
            self._add_parameter(name, value, ParameterCategory.PREPROCESSING)

        # Displacement parameters
        disp_params = {
            'tau': 0.25,
            'lambda_': 0.4,
            'theta': 0.3,
            'nscales': 3,
            'warps': 3,
            'epsilon': 0.01,
            'inner_iterations': 15,
            'outer_iterations': 5,
            'scale_step': 0.5,
            'median_filtering': 5,
            'downscale_factor': 1,
            'disp_vector_stride': 20,
            'disp_arrow_scale': 1.0,
            'd_max': 5.0
        }
        for name, value in disp_params.items():
            self._add_parameter(name, value, ParameterCategory.DISPLACEMENT)

        # Force parameters
        force_params = {
            'young_modulus': 10000.0,  # 10 kPa in Pa
            'poisson_ratio': 0.49,
            'gel_height': None,  # None means infinite
            'lanczos_exp': 1,
            'regularization': 1e-17,
            'auto_gcv': False,
            'force_vector_stride': 20,
            'force_arrow_scale': 1.0,
            'f_max': 1000.0
        }
        for name, value in force_params.items():
            self._add_parameter(name, value, ParameterCategory.FORCE)

        # Stress parameters
        stress_params = {
            'threshold': 0.0,
            'dilation': 10,
            'smoothing_sigma': 10.0,
            'density_factor': 0.025,
            'mesh_algorithm': 'frontal-del.',
            'use_optimization': True,
            'max_stress': 1.0
        }
        for name, value in stress_params.items():
            self._add_parameter(name, value, ParameterCategory.STRESS)

        # Visualization parameters
        viz_params = {
            'save_bead_overlay': False,
            'save_displacement_map': False,
            'save_force_map': False,
            'save_force_cell_overlay': False,
            'save_sigma_xx': False,
            'save_sigma_yy': False,
            'save_shear': False,
            'save_normal_stress': False
        }
        for name, value in viz_params.items():
            self._add_parameter(name, value, ParameterCategory.VISUALIZATION)

    def _add_parameter(self, name: str, value: Any, category: ParameterCategory):
        """Add a new parameter to the manager."""
        self._parameters[name] = Parameter(value=value, category=category)

    def get_value(self, name: str) -> Any:
        """Get the current value of a parameter."""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        return self._parameters[name].value

    def set_value(self, name: str, value: Any):
        """Set the value of a parameter and notify observers."""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")

        param = self._parameters[name]
        if param.value != value:
            param.value = value
            # Notify observers
            for callback in param.callbacks:
                callback(value)
            # Emit signal
            self.parameter_changed.emit(name, value)

    def register_callback(self, name: str, callback: Callable):
        """Register a callback for parameter changes."""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        self._parameters[name].callbacks.add(callback)

    def unregister_callback(self, name: str, callback: Callable):
        """Unregister a callback for parameter changes."""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        self._parameters[name].callbacks.discard(callback)

    def get_category_parameters(self, category: ParameterCategory) -> Dict[str, Any]:
        """Get all parameters belonging to a specific category."""
        return {
            name: param.value
            for name, param in self._parameters.items()
            if param.category == category
        }

    def load_from_file(self, filepath: Path):
        """Load parameters from a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Update parameters from each section
        for section, params in data.items():
            try:
                category = ParameterCategory[section.upper()]
                for name, value in params.items():
                    if name in self._parameters:
                        self.set_value(name, value)
            except KeyError:
                continue  # Skip unknown sections/parameters

    def save_to_file(self, filepath: Path):
        """Save parameters to a YAML file."""
        # Group parameters by category
        data = {}
        for category in ParameterCategory:
            category_params = self.get_category_parameters(category)
            if category_params:
                data[category.name.lower()] = category_params

        # Save to file
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def reset_to_defaults(self):
        """Reset all parameters to their default values."""
        self._parameters.clear()
        self._initialize_default_parameters()