from dataclasses import asdict, fields
from typing import Dict, Any
from enum import Enum, auto
import math
from qtpy.QtCore import QObject, Signal
from napariTFM.backend.parameter_dataclasses import DisplacementParameters, FTTCParameters, StressParameters, UnifiedParameters


class ParameterCategory(Enum):
    """Enum defining different categories of parameters."""
    GENERAL = auto()
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
        """Get a parameter value converted for display in UI controls.

        Young's modulus is shown in kPa (stored in Pa); the two regularizers are
        shown as a base-10 exponent (stored as the actual value). gel_height's
        None->0 display mapping is already applied by get_parameter, so it needs
        no conversion here — everything else is passed through unchanged.
        """
        value = self.get_parameter(name)
        if name == 'young_modulus':
            return value / 1000
        if name in ('regularization', 'bism_regularization'):
            return math.log10(value)
        return value

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value and emit parameter_changed."""
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
            self.parameter_changed.emit(name, value)

    def set_ui_parameter(self, name: str, value: Any) -> None:
        """Set a parameter from a UI control value, converting to internal units."""
        if name == 'young_modulus':
            value = value * 1000
        elif name in ('regularization', 'bism_regularization'):
            value = 10 ** value
        self.set_parameter(name, value)

    def get_displacement_parameters(self) -> DisplacementParameters:
        """Get parameters for displacement service"""
        return self._parameters.to_displacement_parameters()

    def get_fttc_parameters(self) -> FTTCParameters:
        """Get parameters for FTTC service"""
        return self._parameters.to_fttc_parameters()

    def get_stress_parameters(self) -> StressParameters:
        """Get parameters for the stress (BISM) service"""
        return self._parameters.to_stress_parameters()

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of all parameters with their current values
        """
        # Values are stored in internal units already (Pa, actual regularizer
        # value — not the kPa / log10 the UI shows), so serialization is a
        # straight field copy. The sole exception is gel_height, whose None
        # ("infinite thickness") sentinel is written as 0 for YAML/JSON portability.
        parameters = {}
        for field in fields(self._parameters):
            value = getattr(self._parameters, field.name)
            if field.name == 'gel_height' and value is None:
                value = 0
            parameters[field.name] = value
        return parameters

    def reset_all_parameters(self) -> None:
        """Reset all parameters to default values"""
        new_params = UnifiedParameters()
        self._update_all_parameters(new_params)
        for category in ParameterCategory:
            self.parameters_reset.emit(category)

    def _update_all_parameters(self, new_params: UnifiedParameters) -> None:
        """Update all parameters with validation"""
        old_values = asdict(self._parameters)
        new_values = asdict(new_params)

        for name, new_value in new_values.items():
            if old_values.get(name) != new_value:
                self.set_parameter(name, new_value)
