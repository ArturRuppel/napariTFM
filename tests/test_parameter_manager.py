import importlib
import sys
import types

import pytest

sys.modules.setdefault("gmsh", types.ModuleType("gmsh"))
sys.modules.setdefault("solidspy", types.ModuleType("solidspy"))
sys.modules.setdefault("solidspy.assemutil", types.ModuleType("solidspy.assemutil"))
sys.modules.setdefault("solidspy.postprocesor", types.ModuleType("solidspy.postprocesor"))
qtrangeslider = types.ModuleType("qtrangeslider")
qtrangeslider.QRangeSlider = object
sys.modules.setdefault("qtrangeslider", qtrangeslider)

from napariTFM.backend.parameter_dataclasses import (
    DisplacementParameters,
    FTTCParameters,
    MSMParameters,
    PreprocessingParameters,
)
from napariTFM.services.displacement_service import DisplacementService
from napariTFM.services.fttc_service import FTTCService
from napariTFM.services.msm_service import MSMService
from napariTFM.services.preprocessing_service import PreprocessingService
from napariTFM.utilities.parameter_manager import ParameterCategory, ParameterManager


def test_ui_parameter_conversions_round_trip():
    manager = ParameterManager()

    manager.set_ui_parameter("young_modulus", 7.5)
    assert manager.get_parameter("young_modulus") == 7500
    assert manager.get_ui_parameter("young_modulus") == 7.5

    manager.set_ui_parameter("regularization", -3)
    assert manager.get_parameter("regularization") == pytest.approx(1e-3)
    assert manager.get_ui_parameter("regularization") == pytest.approx(-3)

    manager.set_ui_parameter("gel_height", 0)
    assert manager.get_parameter("gel_height") == 0
    assert manager._parameters.gel_height is None
    assert manager.get_ui_parameter("gel_height") == 0

    manager.set_ui_parameter("gel_height", 12.5)
    assert manager.get_parameter("gel_height") == 12.5
    assert manager.get_ui_parameter("gel_height") == 12.5


def test_preprocessing_category_uses_unified_parameter_field_names():
    manager = ParameterManager()

    params = manager.get_category_parameters(ParameterCategory.PREPROCESSING)

    assert "min_intensity_percentile" in params
    assert "max_intensity_percentile" in params
    assert "cell_min_intensity_percentile" in params
    assert "cell_max_intensity_percentile" in params
    assert "min_intensity" not in params
    assert "max_intensity" not in params


def test_parameter_manager_validation_does_not_import_services(monkeypatch):
    for module_name in list(sys.modules):
        if module_name.startswith("napariTFM.services"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    module = importlib.reload(importlib.import_module("napariTFM.utilities.parameter_manager"))

    imported_service_modules = [
        name for name in sys.modules
        if name.startswith("napariTFM.services")
    ]
    assert imported_service_modules == []

    manager = module.ParameterManager()
    assert manager.validate_all_parameters() == (True, "")


def test_service_validate_parameters_delegates_compatible_results():
    assert PreprocessingService.validate_parameters(
        PreprocessingParameters(min_intensity_percentile=80, max_intensity_percentile=20)
    ) == (False, "Invalid intensity percentile range")

    assert DisplacementService.validate_parameters(
        DisplacementParameters(tau=0)
    ) == (False, "tau must be positive")

    assert FTTCService.validate_parameters(
        FTTCParameters(young_modulus=0)
    ) == (False, "Young's modulus must be positive")

    assert MSMService.validate_parameters(
        MSMParameters(density_factor=0.001)
    ) == (False, "Density factor is too low (< 0.005). This may lead to numerical instabilities.")
