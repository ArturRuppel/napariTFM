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
from napariTFM.backend.parameter_validation import (
    validate_displacement_parameters,
    validate_fttc_parameters,
    validate_msm_parameters,
    validate_preprocessing_parameters,
)
from napariTFM.utilities.parameter_manager import ParameterCategory, ParameterManager


STALE_TVL1_PARAMETERS = {"tau", "lambda_", "theta", "warps", "epsilon", "scale_step"}


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


def test_displacement_category_omits_tvl1_only_parameters():
    manager = ParameterManager()

    params = manager.get_category_parameters(ParameterCategory.DISPLACEMENT)

    assert STALE_TVL1_PARAMETERS.isdisjoint(params)
    assert {"nscales", "inner_iterations", "median_filtering"}.issubset(params)
    assert "outer_iterations" not in params


def test_loading_parameters_ignores_removed_tvl1_fields(tmp_path):
    config_path = tmp_path / "old-parameters.yml"
    config_path.write_text(
        "tau: 0.1\n"
        "lambda_: 0.2\n"
        "theta: 0.3\n"
        "warps: 7\n"
        "epsilon: 0.001\n"
        "scale_step: 0.8\n"
        "nscales: 4\n"
    )

    manager = ParameterManager()
    manager.load_from_file(config_path)

    assert manager.get_parameter("nscales") == 4
    for stale_name in STALE_TVL1_PARAMETERS:
        with pytest.raises(ValueError, match=f"Unknown parameter: {stale_name}"):
            manager.get_parameter(stale_name)


def test_parameter_manager_validation_does_not_import_services(monkeypatch):
    removed_package = ".".join(("napariTFM", "services"))

    for module_name in list(sys.modules):
        if module_name.startswith(removed_package):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    module = importlib.reload(importlib.import_module("napariTFM.utilities.parameter_manager"))

    imported_service_modules = [
        name for name in sys.modules
        if name.startswith(removed_package)
    ]
    assert imported_service_modules == []

    manager = module.ParameterManager()
    assert manager.validate_all_parameters() == (True, "")


def test_validation_helpers_return_compatible_results():
    assert validate_preprocessing_parameters(
        PreprocessingParameters(min_intensity_percentile=80, max_intensity_percentile=20)
    ) == (False, "Invalid intensity percentile range")

    assert validate_displacement_parameters(
        DisplacementParameters(nscales=0)
    ) == (False, "nscales must be at least 1")

    assert validate_fttc_parameters(
        FTTCParameters(young_modulus=0)
    ) == (False, "Young's modulus must be positive")

    assert validate_msm_parameters(
        MSMParameters(density_factor=0.001)
    ) == (False, "Density factor is too low (< 0.005). This may lead to numerical instabilities.")
