import importlib
import sys
import types

import pytest

qtrangeslider = types.ModuleType("qtrangeslider")
qtrangeslider.QRangeSlider = object
sys.modules.setdefault("qtrangeslider", qtrangeslider)

from napariTFM.backend.parameter_dataclasses import (
    FTTCParameters,
    StressParameters,
)
from napariTFM.backend.parameter_validation import validate_fttc_parameters
from napariTFM.utilities.parameter_manager import ParameterManager


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


def test_all_parameters_omit_tvl1_only_parameters():
    manager = ParameterManager()

    params = manager.get_all_parameters()

    assert STALE_TVL1_PARAMETERS.isdisjoint(params)
    assert {"disp_method", "disp_device", "piv_window", "piv_overlap", "piv_passes",
            "ilk_radius", "ilk_num_warp", "ffd_level_spacing", "ffd_num_levels",
            "ffd_metric"}.issubset(params)
    assert "outer_iterations" not in params


def test_removed_tvl1_fields_are_rejected():
    """Stale TVL1 knobs from old presets have no field on UnifiedParameters, so
    set_parameter rejects them — the guard that keeps a legacy load from
    reintroducing them (loads filter through valid field names)."""
    manager = ParameterManager()

    for stale_name in STALE_TVL1_PARAMETERS:
        with pytest.raises(ValueError, match=f"Unknown parameter: {stale_name}"):
            manager.get_parameter(stale_name)
        with pytest.raises(ValueError, match=f"Unknown parameter: {stale_name}"):
            manager.set_parameter(stale_name, 0.1)


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


def test_validation_helpers_return_compatible_results():
    assert validate_fttc_parameters(
        FTTCParameters(young_modulus=0)
    ) == (False, "Young's modulus must be positive")


def test_fttc_validation_ignores_visualization_only_params():
    """force_arrow_scale / f_max / force_vector_stride never enter the traction
    solve, so a bad value there must not block force computation."""
    assert validate_fttc_parameters(FTTCParameters(f_max=0)) == (True, "")
    assert validate_fttc_parameters(FTTCParameters(force_arrow_scale=0)) == (True, "")
    assert validate_fttc_parameters(FTTCParameters(force_vector_stride=0)) == (True, "")


def test_fttc_validation_skips_regularization_check_under_auto_gcv():
    """Under auto-GCV the manual regularization is unused, so reg<=0 is fine; with
    auto-GCV off it must still be rejected."""
    assert validate_fttc_parameters(
        FTTCParameters(auto_gcv=True, regularization=0.0)
    ) == (True, "")
    assert validate_fttc_parameters(
        FTTCParameters(auto_gcv=False, regularization=0.0)
    ) == (False, "Regularization parameter must be positive")


def test_stress_parameters_have_no_mask_fields():
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(StressParameters)}
    assert "threshold" not in field_names
    assert "dilation" not in field_names
    assert "smoothing_sigma" not in field_names
