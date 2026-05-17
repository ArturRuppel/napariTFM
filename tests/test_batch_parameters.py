import sys
import types
from types import SimpleNamespace

qtrangeslider = types.ModuleType("qtrangeslider")
qtrangeslider.QRangeSlider = object
sys.modules.setdefault("qtrangeslider", qtrangeslider)
sys.modules.setdefault("gmsh", types.ModuleType("gmsh"))
sys.modules.setdefault("solidspy", types.ModuleType("solidspy"))
sys.modules.setdefault("solidspy.assemutil", types.ModuleType("solidspy.assemutil"))
sys.modules.setdefault("solidspy.postprocesor", types.ModuleType("solidspy.postprocesor"))

from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.widgets.batch_analysis_widget import BatchAnalysisWidget


class _Text:
    def __init__(self, value=""):
        self._value = value

    def text(self):
        return self._value


class _Check:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _List:
    def count(self):
        return 1

    def item(self, index):
        return _Text("/tmp/example")


class _Manager:
    def __init__(self):
        self.parameters = {
            "pixel_size": 0.33,
            "frame_interval": 2.5,
            "young_modulus": 9000,
            "regularization": 1e-6,
            "gel_height": None,
            "registration_mode": "translation",
            "mesh_algorithm": "Frontal-Del.",
        }
        self.set_calls = []

    def get_all_parameters(self):
        return dict(self.parameters)

    def set_ui_parameter(self, name, value):
        self.set_calls.append((name, value))

    def set_parameter(self, name, value):
        self.set_calls.append((name, value))


def test_generate_config_uses_parameter_manager_values():
    fake = SimpleNamespace(
        folder_list_widget=_List(),
        file_inputs={"beads": _Text("beads.tif"), "reference": _Text("ref.tif"), "cells": _Text("")},
        analysis_checkboxes={"preprocess": _Check(True), "calculate_metrics": _Check(False)},
        visualization_checkboxes={
            "bead_overlay": _Check(True),
            "displacement_map": _Check(False),
            "force_map": _Check(False),
            "force_cell_overlay": _Check(False),
            "sigma_xx": _Check(False),
            "sigma_yy": _Check(False),
            "normal_stress": _Check(False),
            "mesh": _Check(False),
        },
        parameter_manager=_Manager(),
        parameter_spins={},
        parameter_combos={},
        parameter_checks={},
    )

    config = BatchAnalysisWidget._generate_config(fake)

    assert config["parameters"]["pixel_size"] == 0.33
    assert config["parameters"]["young_modulus"] == 9000
    assert config["parameters"]["mesh_algorithm"] == "Frontal-Del."


def test_sync_parameters_preserves_mesh_algorithm_case():
    manager = _Manager()
    combo = SimpleNamespace(currentText=lambda: "Frontal-Del.")
    fake = SimpleNamespace(
        blockSignals=lambda value: None,
        parameter_spins={},
        parameter_combos={"mesh_algorithm": combo},
        parameter_checks={},
        parameter_manager=manager,
    )

    BatchAnalysisWidget._sync_parameters_with_manager(fake)

    assert ("mesh_algorithm", "Frontal-Del.") in manager.set_calls
    assert ("mesh_algorithm", "frontal-del.") not in manager.set_calls


def test_batch_fttc_parameters_honor_auto_gcv():
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {
        "parameters": {
            "young_modulus": 5000,
            "poisson_ratio_substrate": 0.5,
            "gel_height": None,
            "lanczos_exp": 1,
            "regularization": 1e-4,
            "auto_gcv": True,
            "force_vector_stride": 20,
            "force_arrow_scale": 1.0,
            "f_max": 500.0,
            "frame_interval": 1.0,
            "pixel_size": 0.1,
            "downscale_factor": 4,
        }
    }

    params = analysis._create_fttc_parameters()

    assert params.auto_gcv is True


def test_batch_displacement_parameters_do_not_require_removed_tvl1_keys():
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {
        "parameters": {
            "nscales": 3,
            "inner_iterations": 15,
            "outer_iterations": 5,
            "median_filtering": 5,
            "downscale_factor": 4,
            "pixel_size": 0.1,
            "frame_interval": 1.0,
            "d_max": 1.0,
            "disp_vector_stride": 20,
            "disp_arrow_scale": 1.0,
        }
    }

    params = analysis._create_displacement_parameters()

    assert params.nscales == 3
    assert params.inner_iterations == 15
    assert params.outer_iterations == 5
