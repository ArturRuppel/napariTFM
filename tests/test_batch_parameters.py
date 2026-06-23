import sys
import types
from types import SimpleNamespace

import yaml
from qtpy.QtWidgets import QApplication, QGroupBox

qtrangeslider = types.ModuleType("qtrangeslider")
qtrangeslider.QRangeSlider = object
sys.modules.setdefault("qtrangeslider", qtrangeslider)
sys.modules.setdefault("gmsh", types.ModuleType("gmsh"))
sys.modules.setdefault("solidspy", types.ModuleType("solidspy"))
sys.modules.setdefault("solidspy.assemutil", types.ModuleType("solidspy.assemutil"))
sys.modules.setdefault("solidspy.postprocesor", types.ModuleType("solidspy.postprocesor"))

from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.utilities.parameter_manager import ParameterManager
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


def _app():
    return QApplication.instance() or QApplication([])


def test_batch_widget_does_not_create_duplicate_analysis_parameter_controls():
    app = _app()
    widget = BatchAnalysisWidget(None, object(), ParameterManager(), object())
    widget.show()
    app.processEvents()

    analysis_parameter_titles = {
        "General Parameters",
        "Preprocessing Parameters",
        "Farneback Displacement Parameters",
        "Force Parameters",
        "Stress Parameters",
    }

    visible_group_titles = {
        group.title()
        for group in widget.findChildren(QGroupBox)
        if group.isVisibleTo(widget)
    }

    assert widget.parameter_spins == {}
    assert widget.parameter_combos == {}
    assert widget.parameter_checks == {}
    assert analysis_parameter_titles.isdisjoint(visible_group_titles)


def test_load_config_writes_parameters_directly_to_parameter_manager(tmp_path):
    app = _app()
    manager = ParameterManager()
    widget = BatchAnalysisWidget(None, object(), manager, object())
    widget.show()
    app.processEvents()

    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "root_folders": [],
                "input_files": {
                    "beads": "beads.tif",
                    "reference": "reference.tif",
                    "cells": "",
                },
                "analysis_steps": {"displacement": True, "force": False},
                "visualizations": {"displacement_map": True, "force_map": False},
                "parameters": {
                    "pixel_size": 0.33,
                    "young_modulus": 9000,
                    "regularization": 1e-6,
                    "registration_mode": "Rigid",
                    "mesh_algorithm": "Delaunay",
                    "auto_gcv": True,
                    "tau": 0.25,
                },
            }
        )
    )

    widget.load_config_from_yaml(str(config_path))

    parameters = manager.get_all_parameters()
    assert parameters["pixel_size"] == 0.33
    assert parameters["young_modulus"] == 9000
    assert parameters["regularization"] == 1e-6
    assert parameters["registration_mode"] == "rigid"
    assert parameters["mesh_algorithm"] == "Delaunay"
    assert parameters["auto_gcv"] is True


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


def test_batch_keeps_batch_specific_controls_visible_after_parameter_slimdown():
    app = _app()
    widget = BatchAnalysisWidget(None, object(), ParameterManager(), object())
    widget.show()
    app.processEvents()

    assert widget.save_config_btn.isVisibleTo(widget)
    assert widget.load_config_btn.isVisibleTo(widget)
    assert widget.run_analysis_btn.isVisibleTo(widget)
    assert widget.folder_list_widget.isVisibleTo(widget)


def test_batch_config_generation_does_not_read_duplicate_parameter_widgets():
    fake = SimpleNamespace(
        folder_list_widget=_List(),
        file_inputs={"beads": _Text("beads.tif"), "reference": _Text("ref.tif"), "cells": _Text("")},
        analysis_checkboxes={"preprocess": _Check(True)},
        visualization_checkboxes={
            "bead_overlay": _Check(False),
            "displacement_map": _Check(False),
            "force_map": _Check(False),
            "force_cell_overlay": _Check(False),
            "sigma_xx": _Check(False),
            "sigma_yy": _Check(False),
            "normal_stress": _Check(False),
            "mesh": _Check(False),
        },
        parameter_manager=_Manager(),
        parameter_spins={"young_modulus": object()},
        parameter_combos={"mesh_algorithm": object()},
        parameter_checks={"auto_gcv": object()},
    )

    config = BatchAnalysisWidget._generate_config(fake)

    assert config["parameters"]["young_modulus"] == 9000
    assert config["parameters"]["mesh_algorithm"] == "Frontal-Del."


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
            "nscales": 10,
            "inner_iterations": 10,
            "median_filtering": 9,
            "downscale_factor": 4,
            "pixel_size": 0.1,
            "frame_interval": 1.0,
            "d_max": 1.0,
            "disp_vector_stride": 20,
            "disp_arrow_scale": 1.0,
        }
    }

    params = analysis._create_displacement_parameters()

    assert params.nscales == 10
    assert params.inner_iterations == 10
    assert params.median_filtering == 9


def test_batch_unified_parameters_ignore_unknown_keys_and_default_missing():
    # Config from an older version carries retired keys and omits some current
    # ones; the unifier must drop the strays and fall back to defaults.
    analysis = BatchAnalysis.__new__(BatchAnalysis)
    analysis.config = {
        "parameters": {
            "rolling_ball_radius": 7,
            "density_factor": 0.02,
            "outer_iterations": 99,   # retired field
            "threshold": 0.5,         # never existed on UnifiedParameters
        }
    }

    unified = analysis._unified_parameters()
    assert unified.rolling_ball_radius == 7
    assert unified.density_factor == 0.02
    assert not hasattr(unified, "outer_iterations")
    assert not hasattr(unified, "threshold")
    # Missing keys default (UnifiedParameters default downscale_factor is 4).
    assert unified.downscale_factor == 4

    # The retired/unknown keys must not leak into any backend dataclass, and the
    # otherwise-undertested preprocessing/MSM mappings flow through correctly.
    assert analysis._create_preprocessing_parameters().rolling_ball_radius == 7
    assert analysis._create_msm_parameters().density_factor == 0.02
