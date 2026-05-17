import sys
import types

from qtpy.QtWidgets import QApplication, QGroupBox, QLabel

sys.modules.setdefault("gmsh", types.ModuleType("gmsh"))
sys.modules.setdefault("solidspy", types.ModuleType("solidspy"))
sys.modules.setdefault("solidspy.assemutil", types.ModuleType("solidspy.assemutil"))
sys.modules.setdefault("solidspy.postprocesor", types.ModuleType("solidspy.postprocesor"))
qtrangeslider = types.ModuleType("qtrangeslider")
qtrangeslider.QRangeSlider = object
sys.modules.setdefault("qtrangeslider", qtrangeslider)

from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.widgets.displacement_analysis_widget import DisplacementParameterPanel


STALE_TVL1_PARAMETERS = {"tau", "lambda_", "theta", "warps", "epsilon", "scale_step"}
ACTIVE_FARNEBACK_PARAMETERS = {
    "nscales",
    "inner_iterations",
    "median_filtering",
    "downscale_factor",
}


def _app():
    return QApplication.instance() or QApplication([])


def _label_texts(widget):
    return {label.text() for label in widget.findChildren(QLabel)}


def _group_titles(widget):
    return {group.title() for group in widget.findChildren(QGroupBox)}


def test_displacement_panel_exposes_farneback_parameters_not_tvl1_controls():
    app = _app()

    panel = DisplacementParameterPanel(ParameterManager())
    panel.show()
    app.processEvents()

    assert STALE_TVL1_PARAMETERS.isdisjoint(panel.parameter_spins)
    assert ACTIVE_FARNEBACK_PARAMETERS.issubset(panel.parameter_spins)
    assert "outer_iterations" not in panel.parameter_spins
    assert any("Farneback" in title for title in _group_titles(panel))
    assert "Lambda:" not in _label_texts(panel)
    assert "Farneback Iterations:" in _label_texts(panel)
    assert "Window Size:" in _label_texts(panel)
