"""Layer-isolation + parameter-panel guards.

The interactive per-stage Run buttons stream through ``begin_*_stream`` (not the
preview path), so the §4 viewer takeover — hide every layer that isn't this
stage's — has to live in ``begin_*_stream`` itself. These lock that isolation
onto the streaming entry points, and pin the parameter panel's grid layout.

(Formerly part of ``test_preprocessing_ui_redesign.py``; the preprocessing-stage
tests were dropped when that stage was removed, leaving these stage-agnostic
ones.)
"""
import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.visualization_manager import VisualizationManager


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeSelectionEvents:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _FakeSelection:
    def __init__(self):
        self.active = None
        self.events = type("Events", (), {"active": _FakeSelectionEvents()})()


class _FakeLayer:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.visible = True
        self.colormap = None
        self.blending = None


class _FakeLayers(list):
    def __init__(self):
        super().__init__()
        self.selection = _FakeSelection()
        self.events = type("Events", (), {"removed": _FakeSelectionEvents()})()

    def __contains__(self, item):
        if isinstance(item, str):
            return any(layer.name == item for layer in self)
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)

    def remove(self, item):
        if isinstance(item, str):
            item = self[item]
        super().remove(item)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayers()
        self.dims = type(
            "Dims",
            (),
            {"current_step": (0,), "events": type("Events", (), {"current_step": _FakeSelectionEvents()})()},
        )()

    def add_image(self, data, **kwargs):
        layer = _FakeLayer(kwargs["name"], data)
        layer.colormap = kwargs.get("colormap")
        layer.blending = kwargs.get("blending")
        layer.visible = kwargs.get("visible", True)
        self.layers.append(layer)
        return layer


def test_isolate_layers_shows_only_kept_layers_hiding_the_rest():
    viewer = _FakeViewer()
    manager = VisualizationManager(viewer, DataManager())
    for name in ("Beads", "Reference", "Force Magnitude"):
        viewer.add_image(np.ones((2, 2), dtype=np.float32), name=name)

    manager.isolate_layers(["Beads"])

    assert viewer.layers["Beads"].visible is True
    assert viewer.layers["Reference"].visible is False
    assert viewer.layers["Force Magnitude"].visible is False


def test_begin_vector_field_stream_isolates_to_its_stage():
    viewer = _FakeViewer()
    manager = VisualizationManager(viewer, DataManager())
    for name in ("Beads", "Cells", "Force Magnitude"):
        viewer.add_image(np.ones((3, 2, 2), dtype=np.float32), name=name)

    manager.begin_vector_field_stream(
        "displacement", 3,
        {"v_max": 5.0, "vector_stride": 2, "arrow_scale": 1.0, "downscale_factor": 1},
    )

    assert viewer.layers["Beads"].visible is False
    assert viewer.layers["Cells"].visible is False
    assert viewer.layers["Force Magnitude"].visible is False


def test_begin_stress_stream_isolates_to_stress_layers():
    viewer = _FakeViewer()
    manager = VisualizationManager(viewer, DataManager())
    for name in ("Beads", "Force Magnitude", "Average Normal Stress"):
        viewer.add_image(np.ones((3, 2, 2), dtype=np.float32), name=name)

    manager.begin_stress_stream(num_frames=3, max_stress=1.0, downscale_factor=1)

    assert viewer.layers["Beads"].visible is False
    assert viewer.layers["Force Magnitude"].visible is False
    assert viewer.layers["Average Normal Stress"].visible is True


def test_isolate_layers_keeps_colorbar_legend_visible():
    viewer = _FakeViewer()
    manager = VisualizationManager(viewer, DataManager())
    for name in (
        "Displacement Magnitude",
        "Displacement Vectors",
        "Displacement (µm) Colorbar",
        "Displacement (µm) Colorbar Label",
    ):
        viewer.add_image(np.ones((2, 2), dtype=np.float32), name=name)
    # Pretend the colorbar manager has rendered its legend layers for this preview.
    manager.colorbar_manager._layer_names = [
        "Displacement (µm) Colorbar",
        "Displacement (µm) Colorbar Label",
    ]

    manager.isolate_layers(["Displacement Magnitude", "Displacement Vectors"])

    # The previewed stage layers stay on, and the legend rides along with them
    # instead of being hidden as an "other" layer.
    assert viewer.layers["Displacement Magnitude"].visible is True
    assert viewer.layers["Displacement Vectors"].visible is True
    assert viewer.layers["Displacement (µm) Colorbar"].visible is True
    assert viewer.layers["Displacement (µm) Colorbar Label"].visible is True


def test_param_panel_uses_section_grid_not_groupbox(app):
    from qtpy.QtWidgets import QGridLayout, QGroupBox
    from napariTFM.utilities.parameter_manager import ParameterManager
    from napariTFM.widgets._widget import WorkflowParameterPanel

    panel = WorkflowParameterPanel(ParameterManager(), section_titles=("Displacement",))

    assert panel.findChildren(QGroupBox) == []
    assert panel.findChild(QGridLayout) is not None


def test_param_panel_packs_two_pairs_per_row(app):
    from qtpy.QtWidgets import QGridLayout
    from napariTFM.utilities.parameter_manager import ParameterManager
    from napariTFM.widgets._widget import WorkflowParameterPanel

    # Displacement has several params; row 0 is the header, row 1 holds the first two.
    panel = WorkflowParameterPanel(ParameterManager(), section_titles=("Displacement",))
    grid = panel.findChild(QGridLayout)

    assert grid.itemAtPosition(1, 0) is not None
    assert grid.itemAtPosition(1, 2) is not None


def test_param_panel_still_registers_controls(app):
    from napariTFM.utilities.parameter_manager import ParameterManager
    from napariTFM.widgets._widget import WorkflowParameterPanel

    panel = WorkflowParameterPanel(ParameterManager(), section_titles=("Displacement",))

    assert "piv_window" in panel.parameter_controls
    assert "d_max" in panel.parameter_controls
