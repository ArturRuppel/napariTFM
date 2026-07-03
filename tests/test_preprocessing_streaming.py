"""Live-streaming preprocessing: frames must fill the output stacks in place,
the viewer must follow the frame being computed, and re-runs must preserve the
contrast/visibility the user set (and not hide unrelated layers)."""

import sys
import types

import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication

for module_name in [
    "napariTFM.utilities.visualization_manager",
    "napariTFM.widgets.preprocessing_widget",
]:
    sys.modules.pop(module_name, None)

for module_name in ["gmsh", "solidspy", "solidspy.assemutil",
                    "solidspy.postprocesor", "solidspy.solutil"]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

from napariTFM.utilities.visualization_manager import VisualizationManager
import napariTFM.widgets.preprocessing_widget as preprocessing_widget
from napariTFM.widgets.preprocessing_widget import PreprocessingController


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


# --- minimal napari fakes -------------------------------------------------

class _Blocker:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeLayer:
    def __init__(self, name, data, colormap=None, contrast_limits=None, visible=True):
        self.name = name
        self.data = data
        self.colormap = colormap
        self.blending = None
        self.visible = visible
        self.contrast_limits = list(contrast_limits) if contrast_limits else [0.0, 1.0]
        self.contrast_limits_range = list(self.contrast_limits)
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class _FakeLayers(list):
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


class _Sub:
    def connect(self, *_):
        pass

    def disconnect(self, *_):
        pass


class _FakeDims:
    def __init__(self):
        self.current_step = (0,)
        self.steps = []
        self.events = type("E", (), {"current_step": _Sub()})()

    def set_current_step(self, axis, value):
        self.steps.append((axis, value))
        step = list(self.current_step)
        if axis < len(step):
            step[axis] = value
        self.current_step = tuple(step)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayers()
        sel = type("Sel", (), {"active": None})()
        sel.events = type("E", (), {"active": _Sub()})()
        self.layers.selection = sel
        self.layers.events = type("E", (), {"removed": _Sub()})()
        self.dims = _FakeDims()
        self.events = type("E", (), {"blocker_all": lambda self_: _Blocker()})()

    def add_image(self, data, **kwargs):
        layer = _FakeLayer(
            kwargs["name"], data,
            colormap=kwargs.get("colormap"),
            contrast_limits=kwargs.get("contrast_limits"),
            visible=kwargs.get("visible", True),
        )
        layer.blending = kwargs.get("blending")
        self.layers.append(layer)
        return layer


class _FakeColorbar:
    layer_names = []

    def clear(self):
        pass


def _make_manager(viewer, data_manager):
    # Build without running __init__'s event wiring against fakes.
    manager = VisualizationManager.__new__(VisualizationManager)
    manager.viewer = viewer
    manager.data_manager = data_manager
    manager._layers = {}
    manager.colorbar_manager = _FakeColorbar()
    return manager


class _DataManager:
    def __init__(self):
        self.preprocessed_bead_stack = None
        self.preprocessed_reference = None
        self.preprocessed_cell_stack = None


# --- visualization-manager streaming -------------------------------------

def test_stream_writes_frame_in_place_and_follows_slider():
    viewer = _FakeViewer()
    dm = _DataManager()
    dm.preprocessed_bead_stack = np.zeros((3, 2, 2), dtype=np.float32)
    manager = _make_manager(viewer, dm)

    manager.begin_preprocessing_stream()
    layer = viewer.layers["Preprocessed Beads"]

    manager.stream_preprocessing_frame("beads", 1, np.full((2, 2), 7.0, dtype=np.float32))

    # Written straight into the pre-allocated stack (same array object).
    assert layer.data is dm.preprocessed_bead_stack
    assert np.all(dm.preprocessed_bead_stack[1] == 7.0)
    assert np.all(dm.preprocessed_bead_stack[0] == 0.0)
    assert layer.refresh_count == 1
    # Slider auto-advanced to the frame just computed.
    assert viewer.dims.steps[-1] == (0, 1)


def test_stream_reference_writes_without_advancing_slider():
    viewer = _FakeViewer()
    dm = _DataManager()
    dm.preprocessed_reference = np.zeros((2, 2), dtype=np.float32)
    manager = _make_manager(viewer, dm)

    manager.begin_preprocessing_stream()
    manager.stream_preprocessing_frame("reference", 0, np.full((2, 2), 5.0, dtype=np.float32))

    assert np.all(dm.preprocessed_reference == 5.0)
    # A 2D reference has no time axis to follow.
    assert viewer.dims.steps == []


def test_begin_stream_preserves_existing_layer_settings_on_rerun():
    viewer = _FakeViewer()
    dm = _DataManager()
    dm.preprocessed_bead_stack = np.zeros((2, 2, 2), dtype=np.float32)
    manager = _make_manager(viewer, dm)

    manager.begin_preprocessing_stream()
    layer = viewer.layers["Preprocessed Beads"]
    # User dials in custom contrast and hides the layer.
    layer.contrast_limits = [0.2, 0.6]
    layer.visible = False

    # Re-run allocates a fresh backing array and rebinds it.
    new_stack = np.zeros((2, 2, 2), dtype=np.float32)
    dm.preprocessed_bead_stack = new_stack
    manager.begin_preprocessing_stream()

    same_layer = viewer.layers["Preprocessed Beads"]
    assert same_layer is layer  # reused, not recreated
    assert same_layer.data is new_stack  # rebound to the new array
    assert same_layer.contrast_limits == [0.2, 0.6]  # contrast preserved
    assert same_layer.visible is False  # visibility preserved


def test_begin_stream_hides_unrelated_layers_but_does_not_force_show():
    viewer = _FakeViewer()
    hidden = viewer.add_image(np.ones((2, 2)), name="Raw beads", visible=True)
    hidden.visible = False  # user hid it on purpose
    viewer.add_image(np.ones((2, 2)), name="Force Magnitude", visible=True)
    dm = _DataManager()
    dm.preprocessed_bead_stack = np.zeros((1, 2, 2), dtype=np.float32)
    manager = _make_manager(viewer, dm)

    manager.begin_preprocessing_stream()

    # A run takes the viewer over: an unrelated *visible* layer (another stage)
    # is hidden so it can't bleed into the preprocessing view (worklist §4)...
    assert viewer.layers["Force Magnitude"].visible is False
    # ...but a layer the user hid stays hidden — the run never force-shows.
    assert "Raw beads" in viewer.layers
    assert viewer.layers["Raw beads"].visible is False


def test_begin_stream_defaults_to_normalized_contrast_window():
    # The live-run default binds zeroed stacks then streams normalized floats,
    # so a first-ever add must keep the [0, 1] window (not scale to the zeros).
    viewer = _FakeViewer()
    dm = _DataManager()
    dm.preprocessed_bead_stack = np.zeros((2, 2, 2), dtype=np.float32)
    manager = _make_manager(viewer, dm)

    manager.begin_preprocessing_stream()

    assert list(viewer.layers["Preprocessed Beads"].contrast_limits) == [0.0, 1.0]


def test_load_autoscales_contrast_to_uint16_range():
    # Regression: a display-only load binds fully-populated uint16 stacks read
    # back from disk (0..65535). The [0, 1] streaming default would clip every
    # non-zero pixel to white; autoscale_contrast must snap the window to the
    # data's actual range so the reloaded image matches what was saved.
    viewer = _FakeViewer()
    dm = _DataManager()
    beads = np.zeros((2, 4, 4), dtype=np.uint16)
    beads[0, 0, 0] = 65535
    beads[1, 1, 1] = 20000
    dm.preprocessed_bead_stack = beads
    manager = _make_manager(viewer, dm)

    manager.begin_preprocessing_stream(autoscale_contrast=True)

    layer = viewer.layers["Preprocessed Beads"]
    assert list(layer.contrast_limits) == [0.0, 65535.0]
    assert list(layer.contrast_limits_range) == [0.0, 65535.0]


def test_load_autoscale_overrides_preserved_streaming_window_on_rerun():
    # A prior live run leaves a layer at [0, 1]; rebinding preserves it. A later
    # display-only load into that same layer must override the stale window so a
    # click after a run doesn't inherit the saturating [0, 1] range.
    viewer = _FakeViewer()
    dm = _DataManager()
    dm.preprocessed_bead_stack = np.zeros((2, 2, 2), dtype=np.float32)
    manager = _make_manager(viewer, dm)
    manager.begin_preprocessing_stream()  # live run: layer created at [0, 1]
    assert list(viewer.layers["Preprocessed Beads"].contrast_limits) == [0.0, 1.0]

    loaded = np.full((2, 2, 2), 30000, dtype=np.uint16)
    loaded[0, 0, 0] = 0
    dm.preprocessed_bead_stack = loaded
    manager.begin_preprocessing_stream(autoscale_contrast=True)

    assert list(viewer.layers["Preprocessed Beads"].contrast_limits) == [0.0, 30000.0]


# --- controller wiring ----------------------------------------------------

class _ParameterManager(QObject):
    parameter_changed = Signal(str, object)
    parameters_reset = Signal(object)

    def get_preprocessing_parameters(self):
        return object()


class _ControllerDataManager:
    def __init__(self, bead=None, ref=None, cell=None):
        self.bead_stack = bead
        self.reference = ref
        self.cell_stack = cell
        self.preprocessed_bead_stack = None
        self.preprocessed_reference = None
        self.preprocessed_cell_stack = None
        self.sets = []

    def set_preprocessed_bead_stack(self, data, source="", dirty=False):
        self.preprocessed_bead_stack = data
        self.sets.append(("beads", data.shape, source, dirty))

    def set_preprocessed_reference(self, data, source="", dirty=False):
        self.preprocessed_reference = data
        self.sets.append(("reference", data.shape, source, dirty))

    def set_preprocessed_cell_stack(self, data, source="", dirty=False):
        self.preprocessed_cell_stack = data
        self.sets.append(("cells", data.shape, source, dirty))


class _RecordingViz:
    def __init__(self):
        self.began = 0
        self.frames = []

    def begin_preprocessing_stream(self):
        self.began += 1

    def stream_preprocessing_frame(self, data_type, frame_index, image):
        self.frames.append((data_type, frame_index, tuple(image.shape)))


def _controller(app, dm, viz):
    return PreprocessingController(
        viewer=_FakeViewer(),
        data_manager=dm,
        parameter_manager=_ParameterManager(),
        visualization_manager=viz,
    )


def test_begin_stream_allocates_zero_stacks_and_counts_total(app):
    dm = _ControllerDataManager(
        bead=np.ones((3, 2, 2), dtype=np.float32),
        ref=np.ones((2, 2), dtype=np.float32),
        cell=np.ones((4, 2, 2), dtype=np.float32),
    )
    viz = _RecordingViz()
    controller = _controller(app, dm, viz)

    controller._begin_stream()

    # 3 beads + 1 reference + 4 cells frames in total.
    assert controller._stream_total == 8
    assert controller._stream_done == 0
    # Output stacks allocated as zeros, matching input shapes, marked generated+dirty.
    assert dm.preprocessed_bead_stack.shape == (3, 2, 2)
    assert np.count_nonzero(dm.preprocessed_bead_stack) == 0
    assert ("beads", (3, 2, 2), "generated", True) in dm.sets
    assert ("reference", (2, 2), "generated", True) in dm.sets
    assert ("cells", (4, 2, 2), "generated", True) in dm.sets
    assert viz.began == 1


def test_on_frame_processed_streams_and_reports_progress(app):
    dm = _ControllerDataManager(bead=np.ones((2, 2, 2), dtype=np.float32))
    viz = _RecordingViz()
    controller = _controller(app, dm, viz)
    controller._stream_total = 2
    controller._stream_done = 0

    progress = []
    controller.progress_updated.connect(lambda *a: progress.append(a))

    controller._on_frame_processed(("beads", 0, 2, np.zeros((2, 2), dtype=np.float32)))
    controller._on_frame_processed(("beads", 1, 2, np.zeros((2, 2), dtype=np.float32)))

    assert viz.frames == [("beads", 0, (2, 2)), ("beads", 1, (2, 2))]
    assert progress[0] == (50, "Processing beads: Frame 1/2")
    assert progress[-1] == (100, "Processing beads: Frame 2/2")
