"""Live-streaming for the displacement / force / stress steps: each frame must
fill its pre-allocated stack in place, the viewer must follow the frame being
computed, and a re-run must preserve the contrast/visibility the user set (and
not hide unrelated layers) — the same behaviour ported from preprocessing."""

import sys
import types

import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication

for module_name in [
    "napariTFM.utilities.visualization_manager",
]:
    sys.modules.pop(module_name, None)

for module_name in ["gmsh", "solidspy", "solidspy.assemutil",
                    "solidspy.postprocesor", "solidspy.solutil"]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

from napariTFM.utilities.visualization_manager import VisualizationManager


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
        self.edge_color = None
        self.contrast_limits = list(contrast_limits) if contrast_limits else [0.0, 1.0]
        self.contrast_limits_range = list(self.contrast_limits)
        self.refresh_count = 0

    @property
    def ndim(self):
        return np.ndim(self.data)

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

    def add_vectors(self, data, **kwargs):
        layer = _FakeLayer(kwargs["name"], data, visible=kwargs.get("visible", True))
        layer.edge_color = kwargs.get("edge_color")
        self.layers.append(layer)
        return layer


class _FakeColorbar:
    layer_names = []

    def clear(self):
        pass

    def show_for_layer(self, *_, **__):
        pass


class _DataManager:
    def __init__(self):
        self.displacement_vector_cache = None
        self.force_vector_cache = None
        self.stress_components = None


def _make_manager(viewer, data_manager):
    manager = VisualizationManager.__new__(VisualizationManager)
    manager.viewer = viewer
    manager.data_manager = data_manager
    manager._layers = {}
    manager.colorbar_manager = _FakeColorbar()
    return manager


def _vis_params(v_max=5.0):
    return {
        'v_max': v_max,
        'vector_stride': 1,
        'arrow_scale': 1.0,
        'downscale_factor': 1,
    }


# --- vector-field streaming (displacement / force) ------------------------

@pytest.mark.parametrize("kind,mag_name,vec_name", [
    ("displacement", "Displacement Magnitude", "Displacement Vectors"),
    ("force", "Force Magnitude", "Force Vectors"),
])
def test_vector_field_streams_frame_in_place_and_follows_slider(kind, mag_name, vec_name):
    viewer = _FakeViewer()
    dm = _DataManager()
    manager = _make_manager(viewer, dm)

    manager.begin_vector_field_stream(kind, num_frames=2, vis_params=_vis_params())
    field = np.ones((4, 4, 2), dtype=np.float32)
    manager.stream_vector_field_frame(kind, 0, field)

    mag = viewer.layers[mag_name]
    # Magnitude stack pre-allocated for both frames; frame 0 filled, frame 1 zero.
    assert mag.data.shape == (2, 4, 4)
    assert np.allclose(mag.data[0], np.sqrt(2.0))
    assert np.all(mag.data[1] == 0.0)
    assert mag.refresh_count == 1
    # Slider auto-advanced to the frame just computed.
    assert viewer.dims.steps[-1] == (0, 0)
    # Vectors layer created and this frame cached for later scrubbing.
    assert vec_name in viewer.layers
    cache_attr = 'displacement_vector_cache' if kind == 'displacement' else 'force_vector_cache'
    cache = getattr(dm, cache_attr)
    assert cache['data'][0] is not None
    assert cache['data'][1] is None  # not computed yet


def test_vector_field_rerun_preserves_magnitude_settings():
    viewer = _FakeViewer()
    dm = _DataManager()
    manager = _make_manager(viewer, dm)

    manager.begin_vector_field_stream('displacement', num_frames=1, vis_params=_vis_params())
    manager.stream_vector_field_frame('displacement', 0, np.ones((4, 4, 2), dtype=np.float32))

    mag = viewer.layers['Displacement Magnitude']
    mag.contrast_limits = [0.2, 0.6]
    mag.visible = False

    # Re-run: lazily rebinds the same magnitude layer to a fresh zeroed stack.
    manager.begin_vector_field_stream('displacement', num_frames=1, vis_params=_vis_params())
    manager.stream_vector_field_frame('displacement', 0, np.ones((4, 4, 2), dtype=np.float32))

    same = viewer.layers['Displacement Magnitude']
    assert same is mag                      # reused, not recreated
    assert same.contrast_limits == [0.2, 0.6]  # contrast preserved
    assert same.visible is False            # visibility preserved


def test_vector_field_rerun_recreates_magnitude_when_ndim_changes():
    """A 2D preview magnitude layer rebound to a 3D stream stack must be
    recreated, not swapped in place.

    napari never refreshes a vispy layer's cached ``_world_to_layer_units_scale``
    on an ndim change, so assigning ``.data`` with a different ndim leaves a
    stale, too-short scale tuple and the next slice raises ``IndexError`` deep in
    vispy. Recreating the layer side-steps that. See ``_rebind_image_layer``.
    """
    viewer = _FakeViewer()
    dm = _DataManager()
    manager = _make_manager(viewer, dm)

    # Simulate a single-frame preview having created a 2D magnitude layer.
    preview = viewer.add_image(
        np.ones((4, 4), dtype=np.float32),
        name='Displacement Magnitude',
        colormap='viridis',
        contrast_limits=(0.0, 5.0),
        visible=True,
    )
    preview.contrast_limits = [0.1, 0.9]
    preview.visible = False
    assert preview.ndim == 2

    # Full stream rebinds the same name to a 3D (num_frames, H, W) stack.
    manager.begin_vector_field_stream('displacement', num_frames=3, vis_params=_vis_params())
    manager.stream_vector_field_frame('displacement', 0, np.ones((4, 4, 2), dtype=np.float32))

    mag = viewer.layers['Displacement Magnitude']
    assert mag is not preview                 # recreated, not swapped
    assert mag.ndim == 3
    assert mag.data.shape == (3, 4, 4)
    assert manager._layers['displacement_magnitude'] is mag  # cache points at new layer
    # The settings that survive a re-run are preserved.
    assert mag.contrast_limits == [0.1, 0.9]
    assert mag.visible is False
    # Exactly one magnitude layer remains (the stale 2D one was removed).
    assert sum(layer.name == 'Displacement Magnitude' for layer in viewer.layers) == 1


def test_vector_field_hides_unrelated_layers_but_does_not_force_show():
    viewer = _FakeViewer()
    hidden = viewer.add_image(np.ones((2, 2)), name="Raw beads", visible=True)
    hidden.visible = False  # user hid it on purpose
    viewer.add_image(np.ones((2, 2)), name="Average Normal Stress", visible=True)
    dm = _DataManager()
    manager = _make_manager(viewer, dm)

    manager.begin_vector_field_stream('displacement', num_frames=1, vis_params=_vis_params())
    manager.stream_vector_field_frame('displacement', 0, np.ones((4, 4, 2), dtype=np.float32))

    # A run takes the viewer over: an unrelated *visible* layer (another stage)
    # is hidden so it can't bleed into the displacement view (worklist §4)...
    assert viewer.layers["Average Normal Stress"].visible is False
    # ...but a layer the user hid stays hidden — the run never force-shows.
    assert "Raw beads" in viewer.layers
    assert viewer.layers["Raw beads"].visible is False


# --- stress streaming (BISM) ------------------------------------------------

def _stress_frame(xx, yy):
    t = np.zeros((4, 4, 2, 2), dtype=np.float32)
    t[..., 0, 0] = xx
    t[..., 1, 1] = yy
    return t


def test_stress_streams_three_components_in_place_and_follows_slider():
    viewer = _FakeViewer()
    dm = _DataManager()
    manager = _make_manager(viewer, dm)

    manager.begin_stress_stream(num_frames=2, max_stress=10.0, downscale_factor=1)
    manager.stream_stress_frame(0, _stress_frame(xx=4.0, yy=2.0))

    for name in ('Normal Stress XX', 'Normal Stress YY', 'Average Normal Stress'):
        assert name in viewer.layers
        assert viewer.layers[name].data.shape == (2, 4, 4)

    comps = dm.stress_components
    assert np.all(comps['sigma_xx'][0] == 4.0)
    assert np.all(comps['sigma_yy'][0] == 2.0)
    assert np.all(comps['sigma_normal'][0] == 3.0)  # (4 + 2) / 2
    assert np.all(comps['sigma_xx'][1] == 0.0)      # frame 1 not computed yet
    # Slider auto-advanced to the streamed frame.
    assert viewer.dims.steps[-1] == (0, 0)


def test_stress_rerun_preserves_layer_settings():
    viewer = _FakeViewer()
    dm = _DataManager()
    manager = _make_manager(viewer, dm)

    manager.begin_stress_stream(num_frames=1, max_stress=10.0, downscale_factor=1)
    manager.stream_stress_frame(0, _stress_frame(xx=1.0, yy=1.0))

    avg = viewer.layers['Average Normal Stress']
    avg.contrast_limits = [-3.0, 3.0]
    avg.visible = False

    manager.begin_stress_stream(num_frames=1, max_stress=10.0, downscale_factor=1)
    manager.stream_stress_frame(0, _stress_frame(xx=1.0, yy=1.0))

    same = viewer.layers['Average Normal Stress']
    assert same is avg
    assert same.contrast_limits == [-3.0, 3.0]
    assert same.visible is False


# --- controller wiring (displacement / force) -----------------------------

class _Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _ParameterManager(QObject):
    parameter_changed = Signal(str, object)
    parameters_reset = Signal(object)

    def __init__(self, params):
        super().__init__()
        self._params = params

    def get_displacement_parameters(self):
        return self._params

    def get_fttc_parameters(self):
        return self._params


class _CtrlDataManager:
    def __init__(self):
        self.preprocessed_bead_stack = np.ones((3, 4, 4), dtype=np.float32)
        self.preprocessed_reference = np.ones((4, 4), dtype=np.float32)
        # Force reads its frame count from the displacement result (2 frames).
        self.displacement_results = _Params(
            displacement_field=np.ones((2, 4, 4, 2), dtype=np.float32)
        )


class _RecordingViz:
    def __init__(self):
        self.begun = []
        self.frames = []

    def begin_vector_field_stream(self, kind, num_frames, vis_params):
        self.begun.append((kind, num_frames, dict(vis_params)))

    def stream_vector_field_frame(self, kind, frame_index, field):
        self.frames.append((kind, frame_index, tuple(field.shape)))


def test_displacement_controller_begin_and_stream(app):
    from napariTFM.widgets.displacement_analysis_widget import DisplacementController

    viz = _RecordingViz()
    dm = _CtrlDataManager()
    params = _Params(d_max=5.0, disp_vector_stride=8, disp_arrow_scale=1.0, downscale_factor=1)
    ctrl = DisplacementController(
        viewer=_FakeViewer(), data_manager=dm,
        parameter_manager=_ParameterManager(params), visualization_manager=viz,
    )

    ctrl._begin_stream(params)
    assert ctrl._stream_total == 3
    assert viz.begun == [('displacement', 3, {
        'v_max': 5.0, 'vector_stride': 8, 'arrow_scale': 1.0, 'downscale_factor': 1,
    })]

    progress = []
    ctrl.progress_updated.connect(lambda *a: progress.append(a))
    ctrl._on_frame_processed((0, 3, np.ones((4, 4, 2), dtype=np.float32)))
    ctrl._on_frame_processed((1, 3, np.ones((4, 4, 2), dtype=np.float32)))

    assert viz.frames == [('displacement', 0, (4, 4, 2)), ('displacement', 1, (4, 4, 2))]
    assert progress[0] == (33, "Processing frame 1/3")
    assert progress[1] == (66, "Processing frame 2/3")


def test_fttc_controller_begin_and_stream(app):
    from napariTFM.widgets.fttc_widget import FTTCController

    viz = _RecordingViz()
    params = _Params(f_max=20.0, force_vector_stride=8, force_arrow_scale=1.0, downscale_factor=1)
    ctrl = FTTCController(
        viewer=_FakeViewer(), data_manager=_CtrlDataManager(),
        parameter_manager=_ParameterManager(params), visualization_manager=viz,
    )

    # Unified _begin_stream(params): the frame count comes from the data
    # manager's displacement result, not a passed-in array.
    ctrl._begin_stream(params)
    assert ctrl._stream_total == 2
    assert viz.begun == [('force', 2, {
        'v_max': 20.0, 'vector_stride': 8, 'arrow_scale': 1.0, 'downscale_factor': 1,
    })]

    progress = []
    ctrl.progress_updated.connect(lambda *a: progress.append(a))
    ctrl._on_frame_processed((0, 2, np.ones((4, 4, 2), dtype=np.float32)))
    ctrl._on_frame_processed((1, 2, np.ones((4, 4, 2), dtype=np.float32)))

    assert viz.frames == [('force', 0, (4, 4, 2)), ('force', 1, (4, 4, 2))]
    assert progress[0] == (50, "Processing frame 1/2")
    assert progress[-1] == (100, "Processing frame 2/2")
