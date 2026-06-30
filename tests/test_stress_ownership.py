import pytest
from qtpy.QtWidgets import QApplication

import napariTFM.widgets.stress_widget as mw


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeDims:
    class _Events:
        class _Step:
            def connect(self, *_):
                return None
        current_step = _Step()
    events = _Events()
    current_step = (0,)


class _FakeLayersSelection:
    active = None

    class _Events:
        class _Active:
            def connect(self, *_):
                return None
        active = _Active()
    events = _Events()


class _FakeLayers:
    selection = _FakeLayersSelection()

    def __iter__(self):
        return iter(())


class _FakeViewer:
    dims = _FakeDims()
    layers = _FakeLayers()


class _FakeDataManager:
    force_results = None
    mask_stack = None
    stress_results = None


class _FakeParameterManager:
    def __init__(self):
        from qtpy.QtCore import QObject, Signal

        class _PM(QObject):
            parameter_changed = Signal(str, object)
            parameters_reset = Signal(object)

        self._pm = _PM()
        self.parameter_changed = self._pm.parameter_changed
        self.parameters_reset = self._pm.parameters_reset

    def get_stress_parameters(self):
        return object()


@pytest.fixture
def stress_widget(app):
    return mw.StressWidget(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )


def test_stress_exposes_action_contract(app, stress_widget):
    w = stress_widget
    assert hasattr(w, "action_states_changed")
    states = w.action_states()
    assert set(states) >= {"run", "preview", "cancel"}
    assert callable(w.run_action)
    assert callable(w.preview_action)
    assert callable(w.cancel_action)


def test_no_per_stage_status_label(app, stress_widget):
    # P2: the shell's one global status label replaces per-stage labels.
    assert not hasattr(stress_widget, "status_label")


def test_parameter_panel_class_is_removed():
    assert not hasattr(mw, "StressParameterPanel")


def test_data_panel_class_is_removed():
    assert not hasattr(mw, "StressDataPanel")


def test_action_panel_class_is_removed():
    assert not hasattr(mw, "StressActionPanel")


def test_controller_has_no_panel_attributes(app):
    controller = mw.StressController(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )
    assert not hasattr(controller, "parameter_panel")
    assert not hasattr(controller, "data_panel")
    assert not hasattr(controller, "action_panel")
    assert not hasattr(controller, "set_panels")


def test_controller_freeze_emits_signal_without_panels(app):
    controller = mw.StressController(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )
    seen = []
    controller.ui_frozen.connect(seen.append)
    controller.freeze_ui()
    controller.unfreeze_ui()
    assert seen == [True, False]


class _CapturingVisualizationManager:
    def __init__(self):
        self.calls = []

    def visualize_masks(self, masks, *args, **kwargs):
        self.calls.append((masks, kwargs))


class _Forces:
    def __init__(self, force_field):
        self.force_field = force_field


class _MaskDataManager:
    """Holds optional force results (to downsample the mask) and a bead stack."""

    def __init__(self, force_results=None, bead_stack=None):
        self.force_results = force_results
        self.bead_stack = bead_stack
        self.mask_stack = None

    def set_mask_stack(self, masks):
        self.mask_stack = masks


def _make_widget(viz, data_manager):
    return mw.StressWidget(
        viewer=_FakeViewer(),
        data_manager=data_manager,
        parameter_manager=_FakeParameterManager(),
        visualization_manager=viz,
    )


def test_mask_layer_scaled_to_fit_beads(app):
    import numpy as np

    # Force grid downsamples the mask to (10, 15); beads are (40, 60).
    force_field = np.zeros((3, 10, 15, 2), dtype=np.float32)
    viz = _CapturingVisualizationManager()
    widget = _make_widget(viz, _MaskDataManager(force_results=_Forces(force_field)))

    raw_mask = np.ones((3, 40, 60), dtype=np.uint8)
    widget._apply_mask_data(raw_mask, warn=False, beads_shape=(40, 60))

    masks, kwargs = viz.calls[-1]
    # Stored/displayed array stays on the downsampled grid (not inflated)...
    assert masks.shape == (3, 10, 15)
    assert widget.data_manager.mask_stack.shape == (3, 10, 15)
    # ...but the layer is scaled by the actual bead/mask xy ratio to fit the beads.
    assert kwargs["scale"] == (1.0, 4.0, 4.0)


def test_mask_layer_scaled_to_fit_beads_without_force_results(app):
    import numpy as np

    # No force results → mask kept at its file resolution (12, 9); beads (36, 36).
    viz = _CapturingVisualizationManager()
    widget = _make_widget(viz, _MaskDataManager(force_results=None))

    raw_mask = np.ones((2, 12, 9), dtype=np.uint8)
    widget._apply_mask_data(raw_mask, warn=False, beads_shape=(36, 36))

    masks, kwargs = viz.calls[-1]
    assert masks.shape == (2, 12, 9)
    assert kwargs["scale"] == (1.0, 3.0, 4.0)


def test_mask_layer_unscaled_when_matching_beads(app):
    import numpy as np

    force_field = np.zeros((2, 32, 32, 2), dtype=np.float32)
    viz = _CapturingVisualizationManager()
    widget = _make_widget(viz, _MaskDataManager(force_results=_Forces(force_field)))

    widget._apply_mask_data(
        np.ones((2, 32, 32), dtype=np.uint8), warn=False, beads_shape=(32, 32)
    )

    _, kwargs = viz.calls[-1]
    assert kwargs["scale"] is None


def test_mask_layer_scale_is_none_without_a_beads_shape(app):
    import numpy as np

    viz = _CapturingVisualizationManager()
    widget = _make_widget(viz, _MaskDataManager())

    widget._apply_mask_data(np.ones((1, 8, 8), dtype=np.uint8), warn=False)

    _, kwargs = viz.calls[-1]
    assert kwargs["scale"] is None
