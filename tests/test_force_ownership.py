import pytest
from qtpy.QtWidgets import QApplication

import napariTFM.widgets.fttc_widget as fw


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
    displacement_results = None
    force_results = None


class _FakeParameterManager:
    def __init__(self):
        from qtpy.QtCore import QObject, Signal

        class _PM(QObject):
            parameter_changed = Signal(str, object)
            parameters_reset = Signal(object)

        self._pm = _PM()
        self.parameter_changed = self._pm.parameter_changed
        self.parameters_reset = self._pm.parameters_reset

    def get_fttc_parameters(self):
        return object()


@pytest.fixture
def force_widget(app):
    return fw.FTTCWidget(
        viewer=_FakeViewer(),
        data_manager=_FakeDataManager(),
        parameter_manager=_FakeParameterManager(),
        visualization_manager=object(),
    )


def test_force_exposes_action_contract(app, force_widget):
    w = force_widget
    assert hasattr(w, "action_states_changed")
    states = w.action_states()
    assert set(states) >= {"run", "preview", "cancel"}
    assert callable(w.run_action)
    assert callable(w.preview_action)
    assert callable(w.cancel_action)


def test_no_per_stage_status_label(app, force_widget):
    # P2: the shell's one global status label replaces per-stage labels.
    assert not hasattr(force_widget, "status_label")


def test_parameter_panel_class_is_removed():
    assert not hasattr(fw, "FTTCParameterPanel")


def test_data_panel_class_is_removed():
    assert not hasattr(fw, "FTTCDataPanel")


def test_action_panel_class_is_removed():
    assert not hasattr(fw, "FTTCActionPanel")


def test_controller_has_no_panel_attributes(app):
    controller = fw.FTTCController(
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
    controller = fw.FTTCController(
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
