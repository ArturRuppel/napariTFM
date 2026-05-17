import sys
import types

import numpy as np
import pytest
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QApplication, QMessageBox, QWidget


class _RangeSlider(QWidget):
    valueChanged = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._value = (0, 1000)

    def setMinimum(self, value):
        pass

    def setMaximum(self, value):
        pass

    def setRange(self, minimum, maximum):
        pass

    def setSingleStep(self, value):
        pass

    def setPageStep(self, value):
        pass

    def setValue(self, value):
        self._value = value
        self.valueChanged.emit(value)

    def value(self):
        return self._value


qtrangeslider_module = types.ModuleType("qtrangeslider")
qtrangeslider_module.QRangeSlider = _RangeSlider
sys.modules["qtrangeslider"] = qtrangeslider_module

for module_name in ["gmsh", "solidspy", "solidspy.assemutil", "solidspy.postprocesor"]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))
for module_name in [
    "napariTFM.utilities.data_manager",
    "napariTFM.widgets._output_directory",
    "napariTFM.widgets._pipeline_data_widget",
]:
    sys.modules.pop(module_name, None)

from napariTFM.backend.displacement_analysis import DisplacementResult
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.utilities.data_manager import DataManager
from napariTFM.widgets._output_directory import ensure_output_dir_for_generated_artifacts
from napariTFM.widgets._pipeline_data_widget import PipelineDataWidget


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeSelection:
    def __init__(self):
        self.active = None


class _FakeLayers(list):
    def __init__(self):
        super().__init__()
        self.selection = _FakeSelection()


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayers()


class _FakeLayer:
    def __init__(self, data, name="Layer", path=None):
        self.data = data
        self.name = name
        if path is not None:
            self.source = type("Source", (), {"path": str(path)})()


def _displacement_result():
    field = np.ones((1, 2, 2, 2), dtype=np.float32)
    return DisplacementResult(
        displacement_field=field,
        original_shape=(2, 2),
        displacement_field_shape=(2, 2),
        parameters=DisplacementParameters(),
        physical_scale={"pixel_size": 1.0, "frame_interval": 1.0},
    )


def test_data_manager_tracks_artifact_metadata_and_callbacks(tmp_path):
    manager = DataManager()
    calls = []
    manager.add_change_callback(lambda: calls.append("changed"))

    manager.set_reference(np.ones((2, 2), dtype=np.float32), path=tmp_path / "ref.tif", source="file")

    state = manager.get_artifact("reference")
    assert state.available
    assert state.path == tmp_path / "ref.tif"
    assert state.source == "file"
    assert not state.dirty
    assert calls == ["changed"]


def test_data_manager_auto_saves_generated_artifacts(tmp_path):
    manager = DataManager()
    manager.set_output_dir(tmp_path)
    manager.set_displacement_results(_displacement_result(), source="generated", dirty=True)

    path = manager.auto_save_artifact("displacement_results")

    state = manager.get_artifact("displacement_results")
    assert path == tmp_path / "displacement_results.npy"
    assert path.exists()
    assert not state.dirty
    assert state.path == path
    loaded = np.load(path, allow_pickle=True).item()
    assert isinstance(loaded, DisplacementResult)


def test_data_manager_loads_result_artifacts_from_npy(tmp_path):
    manager = DataManager()
    path = tmp_path / "displacement.npy"
    np.save(path, _displacement_result())

    manager.load_result_artifact("displacement_results", path)

    state = manager.get_artifact("displacement_results")
    assert state.available
    assert state.path == path
    assert state.source == "file"
    assert not state.dirty


def test_pipeline_data_widget_loads_active_image_layer(app, tmp_path):
    viewer = _FakeViewer()
    data = np.ones((1, 2, 2), dtype=np.float32)
    viewer.layers.selection.active = _FakeLayer(data, name="beads", path=tmp_path / "beads.tif")
    manager = DataManager()
    widget = PipelineDataWidget(viewer, manager)

    widget.load_active_layer_artifact("bead_stack")

    state = manager.get_artifact("bead_stack")
    assert np.array_equal(manager.bead_stack, data)
    assert state.path == tmp_path / "beads.tif"
    assert state.source == "beads"
    assert "(1, 2, 2)" in widget.rows["bead_stack"].info_label.text()


def test_pipeline_data_widget_auto_output_dir_uses_loaded_input_path(app, tmp_path):
    manager = DataManager()
    manager.set_reference(np.ones((2, 2), dtype=np.float32), path=tmp_path / "input" / "ref.tif", source="file")
    widget = PipelineDataWidget(_FakeViewer(), manager)

    widget.auto_output_dir()

    assert manager.output_dir == tmp_path / "input" / "napariTFM_outputs"
    assert widget.output_dir_edit.text() == str(manager.output_dir)


def test_output_directory_helper_prompts_and_sets_directory(monkeypatch, app, tmp_path):
    manager = DataManager()
    chosen = tmp_path / "outputs"
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(
        "napariTFM.widgets._output_directory.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(chosen),
    )

    assert ensure_output_dir_for_generated_artifacts(None, manager)
    assert manager.output_dir == chosen


def test_output_directory_helper_allows_user_to_skip_save(monkeypatch, app):
    manager = DataManager()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    assert not ensure_output_dir_for_generated_artifacts(None, manager)
    assert manager.output_dir is None
