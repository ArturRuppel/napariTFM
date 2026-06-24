from pathlib import Path

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QPushButton

from napariTFM.widgets._project_section import ProjectSection


class _StubParameterManager(QObject):
    parameter_changed = Signal(str, object)

    def __init__(self):
        super().__init__()
        self._values = {"pixel_size": 1.0, "frame_interval": 1.0}
        self.ui_writes = []

    def get_parameter(self, name):
        return self._values[name]

    def get_ui_parameter(self, name):
        return self._values[name]

    def set_ui_parameter(self, name, value):
        self.ui_writes.append((name, value))
        self._values[name] = value
        self.parameter_changed.emit(name, value)

    def reset_all_parameters(self):
        self._values = {"pixel_size": 1.0, "frame_interval": 1.0}


class _StubDataManager:
    def __init__(self):
        self.output_dir = None
        self.set_calls = []
        self._callbacks = []

    def set_output_dir(self, path):
        self.output_dir = Path(path).expanduser() if path else None
        self.set_calls.append(self.output_dir)
        for callback in list(self._callbacks):
            callback()

    def add_change_callback(self, callback):
        self._callbacks.append(callback)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_project_section_contains_general_parameter_controls(app):
    section = ProjectSection(_StubParameterManager())

    assert "pixel_size" in section.parameter_controls
    assert "frame_interval" in section.parameter_controls


def test_project_section_exposes_one_config_save_plus_reset_clear(app):
    section = ProjectSection(_StubParameterManager())

    for name in ["save_config_btn", "load_config_btn", "reset_params_btn", "clear_data_btn"]:
        button = getattr(section, name)
        assert isinstance(button, QPushButton)
    # The params-only save merged into the single config save (P0b).
    assert section.save_config_btn.text() == "Save Config"
    assert section.load_config_btn.text() == "Load Config"
    assert not hasattr(section, "save_params_btn")


def test_project_section_uses_free_text_inputs(app):
    from qtpy.QtWidgets import QLineEdit

    section = ProjectSection(_StubParameterManager())

    for name in ("pixel_size", "frame_interval"):
        assert isinstance(section.parameter_controls[name], QLineEdit)


def test_free_text_field_shows_initial_parameter_value(app):
    section = ProjectSection(_StubParameterManager())

    assert float(section.parameter_controls["pixel_size"].text()) == 1.0


def test_project_section_writes_through_ui_parameter_api(app):
    manager = _StubParameterManager()
    section = ProjectSection(manager)

    field = section.parameter_controls["pixel_size"]
    field.setText("0.108")
    field.editingFinished.emit()

    assert ("pixel_size", 0.108) in manager.ui_writes


def test_free_text_field_reverts_unparseable_input(app):
    manager = _StubParameterManager()
    section = ProjectSection(manager)

    field = section.parameter_controls["pixel_size"]
    field.setText("not-a-number")
    field.editingFinished.emit()

    assert manager.ui_writes == []  # nothing written
    assert float(field.text()) == 1.0  # reverted to last good value


def test_free_text_field_syncs_from_parameter_changed(app):
    manager = _StubParameterManager()
    section = ProjectSection(manager)

    manager.set_ui_parameter("frame_interval", 2.5)

    assert float(section.parameter_controls["frame_interval"].text()) == 2.5


def test_project_section_starts_expanded(app):
    section = ProjectSection(_StubParameterManager())
    section.show()
    app.processEvents()

    assert section._content.isVisible()


def test_project_section_shows_unset_output_directory(app):
    section = ProjectSection(_StubParameterManager(), _StubDataManager())

    assert section.output_dir_label.text() == "No output directory"
    assert section.output_dir_label.toolTip() == ""


def test_project_section_syncs_output_directory_from_data_manager(app, tmp_path):
    data_manager = _StubDataManager()
    section = ProjectSection(_StubParameterManager(), data_manager)

    data_manager.set_output_dir(tmp_path)

    assert section.output_dir_label.text() == str(tmp_path)
    assert section.output_dir_label.toolTip() == str(tmp_path)


def test_project_section_exposes_output_directory_button(app):
    section = ProjectSection(_StubParameterManager(), _StubDataManager())

    assert isinstance(section.choose_output_dir_btn, QPushButton)
    assert section.choose_output_dir_btn.objectName() == "project_choose_output_dir_button"


def test_general_body_uses_section_grid_not_groupbox(app):
    from qtpy.QtWidgets import QGridLayout, QGroupBox

    section = ProjectSection(_StubParameterManager())

    assert section.body.findChildren(QGroupBox) == []
    grid = section.body.findChild(QGridLayout)
    assert grid is not None
    # pixel_size (col 0) and frame_interval (col 2) share the first row
    assert grid.itemAtPosition(0, 0) is not None
    assert grid.itemAtPosition(0, 2) is not None
