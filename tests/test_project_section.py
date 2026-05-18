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


def test_project_section_exposes_save_load_reset_clear_buttons(app):
    section = ProjectSection(_StubParameterManager())

    for name in ["save_params_btn", "load_params_btn", "reset_params_btn", "clear_data_btn"]:
        button = getattr(section, name)
        assert isinstance(button, QPushButton)


def test_project_section_writes_through_ui_parameter_api(app):
    manager = _StubParameterManager()
    section = ProjectSection(manager)

    section.parameter_controls["pixel_size"].setValue(0.108)

    assert ("pixel_size", 0.108) in manager.ui_writes


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
