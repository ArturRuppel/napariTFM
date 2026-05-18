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
