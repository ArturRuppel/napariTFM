import pytest
from qtpy.QtWidgets import QApplication, QWidget

from napariTFM.widgets._stage_section import StageSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_body_visible_regardless_of_params_toggle(app):
    body = QWidget()
    panel = QWidget()
    section = StageSection("Force Analysis", body, parameter_panel=panel)
    section.show()
    app.processEvents()
    # Stage body (action buttons) is always visible.
    assert body.isVisible()


def test_params_button_toggles_only_the_parameter_panel(app):
    body = QWidget()
    panel = QWidget()
    section = StageSection("Force Analysis", body, parameter_panel=panel)
    section.show()
    app.processEvents()

    assert not panel.isVisible()          # collapsed by default
    section.params_btn.setChecked(True)
    app.processEvents()
    assert panel.isVisible()
    assert body.isVisible()               # body unaffected
    section.params_btn.setChecked(False)
    app.processEvents()
    assert not panel.isVisible()


def test_no_inner_section_api(app):
    # add_inner_section was a faux-stage hack; it is gone.
    assert not hasattr(StageSection("X", QWidget()), "add_inner_section")
