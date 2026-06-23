import pytest
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from napariTFM.widgets._collapsible_section import CollapsibleSection
from napariTFM.widgets._stage_section import StageSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _panel():
    p = QWidget()
    return p


def test_param_panel_lives_in_a_collapsible_section(app):
    sec = StageSection("Stage", QLabel("body"), parameter_panel=_panel())
    assert isinstance(sec._param_section, CollapsibleSection)
    # Header of the inner CollapsibleSection is hidden — the stage's own
    # params_btn is the visible toggle.
    assert sec._param_section._toggle.isVisible() is False


def test_body_visible_regardless_of_params_toggle(app):
    sec = StageSection("Stage", QLabel("body"), parameter_panel=_panel())
    sec.show()
    app.processEvents()
    assert sec._content.isVisible() is True
    sec.params_btn.setChecked(True)
    assert sec._content.isVisible() is True
    sec.params_btn.setChecked(False)
    assert sec._content.isVisible() is True


def test_params_button_toggles_only_the_collapsible(app):
    sec = StageSection("Stage", QLabel("body"), parameter_panel=_panel())
    sec.params_btn.setChecked(True)
    assert sec._param_section.is_expanded is True
    sec.params_btn.setChecked(False)
    assert sec._param_section.is_expanded is False


def test_no_param_panel_hides_params_button(app):
    sec = StageSection("Stage", QLabel("body"))
    assert sec.params_btn.isVisible() is False
    assert sec._param_section is None


def test_no_inner_section_api(app):
    sec = StageSection("Stage", QLabel("body"))
    assert not hasattr(sec, "add_inner_section")
