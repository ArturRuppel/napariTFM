import pytest
from qtpy.QtWidgets import QApplication, QWidget

from napariTFM.widgets._stage_section import StageSection
from napariTFM.widgets._ui_style import muted_stage_accent, stage_accent


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_outer_section_uses_explicit_accent(app):
    child = QWidget()
    section = StageSection("Preprocessing", child, accent=stage_accent("preprocessing"))

    assert stage_accent("preprocessing") in section.header_label.styleSheet()


def test_inner_section_inherits_and_mutes_parent_accent(app):
    inner_child = QWidget()
    outer_child = QWidget()
    outer = StageSection("Preprocessing", outer_child, accent=stage_accent("preprocessing"))

    inner = outer.add_inner_section("Parameters", inner_child)

    expected = muted_stage_accent("preprocessing")
    assert expected in inner.header_label.styleSheet()


def test_inner_section_added_to_parent_content(app):
    inner_child = QWidget()
    outer_child = QWidget()
    outer = StageSection("Preprocessing", outer_child)

    inner = outer.add_inner_section("Parameters", inner_child, expanded=False)

    assert inner.parent() is outer._content
    assert isinstance(inner, StageSection)


def test_inner_section_collapsed_by_default(app):
    inner_child = QWidget()
    outer = StageSection("Preprocessing", QWidget(), expanded=True)
    inner = outer.add_inner_section("Parameters", inner_child)
    outer.show()
    app.processEvents()

    assert not inner_child.isVisible()


def test_inner_section_toggle_reveals_inner_child(app):
    inner_child = QWidget()
    outer = StageSection("Preprocessing", QWidget(), expanded=True)
    inner = outer.add_inner_section("Parameters", inner_child)
    outer.show()
    app.processEvents()

    inner._toggle_button.setChecked(True)
    app.processEvents()

    assert inner_child.isVisible()
