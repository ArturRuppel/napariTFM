import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from napariTFM.widgets._collapsible_section import CollapsibleSection


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_starts_collapsed_and_toggles(app):
    inner = QLabel("body")
    sec = CollapsibleSection("Params", inner, expanded=False)
    assert sec.is_expanded is False
    assert sec._content_frame.isVisible() is False
    sec.expand()
    assert sec.is_expanded is True


def test_header_can_be_hidden(app):
    sec = CollapsibleSection("Params", QLabel("body"))
    sec.set_header_visible(False)
    assert sec._toggle.isVisible() is False


def test_outer_accent_styles_header(app):
    sec = CollapsibleSection("Stage", QLabel("body"), accent_color="#3b6fb6")
    assert "#3b6fb6" in sec._toggle.styleSheet()


def test_inner_inherits_ancestor_accent(app):
    outer = CollapsibleSection("Outer", QWidget(), accent_color="#3b6fb6")
    inner = CollapsibleSection("Inner", QLabel("x"))
    inner.setParent(outer)
    inner._maybe_inherit_accent()
    assert inner._effective_accent == "#3b6fb6"


def test_set_accent_color_refreshes_descendants(app):
    outer = CollapsibleSection("Outer", QWidget())
    inner = CollapsibleSection("Inner", QLabel("x"))
    inner.setParent(outer)
    outer.set_accent_color("#aa3344")
    assert inner._effective_accent == "#aa3344"
