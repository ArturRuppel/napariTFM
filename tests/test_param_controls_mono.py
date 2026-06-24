from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication

import pytest

from napariTFM.widgets._param_controls import islider


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_slider_value_label_uses_monospace_font(app):
    s = islider(0, 100, 50)
    assert s._label.font().styleHint() == QFont.StyleHint.Monospace
