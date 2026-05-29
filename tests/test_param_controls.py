import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._param_controls import dslider, islider


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_islider_roundtrips_value(app):
    s = islider(0, 50, 12)
    assert s.value() == 12
    s.setValue(20)
    assert s.value() == 20


def test_dslider_respects_decimals_and_range(app):
    s = dslider(0.0, 10.0, 1.5, step=0.1, decimals=1)
    assert abs(s.value() - 1.5) < 1e-9
    s.setValue(3.3)
    assert abs(s.value() - 3.3) < 1e-9


def test_sliders_emit_value_changed(app):
    s = islider(0, 10, 1)
    seen = []
    s.valueChanged.connect(seen.append)
    s.setValue(7)
    assert seen and seen[-1] == 7
