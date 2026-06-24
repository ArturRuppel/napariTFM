import pytest
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._stage_spine import StageSpine, _node_style


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_node_style_done_is_filled_with_accent():
    fill, ring = _node_style("done", "#2a788e")
    assert fill is not None
    assert fill.name() == "#2a788e"
    assert ring.name() == "#2a788e"


def test_node_style_ready_is_hollow_ring():
    fill, ring = _node_style("ready", "#2a788e")
    assert fill is None
    assert ring.name() == "#2a788e"


def test_node_style_running_is_amber():
    fill, ring = _node_style("running", "#2a788e")
    assert fill is not None and ring.name() == "#e3b341"


def test_node_style_not_started_is_dim_hollow():
    fill, ring = _node_style("not_started", "#2a788e")
    assert fill is None
    assert ring.name() != "#2a788e"


def test_spine_set_status_updates_state(app):
    spine = StageSpine("#2a788e")
    spine.set_status("done")
    assert spine._status == "done"


def test_spine_set_accents_stores_neighbours(app):
    spine = StageSpine("#2a788e")
    spine.set_accents("#2a788e", above="#414487", below="#22a884")
    assert spine._accent_above == "#414487"
    assert spine._accent_below == "#22a884"


def test_spine_has_fixed_gutter_width(app):
    spine = StageSpine("#2a788e")
    assert spine.width() == StageSpine.GUTTER_WIDTH
