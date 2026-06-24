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


def test_node_style_off_is_recessed_and_distinct_from_not_started():
    off_fill, off_ring = _node_style("off", "#2a788e")
    _, idle_ring = _node_style("not_started", "#2a788e")
    assert off_fill is None
    # "off" must be visually distinct from a not-yet-run stage.
    assert off_ring.name() != idle_ring.name()


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


def test_stage_section_owns_a_spine_and_forwards_status(app):
    from napariTFM.widgets._stage_section import StageSection
    from qtpy.QtWidgets import QLabel
    section = StageSection("Force", QLabel("body"), status="ready")
    assert isinstance(section.spine, StageSpine)
    assert section.spine._status == "ready"
    section.set_status("done")
    assert section.spine._status == "done"


def test_stage_section_set_accents_forwards_to_spine(app):
    from napariTFM.widgets._stage_section import StageSection
    from qtpy.QtWidgets import QLabel
    section = StageSection("Force", QLabel("body"))
    section.set_accents("#22a884", above="#2a788e", below="#7ad151")
    assert section.spine._accent_above == "#2a788e"
    assert section.spine._accent_below == "#7ad151"


def test_spine_node_aligns_vertically_with_header_pills(app):
    """The spine status node centres on the header pill row (P8)."""
    from napariTFM.widgets._stage_section import StageSection
    from qtpy.QtWidgets import QWidget

    section = StageSection("Preprocessing", QWidget(), status="ready")
    section.resize(360, 120)
    section.show()
    app.processEvents()

    pill = section.run_cancel_btn
    pill_centre_y = pill.mapTo(section, pill.rect().topLeft()).y() + pill.height() / 2.0
    spine_top_y = section.spine.mapTo(section, section.spine.rect().topLeft()).y()
    node_centre_y = spine_top_y + section.spine.NODE_Y

    assert abs(node_centre_y - pill_centre_y) <= 1
