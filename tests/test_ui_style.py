import pytest
from qtpy.QtWidgets import QApplication, QGridLayout, QLabel

from napariTFM.widgets._ui_style import (
    MUTED_TEXT_COLOR,
    STAGE_ACCENTS,
    caption_style,
    danger_text_style,
    muted_stage_accent,
    section_label_style,
    stage_accent,
    stage_header_style,
    title_style,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_stage_accent_returns_palette_color_for_known_key():
    from napariTFM.widgets._ui_style import ACTIVE_PALETTE
    assert stage_accent("preprocessing") == ACTIVE_PALETTE[STAGE_ACCENTS["preprocessing"]]
    assert stage_accent("displacement") == ACTIVE_PALETTE[STAGE_ACCENTS["displacement"]]


def test_stage_accent_falls_back_to_inputs_for_unknown_key():
    assert stage_accent("nonexistent_stage") == stage_accent("inputs")


def test_muted_stage_accent_reduces_saturation():
    full = stage_accent("preprocessing").lstrip("#")
    muted = muted_stage_accent("preprocessing").lstrip("#")

    assert muted != full
    assert len(muted) == 6


def test_muted_stage_accent_preserves_hue_family():
    muted = muted_stage_accent("preprocessing").lstrip("#")
    r, g, b = int(muted[0:2], 16), int(muted[2:4], 16), int(muted[4:6], 16)
    assert b > r


def test_muted_stage_accent_falls_back_for_unknown_key():
    assert muted_stage_accent("nonexistent") == muted_stage_accent("inputs")


def test_caption_style_uses_muted_text_color():
    style = caption_style()
    assert MUTED_TEXT_COLOR in style
    assert "9pt" in style


def test_title_style_is_bold_and_sized():
    assert title_style() == "font-weight: bold; font-size: 14px;"


def test_section_label_style_is_bold():
    assert section_label_style() == "font-weight: bold;"


def test_danger_text_style_is_red():
    assert danger_text_style() == "color: red;"


def test_stage_header_style_embeds_accent():
    accent = stage_accent("preprocessing")
    style = stage_header_style(accent)
    assert accent in style
    assert "font-weight: bold" in style


def test_muted_accent_desaturates_and_flattens():
    from napariTFM.widgets._ui_style import muted_accent

    out = muted_accent("#3b6fb6")
    assert out.startswith("#") and len(out) == 7
    # idempotent shape: feeding the output back stays a valid hex
    assert muted_accent(out).startswith("#")


def test_layout_constants_present():
    from napariTFM.widgets import _ui_style

    assert _ui_style.TINY_MARGIN == 2
    assert _ui_style.SECTION_MARGIN == 4
    assert _ui_style.TIGHT_SPACING == 4


def test_section_grid_has_four_columns_with_stretchy_fields(app):
    from napariTFM.widgets._ui_style import section_grid

    grid = section_grid()
    assert isinstance(grid, QGridLayout)
    assert grid.columnStretch(0) == 0
    assert grid.columnStretch(1) == 1
    assert grid.columnStretch(2) == 0
    assert grid.columnStretch(3) == 1


def test_add_section_pair_row_places_both_pairs(app):
    from napariTFM.widgets._ui_style import section_grid, add_section_pair_row

    grid = section_grid()
    add_section_pair_row(grid, 0, "Left", QLabel("L"), "Right", QLabel("R"))

    assert grid.itemAtPosition(0, 0) is not None
    assert grid.itemAtPosition(0, 2) is not None


def test_add_section_pair_row_left_only_leaves_right_empty(app):
    from napariTFM.widgets._ui_style import section_grid, add_section_pair_row

    grid = section_grid()
    add_section_pair_row(grid, 0, "Left", QLabel("L"))

    assert grid.itemAtPosition(0, 0) is not None
    assert grid.itemAtPosition(0, 2) is None


def test_add_section_header_spans_all_four_columns(app):
    from napariTFM.widgets._ui_style import section_grid, add_section_header

    grid = section_grid()
    header = add_section_header(grid, 0, QLabel("Title"))

    assert header is not None
    assert grid.itemAtPosition(0, 0) is grid.itemAtPosition(0, 3)
