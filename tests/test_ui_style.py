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


def test_stage_accent_returns_hex_for_known_keys():
    assert stage_accent("preprocessing").startswith("#")
    assert stage_accent("displacement").startswith("#")
    assert stage_accent("preprocessing") != stage_accent("displacement")


def test_stage_accent_samples_active_ramp_in_pipeline_order():
    from napariTFM.widgets import _ui_style
    _ui_style.set_active_theme("Viridis")
    # project/inputs sit at the ramp start; batch at the end.
    assert _ui_style.stage_accent("project") == _ui_style.THEME_RAMPS["Viridis"][0]
    assert _ui_style.stage_accent("batch") == _ui_style.THEME_RAMPS["Viridis"][-1]
    # adjacent pipeline stages are visibly distinct (the anti-mud guarantee).
    order = ["preprocessing", "displacement", "force", "stress"]
    accents = [_ui_style.stage_accent(k) for k in order]
    assert len(set(accents)) == len(accents)


def test_cividis_stages_match_cellflow_ordered_stops():
    """The Cividis spine reuses CellFlow's five ordered cividis stops, mapped
    project(yellow)→stress(dark blue) — project first, stress analysis last."""
    from napariTFM.widgets import _ui_style
    _ui_style.set_active_theme("Cividis")
    assert _ui_style.stage_accent("project") == "#d6c35d"
    assert _ui_style.stage_accent("preprocessing") == "#a79d73"
    assert _ui_style.stage_accent("displacement") == "#7d7c78"
    assert _ui_style.stage_accent("force") == "#555c6d"
    assert _ui_style.stage_accent("stress") == "#243c6e"


def test_stage_accent_falls_back_to_inputs_for_unknown_key():
    assert stage_accent("nonexistent_stage") == stage_accent("inputs")


def test_muted_stage_accent_reduces_saturation():
    full = stage_accent("preprocessing").lstrip("#")
    muted = muted_stage_accent("preprocessing").lstrip("#")

    assert muted != full
    assert len(muted) == 6


def test_muted_stage_accent_preserves_hue_family():
    import colorsys

    def hue(hex_str: str) -> float:
        c = hex_str.lstrip("#")
        r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        return colorsys.rgb_to_hls(r, g, b)[0]

    full = stage_accent("preprocessing")
    muted = muted_stage_accent("preprocessing")
    # Muting drops saturation/lightness but keeps the hue (the colour family).
    assert abs(hue(full) - hue(muted)) < 0.02


def test_muted_stage_accent_falls_back_for_unknown_key():
    assert muted_stage_accent("nonexistent") == muted_stage_accent("inputs")


def test_file_status_color_maps_presence_to_red_green():
    from napariTFM.widgets._ui_style import file_status_color

    # Present (in cache or on disk) reads green; required-and-absent reads red.
    # The hues are muted (quiet, not alarming) but stay clearly red vs green.
    assert file_status_color("present") == "#5e9468"
    assert file_status_color("missing") == "#b05751"
    # Optional-and-absent is a quiet grey, not an alarming red.
    assert file_status_color("optional") == "#5b626d"
    assert file_status_color("error") == "#c2a04e"
    # Unknown states fall back to the optional grey rather than crashing.
    assert file_status_color("nonsense") == "#5b626d"


def test_file_status_state_classifies_availability():
    from napariTFM.widgets._ui_style import file_status_state

    assert file_status_state(available=True, required=True, error=False) == "present"
    assert file_status_state(available=False, required=True, error=False) == "missing"
    assert file_status_state(available=False, required=False, error=False) == "optional"
    assert file_status_state(available=True, required=True, error=True) == "error"


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


def test_stage_header_style_is_accent_pill():
    from napariTFM.widgets._ui_style import muted_accent
    accent = stage_accent("preprocessing")
    style = stage_header_style(accent)
    assert muted_accent(accent) in style
    assert "font-weight: bold" in style
    assert "border-radius" in style
    assert "background-color" in style


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


def test_stage_header_pill_background_is_rgba_of_muted_accent():
    from napariTFM.widgets._ui_style import stage_header_pill_background, muted_accent

    accent = stage_accent("force")
    bg = stage_header_pill_background(accent, alpha=38)
    muted = muted_accent(accent).lstrip("#")
    r, g, b = int(muted[0:2], 16), int(muted[2:4], 16), int(muted[4:6], 16)
    assert bg == f"rgba({r}, {g}, {b}, 38)"


def test_stage_header_disabled_action_color_is_hex():
    from napariTFM.widgets._ui_style import stage_header_disabled_action_color

    out = stage_header_disabled_action_color(stage_accent("displacement"))
    assert out.startswith("#") and len(out) == 7


def test_stage_header_action_button_style_has_state_rules():
    from napariTFM.widgets._ui_style import stage_header_action_button_style

    style = stage_header_action_button_style(stage_accent("preprocessing"))
    assert "QToolButton {" in style
    assert "QToolButton:hover" in style
    assert "QToolButton:checked" in style
    assert "QToolButton:disabled" in style
    assert "border-radius" in style


def test_make_stage_action_button_carries_glyph_and_is_fixed(app):
    from qtpy.QtWidgets import QToolButton
    from napariTFM.widgets._ui_style import make_stage_action_button, STAGE_ACTION_BUTTON_SIZE

    btn = make_stage_action_button(None, "stage_x_run_button", "Run", "▶", stage_accent("force"))
    assert isinstance(btn, QToolButton)
    assert btn.text() == "▶"
    assert btn.objectName() == "stage_x_run_button"
    assert btn.width() == STAGE_ACTION_BUTTON_SIZE
    assert btn.isCheckable() is False


def test_make_stage_action_button_with_icon_name_uses_vector_icon(app):
    from napariTFM.widgets._ui_style import make_stage_action_button

    btn = make_stage_action_button(
        None, "stage_x_run_button", "Run", "▶", stage_accent("force"), icon_name="run"
    )
    assert btn.text() == ""
    assert not btn.icon().isNull()
