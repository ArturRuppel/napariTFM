import pytest

from napariTFM.widgets import _ui_style


@pytest.fixture(autouse=True)
def _restore_theme():
    original = _ui_style.active_theme_name()
    yield
    _ui_style.set_active_theme(original)


def test_theme_names_nonempty_and_contains_default():
    names = _ui_style.theme_names()
    assert len(names) >= 2
    assert _ui_style.active_theme_name() in names


def test_stage_accent_resolves_through_active_ramp():
    _ui_style.set_active_theme("Viridis")
    assert _ui_style.stage_accent("preprocessing") == _ui_style._sample_ramp(
        _ui_style.THEME_RAMPS["Viridis"], _ui_style.STAGE_RAMP_POSITION["preprocessing"]
    )


def test_stage_accent_unknown_key_falls_back_to_inputs():
    assert _ui_style.stage_accent("nope") == _ui_style.stage_accent("inputs")


def test_set_active_theme_changes_resolved_accent():
    names = _ui_style.theme_names()
    other = next(n for n in names if n != _ui_style.active_theme_name())
    before = _ui_style.stage_accent("preprocessing")
    _ui_style.set_active_theme(other)
    after = _ui_style.stage_accent("preprocessing")
    assert _ui_style.active_theme_name() == other
    differs = any(
        _ui_style.stage_accent(k) != before
        for k in ("preprocessing", "displacement", "force", "stress", "batch")
    ) or after != before
    assert differs


@pytest.fixture
def app():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_stage_section_set_accent_restyles_header(app):
    from qtpy.QtWidgets import QWidget
    from napariTFM.widgets._stage_section import StageSection
    from napariTFM.widgets._ui_style import muted_accent

    # The header title is an accent pill whose color is the muted accent.
    section = StageSection("Force Analysis", QWidget(), accent="#111111")
    assert muted_accent("#111111") in section.header_label.styleSheet()
    section.set_accent("#abcdef")
    assert muted_accent("#abcdef") in section.header_label.styleSheet()
