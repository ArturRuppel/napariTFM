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


def test_stage_accent_resolves_through_active_palette():
    semantic = _ui_style.STAGE_ACCENTS["preprocessing"]
    expected = _ui_style.ACTIVE_PALETTE[semantic]
    assert _ui_style.stage_accent("preprocessing") == expected


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


def test_status_states_trimmed_to_computed():
    # 'stale' is never produced by StageDataStatusPanel.refresh(); drop it.
    assert "stale" not in _ui_style.STATUS_COLORS
    assert "stale" not in _ui_style.STATUS_GLYPHS
    # The states that ARE produced/used must remain.
    for s in ("not_started", "ready", "running", "done", "error"):
        assert s in _ui_style.STATUS_COLORS
