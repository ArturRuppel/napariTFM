from napariTFM.widgets._ui_style import (
    STAGE_ACCENTS,
    muted_stage_accent,
    stage_accent,
)


def test_stage_accent_returns_palette_color_for_known_key():
    assert stage_accent("preprocessing") == STAGE_ACCENTS["preprocessing"]
    assert stage_accent("displacement") == STAGE_ACCENTS["displacement"]


def test_stage_accent_falls_back_to_inputs_for_unknown_key():
    assert stage_accent("nonexistent_stage") == STAGE_ACCENTS["inputs"]


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
