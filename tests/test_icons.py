import pytest
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from napariTFM.widgets._icons import (
    ICON_NAMES,
    stage_action_icon,
    stage_action_pixmap,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _opaque_count(pixmap):
    image = pixmap.toImage()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    )


def test_icon_names_cover_the_header_action_set():
    assert set(ICON_NAMES) == {"files", "params", "preview", "run", "cancel", "power", "plus"}


def test_every_action_icon_renders_visible_pixels(app):
    for name in ICON_NAMES:
        pixmap = stage_action_pixmap(name, "#2a788e", size=18)
        assert pixmap.size().width() == 18
        assert _opaque_count(pixmap) > 0, f"{name} rendered blank"


def test_tint_color_changes_the_rendered_pixels(app):
    red = stage_action_pixmap("run", "#ff0000", size=18).toImage()
    blue = stage_action_pixmap("run", "#0000ff", size=18).toImage()
    assert red != blue


def test_stage_action_icon_returns_a_non_null_icon(app):
    icon = stage_action_icon("params", "#2a788e")
    assert isinstance(icon, QIcon)
    assert not icon.isNull()


def test_disabled_pixmap_differs_from_normal_when_disabled_color_given(app):
    icon = stage_action_icon("run", "#2a788e", disabled_color="#3a3a3a", size=18)
    from qtpy.QtCore import QSize

    normal = icon.pixmap(QSize(18, 18), QIcon.Normal).toImage()
    disabled = icon.pixmap(QSize(18, 18), QIcon.Disabled).toImage()
    assert normal != disabled


def test_unknown_icon_name_raises(app):
    with pytest.raises(KeyError):
        stage_action_pixmap("nope", "#2a788e")


def test_plus_icon_is_registered_and_renders_opaque(app):
    from napariTFM.widgets._icons import ICON_NAMES, stage_action_pixmap

    assert "plus" in ICON_NAMES
    pm = stage_action_pixmap("plus", "#7ad151", size=18)
    img = pm.toImage()
    opaque = sum(
        img.pixelColor(x, y).alpha() > 0
        for x in range(img.width())
        for y in range(img.height())
    )
    assert opaque > 0
