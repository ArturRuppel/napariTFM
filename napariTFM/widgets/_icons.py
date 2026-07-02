"""Crisp, accent-tinted vector icons for stage-header action buttons.

Rendered from inline SVG via QtSvg so they re-tint with the active theme and
stay sharp at any DPI — no bundled raster assets. Each body carries a ``{c}``
colour placeholder; stroke icons inherit the wrapper stroke, filled icons
override ``fill`` on their own element.
"""
from __future__ import annotations

from qtpy.QtCore import QByteArray, QRectF, QSize, Qt
from qtpy.QtGui import QIcon, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer

_ICON_BODIES = {
    # magnifier — "show this stage's data"
    "files": '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/>',
    # sliders — "tune this stage's parameters"
    "params": (
        '<line x1="3" y1="8" x2="21" y2="8"/>'
        '<circle cx="9" cy="8" r="2.4" fill="{c}" stroke="none"/>'
        '<line x1="3" y1="16" x2="21" y2="16"/>'
        '<circle cx="15" cy="16" r="2.4" fill="{c}" stroke="none"/>'
    ),
    # eye — "preview this stage"
    "preview": (
        '<path d="M2 12 C5 6 19 6 22 12 C19 18 5 18 2 12 Z"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    # play triangle — "run this stage"
    "run": '<path d="M8 5 L19 12 L8 19 Z" fill="{c}" stroke="none"/>',
    # stop square — "cancel the running stage"
    "cancel": '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="{c}" stroke="none"/>',
    # power — "enable / disable this stage"
    "power": (
        '<path d="M12 3 L12 11"/>'
        '<path d="M6.4 6.8 A8 8 0 1 0 17.6 6.8"/>'
    ),
    # plus — "add folders"
    "plus": '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
    # crosshair target — "auto-select the optimal regularization (GCV)"
    "gcv": (
        '<circle cx="12" cy="12" r="6.5"/>'
        '<line x1="12" y1="1.5" x2="12" y2="5"/>'
        '<line x1="12" y1="19" x2="12" y2="22.5"/>'
        '<line x1="1.5" y1="12" x2="5" y2="12"/>'
        '<line x1="19" y1="12" x2="22.5" y2="12"/>'
    ),
    # page with a folded corner + plus — "start a new project"
    "new": (
        '<path d="M6 3 H14 L18 7 V21 H6 Z"/>'
        '<path d="M14 3 V7 H18"/>'
        '<path d="M9 15 H15"/>'
        '<path d="M12 12 V18"/>'
    ),
    # tabbed folder — "load / open a project or a preset"
    "load": '<path d="M3 6 H9 L11 8 H21 V19 H3 Z"/>',
    # floppy disk — "save a project or a preset"
    "save": (
        '<path d="M5 4 H15.5 L20 8.5 V20 H5 Z"/>'
        '<path d="M8 20 V14 H16 V20"/>'
        '<path d="M9 4 V8 H13.5 V4"/>'
    ),
    # counterclockwise rewind — "reset parameters / reload"
    "reset": (
        '<path d="M5.5 6.5 A8.5 8.5 0 1 0 18 6"/>'
        '<path d="M6.06 8.89 L6.80 4.01 L2.94 7.08"/>'
    ),
}

_SVG_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{c}" stroke-width="{w}" '
    'stroke-linecap="{cap}" stroke-linejoin="{join}">{body}</svg>'
)

ICON_NAMES = tuple(_ICON_BODIES)


def _render(svg: str, size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pixmap


def stage_action_pixmap(
    name: str,
    color: str,
    size: int = 18,
    stroke_width: float = 2.0,
    linecap: str = "round",
    linejoin: str = "round",
) -> QPixmap:
    """Render a single tinted action icon to a transparent pixmap."""
    body = _ICON_BODIES[name].format(c=color)
    svg = _SVG_TEMPLATE.format(
        c=color, w=stroke_width, cap=linecap, join=linejoin, body=body
    )
    return _render(svg, size)


def stage_action_icon(
    name: str,
    color: str,
    disabled_color: str | None = None,
    size: int = 18,
    stroke_width: float = 2.0,
    linecap: str = "round",
    linejoin: str = "round",
) -> QIcon:
    """A QIcon for an action button, optionally carrying a dimmed disabled mode."""
    icon = QIcon(
        stage_action_pixmap(name, color, size, stroke_width, linecap, linejoin)
    )
    if disabled_color is not None:
        icon.addPixmap(
            stage_action_pixmap(
                name, disabled_color, size, stroke_width, linecap, linejoin
            ),
            QIcon.Disabled,
        )
    return icon
