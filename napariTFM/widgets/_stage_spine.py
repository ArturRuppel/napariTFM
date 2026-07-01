"""Left-gutter spine + status node for a workflow stage (the colormap rail)."""
from __future__ import annotations

from typing import Optional, Tuple

from qtpy.QtCore import QEvent, QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from qtpy.QtWidgets import QSizePolicy, QToolTip, QWidget

# status -> node appearance; muted grey for inert, amber for active.
_RUNNING = "#e3b341"
_ERROR = "#d62828"
_DIM = "#6b7484"
_OFF = "#3c424c"  # recessed: a stage deliberately turned off (not "missing")

# How each status reads in a node's tooltip. "done" is the only state that has
# output pixels to bring on screen, so it's the only one that advertises the
# click; the rest describe the stage's state so the empty click isn't a mystery.
_STATUS_TOOLTIP = {
    "done": "computed — click to view",
    "running": "running…",
    "ready": "ready to run (no output yet)",
    "error": "failed",
    "not_started": "not started (no output yet)",
    "off": "disabled",
}


def _mix(hex_a: str, hex_b: str) -> QColor:
    """The colour halfway between two hex values (the rail's boundary tint)."""
    a, b = QColor(hex_a), QColor(hex_b)
    return QColor(
        (a.red() + b.red()) // 2,
        (a.green() + b.green()) // 2,
        (a.blue() + b.blue()) // 2,
    )


def _node_style(status: str, accent: str) -> Tuple[Optional[QColor], QColor]:
    """Return (fill, ring) for a node; fill None means a hollow ring."""
    if status == "done":
        return QColor(accent), QColor(accent)
    if status == "running":
        return QColor(_RUNNING), QColor(_RUNNING)
    if status == "ready":
        return None, QColor(accent)
    if status == "error":
        return QColor(_ERROR), QColor(_ERROR)
    if status == "off":
        return None, QColor(_OFF)
    return None, QColor(_DIM)


class StageSpine(QWidget):
    """A vertical gradient line + a status node, sized to its stage's height."""

    GUTTER_WIDTH = 28
    # Node centre from the top, aligned to the header pill row. The header is
    # the first row of the stage body (zero top margin) and is dominated by the
    # 22px action pills, so its vertical centre sits at ~11px (P8).
    NODE_Y = 11
    NODE_R = 6
    LINE_W = 2
    # Generous click target around the small painted node.
    NODE_HIT_R = 14

    # Clicking the node decodes this stage's output series into the viewer
    # (display-only, on demand) — the dot's status is already shown eagerly.
    clicked = Signal()

    def __init__(self, accent: str, status: str = "not_started", parent=None, *, label: str = ""):
        super().__init__(parent)
        self._accent = accent
        self._accent_above = accent
        self._accent_below = accent
        self._status = status
        # The stage's human name, used only to caption the node's tooltip.
        self._label = label
        # True while the cursor is over the clickable node — drives a hover halo
        # and the pointing-hand cursor so the small dot reads as a button.
        self._hover = False
        # Tooltips/hover only matter over the node, which is a fraction of the
        # tall gutter, so track motion to know when the cursor is on it.
        self.setMouseTracking(True)
        # Fractional completion (0..1) of an in-flight "running" stage, or None
        # when no per-frame progress is known. None falls back to the original
        # solid-fill node (the historical "running" look), which is also what a
        # single-frame stage shows since its only update is 0%→100%.
        self._progress: Optional[float] = None
        self.setFixedWidth(self.GUTTER_WIDTH)
        # Fixed width, but vertically the spine only *follows* its stage's height
        # rather than expanding to grab free space — otherwise each section
        # balloons and the stages drift apart. The box layout still stretches it
        # to fill the row, so its painted segment stays full-height and the rail
        # remains continuous.
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

    def set_status(self, status: str) -> None:
        self._status = status
        if status != "running":
            # Stale progress must not leak into the next run of this stage, or
            # the node would briefly flash a half-filled ring on its next start.
            self._progress = None
        self.update()

    def set_progress(self, fraction: Optional[float]) -> None:
        """Set the in-flight fractional completion (0..1) of a running stage.

        Only visible while ``status == "running"``; harmless to call at other
        times since :meth:`paintEvent` ignores it then. Pass ``None`` to fall
        back to the plain solid-fill "running" node.
        """
        self._progress = None if fraction is None else max(0.0, min(1.0, fraction))
        self.update()

    def set_accents(self, accent: str, above: Optional[str] = None, below: Optional[str] = None) -> None:
        self._accent = accent
        self._accent_above = above or accent
        self._accent_below = below or accent
        self.update()

    def _gradient_stops(self) -> list[Tuple[float, QColor]]:
        """Two stops that make this segment a slice of one uniform rail.

        The top stop is the colour midway between this stage and the one above;
        the bottom stop is midway to the stage below. Because neighbouring
        segments share those midpoints, the rail is continuous across nodes
        instead of jumping — a single gradient, drawn in pieces.
        """
        top = _mix(self._accent_above, self._accent)
        bottom = _mix(self._accent, self._accent_below)
        return [(0.0, top), (1.0, bottom)]

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = self.width() / 2.0
        h = self.height()

        gradient = QLinearGradient(0.0, 0.0, 0.0, float(h))
        for position, color in self._gradient_stops():
            gradient.setColorAt(position, color)
        pen = QPen(QBrush(gradient), self.LINE_W)
        pen.setCapStyle(Qt.FlatCap)
        # A disabled stage recedes: dim and dash its segment of the rail so the
        # colormap visibly "skips" it rather than reading as a broken pipeline.
        if self._status == "off":
            painter.setOpacity(0.45)
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(cx), 0, int(cx), int(h))
        painter.setOpacity(1.0)

        fill, ring = _node_style(self._status, self._accent)
        r = self.NODE_R
        if self._hover and self._status != "off":
            # A soft halo behind the node on hover: the small dot is a button, so
            # it should light up under the cursor the way a button does.
            halo = QColor(ring)
            halo.setAlpha(70)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(halo))
            hr = r + 4
            painter.drawEllipse(QRectF(cx - hr, self.NODE_Y - hr, 2 * hr, 2 * hr))
        if self._status == "off":
            # A short horizontal dash ("—") reads as a skipped/off stage.
            painter.setPen(QPen(ring, 2, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(int(cx - r), self.NODE_Y, int(cx + r), self.NODE_Y)
            painter.end()
            return
        rect = QRectF(cx - r, self.NODE_Y - r, 2 * r, 2 * r)
        if self._status == "running" and self._progress is not None:
            # A pie wedge growing clockwise from 12 o'clock reads as a fill
            # level, so the node itself doubles as a tiny per-stage progress
            # ring instead of just a flat "something is happening" amber dot.
            painter.setPen(QPen(ring, 2))
            painter.setBrush(QBrush(self.palette().color(self.backgroundRole())))
            painter.drawEllipse(rect)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(ring))
            span = -round(360 * 16 * self._progress)
            painter.drawPie(rect, 90 * 16, span)
            painter.end()
            return
        centre = fill if fill is not None else self.palette().color(self.backgroundRole())
        painter.setPen(QPen(ring, 2))
        painter.setBrush(QBrush(centre))
        painter.drawEllipse(rect)
        painter.end()

    def _over_node(self, pos) -> bool:
        """True when *pos* falls within the node's (generous) click target."""
        if self._status == "off":
            return False
        dx = pos.x() - self.width() / 2.0
        dy = pos.y() - self.NODE_Y
        return dx * dx + dy * dy <= self.NODE_HIT_R * self.NODE_HIT_R

    def _tooltip_text(self) -> str:
        phrase = _STATUS_TOOLTIP.get(self._status, self._status)
        return f"{self._label}: {phrase}" if self._label else phrase

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI event
        if self._over_node(event.pos()):
            self.clicked.emit()

    def mouseMoveEvent(self, event) -> None:  # pragma: no cover - GUI event
        over = self._over_node(event.pos())
        self.setCursor(Qt.PointingHandCursor if over else Qt.ArrowCursor)
        if over != self._hover:
            self._hover = over
            self.update()

    def leaveEvent(self, _event) -> None:  # pragma: no cover - GUI event
        if self._hover:
            self._hover = False
            self.update()

    def event(self, event):  # pragma: no cover - GUI event
        # Only surface the tooltip over the node itself, not the whole tall
        # gutter, so hovering the bare rail says nothing misleading.
        if event.type() == QEvent.ToolTip:
            if self._over_node(event.pos()):
                QToolTip.showText(event.globalPos(), self._tooltip_text(), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)
