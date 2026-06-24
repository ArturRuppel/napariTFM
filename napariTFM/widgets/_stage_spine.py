"""Left-gutter spine + status node for a workflow stage (the colormap rail)."""
from __future__ import annotations

from typing import Optional, Tuple

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

# status -> node appearance; muted grey for inert, amber for active.
_RUNNING = "#e3b341"
_ERROR = "#d62828"
_DIM = "#6b7484"
_OFF = "#3c424c"  # recessed: a stage deliberately turned off (not "missing")


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
    NODE_Y = 20      # node centre from the top, aligned to the header row
    NODE_R = 6
    LINE_W = 2

    def __init__(self, accent: str, status: str = "not_started", parent=None):
        super().__init__(parent)
        self._accent = accent
        self._accent_above = accent
        self._accent_below = accent
        self._status = status
        self.setFixedWidth(self.GUTTER_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def set_status(self, status: str) -> None:
        self._status = status
        self.update()

    def set_accents(self, accent: str, above: Optional[str] = None, below: Optional[str] = None) -> None:
        self._accent = accent
        self._accent_above = above or accent
        self._accent_below = below or accent
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = self.width() / 2.0
        h = self.height()

        gradient = QLinearGradient(0.0, 0.0, 0.0, float(h))
        gradient.setColorAt(0.0, QColor(self._accent_above))
        gradient.setColorAt(0.5, QColor(self._accent))
        gradient.setColorAt(1.0, QColor(self._accent_below))
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
        if self._status == "off":
            # A short horizontal dash ("—") reads as a skipped/off stage.
            painter.setPen(QPen(ring, 2, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(int(cx - r), self.NODE_Y, int(cx + r), self.NODE_Y)
            painter.end()
            return
        centre = fill if fill is not None else self.palette().color(self.backgroundRole())
        painter.setPen(QPen(ring, 2))
        painter.setBrush(QBrush(centre))
        painter.drawEllipse(QRectF(cx - r, self.NODE_Y - r, 2 * r, 2 * r))
        painter.end()
