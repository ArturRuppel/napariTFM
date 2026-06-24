"""Experiments list (top-of-panel substrate): mini-rails + selectable rows."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._stage_spine import _node_style
from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    section_label_style,
    stage_accent,
)

# The four pipeline stages a mini-rail summarises (project/batch are not dots).
PIPELINE_STAGES = ("preprocessing", "displacement", "force", "stress")


class MiniRail(QWidget):
    """A compact horizontal row of per-stage status dots for one experiment."""

    DOT_R = 4
    DOT_GAP = 12

    def __init__(self, stages=PIPELINE_STAGES, parent=None):
        super().__init__(parent)
        self.stages = tuple(stages)
        self._statuses = {key: "not_started" for key in self.stages}
        self.setFixedSize(self.DOT_GAP * len(self.stages), 2 * self.DOT_R + 6)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_statuses(self, statuses: dict[str, str]) -> None:
        for key in self.stages:
            if key in statuses:
                self._statuses[key] = statuses[key]
        self.update()

    def appearance(self, stage: str) -> tuple[Optional[str], str]:
        """Return (fill_hex_or_None, ring_hex) for a stage dot — used by tests/paint."""
        fill, ring = _node_style(self._statuses[stage], stage_accent(stage))
        return (fill.name() if fill is not None else None, ring.name())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2.0
        r = self.DOT_R
        for i, stage in enumerate(self.stages):
            cx = self.DOT_GAP * i + self.DOT_GAP / 2.0
            fill, ring = _node_style(self._statuses[stage], stage_accent(stage))
            if self._statuses[stage] == "off":
                painter.setPen(QPen(ring, 2, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
                continue
            centre = fill if fill is not None else self.palette().color(self.backgroundRole())
            painter.setPen(QPen(ring, 1.5))
            painter.setBrush(QBrush(centre))
            painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        painter.end()
