"""A row of red→green status dots for a stage's input/output files.

Each dot is one artifact: red (a required input/output is missing), grey (an
optional input is absent), green (present — in cache or on disk), amber (failed
to load). Inputs sit left of an arrow, outputs to its right, so the row reads as
"these inputs produce these outputs when the stage runs". Clicking a present
(green) dot views it in napari; clicking a missing input dot assigns the active
napari layer; a missing output is inert.
"""
from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from napariTFM.widgets._stage_data_status import (
    DataArtifactSpec,
    artifact_available,
    artifact_info_text,
    artifact_state,
    compute_stage_status,
)
from napariTFM.widgets._ui_style import (
    COMPACT_SPACING,
    caption_style,
    file_status_color,
    file_status_state,
)

DOT_SIZE = 12
_DOT_RADIUS = DOT_SIZE // 2


def _dot_style(color: str, enabled: bool) -> str:
    border = "border: 1px solid rgba(0, 0, 0, 90);"
    base = (
        f"QToolButton {{ background-color: {color}; {border} "
        f"border-radius: {_DOT_RADIUS}px; }}"
    )
    if enabled:
        base += "QToolButton:hover { border: 1px solid white; }"
    return base


class StageFileStatusRow(QWidget):
    """Always-visible row of file-status dots for one workflow stage."""

    def __init__(self, stage_key: str, data_manager: Any, artifacts: list[DataArtifactSpec]):
        super().__init__()
        self.stage_key = stage_key
        self.data_manager = data_manager
        self.artifacts = artifacts
        self.dots: dict[str, QToolButton] = {}
        self._available: dict[str, bool] = {}
        self.setObjectName(f"stage_{stage_key}_file_status_row")
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout()
        layout.setContentsMargins(18, 0, 0, 2)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        inputs = [a for a in self.artifacts if a.role == "input"]
        outputs = [a for a in self.artifacts if a.role == "output"]

        for spec in inputs:
            layout.addWidget(self._make_dot(spec))

        self.arrow = QLabel("→")
        self.arrow.setStyleSheet(caption_style())
        self.arrow.setToolTip(
            "Inputs (left) become outputs (right) when this stage runs.\n"
            "Click a green dot to view it; click a red input dot to assign the "
            "active napari layer."
        )
        layout.addWidget(self.arrow)

        for spec in outputs:
            layout.addWidget(self._make_dot(spec))

        layout.addStretch(1)

    def _make_dot(self, spec: DataArtifactSpec) -> QToolButton:
        dot = QToolButton()
        dot.setObjectName(f"stage_artifact_{spec.key}_dot")
        dot.setFixedSize(DOT_SIZE, DOT_SIZE)
        dot.setAutoRaise(True)
        dot.clicked.connect(lambda _checked=False, s=spec: self._on_dot_clicked(s))
        self.dots[spec.key] = dot
        return dot

    def _on_dot_clicked(self, spec: DataArtifactSpec) -> None:
        if self._available.get(spec.key) and spec.on_view is not None:
            spec.on_view()
        elif not self._available.get(spec.key) and spec.on_action is not None:
            spec.on_action()

    def refresh(self) -> str:
        for spec in self.artifacts:
            available = artifact_available(self.data_manager, spec)
            state = artifact_state(self.data_manager, spec)
            error = bool(getattr(state, "error", "")) if state is not None else False
            self._available[spec.key] = available

            color = file_status_color(file_status_state(available, spec.required, error))
            clickable = (available and spec.on_view is not None) or (
                not available and spec.on_action is not None
            )
            dot = self.dots[spec.key]
            dot.setEnabled(clickable)
            dot.setCursor(Qt.PointingHandCursor if clickable else Qt.ArrowCursor)
            dot.setStyleSheet(_dot_style(color, clickable))
            dot.setToolTip(f"{spec.label} — {artifact_info_text(self.data_manager, spec)}")

        return compute_stage_status(self.data_manager, self.artifacts)
