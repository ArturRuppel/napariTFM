from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._ui_style import (
    ACTION_GLYPHS,
    COMPACT_SPACING,
    STATUS_GLYPHS,
)


@dataclass(frozen=True)
class DataArtifactSpec:
    key: str
    label: str
    attr: str | None
    role: str = "input"
    required: bool = True
    on_view: Callable[[], None] | None = None
    on_action: Callable[[], None] | None = None


class _ArtifactRow(QWidget):
    """Single CellFlow-style artifact row."""

    def __init__(self, spec: DataArtifactSpec):
        super().__init__()
        self.spec = spec
        self.view_btn: QToolButton | None = None
        self.action_btn: QToolButton | None = None

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.setLayout(layout)

        self.glyph_label = QLabel("○")
        self.glyph_label.setFixedWidth(14)
        layout.addWidget(self.glyph_label)

        self.name_label = QLabel(spec.label)
        self.name_label.setMinimumWidth(135)
        layout.addWidget(self.name_label)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.info_label, stretch=1)

        if spec.on_view is not None:
            self.view_btn = QToolButton()
            self.view_btn.setText(ACTION_GLYPHS["view"])
            self.view_btn.setObjectName(f"stage_artifact_{spec.key}_view_btn")
            self.view_btn.setToolTip(f"View {spec.label} in viewer")
            self.view_btn.clicked.connect(spec.on_view)
            layout.addWidget(self.view_btn)

        if spec.on_action is not None:
            self.action_btn = QToolButton()
            glyph = ACTION_GLYPHS["save"] if spec.role == "output" else ACTION_GLYPHS["load"]
            self.action_btn.setText(glyph)
            self.action_btn.setObjectName(f"stage_artifact_{spec.key}_action_btn")
            action_label = "Save" if spec.role == "output" else "Load"
            self.action_btn.setToolTip(f"{action_label} {spec.label}")
            self.action_btn.clicked.connect(spec.on_action)
            layout.addWidget(self.action_btn)

    def refresh(self, available: bool, info_text: str) -> None:
        if available:
            self.glyph_label.setText(STATUS_GLYPHS["available"])
        elif self.spec.required:
            self.glyph_label.setText(STATUS_GLYPHS["missing_required"])
        else:
            self.glyph_label.setText(STATUS_GLYPHS["missing_optional"])

        self.info_label.setText(info_text)

        if self.view_btn is not None:
            self.view_btn.setVisible(available)
        if self.action_btn is not None and self.spec.role == "output":
            self.action_btn.setEnabled(available)

    def refresh_state(self, state, info_text: str) -> None:
        available = getattr(state, "value", None) is not None
        error = getattr(state, "error", "")
        path = getattr(state, "path", None)
        dirty = bool(getattr(state, "dirty", False))

        if error:
            self.glyph_label.setText(STATUS_GLYPHS["error"])
            self.info_label.setText(str(error))
            self.info_label.setToolTip(str(error))
            if self.view_btn is not None:
                self.view_btn.setVisible(available)
            if self.action_btn is not None and self.spec.role == "output":
                self.action_btn.setEnabled(available)
            return

        self.refresh(available=available, info_text=info_text)
        hints = []
        if available and dirty:
            hints.append("Unsaved")
        if available and path is not None and not dirty:
            hints.append(Path(path).name)
            self.info_label.setToolTip(str(path))
        else:
            self.info_label.setToolTip("")
        if hints:
            self.info_label.setText(f"{self.info_label.text()} · {' · '.join(hints)}")


class StageDataStatusPanel(QWidget):
    """Compact, always-visible summary of a stage's data dependencies."""

    def __init__(self, stage_key: str, data_manager: Any, artifacts: list[DataArtifactSpec]):
        super().__init__()
        self.stage_key = stage_key
        self.data_manager = data_manager
        self.artifacts = artifacts
        self.artifact_rows: dict[str, _ArtifactRow] = {}
        self.setObjectName(f"stage_{stage_key}_data_status_panel")
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 0, 0, 2)
        layout.setSpacing(COMPACT_SPACING)
        self.setLayout(layout)

        input_artifacts = [artifact for artifact in self.artifacts if artifact.role == "input"]
        output_artifacts = [artifact for artifact in self.artifacts if artifact.role == "output"]

        if input_artifacts:
            inputs_header = QLabel("Inputs")
            inputs_header.setStyleSheet("color: #999; font-size: 9pt;")
            layout.addWidget(inputs_header)
            for artifact in input_artifacts:
                self._add_row(layout, artifact)

        if output_artifacts:
            outputs_header = QLabel("Outputs")
            outputs_header.setStyleSheet("color: #999; font-size: 9pt;")
            layout.addWidget(outputs_header)
            for artifact in output_artifacts:
                self._add_row(layout, artifact)

        # Legacy compatibility: existing tests and call sites read info labels.
        self.artifact_labels = {
            key: row.info_label for key, row in self.artifact_rows.items()
        }

    def _add_row(self, layout: QVBoxLayout, artifact: DataArtifactSpec):
        row = _ArtifactRow(artifact)
        self.artifact_rows[artifact.key] = row
        layout.addWidget(row)

    def refresh(self) -> str:
        required_inputs_available = True
        output_available = False

        for artifact in self.artifacts:
            state = self._artifact_state(artifact)
            value = state.value if state is not None else self._artifact_value(artifact)
            available = value is not None
            if artifact.role == "input" and artifact.required and not available:
                required_inputs_available = False
            if artifact.role == "output" and available:
                output_available = True

            info_text = self._info_text(artifact, value, available)
            if state is not None:
                self.artifact_rows[artifact.key].refresh_state(state, info_text=info_text)
            else:
                self.artifact_rows[artifact.key].refresh(available=available, info_text=info_text)

        if output_available:
            return "done"
        if required_inputs_available:
            return "ready"
        return "not_started"

    def _info_text(self, artifact: DataArtifactSpec, value: Any, available: bool) -> str:
        if available:
            return self._shape_text(value) or "Loaded"
        return "Missing" if artifact.required else "Optional"

    @staticmethod
    def _shape_text(value: Any) -> str:
        try:
            shape = getattr(value, "shape", None)
            if shape is not None:
                return "×".join(str(size) for size in shape)
        except Exception:
            pass

        for attr in ("displacement_field", "force_field", "stress_tensor"):
            array = getattr(value, attr, None)
            if array is not None and hasattr(array, "shape"):
                return "×".join(str(size) for size in array.shape)
        return ""

    def _artifact_value(self, artifact: DataArtifactSpec):
        if artifact.attr is None:
            return None
        return getattr(self.data_manager, artifact.attr, None)

    def _artifact_state(self, artifact: DataArtifactSpec):
        get_artifact = getattr(self.data_manager, "get_artifact", None)
        if get_artifact is None:
            return None
        try:
            return get_artifact(artifact.key)
        except (KeyError, AttributeError):
            return None
