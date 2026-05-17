from dataclasses import dataclass
from typing import Any

from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

from napariTFM.widgets._ui_style import COMPACT_SPACING


@dataclass(frozen=True)
class DataArtifactSpec:
    key: str
    label: str
    attr: str | None
    role: str = "input"
    required: bool = True


class StageDataStatusPanel(QWidget):
    """Compact, always-visible summary of a stage's data dependencies."""

    def __init__(self, stage_key: str, data_manager: Any, artifacts: list[DataArtifactSpec]):
        super().__init__()
        self.stage_key = stage_key
        self.data_manager = data_manager
        self.artifacts = artifacts
        self.artifact_labels: dict[str, QLabel] = {}
        self.setObjectName(f"stage_{stage_key}_data_status_panel")
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QGridLayout()
        layout.setContentsMargins(18, 0, 0, 2)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(COMPACT_SPACING)

        for index, artifact in enumerate(self.artifacts):
            label = QLabel()
            label.setObjectName(f"stage_{self.stage_key}_{artifact.key}_status_label")
            self.artifact_labels[artifact.key] = label
            layout.addWidget(label, index // 2, index % 2)

        self.setLayout(layout)

    def refresh(self) -> str:
        required_inputs_available = True
        output_available = False

        for artifact in self.artifacts:
            value = self._artifact_value(artifact)
            available = value is not None
            if artifact.role == "input" and artifact.required and not available:
                required_inputs_available = False
            if artifact.role == "output" and available:
                output_available = True
            self.artifact_labels[artifact.key].setText(
                f"{artifact.label}: {'available' if available else 'missing'}"
            )

        if output_available:
            return "done"
        if required_inputs_available:
            return "ready"
        return "not_started"

    def _artifact_value(self, artifact: DataArtifactSpec):
        if artifact.attr is None:
            return None
        return getattr(self.data_manager, artifact.attr, None)
