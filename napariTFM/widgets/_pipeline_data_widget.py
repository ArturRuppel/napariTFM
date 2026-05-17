from pathlib import Path
from typing import Callable, Optional

import numpy as np
from napari.layers import Image
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napariTFM.utilities.data_manager import DataManager


IMAGE_ARTIFACTS = {
    "bead_stack": ("Raw bead stack", "set_bead_stack"),
    "reference": ("Raw reference image", "set_reference"),
    "cell_stack": ("Raw cell stack", "set_cell_stack"),
    "preprocessed_bead_stack": ("Preprocessed bead stack", "set_preprocessed_bead_stack"),
    "preprocessed_reference": ("Preprocessed reference image", "set_preprocessed_reference"),
    "preprocessed_cell_stack": ("Preprocessed cell stack", "set_preprocessed_cell_stack"),
}

RESULT_ARTIFACTS = {
    "displacement_results": "Displacement result",
    "force_results": "Force/traction result",
    "stress_results": "Stress result",
}


class PipelineArtifactRow(QWidget):
    """Single compact row for loading and displaying one pipeline artifact."""

    def __init__(self, key: str, label: str, load_label: str, load_callback: Callable[[str], None]):
        super().__init__()
        self.key = key
        self._load_callback = load_callback

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.status_icon = QLabel("○")
        self.status_icon.setFixedWidth(14)
        self.name_label = QLabel(label)
        self.name_label.setMinimumWidth(135)
        self.info_label = QLabel("Not loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.load_btn = QPushButton(load_label)
        self.load_btn.setFixedWidth(82)
        self.load_btn.clicked.connect(lambda: self._load_callback(self.key))

        layout.addWidget(self.status_icon)
        layout.addWidget(self.name_label)
        layout.addWidget(self.info_label, stretch=1)
        layout.addWidget(self.load_btn)
        self.setLayout(layout)

    def refresh(self, data_manager: DataManager) -> None:
        state = data_manager.get_artifact(self.key)
        if not state.available:
            self.status_icon.setText("○")
            self.info_label.setText("Not loaded")
            return

        self.status_icon.setText("●" if not state.dirty else "◐")
        parts = []
        shape = self._shape_text(state.value)
        if shape:
            parts.append(shape)
        if state.path:
            parts.append(str(state.path))
        elif state.source:
            parts.append(state.source)
        if state.dirty:
            parts.append("unsaved")
        if state.error:
            parts.append(f"error: {state.error}")
        self.info_label.setText(" — ".join(parts) if parts else "Loaded")

    @staticmethod
    def _shape_text(value) -> str:
        if isinstance(value, np.ndarray):
            return str(value.shape)
        for attr in ("displacement_field", "force_field", "stress_tensor"):
            array = getattr(value, attr, None)
            if array is not None:
                return str(array.shape)
        return ""


class PipelineDataWidget(QFrame):
    """Unified source-of-truth widget for pipeline input/output artifacts."""

    data_changed = Signal()

    def __init__(self, viewer, data_manager: DataManager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.rows = {}
        self.setObjectName("pipeline_data_widget")
        self._setup_ui()
        self.data_manager.add_change_callback(self.refresh)
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        output_group = QGroupBox("Output directory")
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Choose an output directory")
        self.choose_output_btn = QPushButton("Choose…")
        self.auto_output_btn = QPushButton("Auto")
        self.choose_output_btn.clicked.connect(self.choose_output_dir)
        self.auto_output_btn.clicked.connect(self.auto_output_dir)
        self.output_dir_edit.editingFinished.connect(self._apply_output_dir_text)
        output_layout.addWidget(self.output_dir_edit, stretch=1)
        output_layout.addWidget(self.choose_output_btn)
        output_layout.addWidget(self.auto_output_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        active_group = QGroupBox("Images from active napari layer")
        active_layout = QVBoxLayout()
        for key, (label, _) in IMAGE_ARTIFACTS.items():
            row = PipelineArtifactRow(key, label, "Use active", self.load_active_layer_artifact)
            self.rows[key] = row
            active_layout.addWidget(row)
        active_group.setLayout(active_layout)
        layout.addWidget(active_group)

        result_group = QGroupBox("Analysis results from file")
        result_layout = QVBoxLayout()
        for key, label in RESULT_ARTIFACTS.items():
            row = PipelineArtifactRow(key, label, "Load .npy", self.load_result_artifact)
            self.rows[key] = row
            result_layout.addWidget(row)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        self.setLayout(layout)

    def closeEvent(self, event):
        self.data_manager.remove_change_callback(self.refresh)
        super().closeEvent(event)

    def refresh(self) -> None:
        output_dir = self.data_manager.output_dir
        self.output_dir_edit.blockSignals(True)
        self.output_dir_edit.setText(str(output_dir) if output_dir else "")
        self.output_dir_edit.blockSignals(False)
        for row in self.rows.values():
            row.refresh(self.data_manager)
        self.data_changed.emit()

    def choose_output_dir(self) -> None:
        start_dir = str(self.data_manager.output_dir or Path.home())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Pipeline Output Directory",
            start_dir,
            QFileDialog.ShowDirsOnly,
        )
        if path:
            self.data_manager.set_output_dir(path)

    def auto_output_dir(self) -> None:
        input_dir = self._infer_input_dir()
        if input_dir is None:
            QMessageBox.warning(
                self,
                "Cannot Infer Output Directory",
                "Load at least one artifact from a file or layer with a source path first."
            )
            return
        self.data_manager.set_output_dir(input_dir / "napariTFM_outputs")

    def _apply_output_dir_text(self) -> None:
        text = self.output_dir_edit.text().strip()
        self.data_manager.set_output_dir(text or None)

    def _infer_input_dir(self) -> Optional[Path]:
        for key in IMAGE_ARTIFACTS:
            state = self.data_manager.get_artifact(key)
            if state.path:
                return state.path.parent
        active_layer = self._active_image_layer()
        source = getattr(active_layer, "source", None)
        source_path = getattr(source, "path", None)
        if source_path:
            return Path(source_path).parent
        return None

    def load_active_layer_artifact(self, key: str) -> None:
        layer = self._active_image_layer()
        if layer is None:
            QMessageBox.warning(self, "No Image Layer", "Select an image layer first.")
            return
        data = np.asarray(layer.data)
        _, setter_name = IMAGE_ARTIFACTS[key]
        setter = getattr(self.data_manager, setter_name)
        source_path = self._layer_path(layer)
        try:
            kwargs = {"path": source_path, "source": layer.name or "active layer"}
            if key.startswith("preprocessed_"):
                kwargs["dirty"] = False
            setter(data, **kwargs)
        except Exception as exc:
            QMessageBox.critical(self, "Could Not Load Layer", str(exc))
            return
        self.refresh()

    def load_result_artifact(self, key: str) -> None:
        start_dir = str(self.data_manager.output_dir or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pipeline Result",
            start_dir,
            "NumPy files (*.npy);;All files (*)",
        )
        if not path:
            return
        try:
            self.data_manager.load_result_artifact(key, path)
        except Exception as exc:
            QMessageBox.critical(self, "Could Not Load Result", str(exc))
            return
        self.refresh()

    def _active_image_layer(self):
        active = self.viewer.layers.selection.active
        if isinstance(active, Image):
            return active
        if active is not None and hasattr(active, "data") and isinstance(active.data, np.ndarray):
            return active
        return None

    @staticmethod
    def _layer_path(layer) -> Optional[Path]:
        source = getattr(layer, "source", None)
        source_path = getattr(source, "path", None)
        return Path(source_path) if source_path else None
