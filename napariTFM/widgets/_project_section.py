from pathlib import Path
from typing import Any

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._stage_section import StageSection
from napariTFM.widgets._ui_style import danger_text_style


_GENERAL_SPECS = [
    ("pixel_size", "Pixel Size (um)", 0.001, 100.0, 0.1, 3),
    ("frame_interval", "Frame Length (min)", 0.001, 1000.0, 0.1, 3),
]


class _GeneralBody(QWidget):
    output_dir_changed = Signal()

    def __init__(self, parameter_manager, data_manager=None):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.data_manager = data_manager
        self.parameter_controls: dict[str, QDoubleSpinBox] = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.setLayout(layout)

        for name, label, min_val, max_val, step, decimals in _GENERAL_SPECS:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            control = QDoubleSpinBox()
            control.setRange(min_val, max_val)
            control.setSingleStep(step)
            control.setDecimals(decimals)
            control.setObjectName(f"workflow_parameter_{name}")
            control.setValue(parameter_manager.get_ui_parameter(name))
            control.valueChanged.connect(
                lambda value, n=name: parameter_manager.set_ui_parameter(n, value)
            )
            self.parameter_controls[name] = control
            row.addWidget(control)
            layout.addLayout(row)

        self.output_dir_label = QLabel("No output directory")
        self.output_dir_label.setObjectName("project_output_dir_label")
        self.choose_output_dir_btn = QPushButton("Output Directory")
        self.choose_output_dir_btn.setObjectName("project_choose_output_dir_button")
        self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)

        output_row = QHBoxLayout()
        output_row.addWidget(self.choose_output_dir_btn)
        output_row.addWidget(self.output_dir_label, stretch=1)
        layout.addLayout(output_row)

        self.save_params_btn = QPushButton("Save Parameters")
        self.load_params_btn = QPushButton("Load Parameters")
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setStyleSheet(danger_text_style())

        button_row1 = QHBoxLayout()
        button_row1.addWidget(self.save_params_btn)
        button_row1.addWidget(self.load_params_btn)
        layout.addLayout(button_row1)

        button_row2 = QHBoxLayout()
        button_row2.addWidget(self.reset_params_btn)
        button_row2.addWidget(self.clear_data_btn)
        layout.addLayout(button_row2)

        parameter_manager.parameter_changed.connect(self._sync_parameter)
        if self.data_manager is not None:
            self.data_manager.add_change_callback(self._sync_output_dir)
        self._sync_output_dir()

    def _sync_parameter(self, name: str, value: Any):
        control = self.parameter_controls.get(name)
        if control is None:
            return
        control.blockSignals(True)
        try:
            control.setValue(value)
        finally:
            control.blockSignals(False)

    def _choose_output_dir(self):
        if self.data_manager is None:
            return
        current = self.data_manager.output_dir or Path.home()
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Pipeline Output Directory",
            str(current),
        )
        if path:
            self.data_manager.set_output_dir(path)
            self.output_dir_changed.emit()

    def _sync_output_dir(self):
        path = getattr(self.data_manager, "output_dir", None)
        if path is None:
            self.output_dir_label.setText("No output directory")
            self.output_dir_label.setToolTip("")
            return
        text = str(path)
        self.output_dir_label.setText(text)
        self.output_dir_label.setToolTip(text)


class ProjectSection(StageSection):
    """Top-of-shell Project section: general parameters + save/load/reset/clear."""

    def __init__(self, parameter_manager, data_manager=None):
        body = _GeneralBody(parameter_manager, data_manager)
        super().__init__("Project", body, accent=None)
        self.body = body
        # Project is not a workflow stage; hide the run/preview action buttons.
        self.run_cancel_btn.setVisible(False)
        self.preview_button.setVisible(False)

    @property
    def parameter_controls(self):
        return self.body.parameter_controls

    @property
    def save_params_btn(self):
        return self.body.save_params_btn

    @property
    def load_params_btn(self):
        return self.body.load_params_btn

    @property
    def reset_params_btn(self):
        return self.body.reset_params_btn

    @property
    def clear_data_btn(self):
        return self.body.clear_data_btn

    @property
    def output_dir_label(self):
        return self.body.output_dir_label

    @property
    def choose_output_dir_btn(self):
        return self.body.choose_output_dir_btn
