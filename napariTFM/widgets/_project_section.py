from typing import Any

from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napariTFM.widgets._stage_section import StageSection


_GENERAL_SPECS = [
    ("pixel_size", "Pixel Size (um)", 0.001, 100.0, 0.1, 3),
    ("frame_interval", "Frame Length (min)", 0.001, 1000.0, 0.1, 3),
]


class _GeneralBody(QWidget):
    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
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

        self.save_params_btn = QPushButton("Save Parameters")
        self.load_params_btn = QPushButton("Load Parameters")
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setStyleSheet("color: red;")

        button_row1 = QHBoxLayout()
        button_row1.addWidget(self.save_params_btn)
        button_row1.addWidget(self.load_params_btn)
        layout.addLayout(button_row1)

        button_row2 = QHBoxLayout()
        button_row2.addWidget(self.reset_params_btn)
        button_row2.addWidget(self.clear_data_btn)
        layout.addLayout(button_row2)

        parameter_manager.parameter_changed.connect(self._sync_parameter)

    def _sync_parameter(self, name: str, value: Any):
        control = self.parameter_controls.get(name)
        if control is None:
            return
        control.blockSignals(True)
        try:
            control.setValue(value)
        finally:
            control.blockSignals(False)


class ProjectSection(StageSection):
    """Top-of-shell Project section: general parameters + save/load/reset/clear."""

    def __init__(self, parameter_manager):
        body = _GeneralBody(parameter_manager)
        super().__init__("Project", body, expanded=True, accent=None)
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
