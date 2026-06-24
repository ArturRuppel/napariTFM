from pathlib import Path
from typing import Any

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from napariTFM.widgets._stage_section import StageSection
from napariTFM.widgets._ui_style import danger_text_style, section_grid, add_section_pair_row, add_section_full_row


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

        grid = section_grid()
        grid.setContentsMargins(8, 8, 8, 8)
        self.setLayout(grid)

        controls = []
        for name, label, min_val, max_val, step, decimals in _GENERAL_SPECS:
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
            controls.append((label, control))

        add_section_pair_row(
            grid, 0,
            controls[0][0], controls[0][1],
            controls[1][0], controls[1][1],
        )

        self.output_dir_label = QLabel("No output directory")
        self.output_dir_label.setObjectName("project_output_dir_label")
        self.choose_output_dir_btn = QPushButton("Output Directory")
        self.choose_output_dir_btn.setObjectName("project_choose_output_dir_button")
        self.choose_output_dir_btn.clicked.connect(self._choose_output_dir)

        output_container = QWidget()
        output_row = QHBoxLayout(output_container)
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.addWidget(self.choose_output_dir_btn)
        output_row.addWidget(self.output_dir_label, stretch=1)
        add_section_full_row(grid, 1, output_container)

        # One save (P0b): the config save carries parameters *and* the experiment
        # table, so the old params-only save is gone.
        self.save_config_btn = QPushButton("Save Config")
        self.load_config_btn = QPushButton("Load Config")
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setStyleSheet(danger_text_style())

        button_container1 = QWidget()
        button_row1 = QHBoxLayout(button_container1)
        button_row1.setContentsMargins(0, 0, 0, 0)
        button_row1.addWidget(self.save_config_btn)
        button_row1.addWidget(self.load_config_btn)
        add_section_full_row(grid, 2, button_container1)

        button_container2 = QWidget()
        button_row2 = QHBoxLayout(button_container2)
        button_row2.setContentsMargins(0, 0, 0, 0)
        button_row2.addWidget(self.reset_params_btn)
        button_row2.addWidget(self.clear_data_btn)
        add_section_full_row(grid, 3, button_container2)

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
    def save_config_btn(self):
        return self.body.save_config_btn

    @property
    def load_config_btn(self):
        return self.body.load_config_btn

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
