# TODO add reset config
# TODO Default value for young's modulus should be 10. Conversion from kPa to Pa correct?
# TODO young's modulus resets to default value as soon as I change it
# TODO add tooltips for parameters
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from qtpy.QtCore import Qt, Signal, QSettings
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget, QGridLayout, QButtonGroup, QRadioButton,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame, QScrollArea,
    QProgressBar, QMessageBox, QListWidget, QCheckBox, QLineEdit,
    QFileDialog, QComboBox
)

from napariTFM.backend.batch_analysis import BatchAnalysis
from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.parameter_manager import ParameterManager, ParameterCategory


class BatchAnalysisWidget(BaseAnalysisWidget):
    """Widget for running batch analysis on multiple folders."""

    batch_completed = Signal(dict)  # Emits results when batch processing completes

    MESH_ALGORITHMS = {
        "Frontal-Del.": "Frontal-Del.",
        "Delaunay": "Delaunay",
        "MeshAdapt": "MeshAdapt",
        "BAMG": "BAMG",
        "FD Quads": "FD Quads",
        "Para. Pack": "Para. Pack"
    }

    # region === Initialization ===
    def __init__(self, viewer, data_manager, parameter_manager: ParameterManager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        self.parameter_manager = parameter_manager
        self.parameter_spins = {}
        self.parameter_combos = {}
        self.parameter_checks = {}
        self.visualization_checkboxes = {}
        self.folder_list = []

        # Block signals during setup
        self.blockSignals(True)
        try:
            self._setup_ui()
            self._connect_signals()

            # Ensure parameters are connected before syncing
            self._connect_parameters()

            # Force an initial sync with parameter manager
            self._sync_widget_with_parameters()

            self._update_ui_state()

            # Connect to parameter manager signals after everything is set up
            self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
            self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)
        finally:
            self.blockSignals(False)

    def _setup_ui(self):
        """Set up the user interface."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        main_layout = QVBoxLayout()

        # Add new file paths and metadata groups first
        main_layout.addWidget(self._create_file_paths_group())
        main_layout.addWidget(self._create_metadata_group())

        # Add existing parameter groups
        main_layout.addWidget(self._create_general_params_group())
        main_layout.addWidget(self._create_preprocessing_params_group())
        main_layout.addWidget(self._create_displacement_params_group())
        main_layout.addWidget(self._create_force_params_group())
        main_layout.addWidget(self._create_stress_params_group())

        # Add modified analysis steps and visualization groups
        main_layout.addWidget(self._create_analysis_steps_group())
        main_layout.addWidget(self._create_visualization_group())
        main_layout.addWidget(self._create_folder_management_group())
        main_layout.addWidget(self._create_status_frame())

        container.setLayout(main_layout)
        scroll.setWidget(container)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect widget signals."""
        # Keep existing signal connections
        self.add_folder_btn.clicked.connect(self._add_folder)
        self.clear_folders_btn.clicked.connect(self._clear_folders)
        self.run_analysis_btn.clicked.connect(self._run_batch_analysis)
        self.save_config_btn.clicked.connect(self._save_config_dialog)
        self.load_config_btn.clicked.connect(self._load_config_dialog)

    def _connect_parameters(self):
        """Connect widget controls to parameter manager."""
        self._block_parameter_widgets(True)
        try:
            # Connect all spinboxes
            for name, spin in self.parameter_spins.items():
                if isinstance(spin, tuple):
                    continue  # Handle special cases separately if needed

                # Create closure for the callback
                def make_callback(name=name):
                    def callback(value):
                        if name == 'young_modulus':
                            value = value * 1000  # Convert kPa to Pa
                        elif name == 'gel_height' and value == 0:
                            value = None
                        self.parameter_manager.set_parameter(name, value)

                    return callback

                spin.valueChanged.connect(make_callback())

                # Set initial value
                try:
                    value = self.parameter_manager.get_parameter(name)
                    if name == 'young_modulus':
                        value = value / 1000  # Convert Pa to kPa for display
                    elif name == 'gel_height':
                        value = 0 if value is None else value
                    self._safe_set_value(spin, value)
                except KeyError:
                    print(f"Warning: Parameter {name} not found")

            # Connect all comboboxes
            for name, combo in self.parameter_combos.items():
                combo.currentTextChanged.connect(
                    lambda text, name=name: self.parameter_manager.set_parameter(name, text)
                )
                try:
                    value = self.parameter_manager.get_parameter(name)
                    self._safe_set_combo_text(combo, value)
                except KeyError:
                    print(f"Warning: Parameter {name} not found")

            # Connect all checkboxes
            for name, checkbox in self.parameter_checks.items():
                checkbox.stateChanged.connect(
                    lambda state, name=name: self.parameter_manager.set_parameter(
                        name, state == Qt.Checked
                    )
                )
                try:
                    value = self.parameter_manager.get_parameter(name)
                    self._safe_set_checked(checkbox, value)
                except KeyError:
                    print(f"Warning: Parameter {name} not found")

        finally:
            self._block_parameter_widgets(False)

    # endregion === Initialization ===

    # region === UI Creation Groups ===
    def _create_metadata_group(self) -> QGroupBox:
        """Create group for metadata fields."""
        group = QGroupBox("Metadata")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        metadata_fields = [
            ("experiment_date", "Experiment Date:"),
            ("cell_type", "Cell Type:"),
            ("substrate", "Substrate:", "PA gel"),  # Default value from config
            ("notes", "Notes:")
        ]

        self.metadata_inputs = {}
        for key, label, *default in metadata_fields:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            line_edit = QLineEdit()
            if default:  # Set default value if provided
                line_edit.setText(default[0])
            self.metadata_inputs[key] = line_edit
            row.addWidget(line_edit)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_general_params_group(self) -> QGroupBox:
        """Create general parameters group."""
        group = QGroupBox("General Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Pixel size
        pixel_row = QHBoxLayout()
        pixel_row.addWidget(QLabel("Pixel Size (µm):"))
        pixel_spin = QDoubleSpinBox()
        pixel_spin.setRange(0.01, 10.0)
        pixel_spin.setSingleStep(0.01)
        pixel_spin.setDecimals(2)
        pixel_spin.setToolTip("Physical size of each pixel in micrometers")
        self.parameter_spins['pixel_size'] = pixel_spin
        pixel_row.addWidget(pixel_spin)
        layout.addLayout(pixel_row)

        # Frame interval
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame Length (min):"))
        frame_spin = QDoubleSpinBox()
        frame_spin.setRange(0.001, 1000.0)
        frame_spin.setSingleStep(0.1)
        frame_spin.setDecimals(1)
        frame_spin.setToolTip("Time between consecutive frames in minutes")
        self.parameter_spins['frame_interval'] = frame_spin
        frame_row.addWidget(frame_spin)
        layout.addLayout(frame_row)

        group.setLayout(layout)
        return group

    def _create_file_paths_group(self) -> QGroupBox:
        """Create group for input/output file paths."""
        group = QGroupBox("File Paths")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Input files section
        input_label = QLabel("Input Files:")
        input_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(input_label)

        input_files = [
            ("beads", "Beads File:", "beads.tif"),
            ("reference", "Reference File:", "reference.tif"),
            ("cells", "Cells File (optional):", "cells.tif")
        ]

        self.file_inputs = {}
        for key, label, default in input_files:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            line_edit = QLineEdit(default)
            self.file_inputs[key] = line_edit
            row.addWidget(line_edit)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_preprocessing_params_group(self) -> QGroupBox:
        """Create preprocessing parameters group."""
        group = QGroupBox("Preprocessing Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Bead/Reference parameters
        bead_params = [
            ("min_intensity_percentile", "Min Intensity (%)", 0, 100, 0.1, 0),
            ("max_intensity_percentile", "Max Intensity (%)", 0, 100, 0.1, 100),
            ("gaussian_sigma", "Gaussian Sigma", 0.0, 10.0, 0.1, 0.0)
        ]

        for name, label, min_val, max_val, step, default in bead_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setDecimals(1)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Cell parameters
        cell_params = [
            ("cell_min_intensity_percentile", "Cell Min Intensity (%)", 0, 100, 0.1, 0),
            ("cell_max_intensity_percentile", "Cell Max Intensity (%)", 0, 100, 0.1, 100),
            ("cell_gaussian_sigma", "Cell Gaussian Sigma", 0.0, 10.0, 0.1, 0.0)
        ]

        for name, label, min_val, max_val, step, default in cell_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setDecimals(1)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Registration mode
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Registration Mode:"))
        reg_combo = QComboBox()
        reg_combo.addItems(['Translation', 'Rigid', 'No registration'])
        reg_combo.setToolTip("Choose registration method")
        self.parameter_combos['registration_mode'] = reg_combo
        reg_row.addWidget(reg_combo)
        layout.addLayout(reg_row)

        group.setLayout(layout)
        return group

    def _create_displacement_params_group(self) -> QGroupBox:
        """Create displacement analysis parameters group."""
        group = QGroupBox("Displacement Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Optical flow parameters
        flow_params = [
            ("tau", "Tau:", 0.01, 1.0, 0.01, 0.25),
            ("lambda_", "Lambda:", 0.01, 1.0, 0.01, 0.4),
            ("theta", "Theta:", 0.1, 1.0, 0.1, 0.3),
            ("nscales", "Pyramid Scales:", 1, 10, 1, 3),
            ("warps", "Warps:", 1, 10, 1, 3),
            ("epsilon", "Epsilon:", 0.001, 0.1, 0.001, 0.01),
            ("inner_iterations", "Inner Iterations:", 1, 50, 1, 15),
            ("outer_iterations", "Outer Iterations:", 1, 20, 1, 5),
            ("scale_step", "Scale Step:", 0.1, 0.99, 0.01, 0.5),
            ("median_filtering", "Median Filter:", 1, 9, 2, 5),
            ("downscale_factor", "Downscale Factor:", 1, 10, 1, 1)
        ]

        for name, label, min_val, max_val, step, default in flow_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            if isinstance(step, int):
                spin = QSpinBox()
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(2)
            if name == "epsilon":
                spin.setDecimals(3)
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Visualization parameters
        vis_params = [
            ("disp_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("disp_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("d_max", "Max Displacement (µm):", 0.1, 200.0, 0.1, 5.0)
        ]

        for name, label, min_val, max_val, step, default in vis_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_force_params_group(self) -> QGroupBox:
        """Create force analysis parameters group."""
        group = QGroupBox("Force Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Material parameters setup
        material_params = [
            ("young_modulus", "Young's Modulus (kPa):", 0.1, 1000, 0.1, 10),
            ("poisson_ratio_substrate", "Poisson's Ratio:", 0, 0.5, 0.01, 0.49),
            ("gel_height", "Gel Height (µm):", 0, 1000, 10, 0)
        ]

        for name, label, min_val, max_val, step, default in material_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if name == "gel_height":
                spin.setSpecialValueText("∞")
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Lanczos exponent (integer spinbox)
        lanczos_row = QHBoxLayout()
        lanczos_row.addWidget(QLabel("Lanczos Exponent:"))
        lanczos_spin = QSpinBox()  # Changed to QSpinBox
        lanczos_spin.setRange(0, 5)
        lanczos_spin.setValue(1)
        self.parameter_spins['lanczos_exp'] = lanczos_spin
        lanczos_row.addWidget(lanczos_spin)
        layout.addLayout(lanczos_row)

        # Regularization parameter (as log10)
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Regularization (10^x):"))
        reg_spin = QDoubleSpinBox()
        reg_spin.setRange(-21, 0)
        reg_spin.setSingleStep(0.5)
        reg_spin.setValue(-4)
        reg_spin.setDecimals(1)
        self.parameter_spins['regularization'] = reg_spin
        reg_row.addWidget(reg_spin)
        layout.addLayout(reg_row)

        # Auto-GCV checkbox
        auto_gcv = QCheckBox("Auto-GCV per frame")
        self.parameter_checks['auto_gcv'] = auto_gcv
        layout.addWidget(auto_gcv)

        # Connect auto-GCV checkbox to enable/disable regularization spinbox
        def toggle_reg_spin(state):
            reg_spin.setEnabled(not state)

        auto_gcv.stateChanged.connect(toggle_reg_spin)

        # Add visualization parameters
        vis_params = [
            ("force_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("force_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("f_max", "Max Force (Pa):", 0.1, 10000.0, 1, 1000.0)
        ]

        for name, label, min_val, max_val, step, default in vis_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_stress_params_group(self) -> QGroupBox:
        """Create stress parameters group with proper parameter manager integration."""
        group = QGroupBox("Stress Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Mask parameters
        mask_params = [
            ("threshold", "Threshold Percentile:", 0, 100, 0.1, 0,
             "Percentile threshold for cell mask generation"),
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 10,
             "Number of pixels to dilate the mask"),
            ("smoothing_sigma", "Boundary Smoothing:", 0.0, 40.0, 0.1, 10,
             "Sigma for Gaussian smoothing of mask boundary")
        ]

        for name, label, min_val, max_val, step, default, tooltip in mask_params:
            row = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setToolTip(tooltip)
            row.addWidget(label_widget)

            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if isinstance(spin, QDoubleSpinBox):
                spin.setDecimals(1)

            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Mesh parameters
        mesh_params = [
            ("density_factor", "Density Factor:", 0.001, 0.1, 0.001, 0.025,
             "Controls mesh density. Lower values create finer meshes."),
            ("mesh_algorithm", "Mesh Algorithm:", None, None, None, "Frontal-Del.",
             "Algorithm used for mesh generation"),
        ]

        for name, label, min_val, max_val, step, default, tooltip in mesh_params:
            row = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setToolTip(tooltip)
            row.addWidget(label_widget)

            if name == "mesh_algorithm":
                combo = QComboBox()
                combo.addItems(self.MESH_ALGORITHMS.keys())
                combo.setCurrentText(default)
                self.parameter_combos[name] = combo
                row.addWidget(combo)
            else:
                spin = QDoubleSpinBox()
                spin.setRange(min_val, max_val)
                spin.setSingleStep(step)
                spin.setValue(default)
                spin.setDecimals(3)
                self.parameter_spins[name] = spin
                row.addWidget(spin)

            layout.addLayout(row)

        # Add optimization checkbox
        self.parameter_checks['use_optimization'] = QCheckBox("Mesh Optimization")
        self.parameter_checks['use_optimization'].setChecked(True)
        self.parameter_checks['use_optimization'].setToolTip(
            "Enable mesh quality optimization after generation"
        )
        layout.addWidget(self.parameter_checks['use_optimization'])

        # Add max stress visualization parameter
        poisson_row = QHBoxLayout()
        poisson_row.addWidget(QLabel("Poisson's Ratio:"))
        poisson_spin = QDoubleSpinBox()
        poisson_spin.setRange(0, 1.0)
        poisson_spin.setSingleStep(0.01)
        poisson_spin.setValue(1.0)
        poisson_spin.setDecimals(2)
        self.parameter_spins['poisson_ratio_cells'] = poisson_spin
        poisson_row.addWidget(poisson_spin)
        layout.addLayout(poisson_row)

        stress_row = QHBoxLayout()
        stress_row.addWidget(QLabel("Max Stress (mN/m):"))
        max_stress_spin = QDoubleSpinBox()
        max_stress_spin.setRange(0.01, 1000.0)
        max_stress_spin.setSingleStep(0.1)
        max_stress_spin.setValue(1.0)
        max_stress_spin.setDecimals(2)
        self.parameter_spins['max_stress'] = max_stress_spin
        stress_row.addWidget(max_stress_spin)
        layout.addLayout(stress_row)

        group.setLayout(layout)
        return group

    def _create_analysis_steps_group(self) -> QGroupBox:
        """Create analysis steps group with checkboxes."""
        group = QGroupBox("Analysis Steps")
        layout = QVBoxLayout()

        steps = [
            ("preprocessing", "Preprocessing"),
            ("displacement", "Displacement"),
            ("force", "Force"),
            ("create_masks", "Create Masks"),
            ("stress", "Stress")
        ]

        self.analysis_checkboxes = {}
        for key, label in steps:
            checkbox = QCheckBox(label)
            checkbox.setChecked(False)  # Default to False as per config
            self.analysis_checkboxes[key] = checkbox
            layout.addWidget(checkbox)

        group.setLayout(layout)
        return group

    def _create_folder_management_group(self) -> QGroupBox:
        """Create folder management group."""
        group = QGroupBox("Folder Management")
        layout = QVBoxLayout()

        # Folder list first
        self.folder_list_widget = QListWidget()
        layout.addWidget(self.folder_list_widget)

        # Create grid layout for buttons
        button_layout = QGridLayout()

        # Add folder management buttons
        self.add_folder_btn = QPushButton("Add Folder")
        self.clear_folders_btn = QPushButton("Clear Folders")
        self.save_config_btn = QPushButton("Save Config")
        self.load_config_btn = QPushButton("Load Config")
        button_layout.addWidget(self.add_folder_btn, 0, 0)
        button_layout.addWidget(self.clear_folders_btn, 0, 1)
        button_layout.addWidget(self.save_config_btn, 1, 0)
        button_layout.addWidget(self.load_config_btn, 1, 1)

        # Add console selection radio buttons
        console_group = QHBoxLayout()
        self.console_group = QButtonGroup()

        self.napari_console_radio = QRadioButton("Run in Napari Console")
        self.new_console_radio = QRadioButton("Run in New Console")
        self.napari_console_radio.setChecked(True)

        self.console_group.addButton(self.napari_console_radio)
        self.console_group.addButton(self.new_console_radio)

        console_group.addWidget(self.napari_console_radio)
        console_group.addWidget(self.new_console_radio)

        layout.addLayout(console_group)

        # Add run button
        self.run_analysis_btn = QPushButton("Run Analysis")
        layout.addWidget(self.run_analysis_btn)

        layout.addLayout(button_layout)
        group.setLayout(layout)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        """Create visualization options group."""
        group = QGroupBox("Visualizations")
        layout = QVBoxLayout()

        # Update visualization options to match parameter manager
        viz_options = [
            ("bead_overlay", "Bead Overlay", "save_bead_overlay"),
            ("displacement_map", "Displacement Map", "save_displacement_map"),
            ("force_map", "Force Map", "save_force_map"),
            ("force_cell_overlay", "Force Cell Overlay", "save_force_cell_overlay"),
            ("sigma_xx", "Sigma XX", "save_sigma_xx"),
            ("sigma_yy", "Sigma YY", "save_sigma_yy"),
            ("normal_stress", "Normal Stress", "save_normal_stress"),
            ("mesh", "Mesh", "save_mesh")
        ]

        self.visualization_checkboxes = {}
        for ui_key, label, param_name in viz_options:
            checkbox = QCheckBox(label)
            try:
                checkbox.setChecked(self.parameter_manager.get_parameter(param_name))
            except KeyError:
                print(f"Warning: Parameter {param_name} not found in parameter manager")
                checkbox.setChecked(True)  # Default to True if parameter not found

            # Store checkbox with parameter name (including 'save_' prefix)
            self.visualization_checkboxes[param_name] = checkbox

            # Connect checkbox to parameter manager
            def make_callback(param_name=param_name):
                def callback(state):
                    self.parameter_manager.set_parameter(param_name, state == Qt.Checked)

                return callback

            checkbox.stateChanged.connect(make_callback())
            layout.addWidget(checkbox)

        group.setLayout(layout)
        return group

    def _create_status_frame(self) -> QFrame:
        """Create status frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    # endregion === UI Creation Groups ===

    # region === Configuration Interface ===

    def generate_config(self) -> dict:
        """
        Generate configuration dictionary based on UI state.
        Items are ordered to match widget layout.
        """
        config = {
            # Input Files section (matches UI order)
            "input_files": {
                key: input.text()
                for key, input in self.file_inputs.items()
            },

            # Metadata section (matches UI order)
            "metadata": {
                key: input.text()
                for key, input in self.metadata_inputs.items()
            },

            # Parameters section - ordered by UI groups
            "parameters": {
                # General parameters
                "pixel_size": self.parameter_spins['pixel_size'].value(),
                "frame_interval": self.parameter_spins['frame_interval'].value(),

                # Preprocessing parameters
                "min_intensity_percentile": self.parameter_spins['min_intensity_percentile'].value(),
                "max_intensity_percentile": self.parameter_spins['max_intensity_percentile'].value(),
                "gaussian_sigma": self.parameter_spins['gaussian_sigma'].value(),
                "cell_min_intensity_percentile": self.parameter_spins['cell_min_intensity_percentile'].value(),
                "cell_max_intensity_percentile": self.parameter_spins['cell_max_intensity_percentile'].value(),
                "cell_gaussian_sigma": self.parameter_spins['cell_gaussian_sigma'].value(),
                "registration_mode": self.parameter_combos['registration_mode'].currentText().lower(),

                # Displacement parameters
                "tau": self.parameter_spins['tau'].value(),
                "lambda_": self.parameter_spins['lambda_'].value(),
                "theta": self.parameter_spins['theta'].value(),
                "nscales": self.parameter_spins['nscales'].value(),
                "warps": self.parameter_spins['warps'].value(),
                "epsilon": self.parameter_spins['epsilon'].value(),
                "inner_iterations": self.parameter_spins['inner_iterations'].value(),
                "outer_iterations": self.parameter_spins['outer_iterations'].value(),
                "scale_step": self.parameter_spins['scale_step'].value(),
                "median_filtering": self.parameter_spins['median_filtering'].value(),
                "downscale_factor": self.parameter_spins['downscale_factor'].value(),
                "disp_vector_stride": self.parameter_spins['disp_vector_stride'].value(),
                "disp_arrow_scale": self.parameter_spins['disp_arrow_scale'].value(),
                "d_max": self.parameter_spins['d_max'].value(),

                # Force parameters
                "young_modulus": self.parameter_spins['young_modulus'].value() * 1000,  # Convert to Pa
                "poisson_ratio_substrate": self.parameter_spins['poisson_ratio_substrate'].value(),
                "gel_height": None if self.parameter_spins['gel_height'].value() == 0 else self.parameter_spins['gel_height'].value(),
                "lanczos_exp": self.parameter_spins['lanczos_exp'].value(),
                "regularization": self.parameter_spins['regularization'].value(),
                "auto_gcv": self.parameter_checks['auto_gcv'].isChecked(),
                "force_vector_stride": self.parameter_spins['force_vector_stride'].value(),
                "force_arrow_scale": self.parameter_spins['force_arrow_scale'].value(),
                "f_max": self.parameter_spins['f_max'].value(),

                # Stress parameters
                "threshold": self.parameter_spins['threshold'].value(),
                "dilation": self.parameter_spins['dilation'].value(),
                "smoothing_sigma": self.parameter_spins['smoothing_sigma'].value(),
                "density_factor": self.parameter_spins['density_factor'].value(),
                "mesh_algorithm": self.parameter_combos['mesh_algorithm'].currentText(),
                "use_optimization": self.parameter_checks['use_optimization'].isChecked(),
                "poisson_ratio_cells": self.parameter_spins['poisson_ratio_cells'].value(),
                "max_stress": self.parameter_spins['max_stress'].value()
            },

            # Analysis steps section
            "analysis_steps": {
                key: checkbox.isChecked()
                for key, checkbox in self.analysis_checkboxes.items()
            },

            # Visualizations section
            "visualizations": {
                key.replace('save_', ''): checkbox.isChecked()
                for key, checkbox in self.visualization_checkboxes.items()
            },

            # Root folders should be last as they're at bottom of UI
            "root_folders": self.folder_list
        }

        return config

    def _save_config_dialog(self):
        """Open a file dialog to save the configuration file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            os.path.expanduser("~"),
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )

        if filepath:
            try:
                self.save_config_to_yaml(filepath)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Configuration saved successfully to:\n{filepath}"
                )
            except IOError as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save configuration:\n{str(e)}"
                )

    def _load_config_dialog(self):
        """Open a file dialog to load a configuration file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            os.path.expanduser("~"),
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )

        if filepath:
            try:
                self.load_config_from_yaml(filepath)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Configuration loaded successfully from:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load configuration:\n{str(e)}"
                )

    def save_config_to_yaml(self, filepath: str) -> None:
        """
        Save the current configuration to a YAML file.

        Parameters:
        -----------
        filepath : str
            The path where the YAML file should be saved. If no extension is provided,
            '.yaml' will be added automatically.

        Raises:
        -------
        IOError: If there is an error writing to the file
        """
        # Add .yaml extension if not present
        if not filepath.lower().endswith(('.yml', '.yaml')):
            filepath += '.yaml'

        try:
            # Generate configuration dictionary
            config = self.generate_config()

            # Save to YAML file
            with open(filepath, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        except Exception as e:
            raise IOError(f"Failed to save configuration to {filepath}: {str(e)}")

    def load_config_from_yaml(self, filepath: str) -> None:
        """
        Load configuration from a YAML file and update the widget state.

        Parameters:
        -----------
        filepath : str
            Path to the YAML configuration file

        Raises:
        -------
        IOError: If there is an error reading the file
        ValueError: If the configuration format is invalid
        """
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)

            # Block signals during update
            self.blockSignals(True)
            try:
                # Update input files
                for key, input_widget in self.file_inputs.items():
                    if key in config.get('input_files', {}):
                        input_widget.setText(config['input_files'][key])

                # Update metadata
                for key, input_widget in self.metadata_inputs.items():
                    if key in config.get('metadata', {}):
                        input_widget.setText(config['metadata'][key])

                # Update parameters
                params = config.get('parameters', {})

                # Create a list to track parameters that fail to update
                failed_updates = []

                for name, value in params.items():
                    try:
                        # Handle special cases first
                        if name == 'young_modulus':
                            value = value / 1000  # Convert Pa to kPa for display
                        elif name == 'gel_height':
                            value = 0 if value is None else value
                        elif name == 'registration_mode':
                            # Ensure first letter is capitalized for combo box
                            value = value.capitalize()

                        # Update spinboxes
                        if name in self.parameter_spins:
                            spin = self.parameter_spins[name]
                            if value is not None:  # Only update if value exists
                                if isinstance(spin, tuple):
                                    # Handle special cases like threshold
                                    spin_widget, slider = spin
                                    self._safe_set_value(spin_widget, value)
                                    self._safe_set_value(slider, value)
                                else:
                                    self._safe_set_value(spin, value)
                                    # Force an update through the parameter manager
                                    if name == 'young_modulus':
                                        self.parameter_manager.set_parameter(name, value * 1000)
                                    else:
                                        self.parameter_manager.set_parameter(name, value)

                        # Update comboboxes
                        elif name in self.parameter_combos:
                            combo = self.parameter_combos[name]
                            if value is not None:
                                self._safe_set_combo_text(combo, str(value))
                                self.parameter_manager.set_parameter(name, str(value))

                        # Update checkboxes
                        elif name in self.parameter_checks:
                            checkbox = self.parameter_checks[name]
                            if value is not None:
                                self._safe_set_checked(checkbox, bool(value))
                                self.parameter_manager.set_parameter(name, bool(value))

                    except Exception as e:
                        failed_updates.append((name, str(e)))

                # Update analysis steps
                for key, checkbox in self.analysis_checkboxes.items():
                    if key in config.get('analysis_steps', {}):
                        checkbox.setChecked(config['analysis_steps'][key])

                # Update visualizations
                for key, checkbox in self.visualization_checkboxes.items():
                    viz_key = key.replace('save_', '')
                    if viz_key in config.get('visualizations', {}):
                        checkbox.setChecked(config['visualizations'][viz_key])

                # Clear and update folder list
                self.folder_list.clear()
                self.folder_list_widget.clear()
                for folder in config.get('root_folders', []):
                    if os.path.exists(folder):
                        self.folder_list.append(folder)
                        self.folder_list_widget.addItem(folder)

                # Report any failures
                if failed_updates:
                    error_msg = "Failed to update the following parameters:\n"
                    for param, error in failed_updates:
                        error_msg += f"- {param}: {error}\n"
                    raise ValueError(error_msg)

            finally:
                self.blockSignals(False)

            # Update UI state
            self._update_ui_state()

            # Force a final parameter sync
            self._sync_parameters_with_manager()

        except Exception as e:
            raise IOError(f"Failed to load configuration from {filepath}: {str(e)}")

    # endregion === Configuration Interface ===

    # region === Parameter Management ===
    def _sync_parameters_with_manager(self):
        """Sync current widget values with parameter manager."""
        try:
            # Block signals during sync
            self.blockSignals(True)

            # Sync spinboxes
            for name, spin in self.parameter_spins.items():
                try:
                    if isinstance(spin, tuple):
                        continue  # Skip special cases if any
                    value = spin.value()
                    if name == 'young_modulus':
                        value = value * 1000  # Convert kPa to Pa
                    elif name == 'gel_height' and value == 0:
                        value = None
                    self.parameter_manager.set_parameter(name, value)
                except Exception as e:
                    print(f"Error syncing parameter {name}: {str(e)}")

            # Sync comboboxes
            for name, combo in self.parameter_combos.items():
                try:
                    self.parameter_manager.set_parameter(name, combo.currentText().lower())
                except Exception as e:
                    print(f"Error syncing combo {name}: {str(e)}")

            # Sync checkboxes
            for name, checkbox in self.parameter_checks.items():
                try:
                    self.parameter_manager.set_parameter(name, checkbox.isChecked())
                except Exception as e:
                    print(f"Error syncing checkbox {name}: {str(e)}")

            # Sync visualization checkboxes
            for param_name, checkbox in self.visualization_checkboxes.items():
                try:
                    self.parameter_manager.set_parameter(param_name, checkbox.isChecked())
                except Exception as e:
                    print(f"Error syncing visualization {param_name}: {str(e)}")

        finally:
            self.blockSignals(False)

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager values."""
        if not hasattr(self, 'parameter_manager') or self.parameter_manager is None:
            print("Warning: No parameter manager available for syncing")
            return

        self._block_parameter_widgets(True)
        try:
            # Sync all parameters
            for name, spin in self.parameter_spins.items():
                value = self.parameter_manager.get_parameter(name)
                if isinstance(spin, tuple):
                    # Handle special cases like threshold
                    spin_widget, slider = spin
                    self._safe_set_value(spin_widget, value)
                    self._safe_set_value(slider, value)
                else:
                    self._safe_set_value(spin, value)

            # Sync combo boxes
            for name, combo in self.parameter_combos.items():
                value = self.parameter_manager.get_parameter(name)
                if value:
                    self._safe_set_combo_text(combo, str(value))

            # Sync checkboxes
            for name, checkbox in self.parameter_checks.items():
                value = self.parameter_manager.get_parameter(name)
                self._safe_set_checked(checkbox, bool(value))

            # Sync visualization checkboxes
            for param_name, checkbox in self.visualization_checkboxes.items():
                value = self.parameter_manager.get_parameter(param_name)
                self._safe_set_checked(checkbox, bool(value))

        finally:
            self._block_parameter_widgets(False)

    def _block_parameter_widgets(self, block: bool):
        """Block or unblock signals for all parameter widgets."""
        widgets = []

        # Add all spinboxes
        for spin in self.parameter_spins.values():
            if isinstance(spin, tuple):  # Handle special cases like threshold
                widgets.extend(spin)
            else:
                widgets.append(spin)

        # Add all comboboxes
        for combo in self.parameter_combos.values():
            widgets.append(combo)

        # Add all checkboxes
        for checkbox in self.parameter_checks.values():
            widgets.append(checkbox)

        # Add visualization checkboxes
        for checkbox in self.visualization_checkboxes.values():
            widgets.append(checkbox)

        for widget in widgets:
            widget.blockSignals(block)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes from parameter manager."""
        if not self.signalsBlocked():
            self._sync_widget_with_parameters()

    def _on_parameters_reset(self, category: ParameterCategory):
        """Handle parameter reset events."""
        self._sync_widget_with_parameters()

    def _safe_set_value(self, widget, value):
        """Safely set widget value with signal blocking and range checking."""
        if value is not None and widget is not None:
            widget.blockSignals(True)
            try:
                # Ensure value is within widget's range
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    value = max(widget.minimum(), min(widget.maximum(), value))
                widget.setValue(value)
            except Exception as e:
                print(f"Error setting widget value: {str(e)}")
            widget.blockSignals(False)

    def _safe_set_combo_text(self, combo, text):
        """Safely set combo box text with signal blocking and case-insensitive matching."""
        if combo is not None and text is not None:
            combo.blockSignals(True)
            try:
                # Try exact match first
                index = combo.findText(str(text), Qt.MatchFixedString)
                if index < 0:
                    # Try case-insensitive match
                    index = combo.findText(str(text), Qt.MatchFixedString | Qt.MatchCaseInsensitive)
                if index >= 0:
                    combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)

    def _safe_set_checked(self, checkbox, checked):
        """Safely set checkbox state with signal blocking."""
        if checkbox is not None:
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(checked))
            checkbox.blockSignals(False)

    # endregion === Parameter Management ===

    # region === Folder Management ===
    def _add_folder(self):
        """Add folder to analysis queue."""
        # Try to load last directory from QSettings
        settings = QSettings()
        last_dir = settings.value("BatchAnalysis/last_folder", os.path.expanduser("~"))

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            last_dir
        )

        if folder:
            # Save the selected directory
            settings.setValue("BatchAnalysis/last_folder", os.path.dirname(folder))

            try:
                # Validate folder contents
                missing_files = self._check_folder_contents(folder)
                if not missing_files:
                    if folder not in self.folder_list:  # Avoid duplicates
                        self.folder_list.append(folder)
                        self.folder_list_widget.addItem(folder)
                        self._update_ui_state()
                    else:
                        QMessageBox.warning(
                            self,
                            "Duplicate Folder",
                            f"The folder:\n{folder}\nis already in the list."
                        )
                else:
                    QMessageBox.warning(
                        self,
                        "Missing Files",
                        f"The following required files are missing:\n{', '.join(missing_files)}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error adding folder {folder}:\n{str(e)}"
                )

    def _clear_folders(self):
        """Clear folder list."""
        self.folder_list.clear()
        self.folder_list_widget.clear()
        self._update_ui_state()

    def _validate_folder(self, folder: str) -> bool:
        """Validate folder contains required files."""
        required_files = ['beads.tif', 'reference.tif']
        return all(os.path.exists(os.path.join(folder, f)) for f in required_files)

    def _check_folder_contents(self, folder: str) -> list:
        """
        Check folder for required files based on config input file names.

        Parameters:
        -----------
        folder : str
            Path to folder to check

        Returns:
        --------
        list
            List of missing required files
        """
        # Get required file names from input file fields
        required_files = [
            self.file_inputs['beads'].text(),
            self.file_inputs['reference'].text()
        ]

        # Check for optional cell file only if specified
        cell_filename = self.file_inputs['cells'].text()
        if cell_filename and cell_filename != "cells.tif":  # Don't check default value
            required_files.append(cell_filename)

        missing = []
        for file in required_files:
            if not file:  # Skip empty filenames
                continue
            if not os.path.exists(os.path.join(folder, file)):
                missing.append(file)
        return missing

    def _validate_all_folders(self) -> bool:
        """Validate all folders in the list still exist and contain required files."""
        invalid_folders = []
        for folder in self.folder_list[:]:  # Create a copy to iterate
            if not os.path.exists(folder):
                invalid_folders.append((folder, "Folder no longer exists"))
                continue

            missing_files = self._check_folder_contents(folder)
            if missing_files:
                invalid_folders.append((folder, f"Missing files: {', '.join(missing_files)}"))

        if invalid_folders:
            message = "The following folders are invalid:\n\n"
            for folder, reason in invalid_folders:
                message += f"{folder}\n{reason}\n\n"
            message += "Would you like to remove these folders from the list?"

            reply = QMessageBox.question(
                self,
                "Invalid Folders",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                for folder, _ in invalid_folders:
                    self.folder_list.remove(folder)
                    # Find and remove item from QListWidget
                    items = self.folder_list_widget.findItems(folder, Qt.MatchExactly)
                    for item in items:
                        self.folder_list_widget.takeItem(self.folder_list_widget.row(item))
                self._update_ui_state()

            return False

        return True

    # endregion === Folder Management ===

    # region === State Management ===
    def _update_ui_state(self):
        """Update UI element states."""
        has_folders = len(self.folder_list) > 0
        self.run_analysis_btn.setEnabled(has_folders)
        self.clear_folders_btn.setEnabled(has_folders)

    def _handle_error(self, error_message: str):
        """Handle error by showing message box."""
        QMessageBox.critical(self, "Error", error_message)

    # endregion === State Management ===

    # region === Execution ===

    def _run_batch_analysis(self):
        """Run batch analysis according to selected console option."""
        if not self.folder_list:
            QMessageBox.warning(self, "No Folders", "Please add folders to analyze first.")
            return

        # Validate folders before proceeding
        if not self._validate_all_folders():
            return

        try:
            # Generate configuration dictionary
            config = self.generate_config()

            # Create temporary YAML file to store configuration
            with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_yaml:
                yaml.safe_dump(config, temp_yaml, default_flow_style=False)
                config_path = temp_yaml.name

            if self.napari_console_radio.isChecked():
                # Run directly in napari console
                print("Starting batch analysis in napari console...")

                # Create output directories
                for folder in config["root_folders"]:
                    tfm_data_dir = Path(folder) / "TFM_data"
                    tfm_data_dir.mkdir(exist_ok=True)

                # Run analysis
                analyzer = BatchAnalysis(config)
                analyzer.process_all_folders()

                # Clean up config file
                Path(config_path).unlink()

            else:
                # Run in new console - implementation remains the same
                self._run_in_new_console(config_path)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start batch analysis: {str(e)}"
            )
            # Clean up temporary files
            if 'config_path' in locals():
                try:
                    Path(config_path).unlink()
                except:
                    pass

    def _run_in_new_console(self, config_path: str):
        """Run analysis in a new console window."""
        config_path_forward = str(Path(config_path)).replace('\\', '/')

        script_content = f'''
import sys
from pathlib import Path

# Add parent directory to Python path to find napariTFM package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from napariTFM.backend.batch_analysis import BatchAnalysis

# Create analyzer instance and process folders
config_path = "{config_path_forward}"
analyzer = BatchAnalysis.from_yaml(config_path)
analyzer.process_all_folders()

# Clean up temporary config file
Path(config_path).unlink()
'''

        # Create temporary Python script file
        with NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            temp_script.write(script_content)
            script_path = temp_script.name

        # Launch new Python console running the script
        self._launch_console(script_path)

    def _launch_console(self, script_path: str):
        """Launch the appropriate console based on platform."""
        python_executable = sys.executable

        if sys.platform == 'win32':
            subprocess.Popen(['start', 'cmd', '/k', python_executable, script_path],
                             shell=True)
        else:
            if sys.platform == 'darwin':
                subprocess.Popen(['open', '-a', 'Terminal',
                                  python_executable, script_path])
            else:
                terminals = ['gnome-terminal', 'xterm', 'konsole']
                for terminal in terminals:
                    try:
                        subprocess.Popen([terminal, '--', python_executable,
                                          script_path])
                        break
                    except FileNotFoundError:
                        continue
                else:
                    raise RuntimeError("No suitable terminal emulator found")

        QMessageBox.information(
            self,
            "Analysis Started",
            "Batch analysis has been started in a new console window.\n"
            "The analysis will continue running even if you close napari."
        )

    # endregion === Execution ===
