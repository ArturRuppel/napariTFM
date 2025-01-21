from typing import List, Dict, Optional
import os
import sys
from pathlib import Path
import numpy as np
import yaml
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget, QGridLayout,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame, QScrollArea,
    QProgressBar, QMessageBox, QListWidget, QCheckBox,
    QFileDialog, QComboBox
)

from .base_widget import BaseAnalysisWidget


class BatchAnalysisWidget(BaseAnalysisWidget):
    """Widget for running batch analysis on multiple folders."""

    batch_completed = Signal(dict)  # Emits results when batch processing completes

    def __init__(self, viewer, data_manager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        self.parameter_spins = {}
        self.parameter_combos = {}
        self.parameter_checks = {}
        self.analysis_checkboxes = {}
        self.visualization_checkboxes = {}
        self.folder_list = []

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

        # Add dictionary to store default parameter values
        self.default_parameters = self._get_parameter_dict()

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
        pixel_spin.setValue(1.0)
        pixel_spin.setDecimals(3)
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
        frame_spin.setValue(1.0)
        frame_spin.setDecimals(3)
        frame_spin.setToolTip("Time between consecutive frames in minutes")
        self.parameter_spins['frame_interval'] = frame_spin
        frame_row.addWidget(frame_spin)
        layout.addLayout(frame_row)

        group.setLayout(layout)
        return group

    def _create_preprocessing_params_group(self) -> QGroupBox:
        """Create preprocessing parameters group."""
        group = QGroupBox("Preprocessing Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Bead/Reference parameters
        bead_params = [
            ("min_spinbox", "Min Intensity (%)", 0, 100, 0.1, 0),
            ("max_spinbox", "Max Intensity (%)", 0, 100, 0.1, 100),
            ("gaussian_sigma_spin", "Gaussian Sigma", 0.0, 10.0, 0.1, 0.0)
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
            ("cell_min_spinbox", "Cell Min Intensity (%)", 0, 100, 0.1, 0),
            ("cell_max_spinbox", "Cell Max Intensity (%)", 0, 100, 0.1, 100),
            ("cell_gaussian_sigma_spin", "Cell Gaussian Sigma", 0.0, 10.0, 0.1, 0.0)
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
        reg_combo.addItems(['No registration', 'Translation', 'Rigid'])
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

        # Material parameters
        params = [
            ("young_spin", "Young's Modulus (kPa):", 0.1, 1000, 0.1, 10),
            ("poisson_spin", "Poisson's Ratio:", 0, 0.5, 0.01, 0.49),
            ("height_spin", "Gel Height (µm):", 0, 1000, 10, 0),
            ("lanczos_exp_spin", "Lanczos Exponent:", 0, 5, 1, 1),
            ("regularization_spin", "Regularization (10^x):", -21, 0, 0.5, -17)
        ]

        for name, label, min_val, max_val, step, default in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if name == "height_spin":
                spin.setSpecialValueText("∞")
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Auto-GCV checkbox
        self.parameter_checks['auto_gcv'] = QCheckBox("Auto-GCV per frame")
        layout.addWidget(self.parameter_checks['auto_gcv'])

        # Visualization parameters
        vis_params = [
            ("force_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("force_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("f_max", "Max Force (Pa):", 0.1, 10000.0, 0.1, 1000.0)
        ]

        for name, label, min_val, max_val, step, default in vis_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_stress_params_group(self) -> QGroupBox:
        """Create stress analysis parameters group."""
        group = QGroupBox("Stress Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Mask parameters first
        mask_params = [
            ("threshold", "Threshold Percentile:", 0, 100, 0.1, 0),
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 10),
            ("smoothing_sigma", "Boundary Smoothing:", 0.0, 40.0, 0.1, 10)
        ]

        for name, label, min_val, max_val, step, default in mask_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if isinstance(spin, QDoubleSpinBox):
                spin.setDecimals(2)
            if name == "threshold":
                spin.setSuffix("%")
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Then mesh parameters
        # Density factor
        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Density Factor:"))
        density_spin = QDoubleSpinBox()
        density_spin.setRange(0.001, 0.1)
        density_spin.setSingleStep(0.001)
        density_spin.setValue(0.025)
        density_spin.setDecimals(3)
        self.parameter_spins['density_factor'] = density_spin
        density_row.addWidget(density_spin)
        layout.addLayout(density_row)

        # Mesh algorithm selection
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Mesh Algorithm:"))
        algo_combo = QComboBox()
        algo_combo.addItems([
            "Frontal-Del.",
            "Delaunay",
            "MeshAdapt",
            "BAMG",
            "FD Quads",
            "Para. Pack"
        ])
        self.parameter_combos['mesh_algorithm'] = algo_combo
        algo_row.addWidget(algo_combo)
        layout.addLayout(algo_row)

        # Mesh optimization checkbox
        self.parameter_checks['use_optimization'] = QCheckBox("Mesh Optimization")
        self.parameter_checks['use_optimization'].setChecked(True)
        layout.addWidget(self.parameter_checks['use_optimization'])

        # Visualization parameter at the end
        max_stress_row = QHBoxLayout()
        max_stress_row.addWidget(QLabel("Max Stress (mN/m):"))
        max_stress_spin = QDoubleSpinBox()
        max_stress_spin.setRange(0.01, 1000.0)
        max_stress_spin.setSingleStep(0.1)
        max_stress_spin.setValue(1.0)
        max_stress_spin.setDecimals(2)
        self.parameter_spins['max_stress'] = max_stress_spin
        max_stress_row.addWidget(max_stress_spin)
        layout.addLayout(max_stress_row)

        group.setLayout(layout)
        return group
    def _setup_ui(self):
        """Set up the user interface."""
        # Keep existing scroll area setup
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Main container
        container = QWidget()
        main_layout = QVBoxLayout()

        # Add widget groups
        main_layout.addWidget(self._create_general_params_group())
        main_layout.addWidget(self._create_preprocessing_params_group())
        main_layout.addWidget(self._create_displacement_params_group())
        main_layout.addWidget(self._create_force_params_group())
        main_layout.addWidget(self._create_stress_params_group())
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

        # Add new signal connections
        self.save_params_btn.clicked.connect(self._save_parameters)
        self.load_params_btn.clicked.connect(self._load_parameters)
        self.reset_params_btn.clicked.connect(self._reset_parameters)

    def _get_parameter_dict(self) -> dict:
        """Get dictionary of current parameter values."""
        params = {}

        # Add values from spinboxes
        for name, spin in self.parameter_spins.items():
            params[name] = spin.value()

        # Add values from checkboxes
        for name, check in self.parameter_checks.items():
            params[name] = check.isChecked()

        # Add values from comboboxes
        for name, combo in self.parameter_combos.items():
            params[name] = combo.currentText().lower()

        for name, checkbox in self.visualization_checkboxes.items():
            params[f'save_{name}'] = checkbox.isChecked()

        return params

    def _set_parameters(self, params: dict):
        """Set parameter values in UI elements."""
        # Update spinbox values
        for name, value in params.items():
            if name in self.parameter_spins:
                self.parameter_spins[name].setValue(value)

        # Update checkbox values
        for name, value in params.items():
            if name in self.parameter_checks:
                self.parameter_checks[name].setChecked(value)

        # Update combobox values
        for name, value in params.items():
            if name in self.parameter_combos:
                index = self.parameter_combos[name].findText(
                    value.capitalize(),
                    Qt.MatchFixedString
                )
                if index >= 0:
                    self.parameter_combos[name].setCurrentIndex(index)

        # Update visualization checkboxes
        for name, value in params.items():
            if name.startswith('viz_'):
                viz_name = name[4:]  # Remove 'viz_' prefix
                if viz_name in self.visualization_checkboxes:
                    self.visualization_checkboxes[viz_name].setChecked(value)

    def _save_parameters(self):
        """Save current parameters to a YAML file."""
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Parameters",
                "",
                "YAML Files (*.yaml *.yml)",
            )

            if not file_path:
                return

            # Add .yaml extension if not present
            if not file_path.lower().endswith(('.yaml', '.yml')):
                file_path += '.yaml'

            # Get current parameters
            params = self._get_parameter_dict()

            # Add visualization parameters
            for name, checkbox in self.visualization_checkboxes.items():
                params[f'viz_{name}'] = checkbox.isChecked()

            # Add comments for parameter groups
            yaml_str = "# TFM Analysis Parameters\n\n"

            # General parameters
            yaml_str += "# General Parameters\n"
            yaml_str += yaml.dump({'general': {
                'pixelsize': params['pixelsize']
            }}, default_flow_style=False)
            yaml_str += "\n"

            # Preprocessing parameters
            yaml_str += "# Preprocessing Parameters\n"
            preproc_params = {k: v for k, v in params.items() if k in [
                'min_intensity', 'max_intensity', 'gaussian_sigma',
                'enable_gaussian', 'cell_min_intensity', 'cell_max_intensity',
                'cell_gaussian_sigma', 'enable_cell_gaussian',
                'enable_registration', 'registration_mode'
            ]}
            yaml_str += yaml.dump({'preprocessing': preproc_params}, default_flow_style=False)
            yaml_str += "\n"

            # Displacement parameters
            yaml_str += "# Displacement Analysis Parameters\n"
            disp_params = {k: v for k, v in params.items() if k in [
                'tau', 'lambda_', 'theta', 'nscales', 'warps', 'epsilon',
                'inner_iterations', 'outer_iterations', 'scale_step',
                'median_filtering', 'downscale_factor', 'disp_vector_stride',
                'disp_arrow_scale', 'd_max'
            ]}
            yaml_str += yaml.dump({'displacement': disp_params}, default_flow_style=False)
            yaml_str += "\n"

            # Force parameters
            yaml_str += "# Force Analysis Parameters\n"
            force_params = {k: v for k, v in params.items() if k in [
                'youngs_modulus', 'poisson_ratio', 'gel_height', 'lanczos_exp',
                'regularization', 'force_vector_stride', 'force_arrow_scale',
                'f_max', 'auto_gcv'
            ]}
            yaml_str += yaml.dump({'force': force_params}, default_flow_style=False)
            yaml_str += "\n"

            # Stress parameters
            yaml_str += "# Stress Analysis Parameters\n"
            stress_params = {k: v for k, v in params.items() if k in [
                'target_nodes', 'boundary_refinement', 'gradient_refinement',
                'dilation', 'max_stress'
            ]}
            yaml_str += yaml.dump({'stress': stress_params}, default_flow_style=False)
            yaml_str += "\n"

            # Add visualization parameters
            yaml_str += "# Visualization Parameters\n"
            viz_params = {k: v for k, v in params.items() if k.startswith('viz_')}
            yaml_str += yaml.dump({'visualization': viz_params}, default_flow_style=False)

            # Save to file
            with open(file_path, 'w') as f:
                f.write(yaml_str)

            QMessageBox.information(self, "Success", "Parameters saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {str(e)}")

    def _load_parameters(self):
        """Load parameters from a YAML file."""
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Parameters",
                "",
                "YAML Files (*.yaml *.yml)",
            )

            if not file_path:
                return

            # Load parameters
            with open(file_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Flatten nested dictionary
            params = {}
            for section in yaml_data.values():
                params.update(section)

            # Update UI with loaded parameters
            self._set_parameters(params)

            # Update visualization checkboxes
            for name, checkbox in self.visualization_checkboxes.items():
                viz_param = f'viz_{name}'
                if viz_param in params:
                    checkbox.setChecked(params[viz_param])

            QMessageBox.information(self, "Success", "Parameters loaded successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

    def _reset_parameters(self):
        """Reset parameters to default values."""
        try:
            # Ask for confirmation
            reply = QMessageBox.question(
                self,
                "Confirm Reset",
                "Are you sure you want to reset all parameters to default values?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._set_parameters(self.default_parameters)
                QMessageBox.information(self, "Success", "Parameters reset to defaults!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reset parameters: {str(e)}")

    def _create_analysis_steps_group(self) -> QGroupBox:
        """Create analysis steps group with checkboxes."""
        group = QGroupBox("Analysis Steps")
        layout = QVBoxLayout()

        steps = [
            "preprocessing", "displacement", "force", "stress"
        ]

        for step in steps:
            checkbox = QCheckBox(step.capitalize())
            checkbox.setChecked(True)
            self.analysis_checkboxes[step] = checkbox
            layout.addWidget(checkbox)

        group.setLayout(layout)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        """Create visualization options group."""
        group = QGroupBox("Save Visualizations")
        layout = QVBoxLayout()

        viz_options = [
            "bead_overlay",
            "displacement_map",
            "force_map",
            "force_cell_overlay",
            "sigma_xx",
            "sigma_yy",
            "shear",
            "normal_stress"
        ]

        for viz in viz_options:
            checkbox = QCheckBox(viz.replace("_", " ").title())
            self.visualization_checkboxes[viz] = checkbox
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

        # Add parameter management buttons (top row)
        self.save_params_btn = QPushButton("Save Parameters")
        self.load_params_btn = QPushButton("Load Parameters")
        self.reset_params_btn = QPushButton("Reset Parameters")
        button_layout.addWidget(self.save_params_btn, 0, 0)  # row 0, col 0
        button_layout.addWidget(self.load_params_btn, 0, 1)  # row 0, col 1
        button_layout.addWidget(self.reset_params_btn, 0, 2)  # row 0, col 2

        # Add folder management and run buttons (bottom row)
        self.add_folder_btn = QPushButton("Add Folder")
        self.clear_folders_btn = QPushButton("Clear Folders")
        self.run_analysis_btn = QPushButton("Run Analysis")
        button_layout.addWidget(self.add_folder_btn, 1, 0)  # row 1, col 0
        button_layout.addWidget(self.clear_folders_btn, 1, 1)  # row 1, col 1
        button_layout.addWidget(self.run_analysis_btn, 1, 2)  # row 1, col 2

        layout.addLayout(button_layout)
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

    def _run_batch_analysis(self):
        """Run batch analysis by launching a new Python console."""
        if not self.folder_list:
            QMessageBox.warning(self, "Warning", "No folders queued for analysis")
            return

        try:
            # Get first folder path since we'll create script there
            base_folder = os.path.dirname(self.folder_list[0])
            script_path = os.path.join(base_folder, 'batch_analysis.py')

            # Create master script that processes all folders
            script_lines = []

            # Add imports
            script_lines.extend([
                "import os",
                "import sys",
                "import numpy as np",
                "import tifffile",
                "from napariTFM.preprocessing import PreprocessingParameters, ImagePreprocessor",
                "from napariTFM.displacement_analysis import TVL1Parameters, DisplacementAnalyzer",
                "from napariTFM.fttc import FTTC",
                "from napariTFM.msm import MonolayerStressMicroscopy",
                "from napariTFM.batch_analysis_visualizations import BatchVisualizationSaver\n"
            ])

            # Add folder list
            script_lines.append("folders = [")
            for folder in self.folder_list:
                script_lines.append(f"    r'{folder}',")
            script_lines.append("]\n")

            # Get parameters dictionary
            params = self._get_parameter_dict()
            script_lines.append(f"params = {repr(params)}\n")

            # Get enabled steps
            enabled_steps = [step for step, checkbox in self.analysis_checkboxes.items()
                             if checkbox.isChecked()]
            script_lines.append(f"enabled_steps = {repr(enabled_steps)}\n")

            # Add analysis function
            script_lines.extend([
                "def run_analysis(folder):",
                "    # Preprocessing",
                "    if 'preprocessing' in enabled_steps:",
                "        print('Running preprocessing...')",
                "        preprocessor = ImagePreprocessor()",
                "        # Load input data",
                "        try:",
                "            beads = tifffile.imread(os.path.join(folder, 'beads.tif'))",
                "            reference = tifffile.imread(os.path.join(folder, 'reference.tif'))",
                "            cells = None",
                "            if os.path.exists(os.path.join(folder, 'cells.tif')):",
                "                cells = tifffile.imread(os.path.join(folder, 'cells.tif'))",
                "",
                "            # Create preprocessing parameters",
                "            prep_params = PreprocessingParameters(",
                "                min_intensity_percentile=params['min_intensity'] / 100,",
                "                max_intensity_percentile=params['max_intensity'] / 100,",
                "                enable_gaussian_filter=params['enable_gaussian'],",
                "                gaussian_sigma=params['gaussian_sigma'],",
                "                cell_min_intensity_percentile=params['cell_min_intensity'] / 100,",
                "                cell_max_intensity_percentile=params['cell_max_intensity'] / 100,",
                "                enable_cell_gaussian_filter=params['enable_cell_gaussian'],",
                "                cell_gaussian_sigma=params['cell_gaussian_sigma'],",
                "                enable_registration=params['enable_registration'],",
                "                registration_mode=params['registration_mode']",
                "            )",
                "",
                "            # Update preprocessor parameters",
                "            preprocessor.update_parameters(prep_params)",
                "",
                "            # Process data",
                "            results = preprocessor.preprocess_all(beads, reference, cells)",
                "",
                "            # Save results",
                "            if 'beads' in results:",
                "                processed_beads, _ = results['beads']",
                "                processed_beads = (processed_beads * 65535).astype(np.uint16)",
                "                tifffile.imwrite(os.path.join(folder, 'preprocessed_beads.tif'),",
                "                                processed_beads)",
                "",
                "            if 'reference' in results:",
                "                processed_reference, _ = results['reference']",
                "                processed_reference = (processed_reference * 65535).astype(np.uint16)",
                "                tifffile.imwrite(os.path.join(folder, 'preprocessed_reference.tif'),",
                "                                processed_reference)",
                "",
                "            if 'cells' in results:",
                "                processed_cells, _ = results['cells']",
                "                processed_cells = (processed_cells * 65535).astype(np.uint16)",
                "                tifffile.imwrite(os.path.join(folder, 'preprocessed_cells.tif'),",
                "                                processed_cells)",
                "            # Add visualization saving\n"
                "            if params.get('save_bead_overlay', False):\n"
                "                try:\n"
                "                    viz_saver = BatchVisualizationSaver(folder)\n"
                "                    processed_beads = tifffile.imread(os.path.join(folder, 'preprocessed_beads.tif'))\n"
                "                    processed_reference = tifffile.imread(os.path.join(folder, 'preprocessed_reference.tif'))\n"
                "                    \n"
                "                    # Convert from uint16 back to float [0,1] range\n"
                "                    processed_beads = processed_beads.astype(float) / 65535\n"
                "                    processed_reference = processed_reference.astype(float) / 65535\n"
                "                    \n"
                "                    viz_saver.save_bead_overlay(processed_beads, processed_reference)\n"
                "                    print('Saved bead overlay visualization')\n"
                "                except Exception as e:\n"
                "                    print(f'Failed to save bead overlay visualization: {str(e)}')\n"
                "",
                "            print('Preprocessing completed successfully')",
                "",
                "        except Exception as e:",
                "            print(f'Preprocessing failed: {str(e)}')",
                "            return",
                "",
                "    # Displacement Analysis",
                "    if 'displacement' in enabled_steps:",
                "        print('Running displacement analysis...')",
                "        try:",
                "            # Create displacement parameters",
                "            disp_params = TVL1Parameters(",
                "                tau=params['tau'],",
                "                lambda_=params['lambda_'],",
                "                theta=params['theta'],",
                "                nscales=params['nscales'],",
                "                warps=params['warps'],",
                "                epsilon=params['epsilon'],",
                "                inner_iterations=params['inner_iterations'],",
                "                outer_iterations=params['outer_iterations'],",
                "                scale_step=params['scale_step'],",
                "                median_filtering=params['median_filtering']",
                "            )",
                "",
                "            # Initialize analyzer",
                "            analyzer = DisplacementAnalyzer(disp_params)",
                "",
                "            # Load data",
                "            beads = tifffile.imread(os.path.join(folder, 'preprocessed_beads.tif'))",
                "            reference = tifffile.imread(os.path.join(folder, 'preprocessed_reference.tif'))",
                "",
                "            # Calculate flows",
                "            flows = []",
                "            for i in range(len(beads)):",
                "                print(f'Processing frame {i+1}/{len(beads)}')",
                "                flow = analyzer.calculate_flow(reference, beads[i])",
                "                # Apply downscaling if factor > 1",
                "                if params['downscale_factor'] > 1:",
                "                    flow = analyzer.downscale_flow(flow, params['downscale_factor'])",
                "                ",
                "                flows.append(flow * params['pixelsize'])",
                "",
                "            # Save results",
                "            displacement_results = {",
                "                'flows': flows,",
                "                'parameters': {",
                "                    'pixelsize': params['pixelsize'],",
                "                    'downscale_factor': params['downscale_factor'],",
                "                    'arrow_scale': params['disp_arrow_scale'],",
                "                    'vector_stride': params['disp_vector_stride'],",
                "                    'd_max': params['d_max']",
                "                }",
                "            }",
                "",
                "            np.save(os.path.join(folder, 'displacement.npy'), displacement_results)",

                "            # Save displacement visualizations",
                "            if params.get('save_displacement_map', False):",
                "                print('Creating displacement visualization...')",
                "                try:",
                "                    visualizer = BatchVisualizationSaver(folder)",
                "                    visualizer.save_displacement_visualization(",
                "                        displacement_results,",
                "                        fps=10",
                "                    )",
                "                    print('Displacement visualization saved successfully')",
                "                except Exception as e:",
                "                    print(f'Failed to create displacement visualization: {str(e)}')\n",

                "            print('Displacement analysis completed successfully')",
                "",
                "        except Exception as e:",
                "            print(f'Displacement analysis failed: {str(e)}')",
                "            return",
                "",
                "    # Force Calculation",
                "    if 'force' in enabled_steps:",
                "        print('Running force calculation...')",
                "        try:",
                "            # Load displacement data",
                "            disp_data = np.load(os.path.join(folder, 'displacement.npy'),",
                "                               allow_pickle=True).item()",
                "            flows = disp_data['flows']",
                "",
                "            # Initialize FTTC calculator",
                "            calculator = FTTC(",
                "                E=params['youngs_modulus'],",
                "                nu=params['poisson_ratio'],",
                "                mesh_size=1,",
                "                lanczos_exp=params['lanczos_exp'],",
                "                gel_height=None if params['gel_height'] == 0 else params['gel_height'] * 1e-6  # Convert to meters if not zero",
                "            )",
                "",
                "            # Process each frame",
                "            tx_list = []",
                "            ty_list = []",
                "",
                "            for i, flow in enumerate(flows):",
                "                print(f'Processing frame {i+1}/{len(flows)}')",
                "                u_data = flow[..., 0]",
                "                v_data = flow[..., 1]",
                "",
                "                # Calculate traction forces",
                "                x = np.arange(u_data.shape[1])",
                "                y = np.arange(u_data.shape[0])",
                "                dx = params['pixelsize'] * params['downscale_factor']",
                "                set_lam = None if params['auto_gcv'] else 10**params['regularization']",
                "",
                "                xy, fnorm, f, urec, u, energy, force, Ftf, Fturec = calculator.calculate_traction(",
                "                    x=x,",
                "                    y=y,",
                "                    u_data=u_data,",
                "                    v_data=v_data,",
                "                    dx=dx,",
                "                    set_lam=set_lam",
                "                )",
                "",
                "                # Store results",
                "                tx_list.append(f[0])",
                "                ty_list.append(f[1])",
                "",
                "            # Save results",
                "            force_results = {",
                "                'tx': np.array(tx_list),",
                "                'ty': np.array(ty_list),",
                "                'parameters': {",
                "                    'youngs_modulus': params['youngs_modulus'],",
                "                    'poisson_ratio': params['poisson_ratio'],",
                "                    'gel_height': None if params['gel_height'] == 0 else params['gel_height'] * 1e-6,",
                "                    'pixelsize': dx,",
                "                    'vector_stride': params['force_vector_stride'],",
                "                    'arrow_scale': params['force_arrow_scale'],",
                "                    'f_max': params['f_max']",
                "                }",
                "            }",
                "",
                "            np.save(os.path.join(folder, 'traction_forces.npy'), force_results)",
                "",
                "            # Save force visualization",
                "            if params.get('save_force_map', False):",
                "                print('Creating force visualization...')",
                "                try:",
                "                    visualizer = BatchVisualizationSaver(folder)",
                "                    visualizer.save_force_visualization(",
                "                        force_results,",
                "                        fps=10",
                "                    )",
                "                    print('Force visualization saved successfully')",
                "                except Exception as e:",
                "                    print(f'Failed to create force visualization: {str(e)}')",
                "            # Save force visualization",
                "            if params.get('save_force_map', False):",
                "                print('Creating force visualization...')",
                "                try:",
                "                    visualizer = BatchVisualizationSaver(folder)",
                "                    visualizer.save_force_visualization(",
                "                        force_results,",
                "                        fps=10",
                "                    )",
                "                    print('Force visualization saved successfully')",
                "                except Exception as e:",
                "                    print(f'Failed to create force visualization: {str(e)}')",
                "",
                "            # Save force cell overlay visualization",
                "            if params.get('save_force_cell_overlay', False):",
                "                print('Creating force cell overlay visualization...')",
                "                try:",
                "                    visualizer = BatchVisualizationSaver(folder)",
                "                    cell_images = tifffile.imread(os.path.join(folder, 'preprocessed_cells.tif'))",
                "                    visualizer.save_force_cell_overlay(",
                "                        force_results,",
                "                        cell_images,",
                "                        fps=10",
                "                    )",
                "                    print('Force cell overlay visualization saved successfully')",
                "                except Exception as e:",
                "                    print(f'Failed to create force cell overlay visualization: {str(e)}')",
                "",
                "            print('Force calculation completed successfully')",
                "",
                "        except Exception as e:",
                "            print(f'Force calculation failed: {str(e)}')",
                "            return",
                "",
                "",
                "    # Stress Analysis",
                "    if 'stress' in enabled_steps:",
                "        print('Running stress analysis...')",
                "        try:",
                "            # Load force data",
                "            force_data = np.load(os.path.join(folder, 'traction_forces.npy'),",
                "                               allow_pickle=True).item()",
                "",
                "            # Try to load masks.tif first",
                "            mask_path = os.path.join(folder, 'masks.tif')",
                "            cell_data = None",
                "            if os.path.exists(mask_path):",
                "                cell_data = tifffile.imread(mask_path)",
                "            elif os.path.exists(os.path.join(folder, 'preprocessed_cells.tif')):",
                "                cell_data = tifffile.imread(os.path.join(folder, 'preprocessed_cells.tif'))",
                "            else:",
                "                raise FileNotFoundError('Neither masks.tif nor preprocessed_cells.tif found. Cannot proceed with stress analysis.')",
                "",
                "            # Initialize MSM calculator",
                "            msm = MonolayerStressMicroscopy(",
                "                pixelsize=params['pixelsize'] * params['downscale_factor'],",
                "                sigma=params['poisson_ratio'],",
                "                youngs_modulus=params['youngs_modulus'],",
                "                target_nodes=params['target_nodes'],",
                "                boundary_refinement=params['boundary_refinement'],",
                "                gradient_refinement=params['gradient_refinement']",
                "            )",
                "",
                "            # Create masks",
                "            print('Generating masks...')",
                "            masks = msm.create_mask_from_cells(",
                "                cell_data,",
                "                dilation_pixels=params['dilation'],",
                "                smoothing_sigma=params['smoothing_sigma']",
                "            )",
                "            if len(masks.shape) == 2:",
                "                masks = masks[np.newaxis, ...]  # Add time dimension if single mask",
                "",
                "            # Save masks",
                "            tifffile.imwrite(os.path.join(folder, 'masks.tif'), masks.astype(np.uint8) * 255)",
                "            print('Masks saved successfully')",
                "",
                "            # Use first mask for stress calculation (assuming static cell boundary)",
                "            mask = masks[0]",
                "",
                "            # Calculate stress tensors",
                "            stress_tensors = []",
                "            for i, (tx, ty) in enumerate(zip(force_data['tx'], force_data['ty'])):",
                "                print(f'Processing frame {i+1}/{len(force_data[\"tx\"])}')",
                "                stress = msm.calculate_stress_field(tx, ty, mask) / (params['pixelsize'] * 1e-6)",
                "                stress_tensors.append(stress)",
                "",
                "            # Save results",
                "            stress_results = {",
                "                'stress_tensor': np.array(stress_tensors),",
                "                'parameters': {",
                "                    'pixelsize': params['pixelsize'],",
                "                    'youngs_modulus': params['youngs_modulus'],",
                "                    'target_nodes': params['target_nodes'],",
                "                    'max_stress': params['max_stress'],",
                "                    'dilation': params['dilation'],",
                "                    'smoothing_sigma': params['smoothing_sigma'],",
                "                    'save_sigma_xx': params.get('save_sigma_xx', False),",
                "                    'save_sigma_yy': params.get('save_sigma_yy', False),",
                "                    'save_shear': params.get('save_shear', False),",
                "                    'save_normal_stress': params.get('save_normal_stress', False)",
                "                }",
                "            }",
                "",
                "            np.save(os.path.join(folder, 'stress_tensor.npy'), stress_results)",
                "            # Save stress visualizations",
                "            if any(params.get(f'save_{comp}', False) for comp in ['sigma_xx', 'sigma_yy', 'shear', 'normal_stress']):",
                "                print('Creating stress visualizations...')",
                "                try:",
                "                    visualizer = BatchVisualizationSaver(folder)",
                "                    visualizer.save_stress_visualization(",
                "                        stress_results,",
                "                        fps=10",
                "                    )",
                "                    print('Stress visualizations saved successfully')",
                "                except Exception as e:",
                "                    print(f'Failed to create stress visualizations: {str(e)}')",

                "            print('Stress analysis completed successfully')",
                "",
                "        except Exception as e:",
                "            print(f'Stress analysis failed: {str(e)}')",
                "            return",
                "",
                "print('\\nBatch processing started...')",
                "for folder in folders:",
                "    print(f'\\nProcessing folder: {folder}')",
                "    run_analysis(folder)",
                "print('\\nBatch processing completed!')",
            ])

            # Write script
            with open(script_path, 'w') as f:
                f.write('\n'.join(script_lines))

            # Launch new Python console with script
            if sys.platform == 'win32':
                os.system(f'start cmd /k python "{script_path}"')
            else:
                os.system(f'x-terminal-emulator -e python "{script_path}"')

        except Exception as e:
            self._handle_error(str(e))

    def _add_folder(self):
        """Add folder to analysis queue."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            os.path.expanduser("~")
        )

        if folder:
            # Validate folder contents
            if self._validate_folder(folder):
                self.folder_list.append(folder)
                self.folder_list_widget.addItem(folder)
                self._update_ui_state()
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Folder",
                    "Folder must contain at least beads.tif and reference.tif"
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

    def _update_ui_state(self):
        """Update UI element states."""
        has_folders = len(self.folder_list) > 0
        self.run_analysis_btn.setEnabled(has_folders)
        self.clear_folders_btn.setEnabled(has_folders)

        # Parameter management buttons are always enabled
        self.save_params_btn.setEnabled(True)
        self.load_params_btn.setEnabled(True)
        self.reset_params_btn.setEnabled(True)

    def _handle_error(self, error_message: str):
        """Handle error by showing message box."""
        QMessageBox.critical(self, "Error", error_message)
