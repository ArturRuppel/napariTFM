from typing import List, Dict, Optional
import os
import sys
from pathlib import Path
import subprocess
import tifffile
import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget,
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

    def _setup_ui(self):
        """Set up the user interface."""
        # Create scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Main container
        container = QWidget()
        main_layout = QVBoxLayout()

        # Add parameter groups
        main_layout.addWidget(self._create_general_params_group())
        main_layout.addWidget(self._create_preprocessing_params_group())
        main_layout.addWidget(self._create_displacement_params_group())
        main_layout.addWidget(self._create_force_params_group())
        main_layout.addWidget(self._create_stress_params_group())

        # Analysis steps and visualization options
        main_layout.addWidget(self._create_analysis_steps_group())
        main_layout.addWidget(self._create_visualization_group())

        # Folder management
        main_layout.addWidget(self._create_folder_management_group())

        # Status frame
        main_layout.addWidget(self._create_status_frame())

        container.setLayout(main_layout)
        scroll.setWidget(container)

        # Main widget layout
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _create_general_params_group(self) -> QGroupBox:
        """Create general parameters group."""
        group = QGroupBox("General Parameters")
        layout = QVBoxLayout()

        # Pixel size (shared between all analyses)
        row = QHBoxLayout()
        row.addWidget(QLabel("Pixel Size (µm):"))
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 10.0)
        spin.setSingleStep(0.01)
        spin.setValue(1.0)
        spin.setDecimals(3)
        self.parameter_spins['pixelsize'] = spin
        row.addWidget(spin)
        layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_preprocessing_params_group(self) -> QGroupBox:
        """Create preprocessing parameters group."""
        group = QGroupBox("Preprocessing Parameters")
        layout = QVBoxLayout()

        # Bead/Reference parameters
        params = [
            ("min_intensity", "Min Intensity (%):", 0, 100, 0.1, 0),
            ("max_intensity", "Max Intensity (%):", 0, 100, 0.1, 100),
            ("gaussian_sigma", "Gaussian Sigma:", 0.1, 10.0, 0.1, 1.0),
        ]

        for name, label, min_val, max_val, step, default in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Gaussian filter checkbox
        self.parameter_checks['enable_gaussian'] = QCheckBox("Enable Gaussian Filter")
        layout.addWidget(self.parameter_checks['enable_gaussian'])

        # Cell parameters
        cell_params = [
            ("cell_min_intensity", "Cell Min Intensity (%):", 0, 100, 0.1, 0),
            ("cell_max_intensity", "Cell Max Intensity (%):", 0, 100, 0.1, 100),
            ("cell_gaussian_sigma", "Cell Gaussian Sigma:", 0.1, 10.0, 0.1, 1.0),
        ]

        for name, label, min_val, max_val, step, default in cell_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Cell Gaussian filter checkbox
        self.parameter_checks['enable_cell_gaussian'] = QCheckBox("Enable Cell Gaussian Filter")
        layout.addWidget(self.parameter_checks['enable_cell_gaussian'])

        # Registration parameters
        self.parameter_checks['enable_registration'] = QCheckBox("Enable Registration")
        layout.addWidget(self.parameter_checks['enable_registration'])

        row = QHBoxLayout()
        row.addWidget(QLabel("Registration Mode:"))
        combo = QComboBox()
        combo.addItems(['Translation', 'Rigid'])
        self.parameter_combos['registration_mode'] = combo
        row.addWidget(combo)
        layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_displacement_params_group(self) -> QGroupBox:
        """Create displacement analysis parameters group."""
        group = QGroupBox("Displacement Parameters")
        layout = QVBoxLayout()

        params = [
            ("tau", "Tau:", 0.01, 1.0, 0.01, 0.25),
            ("lambda_", "Lambda:", 0.01, 1.0, 0.01, 0.4),
            ("theta", "Theta:", 0.1, 1.0, 0.1, 0.3),
            ("nscales", "Pyramid Scales:", 1, 10, 1, 3),
            ("warps", "Warps:", 1, 10, 1, 3),
            ("epsilon", "Epsilon:", 0.001, 0.1, 0.001, 0.01),
            ("inner_iterations", "Inner Iterations:", 1, 50, 1, 15),
            ("outer_iterations", "Outer Iterations:", 1, 20, 1, 5),
            ("scale_step", "Scale Step:", 0.1, 0.99, 0.01, 0.5),
            ("median_filtering", "Median Filter Size:", 1, 9, 2, 5),
            ("downscale_factor", "Downscale Factor:", 1, 10, 1, 1),
            ("disp_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("disp_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("d_max", "Max Displacement (µm):", 0.1, 200.0, 0.1, 10.0),
        ]

        for name, label, min_val, max_val, step, default in params:
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

        group.setLayout(layout)
        return group

    def _create_force_params_group(self) -> QGroupBox:
        """Create force analysis parameters group."""
        group = QGroupBox("Force Parameters")
        layout = QVBoxLayout()

        params = [
            ("youngs_modulus", "Young's Modulus (Pa):", 100, 1000000, 100, 10000),
            ("poisson_ratio", "Poisson Ratio:", 0, 0.5, 0.01, 0.5),
            ("gel_height", "Gel Height (µm):", 0, 1000, 1, 0),
            ("lanczos_exp", "Lanczos Exponent:", 0, 5, 1, 1),
            ("regularization", "Regularization (10^x):", -21, 0, 0.5, -17),
            ("force_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("force_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("f_max", "Max Force (Pa):", 0.1, 10000.0, 0.1, 1000.0),
        ]

        for name, label, min_val, max_val, step, default in params:
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

        # Auto-GCV checkbox
        self.parameter_checks['auto_gcv'] = QCheckBox("Auto-GCV per frame")
        layout.addWidget(self.parameter_checks['auto_gcv'])

        group.setLayout(layout)
        return group

    def _create_stress_params_group(self) -> QGroupBox:
        """Create stress analysis parameters group."""
        group = QGroupBox("Stress Parameters")
        layout = QVBoxLayout()

        params = [
            ("target_nodes", "Target Nodes:", 100, 10000, 100, 1000),
            ("boundary_refinement", "Boundary Refinement:", 1.0, 5.0, 0.1, 2.0),
            ("gradient_refinement", "Gradient Refinement:", 1.0, 5.0, 0.1, 1.5),
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 0),
            ("max_stress", "Max Stress (mN/m):", 0.1, 1000.0, 0.1, 10.0),
        ]

        for name, label, min_val, max_val, step, default in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            if isinstance(step, int):
                spin = QSpinBox()
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(2)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

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

        # Buttons
        button_layout = QHBoxLayout()
        self.add_folder_btn = QPushButton("Add Folder")
        self.clear_folders_btn = QPushButton("Clear Folders")
        self.run_analysis_btn = QPushButton("Run Analysis")

        button_layout.addWidget(self.add_folder_btn)
        button_layout.addWidget(self.clear_folders_btn)
        button_layout.addWidget(self.run_analysis_btn)

        layout.addLayout(button_layout)

        # Folder list
        self.folder_list_widget = QListWidget()
        layout.addWidget(self.folder_list_widget)

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


    def _connect_signals(self):
        """Connect widget signals."""
        self.add_folder_btn.clicked.connect(self._add_folder)
        self.clear_folders_btn.clicked.connect(self._clear_folders)
        self.run_analysis_btn.clicked.connect(self._run_batch_analysis)

        # Parameter inheritance connections
        for spin in self.parameter_spins.values():
            spin.valueChanged.connect(self._sync_parameters)

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

    def _sync_parameters(self):
        """Synchronize parameters with other widgets."""
        # This will be implemented when integrating with other widgets
        pass

    def _create_analysis_script(self, folder: str) -> str:
        """Create Python script for running analysis on a folder."""
        script = f"""
import os
import numpy as np
import tifffile
from napariTFM.preprocessing import PreprocessingParameters, ImagePreprocessor
from napariTFM.displacement_analysis import TVL1Parameters, DisplacementAnalyzer
from napariTFM.fttc import FTTC
from napariTFM.msm import MonolayerStressMicroscopy

def run_analysis(folder):
    # Parameters
    params = {self._get_parameter_dict()}

    # Analysis steps
    steps = {self._get_enabled_steps()}

    # Preprocessing
    if 'preprocessing' in steps:
        preprocessor = ImagePreprocessor()
        # Load data
        beads = tifffile.imread(os.path.join(folder, 'beads.tif'))
        reference = tifffile.imread(os.path.join(folder, 'reference.tif'))
        cells = None
        if os.path.exists(os.path.join(folder, 'cells.tif')):
            cells = tifffile.imread(os.path.join(folder, 'cells.tif'))

        # Process
        results = preprocessor.preprocess_all(beads, reference, cells)

        # Save results
        if 'beads' in results:
            processed_beads, _ = results['beads']
            tifffile.imwrite(os.path.join(folder, 'preprocessed_beads.tif'), 
                           processed_beads.astype(np.uint16))

        if 'reference' in results:
            processed_reference, _ = results['reference']
            tifffile.imwrite(os.path.join(folder, 'preprocessed_reference.tif'),
                           processed_reference.astype(np.uint16))

        if 'cells' in results:
            processed_cells, _ = results['cells']
            tifffile.imwrite(os.path.join(folder, 'preprocessed_cells.tif'),
                           processed_cells.astype(np.uint16))

    # Displacement Analysis
    if 'displacement' in steps:
        if os.path.exists(os.path.join(folder, 'preprocessed_beads.tif')):
            analyzer = DisplacementAnalyzer()
            beads = tifffile.imread(os.path.join(folder, 'preprocessed_beads.tif'))
            reference = tifffile.imread(os.path.join(folder, 'preprocessed_reference.tif'))

            flows = []
            for i in range(len(beads)):
                flow = analyzer.calculate_flow(reference, beads[i])
                flows.append(flow)

            np.save(os.path.join(folder, 'displacement.npy'), flows)

    # Force Calculation
    if 'force' in steps:
        if os.path.exists(os.path.join(folder, 'displacement.npy')):
            flows = np.load(os.path.join(folder, 'displacement.npy'))
            calculator = FTTC(params['youngs_modulus'], params['poisson_ratio'])

            tx = []
            ty = []
            for flow in flows:
                result = calculator.calculate_traction(
                    flow[..., 0], flow[..., 1], params['pixelsize']
                )
                tx.append(result[0])
                ty.append(result[1])

            np.save(os.path.join(folder, 'traction_forces.npy'), 
                   {'tx': tx, 'ty': ty})

    # Stress Analysis
    if 'stress' in steps:
        if os.path.exists(os.path.join(folder, 'traction_forces.npy')):
            forces = np.load(os.path.join(folder, 'traction_forces.npy'), 
                           allow_pickle=True).item()

            cells = tifffile.imread(os.path.join(folder, 'preprocessed_cells.tif'))
            mask = cells > 0

            analyzer = MonolayerStressMicroscopy(
                pixelsize=params['pixelsize'],
                target_nodes=params['target_nodes'],
                boundary_refinement=params['boundary_refinement']
            )

            stress_tensors = []
            for tx, ty in zip(forces['tx'], forces['ty']):
                stress = analyzer.calculate_stress_field(tx, ty, mask)
                stress_tensors.append(stress)

            np.save(os.path.join(folder, 'stress_tensor.npy'), 
                   np.array(stress_tensors))

# Run analysis
run_analysis('{folder}')
"""
        return script

    def _get_parameter_dict(self) -> dict:
        """Get dictionary of current parameter values."""
        return {name: spin.value() for name, spin in self.parameter_spins.items()}

    def _get_enabled_steps(self) -> List[str]:
        """Get list of enabled analysis steps."""
        return [step for step, checkbox in self.analysis_checkboxes.items()
                if checkbox.isChecked()]

    def _run_batch_analysis(self):
        """Run batch analysis on queued folders."""
        if not self.folder_list:
            QMessageBox.warning(self, "Warning", "No folders queued for analysis")
            return

        try:
            self._set_controls_enabled(False)

            for i, folder in enumerate(self.folder_list):
                progress = (i / len(self.folder_list)) * 100
                self._update_status(f"Processing folder {i + 1}/{len(self.folder_list)}:\n{folder}", progress)

                # Create and run analysis script
                script = self._create_analysis_script(folder)
                script_path = os.path.join(folder, 'analysis_script.py')

                with open(script_path, 'w') as f:
                    f.write(script)

                # Run script in new Python process
                process = subprocess.Popen([sys.executable, script_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    raise RuntimeError(f"Analysis failed:\n{stderr.decode()}")

                # Clean up script
                os.remove(script_path)

            self._update_status("Batch analysis completed successfully", 100)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _update_ui_state(self):
        """Update UI element states."""
        has_folders = len(self.folder_list) > 0
        self.run_analysis_btn.setEnabled(has_folders)
        self.clear_folders_btn.setEnabled(has_folders)