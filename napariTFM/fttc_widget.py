import os
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QFileDialog, QCheckBox,
    QDoubleSpinBox, QPushButton, QFrame, QProgressBar, QMessageBox,
    QWidget, QSizePolicy, QSpinBox
)

from .base_widget import BaseAnalysisWidget
from .colorbar import ColorbarManager
from .data_manager import DataManager
from .fttc import FTTC
from .visualization_manager import VisualizationManager


class FTTCWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method."""

    force_calculated = Signal(dict)
    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        # Keep existing initialization
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize parameters (keep existing ones)
        self.young_modulus = 10  # kPa (will be converted to Pa internally)
        self.poisson_ratio = 0.49
        self.gel_height = None  # μm (None means infinite)
        self._pixel_size = None  # Will be set from data manager
        self._downscale_factor = None  # Will be set from data manager
        self.regularization = 1e-6
        self.mesh_size = 1  # hardcoded to 1
        self.lanczos_exp = 1

        # Add flag to track if analysis is running
        self.is_analysis_running = False

        # Initialize calculator
        self.calculator = None
        self.colorbar_manager = ColorbarManager()
        self.visualization_params = {}

        self._setup_ui()
        self._connect_signals()
        self._register_controls()
        self._update_ui_state()

    def _connect_signals(self):
        """Connect all widget signals."""
        # Keep existing signal connections
        self.young_spin.valueChanged.connect(self._update_parameters)
        self.poisson_spin.valueChanged.connect(self._update_parameters)
        self.height_spin.valueChanged.connect(self._update_parameters)
        self.lanczos_exp_spin.valueChanged.connect(self._update_parameters)
        self.regularization_spin.valueChanged.connect(self._update_parameters)

        # Action buttons
        self.calculate_btn.clicked.connect(self.calculate_forces)
        self.preview_btn.clicked.connect(self.preview_force)
        self.save_force_btn.clicked.connect(self._save_force_data)
        self.load_force_btn.clicked.connect(self._load_force_data)
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        self.clear_data_btn.clicked.connect(self._clear_data)

        # Add GCV control connections
        self.gcv_button.clicked.connect(self._auto_select_gcv)
        self.auto_gcv_checkbox.stateChanged.connect(self._toggle_auto_gcv)

    def _auto_select_gcv(self):
        """Calculate optimal regularization parameter for current frame using GCV."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating optimal regularization parameter...", 0)

            # Initialize calculator if needed
            self._initialize_calculator()

            # Get current frame data
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            current_frame = self.viewer.dims.current_step[0]
            flow = flows[current_frame]

            # Get spatial coordinates and prepare data
            shape = flow.shape[:-1]
            pos = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy'))
            vec = np.array([flow[..., 0], flow[..., 1]])

            # Scale displacements
            pix_per_mu = self.mesh_size / (self._pixel_size * self._downscale_factor)
            vec = pix_per_mu * vec

            # Calculate optimal regularization parameter
            lam = self.calculator._find_regularization(pos, vec)

            # Update UI with new value (log scale)
            self.regularization_spin.setValue(np.log10(lam))
            self.regularization = lam

            self._set_controls_enabled(True)
            self._update_status(f"Optimal regularization parameter: {lam:.2e}", 100)

        except Exception as e:
            self._handle_error(str(e))
            self._set_controls_enabled(True)

    def _toggle_auto_gcv(self, state):
        """Enable or disable automatic GCV calculation per frame."""
        self.regularization_spin.setEnabled(not state)
        self.gcv_button.setEnabled(not state)

    def calculate_forces(self):
        """Calculate traction forces using the FTTC calculator."""
        try:
            if not self._validate_input_data():
                return

            # Set analysis running flag and disable controls
            self.is_analysis_running = True
            self._set_controls_enabled(False)
            self._update_status("Starting force calculation...", 0)

            # Get displacement data
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            self.total_frames = len(flows)

            # Initialize result arrays
            self.force_results = {
                'tx': [],
                'ty': []
            }
            self.flows = flows
            self.current_frame = 0

            # Initialize calculator if needed
            self._initialize_calculator()

            # Start processing first frame
            self._process_next_frame()

        except Exception as e:
            self._handle_error(str(e))
            self.is_analysis_running = False
            self._set_controls_enabled(True)

    def _process_next_frame(self):
        """Process the next frame in the sequence."""
        try:
            if self.current_frame >= self.total_frames:
                self._finalize_force_results()
                return

            # Get spatial coordinates for current frame
            shape = self.flows[self.current_frame].shape[:-1]
            x = np.arange(shape[1])  # Width
            y = np.arange(shape[0])  # Height

            # Determine regularization parameter
            if self.auto_gcv_checkbox.isChecked():
                # Calculate optimal regularization for this frame
                pos = np.array(np.meshgrid(x, y, indexing='xy'))
                vec = np.array([
                    self.flows[self.current_frame][..., 0],
                    self.flows[self.current_frame][..., 1]
                ])
                pix_per_mu = self.mesh_size / (self._pixel_size * self._downscale_factor)
                vec = pix_per_mu * vec
                lam = self.calculator._find_regularization(pos, vec)
            else:
                lam = self.regularization

            # Create worker for current frame
            worker = self.calculator.calculate_traction(
                x=x,
                y=y,
                u_data=self.flows[self.current_frame][..., 0],
                v_data=self.flows[self.current_frame][..., 1],
                dx=self._pixel_size * self._downscale_factor,
                set_lam=lam
            )

            # Connect worker signals with proper error handling
            worker.returned.connect(self._handle_force_frame)
            worker.finished.connect(self._on_frame_finished)
            worker.errored.connect(self._handle_error)

            # Start the worker
            worker.start()

        except Exception as e:
            self._handle_error(str(e))


    def _handle_force_frame(self, results):
        """Handle results from a single frame calculation."""
        try:
            # Unpack results
            (_, _), _, f, _, _, energy, force, _, _ = results

            # Store force components
            self.force_results['tx'].append(f[0])
            self.force_results['ty'].append(f[1])

            # Update progress
            progress = (self.current_frame + 1) / self.total_frames * 100
            self._update_status(
                f"Processing frame {self.current_frame + 1}/{self.total_frames}...\n"
                f"Computing forces...",
                progress
            )

        except Exception as e:
            self._handle_error(str(e))

    def _on_frame_finished(self):
        """Handle completion of a frame calculation."""
        try:
            # Increment frame counter
            self.current_frame += 1

            # Process next frame or finalize
            if self.current_frame < self.total_frames:
                self._process_next_frame()
            else:
                self._finalize_force_results()

        except Exception as e:
            self._handle_error(str(e))

    def _finalize_force_results(self):
        """Finalize and handle the complete force calculation results."""
        try:
            # Convert lists to arrays
            self.force_results['tx'] = np.stack(self.force_results['tx'])
            self.force_results['ty'] = np.stack(self.force_results['ty'])

            # Store calculation parameters
            visualization_params = {
                'vector_stride': self.visualization_params['vector_stride'].value(),
                'arrow_scale': self.visualization_params['arrow_scale'].value(),
                'f_max': self.visualization_params['f_max'].value()
            }

            displacement_results = self.data_manager.displacement_results
            downscale_factor = displacement_results.get('parameters', {}).get('downscale_factor', 1)

            self.force_results['parameters'] = {
                'young_modulus': self.young_modulus,
                'poisson_ratio': self.poisson_ratio,
                'gel_height': self.gel_height,
                'pixel_size': self._pixel_size,
                'regularization': self.regularization,
                'mesh_size': self.mesh_size,
                'lanczos_exp': self.lanczos_exp,
                'downscale_factor': downscale_factor,
                'visualization': visualization_params
            }

            # Handle final results
            self._handle_force_results(self.force_results)

            # Clean up
            self.is_analysis_running = False
            self._set_controls_enabled(True)
            self.current_frame = 0
            self.total_frames = 0
            self.flows = None

        except Exception as e:
            self._handle_error(str(e))
            self.is_analysis_running = False
            self._set_controls_enabled(True)

    def _handle_error(self, error):
        """Handle errors during calculation."""
        error_message = str(error)
        self._update_status(f"Error: {error_message}", 0)

        # Reset analysis state
        self.is_analysis_running = False
        self._set_controls_enabled(True)

        # Clean up calculation state
        self.current_frame = 0
        self.total_frames = 0
        self.flows = None

        # Show error dialog
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred during calculation:\n{error_message}"
        )

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Check displacement results availability
        has_displacement = False
        if hasattr(self.data_manager, 'displacement_results'):
            results = self.data_manager.displacement_results
            if results and isinstance(results, dict) and 'flows' in results:
                flows = results['flows']
                try:
                    # Convert to numpy array if it isn't already
                    if not isinstance(flows, np.ndarray):
                        flows = np.array(flows)
                    self.displacement_status.setText(f"Displacement data: {flows.shape}")
                    has_displacement = True  # Set to True only if we successfully validated the data
                except Exception as e:
                    self.displacement_status.setText(f"Displacement data: Error ({str(e)})")
            else:
                self.displacement_status.setText("Displacement data: Not loaded")
        else:
            self.displacement_status.setText("Displacement data: Not loaded")

        # Check force results
        if hasattr(self.data_manager, 'force_results') and self.data_manager.force_results is not None:
            results = self.data_manager.force_results
            if isinstance(results, dict) and 'tx' in results and 'ty' in results:
                try:
                    shape = results['tx'].shape
                    if len(shape) > 0:  # Check if shape is not empty
                        self.force_status.setText(f"Force results: {shape}")
                    else:
                        self.force_status.setText("Force results: Invalid shape")
                except Exception as e:
                    self.force_status.setText(f"Force results: Error ({str(e)})")
            else:
                self.force_status.setText("Force results: Not loaded")
        else:
            self.force_status.setText("Force results: Not loaded")

        # Update button states based on data availability and analysis state
        if self.is_analysis_running:
            # If analysis is running, disable the buttons regardless of data availability
            self.calculate_btn.setEnabled(False)
            self.preview_btn.setEnabled(False)
            self.status_label.setText("Analysis in progress...")
        else:
            # If no analysis is running, enable buttons if we have valid displacement data
            self.calculate_btn.setEnabled(has_displacement)
            self.preview_btn.setEnabled(has_displacement)

            if has_displacement:
                self.status_label.setText("Ready for force calculation")
            else:
                self.status_label.setText("Missing required displacement data")

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all registered controls."""
        self.is_analysis_running = not enabled

        # Only update controls if they're registered
        if hasattr(self, '_controls'):
            for control in self._controls:
                if control in [self.calculate_btn, self.preview_btn]:
                    # Skip these buttons as they're handled in _update_ui_state
                    continue
                control.setEnabled(enabled)

        # Update UI state to handle the calculation buttons
        self._update_ui_state()

    def preview_force(self):
        """Preview force calculation on current frame."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating forces...", 0)

            # Initialize calculator
            self._initialize_calculator()

            # Get current frame data
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            current_frame = self.viewer.dims.current_step[0]
            flow = flows[current_frame]

            # Get spatial coordinates
            x = np.arange(flow.shape[1])
            y = np.arange(flow.shape[0])

            # Create and start worker
            worker = self.calculator.calculate_traction(
                x=x,
                y=y,
                u_data=flow[..., 0],
                v_data=flow[..., 1],
                dx=self._pixel_size * self._downscale_factor,
                set_lam=self.regularization
            )

            # Connect worker signals
            worker.returned.connect(self._handle_preview_results)
            worker.finished.connect(lambda: self._set_controls_enabled(True))
            worker.errored.connect(self._handle_error)

            # Start the worker
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self._set_controls_enabled(True)

    def _handle_preview_results(self, results):
        """Handle the preview calculation results."""
        try:
            # Unpack results
            (_, _), fnorm, f, urec, u, energy, force, _, _ = results

            # Get current visualization parameters
            vector_stride = self.visualization_params['vector_stride'].value()
            arrow_scale = self.visualization_params['arrow_scale'].value()
            f_max = self.visualization_params['f_max'].value()

            # Update visualization
            self.visualization_manager.visualize_force_preview(
                f[0], f[1],
                f_max=f_max,
                vector_stride=vector_stride,
                arrow_scale=arrow_scale,
                downscale_factor=self.data_manager.displacement_results.get('parameters', {}).get('downscale_factor', 1)
            )

            # Handle layer visibility and ordering
            self._handle_visualization_layers()

            # Update colorbar
            self.colorbar_manager.update_limits(0, f_max)

            # Show statistics
            magnitude = np.sqrt(f[0] ** 2 + f[1] ** 2)
            self._update_status(
                f"Preview statistics:\n"
                f"Max force: {np.max(magnitude):.2f} Pa\n"
                f"Mean force: {np.mean(magnitude):.2f} Pa\n"
                f"Median force: {np.median(magnitude):.2f} Pa\n"
                f"Energy: {energy:.2e} J\n"
                f"Total force: {force:.2e} N",
                100
            )

        except Exception as e:
            self._handle_error(str(e))
            self._set_controls_enabled(True)

    def _start_next_frame(self):
        """Start calculation for next frame if available."""
        try:
            self.current_frame += 1

            if self.current_frame < self.total_frames:
                # Start next frame calculation
                shape = self.flows[0].shape[:-1]
                x = np.arange(shape[1])
                y = np.arange(shape[0])

                worker = self.calculator.calculate_traction(
                    x=x,
                    y=y,
                    u_data=self.flows[self.current_frame][..., 0],
                    v_data=self.flows[self.current_frame][..., 1],
                    dx=self._pixel_size,
                    set_lam=self.regularization
                )

                worker.returned.connect(self._handle_force_frame)
                worker.finished.connect(self._start_next_frame)
                worker.errored.connect(self._handle_error)

                worker.start()
            else:
                # All frames complete, finalize results
                self._finalize_force_results()

        except Exception as e:
            self._handle_error(str(e))
            self._set_controls_enabled(True)

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress bar."""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)

    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
                       self.young_spin,
                       self.poisson_spin,
                       self.height_spin,
                       self.lanczos_exp_spin,
                       self.regularization_spin,
                       self.calculate_btn,
                       self.preview_btn,
                       self.reset_params_btn,
                       self.save_force_btn,
                       self.load_force_btn,
                       self.clear_data_btn,
                       self.progress_bar,
                       self.status_label,
                       self.displacement_status,
                       self.force_status,
                       self.gcv_button,
                       self.auto_gcv_checkbox
                   ] + list(self.visualization_params.values())

        for control in controls:
            self.register_control(control)

    def _create_material_params_group(self) -> QGroupBox:
        """Create the material parameters group."""
        group = QGroupBox("Material Parameters")
        layout = QVBoxLayout()

        # Add reset parameters button at the top
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.reset_params_btn.setToolTip("Reset all parameters to their default values")
        layout.addWidget(self.reset_params_btn)

        # Create spinboxes
        self.young_spin = QDoubleSpinBox()
        self.poisson_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        self.lanczos_exp_spin = QSpinBox()

        params = [
            ("Young's Modulus (kPa):", self.young_spin, 0.1, 1000, 0.1, self.young_modulus,
             "Elastic modulus of the gel substrate in kilopascals (kPa)"),
            ("Poisson Ratio:", self.poisson_spin, 0, 0.5, 0.01, self.poisson_ratio,
             "Poisson's ratio of the gel substrate (typically 0.45-0.49 for hydrogels)"),
            ("Gel Height (μm):", self.height_spin, 0, 1000, 10, 0,
             "Thickness of the gel substrate in micrometers. Set to 0 for infinite thickness"),
            ("Lanczos Exponent:", self.lanczos_exp_spin, 0, 5, 1, self.lanczos_exp,
             "Exponent for Lanczos interpolation. Higher values increase smoothing")
        ]

        for label_text, spin, min_val, max_val, step, default, tooltip in params:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(150)
            label.setToolTip(tooltip)
            row.addWidget(label)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setToolTip(tooltip)
            if label_text.startswith("Gel Height"):
                spin.setSpecialValueText("∞")
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_calculation_params_group(self) -> QGroupBox:
        """Create the calculation parameters group."""
        group = QGroupBox("Calculation Parameters")
        layout = QVBoxLayout()

        # Regularization controls container
        reg_container = QGroupBox("Regularization")
        reg_container_layout = QVBoxLayout()

        # Regularization parameter with log scale
        reg_layout = QHBoxLayout()
        reg_layout.addWidget(QLabel("Parameter (10^x):"))
        self.regularization_spin = QDoubleSpinBox()
        self.regularization_spin.setRange(-21, 0)
        self.regularization_spin.setValue(-17)
        self.regularization_spin.setSingleStep(0.5)
        self.regularization_spin.setDecimals(1)
        self.regularization_spin.setToolTip(
            "Tikhonov regularization parameter as a power of 10.\n"
            "Lower values give more detailed but potentially noisier results"
        )
        reg_layout.addWidget(self.regularization_spin)
        reg_container_layout.addLayout(reg_layout)

        # Add checkbox and GCV button
        gcv_layout = QHBoxLayout()
        self.auto_gcv_checkbox = QCheckBox("Auto-GCV per frame")
        self.auto_gcv_checkbox.setToolTip(
            "Automatically optimize regularization parameter for each frame\n"
            "using Generalized Cross-Validation"
        )
        gcv_layout.addWidget(self.auto_gcv_checkbox)

        self.gcv_button = QPushButton("Auto-select (GCV)")
        self.gcv_button.setToolTip(
            "Calculate optimal regularization parameter for current frame\n"
            "using Generalized Cross-Validation"
        )
        gcv_layout.addWidget(self.gcv_button)

        reg_container_layout.addLayout(gcv_layout)
        reg_container.setLayout(reg_container_layout)
        layout.addWidget(reg_container)

        group.setLayout(layout)
        return group

    def _create_visualization_parameters_group(self) -> QGroupBox:
        """Create the visualization parameters group."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        # Vector stride
        stride_layout = QHBoxLayout()
        stride_layout.addWidget(QLabel("Vector Stride:"))
        self.visualization_params['vector_stride'] = QSpinBox()
        self.visualization_params['vector_stride'].setRange(1, 100)
        self.visualization_params['vector_stride'].setValue(20)
        self.visualization_params['vector_stride'].setToolTip(
            "Display every nth force vector.\n"
            "Higher values show fewer vectors but improve visibility"
        )
        stride_layout.addWidget(self.visualization_params['vector_stride'])
        layout.addLayout(stride_layout)

        # Arrow scale
        arrow_layout = QHBoxLayout()
        arrow_layout.addWidget(QLabel("Arrow Scale:"))
        self.visualization_params['arrow_scale'] = QDoubleSpinBox()
        self.visualization_params['arrow_scale'].setRange(0.1, 50.0)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['arrow_scale'].setToolTip(
            "Scale factor for force vector arrows.\n"
            "Adjust to make vectors more or less visible"
        )
        arrow_layout.addWidget(self.visualization_params['arrow_scale'])
        layout.addLayout(arrow_layout)

        # Maximum force
        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Force (Pa):"))
        self.visualization_params['f_max'] = QDoubleSpinBox()
        self.visualization_params['f_max'].setRange(0.1, 10000.0)
        self.visualization_params['f_max'].setValue(1000.0)
        self.visualization_params['f_max'].setToolTip(
            "Maximum force value for color scaling.\n"
            "Forces above this value will be shown at maximum intensity"
        )
        fmax_layout.addWidget(self.visualization_params['f_max'])
        layout.addLayout(fmax_layout)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        button_grid = QHBoxLayout()

        # Create left column (Preview and Save)
        left_column = QVBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.preview_btn.setToolTip(
            "Calculate and display forces for the current frame only"
        )
        self.save_force_btn = QPushButton("Save Force Data")
        self.save_force_btn.setToolTip(
            "Save calculated force data to file for later use"
        )
        self.save_force_btn.setEnabled(False)
        left_column.addWidget(self.preview_btn)
        left_column.addWidget(self.save_force_btn)

        # Create right column (Calculate and Load)
        right_column = QVBoxLayout()
        self.calculate_btn = QPushButton("Calculate Forces")
        self.calculate_btn.setToolTip(
            "Calculate forces for all frames in the dataset"
        )
        self.load_force_btn = QPushButton("Load Force Data")
        self.load_force_btn.setToolTip(
            "Load previously saved force calculation results"
        )
        right_column.addWidget(self.calculate_btn)
        right_column.addWidget(self.load_force_btn)

        button_grid.addLayout(left_column)
        button_grid.addLayout(right_column)

        layout.addLayout(button_grid)
        frame.setLayout(layout)
        return frame

    def _handle_force_results(self, force_results):
        """Handle the completed force calculation results."""
        try:
            if not isinstance(force_results, dict):
                return

            # Update data manager and visualization
            self.data_manager.force_results = force_results
            self.visualization_manager.visualize_force_results(
                force_results,
                downscale_factor=force_results['parameters'].get('downscale_factor', 1)
            )
            self._handle_visualization_layers()

            # Update colorbar
            f_max = force_results['parameters']['visualization']['f_max']
            if f_max is not None:
                self.colorbar_manager.update_limits(0, f_max)

            # Get and display statistics from last frame
            magnitude = np.sqrt(force_results['tx'][-1] ** 2 + force_results['ty'][-1] ** 2)
            stats = {
                'mean_force': np.mean(magnitude),
                'max_force': np.max(magnitude),
                'median_force': np.median(magnitude)
            }

            stats_text = (
                f"Mean force: {stats['mean_force']:.2f} Pa\n"
                f"Max force: {stats['max_force']:.2f} Pa\n"
                f"Median force: {stats['median_force']:.2f} Pa"
            )
            self._update_status(stats_text, 100)

            # Enable save button and emit results
            self.save_force_btn.setEnabled(True)
            self.force_calculated.emit(force_results)

            # Update UI state to show new data status
            self._update_ui_state()

        except Exception as e:
            self._handle_error(self.create_error(
                message="Force calculation failed",
                details=str(e),
                recovery_hint="Check input data and parameters"
            ))

    def _load_force_data(self):
        """Load force data from files."""
        try:
            # Get file path
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Force Data File",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if file_path:
                # Load the force data
                force_data = np.load(file_path, allow_pickle=True).item()

                # Convert force components to numpy arrays if they aren't already
                tx = np.array(force_data['tx'])
                ty = np.array(force_data['ty'])

                parameters = force_data['parameters']

                # Update UI parameters with loaded values
                self._load_parameters_to_ui(parameters)

                # Create results dictionary with proper parameter structure
                results = {
                    'tx': tx,
                    'ty': ty,
                    'parameters': {
                        'young_modulus': parameters['youngs_modulus'],
                        'poisson_ratio': parameters['poisson_ratio'],
                        'gel_height': None if parameters.get('gel_height') is None else parameters['gel_height'] * 1e6,
                        'pixel_size': parameters['pixelsize'],
                        'regularization': parameters['regularization'],
                        'mesh_size': self.mesh_size,
                        'lanczos_exp': parameters['lanczos_exp'],
                        'downscale_factor': parameters.get('downscale_factor', 1),
                        'visualization': {
                            'vector_stride': parameters['vector_stride'],
                            'arrow_scale': parameters['arrow_scale'],
                            'f_max': parameters['f_max']
                        }
                    }
                }

                # Update all parameters in the calculator
                self._update_parameters()

                # Update data manager and visualization
                self.data_manager.force_results = results
                self.visualization_manager.visualize_force_results(
                    results,
                    downscale_factor=parameters.get('downscale_factor', 1)
                )
                self._handle_visualization_layers()

                # Update colorbar with loaded f_max
                self.colorbar_manager.update_limits(0, parameters['f_max'])

                # Enable save button and emit results
                self.save_force_btn.setEnabled(True)
                self.force_calculated.emit(results)

                # Update UI state to show new data status
                self._update_ui_state()

                self._update_status(f"Force data successfully loaded from:\n{file_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load force data: {str(e)}"
            )
            # Print the full error for debugging
            import traceback
            traceback.print_exc()

    def _create_data_status_group(self) -> QGroupBox:
        """Create the data status group."""
        group = QGroupBox("Data Status")
        layout = QVBoxLayout()

        # Status labels stacked vertically
        self.displacement_status = QLabel("Displacement data: Not loaded")
        self.force_status = QLabel("Force results: Not loaded")

        # Add status labels
        layout.addWidget(self.displacement_status)
        layout.addWidget(self.force_status)

        # Add clear data button
        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setToolTip("Clear all loaded data and force calculation results")
        self.clear_data_btn.setStyleSheet("QPushButton { color: red; }")
        layout.addWidget(self.clear_data_btn)

        group.setLayout(layout)
        return group

    def _clear_data(self):
        """Clear all force calculation and displacement data"""
        try:
            # Clear force results
            if hasattr(self.data_manager, 'force_results'):
                self.data_manager.force_results = None

            # Clear displacement results
            if hasattr(self.data_manager, 'displacement_results'):
                self.data_manager.displacement_results = None

            # Disable save button when data is cleared
            self.save_force_btn.setEnabled(False)

            # Update UI
            self._update_ui_state()
            self._update_status("All data cleared")

        except Exception as e:
            self._handle_error(str(e))

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create colorbar container
        colorbar_container = QWidget()
        colorbar_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(6, 6, 6, 6)

        # Create colorbar
        colorbar_group = self.create_colorbar_widget(
            colormap_name='inferno',
            label="Force (Pa)",
            clim=(1000, 0),
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        colorbar_layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        colorbar_layout.addStretch()
        colorbar_container.setLayout(colorbar_layout)
        main_layout.addWidget(colorbar_container)

        # Right side container
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Add parameter groups
        right_layout.addWidget(self._create_data_status_group())  # Add the new data status group
        right_layout.addWidget(self._create_material_params_group())
        right_layout.addWidget(self._create_calculation_params_group())
        right_layout.addWidget(self._create_visualization_parameters_group())
        right_layout.addWidget(self._create_action_buttons())
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(350)

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

    def _update_parameters(self):
        """Update parameters from UI controls and data manager."""
        # Update basic parameters
        self.young_modulus = self.young_spin.value() * 1000
        self.poisson_ratio = self.poisson_spin.value()
        self.regularization = 10 ** self.regularization_spin.value()
        self.mesh_size = 1  # hardcoded to 1, artifact from older version
        self.lanczos_exp = self.lanczos_exp_spin.value()

        # Handle gel height
        height_value = self.height_spin.value()
        self.gel_height = None if height_value == 0 else height_value

        # Get pixel size and downscale_factor from data manager
        if self.data_manager.displacement_results:
            disp_params = self.data_manager.displacement_results.get('parameters', {})
            if disp_params.get('pixel_size') is not None:
                self._pixel_size = disp_params.get('pixel_size')
            if disp_params.get('downscale_factor') is not None:
                self._downscale_factor = disp_params.get('downscale_factor')

        # Update UI state
        self._update_ui_state()

    def reset_parameters(self):
        """Reset all parameters to defaults."""
        self.young_spin.setValue(10)  # 10 kPa
        self.poisson_spin.setValue(0.49)
        self.height_spin.setValue(0)
        self.mesh_size = 1  # hardcoded to 1
        self.lanczos_exp_spin.setValue(1)
        self.regularization_spin.setValue(-17)  # 10^-17
        self.auto_gcv_checkbox.setChecked(False)

        self.visualization_params['vector_stride'].setValue(20)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['f_max'].setValue(1000.0)

        self._update_status("Parameters reset to defaults")

    def _load_parameters_to_ui(self, params: dict):
        """Load parameters from dictionary to UI controls."""
        try:
            # Update spinboxes with loaded values
            if 'young_modulus' in params:
                self.young_spin.setValue(params['young_modulus'])
            if 'poisson_ratio' in params:
                self.poisson_spin.setValue(params['poisson_ratio'])
            if 'gel_height' in params:
                self.height_spin.setValue(0 if params['gel_height'] is None else params['gel_height'])
            if 'pixel_size' in params:
                self.pixel_spin.setValue(params['pixel_size'])
            if 'lanczos_exp' in params:
                self.lanczos_exp_spin.setValue(params['lanczos_exp'])
            if 'regularization' in params:
                self.regularization_spin.setValue(np.log10(params['regularization']))

            # Update visualization parameters if available
            vis_params = params.get('visualization', {})
            if vis_params:
                if 'vector_stride' in vis_params:
                    self.visualization_params['vector_stride'].setValue(vis_params['vector_stride'])
                if 'arrow_scale' in vis_params:
                    self.visualization_params['arrow_scale'].setValue(vis_params['arrow_scale'])
                if 'f_max' in vis_params:
                    self.visualization_params['f_max'].setValue(vis_params['f_max'])

            self._update_parameters()

        except Exception as e:
            self._handle_error(self.create_error(
                message="Failed to load parameters to UI",
                details=str(e),
                recovery_hint="Check parameter values and ranges"
            ))

    def _create_status_frame(self) -> QFrame:
        """Create the status and progress frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.colorbar_manager is not None:
                self.colorbar_manager.cleanup()
                self.colorbar_manager = None

            self.calculator = None
        except Exception as e:
            self._handle_error(f"Cleanup failed: {str(e)}")

        super().cleanup()

    def _initialize_calculator(self):
        """Initialize the FTTC calculator with current parameters."""
        self._update_parameters()

        # Convert gel height from μm to m if specified
        gel_height_p = None if self.gel_height is None else self.gel_height / (self.pixel_size * self._downscale_factor)

        self.calculator = FTTC(
            E=self.young_modulus,
            nu=self.poisson_ratio,
            mesh_size=self.mesh_size,
            lanczos_exp=self.lanczos_exp,
            gel_height=gel_height_p
        )

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        if not self.data_manager.displacement_results:
            return False

        if 'flows' not in self.data_manager.displacement_results:
            return False

        # Get flows data
        flows = self.data_manager.displacement_results['flows']

        # Check if flows is None or empty
        if flows is None or not isinstance(flows, (list, np.ndarray)) or len(flows) == 0:
            return False

        # Ensure proper shape
        if not isinstance(flows, np.ndarray):
            flows = np.array(flows)

        if flows.ndim < 4:  # Should be (frames, height, width, 2)
            return False

        return True

    def _save_force_data(self):
        """Save force data to files."""
        if not hasattr(self.data_manager, 'force_results') or not self.data_manager.force_results:
            QMessageBox.warning(self, "Warning", "No force data to save.")
            return

        try:
            # Get file path from user
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Force Data",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if save_path:
                results = self.data_manager.force_results

                # Safely handle gel height conversion
                gel_height = results['parameters'].get('gel_height')
                if gel_height is not None and gel_height != 0:
                    gel_height_m = gel_height * 1e-6
                else:
                    gel_height_m = None

                # Structure the data to match batch script format
                force_results = {
                    'tx': np.array(results['tx']),
                    'ty': np.array(results['ty']),
                    'parameters': {
                        'youngs_modulus': float(results['parameters']['young_modulus']),
                        'poisson_ratio': float(results['parameters']['poisson_ratio']),
                        'gel_height': gel_height_m,
                        'pixelsize': float(results['parameters']['pixel_size']),
                        'regularization': float(results['parameters']['regularization']),
                        'lanczos_exp': int(results['parameters']['lanczos_exp']),
                        'downscale_factor': int(results['parameters'].get('downscale_factor', 1)),
                        'vector_stride': int(results['parameters']['visualization']['vector_stride']),
                        'arrow_scale': float(results['parameters']['visualization']['arrow_scale']),
                        'f_max': float(results['parameters']['visualization']['f_max'])
                    }
                }

                # Save as single .npy file with all data
                np.save(save_path, force_results)

                self._update_status(f"Force data successfully saved to:\n{save_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save force data: {str(e)}"
            )

    def _handle_visualization_layers(self):
        """Handle layer visibility and ordering for better force visualization."""
        from qtpy.QtCore import QTimer

        def update_visibility():
            # Track indices and set initial visibility
            magnitude_index = None
            vectors_index = None

            for i, layer in enumerate(self.viewer.layers):
                # Hide all layers by default
                layer.visible = False

                if layer.name == 'Force Magnitude':
                    layer.visible = True  # Show magnitude by default for forces
                    magnitude_index = i
                elif layer.name == 'Force Vectors':
                    layer.visible = True
                    vectors_index = i

                # Move vectors to top if they exist
                if vectors_index is not None:
                    self.viewer.layers.move(vectors_index, -1)

                # Move magnitude above overlay but below vectors
                if magnitude_index is not None:
                    self.viewer.layers.move(magnitude_index, -2)

        # Wait a brief moment for layers to be created
        QTimer.singleShot(10, update_visibility)
