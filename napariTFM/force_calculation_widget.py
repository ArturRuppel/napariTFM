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


class ForceCalculationWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method."""

    force_calculated = Signal(dict)
    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize parameters
        self.young_modulus = 10000  # Pa
        self.poisson_ratio = 0.49
        self.gel_height = None  # μm (None means infinite)
        self._pixel_size = 0.1  # μm/pixel
        self.regularization = 1e-6
        self.filter_sigma = 2.0  # pixels
        self.mesh_size = 1  # hardcoded to 1
        self.lanczos_exp = 1
        self._using_inherited_pixel_size = False

        # Initialize calculator
        self.calculator = None
        self.colorbar_manager = ColorbarManager()
        self.visualization_params = {}

        self._setup_ui()
        self._connect_signals()
        self._register_controls()
        self._update_ui_state()

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
        self.regularization_spin.setRange(-21, 0)  # 10^-12 to 10^0
        self.regularization_spin.setValue(-17)  # Default to 10^-6
        self.regularization_spin.setSingleStep(0.5)
        self.regularization_spin.setDecimals(1)
        reg_layout.addWidget(self.regularization_spin)
        reg_container_layout.addLayout(reg_layout)

        # Add checkbox and GCV button
        gcv_layout = QHBoxLayout()
        self.auto_gcv_checkbox = QCheckBox("Auto-GCV per frame")
        self.auto_gcv_checkbox.stateChanged.connect(self._on_auto_gcv_changed)
        gcv_layout.addWidget(self.auto_gcv_checkbox)

        self.gcv_button = QPushButton("Auto-select (GCV)")
        self.gcv_button.clicked.connect(self._set_regularization_with_gcv)
        gcv_layout.addWidget(self.gcv_button)

        reg_container_layout.addLayout(gcv_layout)
        reg_container.setLayout(reg_container_layout)
        layout.addWidget(reg_container)

        # Filter sigma
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Sigma:"))
        self.filter_sigma_spin = QDoubleSpinBox()
        self.filter_sigma_spin.setRange(0, 10)
        self.filter_sigma_spin.setValue(2.0)
        self.filter_sigma_spin.setSingleStep(0.1)
        filter_layout.addWidget(self.filter_sigma_spin)
        layout.addLayout(filter_layout)

        group.setLayout(layout)
        return group

    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
                       self.young_spin,
                       self.poisson_spin,
                       self.height_spin,
                       self.pixel_spin,
                       self.lanczos_exp_spin,
                       self.regularization_spin,
                       self.filter_sigma_spin,
                       self.calculate_btn,
                       self.preview_btn,
                       self.reset_params_btn,
                       self.save_force_btn,
                       self.load_force_btn,
                       self.progress_bar,
                       self.status_label,
                       self.gcv_button,
                       self.auto_gcv_checkbox
                   ] + list(self.visualization_params.values())

        for control in controls:
            self.register_control(control)

    def _on_auto_gcv_changed(self, state):
        """Handle changes to the auto-GCV checkbox state."""
        is_checked = state == Qt.Checked
        self.gcv_button.setEnabled(not is_checked)
        self.regularization_spin.setEnabled(not is_checked)

    def reset_parameters(self):
        """Reset all parameters to defaults."""
        self._using_inherited_pixel_size = False
        self.young_spin.setValue(10000)
        self.poisson_spin.setValue(0.49)
        self.height_spin.setValue(0)
        self.pixel_spin.setValue(0.1)
        self.mesh_size = 1  # hardcoded to 1
        self.lanczos_exp_spin.setValue(1)
        self.regularization_spin.setValue(-17)  # 10^-17
        self.filter_sigma_spin.setValue(2.0)
        self.auto_gcv_checkbox.setChecked(False)

        self.visualization_params['vector_stride'].setValue(20)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['f_max'].setValue(1000.0)

        self.pixel_spin.setStyleSheet("")
        self._update_status("Parameters reset to defaults")

    def _create_material_params_group(self) -> QGroupBox:
        """Create the material parameters group."""
        group = QGroupBox("Material Parameters")
        layout = QVBoxLayout()

        # Add reset parameters button at the top
        self.reset_params_btn = QPushButton("Reset Parameters")
        layout.addWidget(self.reset_params_btn)

        # Create spinboxes
        self.young_spin = QDoubleSpinBox()
        self.poisson_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        self.pixel_spin = QDoubleSpinBox()
        self.lanczos_exp_spin = QSpinBox()

        params = [
            ("Young's Modulus (Pa):", self.young_spin, 100, 1000000, 100, self.young_modulus),
            ("Poisson Ratio:", self.poisson_spin, 0, 0.5, 0.01, self.poisson_ratio),
            ("Gel Height (μm):", self.height_spin, 0, 1000, 10, 0),
            ("Pixel Size (μm):", self.pixel_spin, 0.001, 10, 0.1, self._pixel_size),
            ("Lanczos Exponent:", self.lanczos_exp_spin, 0, 5, 1, self.lanczos_exp)
        ]

        for label_text, spin, min_val, max_val, step, default in params:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(120)
            row.addWidget(label)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if label_text.startswith("Gel Height"):
                spin.setSpecialValueText("∞")
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect all widget signals."""
        # Parameter updates
        self.young_spin.valueChanged.connect(self._update_parameters)
        self.poisson_spin.valueChanged.connect(self._update_parameters)
        self.height_spin.valueChanged.connect(self._update_parameters)
        self.pixel_spin.valueChanged.connect(self._on_pixel_size_changed)
        self.lanczos_exp_spin.valueChanged.connect(self._update_parameters)
        self.regularization_spin.valueChanged.connect(self._update_parameters)
        self.filter_sigma_spin.valueChanged.connect(self._update_parameters)

        # Action buttons
        self.calculate_btn.clicked.connect(self.calculate_forces)
        self.preview_btn.clicked.connect(self.preview_force)
        self.save_force_btn.clicked.connect(self._save_force_data)
        self.load_force_btn.clicked.connect(self._load_force_data)
        self.reset_params_btn.clicked.connect(self.reset_parameters)

    def _update_parameters(self):
        """Update parameters from UI controls."""
        # Update basic parameters
        self.young_modulus = self.young_spin.value()
        self.poisson_ratio = self.poisson_spin.value()
        self.regularization = 10 ** self.regularization_spin.value()
        self.filter_sigma = self.filter_sigma_spin.value()
        self.mesh_size = 1  # hardcoded to 1, removed from UI
        self.lanczos_exp = self.lanczos_exp_spin.value()

        # Handle gel height
        height_value = self.height_spin.value()
        self.gel_height = None if height_value == 0 else height_value

        # Handle pixel size inheritance
        if not self._using_inherited_pixel_size:
            self._pixel_size = self.pixel_spin.value()

        # Update UI state
        self._update_ui_state()

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
            if 'filter_sigma' in params:
                self.filter_sigma_spin.setValue(params['filter_sigma'])

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
        stride_layout.addWidget(self.visualization_params['vector_stride'])
        layout.addLayout(stride_layout)

        # Arrow scale
        arrow_layout = QHBoxLayout()
        arrow_layout.addWidget(QLabel("Arrow Scale:"))
        self.visualization_params['arrow_scale'] = QDoubleSpinBox()
        self.visualization_params['arrow_scale'].setRange(0.1, 50.0)
        self.visualization_params['arrow_scale'].setValue(1.0)
        arrow_layout.addWidget(self.visualization_params['arrow_scale'])
        layout.addLayout(arrow_layout)

        # Maximum force
        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Force (Pa):"))
        self.visualization_params['f_max'] = QDoubleSpinBox()
        self.visualization_params['f_max'].setRange(0.1, 10000.0)
        self.visualization_params['f_max'].setValue(1000.0)
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
        self.save_force_btn = QPushButton("Save Force Data")
        self.save_force_btn.setEnabled(False)
        left_column.addWidget(self.preview_btn)
        left_column.addWidget(self.save_force_btn)

        # Create right column (Calculate and Load)
        right_column = QVBoxLayout()
        self.calculate_btn = QPushButton("Calculate Forces")
        self.load_force_btn = QPushButton("Load Force Data")
        right_column.addWidget(self.calculate_btn)
        right_column.addWidget(self.load_force_btn)

        button_grid.addLayout(left_column)
        button_grid.addLayout(right_column)

        layout.addLayout(button_grid)
        frame.setLayout(layout)
        return frame

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
        gel_height_m = None if self.gel_height is None else self.gel_height * 1e-6

        self.calculator = FTTC(
            E=self.young_modulus,
            nu=self.poisson_ratio,
            mesh_size=self.mesh_size,
            lanczos_exp=self.lanczos_exp,
            gel_height=gel_height_m
        )

    def calculate_forces(self):
        """Calculate traction forces using the FTTC calculator."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating forces...", 0)

            # Get displacement data
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            downscale_factor = displacement_results.get('parameters', {}).get('downscale_factor', 1)
            num_frames = len(flows)

            # Initialize calculator with current parameters
            self._initialize_calculator()

            # Initialize result arrays
            force_results = {
                'tx': [],
                'ty': []
            }

            # Get spatial coordinates
            shape = flows[0].shape[:-1]  # Exclude channel dimension
            x = np.arange(shape[1])  # Width
            y = np.arange(shape[0])  # Height

            # Process each frame
            for i, flow in enumerate(flows):
                progress = (i + 1) / num_frames * 100
                self._update_status(f"Processing frame {i + 1}/{num_frames}...", progress)

                # Extract u and v components
                u_data = flow[..., 0]  # x displacement
                v_data = flow[..., 1]  # y displacement

                # If auto-GCV is enabled, calculate optimal regularization for this frame
                if self.auto_gcv_checkbox.isChecked():
                    xx, yy = np.meshgrid(x, y, indexing='ij')
                    pos0 = np.array([xx.flatten(), yy.flatten()])
                    vec0 = np.array([
                        u_data.flatten() * self._pixel_size,
                        v_data.flatten() * self._pixel_size
                    ])
                    self.regularization = self.calculator._find_regularization(pos0, vec0)
                    self._update_status(
                        f"Frame {i + 1}: Using GCV-optimized regularization {self.regularization:.2e}",
                        progress
                    )

                # Calculate forces using FTTC
                (_, _), _, f, _, _, energy, force, _, _ = self.calculator.calculate_traction(
                    x=x,
                    y=y,
                    u_data=u_data,
                    v_data=v_data,
                    dx=self._pixel_size,
                    set_lam=self.regularization
                )

                # Store force components
                force_results['tx'].append(f[0])
                force_results['ty'].append(f[1])

                # Log energy and total force for debugging
                self._update_status(
                    f"Frame {i + 1} - Energy: {energy:.2e} J, Total Force: {force:.2e} N",
                    progress
                )

            # Convert lists to arrays
            force_results['tx'] = np.stack(force_results['tx'])
            force_results['ty'] = np.stack(force_results['ty'])

            # Store calculation parameters
            visualization_params = {
                'vector_stride': self.visualization_params['vector_stride'].value(),
                'arrow_scale': self.visualization_params['arrow_scale'].value(),
                'f_max': self.visualization_params['f_max'].value()
            }

            force_results['parameters'] = {
                'young_modulus': self.young_modulus,
                'poisson_ratio': self.poisson_ratio,
                'gel_height': self.gel_height,
                'pixel_size': self._pixel_size,
                'regularization': self.regularization,
                'mesh_size': self.mesh_size,
                'lanczos_exp': self.lanczos_exp,
                'filter_sigma': self.filter_sigma,
                'downscale_factor': downscale_factor,  # Store downscale factor in parameters
                'visualization': visualization_params
            }

            # Update data manager and visualization
            self.data_manager.force_results = force_results
            self.visualization_manager.visualize_force_results(
                force_results,
                # visualization_params,
                downscale_factor=downscale_factor
            )

            # Update colorbar
            f_max = visualization_params['f_max']
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

        except Exception as e:
            self._handle_error(self.create_error(
                message="Force calculation failed",
                details=str(e),
                recovery_hint="Check input data and parameters"
            ))
        finally:
            self._set_controls_enabled(True)

    def preview_force(self):
        """Preview force calculation on current frame."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating forces...", 0)

            # Get displacement data and parameters
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            # Get downscale factor from displacement results
            downscale_factor = displacement_results.get('parameters', {}).get('downscale_factor', 1)

            # Get current frame index
            current_frame = self.viewer.dims.current_step[0]
            if current_frame >= len(flows):
                current_frame = 0

            # Get flow for current frame
            flow = flows[current_frame]

            # Extract displacement components
            u_data = flow[..., 0]  # x displacement
            v_data = flow[..., 1]  # y displacement

            x = np.arange(u_data.shape[1])
            y = np.arange(u_data.shape[0])
            dx = self._pixel_size

            # Initialize calculator with current parameters
            self._initialize_calculator()

            # Calculate forces for current frame
            xy, fnorm, f, urec, u, energy, force, Ftf, Fturec = self.calculator.calculate_traction(
                x=x,
                y=y,
                u_data=u_data,
                v_data=v_data,
                dx=dx,
                set_lam=self.regularization
            )

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
                downscale_factor=downscale_factor
            )

            # Update colorbar
            self.colorbar_manager.update_limits(0, f_max)

            # Show statistics and calculation results
            self._update_status(
                f"Preview statistics:\n"
                f"Max force: {np.max(fnorm):.2f} Pa\n"
                f"Mean force: {np.mean(fnorm):.2f} Pa\n"
                f"Median force: {np.median(fnorm):.2f} Pa\n"
                f"Energy: {energy:.2e} J\n"
                f"Total force: {force:.2e} N",
                100
            )

        except Exception as e:
            self._handle_error(self.create_error(
                message="Force preview failed",
                details=str(e),
                recovery_hint="Check input data and parameters"
            ))
        finally:
            self._set_controls_enabled(True)

    def _set_regularization_with_gcv(self):
        """Handle GCV-based regularization parameter selection."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Optimizing regularization parameter...", 0)

            # Initialize calculator if not already done
            self._initialize_calculator()

            # Get displacement data for current frame
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            current_frame = self.viewer.dims.current_step[0]
            flow = flows[current_frame]

            # Get spatial coordinates
            shape = flow.shape[:-1]  # This needs to match displacement field shape
            x = np.arange(shape[1])  # Width
            y = np.arange(shape[0])  # Height

            # Create proper position and displacement arrays for GCV
            xx, yy = np.meshgrid(x, y, indexing='ij')
            pos0 = np.array([xx.flatten(), yy.flatten()])
            vec0 = np.array([
                flow[..., 0].flatten() * self._pixel_size,  # Scale by pixel size
                flow[..., 1].flatten() * self._pixel_size
            ])

            # Calculate optimal regularization using the calculator's built-in GCV
            reg_param = self.calculator._find_regularization(pos0, vec0)

            # Update UI and calculator
            log_reg = np.log10(reg_param)
            self.regularization_spin.setValue(log_reg)
            self.regularization = reg_param

            self._update_status(
                f"Regularization parameter set to {reg_param:.2e}",
                100
            )

        except Exception as e:
            self._update_status(
                f"Error: Failed to optimize regularization parameter - {str(e)}",
                0
            )
        finally:
            self._set_controls_enabled(True)

    def _on_pixel_size_changed(self, value):
        """Handle manual changes to pixel size."""
        if not self._using_inherited_pixel_size:
            self._pixel_size = value
            self._update_parameters()

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Check if required data is available
        has_required_data = self._validate_input_data()

        # Update button states
        self.calculate_btn.setEnabled(has_required_data)
        self.preview_btn.setEnabled(has_required_data)

        # Handle pixel size inheritance if displacement results are available
        if self.data_manager.displacement_results:
            disp_params = self.data_manager.displacement_results.get('parameters', {})
            base_pixel_size = disp_params.get('pixel_size')
            downscale_factor = disp_params.get('downscale_factor', 1)

            if base_pixel_size is not None:
                # Calculate effective pixel size considering downscaling
                inherited_pixel_size = base_pixel_size * downscale_factor

                if self._using_inherited_pixel_size:
                    self._pixel_size = inherited_pixel_size
                    self.pixel_spin.setValue(inherited_pixel_size)
                    self.pixel_spin.setStyleSheet("color: gray;")
                    self.pixel_spin.setToolTip(
                        f"Inherited from displacement analysis\n"
                        f"Base pixel size: {base_pixel_size} μm\n"
                        f"Downscale factor: {downscale_factor}"
                    )
            else:
                self._using_inherited_pixel_size = False
                self.pixel_spin.setStyleSheet("")
                self.pixel_spin.setToolTip("")
        else:
            # No displacement results available
            self._using_inherited_pixel_size = False
            self.pixel_spin.setStyleSheet("")
            self.pixel_spin.setToolTip("")

            if not has_required_data:
                self._update_status("Required displacement data not available")

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

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress bar."""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all registered controls."""
        for control in self._controls:
            control.setEnabled(enabled)

    def _save_force_data(self):
        """Save force data to files."""
        if not hasattr(self.data_manager, 'force_results') or not self.data_manager.force_results:
            QMessageBox.warning(self, "Warning", "No force data to save.")
            return

        try:
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Force Data",
                os.path.expanduser("~")
            )

            if save_dir:
                results = self.data_manager.force_results

                # Save force components
                np.save(os.path.join(save_dir, 't_x.npy'), results['tx'])
                np.save(os.path.join(save_dir, 't_y.npy'), results['ty'])

                # Save parameters
                params = results.get('parameters', {})
                if params:
                    np.save(os.path.join(save_dir, 'parameters.npy'), params)

                self._update_status(f"Force data successfully saved to:\n{save_dir}", 100)


        except Exception as e:
            self._handle_error(self.create_error(
                message="Failed to save force data",
                details=str(e),
                recovery_hint="Check file permissions and disk space"
            ))

    def _load_force_data(self):
        """Load force data from files."""
        try:
            # Get directory containing the files
            load_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory Containing Force Data",
                os.path.expanduser("~")
            )

            if load_dir:
                # Check required files exist
                t_x_path = os.path.join(load_dir, 't_x.npy')
                t_y_path = os.path.join(load_dir, 't_y.npy')

                if not (os.path.exists(t_x_path) and os.path.exists(t_y_path)):
                    raise FileNotFoundError("Could not find t_x.npy and t_y.npy in selected directory")

                # Load the force components
                tx = np.load(t_x_path)
                ty = np.load(t_y_path)

                # Try to load parameters, use current if not available
                try:
                    params_path = os.path.join(load_dir, 'parameters.npy')
                    params = np.load(params_path, allow_pickle=True).item() if os.path.exists(params_path) else {}
                except:
                    params = {}

                # Get current visualization parameters from UI
                visualization_params = {
                    'vector_stride': self.visualization_params['vector_stride'].value(),
                    'arrow_scale': self.visualization_params['arrow_scale'].value(),
                    'f_max': self.visualization_params['f_max'].value()
                }

                # Create results dictionary
                results = {
                    'tx': tx,
                    'ty': ty,
                    'parameters': {
                        'young_modulus': params.get('young_modulus', self.young_modulus),
                        'poisson_ratio': params.get('poisson_ratio', self.poisson_ratio),
                        'gel_height': params.get('gel_height', self.gel_height),
                        'pixel_size': params.get('pixel_size', self._pixel_size),
                        'regularization': params.get('regularization', self.regularization),
                        'mesh_size': params.get('mesh_size', self.mesh_size),
                        'lanczos_exp': params.get('lanczos_exp', self.lanczos_exp),
                        'filter_sigma': params.get('filter_sigma', self.filter_sigma),
                        'visualization': visualization_params
                    }
                }

                # Update data manager and visualization
                self.data_manager.force_results = results
                self.visualization_manager.visualize_force_results(
                    results,
                    # visualization_params,
                    downscale_factor=1
                )

                # Update UI with loaded parameters
                if params:
                    self._load_parameters_to_ui(params)

                # Enable save button and emit results
                self.save_force_btn.setEnabled(True)
                self.force_calculated.emit(results)

                self._update_status(f"Force data successfully loaded from:\n{load_dir}", 100)


        except Exception as e:
            self._handle_error(self.create_error(
                message="Failed to load force data",
                details=str(e),
                recovery_hint="Verify file format and contents"
            ))
