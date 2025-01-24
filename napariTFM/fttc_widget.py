import os
from typing import Optional
from napari.qt.threading import thread_worker

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
from .displacement_analysis_widget import DisplacementAnalysisWidget
from .fttc import FTTC
from .parameter_manager import ParameterManager, ParameterCategory
from .visualization_manager import VisualizationManager


class FTTCWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method."""

    force_calculated = Signal(dict)

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            parameter_manager: ParameterManager,
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Store reference to parameter manager
        self.parameter_manager = parameter_manager

        # Initialize calculator and UI-related attributes
        self.calculator = None
        self.colorbar_manager = ColorbarManager()
        self.visualization_params = {}
        self.is_analysis_running = False

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()
        self._register_controls()
        self._update_ui_state()

        # Connect to parameter manager signals

        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)

        # Initialize widget with current parameter values
        self._sync_widget_with_parameters()

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        # Get displacement field from data manager
        displacement_field = self.data_manager.displacement_field
        displacement_params = self.data_manager.displacement_params

        if displacement_field is None or displacement_params is None:
            return False

        # Check if displacement field has proper shape and type
        if not isinstance(displacement_field, np.ndarray):
            return False

        # Displacement field should be 4D: (time, x, y, 2)
        if displacement_field.ndim != 4:
            return False

        return True

    def calculate_forces(self):
        """Calculate traction forces for all frames."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Starting force calculation...", 0)

            # Get displacement field from data manager
            displacement_field = self.data_manager.displacement_field
            displacement_params = self.data_manager.displacement_params

            self._initialize_calculator()

            # Get parameters from parameter manager
            regularization = self.parameter_manager.get_value('regularization')
            auto_gcv = self.parameter_manager.get_value('auto_gcv')

            # Get pixel size and scaling factors
            pixel_size = displacement_params["pixel_size"]
            downscale_factor = displacement_params.get("downscale_factor", 1)
            forcemap_pixel_size = pixel_size * downscale_factor

            # Create and configure worker for batch processing
            @thread_worker
            def process_frames():
                force_results = {
                    'tx': [],
                    'ty': []
                }

                for frame_idx, frame_data in enumerate(displacement_field):
                    # Convert displacements from micrometers to pixels
                    u_data = frame_data[..., 0] / forcemap_pixel_size
                    v_data = frame_data[..., 1] / forcemap_pixel_size

                    shape = frame_data.shape[:-1]
                    x = np.arange(shape[1])
                    y = np.arange(shape[0])
                    pos = np.array(np.meshgrid(x, y, indexing='xy'))
                    vec = np.array([u_data.flatten(), v_data.flatten()])

                    if auto_gcv:
                        lam = self.calculator._find_regularization(pos, vec)
                    else:
                        lam = regularization

                    # Calculate forces
                    (_, _), f = self.calculator._perform_tfm(pos, vec, lam)

                    # Reshape force components and convert to Pascal
                    fx = f[0].reshape(shape)
                    fy = f[1].reshape(shape)
                    forces = np.stack([fx, fy])

                    force_results['tx'].append(forces[0])
                    force_results['ty'].append(forces[1])

                    yield frame_idx, forces

                return force_results

            # Create worker and connect signals
            worker = process_frames()

            def handle_frame(result):
                try:
                    frame_idx, forces = result
                    # Calculate statistics for progress update
                    magnitude = np.sqrt(forces[0] ** 2 + forces[1] ** 2)

                    progress = (frame_idx + 1) / len(displacement_field) * 100
                    self._update_status(
                        f"Processing frame {frame_idx + 1}/{len(displacement_field)}...\n"
                        f"Mean force: {np.mean(magnitude):.2f} Pa\n"
                        f"Max force: {np.max(magnitude):.2f} Pa\n",
                        progress
                    )
                except Exception as e:
                    self._handle_error(f"Error processing frame {frame_idx}: {str(e)}")

            def handle_completion(force_results):
                try:
                    if not force_results['tx']:
                        raise ValueError("No frames were successfully processed")

                    # Convert lists to arrays
                    force_results['tx'] = np.stack(force_results['tx'])
                    force_results['ty'] = np.stack(force_results['ty'])

                    # Create force parameters dictionary
                    force_params = {
                        'young_modulus': self.parameter_manager.get_value('young_modulus'),
                        'poisson_ratio': self.parameter_manager.get_value('poisson_ratio'),
                        'gel_height': self.parameter_manager.get_value('gel_height'),
                        'pixel_size': pixel_size,
                        'frame_interval': self.parameter_manager.get_value('frame_interval'),
                        'regularization': regularization,
                        'lanczos_exp': self.parameter_manager.get_value('lanczos_exp'),
                        'downscale_factor': downscale_factor,
                        'visualization': {
                            'vector_stride': self.parameter_manager.get_value('force_vector_stride'),
                            'arrow_scale': self.parameter_manager.get_value('force_arrow_scale'),
                            'f_max': self.parameter_manager.get_value('f_max')
                        }
                    }

                    # Add parameters to results
                    force_results['parameters'] = force_params

                    # Store results in data manager
                    self.data_manager.set_force_results(
                        np.stack([force_results['tx'], force_results['ty']], axis=-1),
                        force_params
                    )

                    # Update visualization
                    self.visualization_manager.visualize_force_results(
                        force_results,
                        downscale_factor=force_params['downscale_factor']
                    )
                    self._handle_visualization_layers()

                    # Update colorbar
                    f_max = self.parameter_manager.get_value('f_max')
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
                    self._handle_error(f"Error finalizing results: {str(e)}")
                finally:
                    self._set_controls_enabled(True)

            worker.yielded.connect(handle_frame)
            worker.returned.connect(handle_completion)
            worker.errored.connect(self._handle_error)

            # Start processing
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self._set_controls_enabled(True)

    def _load_displacement(self):
        """Load displacement data from files."""
        try:
            # Get file path
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Displacement Data File",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if file_path:
                # Load data using numpy
                displacement_data = np.load(file_path, allow_pickle=True).item()

                # Validate the loaded data
                if 'flows' not in displacement_data:
                    raise ValueError("Invalid displacement data: 'flows' not found")

                flows = np.array(displacement_data['flows'])
                parameters = displacement_data.get('parameters', {})

                # Handle flow array reshaping if needed
                if len(flows.shape) == 3:  # If flows is (frames, height*2, width)
                    frames, height_doubled, width = flows.shape
                    height = height_doubled // 2
                    flows = flows.reshape(frames, 2, height, width).transpose(0, 2, 3, 1)

                # Update data manager with new format
                self.data_manager.set_displacement_results(flows, parameters)

                # Create results dictionary for visualization
                results = {
                    'flows': displacement_data['flows'],
                    'parameters': parameters,
                    'visualization_params': parameters['visualization_params'],
                    'original_shape': displacement_data['flows'].shape[1:3],
                    'flow_shape': displacement_data['flows'].shape[1:3],
                    'units': 'micrometers'
                }

                # Update visualization
                self.visualization_manager.visualize_displacement_results(
                    results,
                    downscale_factor=parameters.get('downscale_factor', 1)
                )

                self.colorbar_manager.update_limits(0, parameters.get('d_max', 10.0))

                self._update_parent_calibration(parameters['pixel_size'], parameters.get('frame_interval', 1.0))

                # Update UI state
                self._update_ui_state()

                self._update_status(f"Displacement data successfully loaded from:\n{file_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load displacement data: {str(e)}"
            )

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Check displacement field availability
        has_displacement = False
        displacement_field = getattr(self.data_manager, '_displacement_field', None)

        if displacement_field is not None and isinstance(displacement_field, np.ndarray):
            try:
                self.displacement_status.setText(f"Loaded : {displacement_field.shape}")
                has_displacement = True
            except Exception as e:
                self.displacement_status.setText(f"Error ({str(e)})")
        else:
            self.displacement_status.setText("Not loaded")

        # Update button states based on data availability and analysis state
        if self.is_analysis_running:
            self.calculate_btn.setEnabled(False)
            self.preview_btn.setEnabled(False)
            self.gcv_button.setEnabled(False)
            self.status_label.setText("Analysis in progress...")
        else:
            self.calculate_btn.setEnabled(has_displacement)
            self.preview_btn.setEnabled(has_displacement)
            self.gcv_button.setEnabled(has_displacement and not self.auto_gcv_checkbox.isChecked())

            if has_displacement:
                self.status_label.setText("Ready for force calculation")
            else:
                self.status_label.setText("Missing required displacement data")

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager values"""
        if not hasattr(self, 'parameter_manager') or self.parameter_manager is None:
            print("Warning: No parameter manager available for syncing")
            return

        # Block signals temporarily
        self._block_parameter_widgets(True)

        try:
            # Sync material parameters
            self.young_spin.setValue(self.parameter_manager.get_value('young_modulus') / 1000)  # Convert Pa to kPa
            self.poisson_spin.setValue(self.parameter_manager.get_value('poisson_ratio'))
            self.height_spin.setValue(self.parameter_manager.get_value('gel_height') or 0)
            self.lanczos_exp_spin.setValue(self.parameter_manager.get_value('lanczos_exp'))

            # Handle regularization parameter - store and display as log10
            reg_value = self.parameter_manager.get_value('regularization')
            if reg_value is not None and reg_value > 0:
                self.regularization_spin.setValue(np.log10(reg_value))
            else:
                self.regularization_spin.setValue(-4)  # Default value

            # Sync auto GCV checkbox
            auto_gcv = self.parameter_manager.get_value('auto_gcv')
            self.auto_gcv_checkbox.setChecked(bool(auto_gcv))
            self.regularization_spin.setEnabled(not auto_gcv)
            self.gcv_button.setEnabled(not auto_gcv)

            # Sync visualization parameters using the dictionary
            if hasattr(self, 'visualization_params'):
                self.visualization_params['vector_stride'].setValue(
                    self.parameter_manager.get_value('force_vector_stride'))
                self.visualization_params['arrow_scale'].setValue(
                    self.parameter_manager.get_value('force_arrow_scale'))
                self.visualization_params['f_max'].setValue(
                    self.parameter_manager.get_value('f_max'))

        except Exception as e:
            print(f"Error syncing parameters: {str(e)}")

        finally:
            # Restore signal handling
            self._block_parameter_widgets(False)

    def _on_auto_gcv_changed(self, state):
        """Handle auto GCV checkbox state changes."""
        is_checked = state == Qt.Checked
        self.regularization_spin.setEnabled(not is_checked)
        self.gcv_button.setEnabled(not is_checked)
        self.parameter_manager.set_value('auto_gcv', is_checked)

    def preview_force(self):
        """Preview force calculation on current frame."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating forces...", 0)

            # Initialize calculator
            self._initialize_calculator()

            # Get displacement field data and parameters
            displacement_field = self.data_manager.displacement_field
            displacement_params = self.data_manager.displacement_params
            current_frame = self.viewer.dims.current_step[0]
            frame_data = displacement_field[current_frame]

            # Get spatial coordinates
            shape = frame_data.shape[:-1]
            x = np.arange(shape[1])
            y = np.arange(shape[0])

            forcemap_pixel_size = displacement_params["pixel_size"] * displacement_params["downscale_factor"]
            print(forcemap_pixel_size)
            print(np.nanmean(np.abs(frame_data)))

            # Create and start worker
            worker = self.calculator.calculate_traction(
                x=x,
                y=y,
                u_data=frame_data[..., 0],
                v_data=frame_data[..., 1],
                dx=forcemap_pixel_size,
                set_lam=self.parameter_manager.get_value('regularization')
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

    def _update_parameters(self):
        """Update parameters in the parameter manager"""
        try:
            # Block signals temporarily
            self.blockSignals(True)

            # Update material parameters
            self.parameter_manager.set_value('young_modulus', self.young_spin.value() * 1000)  # Convert kPa to Pa
            self.parameter_manager.set_value('poisson_ratio', self.poisson_spin.value())
            self.parameter_manager.set_value('gel_height', None if self.height_spin.value() == 0 else self.height_spin.value())
            self.parameter_manager.set_value('lanczos_exp', self.lanczos_exp_spin.value())

            # Update regularization parameters - ensure we never set None or invalid values
            reg_value = 10 ** self.regularization_spin.value()
            if reg_value <= 0:
                reg_value = 1e-4  # Set minimum value
            self.parameter_manager.set_value('regularization', reg_value)
            self.parameter_manager.set_value('auto_gcv', self.auto_gcv_checkbox.isChecked())

            # Update visualization parameters
            self.parameter_manager.set_value('force_vector_stride',
                                             self.visualization_params['vector_stride'].value())
            self.parameter_manager.set_value('force_arrow_scale',
                                             self.visualization_params['arrow_scale'].value())
            self.parameter_manager.set_value('f_max',
                                             self.visualization_params['f_max'].value())

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.blockSignals(False)

    def _on_parameter_changed(self, param_name: str, value: object):
        """Handle parameter changes from the parameter manager"""
        # Only update if the change didn't come from this widget
        if not self.signalsBlocked():
            self._sync_widget_with_parameters()

    def _block_parameter_widgets(self, block: bool):
        """Block or unblock signals for all parameter-related widgets"""
        widgets = [
            self.young_spin,
            self.poisson_spin,
            self.height_spin,
            self.lanczos_exp_spin,
            self.regularization_spin,
            self.auto_gcv_checkbox,
            *self.visualization_params.values()
        ]
        for widget in widgets:
            widget.blockSignals(block)

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
            flows = self.data_manager.displacement_field
            current_frame = self.viewer.dims.current_step[0]
            flow = flows[current_frame]



            # Get pixel size and downscale factor from displacement results
            disp_params = self.data_manager.displacement_params
            pixel_size = disp_params.get('pixel_size')
            downscale_factor = disp_params.get('downscale_factor', 1)

            # Get spatial coordinates and prepare data
            shape = flow.shape[:-1]
            pos = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy'))
            vec = np.array([flow[..., 0], flow[..., 1]]) / (pixel_size * downscale_factor) # convert to pixel

            # Calculate optimal regularization parameter
            lam = self.calculator._find_regularization(pos, vec)

            # Update UI and parameter manager with new value (log scale)
            self.regularization_spin.setValue(np.log10(lam))
            self.parameter_manager.set_value('regularization', lam)

            self._set_controls_enabled(True)
            self._update_status(f"Optimal regularization parameter: {lam:.2e}", 100)

        except Exception as e:
            self._handle_error(str(e))
            self._set_controls_enabled(True)

    def _initialize_calculator(self):
        """Initialize the FTTC calculator with current parameters."""
        try:
            # Get parameters from parameter manager
            young_modulus = self.parameter_manager.get_value('young_modulus')
            poisson_ratio = self.parameter_manager.get_value('poisson_ratio')
            gel_height = self.parameter_manager.get_value('gel_height')
            lanczos_exp = self.parameter_manager.get_value('lanczos_exp')

            # Get pixel size and downscale factor from displacement parameters
            disp_params = self.data_manager.displacement_params
            pixel_size = disp_params.get('pixel_size')
            downscale_factor = disp_params.get('downscale_factor', 1)

            # Convert gel height from μm to pixels if specified
            gel_height_p = None if gel_height is None else gel_height / (pixel_size * downscale_factor)

            self.calculator = FTTC(
                E=young_modulus,
                pixelsize=pixel_size * downscale_factor * 1e-6,
                nu=poisson_ratio,
                lanczos_exp=lanczos_exp,
                gel_height=gel_height_p
            )

        except Exception as e:
            self._handle_error(f"Error initializing calculator: {str(e)}")

    def reset_parameters(self):
        """Reset force-specific parameters to defaults."""
        try:
            # Reset only force parameters
            self.parameter_manager.reset_category_to_defaults(ParameterCategory.FORCE)

            # Synchronize widget values with reset parameters
            self._sync_widget_with_parameters()

            # Reinitialize calculator with new parameters
            self._initialize_calculator()

            self._update_status("Force parameters reset to defaults")

        except Exception as e:
            self._handle_error(f"Error resetting parameters: {str(e)}")

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

        # Add widgets to right container
        right_layout.addWidget(self._create_data_loading_group())
        right_layout.addWidget(self._create_parameters_group())
        right_layout.addWidget(self._create_action_buttons())
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(360)

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)
        self._register_controls()

    def _create_parameters_group(self) -> QGroupBox:
        """Create a consolidated parameters group."""
        # Create all spin boxes first
        self.young_spin = QDoubleSpinBox()
        self.poisson_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        self.lanczos_exp_spin = QSpinBox()
        self.regularization_spin = QDoubleSpinBox()
        self.auto_gcv_checkbox = QCheckBox("Auto-GCV per frame")
        self.gcv_button = QPushButton("Auto-select (GCV)")

        # Initialize visualization parameters
        if not hasattr(self, 'visualization_params'):
            self.visualization_params = {}
        if 'vector_stride' not in self.visualization_params:
            self.visualization_params['vector_stride'] = QSpinBox()
        if 'arrow_scale' not in self.visualization_params:
            self.visualization_params['arrow_scale'] = QDoubleSpinBox()
        if 'f_max' not in self.visualization_params:
            self.visualization_params['f_max'] = QSpinBox()

        # Main parameters group
        group = QGroupBox("Parameters")
        main_layout = QVBoxLayout()

        # Create sections for better organization
        material_params_group = QGroupBox("Material Parameters")
        material_params_layout = QVBoxLayout()
        regularization_params_group = QGroupBox("Regularization Parameters")
        regularization_params_layout = QVBoxLayout()
        vis_params_group = QGroupBox("Visualization Parameters")
        vis_params_layout = QVBoxLayout()

        # Material parameters setup
        params = [
            ("Young's Modulus (kPa):", self.young_spin, 0.1, 1000, 0.1, 10,
             "Elastic modulus of the gel substrate in kilopascals (kPa)"),
            ("Poisson Ratio:", self.poisson_spin, 0, 0.5, 0.01, 0.49,
             "Poisson's ratio of the gel substrate (typically 0.45-0.49 for hydrogels)"),
            ("Gel Height (μm):", self.height_spin, 0, 1000, 10, 0,
             "Thickness of the gel substrate in micrometers. Set to 0 for infinite thickness"),
            ("Lanczos Exponent:", self.lanczos_exp_spin, 0, 5, 1, 1,
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
            material_params_layout.addLayout(row)

        material_params_group.setLayout(material_params_layout)

        # Regularization parameters
        regularization_container = QWidget()
        regularization_container_layout = QVBoxLayout()

        # Regularization value control
        reg_layout = QHBoxLayout()
        reg_layout.addWidget(QLabel("Parameter (10^x):"))
        self.regularization_spin.setRange(-21, 0)
        self.regularization_spin.setValue(-4)
        self.regularization_spin.setSingleStep(0.5)
        self.regularization_spin.setDecimals(1)
        self.regularization_spin.setToolTip(
            "Tikhonov regularization parameter as a power of 10.\n"
            "Lower values give more detailed but potentially noisier results"
        )
        reg_layout.addWidget(self.regularization_spin)
        regularization_container_layout.addLayout(reg_layout)

        # GCV controls
        gcv_layout = QHBoxLayout()
        self.auto_gcv_checkbox.setToolTip(
            "Automatically optimize regularization parameter for each frame\n"
            "using Generalized Cross-Validation"
        )
        gcv_layout.addWidget(self.auto_gcv_checkbox)

        self.gcv_button.setToolTip(
            "Calculate optimal regularization parameter for current frame\n"
            "using Generalized Cross-Validation"
        )
        gcv_layout.addWidget(self.gcv_button)
        regularization_container_layout.addLayout(gcv_layout)

        regularization_container.setLayout(regularization_container_layout)
        regularization_params_layout.addWidget(regularization_container)
        regularization_params_group.setLayout(regularization_params_layout)

        # Visualization parameters
        # Vector stride
        stride_layout = QHBoxLayout()
        stride_layout.addWidget(QLabel("Vector Stride:"))
        self.visualization_params['vector_stride'].setRange(1, 100)
        self.visualization_params['vector_stride'].setValue(20)
        self.visualization_params['vector_stride'].setToolTip(
            "Display every nth vector in the visualization.\n"
            "Higher values show fewer vectors but improve clarity"
        )
        stride_layout.addWidget(self.visualization_params['vector_stride'])
        vis_params_layout.addLayout(stride_layout)

        # Arrow scale
        arrow_layout = QHBoxLayout()
        arrow_layout.addWidget(QLabel("Arrow Scale:"))
        self.visualization_params['arrow_scale'].setRange(0.1, 50.0)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['arrow_scale'].setToolTip(
            "Scale factor for force vector arrows.\n"
            "Adjust to make vectors more or less visible"
        )
        arrow_layout.addWidget(self.visualization_params['arrow_scale'])
        vis_params_layout.addLayout(arrow_layout)

        # Maximum force
        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Force (Pa):"))
        self.visualization_params['f_max'].setRange(0.1, 10000.0)
        self.visualization_params['f_max'].setValue(1000.0)
        self.visualization_params['f_max'].setSingleStep(1)

        self.visualization_params['f_max'].setToolTip(
            "Maximum force value for color scaling.\n"
            "Forces above this value will be shown at maximum intensity"
        )
        fmax_layout.addWidget(self.visualization_params['f_max'])
        vis_params_layout.addLayout(fmax_layout)

        vis_params_group.setLayout(vis_params_layout)

        # Add all parameter groups to main layout
        main_layout.addWidget(material_params_group)
        main_layout.addWidget(regularization_params_group)
        main_layout.addWidget(vis_params_group)

        # Add reset parameters button at the bottom
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.reset_params_btn.setToolTip("Reset all parameters to their default values")
        main_layout.addWidget(self.reset_params_btn)

        main_layout.addStretch()
        group.setLayout(main_layout)
        return group

    def _connect_signals(self):
        """Connect all widget signals."""
        # Keep existing connections
        self.young_spin.valueChanged.connect(self._update_parameters)
        self.poisson_spin.valueChanged.connect(self._update_parameters)
        self.height_spin.valueChanged.connect(self._update_parameters)
        self.lanczos_exp_spin.valueChanged.connect(self._update_parameters)
        self.regularization_spin.valueChanged.connect(self._update_parameters)
        self.load_displacement_btn.clicked.connect(self._load_displacement)
        self.auto_gcv_checkbox.stateChanged.connect(self._on_auto_gcv_changed)

        # Action buttons
        self.calculate_btn.clicked.connect(self.calculate_forces)
        self.preview_btn.clicked.connect(self.preview_force)
        self.save_force_btn.clicked.connect(self._save_force_data)
        self.load_force_btn.clicked.connect(self._load_force_data)
        self.reset_params_btn.clicked.connect(self.reset_parameters)

        # Add GCV control connections
        self.gcv_button.clicked.connect(self._auto_select_gcv)
        self.auto_gcv_checkbox.stateChanged.connect(self._toggle_auto_gcv)

        # Connect visualization parameter changes
        self.visualization_params['vector_stride'].valueChanged.connect(self._update_parameters)
        self.visualization_params['arrow_scale'].valueChanged.connect(self._update_parameters)
        self.visualization_params['f_max'].valueChanged.connect(self._update_parameters)

    def _toggle_auto_gcv(self, state):
        """Enable or disable automatic GCV calculation per frame."""
        self.regularization_spin.setEnabled(not state)
        self.gcv_button.setEnabled(not state)

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

    def _handle_preview_results(self, results):
        """Handle the preview calculation results."""
        try:
            # Unpack results
            (_, _), f = results  # in N / px²

            displacement_params = self.data_manager.displacement_params
            forcemap_pixel_size = displacement_params["pixel_size"] * displacement_params["downscale_factor"]

            # Get visualization parameters
            vector_stride = self.visualization_params['vector_stride'].value()
            arrow_scale = self.visualization_params['arrow_scale'].value()
            f_max = self.visualization_params['f_max'].value()

            # Get downscale factor from displacement parameters
            downscale_factor = 1  # default value
            if hasattr(self.data_manager, '_displacement_params'):
                disp_params = self.data_manager.displacement_params
                if disp_params is not None:
                    downscale_factor = disp_params.get('downscale_factor', 1)

            # Update visualization
            self.visualization_manager.visualize_force_preview(
                f[0], f[1],
                f_max=f_max,
                vector_stride=vector_stride,
                arrow_scale=arrow_scale,
                downscale_factor=downscale_factor
            )

            # Handle layer visibility and ordering
            self._handle_visualization_layers()

            # Update colorbar
            self.colorbar_manager.update_limits(0, f_max)

            # Calculate and show statistics
            magnitude = np.sqrt(f[0] ** 2 + f[1] ** 2)

            self._update_status(
                f"Preview statistics:\n"
                f"Max force: {np.max(magnitude):.2f} Pa\n"
                f"Mean force: {np.mean(magnitude):.2f} Pa\n"
                f"Median force: {np.median(magnitude):.2f} Pa\n",
                100
            )

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
                       self.progress_bar,
                       self.status_label,
                       self.displacement_status,
                       self.gcv_button,
                       self.auto_gcv_checkbox
                   ] + list(self.visualization_params.values())

        for control in controls:
            self.register_control(control)

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
        self.save_force_btn = QPushButton("Save Traction Forces")
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

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data status group."""
        group = QGroupBox("Input Data")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Initialize button and status label for displacement data
        self.load_displacement_btn = QPushButton("Load Displacements")
        self.load_displacement_btn.setToolTip("Load displacement data from file")
        self.displacement_status = QLabel("Not loaded")

        # Add button and status label in a row
        displacement_layout = QHBoxLayout()
        displacement_layout.addWidget(self.load_displacement_btn)
        displacement_layout.addWidget(self.displacement_status)
        layout.addLayout(displacement_layout)

        group.setLayout(layout)
        return group

    def _load_parameters_to_ui(self, params: dict):
        """Load parameters from dictionary to UI controls."""
        try:
            # Update parent widget calibration values
            if 'pixel_size' in params and 'frame_interval' in params:
                self._update_parent_calibration(params['pixel_size'], params['frame_interval'])

            # Update spinboxes with loaded values
            if 'young_modulus' in params:
                self.young_spin.setValue(params['young_modulus'] / 1000) # convert from Pa to kPa
            if 'poisson_ratio' in params:
                self.poisson_spin.setValue(params['poisson_ratio'])
            if 'gel_height' in params:
                self.height_spin.setValue(0 if params['gel_height'] is None else params['gel_height'])
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

    def _save_force_data(self):
        """Save force data to files."""
        if not hasattr(self.data_manager, '_force_field'):
            QMessageBox.warning(self, "Warning", "No force data to save.")
            return

        try:
            # Get file path from user
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Traction Forces",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if save_path:
                force_field = self.data_manager.force_field
                params = self.data_manager.force_params

                force_results = {
                    'force_field': force_field,
                    'parameters': params
                }

                # Save as single .npy file with all data
                np.save(save_path, force_results)

                self._update_status(f"Force data successfully saved to:\n{save_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save traction force data: {str(e)}"
            )

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

                # Update data manager and visualization
                self.data_manager.set_force_results(force_data['force_field'], force_data['parameters'])

                # Convert force components to numpy arrays if they aren't already
                tx = np.array(force_data['force_field'][..., 0])
                ty = np.array(force_data['force_field'][..., 1])

                parameters = force_data['parameters']

                # Update UI parameters with loaded values
                self._load_parameters_to_ui(parameters)

                # Create results dictionary with proper parameter structure for visualizer
                results = {
                    'tx': tx,
                    'ty': ty,
                    'parameters': parameters
                }

                # Update all parameters in the calculator
                self._update_parameters()

                # Update visualization
                self.visualization_manager.visualize_force_results(
                    results,
                    downscale_factor=parameters.get('downscale_factor', 1)
                )
                self._handle_visualization_layers()

                # Update colorbar with loaded f_max
                f_max = parameters.get('visualization', {}).get('f_max')
                self.colorbar_manager.update_limits(0, f_max)



                # Enable save button and emit results
                self.save_force_btn.setEnabled(True)
                self.force_calculated.emit(results)

                # Update UI state to show new data status
                self._update_ui_state()

                self._update_status(
                    f"Force data successfully loaded from:\n"
                    f"{file_path}\n"
                    f"Pixel size: {parameters['pixel_size']} µm\n"
                    f"Frame interval: {parameters.get('frame_interval', 1.0)} min",
                    100
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load force data: {str(e)}"
            )
            import traceback
            traceback.print_exc()
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

    def _update_parent_calibration(self, pixel_size: float, frame_interval: float):
        """Update calibration values in parent widget."""
        try:
            # Find parent widget instance
            parent = self
            while parent is not None:
                if hasattr(parent, 'pixel_spin') and hasattr(parent, 'frame_spin'):
                    break
                parent = parent.parent()

            if parent is not None:
                # Update calibration values
                parent.pixel_spin.setValue(pixel_size)
                parent.frame_spin.setValue(frame_interval)
            else:
                self._update_status("Warning: Could not update calibration in parent widget", 100)

        except Exception as e:
            self._handle_error(f"Failed to update calibration: {str(e)}")
