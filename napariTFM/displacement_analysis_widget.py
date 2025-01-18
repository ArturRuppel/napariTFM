import os

import napari
import numpy as np
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame,
    QProgressBar, QMessageBox, QSizePolicy, QFileDialog
)

from .base_widget import BaseAnalysisWidget
from .colorbar import ColorbarManager
from .data_manager import DataManager
from .displacement_analysis import DisplacementAnalyzer, TVL1Parameters
from .visualization_manager import VisualizationManager


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements using TV-L1 optical flow."""

    displacement_calculated = Signal(dict)  # Emits displacement results

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager",
                 visualization_manager: "VisualizationManager"):
        super().__init__(viewer, data_manager, visualization_manager)

        self.analyzer = DisplacementAnalyzer()
        self.colorbar_manager = ColorbarManager()
        self.current_flow = None
        self.parameter_spins = {}
        self.visualization_params = {}

        # Store default parameter values (removed pixel_size)
        self.default_parameters = {
            'tau': 0.25,
            'lambda_': 0.4,
            'theta': 0.3,
            'nscales': 3,
            'warps': 3,
            'epsilon': 0.01,
            'inner_iterations': 15,
            'outer_iterations': 5,
            'scale_step': 0.5,
            'median_filtering': 5,
            'downscale_factor': 1,
            'vector_stride': 20,
            'arrow_scale': 1.0,
            'd_max': 5.0
        }

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def analyze_all_frames(self):
        """Analyze displacement for all frames using a thread worker."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Starting analysis...", 0)

            # Get parameters
            vis_params = {
                'd_max': self.visualization_params['d_max'].value(),
                'vector_stride': self.visualization_params['vector_stride'].value(),
                'arrow_scale': self.visualization_params['arrow_scale'].value()
            }
            downscale_factor = self.parameter_spins['downscale_factor'].value()
            pixel_size = self.pixel_size

            # Get input data
            reference = self.data_manager.displacement_reference_image
            bead_stack = self.data_manager.displacement_bead_stack

            # Create and start worker
            worker = self.analyzer.analyze_displacement(
                reference=reference,
                bead_stack=bead_stack,
                pixel_size=pixel_size,
                downscale_factor=downscale_factor,
                visualization_params=vis_params
            )

            # Connect worker signals
            worker.yielded.connect(self._handle_progress)
            worker.returned.connect(self._handle_displacement_results)
            worker.finished.connect(lambda: self._set_controls_enabled(True))
            worker.errored.connect(self._handle_error)

            # Start the worker
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self._set_controls_enabled(True)

    def _handle_progress(self, update_dict):
        """Handle progress updates from the worker."""
        progress = update_dict['progress']
        message = update_dict['message']
        self._update_status(message, int(progress))

    def _handle_displacement_results(self, results):
        """Handle the completed displacement analysis results."""
        try:
            # Update data manager
            self.data_manager.displacement_results = results

            # Update visualization
            self.visualization_manager.visualize_displacement_results(
                results,
                downscale_factor=results['parameters']['downscale_factor']
            )

            # Handle layer visibility and ordering
            self._handle_visualization_layers()

            # Update colorbar with current d_max
            d_max = results['visualization_params']['d_max']
            self.colorbar_manager.update_limits(0, d_max)

            # Enable save button and emit results
            self.save_displacement_btn.setEnabled(True)
            self.displacement_calculated.emit(results)

            # Update status with statistics
            stats = self.visualization_manager.get_displacement_statistics(results['flows'][0])
            self._update_status(
                f"Analysis complete\n"
                f"Max displacement: {stats['max']:.2f} µm\n"
                f"Mean displacement: {stats['mean']:.2f} µm\n"
                f"Flow field resolution: {results['flow_shape']} (from {results['original_shape']})",
                100
            )

        except Exception as e:
            self._handle_error(str(e))

    def _clear_data(self):
        """Clear all displacement analysis data"""
        try:
            # Clear data from manager
            self.data_manager.clear_displacement_data()

            # Disable save button when data is cleared
            self.save_displacement_btn.setEnabled(False)

            # Update UI
            self._update_ui_state()
            self._update_status("All displacement analysis data cleared")

        except Exception as e:
            self._handle_error(str(e))

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        # Create 2x2 grid for buttons
        button_grid = QHBoxLayout()

        # Create left column (Preview and Save)
        left_column = QVBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.preview_btn.setToolTip("Calculate and visualize displacement for the current frame")

        self.save_displacement_btn = QPushButton("Save Displacement")
        self.save_displacement_btn.setToolTip("Save the current displacement analysis results to a file")
        # Initialize save button as disabled
        self.save_displacement_btn.setEnabled(False)

        left_column.addWidget(self.preview_btn)
        left_column.addWidget(self.save_displacement_btn)

        # Create right column (Analyze and Load)
        right_column = QVBoxLayout()
        self.analyze_btn = QPushButton("Analyze All Frames")
        self.analyze_btn.setToolTip("Calculate displacement for all frames in the sequence")

        self.load_displacement_btn = QPushButton("Load Displacement")
        self.load_displacement_btn.setToolTip("Load previously saved displacement analysis results")

        right_column.addWidget(self.analyze_btn)
        right_column.addWidget(self.load_displacement_btn)

        # Add columns to grid
        button_grid.addLayout(left_column)
        button_grid.addLayout(right_column)

        layout.addLayout(button_grid)
        frame.setLayout(layout)
        return frame

    def _handle_visualization_layers(self):
        """Handle layer visibility and ordering for better data visualization."""
        from qtpy.QtCore import QTimer

        def update_visibility():
            # Then update layer visibility and order
            magnitude_index = None
            vectors_index = None

            # First pass: collect indices and set visibility
            for i, layer in enumerate(self.viewer.layers):
                # Hide all layers by default
                layer.visible = False

                # Keep track of indices and set visibility for our layers of interest
                if layer.name == 'Displacement Magnitude':
                    layer.visible = True
                    magnitude_index = i
                elif layer.name == 'Displacement Vectors':
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

    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating displacement...", 0)

            # Get reference and bead data
            reference = self.data_manager.displacement_reference_image
            bead_stack = self.data_manager.displacement_bead_stack

            # Get current frame, handling both 2D and 3D cases
            if bead_stack.ndim == 2:
                moving = bead_stack  # Single frame case
            else:
                # Make sure we're using a valid frame index
                current_frame = min(self.viewer.dims.current_step[0], bead_stack.shape[0] - 1)
                moving = bead_stack[current_frame]

            # Calculate initial flow in pixels
            flow_pixels = self.analyzer.calculate_flow(reference, moving)

            # Apply downscaling if factor > 1
            downscale_factor = self.parameter_spins['downscale_factor'].value()
            if downscale_factor > 1:
                flow_pixels = self.analyzer.downscale_flow(flow_pixels, downscale_factor)

            # Convert flow to micrometers using pixel size
            pixel_size = self.pixel_size
            self.current_flow = flow_pixels * pixel_size

            # Get visualization parameters
            vis_params = {
                'd_max': self.visualization_params['d_max'].value(),
                'vector_stride': self.visualization_params['vector_stride'].value(),
                'arrow_scale': self.visualization_params['arrow_scale'].value()
            }

            # Update visualization
            self.visualization_manager.visualize_displacement_preview(
                self.current_flow,
                vis_params['d_max'],
                vis_params['vector_stride'],
                vis_params['arrow_scale'],
                downscale_factor=downscale_factor
            )

            # Handle layer visibility and ordering
            self._handle_visualization_layers()

            # Update colorbar
            self.colorbar_manager.update_limits(0, vis_params['d_max'])

            # Update status with statistics
            stats = self.visualization_manager.get_displacement_statistics(self.current_flow)
            original_shape = reference.shape
            downscaled_shape = self.current_flow.shape[:2]
            self._update_status(
                f"Max displacement: {stats['max']:.2f} µm\n"
                f"Mean displacement: {stats['mean']:.2f} µm\n"
                f"Flow field resolution: {downscaled_shape} \n(from {original_shape})",
                100
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Add reset parameters button at the top
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.reset_params_btn.setToolTip("Reset all parameters to their default values")
        layout.addWidget(self.reset_params_btn)

        # Create sections for better organization
        flow_params_group = QGroupBox("Optical Flow Parameters")
        flow_params_layout = QVBoxLayout()

        scaling_group = QGroupBox("Scaling Parameters")
        scaling_layout = QVBoxLayout()

        # Define core optical flow parameters with tooltips
        flow_params = [
            ("tau", "Tau:", 0.01, 1.0, 0.01, 0.25,
             "Time step for optical flow computation. Lower values give more accurate but slower results"),
            ("lambda_", "Lambda:", 0.01, 1.0, 0.01, 0.4,
             "Regularization parameter. Higher values produce smoother flow fields"),
            ("theta", "Theta:", 0.1, 1.0, 0.1, 0.3,
             "Weight parameter for the divergence term. Controls flow field smoothness"),
            ("nscales", "Pyramid Scales:", 1, 10, 1, 3,
             "Number of pyramid levels. More levels handle larger displacements but increase computation time"),
            ("warps", "Warps:", 1, 10, 1, 3,
             "Number of warping steps per scale. More warps increase accuracy for large displacements"),
            ("epsilon", "Epsilon:", 0.001, 0.1, 0.001, 0.01,
             "Stopping criterion threshold. Lower values give more precise results but longer computation times"),
            ("inner_iterations", "Inner Iterations:", 1, 50, 1, 15,
             "Maximum number of inner iterations. More iterations improve accuracy but increase computation time"),
            ("outer_iterations", "Outer Iterations:", 1, 20, 1, 5,
             "Maximum number of outer iterations. More iterations improve accuracy but increase computation time"),
            ("scale_step", "Scale Step:", 0.1, 0.99, 0.01, 0.5,
             "Scale factor between pyramid levels. Lower values create more pyramid levels"),
            ("median_filtering", "Median Filter Size:", 1, 9, 2, 5,
             "Size of median filter for post-processing. Larger values remove more noise but may lose detail"),
        ]

        # Add optical flow parameters with tooltips
        for param_name, label, min_val, max_val, step, default, tooltip in flow_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            spin = QDoubleSpinBox() if isinstance(default, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setToolTip(tooltip)

            self.parameter_spins[param_name] = spin
            row.addWidget(spin)
            flow_params_layout.addLayout(row)

        # Add downscale factor to scaling parameters
        downscale_row = QHBoxLayout()
        downscale_row.addWidget(QLabel("Local Averaging Factor:"))
        downscale_spin = QSpinBox()
        downscale_spin.setRange(1, 10)
        downscale_spin.setSingleStep(1)
        downscale_spin.setValue(1)
        downscale_spin.setToolTip(
            "Factor for spatial averaging of displacement field.\n"
            "1 = no averaging (full resolution)\n"
            "Higher values reduce resolution but improve signal-to-noise ratio\n"
            "and reduce computational cost for subsequent steps."
        )
        self.parameter_spins['downscale_factor'] = downscale_spin
        downscale_row.addWidget(downscale_spin)
        scaling_layout.addLayout(downscale_row)

        flow_params_group.setLayout(flow_params_layout)
        scaling_group.setLayout(scaling_layout)

        layout.addWidget(flow_params_group)
        layout.addWidget(scaling_group)
        layout.addStretch()

        group.setLayout(layout)
        return group

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group."""
        group = QGroupBox("Data")
        layout = QVBoxLayout()

        # Load buttons and status labels
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.load_beads_btn.setToolTip("Load the time series of bead images for displacement analysis")

        self.load_reference_btn = QPushButton("Load Reference Image")
        self.load_reference_btn.setToolTip("Load the reference (initial) bead image")

        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setToolTip("Clear all loaded data and analysis results")
        self.clear_data_btn.setStyleSheet("QPushButton { color: red; }")

        self.bead_status = QLabel("Not loaded")
        self.reference_status = QLabel("Not loaded")

        # Add with status labels
        for btn, status in [
            (self.load_beads_btn, self.bead_status),
            (self.load_reference_btn, self.reference_status),
        ]:
            row = QHBoxLayout()
            row.addWidget(btn)
            row.addWidget(status)
            layout.addLayout(row)

        # Add clear button
        layout.addWidget(self.clear_data_btn)

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
            "Display every nth vector in the visualization. Higher values show fewer vectors but improve clarity"
        )
        stride_layout.addWidget(self.visualization_params['vector_stride'])
        layout.addLayout(stride_layout)

        # Arrow scale
        arrow_layout = QHBoxLayout()
        arrow_layout.addWidget(QLabel("Arrow Scale:"))
        self.visualization_params['arrow_scale'] = QDoubleSpinBox()
        self.visualization_params['arrow_scale'].setRange(0.1, 2.0)
        self.visualization_params['arrow_scale'].setSingleStep(0.1)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['arrow_scale'].setToolTip(
            "Scale factor for arrow length in the visualization. Adjust to make displacement vectors more visible"
        )
        arrow_layout.addWidget(self.visualization_params['arrow_scale'])
        layout.addLayout(arrow_layout)

        # Maximum displacement (now in µm)
        dmax_layout = QHBoxLayout()
        dmax_layout.addWidget(QLabel("Max Displacement (µm):"))
        self.visualization_params['d_max'] = QDoubleSpinBox()
        self.visualization_params['d_max'].setRange(0.1, 200.0)
        self.visualization_params['d_max'].setSingleStep(1.0)
        self.visualization_params['d_max'].setValue(5.0)
        self.visualization_params['d_max'].setToolTip(
            "Maximum displacement value for color scaling (in µm). Adjust to optimize the color range of the visualization"
        )
        dmax_layout.addWidget(self.visualization_params['d_max'])
        layout.addLayout(dmax_layout)

        group.setLayout(layout)
        return group

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create colorbar widget
        colorbar_container = QWidget()
        colorbar_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(6, 6, 6, 6)

        colorbar_group = self.create_colorbar_widget(
            colormap_name='viridis',
            label="Displacement (µm)",
            clim=(10, 0),
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        colorbar_layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        colorbar_layout.addStretch()
        colorbar_container.setLayout(colorbar_layout)

        main_layout.addWidget(colorbar_container)

        # Container for right side content
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        right_layout.addWidget(self._create_data_loading_group())
        right_layout.addWidget(self._create_parameters_group())
        right_layout.addWidget(self._create_visualization_parameters_group())
        right_layout.addWidget(self._create_action_buttons())
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(350)  # Compromised width between 300 and 400

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)
        self._register_controls()

    def _connect_signals(self):
        """Connect all widget signals."""
        # Existing connections
        self.load_beads_btn.clicked.connect(lambda: self._load_data('beads'))
        self.load_reference_btn.clicked.connect(lambda: self._load_data('reference'))
        self.clear_data_btn.clicked.connect(self._clear_data)
        self.preview_btn.clicked.connect(self.preview_displacement)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)

        # Parameter change connections
        for spin in self.parameter_spins.values():
            spin.valueChanged.connect(self.update_parameters)

        # New button connections
        self.save_displacement_btn.clicked.connect(self._save_displacement)
        self.load_displacement_btn.clicked.connect(self._load_displacement)
        self.reset_params_btn.clicked.connect(self._reset_parameters)

    def _reset_parameters(self):
        """Reset all parameters to their default values."""
        for param_name, default_value in self.default_parameters.items():
            if param_name in self.parameter_spins:
                self.parameter_spins[param_name].setValue(default_value)
            elif param_name in self.visualization_params:
                self.visualization_params[param_name].setValue(default_value)

        self.update_parameters()
        self._update_status("Parameters have been reset to default values")

    def _save_displacement(self):
        """Save displacement data to files."""
        if not hasattr(self.data_manager, 'displacement_results') or not self.data_manager.displacement_results:
            QMessageBox.warning(self, "Warning", "No displacement data to save.")
            return

        try:
            # Get file path
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Displacement Data",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if save_path:
                # Get current parameters from UI
                tvl1_params = {
                    'tau': self.parameter_spins['tau'].value(),
                    'lambda': self.parameter_spins['lambda_'].value(),
                    'theta': self.parameter_spins['theta'].value(),
                    'nscales': self.parameter_spins['nscales'].value(),
                    'warps': self.parameter_spins['warps'].value(),
                    'epsilon': self.parameter_spins['epsilon'].value(),
                    'inner_iterations': self.parameter_spins['inner_iterations'].value(),
                    'outer_iterations': self.parameter_spins['outer_iterations'].value(),
                    'scale_step': self.parameter_spins['scale_step'].value(),
                    'median_filtering': self.parameter_spins['median_filtering'].value()
                }

                results = self.data_manager.displacement_results
                flows = np.array(results['flows'])

                # Package everything into a single dictionary
                displacement_data = {
                    'flows': flows,
                    'parameters': {
                        'tvl1_params': tvl1_params,
                        'pixel_size': self.pixel_size,
                        'frame_interval': self.frame_length,  # Added frame interval
                        'downscale_factor': self.parameter_spins['downscale_factor'].value(),
                        'arrow_scale': self.visualization_params['arrow_scale'].value(),
                        'vector_stride': self.visualization_params['vector_stride'].value(),
                        'd_max': self.visualization_params['d_max'].value()
                    }
                }

                # Save using numpy
                np.save(save_path, displacement_data)
                self._update_status(
                    f"Displacement data successfully saved to:\n{save_path}\n"
                    f"Pixel size: {self.pixel_size} µm\n"
                    f"Frame interval: {self.frame_length} min",
                    100
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save displacement data: {str(e)}"
            )

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
                flows = np.array(displacement_data['flows'])
                parameters = displacement_data['parameters']

                # Update calibration in parent widget
                if 'pixel_size' in parameters and 'frame_interval' in parameters:
                    self._update_parent_calibration(
                        parameters['pixel_size'],
                        parameters['frame_interval']
                    )

                # Handle flow array reshaping if needed
                if len(flows.shape) == 3:  # If flows is (frames, height*2, width)
                    frames, height_doubled, width = flows.shape
                    height = height_doubled // 2
                    flows = flows.reshape(frames, 2, height, width).transpose(0, 2, 3, 1)

                # Update optical flow parameters if available
                if 'tvl1_params' in parameters:
                    tvl1_params = parameters['tvl1_params']
                    param_mapping = {
                        'tau': 'tau',
                        'lambda': 'lambda_',
                        'theta': 'theta',
                        'nscales': 'nscales',
                        'warps': 'warps',
                        'epsilon': 'epsilon',
                        'inner_iterations': 'inner_iterations',
                        'outer_iterations': 'outer_iterations',
                        'scale_step': 'scale_step',
                        'median_filtering': 'median_filtering'
                    }

                    for saved_name, ui_name in param_mapping.items():
                        if saved_name in tvl1_params and ui_name in self.parameter_spins:
                            self.parameter_spins[ui_name].setValue(tvl1_params[saved_name])

                # Update other parameters
                if 'downscale_factor' in parameters:
                    self.parameter_spins['downscale_factor'].setValue(parameters['downscale_factor'])
                if 'arrow_scale' in parameters:
                    self.visualization_params['arrow_scale'].setValue(parameters['arrow_scale'])
                if 'vector_stride' in parameters:
                    self.visualization_params['vector_stride'].setValue(parameters['vector_stride'])
                if 'd_max' in parameters:
                    self.visualization_params['d_max'].setValue(parameters['d_max'])

                # Update analyzer parameters
                self.update_parameters()

                # Create results dictionary with updated pixel size
                results = {
                    'flows': flows,
                    'parameters': {
                        'tvl1_params': tvl1_params,
                        'downscale_factor': parameters.get('downscale_factor', 1),
                        'pixel_size': self.pixel_size  # Use current pixel size
                    },
                    'visualization_params': {
                        'd_max': parameters.get('d_max', 10.0),
                        'vector_stride': parameters.get('vector_stride', 20),
                        'arrow_scale': parameters.get('arrow_scale', 1.0)
                    },
                    'original_shape': flows.shape[1:3],
                    'flow_shape': flows.shape[1:3],
                    'units': 'micrometers'
                }

                # Update data manager and visualization
                self.data_manager.displacement_results = results
                self.visualization_manager.visualize_displacement_results(
                    results,
                    downscale_factor=parameters.get('downscale_factor', 1)
                )

                # Handle layer visibility and ordering
                self._handle_visualization_layers()

                # Update colorbar with loaded d_max
                self.colorbar_manager.update_limits(0, parameters.get('d_max', 10.0))

                # Enable save button
                self.save_displacement_btn.setEnabled(True)

                # Emit the displacement_calculated signal with the results
                self.displacement_calculated.emit(results)

                self._update_status(
                    f"Displacement data successfully loaded from:\n"
                    f"{file_path}\n"
                    f"Pixel size: {self.pixel_size} µm\n"
                    f"Frame interval: {self.frame_length} min",
                    100
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load displacement data: {str(e)}"
            )
            # Print the full error for debugging
            import traceback
            traceback.print_exc()

    def _update_parent_calibration(self, pixel_size: float, frame_interval: float):
        """Update calibration values in parent widget."""
        try:
            # Find parent widget instance (napariTFMWidget)
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

    def _on_displacement_completed(self, results):
        """Handle completion of displacement analysis"""
        super()._on_displacement_completed(results)
        self.save_displacement_btn.setEnabled(True)

    def update_parameters(self):
        """Update analysis parameters."""
        try:
            params = TVL1Parameters(
                tau=self.parameter_spins['tau'].value(),
                lambda_=self.parameter_spins['lambda_'].value(),
                theta=self.parameter_spins['theta'].value(),
                nscales=self.parameter_spins['nscales'].value(),
                warps=self.parameter_spins['warps'].value(),
                epsilon=self.parameter_spins['epsilon'].value(),
                inner_iterations=self.parameter_spins['inner_iterations'].value(),
                outer_iterations=self.parameter_spins['outer_iterations'].value(),
                scale_step=self.parameter_spins['scale_step'].value(),
                median_filtering=self.parameter_spins['median_filtering'].value(),
                downscale_factor=self.parameter_spins['downscale_factor'].value()
            )
            self.analyzer = DisplacementAnalyzer(params)

        except ValueError as e:
            self._handle_error(str(e))

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Only look at displacement input data, ignore preprocessing
        reference = self.data_manager.displacement_reference_image
        bead_stack = self.data_manager.displacement_bead_stack

        has_reference = reference is not None
        has_beads = bead_stack is not None

        # Update status labels with shape information if data exists
        if has_reference:
            self.reference_status.setText(f"Loaded: {reference.shape}")
        else:
            self.reference_status.setText("Not loaded")

        if has_beads:
            self.bead_status.setText(f"Loaded: {bead_stack.shape}")
        else:
            self.bead_status.setText("Not loaded")

        can_analyze = has_beads and has_reference
        self.analyze_btn.setEnabled(can_analyze)
        self.preview_btn.setEnabled(can_analyze)

        if not can_analyze:
            missing = []
            if not has_beads:
                missing.append("bead stack")
            if not has_reference:
                missing.append("reference image")
            self.status_label.setText(f"Missing required data: {', '.join(missing)}")
        else:
            self.status_label.setText("Ready for analysis")

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        reference = self.data_manager.displacement_reference_image
        bead_stack = self.data_manager.displacement_bead_stack

        if reference is None:
            QMessageBox.warning(self, "Error", "Reference image required")
            return False

        if bead_stack is None:
            QMessageBox.warning(self, "Error", "Bead stack required")
            return False

        return True

    def _load_data(self, data_type: str):
        """Load data from active layer."""
        active_layer = self._get_active_image_layer()
        if active_layer is None:
            QMessageBox.warning(self, "Warning", "No active image layer")
            return

        try:
            data = active_layer.data

            if data_type == 'beads':
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                if data.ndim != 3:
                    raise ValueError("Bead stack must be 3D (frames, height, width)")
                self.data_manager.set_displacement_bead_stack(data)
            else:  # reference
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")
                self.data_manager.set_displacement_reference_image(data)

            self._update_ui_state()

        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _create_status_frame(self) -> QFrame:
        """Create the status and progress frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if hasattr(self.data_manager, 'displacement_results'):
            self.visualization_manager.update_displacement_frame(
                self.viewer.dims.current_step[0]
            )

    def cleanup(self):
        """Clean up resources."""
        try:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)

            if self.colorbar_manager is not None:
                self.colorbar_manager.cleanup()
                self.colorbar_manager = None

        except Exception:
            pass

        super().cleanup()

    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
            self.load_beads_btn,
            self.load_reference_btn,
            *self.parameter_spins.values(),
            *self.visualization_params.values(),
            self.analyze_btn,
            self.preview_btn,
            self.progress_bar,
            self.status_label,
            self.bead_status,
            self.reference_status,
        ]

        for control in controls:
            self.register_control(control)
