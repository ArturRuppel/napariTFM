import napari
import numpy as np
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame,
    QProgressBar, QMessageBox, QSizePolicy
)

from .base_widget import BaseAnalysisWidget
from .colorbar import ColorbarManager
from .data_manager import DataManager
from .displacement_analysis import DisplacementAnalyzer, TVL1Parameters
from .visualization_manager import VisualizationManager

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame,
    QProgressBar, QMessageBox, QSizePolicy, QFileDialog
)
from qtpy.QtCore import Signal, Qt
import numpy as np
import os


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

        # Store default parameter values
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
            'pixel_size': 1.0,
            'vector_stride': 20,
            'arrow_scale': 1.0,
            'd_max': 10.0
        }

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group."""
        group = QGroupBox("Data")
        layout = QVBoxLayout()

        # Load buttons and status labels
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.load_reference_btn = QPushButton("Load Reference Image")
        self.clear_data_btn = QPushButton("Clear All Data")
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

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Add reset parameters button at the top
        self.reset_params_btn = QPushButton("Reset Parameters")
        layout.addWidget(self.reset_params_btn)

        # Create sections for better organization
        flow_params_group = QGroupBox("Optical Flow Parameters")
        flow_params_layout = QVBoxLayout()

        scaling_group = QGroupBox("Scaling Parameters")
        scaling_layout = QVBoxLayout()

        # Define core optical flow parameters
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
            ("median_filtering", "Median Filter Size:", 1, 9, 2, 5),
        ]

        # Add optical flow parameters
        for param_name, label, min_val, max_val, step, default in flow_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            spin = QDoubleSpinBox() if isinstance(default, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)

            self.parameter_spins[param_name] = spin
            row.addWidget(spin)
            flow_params_layout.addLayout(row)

        # Add scaling parameters
        # Downscale factor
        downscale_row = QHBoxLayout()
        downscale_row.addWidget(QLabel("Local Averaging Factor:"))
        downscale_spin = QSpinBox()
        downscale_spin.setRange(1, 10)
        downscale_spin.setSingleStep(1)
        downscale_spin.setValue(1)
        downscale_spin.setToolTip("Factor for spatial averaging of displacement field (1 = no averaging)")
        self.parameter_spins['downscale_factor'] = downscale_spin
        downscale_row.addWidget(downscale_spin)
        scaling_layout.addLayout(downscale_row)

        # Pixel size
        pixel_size_row = QHBoxLayout()
        pixel_size_row.addWidget(QLabel("Pixel Size (µm):"))
        pixel_size_spin = QDoubleSpinBox()
        pixel_size_spin.setRange(0.01, 100.0)
        pixel_size_spin.setSingleStep(0.01)
        pixel_size_spin.setValue(1.0)
        pixel_size_spin.setDecimals(3)
        pixel_size_spin.setToolTip("Size of one pixel in micrometers")
        self.parameter_spins['pixel_size'] = pixel_size_spin
        pixel_size_row.addWidget(pixel_size_spin)
        scaling_layout.addLayout(pixel_size_row)

        flow_params_group.setLayout(flow_params_layout)
        scaling_group.setLayout(scaling_layout)

        layout.addWidget(flow_params_group)
        layout.addWidget(scaling_group)
        layout.addStretch()

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        # Create 2x2 grid for buttons
        button_grid = QHBoxLayout()

        # Create left column (Preview and Save)
        left_column = QVBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.save_displacement_btn = QPushButton("Save Displacement")
        self.save_displacement_btn.setEnabled(False)
        left_column.addWidget(self.preview_btn)
        left_column.addWidget(self.save_displacement_btn)

        # Create right column (Analyze and Load)
        right_column = QVBoxLayout()
        self.analyze_btn = QPushButton("Analyze All Frames")
        self.load_displacement_btn = QPushButton("Load Displacement")
        right_column.addWidget(self.analyze_btn)
        right_column.addWidget(self.load_displacement_btn)

        # Add columns to grid
        button_grid.addLayout(left_column)
        button_grid.addLayout(right_column)

        layout.addLayout(button_grid)
        frame.setLayout(layout)
        return frame

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

    def analyze_all_frames(self):
        """Analyze displacement for all frames."""
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
            pixel_size = self.parameter_spins['pixel_size'].value()

            reference = self.data_manager.displacement_reference_image
            bead_stack = self.data_manager.displacement_bead_stack
            total_frames = len(bead_stack)

            # Process all frames
            flows = []
            for i in range(total_frames):
                frame_progress = (i + 1) / total_frames * 100
                self._update_status(
                    f"Processing frame {i + 1}/{total_frames}...\n"
                    f"Computing optical flow...",
                    frame_progress * 0.4
                )

                # Calculate flow in pixels
                flow_pixels = self.analyzer.calculate_flow(reference, bead_stack[i])

                # Downscale if requested
                if downscale_factor > 1:
                    self._update_status(
                        f"Processing frame {i + 1}/{total_frames}...\n"
                        f"Downscaling flow field...",
                        frame_progress * 0.8
                    )
                    flow_pixels = self.analyzer.downscale_flow(flow_pixels, downscale_factor)

                # Convert to micrometers and store
                flow_microns = flow_pixels * pixel_size
                flows.append(flow_microns)

            # Package results
            results = {
                'flows': flows,
                'parameters': {
                    'tvl1_params': self.analyzer.params,
                    'downscale_factor': downscale_factor,
                    'pixel_size': pixel_size
                },
                'visualization_params': vis_params,
                'original_shape': reference.shape,
                'flow_shape': flows[0].shape[:2],
                'units': 'micrometers'
            }

            # Update visualization
            self.data_manager.displacement_results = results
            self.visualization_manager.visualize_displacement_results(
                results,
                downscale_factor=downscale_factor
            )

            # Update colorbar with current d_max
            d_max = self.visualization_params['d_max'].value()
            self.colorbar_manager.update_limits(0, d_max)

            # Enable save button and emit results
            self.save_displacement_btn.setEnabled(True)
            self.displacement_calculated.emit(results)

            # Update status with statistics
            stats = self.visualization_manager.get_displacement_statistics(flows[0])
            self._update_status(
                f"Analysis complete\n"
                f"Max displacement: {stats['max']:.2f} µm\n"
                f"Mean displacement: {stats['mean']:.2f} µm\n"
                f"Flow field resolution: {flows[0].shape[:2]} (from {reference.shape})",
                100
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)
    def _save_displacement(self):
        """Save displacement data to files."""
        if not hasattr(self.data_manager, 'displacement_results') or not self.data_manager.displacement_results:
            QMessageBox.warning(self, "Warning", "No displacement data to save.")
            return

        try:
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Displacement Data",
                os.path.expanduser("~")
            )

            if save_dir:
                results = self.data_manager.displacement_results
                flows = np.array(results['flows'])

                displacement_results = {
                    'flows': flows,
                    'parameters': {
                        'pixelsize': results['parameters']['pixel_size'],
                        'downscale_factor': results['parameters']['downscale_factor'],
                        'arrow_scale': results['visualization_params']['arrow_scale'],
                        'vector_stride': results['visualization_params']['vector_stride'],
                        'd_max': results['visualization_params']['d_max']
                    }
                }

                # Save files
                np.save(os.path.join(save_dir, 'displacement.npy'), displacement_results)

                self._update_status(f"Displacement data successfully saved to:\n{save_dir}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save displacement data: {str(e)}"
            )

    def _load_displacement(self):
        """Load displacement data from files."""
        try:
            # Get file to load
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Displacement Data File",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if file_path:
                # Load the displacement data
                displacement_data = np.load(file_path, allow_pickle=True).item()

                # Convert flows to numpy array if it isn't already
                flows = np.array(displacement_data['flows'])
                if len(flows.shape) == 3:  # If flows is (frames, height*2, width)
                    frames, height_doubled, width = flows.shape
                    height = height_doubled // 2
                    # Reshape to (frames, height, width, 2)
                    flows = flows.reshape(frames, 2, height, width).transpose(0, 2, 3, 1)

                parameters = displacement_data['parameters']

                # Update UI parameters with loaded values
                if 'pixelsize' in parameters:
                    self.parameter_spins['pixel_size'].setValue(parameters['pixelsize'])
                if 'downscale_factor' in parameters:
                    self.parameter_spins['downscale_factor'].setValue(parameters['downscale_factor'])
                if 'arrow_scale' in parameters:
                    self.visualization_params['arrow_scale'].setValue(parameters['arrow_scale'])
                if 'vector_stride' in parameters:
                    self.visualization_params['vector_stride'].setValue(parameters['vector_stride'])
                if 'd_max' in parameters:
                    self.visualization_params['d_max'].setValue(parameters['d_max'])

                # Create results dictionary
                results = {
                    'flows': flows,
                    'parameters': {
                        'tvl1_params': self.analyzer.params,
                        'downscale_factor': parameters['downscale_factor'],
                        'pixel_size': parameters['pixelsize']
                    },
                    'visualization_params': {
                        'd_max': parameters['d_max'],
                        'vector_stride': parameters['vector_stride'],
                        'arrow_scale': parameters['arrow_scale']
                    },
                    'original_shape': flows.shape[1:3],
                    'flow_shape': flows.shape[1:3],
                    'units': 'micrometers'
                }

                # Update data manager and visualization
                self.data_manager.displacement_results = results
                self.visualization_manager.visualize_displacement_results(
                    results,
                    downscale_factor=parameters['downscale_factor']
                )

                # Update colorbar with loaded d_max
                self.colorbar_manager.update_limits(0, parameters['d_max'])

                # Enable save button
                self.save_displacement_btn.setEnabled(True)

                # Emit the displacement_calculated signal with the results
                self.displacement_calculated.emit(results)

                self._update_status(f"Displacement data successfully loaded from:\n{file_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load displacement data: {str(e)}"
            )
            # Print the full error for debugging
            import traceback
            traceback.print_exc()
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

            # Safely determine current frame
            if bead_stack.ndim == 2:
                # Handle 2D case
                moving = bead_stack
            else:
                # Handle 3D case
                if len(self.viewer.dims.current_step) > 0:
                    current_frame = self.viewer.dims.current_step[0]
                    # Ensure frame index is valid
                    if current_frame >= bead_stack.shape[0]:
                        current_frame = 0
                else:
                    current_frame = 0
                moving = bead_stack[current_frame]

            # Calculate initial flow in pixels
            flow_pixels = self.analyzer.calculate_flow(reference, moving)

            # Apply downscaling if factor > 1
            downscale_factor = self.parameter_spins['downscale_factor'].value()
            if downscale_factor > 1:
                flow_pixels = self.analyzer.downscale_flow(flow_pixels, downscale_factor)

            # Convert flow to micrometers
            pixel_size = self.parameter_spins['pixel_size'].value()
            self.current_flow = flow_pixels * pixel_size

            # Get visualization parameters
            vis_params = {
                'd_max': self.visualization_params['d_max'].value(),
                'vector_stride': self.visualization_params['vector_stride'].value(),
                'arrow_scale': self.visualization_params['arrow_scale'].value()
            }

            # Update visualization through visualization manager
            self.visualization_manager.visualize_displacement_preview(
                self.current_flow,  # Already in µm
                vis_params['d_max'],
                vis_params['vector_stride'],
                vis_params['arrow_scale'],
                downscale_factor=downscale_factor
            )

            # Update colorbar with current d_max
            d_max = self.visualization_params['d_max'].value()
            self.colorbar_manager.update_limits(0, d_max)

            # Update status with displacement statistics
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
        self.visualization_params['vector_stride'].setToolTip("Display every nth vector")
        stride_layout.addWidget(self.visualization_params['vector_stride'])
        layout.addLayout(stride_layout)

        # Arrow scale
        arrow_layout = QHBoxLayout()
        arrow_layout.addWidget(QLabel("Arrow Scale:"))
        self.visualization_params['arrow_scale'] = QDoubleSpinBox()
        self.visualization_params['arrow_scale'].setRange(0.1, 50.0)
        self.visualization_params['arrow_scale'].setSingleStep(0.5)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['arrow_scale'].setToolTip("Scale factor for arrow length")
        arrow_layout.addWidget(self.visualization_params['arrow_scale'])
        layout.addLayout(arrow_layout)

        # Maximum displacement (now in µm)
        dmax_layout = QHBoxLayout()
        dmax_layout.addWidget(QLabel("Max Displacement (µm):"))
        self.visualization_params['d_max'] = QDoubleSpinBox()
        self.visualization_params['d_max'].setRange(0.1, 200.0)
        self.visualization_params['d_max'].setSingleStep(1.0)
        self.visualization_params['d_max'].setValue(10.0)
        self.visualization_params['d_max'].setToolTip("Maximum displacement for color scaling (in µm)")
        dmax_layout.addWidget(self.visualization_params['d_max'])
        layout.addLayout(dmax_layout)

        group.setLayout(layout)
        return group

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

    def _clear_data(self):
        """Clear all displacement analysis data"""
        try:
            # Clear data from manager
            self.data_manager.clear_displacement_data()

            # Update UI
            self._update_ui_state()
            self._update_status("All displacement analysis data cleared")

        except Exception as e:
            self._handle_error(str(e))

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
