from typing import Optional, Dict
import numpy as np
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget,
    QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
    QCheckBox, QFrame, QSizePolicy, QScrollArea,
    QProgressBar, QMessageBox
)
import napari

from .base_widget import BaseAnalysisWidget
from .displacement_analysis import DisplacementAnalyzer, TVL1Parameters


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements using TV-L1 optical flow."""

    displacement_calculated = Signal(dict)  # Emits displacement results

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager",
                 visualization_manager: "VisualizationManager"):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize analyzer
        self.analyzer = DisplacementAnalyzer()

        # Initialize state variables
        self.current_flow = None
        self.parameter_spins = {}  # Initialize dictionary before UI setup

        # Setup UI first
        self._setup_ui()

        # Connect to viewer events
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

        # Connect signals after UI is fully set up
        self._connect_signals()

        # Update initial UI state
        self._update_ui_state()

    def _setup_ui(self):
        """Set up the user interface."""
        # Create scroll area and container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Add all component groups
        main_layout.addWidget(self._create_data_loading_group())
        main_layout.addWidget(self._create_parameters_group())
        main_layout.addWidget(self._create_action_buttons())
        main_layout.addWidget(self._create_status_frame())

        container.setLayout(main_layout)
        scroll.setWidget(container)

        # Set the final layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setLayout(layout)

        self._register_controls()

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # TV-L1 parameters including d_max
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
            ("d_max", "Max Displacement:", 0.1, 100.0, 0.1, 10.0)
        ]

        # Add all parameters
        for param_name, label, min_val, max_val, step, default in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            spin = QDoubleSpinBox() if isinstance(default, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)

            self.parameter_spins[param_name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        layout = QHBoxLayout()

        self.preview_btn = QPushButton("Preview Current Frame")
        self.analyze_btn = QPushButton("Analyze All Frames")

        layout.addWidget(self.preview_btn)
        layout.addWidget(self.analyze_btn)

        frame.setLayout(layout)
        return frame

    def update_parameters(self):
        """Update analysis and visualization parameters."""
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
                median_filtering=self.parameter_spins['median_filtering'].value()
            )
            self.analyzer = DisplacementAnalyzer(params)

            # Update d_max in visualization manager
            self.visualization_manager.set_d_max(self.parameter_spins['d_max'].value())

        except ValueError as e:
            self._handle_error(str(e))

    def _connect_signals(self):
        """Connect all widget signals."""
        # Data loading
        self.load_beads_btn.clicked.connect(lambda: self._load_data('beads'))
        self.load_reference_btn.clicked.connect(lambda: self._load_data('reference'))
        self.load_cells_btn.clicked.connect(lambda: self._load_data('cells'))

        # Parameter updates
        for spin in self.parameter_spins.values():
            spin.valueChanged.connect(self.update_parameters)

        # Action buttons
        self.preview_btn.clicked.connect(self.preview_displacement)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group."""
        group = QGroupBox("Data")
        layout = QVBoxLayout()

        # Load buttons
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.load_reference_btn = QPushButton("Load Reference Image")
        self.load_cells_btn = QPushButton("Load Cell Stack (Optional)")

        # Status labels
        self.bead_status = QLabel("Not loaded")
        self.reference_status = QLabel("Not loaded")
        self.cell_status = QLabel("Not loaded")

        # Add with status labels
        for btn, status in [
            (self.load_beads_btn, self.bead_status),
            (self.load_reference_btn, self.reference_status),
            (self.load_cells_btn, self.cell_status)
        ]:
            row = QHBoxLayout()
            row.addWidget(btn)
            row.addWidget(status)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

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

    def _update_d_max(self):
        """Update visualization manager's d_max value."""
        self.visualization_manager.set_d_max(self.d_max_spin.value())

    def analyze_all_frames(self):
        """Analyze displacement for all frames."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Starting analysis...", 0)

            # Get input data
            reference = (self.data_manager.preprocessed_reference if self.data_manager.preprocessed_reference is not None
                         else self.data_manager.reference_image)
            bead_stack = (self.data_manager.preprocessed_bead_stack if self.data_manager.preprocessed_bead_stack is not None
                          else self.data_manager.bead_stack)

            results = {
                'flows': [],
                'magnitudes': [],
                'parameters': self.analyzer.params
            }

            # Process each frame
            for i in range(len(bead_stack)):
                progress = (i + 1) / len(bead_stack) * 100
                self._update_status(f"Processing frame {i + 1}/{len(bead_stack)}...", progress)

                # Calculate flow
                flow = self.analyzer.calculate_flow(reference, bead_stack[i])
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

                results['flows'].append(flow)
                results['magnitudes'].append(magnitude)

                # Update visualization for current frame
                if i == self.viewer.dims.current_step[0]:
                    self.current_flow = flow
                    self._update_visualization(flow, reference, bead_stack[i])

            # Save and emit results
            self.data_manager.displacement_results = results
            self.displacement_calculated.emit(results)

            self._update_status("Analysis complete", 100)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _update_visualization(self, flow: np.ndarray, reference: np.ndarray, moving: np.ndarray):
        """Update displacement visualization."""
        # Get cell data if available
        cells = None
        if self.data_manager.preprocessed_cell_stack is not None:
            cells = self.data_manager.preprocessed_cell_stack[self.viewer.dims.current_step[0]]
        elif self.data_manager.cell_stack is not None:
            cells = self.data_manager.cell_stack[self.viewer.dims.current_step[0]]

        # Update visualization with all components enabled
        self.visualization_manager.update_displacement_visualization(
            reference=reference,
            moving=moving,
            flow=flow,
            cells=cells,
            show_overlay=True,
            show_vectors=True,
            show_magnitude=True,
            vector_stride=20
        )

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if hasattr(self.data_manager, 'displacement_results'):
            results = self.data_manager.displacement_results
            if results and 'flows' in results:
                current_frame = self.viewer.dims.current_step[0]
                if current_frame < len(results['flows']):
                    flow = results['flows'][current_frame]
                    reference = (self.data_manager.preprocessed_reference if self.data_manager.preprocessed_reference is not None
                                 else self.data_manager.reference_image)
                    moving = (self.data_manager.preprocessed_bead_stack[current_frame] if self.data_manager.preprocessed_bead_stack is not None
                              else self.data_manager.bead_stack[current_frame])
                    self._update_visualization(flow, reference, moving)

    def _load_data(self, data_type: str):
        """Load data from active layer."""
        active_layer = self._get_active_image_layer()
        if active_layer is None:
            QMessageBox.warning(self, "Warning", "No active image layer")
            return

        try:
            data = active_layer.data

            # Ensure 3D data for stacks
            if data_type in ['beads', 'cells']:
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                if data.ndim != 3:
                    raise ValueError(f"{data_type} stack must be 3D (frames, height, width)")
            else:  # reference
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")

            # Set data in manager
            if data_type == 'beads':
                self.data_manager.bead_stack = data
                self.bead_status.setText(f"Loaded: {data.shape}")
            elif data_type == 'reference':
                self.data_manager.reference_image = data
                self.reference_status.setText(f"Loaded: {data.shape}")
            elif data_type == 'cells':
                self.data_manager.cell_stack = data
                self.cell_status.setText(f"Loaded: {data.shape}")

            self._update_ui_state()

        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Check for either preprocessed or raw data
        has_reference = (self.data_manager.preprocessed_reference is not None or
                         self.data_manager.reference_image is not None)
        has_beads = (self.data_manager.preprocessed_bead_stack is not None or
                     self.data_manager.bead_stack is not None)

        # Enable/disable analyze button
        can_analyze = has_beads and has_reference
        self.analyze_btn.setEnabled(can_analyze)

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        has_reference = (self.data_manager.preprocessed_reference is not None or
                         self.data_manager.reference_image is not None)
        has_beads = (self.data_manager.preprocessed_bead_stack is not None or
                     self.data_manager.bead_stack is not None)

        if not has_reference:
            QMessageBox.warning(self, "Error", "Reference image required")
            return False

        if not has_beads:
            QMessageBox.warning(self, "Error", "Bead stack required")
            return False

        return True

    def _load_active_layer(self, data_type: str):
        """Load the currently active layer as the specified data type"""
        active_layer = self._get_active_image_layer()
        if active_layer is None:
            QMessageBox.warning(self, "Warning", "No active image layer")
            return

        try:
            data = active_layer.data
            if data_type in ['beads', 'cells']:
                # Ensure 3D data for stacks
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                if data.ndim != 3:
                    raise ValueError(f"{data_type} stack must be 3D (frames, height, width)")

            elif data_type == 'reference':
                # Ensure 2D data for reference
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")

            # Set data in manager using the appropriate method
            if data_type == 'beads':
                self.data_manager.set_bead_stack(data)
                self.bead_status.setText(f"Loaded: {data.shape}")
            elif data_type == 'reference':
                self.data_manager.set_reference_image(data)
                self.reference_status.setText(f"Loaded: {data.shape}")
            elif data_type == 'cells':
                self.data_manager.set_cell_stack(data)
                self.cell_status.setText(f"Loaded: {data.shape}")

            # Update UI state after successful load
            self._update_ui_state()
            self._update_status(f"Loaded {data_type} data successfully")

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _update_parameters(self):
        """Mark parameters as modified but don't trigger immediate update."""
        self.parameters_modified = True

    def _apply_parameters(self):
        """Actually update the analyzer parameters."""
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
                median_filtering=self.parameter_spins['median_filtering'].value()
            )
            self.analyzer = DisplacementAnalyzer(params)
            self.parameters_modified = False
        except ValueError as e:
            self._handle_error(str(e))

    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating displacement...", 0)

            # Get current frame index and data
            current_frame = self.viewer.dims.current_step[0]

            # Use preprocessed data if available, otherwise use raw data
            reference = self.data_manager.preprocessed_reference
            if reference is None:
                reference = self.data_manager.reference_image

            bead_stack = self.data_manager.preprocessed_bead_stack
            if bead_stack is None:
                bead_stack = self.data_manager.bead_stack

            moving = bead_stack[current_frame]

            # Calculate flow
            self.current_flow = self.analyzer.calculate_flow(reference, moving)

            self._update_status("Updating visualization...", 50)

            # Get cell data if available
            cells = None
            if self.data_manager.preprocessed_cell_stack is not None:
                cells = self.data_manager.preprocessed_cell_stack[current_frame]
            elif self.data_manager.cell_stack is not None:
                cells = self.data_manager.cell_stack[current_frame]

            # Update visualization
            self._update_visualization(self.current_flow, reference, moving)

            # Update status with displacement statistics
            stats = self.visualization_manager.get_displacement_statistics(self.current_flow)
            self._update_status(
                f"Max displacement: {stats['max']:.2f} pixels\n"
                f"Mean displacement: {stats['mean']:.2f} pixels",
                100
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _update_visualization_if_active(self, force: bool = False):
        """Update visualization if there's active flow data."""
        if self.current_flow is not None and (force or not self.parameters_modified):
            self._update_visualization(self.current_flow)

    def cleanup(self):
        """Clean up resources before widget is destroyed."""
        # Clear visualization
        self.visualization_manager._clear_displacement_layers()

        # Clear references
        self.current_flow = None

        super().cleanup()

    def reset(self):
        """Reset widget to initial state."""
        self.cleanup()

        # Reset parameters to defaults
        for param_name, spin in self.parameter_spins.items():
            if param_name == 'd_max':
                spin.setValue(10.0)
            else:
                default_value = getattr(TVL1Parameters(), param_name)
                spin.setValue(default_value)

        # Reset checkboxes
        self.show_overlay_check.setChecked(False)
        self.show_vectors_check.setChecked(False)
        self.show_magnitude_check.setChecked(False)
        self.show_cells_check.setChecked(False)

        # Reset vector stride
        self.vector_stride_spin.setValue(20)

        # Update UI
        self._update_ui_state()
        self._update_status("Widget reset to default state")

    def _create_visualization_group(self) -> QGroupBox:
        """Create the visualization options group."""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        # Visualization options
        self.show_overlay_check = QCheckBox("Show Image Overlay")
        self.show_vectors_check = QCheckBox("Show Flow Vectors")
        self.show_magnitude_check = QCheckBox("Show Magnitude Heatmap")
        self.show_cells_check = QCheckBox("Show Cell Overlay")

        # Vector display options
        vector_layout = QHBoxLayout()
        vector_layout.addWidget(QLabel("Vector Stride:"))
        self.vector_stride_spin = QSpinBox()
        self.vector_stride_spin.setRange(5, 50)
        self.vector_stride_spin.setValue(20)
        vector_layout.addWidget(self.vector_stride_spin)
        vector_layout.addStretch()

        # Add all elements
        layout.addWidget(self.show_overlay_check)
        layout.addWidget(self.show_vectors_check)
        layout.addWidget(self.show_magnitude_check)
        layout.addWidget(self.show_cells_check)
        layout.addLayout(vector_layout)

        group.setLayout(layout)
        return group

    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
            # Data loading controls
            self.load_beads_btn,
            self.load_reference_btn,
            self.load_cells_btn,

            # Analysis parameters
            *self.parameter_spins.values(),

            # Action buttons
            self.analyze_btn,

            # Status elements
            self.progress_bar,
            self.status_label,
            self.bead_status,
            self.reference_status,
            self.cell_status
        ]

        for control in controls:
            self.register_control(control)

    def _save_results(self, results: Dict):
        """Save displacement analysis results."""
        try:
            # Store results in data manager
            self.data_manager.displacement_results = results

            # Emit results signal
            self.displacement_calculated.emit(results)

            # Update status
            self._update_status("Results saved successfully")

        except Exception as e:
            self._handle_error(f"Failed to save results: {str(e)}")

    def generate_report(self):
        """Generate a summary report of the displacement analysis."""
        if not hasattr(self.data_manager, 'displacement_results'):
            QMessageBox.warning(self, "Error", "No analysis results available")
            return

        try:
            results = self.data_manager.displacement_results
            report = (
                "Displacement Analysis Report\n"
                "==========================\n\n"
                f"Number of frames analyzed: {len(results['flows'])}\n"
                f"Analysis parameters:\n"
                f"  - Tau: {self.analyzer.params.tau}\n"
                f"  - Lambda: {self.analyzer.params.lambda_}\n"
                f"  - Scales: {self.analyzer.params.nscales}\n"
                "\nDisplacement Statistics:\n"
            )

            # Add frame-by-frame statistics
            for i, flow in enumerate(results['flows']):
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                report += (
                    f"\nFrame {i}:\n"
                    f"  Max displacement: {magnitude.max():.2f} pixels\n"
                    f"  Mean displacement: {magnitude.mean():.2f} pixels\n"
                    f"  Std deviation: {magnitude.std():.2f} pixels\n"
                )

            # Show report in a message box
            QMessageBox.information(self, "Analysis Report", report)

        except Exception as e:
            self._handle_error(f"Failed to generate report: {str(e)}")
