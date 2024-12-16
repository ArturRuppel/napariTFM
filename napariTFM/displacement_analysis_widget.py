from typing import Optional, Dict
import numpy as np
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
    QCheckBox, QFrame, QSizePolicy, QScrollArea,
    QProgressBar, QMessageBox, QWidget
)
import napari

from .base_widget import BaseAnalysisWidget
from .displacement_analysis import DisplacementAnalyzer, DisplacementParameters


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements."""

    # Signals
    displacement_calculated = Signal(dict)  # Emits displacement results

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize analyzer
        self.analyzer = DisplacementAnalyzer()

        # Initialize state variables
        self.displacement_layer = None
        self.vector_layer = None
        self.preview_enabled = False

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create container widget
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Add title
        title = QLabel("Displacement Analysis")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(title)

        # Add all component groups
        main_layout.addWidget(self._create_method_group())
        main_layout.addWidget(self._create_parameters_group())
        main_layout.addWidget(self._create_visualization_group())
        main_layout.addWidget(self._create_action_buttons())
        main_layout.addWidget(self._create_status_frame())

        container.setLayout(main_layout)
        scroll.setWidget(container)

        # Set the final layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setLayout(layout)

        # Connect signals
        self._connect_signals()

        # Register controls
        self._register_controls()

    def _create_parameters_group(self) -> QGroupBox:
        """Create the parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Bead detection parameters
        layout.addWidget(QLabel("Bead Detection:"))
        self.min_distance_spin = self._add_parameter_row(
            layout, "Min Distance:", QSpinBox,
            (1, 20, 1, self.analyzer.params.min_distance)
        )
        self.threshold_spin = self._add_parameter_row(
            layout, "Threshold:", QDoubleSpinBox,
            (0.01, 1.0, 0.01, self.analyzer.params.threshold_rel)
        )
        self.sigma_spin = self._add_parameter_row(
            layout, "Smoothing:", QDoubleSpinBox,
            (0.1, 5.0, 0.1, self.analyzer.params.sigma)
        )

        layout.addSpacing(10)

        # Tracking parameters
        layout.addWidget(QLabel("Tracking:"))
        self.max_displacement_spin = self._add_parameter_row(
            layout, "Max Displacement:", QSpinBox,
            (10, 500, 1, self.analyzer.params.max_displacement)
        )

        layout.addSpacing(10)

        # Optical flow parameters
        layout.addWidget(QLabel("Optical Flow:"))
        self.pyr_scale_spin = self._add_parameter_row(
            layout, "Pyramid Scale:", QDoubleSpinBox,
            (0.1, 0.99, 0.01, self.analyzer.params.pyr_scale)
        )
        self.levels_spin = self._add_parameter_row(
            layout, "Pyramid Levels:", QSpinBox,
            (1, 10, 1, self.analyzer.params.levels)
        )
        self.winsize_spin = self._add_parameter_row(
            layout, "Window Size:", QSpinBox,
            (5, 65, 2, self.analyzer.params.winsize)
        )

        group.setLayout(layout)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        """Create the visualization options group."""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        self.show_vectors_check = QCheckBox("Show Flow Vectors")
        self.show_magnitude_check = QCheckBox("Show Magnitude Heatmap")
        self.show_beads_check = QCheckBox("Show Tracked Beads")

        layout.addWidget(self.show_vectors_check)
        layout.addWidget(self.show_magnitude_check)
        layout.addWidget(self.show_beads_check)

        group.setLayout(layout)
        return group

    def _update_visualization(self, flow: Optional[np.ndarray] = None,
                              results: Optional[Dict] = None):
        """Update displacement visualization."""
        try:
            # Clear existing layers
            self._clear_layers()

            if flow is None or results is None:
                return

            # Add magnitude layer if enabled
            if self.show_magnitude_check.isChecked():
                self.displacement_layer = self.viewer.add_image(
                    results['magnitude'],
                    name='Displacement Magnitude',
                    colormap='inferno',
                    blending='additive',
                    opacity=0.7
                )

            # Add vector layer if enabled
            if self.show_vectors_check.isChecked():
                if self.show_beads_check.isChecked() and 'bead_positions' in results:
                    # Show vectors only at bead positions
                    vectors = self.analyzer.get_vector_coordinates(
                        flow,
                        bead_positions=results['bead_positions']
                    )
                else:
                    # Show vectors on regular grid
                    vectors = self.analyzer.get_vector_coordinates(flow, stride=20)

                self.vector_layer = self.viewer.add_vectors(
                    vectors,
                    name='Displacement Vectors',
                    edge_width=0.5,
                    length=1.0,
                    edge_color='yellow'
                )

            # Add bead positions if enabled
            if self.show_beads_check.isChecked() and 'ref_beads' in results:
                self.viewer.add_points(
                    results['ref_beads'],
                    name='Reference Beads',
                    size=5,
                    face_color='blue',
                    opacity=0.7
                )
                if 'frame_beads' in results:
                    self.viewer.add_points(
                        results['frame_beads'],
                        name='Current Beads',
                        size=5,
                        face_color='red',
                        opacity=0.7
                    )

        except Exception as e:
            self._handle_error(f"Visualization update failed: {str(e)}")

    def _update_parameters(self):
        """Update analyzer parameters from UI values."""
        try:
            params = DisplacementParameters(
                # Bead detection
                min_distance=self.min_distance_spin.value(),
                threshold_rel=self.threshold_spin.value(),
                sigma=self.sigma_spin.value(),
                # Tracking
                max_displacement=self.max_displacement_spin.value(),
                # Optical flow
                pyr_scale=self.pyr_scale_spin.value(),
                levels=self.levels_spin.value(),
                winsize=self.winsize_spin.value(),
                iterations=self.analyzer.params.iterations,
                poly_n=self.analyzer.params.poly_n,
                poly_sigma=self.analyzer.params.poly_sigma
            )

            params.validate()
            self.analyzer.update_parameters(params)

            # Update preview if enabled
            if hasattr(self, 'preview_enabled') and self.preview_enabled:
                self.preview_displacement()

        except ValueError as e:
            self._handle_error(str(e))

    def _clear_layers(self):
        """Remove displacement visualization layers."""
        layer_names = [
            'Displacement Magnitude',
            'Displacement Vectors',
            'Reference Beads',
            'Current Beads'
        ]
        for layer_name in layer_names:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        self.displacement_layer = None
        self.vector_layer = None

    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
            self.min_distance_spin,
            self.threshold_spin,
            self.sigma_spin,
            self.max_displacement_spin,
            self.pyr_scale_spin,
            self.levels_spin,
            self.winsize_spin,
            self.show_vectors_check,
            self.show_magnitude_check,
            self.show_beads_check,
            self.preview_btn,
            self.analyze_btn
        ]

        for control in controls:
            self.register_control(control)

    def _connect_signals(self):
        """Connect widget signals."""
        # Parameter update signals
        for spin in [
            self.min_distance_spin,
            self.threshold_spin,
            self.sigma_spin,
            self.max_displacement_spin,
            self.pyr_scale_spin,
            self.levels_spin,
            self.winsize_spin
        ]:
            spin.valueChanged.connect(self._update_parameters)

        # Visualization update signals
        self.show_vectors_check.toggled.connect(self._update_visualization)
        self.show_magnitude_check.toggled.connect(self._update_visualization)
        self.show_beads_check.toggled.connect(self._update_visualization)

        # Action signals
        self.preview_btn.clicked.connect(self.preview_displacement)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)

    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self.data_manager.has_required_registration_data():
                raise ValueError("Reference image and bead stack required")

            # Get current frame
            current_frame = self.viewer.dims.current_step[0]
            bead_frame = self.data_manager.preprocessed_bead_stack[current_frame]
            reference = self.data_manager.preprocessed_reference

            # Calculate displacement
            flow, results = self.analyzer.analyze_frame(reference, bead_frame)
            self.current_results = results  # Store for visualization updates

            # Update visualization
            self._update_visualization(flow, results)

            # Update status
            self._update_status(
                f"Max displacement: {results['max_displacement']:.2f} pixels\n"
                f"Mean displacement: {results['mean_displacement']:.2f} pixels\n"
                f"Tracked beads: {results['num_tracked_beads']}"
            )

        except Exception as e:
            self._handle_error(str(e))

    def analyze_all_frames(self):
        """Analyze displacement for all frames."""
        try:
            if not self.data_manager.has_required_registration_data():
                raise ValueError("Reference image and bead stack required")

            self._set_controls_enabled(False)
            self._update_status("Starting analysis...", 0)

            def progress_callback(progress):
                self._update_status(
                    f"Processing... {progress:.1f}%",
                    int(progress)
                )

            # Run analysis
            results = self.analyzer.analyze_stack(
                self.data_manager.preprocessed_reference,
                self.data_manager.preprocessed_bead_stack,
                progress_callback
            )

            # Process results
            flows, frame_results = zip(*results)

            # Store results in data manager
            self.data_manager.displacement_results = {
                'flows': flows,
                'frame_results': frame_results,
                'parameters': self.analyzer.params
            }

            # Emit results
            self.displacement_calculated.emit(self.data_manager.displacement_results)

            # Update visualization for current frame
            current_frame = self.viewer.dims.current_step[0]
            self._update_visualization(flows[current_frame], frame_results[current_frame])

            self._update_status(
                f"Analysis complete. Tracked {frame_results[current_frame]['num_tracked_beads']} beads.",
                100
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if hasattr(self, 'current_results') and self.data_manager.displacement_results is not None:
            current_frame = self.viewer.dims.current_step[0]
            flows = self.data_manager.displacement_results['flows']
            frame_results = self.data_manager.displacement_results['frame_results']

            if 0 <= current_frame < len(flows):
                self._update_visualization(flows[current_frame], frame_results[current_frame])

    def _create_method_group(self) -> QGroupBox:
        """Create the method selection group."""
        group = QGroupBox("Analysis Method")
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(['Farneback Optical Flow', 'Feature Tracking'])
        layout.addWidget(self.method_combo)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout()

        self.preview_btn = QPushButton("Preview")
        self.analyze_btn = QPushButton("Analyze All Frames")

        layout.addWidget(self.preview_btn)
        layout.addWidget(self.analyze_btn)

        frame.setLayout(layout)
        return frame

    def _create_status_frame(self) -> QFrame:
        """Create the status frame."""
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _add_parameter_row(self, layout: QVBoxLayout, label: str,
                           spin_type: type, spin_params: tuple) -> QSpinBox:
        """Helper to add a parameter row with consistent styling."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))

        spinbox = spin_type()
        min_val, max_val, step, default = spin_params

        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setValue(default)

        row.addWidget(spinbox)
        row.addStretch()

        layout.addLayout(row)
        return spinbox

    def _on_method_changed(self, method: str):
        """Handle change of analysis method."""
        if method == 'Feature Tracking':
            QMessageBox.information(
                self,
                "Feature Tracking",
                "Feature tracking method is not yet implemented."
            )
            self.method_combo.setCurrentText('Farneback Optical Flow')

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        has_data = self.data_manager.has_required_registration_data()
        self.preview_btn.setEnabled(has_data)
        self.analyze_btn.setEnabled(has_data)

        if not has_data:
            self.status_label.setText("Load reference image and bead stack to begin analysis")

    def cleanup(self):
        """Clean up resources and layers."""
        self._clear_layers()
        super().cleanup()