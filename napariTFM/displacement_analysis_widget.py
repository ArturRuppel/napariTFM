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

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

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
            label="Displacement (pixels)",
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
        right_container.setFixedWidth(300)

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)
        self._register_controls()

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group."""
        group = QGroupBox("Data")
        layout = QVBoxLayout()

        # Load buttons and status labels
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.load_reference_btn = QPushButton("Load Reference Image")
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

        group.setLayout(layout)
        return group

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
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
            ("median_filtering", "Median Filter Size:", 1, 9, 2, 5)
        ]

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

        # Maximum displacement
        dmax_layout = QHBoxLayout()
        dmax_layout.addWidget(QLabel("Max Displacement:"))
        self.visualization_params['d_max'] = QDoubleSpinBox()
        self.visualization_params['d_max'].setRange(0.1, 200.0)
        self.visualization_params['d_max'].setSingleStep(1.0)
        self.visualization_params['d_max'].setValue(10.0)
        self.visualization_params['d_max'].setToolTip("Maximum displacement for color scaling")
        dmax_layout.addWidget(self.visualization_params['d_max'])
        layout.addLayout(dmax_layout)

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

    def _connect_signals(self):
        """Connect all widget signals."""
        self.load_beads_btn.clicked.connect(lambda: self._load_data('beads'))
        self.load_reference_btn.clicked.connect(lambda: self._load_data('reference'))

        for spin in self.parameter_spins.values():
            spin.valueChanged.connect(self.update_parameters)

        self.preview_btn.clicked.connect(self.preview_displacement)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

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
                self.data_manager.bead_stack = data
            else:  # reference
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")
                self.data_manager.reference_image = data

            self._update_ui_state()

        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        reference = (self.data_manager.preprocessed_reference or
                     self.data_manager.reference_image)
        bead_stack = (self.data_manager.preprocessed_bead_stack or
                      self.data_manager.bead_stack)

        has_reference = reference is not None
        has_beads = bead_stack is not None

        self.reference_status.setText("Loaded: " + str(reference.shape) if has_reference else "Not loaded")
        self.bead_status.setText("Loaded: " + str(bead_stack.shape) if has_beads else "Not loaded")

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

    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating displacement...", 0)

            current_frame = self.viewer.dims.current_step[0]
            reference = (self.data_manager.preprocessed_reference or
                         self.data_manager.reference_image)
            bead_stack = (self.data_manager.preprocessed_bead_stack or
                          self.data_manager.bead_stack)
            moving = bead_stack[current_frame]

            self.current_flow = self.analyzer.calculate_flow(reference, moving)

            # Delegate visualization to visualization manager
            self.visualization_manager.visualize_displacement_preview(
                self.current_flow,
                reference,
                moving,
                self.visualization_params['d_max'].value(),
                self.visualization_params['vector_stride'].value(),
                self.visualization_params['arrow_scale'].value()
            )

            # Update colorbar
            self.colorbar_manager.update_limits(0, self.visualization_params['d_max'].value())

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

    def analyze_all_frames(self):
        """Analyze displacement for all frames."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Starting analysis...", 0)

            reference = (self.data_manager.preprocessed_reference or
                         self.data_manager.reference_image)
            bead_stack = (self.data_manager.preprocessed_bead_stack or
                          self.data_manager.bead_stack)

            # Process all frames
            flows = []
            for i in range(len(bead_stack)):
                progress = (i + 1) / len(bead_stack) * 100
                self._update_status(f"Processing frame {i + 1}/{len(bead_stack)}...", progress)

                flow = self.analyzer.calculate_flow(reference, bead_stack[i])
                flows.append(flow)

            # Package results
            results = {
                'flows': flows,
                'parameters': self.analyzer.params,
                'visualization_params': {
                    'd_max': self.visualization_params['d_max'].value(),
                    'vector_stride': self.visualization_params['vector_stride'].value(),
                    'arrow_scale': self.visualization_params['arrow_scale'].value()
                }
            }

            # Store results and update visualization
            self.data_manager.displacement_results = results
            self.visualization_manager.visualize_displacement_results(results)

            # Emit results and update status
            self.displacement_calculated.emit(results)
            self._update_status("Analysis complete", 100)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        reference = (self.data_manager.preprocessed_reference or
                     self.data_manager.reference_image)
        bead_stack = (self.data_manager.preprocessed_bead_stack or
                      self.data_manager.bead_stack)

        if reference is None:
            QMessageBox.warning(self, "Error", "Reference image required")
            return False

        if bead_stack is None:
            QMessageBox.warning(self, "Error", "Bead stack required")
            return False

        return True

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
                median_filtering=self.parameter_spins['median_filtering'].value()
            )
            self.analyzer = DisplacementAnalyzer(params)

        except ValueError as e:
            self._handle_error(str(e))

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



