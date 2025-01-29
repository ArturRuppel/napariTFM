import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import Signal, Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox,
    QSizePolicy, QFrame, QProgressBar
)
from napari.qt.threading import thread_worker

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.data_manager_old import DataManager
from napariTFM.parameter_manager_old import ParameterManager
from napariTFM.visualization_manager import VisualizationManager
from napariTFM.services.displacement_service import DisplacementService, DisplacementParameters


class DisplacementParameterPanel(QWidget):
    """Panel for handling all displacement parameter inputs."""

    parameter_changed = Signal()
    parameter_value_changed = Signal(str, object)

    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_widgets = {}
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create parameter groups
        layout.addWidget(self._create_flow_parameters())
        layout.addWidget(self._create_analysis_parameters())
        layout.addWidget(self._create_visualization_parameters())

        # Add reset button
        self.reset_params_btn = QPushButton("Reset Parameters")
        layout.addWidget(self.reset_params_btn)

        self.setLayout(layout)

    def _create_flow_parameters(self) -> QGroupBox:
        """Create optical flow parameter group."""
        group = QGroupBox("Optical Flow Parameters")
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
            ("median_filtering", "Median Filter:", 1, 9, 2, 5),
        ]

        for name, label, min_val, max_val, step, default in params:
            widget = self._create_parameter_widget(name, label, min_val, max_val, step, default)
            layout.addLayout(widget)

        group.setLayout(layout)
        return group

    def _create_analysis_parameters(self) -> QGroupBox:
        """Create analysis parameter group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Downscale factor
        layout.addLayout(
            self._create_parameter_widget(
                "downscale_factor", "Downscale Factor:",
                1, 10, 1, 1
            )
        )

        group.setLayout(layout)
        return group

    def _create_visualization_parameters(self) -> QGroupBox:
        """Create visualization parameter group."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        params = [
            ("d_max", "Max Displacement (µm):", 0.1, 200.0, 0.1, 10.0),
            ("disp_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("disp_arrow_scale", "Arrow Scale:", 0.1, 10.0, 0.1, 1.0),
        ]

        for name, label, min_val, max_val, step, default in params:
            widget = self._create_parameter_widget(name, label, min_val, max_val, step, default)
            layout.addLayout(widget)

        group.setLayout(layout)
        return group

    def _create_parameter_widget(self, name, label, min_val, max_val, step, default) -> QHBoxLayout:
        """Create a parameter widget with label and input."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))

        if isinstance(step, int):
            spin = QSpinBox()
        else:
            spin = QDoubleSpinBox()
            if name == "epsilon":
                spin.setDecimals(3)
            else:
                spin.setDecimals(2)

        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(default)

        self.parameter_widgets[name] = spin
        layout.addWidget(spin)

        return layout

    def _connect_signals(self):
        """Connect widget signals to parameter manager."""
        for name, widget in self.parameter_widgets.items():
            widget.valueChanged.connect(
                lambda value, n=name: self._on_value_changed(n, value)
            )

        self.parameter_manager.parameter_changed.connect(self._update_widget_value)

    def _on_value_changed(self, param_name: str, value: object):
        """Handle parameter value changes."""
        self.parameter_manager.set_value(param_name, value)
        self.parameter_value_changed.emit(param_name, value)
        self.parameter_changed.emit()

    def _update_widget_value(self, param_name: str, value: object):
        """Update widget when parameter changes externally."""
        if param_name in self.parameter_widgets:
            widget = self.parameter_widgets[param_name]
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def freeze_ui(self, freeze=True):
        """Disable/enable interactive elements in parameter panel"""
        for widget in self.parameter_widgets.values():
            widget.setEnabled(not freeze)
        self.reset_params_btn.setEnabled(not freeze)


class DisplacementDataPanel(QWidget):
    """Panel for handling data loading and status display."""

    data_loaded = Signal(str)  # Emits data type that was loaded
    reference_loaded = Signal(object)
    beads_loaded = Signal(object)

    def __init__(self, data_manager, viewer):
        super().__init__()
        self.data_manager = data_manager
        self.viewer = viewer
        self.controller = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create data input group
        data_group = QGroupBox("Input Data")
        group_layout = QVBoxLayout()

        # Reference data row
        ref_layout = QHBoxLayout()
        self.load_reference_btn = QPushButton("Load Reference")
        self.reference_status = QLabel("Not loaded")
        ref_layout.addWidget(self.load_reference_btn)
        ref_layout.addWidget(self.reference_status)
        group_layout.addLayout(ref_layout)

        # Bead data row
        bead_layout = QHBoxLayout()
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.bead_status = QLabel("Not loaded")
        bead_layout.addWidget(self.load_beads_btn)
        bead_layout.addWidget(self.bead_status)
        group_layout.addLayout(bead_layout)

        data_group.setLayout(group_layout)
        layout.addWidget(data_group)
        self.setLayout(layout)

    def set_controller(self, controller):
        """Set the controller and connect signals."""
        self.controller = controller
        self._connect_signals()

    def _connect_signals(self):
        """Connect UI signals to controller methods."""
        self.load_reference_btn.clicked.connect(
            lambda: self._load_data('reference')
        )
        self.load_beads_btn.clicked.connect(
            lambda: self._load_data('beads')
        )

    def _load_data(self, data_type: str):
        """Load data from active layer."""
        active_layer = self._get_active_layer()
        if active_layer is None:
            return

        try:
            data = active_layer.data

            if data_type == 'beads':
                self.data_manager.set_preprocessing_results(bead_stack=data)
                self.bead_status.setText(f"Loaded: {data.shape}")
                self.beads_loaded.emit(data)
            else:  # reference
                self.data_manager.set_preprocessing_results(reference=data)
                self.reference_status.setText(f"Loaded: {data.shape}")
                self.reference_loaded.emit(data)

            self.data_loaded.emit(data_type)

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _get_active_layer(self):
        """Get the currently active napari layer."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(
                self,
                "No Layer Selected",
                "Please select an image layer first."
            )
        return active_layer

    def freeze_ui(self, freeze=True):
        """Disable/enable interactive elements in data panel"""
        self.load_reference_btn.setEnabled(not freeze)
        self.load_beads_btn.setEnabled(not freeze)

    def update_data_status(self):
        """Update both status labels based on data manager state."""
        if self.data_manager.preprocessed_reference is not None:
            ref_shape = self.data_manager.preprocessed_reference.shape
            self.reference_status.setText(f"Loaded: {ref_shape}")
        else:
            self.reference_status.setText("Not loaded")

        if self.data_manager.preprocessed_bead_stack is not None:
            bead_shape = self.data_manager.preprocessed_bead_stack.shape
            self.bead_status.setText(f"Loaded: {bead_shape}")
        else:
            self.bead_status.setText("Not loaded")


class DisplacementActionPanel(QWidget):
    """Panel for displacement analysis actions and progress display."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create grid of button pairs
        button_layout = QVBoxLayout()

        # Row 1: Preview and Calculate
        row1_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.analyze_btn = QPushButton("Calculate All Frames")
        row1_layout.addWidget(self.preview_btn)
        row1_layout.addWidget(self.analyze_btn)
        button_layout.addLayout(row1_layout)

        # Row 2: Save and Load
        row2_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Displacements")
        self.load_btn = QPushButton("Load Displacements")
        row2_layout.addWidget(self.save_btn)
        row2_layout.addWidget(self.load_btn)
        button_layout.addLayout(row2_layout)

        layout.addLayout(button_layout)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel Operation")
        layout.addWidget(self.cancel_btn)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect action panel buttons to controller methods."""
        self.preview_btn.clicked.connect(self.controller.preview_displacement)
        self.analyze_btn.clicked.connect(self.controller.calculate_all_frames)
        self.save_btn.clicked.connect(self.controller.save_results)
        self.load_btn.clicked.connect(self.controller.load_results)
        self.cancel_btn.clicked.connect(self.controller.cancel_operation)

    def freeze_ui(self, freeze=True):
        """Disable/enable action buttons (keep cancel enabled)"""
        buttons = [
            self.preview_btn, self.analyze_btn,
            self.save_btn, self.load_btn
        ]
        for btn in buttons:
            btn.setEnabled(not freeze)
        self.cancel_btn.setEnabled(True)

    def update_button_states(self, has_reference: bool = False,
                             has_beads: bool = False,
                             has_results: bool = False):
        """Update button states based on data availability."""
        self.preview_btn.setEnabled(has_reference and has_beads)
        self.analyze_btn.setEnabled(has_reference and has_beads)
        self.save_btn.setEnabled(has_results)
        self.load_btn.setEnabled(True)


class DisplacementController(QObject):
    """Controller coordinating displacement analysis components."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)  # Results object
    analysis_failed = Signal(str)  # Error message
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    def __init__(self, viewer, service, data_manager, parameter_manager,
                 visualization_manager, data_panel):
        super().__init__()
        self.viewer = viewer
        self.service = service
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.data_panel = data_panel
        self.active_workers = []

        # Initialize panel attributes
        self.parameter_panel = None
        self.action_panel = None

    def set_panels(self, parameter_panel, action_panel):
        """Set the parameter and action panels after initialization."""
        self.parameter_panel = parameter_panel
        self.action_panel = action_panel

    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self._update_progress(0, "Calculating displacement preview...")

            # Get current frame data and parameters
            current_frame = self.viewer.dims.current_step[0]
            params = self._get_current_parameters()

            # Get reference and moving image
            reference = self.data_manager.preprocessed_reference
            moving = self.data_manager.preprocessed_bead_stack[current_frame]

            # Calculate displacement field
            result = self.service.calculate_displacement_field(reference, moving, params)

            # Update visualization
            self.visualization_manager.visualize_displacement_preview(
                result.flow,
                params.d_max,
                params.vector_stride,
                params.arrow_scale,
                downscale_factor=params.downscale_factor
            )

            # Update status with statistics
            stats = self.visualization_manager.get_displacement_statistics(result.flow)
            self._update_progress(
                100,
                f"Maximum displacement: {stats['max']:.2f} µm\n"
                f"Mean displacement: {stats['mean']:.2f} µm"
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.unfreeze_ui()

    def calculate_all_frames(self):
        """Calculate displacements for all frames."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self._update_progress(0, "Starting displacement analysis...")

            params = self._get_current_parameters()
            reference = self.data_manager.preprocessed_reference
            bead_stack = self.data_manager.preprocessed_bead_stack

            # Create and start worker
            worker = self._create_displacement_worker(reference, bead_stack, params)
            worker.running = True
            self.active_workers.append(worker)

            def on_yielded(progress_data):
                result, current_frame, total_frames = progress_data
                progress = int((current_frame + 1) / total_frames * 100)
                self._update_progress(
                    progress,
                    f"Processing frame {current_frame + 1}/{total_frames}..."
                )

            def on_returned(results):
                # Process results
                flows = [r.flow for r in results]
                flow_stack = np.stack(flows)

                # Update data manager
                self.data_manager.set_displacement_results(
                    flow_stack,
                    {
                        'pixel_size': params.pixel_size,
                        'frame_interval': params.frame_interval,
                        'downscale_factor': params.downscale_factor,
                        'visualization_params': {
                            'd_max': params.d_max,
                            'vector_stride': params.vector_stride,
                            'arrow_scale': params.arrow_scale
                        }
                    }
                )

                # Update visualization
                self.visualization_manager.visualize_displacement_results({
                    'flows': flow_stack,
                    'parameters': params,
                    'original_shape': results[0].original_shape,
                    'displacement_field_shape': results[0].displacement_field_shape,
                    'units': 'micrometers'
                })

                self._update_progress(100, "Analysis completed successfully")
                self.analysis_completed.emit(results)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            def on_errored(exc):
                error_msg = f"Analysis failed: {str(exc)}"
                self._update_progress(0, error_msg)
                self.analysis_failed.emit(error_msg)
                QMessageBox.critical(None, "Error", error_msg)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            worker.yielded.connect(on_yielded)
            worker.returned.connect(on_returned)
            worker.errored.connect(on_errored)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    @thread_worker
    def _create_displacement_worker(self, reference, bead_stack, params):
        """Thread worker for displacement calculation."""
        generator = self.service.calculate_flow_stack(
            reference=reference,
            bead_stack=bead_stack,
            params=params
        )

        try:
            while True:
                yield next(generator)
        except StopIteration as e:
            return e.value

    def save_results(self):
        """Save displacement results to file."""
        try:
            # Implement save functionality similar to MSM widget
            pass
        except Exception as e:
            self._handle_error(str(e))

    def load_results(self):
        """Load displacement results from file."""
        try:
            # Implement load functionality similar to MSM widget
            pass
        except Exception as e:
            self._handle_error(str(e))

    def cancel_operation(self):
        """Cancel all running operations."""
        for worker in self.active_workers:
            try:
                worker.running = False
                worker.quit()
                worker.wait(500)
                if worker.isRunning():
                    worker.terminate()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()
        self._update_progress(0, "Operations cancelled")
        self.unfreeze_ui()

    def _get_current_parameters(self) -> DisplacementParameters:
        """Get current parameters from parameter manager."""
        return DisplacementParameters(
            tau=self.parameter_manager.get_value('tau'),
            lambda_=self.parameter_manager.get_value('lambda_'),
            theta=self.parameter_manager.get_value('theta'),
            nscales=self.parameter_manager.get_value('nscales'),
            warps=self.parameter_manager.get_value('warps'),
            epsilon=self.parameter_manager.get_value('epsilon'),
            inner_iterations=self.parameter_manager.get_value('inner_iterations'),
            outer_iterations=self.parameter_manager.get_value('outer_iterations'),
            scale_step=self.parameter_manager.get_value('scale_step'),
            median_filtering=self.parameter_manager.get_value('median_filtering'),
            downscale_factor=self.parameter_manager.get_value('downscale_factor'),
            pixel_size=self.parameter_manager.get_value('pixel_size'),
            frame_interval=self.parameter_manager.get_value('frame_interval'),
            d_max=self.parameter_manager.get_value('d_max'),
            vector_stride=self.parameter_manager.get_value('disp_vector_stride'),
            arrow_scale=self.parameter_manager.get_value('disp_arrow_scale')
        )

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.preprocessed_reference is None:
            QMessageBox.warning(None, "Warning", "No reference image loaded")
            return False
        if self.data_manager.preprocessed_bead_stack is None:
            QMessageBox.warning(None, "Warning", "No bead stack loaded")
            return False
        return True

    def _update_progress(self, progress: int, message: str):
        """Update progress information."""
        self.progress_updated.emit(progress, message)

    def _handle_error(self, error_msg: str):
        """Handle errors uniformly."""
        self._update_progress(0, f"Error: {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)

    def freeze_ui(self):
        """Disable all interactive UI elements."""
        if self.data_panel:
            self.data_panel.freeze_ui(True)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(True)
        if self.action_panel:
            self.action_panel.freeze_ui(True)
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Re-enable UI elements and refresh state."""
        if self.data_panel:
            self.data_panel.freeze_ui(False)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(False)
        if self.action_panel:
            self.action_panel.freeze_ui(False)
        self.ui_frozen.emit(False)


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements using optical flow."""

    displacement_calculated = Signal(object)

    def __init__(self, viewer: Viewer, data_manager: DataManager,
                 parameter_manager: ParameterManager,
                 visualization_manager: VisualizationManager):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize service and managers
        self.service = DisplacementService()
        self.parameter_panel = DisplacementParameterPanel(parameter_manager)
        self.data_panel = DisplacementDataPanel(data_manager, viewer)

        # Initialize controller
        self.controller = DisplacementController(
            viewer=viewer,
            service=self.service,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
            data_panel=self.data_panel
        )

        # Initialize action panel
        self.action_panel = DisplacementActionPanel(self.controller)

        # Set panels in controller
        self.controller.set_panels(self.parameter_panel, self.action_panel)

        # Connect data panel to controller
        self.data_panel.set_controller(self.controller)

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left side: Colorbar
        colorbar_container = QWidget()
        colorbar_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(0, 0, 0, 0)

        self.colorbar_manager = ColorbarManager()
        colorbar_group = self.create_colorbar_widget(
            colormap_name='viridis',
            label="Displacement (µm)",
            clim=(0, 10),
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        colorbar_layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        colorbar_container.setLayout(colorbar_layout)

        main_layout.addWidget(colorbar_container)

        # Right side: Control panels
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Add panels
        right_layout.addWidget(self.data_panel)
        right_layout.addWidget(self.parameter_panel)
        right_layout.addWidget(self.action_panel)

        # Add status panel
        status_frame = QFrame()
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)
        right_layout.addWidget(status_frame)

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(360)

        main_layout.addWidget(right_container)

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)

        # Connect data panel signals
        self.data_panel.data_loaded.connect(self._on_data_loaded)

        # Connect parameter panel changes
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.displacement_field is not None:
            self.visualization_manager.update_displacement_frame(
                self.viewer.dims.current_step[0]
            )

    def _on_data_loaded(self, data_type: str):
        """Handle data loading events."""
        self.data_panel.update_data_status()
        self.action_panel.update_button_states(
            has_reference=self.data_manager.preprocessed_reference is not None,
            has_beads=self.data_manager.preprocessed_bead_stack is not None,
            has_results=self.data_manager.displacement_field is not None
        )

    def _on_parameter_changed(self):
        """Handle parameter changes."""
        if hasattr(self, 'preview_active') and self.preview_active:
            self.controller.preview_displacement()

    def _on_analysis_completed(self, results):
        """Handle completed analysis."""
        self.displacement_calculated.emit(results)

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_status(0, f"Analysis failed: {error_msg}")

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()
