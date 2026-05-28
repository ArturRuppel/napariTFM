import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton, QMessageBox, QSpacerItem,
    QSizePolicy, QFrame, QProgressBar
)

from napariTFM.widgets._base_widget import BaseAnalysisWidget
from napariTFM.widgets._output_directory import ensure_output_dir_for_generated_artifacts
from napariTFM.backend.displacement_analysis import (
    DisplacementResult,
    calculate_displacement_field,
)
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager

class DisplacementController(QObject):
    """Controller coordinating displacement analysis components."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)  # Results object
    analysis_failed = Signal(str)  # Error message
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    # region === Initialization
    def __init__(self, viewer, data_manager, parameter_manager,
                 visualization_manager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []

    # endregion === Initialization

    # region === Processing Execution
    def get_displacement_statistics(self, flow: np.ndarray) -> dict:
        """Calculate displacement statistics."""
        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
        return {
            'max': magnitude.max(),
            'mean': magnitude.mean(),
            'std': magnitude.std(),
            'median': np.median(magnitude)
        }
    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Calculating displacement preview...")

            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            moving = self.data_manager.preprocessed_bead_stack[current_frame]
            reference = self.data_manager.preprocessed_reference

            params = self.parameter_manager.get_displacement_parameters()

            # Calculate displacement field for single frame
            result = calculate_displacement_field(reference, moving, params)

            # Process generator to get the result
            try:
                while True:
                    displacement_field, frame, total = next(result)
                    self.progress_updated.emit(
                        int((frame + 1) / total * 100),
                        f"Processing preview frame..."
                    )
            except StopIteration as e:
                # Get final result from generator
                final_result = e.value

            if final_result is None:
                raise RuntimeError("Preview calculation failed")

            self.analysis_completed.emit(final_result)

            # Update visualization with preview result
            self.visualization_manager.visualize_displacement_preview(
                final_result.displacement_field[0],  # Single frame result
                params.d_max,
                params.disp_vector_stride,
                params.disp_arrow_scale,
                downscale_factor=params.downscale_factor
            )

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the displacement layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Displacement Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Displacement Magnitude':
                    magnitude_layer = layer
                    layer.visible = True
                else:
                    layer.visible = False

            # Move layers to desired positions if they exist
            if magnitude_layer is not None:
                current_index = self.viewer.layers.index(magnitude_layer)
                # Move magnitude layer to second from top (-2)
                if current_index != -2:
                    self.viewer.layers.move(current_index, -2)

            if vector_layer is not None:
                current_index = self.viewer.layers.index(vector_layer)
                # Move vector layer to top (-1)
                if current_index != -1:
                    self.viewer.layers.move(current_index, -1)

            # Update status with statistics
            stats = self.get_displacement_statistics(final_result.displacement_field[0])
            self.progress_updated.emit(
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
            self.progress_updated.emit(0, "Starting displacement analysis...")

            params = self.parameter_manager.get_displacement_parameters()

            # Create worker for processing
            worker = self._create_displacement_worker(
                self.data_manager.preprocessed_reference,
                self.data_manager.preprocessed_bead_stack,
                params
            )

            self.active_workers.append(worker)
            self.analysis_started.emit()

            worker.yielded.connect(self._handle_progress)
            worker.returned.connect(self._handle_analysis_results)
            worker.errored.connect(self._handle_error)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    @thread_worker
    def _create_displacement_worker(self, reference, bead_stack, params):
        """Create worker for processing data."""
        try:
            displacement_generator = calculate_displacement_field(reference, bead_stack, params)

            # Process all frames through the generator
            try:
                while True:
                    displacement_field, frame, total = next(displacement_generator)
                    yield {
                        'progress': (frame + 1) / total * 100,
                        'message': f"Processing frame {frame + 1}/{total}"
                    }
            except StopIteration as e:
                # Return the final result from the generator
                return e.value

        except Exception as e:
            raise ValueError(f"Displacement calculation failed: {str(e)}")

    def cancel_operation(self):
        """Cancel all running operations."""
        for worker in self.active_workers:
            try:
                worker.quit()
                worker.wait(500)
                if worker.isRunning():
                    worker.terminate()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()

        # Clear any partial results when canceling
        self.data_manager.set_displacement_results(None)

        # Update UI
        self.progress_updated.emit(0, "Operations cancelled")
        self.unfreeze_ui()

        # Force UI state update to reflect cleared results
        self.data_updated.emit('displacement')
    # endregion === Processing Execution

    # region === Data Management
    def load_active_layer(self, data_type: str):
        """Load the currently active layer as the specified data type."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(None, "Error", "No active image layer found")
            return

        try:
            data = active_layer.data

            # Handle data based on type
            if data_type == 'beads':
                # Convert 2D data to 3D with single frame if needed
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                elif data.ndim != 3:
                    raise ValueError("Bead stack must be 2D or 3D (frames, height, width)")
                self.data_manager.set_preprocessed_bead_stack(data)

            elif data_type == 'reference':
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")
                self.data_manager.set_preprocessed_reference(data)
            else:
                raise ValueError(f"Invalid data type: {data_type}")

            # Update UI state and emit signal
            self.data_updated.emit(data_type)

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.preprocessed_reference is None:
            QMessageBox.warning(None, "Warning", "No reference image loaded")
            return False
        if self.data_manager.preprocessed_bead_stack is None:
            QMessageBox.warning(None, "Warning", "No bead stack loaded")
            return False
        return True

    def _handle_analysis_results(self, result: DisplacementResult):
        """Handle completed analysis results."""
        try:
            if result is None:
                raise RuntimeError("Analysis failed to produce results")

            # Update data manager
            self.data_manager.set_displacement_results(result, source="generated", dirty=True)

            # Update visualization
            self.visualization_manager.visualize_displacement_results()

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the displacement layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Displacement Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Displacement Magnitude':
                    magnitude_layer = layer
                    layer.visible = True
                else:
                    layer.visible = False

            # Move layers to desired positions if they exist
            if magnitude_layer is not None:
                current_index = self.viewer.layers.index(magnitude_layer)
                # Move magnitude layer to second from top (-2)
                if current_index != -2:
                    self.viewer.layers.move(current_index, -2)

            if vector_layer is not None:
                current_index = self.viewer.layers.index(vector_layer)
                # Move vector layer to top (-1)
                if current_index != -1:
                    self.viewer.layers.move(current_index, -1)

            self.progress_updated.emit(100, "Analysis completed successfully")
            self.analysis_completed.emit(result)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.active_workers.clear()
            self.unfreeze_ui()

    def _handle_progress(self, progress_info: dict):
        """Handle progress updates."""
        self.progress_updated.emit(
            progress_info['progress'],
            progress_info['message']
        )

    def _handle_error(self, error_msg: str):
        """Handle errors."""
        self.progress_updated.emit(0, f"Error: {error_msg}")
        self.analysis_failed.emit(error_msg)
        QMessageBox.critical(None, "Error", error_msg)

    # endregion === Data Management

    # region === State Management
    def freeze_ui(self):
        """Signal the owning widget to disable interactive controls."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal the owning widget to re-enable controls."""
        self.ui_frozen.emit(False)

    # endregion === State Management


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements using optical flow."""

    displacement_calculated = Signal(object)

    # region === Initialization
    def __init__(
            self,
            viewer: Viewer,
            data_manager: DataManager,
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        self.parameter_manager = parameter_manager

        # Initialize controller
        self.controller = DisplacementController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
        )

        # Set up the UI
        self._setup_ui()

        # Connect signals after UI is fully set up
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

        # Initialize UI state
        self._update_ui_state()



    # endregion

    # region === UI Creation
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        content_container = self._create_content_container()
        main_layout.addWidget(content_container)

        self.setLayout(main_layout)

    def _create_content_container(self) -> QWidget:
        """Create the main content container."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        layout = QVBoxLayout()

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_action_row())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll

    def _create_action_row(self) -> QWidget:
        """Build widget-owned action buttons (proxied by the stage header)."""
        container = QWidget()
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.preview_btn.setToolTip(
            "Calculate and visualize displacement for the current frame only"
        )
        self.process_btn = QPushButton("Calculate All Frames")
        self.process_btn.setToolTip(
            "Calculate displacements for all frames in the dataset"
        )
        self.preview_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.process_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        row.addWidget(self.preview_btn)
        row.addWidget(self.process_btn)
        layout.addLayout(row)

        self.cancel_btn = QPushButton("Cancel Operation")
        self.cancel_btn.setToolTip("Cancel the current operation")
        layout.addWidget(self.cancel_btn)

        container.setLayout(layout)
        return container

    def _create_status_frame(self) -> QFrame:
        """Create the status display frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)  # Enable text wrapping
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    # endregion

    # region === Signal Handling
    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Wire widget-owned action buttons to controller operations
        self.preview_btn.clicked.connect(self.controller.preview_displacement)
        self.process_btn.clicked.connect(self.controller.calculate_all_frames)
        self.cancel_btn.clicked.connect(self.controller.cancel_operation)

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    # endregion

    # region === State Management
    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_reference = self.data_manager.preprocessed_reference is not None
        has_beads = self.data_manager.preprocessed_bead_stack is not None

        can_analyze = has_reference and has_beads
        self.preview_btn.setEnabled(can_analyze)
        self.process_btn.setEnabled(can_analyze)
        self.cancel_btn.setEnabled(True)

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        self.preview_btn.setEnabled(not frozen)
        self.process_btn.setEnabled(not frozen)
        # Cancel button always enabled
        self.cancel_btn.setEnabled(True)

    def load_active_layer(self, data_type: str):
        """Delegate input-layer loading to the controller (called by the shell)."""
        self.controller.load_active_layer(data_type)

    def _has_required_data(self) -> bool:
        """Check if required data for processing is available."""
        return (
                self.data_manager.preprocessed_reference is not None and
                self.data_manager.preprocessed_bead_stack is not None
        )

    # endregion

    # region === Results Handling

    def _on_analysis_completed(self, results):
        """Handle completed analysis."""
        self._update_ui_state()

        if ensure_output_dir_for_generated_artifacts(self, self.data_manager):
            try:
                self.data_manager.auto_save_artifact("displacement_results")
            except Exception as exc:
                self.data_manager.mark_artifact_error("displacement_results", str(exc))
                QMessageBox.warning(self, "Auto-save Failed", str(exc))

        self.displacement_calculated.emit(results)

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_ui_state()
        QMessageBox.critical(self, "Error", error_msg)

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.displacement_results is not None:
            self.visualization_manager.update_displacement_frame(
                self.viewer.dims.current_step[0]
            )

    # endregion

    # region === Cleanup
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()

    # endregion
