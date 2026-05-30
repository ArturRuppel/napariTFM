import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QObject
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (QPushButton, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout,
                            QSizePolicy, QProgressBar, QLabel, QFrame, QSpacerItem)
from qtpy.QtWidgets import QFileDialog

from napariTFM.backend.fttc import FTTCResult, calculate_force_field, find_optimal_regularization
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager
from napariTFM.widgets._base_widget import BaseAnalysisWidget
from napariTFM.widgets._output_directory import ensure_output_dir_for_generated_artifacts


class FTTCController(QObject):
    """Controller coordinating FTTC analysis components."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)
    analysis_failed = Signal(str)
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    def __init__(self, viewer, data_manager, parameter_manager,
                 visualization_manager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []

    def preview_force(self):
        """Preview force calculation for current frame."""

        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Calculating force preview...")

            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]

            params = self.parameter_manager.get_fttc_parameters()

            # Create worker for processing
            worker = self._create_preview_worker(displacement_field, params)
            worker.returned.connect(self._handle_preview_results)
            worker.errored.connect(self._handle_error)
            worker.finished.connect(lambda: self.unfreeze_ui())

            self.active_workers.append(worker)
            worker.start()


        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def calculate_forces(self):
        """Calculate forces for all frames."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Starting force calculation...")

            params = self.parameter_manager.get_fttc_parameters()

            # Create worker for processing
            displacement_field = self.data_manager.displacement_results.displacement_field
            worker = self._create_force_worker(displacement_field, params)

            self.active_workers.append(worker)
            self.analysis_started.emit()

            worker.yielded.connect(self._handle_progress)
            worker.returned.connect(self._handle_analysis_results)
            worker.errored.connect(self._handle_error)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def calculate_optimal_regularization(self):
        """Calculate optimal regularization parameter using GCV."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Calculating optimal regularization...")

            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]

            # Create worker for GCV calculation
            worker = self._create_gcv_worker(displacement_field)

            worker.returned.connect(self._handle_gcv_results)
            worker.errored.connect(self._handle_error)
            worker.finished.connect(lambda: self.unfreeze_ui())

            self.active_workers.append(worker)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def cancel_operation(self):
        """Cancel any running operations."""
        for worker in self.active_workers:
            try:
                worker.quit()
                worker.wait()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()
        self.progress_updated.emit(0, "Operation cancelled")
        self.unfreeze_ui()

    @thread_worker
    def _create_preview_worker(self, displacement_field, params):
        """Create worker for preview calculation."""
        try:
            result = calculate_force_field(displacement_field[np.newaxis, ...], params)
            # Process generator to get result
            try:
                while True:
                    next(result)
            except StopIteration as e:
                return e.value

        except Exception as e:
            raise ValueError(f"Preview calculation failed: {str(e)}")

    @thread_worker
    def _create_force_worker(self, displacement_field, params):
        """Create worker for full force calculation."""
        try:
            force_generator = calculate_force_field(displacement_field, params)

            # Process all frames through the generator
            try:
                while True:
                    force_field, frame, total = next(force_generator)
                    yield {
                        'progress': frame / total * 100,
                        'message': f"Processing frame {frame}/{total}"
                    }
            except StopIteration as e:
                # Return the final result from the generator
                return e.value

        except Exception as e:
            raise ValueError(f"Force calculation failed: {str(e)}")

    @thread_worker
    def _create_gcv_worker(self, displacement_field):
        """Create worker for GCV calculation."""
        try:
            params = self.parameter_manager.get_fttc_parameters()
            optimal_reg = find_optimal_regularization(displacement_field, params)
            return optimal_reg
        except Exception as e:
            raise ValueError(f"GCV calculation failed: {str(e)}")

    def _handle_preview_results(self, result: FTTCResult):
        """Handle preview calculation results."""
        try:
            if result is None:
                raise RuntimeError("Preview calculation failed to produce results")

            # Update visualization for preview
            self.visualization_manager.visualize_force_preview(
                result.force_field[0],
                result.parameters.f_max,
                result.parameters.force_vector_stride,
                result.parameters.force_arrow_scale,
                downscale_factor=result.parameters.downscale_factor
            )

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the force layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Force Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Force Magnitude':
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

            # Calculate and show statistics
            magnitude = np.sqrt(np.sum(result.force_field[0] ** 2, axis=-1))
            self.progress_updated.emit(
                100,
                f"Preview statistics:\n"
                f"Max force: {np.max(magnitude):.2f} Pa\n"
                f"Mean force: {np.mean(magnitude):.2f} Pa"
            )

        except Exception as e:
            self._handle_error(str(e))

    def _handle_analysis_results(self, result: FTTCResult):
        """Handle complete analysis results."""
        try:
            if result is None:
                raise RuntimeError("Analysis failed to produce results")

            # Update data manager
            self.data_manager.set_force_results(result, source="generated", dirty=True)

            # Update visualization
            self.visualization_manager.visualize_force_results()

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the force layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Force Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Force Magnitude':
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

    def _handle_gcv_results(self, regularization: float):
        """Handle GCV calculation results."""
        if regularization is not None:
            # Store the actual optimal value; the UI converts it to the exponent for display.
            self.parameter_manager.set_parameter('regularization', regularization)
            self.progress_updated.emit(
                100,
                f"Optimal regularization parameter: {regularization:.2e}"
            )
        else:
            self._handle_error("GCV calculation failed")

    def _handle_progress(self, progress_info: dict):
        """Handle progress updates."""
        self.progress_updated.emit(
            progress_info['progress'],
            progress_info['message']
        )

    def _handle_error(self, error_msg: str):
        """Handle errors during processing."""
        self.progress_updated.emit(0, f"Error: {error_msg}")
        self.analysis_failed.emit(error_msg)
        QMessageBox.critical(None, "Error", error_msg)

    def _validate_prerequisites(self) -> bool:
        """Check if required data and parameters are available."""
        if self.data_manager.displacement_results is None:
            QMessageBox.warning(None, "Warning", "No displacement data loaded")
            return False
        return True

    def freeze_ui(self):
        """Signal the owning widget to disable interactive controls."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal the owning widget to re-enable controls."""
        self.ui_frozen.emit(False)


class FTTCWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method."""

    force_calculated = Signal(FTTCResult)
    action_states_changed = Signal()

    def __init__(
            self,
            viewer: Viewer,
            data_manager: DataManager,
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Store managers
        self.parameter_manager = parameter_manager

        # Header-proxied action enablement state
        self._action_enabled = {"run": False, "preview": False, "cancel": True}

        # Initialize controller
        self.controller = FTTCController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
        )

        # Set up UI
        self._setup_ui()

        # Connect signals
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

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
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_action_row())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        return container

    def _create_action_row(self) -> QWidget:
        """Build widget-owned action buttons (run/preview/cancel proxied by the stage header)."""
        container = QWidget()
        layout = QVBoxLayout()

        self.gcv_btn = QPushButton("Auto-select Regularization (GCV)")
        self.gcv_btn.setToolTip(
            "Calculate the optimal regularization parameter for the current frame\n"
            "using Generalized Cross-Validation"
        )
        layout.addWidget(self.gcv_btn)

        container.setLayout(layout)
        return container

    def _create_status_frame(self) -> QFrame:
        """Create the status display frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Wire widget-owned action button to controller operation
        self.gcv_btn.clicked.connect(self.controller.calculate_optimal_regularization)

        # Connect to layer selection changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def action_states(self):
        return dict(self._action_enabled)

    def run_action(self):
        self.controller.calculate_forces()

    def preview_action(self):
        self.controller.preview_force()

    def cancel_action(self):
        self.controller.cancel_operation()

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_displacement = self.data_manager.displacement_results is not None

        self._action_enabled["preview"] = has_displacement
        self._action_enabled["run"] = has_displacement
        self.gcv_btn.setEnabled(has_displacement)
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        self._action_enabled["preview"] = not frozen
        self._action_enabled["run"] = not frozen
        self.gcv_btn.setEnabled(not frozen)
        # Cancel action always enabled
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _on_analysis_completed(self, results: FTTCResult):
        """Handle completed analysis."""
        if ensure_output_dir_for_generated_artifacts(self, self.data_manager):
            try:
                self.data_manager.auto_save_artifact("force_results")
            except Exception as exc:
                self.data_manager.mark_artifact_error("force_results", str(exc))
                QMessageBox.warning(self, "Auto-save Failed", str(exc))

        # Emit results
        self.force_calculated.emit(results)
        self._update_ui_state()

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_status(0, f"Error: {error_msg}")
        self._update_ui_state()

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.force_results is not None:
            self.visualization_manager.update_force_frame(
                self.viewer.dims.current_step[0]
            )

    def load_result_artifact(self, key: str):
        path = self._choose_result_path(key)
        if not path:
            return
        self.data_manager.load_result_artifact(key, path)
        show_displacement = getattr(self.visualization_manager, "visualize_displacement_results", None)
        if key == "displacement_results" and show_displacement is not None:
            show_displacement()
        self._update_ui_state()

    def _choose_result_path(self, key: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {key.replace('_', ' ')}",
            "",
            "NumPy Files (*.npy)",
        )
        return path

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()
