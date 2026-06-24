from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QObject
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFrame, QCheckBox, QApplication,
    QMessageBox, QSizePolicy
)
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QWidget
)

from napariTFM.widgets._base_widget import BaseAnalysisWidget
from napariTFM.utilities.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.backend.preprocessing import preprocess_frame, preprocess_stack

class PreprocessingController(QObject):
    """Controller coordinating UI components and data processing."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    preprocessing_started = Signal()
    preprocessing_completed = Signal(dict)  # Results dictionary
    preprocessing_failed = Signal(str)  # Error message
    preview_updated = Signal()
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    # region === Initialization
    def __init__(self, viewer, data_manager, parameter_manager, visualization_manager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []

        self.preview_enabled = False

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)

        # Connect to viewer events for frame changes
        self.connect_viewer_events()

    def connect_viewer_events(self):
        """Connect to viewer dimension events for frame changes."""
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

    def _on_frame_changed(self, event=None):
        """Handle frame change events from the viewer."""
        if self.preview_enabled:
            self._update_preview()

    # endregion === Initialization

    # region === Processing Execution
    def run_preprocessing(self):
        """Execute preprocessing on loaded data."""
        try:
            if self.data_manager.bead_stack is None:
                raise ValueError("Bead stack must be loaded before preprocessing")
            if self.data_manager.reference is None:
                raise ValueError("Reference image must be loaded before preprocessing")

            worker = self._create_processing_worker()
            self.active_workers.append(worker)
            self.preprocessing_started.emit()
            self.freeze_ui()

            worker.yielded.connect(lambda x: self.progress_updated.emit(*x))
            worker.returned.connect(self._handle_preprocessing_results)
            worker.errored.connect(self._handle_preprocessing_error)
            worker.start()

        except Exception as e:
            self.preprocessing_failed.emit(str(e))
            QMessageBox.critical(None, "Error", str(e))
            self.unfreeze_ui()

    @thread_worker
    def _create_processing_worker(self):
        """Create worker for processing data."""
        params = self.parameter_manager.get_preprocessing_parameters()

        results = []

        # Process bead stack with generator
        if self.data_manager.bead_stack is not None:
            for result, frame, total in preprocess_stack(
                    image_stack=self.data_manager.bead_stack,
                    params=params,
                    reference_image=self.data_manager.reference
            ):
                results.append(result)
                yield frame / total * 100, f"Processing beads: Frame {frame + 1}/{total}"

        # Process reference image
        if self.data_manager.reference is not None:
            results.append(preprocess_frame(self.data_manager.reference, params))

        # Process cell stack if available
        if self.data_manager.cell_stack is not None:
            start_progress = len(results)
            for result, frame, total in preprocess_stack(
                    image_stack=self.data_manager.cell_stack,
                    params=params,
                    is_cell=True
            ):
                results.append(result)
                yield start_progress + frame / total * 100, f"Processing cells: Frame {frame + 1}/{total}"

        return results

    def cancel_all_operations(self):
        """Cancel all running background operations."""
        if not self.active_workers:
            # No active workers, just update status
            self.progress_updated.emit(0, "No active operations to cancel")
            return

        for worker in self.active_workers:
            try:
                worker.quit()  # This should be sufficient for napari workers
            except Exception as e:
                print(f"Warning: Could not quit worker cleanly: {str(e)}")

        self.active_workers.clear()

        # Update UI status and ensure responsiveness
        self.progress_updated.emit(0, "Operations cancelled")
        QApplication.processEvents()
        self.unfreeze_ui()

    # endregion === Processing Execution

    # region === Parameter Handling
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        if self.preview_enabled:
            self._update_preview()

    def _on_parameters_reset(self, category: ParameterCategory):
        """Handle parameter reset events."""
        if category == ParameterCategory.PREPROCESSING and self.preview_enabled:
            self._update_preview()

    # endregion === Parameter Handling

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
                self.data_manager.set_bead_stack(data)

            elif data_type == 'reference':
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")
                self.data_manager.set_reference(data)

            elif data_type == 'cells':
                # Convert 2D data to 3D with single frame if needed
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                elif data.ndim != 3:
                    raise ValueError("Cell stack must be 2D or 3D (frames, height, width)")
                self.data_manager.set_cell_stack(data)
            else:
                raise ValueError(f"Invalid data type: {data_type}")

            # Update UI state and emit signal
            self.data_updated.emit(data_type)
            if self.preview_enabled:
                self._update_preview()

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode."""
        try:
            self.preview_enabled = enabled

            if enabled:
                self._update_preview()
            else:
                self.visualization_manager.handle_preprocessing_preview({}, enable=False)

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))
            preview_check = getattr(self, "preview_check", None)
            if preview_check is not None:
                preview_check.setChecked(False)
            self.preview_enabled = False

    def _update_preview(self):
        """Update preview with the current frame from all loaded inputs."""
        if not self.preview_enabled:
            return

        try:
            params = self.parameter_manager.get_preprocessing_parameters()
            preview_frames = {}
            statuses = []

            for data_type, data, is_cell in (
                    ('beads', self.data_manager.bead_stack, False),
                    ('reference', self.data_manager.reference, False),
                    ('cells', self.data_manager.cell_stack, True),
            ):
                if data is None:
                    continue

                if data.ndim == 3:
                    current_step = min(self.viewer.dims.current_step[0], data.shape[0] - 1)
                    frame = data[current_step].copy()
                    frame_info = f" frame {current_step + 1}/{data.shape[0]}"
                else:
                    frame = data.copy()
                    frame_info = ""

                result = preprocess_frame(frame, params, is_cell=is_cell)
                preview_frames[data_type] = result.processed_image
                statuses.append(f"{data_type}{frame_info}")

            if not preview_frames:
                raise ValueError("No preprocessing input data available")

            self.visualization_manager.handle_preprocessing_preview(preview_frames, enable=True)
            self.progress_updated.emit(100, "Preview: " + ", ".join(statuses))

        except Exception as e:
            self.progress_updated.emit(0, f"Preview failed: {str(e)}")
            self.visualization_manager.handle_preprocessing_preview({}, enable=False)
            self.preview_enabled = False
            preview_check = getattr(self, "preview_check", None)
            if preview_check is not None:
                preview_check.blockSignals(True)
                preview_check.setChecked(False)
                preview_check.blockSignals(False)

    def _handle_preprocessing_error(self, error):
        """Handle preprocessing error."""
        error_msg = str(error)
        self.preprocessing_failed.emit(error_msg)
        self.progress_updated.emit(0, f"Error: {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)
        self.unfreeze_ui()

    def _handle_preprocessing_results(self, results):
        """Handle successful preprocessing results."""
        try:
            if results is None:
                return

            # Process all results at once
            processed_images = [r.processed_image for r in results]

            # Update data manager
            if self.data_manager.bead_stack is not None:
                n_beads = self.data_manager.bead_stack.shape[0]
                self.data_manager.set_preprocessed_bead_stack(np.stack(processed_images[:n_beads]), source="generated", dirty=True)
                processed_images = processed_images[n_beads:]

            if self.data_manager.reference is not None:
                self.data_manager.set_preprocessed_reference(processed_images[0], source="generated", dirty=True)
                processed_images = processed_images[1:]

            if self.data_manager.cell_stack is not None:
                n_cells = self.data_manager.cell_stack.shape[0]
                self.data_manager.set_preprocessed_cell_stack(np.stack(processed_images[:n_cells]), source="generated", dirty=True)

            # Update visualization
            self.visualization_manager.update_preprocessing_visualization()

            # Manage layer visibility
            preprocessing_layers = {
                'Preprocessed Beads',
                'Preprocessed Reference',
                'Preprocessed Cells',
            }
            for layer in self.viewer.layers:
                layer.visible = layer.name in preprocessing_layers

            # Get current parameters for the completion signal.
            # Preview-only (ROADMAP §4): preprocessed stacks are held in memory
            # and shown in napari; nothing is written to disk. Batch is the only
            # path to persisted data.
            current_params = self.parameter_manager.get_preprocessing_parameters()
            self.progress_updated.emit(100, "Preprocessing complete")
            self.preprocessing_completed.emit({'parameters': current_params.__dict__})

        except Exception as e:
            error_msg = f"Error handling preprocessing results: {str(e)}"
            self.preprocessing_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
        finally:
            self.unfreeze_ui()

    # endregion === Data Management

    # region === State Management
    def freeze_ui(self):
        """Signal that interactive UI elements should be disabled."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal that interactive UI elements should be re-enabled."""
        self.ui_frozen.emit(False)

    # endregion === State Management


class PreprocessingWidget(BaseAnalysisWidget):
    """Main preprocessing widget integrating all components."""

    preprocessing_completed = Signal(dict)  # Emits processed data
    action_states_changed = Signal()

    # region === Initialization
    def __init__(
            self,
            viewer: Viewer,
            data_manager,
            parameter_manager: ParameterManager,
            visualization_manager
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize managers
        self.parameter_manager = parameter_manager

        # Per-action enablement consumed by the header (StageSection)
        self._action_enabled = {"run": False, "preview": False, "cancel": True}

        # Initialize controller
        self.controller = PreprocessingController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager
        )

        # Set up UI and connections
        self._setup_ui()
        self.controller.preview_check = self.preview_check
        self._connect_signals()
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
        """Create the main content container.

        The stage widget no longer owns a scroll area or a fixed width — the
        shell's single scroll area owns layout, so the body reflows to the dock
        width (CellFlow model).
        """
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_frame = self._create_preview_frame()
        self.preview_frame.setVisible(False)
        layout.addWidget(self.preview_frame)
        self.action_frame = self._create_action_frame()
        self.action_frame.setVisible(False)
        layout.addWidget(self.action_frame)
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        return container

    def _create_preview_frame(self) -> QFrame:
        """Create compatibility preview control frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.preview_check = QCheckBox("Show Preview")
        layout.addWidget(self.preview_check)

        frame.setLayout(layout)
        return frame

    def _create_action_frame(self) -> QFrame:
        """Create the (hidden) action frame.

        Run/cancel are now driven by the header (StageSection) via the
        signal-driven action contract; this frame intentionally holds no
        buttons of its own.
        """
        frame = QFrame()
        layout = QVBoxLayout()
        frame.setLayout(layout)
        return frame

    def _create_status_frame(self) -> QFrame:
        """Create status display frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    # endregion

    # region === Signal Handling
    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect preview controls
        self.preview_check.toggled.connect(self._on_preview_toggled)

        # Run/cancel are driven by the header via the action contract
        # (run_action / cancel_action); no body button wiring here.

        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.controller.preprocessing_failed.connect(self._on_preprocessing_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)
        # Parameter-change-driven preview is handled by the controller via
        # ParameterManager.parameter_changed; no panel wiring needed here.

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _on_preview_toggled(self, enabled: bool):
        """Handle preview toggle."""
        self.controller.toggle_preview(enabled)

    def _on_process_clicked(self):
        """Handle process button click."""
        try:
            if self.preview_check.isChecked():
                self.preview_check.setChecked(False)
            self.controller.run_preprocessing()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # endregion

    # region === Action Contract
    def action_states(self):
        return dict(self._action_enabled)

    def run_action(self):
        self._on_process_clicked()

    def preview_action(self):
        self.preview_check.setChecked(not self.preview_check.isChecked())

    def cancel_action(self):
        self.controller.cancel_all_operations()

    # endregion

    # region === State Management
    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.status_label.setText(message)

    def _has_required_data(self) -> bool:
        """Check if required data for processing is loaded."""
        return (self.data_manager.bead_stack is not None and
                self.data_manager.reference is not None)

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update preview controls
        has_any_data = (
                self.data_manager.bead_stack is not None or
                self.data_manager.reference is not None or
                self.data_manager.cell_stack is not None
        )
        self.preview_check.setEnabled(has_any_data)
        self._action_enabled["preview"] = has_any_data

        # Uncheck preview if no data
        if not has_any_data and self.preview_check.isChecked():
            self.preview_check.setChecked(False)

        # Update action enablement - now uses _has_required_data()
        self._action_enabled["run"] = self._has_required_data()
        self.action_states_changed.emit()

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze."""
        # Disable preview and run action during processing
        self.preview_check.setEnabled(not frozen)
        self._action_enabled["preview"] = not frozen
        self._action_enabled["run"] = not frozen and self._has_required_data()

        # Cancel action is always enabled
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _check_preprocessed_data(self) -> bool:
        """Check availability of preprocessed data."""
        return (self.data_manager.preprocessed_bead_stack is not None or
                self.data_manager.preprocessed_reference is not None or
                self.data_manager.preprocessed_cell_stack is not None)

    # endregion

    # region === Results Handling
    def _on_preprocessing_completed(self, results):
        """Handle preprocessing completion."""
        self.preprocessing_completed.emit(results)

    def _on_preprocessing_failed(self, error_msg: str):
        """Handle preprocessing failure."""
        QMessageBox.critical(self, "Error", error_msg)

    # endregion

    # region === Cleanup
    def cleanup(self):
        """Clean up resources."""
        self.visualization_manager.cleanup()
        super().cleanup()

    # endregion
