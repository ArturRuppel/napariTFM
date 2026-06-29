import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QSpacerItem,
    QSizePolicy
)

from napariTFM.widgets._base_widget import BaseAnalysisWidget
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

        # Live-streaming progress counters: a run fills the magnitude stack and
        # vector cache frame by frame, so progress is "frames done / total".
        self._stream_total = 0
        self._stream_done = 0

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
                        "Processing preview frame..."
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

            # Bind the magnitude + vectors layers up front so each frame can
            # stream in live (preserving the contrast/visibility the user set)
            # and the slider can follow it, instead of the whole result landing
            # only when the run finishes.
            self._begin_stream(params)

            # Create worker for processing
            worker = self._create_displacement_worker(
                self.data_manager.preprocessed_reference,
                self.data_manager.preprocessed_bead_stack,
                params
            )

            self.active_workers.append(worker)
            self.analysis_started.emit()

            worker.yielded.connect(self._on_frame_processed)
            worker.returned.connect(self._handle_analysis_results)
            worker.errored.connect(self._handle_error)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def _begin_stream(self, params):
        """Allocate the live displacement layers and reset the frame counters."""
        self._stream_total = self.data_manager.preprocessed_bead_stack.shape[0]
        self._stream_done = 0
        vis_params = {
            'v_max': params.d_max,
            'vector_stride': params.disp_vector_stride,
            'arrow_scale': params.disp_arrow_scale,
            'downscale_factor': params.downscale_factor,
        }
        self.visualization_manager.begin_vector_field_stream(
            'displacement', self._stream_total, vis_params
        )

    @thread_worker
    def _create_displacement_worker(self, reference, bead_stack, params):
        """Process every frame, yielding each displacement field for live streaming."""
        try:
            displacement_generator = calculate_displacement_field(reference, bead_stack, params)

            # Process all frames through the generator, streaming each one.
            try:
                while True:
                    displacement_field, frame, total = next(displacement_generator)
                    # The backend yields a 1-based frame number; the stack index
                    # (and the slider) want it 0-based.
                    yield frame - 1, total, displacement_field
            except StopIteration as e:
                # Return the final result from the generator
                return e.value

        except Exception as e:
            raise ValueError(f"Displacement calculation failed: {str(e)}")

    def _on_frame_processed(self, payload):
        """Stream one freshly computed displacement frame into the viewer (GUI thread)."""
        frame_index, total, flow_frame = payload
        self._stream_done += 1
        progress = int(self._stream_done / max(self._stream_total, 1) * 100)
        self.progress_updated.emit(
            progress, f"Processing frame {frame_index + 1}/{total}"
        )
        self.visualization_manager.stream_vector_field_frame(
            'displacement', frame_index, flow_frame
        )

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
        """Finalize a streamed run.

        The magnitude stack and vector cache were filled in place as frames
        arrived and are already on screen, so there is nothing left to assemble —
        just store the full result for downstream steps and announce completion.
        Visibility of other layers is left untouched (the user's to set).
        """
        try:
            if result is None:
                raise RuntimeError("Analysis failed to produce results")

            # Store the full result; the live layers already reflect it.
            self.data_manager.set_displacement_results(result, source="generated", dirty=True)

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
    action_states_changed = Signal()

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

        self._action_enabled = {"run": False, "preview": False, "cancel": True}

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
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        layout = QVBoxLayout()

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_action_row())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        container.setLayout(layout)
        return container

    def _create_action_row(self) -> QWidget:
        """Action buttons live on the stage header; this container is empty."""
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        return container

    # endregion

    # region === Signal Handling
    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals. Progress flows to the shell's one global
        # status label (P2); no local status slot.
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    # endregion

    # region === State Management
    def action_states(self):
        return dict(self._action_enabled)

    def run_action(self):
        self.controller.calculate_all_frames()

    def preview_action(self):
        self.controller.preview_displacement()

    def cancel_action(self):
        self.controller.cancel_operation()

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_reference = self.data_manager.preprocessed_reference is not None
        has_beads = self.data_manager.preprocessed_bead_stack is not None

        can_analyze = has_reference and has_beads
        self._action_enabled["preview"] = can_analyze
        self._action_enabled["run"] = can_analyze
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        self._action_enabled["preview"] = not frozen
        self._action_enabled["run"] = not frozen
        # Cancel button always enabled
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

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
        """Handle completed analysis.

        Preview-only (ROADMAP §4): the result is held in memory and shown in
        napari for parameter tuning; nothing is written to disk. Batch is the
        only path to persisted data.
        """
        self._update_ui_state()
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
