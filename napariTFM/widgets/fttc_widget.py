import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QObject
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QMessageBox, QHBoxLayout

from napariTFM.backend.fttc import FTTCResult, calculate_force_field, find_optimal_regularization
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager
from napariTFM.widgets._base_widget import BaseAnalysisWidget


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

        # Live-streaming progress counters: a run fills the magnitude stack and
        # vector cache frame by frame, so progress is "frames done / total".
        self._stream_total = 0
        self._stream_done = 0

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

            # Bind the force magnitude + vectors layers up front so each frame
            # streams in live (preserving the contrast/visibility the user set)
            # and the slider follows it, rather than landing all at once.
            self._begin_stream(displacement_field, params)

            worker = self._create_force_worker(displacement_field, params)

            self.active_workers.append(worker)
            self.analysis_started.emit()

            worker.yielded.connect(self._on_frame_processed)
            worker.returned.connect(self._handle_analysis_results)
            worker.errored.connect(self._handle_error)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def _begin_stream(self, displacement_field, params):
        """Allocate the live force layers and reset the frame counters."""
        self._stream_total = displacement_field.shape[0]
        self._stream_done = 0
        vis_params = {
            'v_max': params.f_max,
            'vector_stride': params.force_vector_stride,
            'arrow_scale': params.force_arrow_scale,
            'downscale_factor': params.downscale_factor,
        }
        self.visualization_manager.begin_vector_field_stream(
            'force', self._stream_total, vis_params
        )

    def _on_frame_processed(self, payload):
        """Stream one freshly computed force frame into the viewer (GUI thread)."""
        frame_index, total, force_frame = payload
        self._stream_done += 1
        progress = int(self._stream_done / max(self._stream_total, 1) * 100)
        self.progress_updated.emit(
            progress, f"Processing frame {frame_index + 1}/{total}"
        )
        self.visualization_manager.stream_vector_field_frame(
            'force', frame_index, force_frame
        )

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
        """Process every frame, yielding each force field for live streaming."""
        try:
            force_generator = calculate_force_field(displacement_field, params)

            # Process all frames through the generator, streaming each one.
            try:
                while True:
                    force_field, frame, total = next(force_generator)
                    # The backend yields a 1-based frame number; the stack index
                    # (and the slider) want it 0-based.
                    yield frame - 1, total, force_field
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
                elif self.visualization_manager.colorbar_manager.is_colorbar_layer(layer.name):
                    # Keep the scale legend visible alongside the preview.
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
        """Finalize a streamed run.

        The magnitude stack and vector cache were filled in place as frames
        arrived and are already on screen, so just store the full result for
        downstream steps and announce completion. Visibility of other layers is
        left untouched (the user's to set).
        """
        try:
            if result is None:
                raise RuntimeError("Analysis failed to produce results")

            # Store the full result; the live layers already reflect it.
            self.data_manager.set_force_results(result, source="generated", dirty=True)

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
        self._action_enabled = {
            "run": False, "preview": False, "cancel": True, "gcv": False,
        }

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
        """Set up the user interface.

        All stage actions — run/preview/cancel plus GCV auto-select — live in
        the stage header (P7), so this widget owns no visible body content.
        """
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals. Progress flows to the shell's one global
        # status label (P2); no local status slot.
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Connect to layer selection changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def action_states(self):
        return dict(self._action_enabled)

    def run_action(self):
        self.controller.calculate_forces()

    def preview_action(self):
        self.controller.preview_force()

    def cancel_action(self):
        self.controller.cancel_operation()

    def gcv_action(self):
        self.controller.calculate_optimal_regularization()

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_displacement = self.data_manager.displacement_results is not None

        self._action_enabled["preview"] = has_displacement
        self._action_enabled["run"] = has_displacement
        self._action_enabled["gcv"] = has_displacement
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        self._action_enabled["preview"] = not frozen
        self._action_enabled["run"] = not frozen
        self._action_enabled["gcv"] = not frozen
        # Cancel action always enabled
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _on_analysis_completed(self, results: FTTCResult):
        """Handle completed analysis.

        Preview-only (ROADMAP §4): result held in memory and shown in napari;
        nothing written to disk. Batch is the only path to persisted data.
        """
        self.force_calculated.emit(results)
        self._update_ui_state()

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self.controller.progress_updated.emit(0, f"Error: {error_msg}")
        self._update_ui_state()

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.force_results is not None:
            self.visualization_manager.update_force_frame(
                self.viewer.dims.current_step[0]
            )

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()
