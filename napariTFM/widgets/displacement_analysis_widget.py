import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QMessageBox

from napariTFM.widgets._base_widget import VectorStageController, BaseAnalysisWidget
from napariTFM.backend.displacement_analysis import calculate_displacement_field
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager


class DisplacementController(VectorStageController):
    """Displacement stage: dense optical-flow displacement over the bead stack.

    A thin :class:`VectorStageController` subclass — the run/cancel lifecycle is
    inherited; this class only supplies the stage's spec (kind, result setter),
    the run hooks, and the synchronous preview.
    """

    STAGE_KIND = 'displacement'
    RESULT_SETTER = 'set_displacement_results'

    # region === Run lifecycle hooks (template lives in the base) ===
    def _validate(self):
        if self.data_manager.preprocessed_reference is None:
            raise ValueError("No reference image loaded")
        if self.data_manager.preprocessed_bead_stack is None:
            raise ValueError("No bead stack loaded")

    def _run_params(self):
        return self.parameter_manager.get_displacement_parameters()

    def _stream_frame_count(self):
        return self.data_manager.preprocessed_bead_stack.shape[0]

    def _vis_params(self, params):
        return {
            'v_max': params.d_max,
            'vector_stride': params.disp_vector_stride,
            'arrow_scale': params.disp_arrow_scale,
            'downscale_factor': params.downscale_factor,
        }

    def _build_worker(self, params):
        return self._run_worker(
            self.data_manager.preprocessed_reference,
            self.data_manager.preprocessed_bead_stack,
            params,
        )

    @thread_worker
    def _run_worker(self, reference, bead_stack, params):
        """Process every frame, yielding each displacement field for live streaming."""
        try:
            gen = calculate_displacement_field(reference, bead_stack, params)
            try:
                while True:
                    displacement_field, frame, total = next(gen)
                    # The backend yields a 1-based frame number; the stack index
                    # (and the slider) want it 0-based.
                    yield frame - 1, total, displacement_field
            except StopIteration as e:
                return e.value
        except Exception as e:
            raise ValueError(f"Displacement calculation failed: {str(e)}")
    # endregion

    # region === Preview (synchronous, GUI thread) ===
    def preview_displacement(self):
        """Preview displacement calculation on the current frame."""
        try:
            self._validate()
        except Exception as e:
            QMessageBox.warning(None, "Warning", str(e))
            return

        self.freeze_ui()
        try:
            self.progress_updated.emit(0, "Calculating displacement preview...")
            current_frame = self._current_frame()

            moving = self.data_manager.preprocessed_bead_stack[current_frame]
            reference = self.data_manager.preprocessed_reference
            params = self.parameter_manager.get_displacement_parameters()

            gen = calculate_displacement_field(reference, moving, params)
            final_result = None
            try:
                while True:
                    _, frame, total = next(gen)
                    self.progress_updated.emit(
                        int((frame + 1) / total * 100), "Processing preview frame..."
                    )
            except StopIteration as e:
                final_result = e.value

            if final_result is None:
                raise RuntimeError("Preview calculation failed")

            self.analysis_completed.emit(final_result)

            self.visualization_manager.visualize_displacement_preview(
                final_result.displacement_field[0],  # Single frame result
                params.d_max,
                params.disp_vector_stride,
                params.disp_arrow_scale,
                downscale_factor=params.downscale_factor,
            )

            # Show only the displacement layers (magnitude below, vectors on top).
            self.visualization_manager.bring_layers_to_front([
                ('Displacement Magnitude', True),
                ('Displacement Vectors', True),
            ])

            stats = self.get_displacement_statistics(final_result.displacement_field[0])
            self.progress_updated.emit(
                100,
                f"Maximum displacement: {stats['max']:.2f} µm\n"
                f"Mean displacement: {stats['mean']:.2f} µm",
            )
        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.unfreeze_ui()

    def get_displacement_statistics(self, flow: np.ndarray) -> dict:
        """Calculate displacement statistics."""
        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
        return {
            'max': magnitude.max(),
            'mean': magnitude.mean(),
            'std': magnitude.std(),
            'median': np.median(magnitude),
        }
    # endregion

    # region === Data Management ===
    def load_active_layer(self, data_type: str):
        """Load the currently active layer as the specified data type."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(None, "Error", "No active image layer found")
            return

        try:
            data = active_layer.data

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

            self.data_updated.emit(data_type)

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))
    # endregion


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
        super().__init__(viewer, data_manager, parameter_manager, visualization_manager)

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
        """Set up the user interface.

        All stage actions (run/preview/cancel) live on the stage header, so this
        widget owns no visible body content — an empty layout that collapses to
        zero height, matching the force stage.
        """
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

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
    def run_action(self):
        self.controller.run()

    def preview_action(self):
        self.controller.preview_displacement()

    def cancel_action(self):
        self.controller.cancel()

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_reference = self.data_manager.preprocessed_reference is not None
        has_beads = self.data_manager.preprocessed_bead_stack is not None

        can_analyze = has_reference and has_beads
        self._action_enabled["preview"] = can_analyze
        self._action_enabled["run"] = can_analyze
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def load_active_layer(self, data_type: str):
        """Delegate input-layer loading to the controller (called by the shell)."""
        self.controller.load_active_layer(data_type)

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
