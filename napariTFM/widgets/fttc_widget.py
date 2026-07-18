import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QMessageBox, QHBoxLayout

from napariTFM.backend.fttc import FTTCResult, calculate_force_field, find_optimal_regularization
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager
from napariTFM.widgets._base_widget import VectorStageController, BaseAnalysisWidget


class FTTCController(VectorStageController):
    """FTTC stage: traction forces from the displacement field.

    A thin :class:`VectorStageController` subclass — the run/cancel lifecycle is
    inherited; this class supplies the stage spec, the run hooks, the synchronous
    preview, and the (force-only) synchronous GCV auto-regularization action.
    """

    STAGE_KIND = 'force'
    RESULT_SETTER = 'set_force_results'

    # region === Run lifecycle hooks (template lives in the base) ===
    def _validate(self):
        if self.data_manager.displacement_results is None:
            raise ValueError("No displacement data loaded")

    def _run_params(self):
        return self.parameter_manager.get_fttc_parameters()

    def _stream_frame_count(self):
        return self.data_manager.displacement_results.displacement_field.shape[0]

    def _vis_params(self, params):
        return {
            'v_max': params.f_max,
            'vector_stride': params.force_vector_stride,
            'arrow_scale': params.force_arrow_scale,
            'downscale_factor': params.downscale_factor,
        }

    def _build_worker(self, params):
        return self._run_worker(
            self.data_manager.displacement_results.displacement_field, params,
            self._support_mask(params),
        )

    def _support_mask(self, params, frame=None):
        """The support mask for the mask-consuming inversions, or None.

        Reuses the same externally-loaded mask the Stress stage consumes
        (``data_manager.mask_stack``). It is read only when Mask Confinement > 0:
        by the confined forward solver, or (when L1 Sparsity > 0 as well) by the
        sparse L1 solver as a hard support. With confinement off the mask is
        withheld, so L1 runs as pure sparsity, as if no mask were loaded. Plain
        FTTC never reads it.

        ``frame`` selects a single mask slice for a single-frame solve (the
        preview): the backend indexes the mask by the *displacement stack* frame,
        so a preview — which passes a 1-frame stack — must hand it the matching
        mask frame, not the whole stack (else it always uses frame 0). The run
        path passes no frame and gets the full stack, aligned frame-by-frame.
        """
        if params.fwd_mask_strength <= 0:
            return None
        mask = getattr(self.data_manager, "mask_stack", None)
        if mask is None or frame is None:
            return mask
        mask = np.asarray(mask)
        if mask.ndim > 2:
            return mask[min(int(frame), mask.shape[0] - 1)]
        return mask

    @thread_worker
    def _run_worker(self, displacement_field, params, mask=None):
        """Process every frame, yielding each force field for live streaming."""
        try:
            gen = calculate_force_field(displacement_field, params, mask=mask)
            try:
                while True:
                    force_field, frame, total = next(gen)
                    # The backend yields a 1-based frame number; the stack index
                    # (and the slider) want it 0-based.
                    yield frame - 1, total, force_field
            except StopIteration as e:
                return e.value
        except Exception as e:
            raise ValueError(f"Force calculation failed: {str(e)}")
    # endregion

    # region === Preview & GCV (synchronous, GUI thread) ===
    def preview_force(self):
        """Preview force calculation for the current frame (synchronous)."""
        try:
            self._validate()
        except Exception as e:
            QMessageBox.warning(None, "Warning", str(e))
            return

        self.freeze_ui()
        try:
            self.progress_updated.emit(0, "Calculating force preview...")
            current_frame = self._current_frame()

            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]
            params = self.parameter_manager.get_fttc_parameters()

            result = self._compute_preview(displacement_field, params, current_frame)
            if result is None:
                raise RuntimeError("Preview calculation failed to produce results")

            self.visualization_manager.visualize_force_preview(
                result.force_field[0],
                result.parameters.f_max,
                result.parameters.force_vector_stride,
                result.parameters.force_arrow_scale,
                downscale_factor=result.parameters.downscale_factor,
            )

            # Show only the force layers (magnitude below, vectors on top).
            self.visualization_manager.bring_layers_to_front([
                ('Force Magnitude', True),
                ('Force Vectors', True),
            ])

            magnitude = np.sqrt(np.sum(result.force_field[0] ** 2, axis=-1))
            self.progress_updated.emit(
                100,
                f"Preview statistics:\n"
                f"Max force: {np.max(magnitude):.2f} Pa\n"
                f"Mean force: {np.mean(magnitude):.2f} Pa",
            )
        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.unfreeze_ui()

    def _compute_preview(self, displacement_field, params, frame=None) -> FTTCResult:
        """Run the single-frame force calculation to completion and return it.

        ``frame`` is the source stack index being previewed; it selects the
        matching mask slice for the forward method's confinement prior.
        """
        gen = calculate_force_field(displacement_field[np.newaxis, ...], params,
                                    mask=self._support_mask(params, frame))
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def calculate_optimal_regularization(self):
        """Compute the GCV-optimal regularization for the current frame (synchronous)."""
        try:
            self._validate()
        except Exception as e:
            QMessageBox.warning(None, "Warning", str(e))
            return

        self.freeze_ui()
        try:
            self.progress_updated.emit(0, "Calculating optimal regularization...")
            current_frame = self._current_frame()

            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]
            params = self.parameter_manager.get_fttc_parameters()

            optimal_reg = find_optimal_regularization(displacement_field, params)
            if optimal_reg is None:
                raise RuntimeError("GCV calculation failed")

            # Store the actual optimal value; the UI converts it to the exponent for display.
            self.parameter_manager.set_parameter('regularization', optimal_reg)
            self.progress_updated.emit(
                100, f"Optimal regularization parameter: {optimal_reg:.2e}"
            )
        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.unfreeze_ui()
    # endregion


class FTTCWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method."""

    force_calculated = Signal(FTTCResult)

    def __init__(
            self,
            viewer: Viewer,
            data_manager: DataManager,
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, parameter_manager, visualization_manager)

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

    def run_action(self):
        self.controller.run()

    def preview_action(self):
        self.controller.preview_force()

    def cancel_action(self):
        self.controller.cancel()

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

    def _on_analysis_completed(self, results: FTTCResult):
        """Handle completed analysis.

        Preview-only: result held in memory and shown in napari;
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
