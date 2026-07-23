import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QMessageBox, QHBoxLayout

from napariTFM.backend.fttc import (
    FTTCResult, calculate_force_field, find_bayesian_regularization, find_gcv_regularization,
)
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager
from napariTFM.widgets._base_widget import VectorStageController, BaseAnalysisWidget


class FTTCController(VectorStageController):
    """FTTC stage: traction forces from the displacement field.

    A thin :class:`VectorStageController` subclass — the run/cancel lifecycle is
    inherited; this class supplies the stage spec, the run hooks, the synchronous
    preview, and the (force-only) synchronous Bayesian auto-λ action.
    """

    STAGE_KIND = 'force'
    RESULT_SETTER = 'set_force_results'

    def set_displacement_loader(self, loader) -> None:
        """Wire a callback that pulls the active experiment's displacement off disk
        into memory (data-only) if it is not resident. Injected by the shell so this
        stage can run straight from a done-on-disk displacement without it first
        being viewed. See napariTFMWidget._ensure_displacement_resident."""
        self._displacement_loader = loader

    # region === Run lifecycle hooks (template lives in the base) ===
    def _validate(self):
        # The solver reads the displacement field from memory; if it is only on disk
        # (a done experiment not yet viewed), pull it in now — the Preview/Run that
        # got here is a compute the user just asked for.
        loader = getattr(self, "_displacement_loader", None)
        if self.data_manager.displacement_results is None and loader is not None:
            loader()
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
        """The mask used for post-hoc force clipping, or None.

        Reuses the same externally-loaded mask the Stress stage consumes
        (``data_manager.mask_stack``). It is read only when Clip Outside Mask > 0.
        The selected force solver runs normally; the backend clips the completed
        force frame outside mask + radius.

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

    # region === Preview (async single-shot) & Bayesian auto-λ (synchronous, GUI thread) ===
    def preview_force(self, *, displacement_result=None, completion=None):
        """Preview force for the current frame on a worker thread.

        The single-frame inversion can spike well above its usual cost (a heavy
        forward/L1 solve, GPU contention); run inline that spell freezes the
        window, so the compute goes off-thread. GUI/data reads happen here on the
        main thread; the napari visualization returns to it in
        :meth:`_show_force_preview`.
        """
        transient_result = displacement_result is not None
        if displacement_result is None:
            try:
                self._validate()
            except Exception as e:
                QMessageBox.warning(None, "Warning", str(e))
                return False
            displacement_result = self.data_manager.displacement_results

        current_frame = self._current_frame()
        result_frame = 0 if transient_result else current_frame
        displacement_field = displacement_result.displacement_field[result_frame]
        params = self.parameter_manager.get_fttc_parameters()

        worker = self._preview_worker(displacement_field, params, current_frame)
        self._start_preview_worker(
            worker, self._show_force_preview,
            status="Calculating force preview...",
            completion=completion,
        )
        return True

    @thread_worker
    def _preview_worker(self, displacement_field, params, frame):
        """Single-frame force solve (worker thread; no napari access)."""
        result = self._compute_preview(displacement_field, params, frame)
        if result is None:
            raise RuntimeError("Preview calculation failed to produce results")
        return result

    def _show_force_preview(self, result):
        """Paint the previewed force field (GUI thread)."""
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

    def _compute_preview(self, displacement_field, params, frame=None) -> FTTCResult:
        """Run the single-frame force calculation to completion and return it.

        ``frame`` is the source stack index being previewed; it selects the
        matching mask slice for post-hoc force clipping.
        """
        gen = calculate_force_field(displacement_field[np.newaxis, ...], params,
                                    mask=self._support_mask(params, frame))
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def calculate_optimal_regularization(self):
        """Estimate the Bayesian-L2 ridge λ on the current frame and freeze it for the batch.

        The auto-λ button (synchronous). It runs the evidence maximization on this frame and
        stores the inferred λ in ``bayesian_lambda``; every frame of the experiment then
        reconstructs with that same λ (the forward operator is identical across frames, so the λ
        transfers exactly), keeping the traction maps comparable. Runs Bayesian-L2 regardless of
        whether the ``bayesian_l2`` checkbox is set, so the estimate is available to freeze.

        Uses the loaded foreground mask (if any) directly — not gated on force clipping — so BL2
        reads the noise from the cell-free exterior (the paper's more robust estimate); with no
        mask it infers the noise instead (ABL2).
        """
        try:
            self._validate()
        except Exception as e:
            QMessageBox.warning(None, "Warning", str(e))
            return

        self.freeze_ui()
        try:
            self.progress_updated.emit(0, "Estimating Bayesian regularization...")
            current_frame = self._current_frame()

            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]
            params = self.parameter_manager.get_fttc_parameters()

            mask_stack = getattr(self.data_manager, "mask_stack", None)
            mask = None
            if mask_stack is not None:
                mask_stack = np.asarray(mask_stack)
                mask = (mask_stack[min(int(current_frame), mask_stack.shape[0] - 1)]
                        if mask_stack.ndim > 2 else mask_stack)

            bayesian_lambda = find_bayesian_regularization(displacement_field, params, mask)
            if bayesian_lambda is None:
                raise RuntimeError("Bayesian regularization estimate failed")

            # Freeze the estimate; every frame reuses this λ for a comparable reconstruction.
            self.parameter_manager.set_parameter('bayesian_lambda', bayesian_lambda)
            self.parameter_manager.set_parameter('bayesian_per_frame', False)
            self.progress_updated.emit(
                100, f"Bayesian λ frozen for batch: {bayesian_lambda:.3g}"
            )
        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.unfreeze_ui()

    def calculate_gcv_regularization(self):
        """Fill the manual λ slider with the GCV-optimal value for the current frame.

        The FTTC+GCV one-shot button. GCV picks λ for the Fourier FTTC operator (the same
        scalar the slider sets), so the result is written straight back into ``regularization``
        where it stays editable. Synchronous; runs on the GUI thread like the Bayesian freeze.
        """
        try:
            self._validate()
        except Exception as e:
            QMessageBox.warning(None, "Warning", str(e))
            return

        self.freeze_ui()
        try:
            self.progress_updated.emit(0, "Estimating GCV regularization...")
            current_frame = self._current_frame()
            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]
            params = self.parameter_manager.get_fttc_parameters()

            lam = find_gcv_regularization(displacement_field, params)
            if lam is None or not np.isfinite(lam) or lam <= 0:
                raise RuntimeError("GCV regularization estimate failed")

            self.parameter_manager.set_parameter('regularization', float(lam))
            self.progress_updated.emit(100, f"GCV λ set: {lam:.3g}")
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
            "run": False, "preview": False, "cancel": True, "bayesian": False,
        }

        # Optional shell-injected predicate: True when displacement is usable as this
        # stage's input even if not resident (i.e. done on disk for the active
        # experiment). Lets Preview/Run enable straight from disk. See
        # set_displacement_available_check / napariTFMWidget._displacement_available.
        self._displacement_available_check = None

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

        All stage actions — run/preview/cancel plus Bayesian auto-λ — live in
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
        return self.controller.preview_force()

    def cancel_action(self):
        self.controller.cancel()

    def bayesian_action(self):
        self.controller.calculate_optimal_regularization()

    def gcv_action(self):
        self.controller.calculate_gcv_regularization()

    def set_displacement_available_check(self, check) -> None:
        """Wire a predicate reporting whether displacement is available as this
        stage's input (resident OR done on disk for the active experiment). Injected
        by the shell so Preview/Run enable from the on-disk status, not only when the
        field is resident. See napariTFMWidget._displacement_available."""
        self._displacement_available_check = check
        self._update_ui_state()

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        has_displacement = self.data_manager.displacement_results is not None
        # Also enable from disk: displacement done for the active experiment is a
        # valid input, pulled into memory on demand when Preview/Run actually fires.
        if not has_displacement and self._displacement_available_check is not None:
            try:
                has_displacement = bool(self._displacement_available_check())
            except Exception:
                has_displacement = False

        self._action_enabled["preview"] = has_displacement
        self._action_enabled["run"] = has_displacement
        self._action_enabled["bayesian"] = has_displacement
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
