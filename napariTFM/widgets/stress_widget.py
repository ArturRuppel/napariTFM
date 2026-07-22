from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QMessageBox

from napariTFM.backend.parameter_dataclasses import StressParameters
from napariTFM.backend.bism import calculate_bism_stresses
from napariTFM.backend.stress import process_mask_data
from napariTFM.widgets._base_widget import BaseAnalysisController, BaseAnalysisWidget
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.utilities.visualization_manager import VisualizationManager


class StressController(BaseAnalysisController):
    """Stress stage (BISM). Uses the base run/cancel lifecycle; its streaming and
    finalize hooks differ from the vector stages (three stress layers, a
    traction-reconstruction R² report), so it fills the hooks directly rather
    than subclassing :class:`VectorStageController`.
    """

    def set_force_loader(self, loader) -> None:
        """Wire a callback that pulls the active experiment's force off disk
        into memory (data-only) if it is not resident. Injected by the shell so this
        stage can run straight from a done-on-disk force without it first
        being viewed. See napariTFMWidget._ensure_force_resident."""
        self._force_loader = loader

    # region === Run lifecycle hooks (template lives in the base) ===
    def _validate(self):
        if self.data_manager.mask_stack is None:
            raise ValueError("No mask loaded. Please load a mask first.")
        # The solver reads the force field from memory; if it is only on disk
        # (a done experiment not yet viewed), pull it in now — the Preview/Run that
        # got here is a compute the user just asked for.
        loader = getattr(self, "_force_loader", None)
        if self.data_manager.force_results is None and loader is not None:
            loader()
        if self.data_manager.force_results is None:
            raise ValueError("No force data available. Please calculate forces first.")

    def _run_params(self) -> StressParameters:
        return self.parameter_manager.get_stress_parameters()

    def _begin_stream(self, params):
        """Allocate the three live stress layers and reset the frame counters."""
        force_results = self.data_manager.force_results
        self._stream_total = force_results.force_field.shape[0]
        self._stream_done = 0
        self.visualization_manager.begin_stress_stream(
            num_frames=self._stream_total,
            max_stress=params.max_stress,
            downscale_factor=force_results.parameters.downscale_factor,
        )

    def _build_worker(self, params):
        return self._run_worker(
            self.data_manager.force_results.force_field,
            self.data_manager.mask_stack,
            params,
        )

    @thread_worker
    def _run_worker(self, force_field, masks, params):
        """Process every frame, yielding each new stress-tensor frame for streaming."""
        gen = calculate_bism_stresses(force_field=force_field, masks=masks, params=params)
        try:
            while True:
                result, current_frame, total = next(gen)
                # ``stress_tensor`` is the cumulative stack; the newest frame is
                # its last slice. Hand it off 0-based.
                yield current_frame - 1, total, result.stress_tensor[-1]
        except StopIteration as e:
            return e.value

    def _on_frame_processed(self, payload):
        """Stream one freshly computed stress frame into the viewer (GUI thread)."""
        frame_index, total, stress_tensor_frame = payload
        progress = int((frame_index + 1) / max(total, 1) * 100)
        self.progress_updated.emit(
            progress, f"Calculating stress: Frame {frame_index + 1}/{total}"
        )
        self.visualization_manager.stream_stress_frame(frame_index, stress_tensor_frame)

    def _finalize(self, result):
        """Commit the streamed stress result and report the reconstruction R²."""
        if result is None:
            raise RuntimeError("Analysis failed to produce results")
        # The three stress stacks were filled in place as frames arrived and are
        # already on screen; just store the full result for downstream steps.
        self.data_manager.set_stress_results(result, dirty=True)

        r2 = result.r2_traction
        r2_text = f"{r2:.4f}" if r2 is not None else "n/a"
        self.progress_updated.emit(
            100,
            f"Analysis completed successfully (BISM)\n"
            f"Mean traction-reconstruction R²: {r2_text}",
        )
        self.analysis_completed.emit(result)
    # endregion

    # region === Preview (async single-shot; template lives in the base) ===
    def preview_current_frame(self, *, force_result=None, completion=None):
        """Preview the stress field for the current frame on a worker thread.

        The single-frame BISM inversion can spike above its usual cost; run inline
        that spell freezes the window, so the compute goes off-thread. GUI/data
        reads (current frame, params, the mask + force slices) happen here on the
        main thread; the napari visualization returns to it in
        :meth:`_show_stress_preview`.
        """
        transient_result = force_result is not None
        try:
            if self.data_manager.mask_stack is None:
                raise ValueError("No mask loaded. Please load a mask first.")
            if force_result is None:
                self._validate()
                force_result = self.data_manager.force_results
        except Exception as e:
            QMessageBox.warning(None, "Warning", str(e))
            return False

        current_frame = self._current_frame()
        params = self._get_current_parameters()
        masks = self.data_manager.mask_stack

        mask = masks[current_frame] if masks.ndim > 2 else masks
        result_frame = 0 if transient_result else current_frame
        force_field = (
            force_result.force_field[result_frame]
            if force_result.force_field.ndim > 3
            else force_result.force_field
        )
        downscale = force_result.parameters.downscale_factor

        worker = self._preview_worker(force_field, mask, params)
        self._start_preview_worker(
            worker,
            lambda result: self._show_stress_preview(result, params, current_frame, downscale),
            status="Generating stress preview...",
            completion=completion,
        )
        return True

    @thread_worker
    def _preview_worker(self, force_field, mask, params):
        """Single-frame BISM stress solve (worker thread; no napari access)."""
        gen = calculate_bism_stresses(
            force_field=force_field[np.newaxis, ...],
            masks=mask[np.newaxis, ...],
            params=params,
        )
        try:
            result, _, _ = next(gen)
        except StopIteration:
            raise ValueError("Stress calculation failed to produce results")
        return result

    def _show_stress_preview(self, result, params, current_frame, downscale):
        """Paint the previewed stress field (GUI thread)."""
        self.visualization_manager.visualize_stress_preview(
            result.stress_tensor,
            max_stress=params.max_stress,
            downscale_factor=downscale,
        )

        # Show only the average-normal-stress layer on top; keep XX/YY loaded
        # (for scrubbing) but hidden, stacked beneath it.
        self.visualization_manager.bring_layers_to_front([
            ('Normal Stress XX', False),
            ('Normal Stress YY', False),
            ('Average Normal Stress', True),
        ])

        r2 = result.r2_traction
        r2_text = f"{r2:.4f}" if r2 is not None else "n/a"
        self.progress_updated.emit(
            100,
            f"Stress preview generated for frame {current_frame} (BISM)\n"
            f"Traction-reconstruction R²: {r2_text}",
        )

    def _get_current_parameters(self) -> StressParameters:
        """Get current stress (BISM) parameters from parameter manager."""
        return self.parameter_manager.get_stress_parameters()
    # endregion


class StressWidget(BaseAnalysisWidget):
    """Widget for Bayesian Inversion Stress Microscopy (BISM) analysis."""
    stress_calculated = Signal(object)  # Emits stress analysis results

    def __init__(
            self,
            viewer: Viewer,
            data_manager: DataManager,
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, parameter_manager, visualization_manager)

        # Action enablement consumed by the stage header via the action contract
        self._action_enabled = {
            "run": False, "preview": False, "cancel": True,
        }

        # Optional shell-injected predicate: True when force is usable as this
        # stage's input even if not resident (i.e. done on disk for the active
        # experiment). Lets Preview/Run enable straight from disk. See
        # set_force_available_check / napariTFMWidget._force_available.
        self._force_available_check = None

        # Monotonic token guarding the async mask loader against stale reads from
        # a superseded experiment-row click. ``_mask_workers`` holds every
        # in-flight read so it isn't garbage-collected mid-run; each removes
        # itself on completion.
        self._mask_load_token = 0
        self._mask_workers = []

        # Get initial parameters from parameter manager
        self.stress_params = parameter_manager.get_stress_parameters()

        # Initialize controller (owns no panels; emits ui_frozen)
        self.controller = StressController(
            viewer=viewer,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
        )

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Keep service parameters synced with the shared parameter manager
        parameter_manager.parameters_reset.connect(self._update_stress_parameters)
        parameter_manager.parameter_changed.connect(self._handle_parameter_change)

        self.controller.unfreeze_ui()

    def _setup_ui(self):
        """Set up the user interface.

        All stage actions — run/preview/cancel — live in the stage header (P7),
        so this widget owns no visible body content.
        """
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect all widget signals."""
        # Progress flows to the shell's one global status label (P2); no local
        # status slot.
        self.controller.analysis_started.connect(self._on_analysis_started)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Update enablement when the active layer changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def load_result_artifact(self, key: str):
        """Load an external input layer. Only the mask is loadable (masks are an external input); analysis results chain in-memory and are
        never read from disk interactively."""
        if key == "mask_stack":
            self._load_mask_stack_from_active_layer()
        self._update_ui_state()

    def _load_mask_stack_from_active_layer(self):
        active_layer = self.viewer.layers.selection.active
        if active_layer is None or not hasattr(active_layer, "data"):
            QMessageBox.warning(self, "No Layer Selected", "Please select an image layer to load masks from.")
            return
        mask_data = active_layer.data
        if not isinstance(mask_data, np.ndarray):
            QMessageBox.warning(self, "Invalid Layer", "Selected layer is not a valid image layer.")
            return
        # Manual load happens after inputs are in memory, so read the bead size
        # straight off the data manager to fit the mask to it in the viewer.
        self._apply_mask_data(mask_data, warn=True, beads_shape=self._loaded_beads_shape())

    def load_mask_from_file(self, mask_path, beads_shape=None) -> None:
        """Load a mask stack from a TIFF on disk into memory, off the UI thread.

        Used to auto-load an experiment's discovered ``masks.tif`` on selection so
        the Stress stage's Run/Preview enable without a manual layer load. Silent
        and best-effort: a missing or unreadable file is a no-op (the user can
        still load a mask manually).

        The read is a full-stack ``imread`` — hundreds of ms on a large mask — so
        it runs on a napari ``thread_worker`` rather than freezing the selection
        click; the decoded array is applied on the GUI thread. A monotonic token
        supersedes an in-flight read from a previous row click: a superseded read
        still runs to completion (a function worker can't be interrupted) but is
        dropped by the token check when it lands, so rapid clicking never paints a
        stale experiment's mask.

        ``beads_shape`` is the bead image's ``(height, width)``; when given, the
        mask's visualization layer is scaled so its xy dimensions fit the beads.
        The caller supplies it because the bead stack itself loads asynchronously
        and may not be in memory yet on selection.
        """
        from pathlib import Path

        mask_path = Path(mask_path)

        # Bump the token so any in-flight read from a previous row click is
        # recognized as stale when it lands.
        self._mask_load_token += 1
        token = self._mask_load_token

        if not mask_path.exists():
            return

        worker = self._read_mask_worker(mask_path, token)
        self._mask_workers.append(worker)
        worker.returned.connect(
            lambda payload, _w=worker, _bs=beads_shape: self._on_mask_read(payload, _w, _bs)
        )
        worker.start()

    @thread_worker
    def _read_mask_worker(self, mask_path, token):
        """Read a mask TIFF off the UI thread; returns ``(token, array)``.

        Returns ``(token, None)`` on any read error so the GUI-thread slot can
        stale-check and quietly drop it — a missing or unreadable mask is not
        fatal (the user can still load one manually).
        """
        import tifffile

        try:
            mask_data = tifffile.imread(str(mask_path))
        except Exception as e:  # pragma: no cover - defensive
            print(f"Could not read mask file {mask_path}: {e}")
            return token, None
        return token, mask_data

    def _on_mask_read(self, payload, worker, beads_shape) -> None:
        """Apply a mask array read off-thread (GUI thread).

        Drops the finished worker from the in-flight list, then stale-checks the
        token before touching the data manager or viewer so a superseded read
        cannot overwrite the current row's mask.
        """
        try:
            self._mask_workers.remove(worker)
        except ValueError:
            pass
        token, mask_data = payload
        if token != self._mask_load_token:
            return
        if not isinstance(mask_data, np.ndarray):
            return
        self._apply_mask_data(mask_data, warn=False, beads_shape=beads_shape)

    def _apply_mask_data(self, mask_data: np.ndarray, *, warn: bool, beads_shape=None) -> None:
        """Resize a raw mask onto the analysis grid, store it, and show it.

        Shared by the manual (active-layer) and automatic (from-file) paths.
        ``warn`` pops the conversion/resize notices for the manual path but stays
        silent for the auto-load so selecting an experiment is quiet.
        ``beads_shape`` is the bead image ``(height, width)`` the displayed mask
        should be scaled to fit (``None`` leaves it at its own size).
        """
        force_results = self.data_manager.force_results
        force_field = force_results.force_field if force_results is not None else None
        processed_masks, warnings = process_mask_data(mask_data, force_field)
        if warn:
            for warning in warnings:
                QMessageBox.warning(self, "Warning", warning)
        self.data_manager.set_mask_stack(processed_masks)
        # The stored mask sits on the (downsampled) force grid, so the labels layer
        # is smaller than the full-resolution bead image. Scale the *visualization*
        # layer so its xy dimensions fit the beads, without inflating the array.
        self.visualization_manager.visualize_masks(
            processed_masks,
            scale=self._mask_display_scale(processed_masks, beads_shape),
        )
        self._update_ui_state()

    def _loaded_beads_shape(self):
        """Bead image ``(height, width)`` from the data manager, or ``None``."""
        beads = getattr(self.data_manager, "bead_stack", None)
        if beads is not None and beads.ndim >= 2:
            return tuple(beads.shape[-2:])
        return None

    def _mask_display_scale(self, masks: np.ndarray, beads_shape):
        """Per-axis world scale that fits the displayed mask to the bead image.

        Compares the actual mask and bead xy shapes; ``None`` when there is no
        bead shape to fit to or the mask already matches it.
        """
        if beads_shape is None:
            return None
        my, mx = masks.shape[-2:]
        ty, tx = beads_shape[-2:]
        if my == 0 or mx == 0 or (my, mx) == (ty, tx):
            return None
        return (1.0, ty / my, tx / mx)

    def _update_stress_parameters(self, category: ParameterCategory):
        """Refresh cached stress parameters when a parameter reset occurs."""
        if category == ParameterCategory.STRESS:
            self.stress_params = self.parameter_manager.get_stress_parameters()

    def _handle_parameter_change(self, param_name: str, value: Any):
        """Refresh cached stress parameters when an individual parameter changes."""
        stress_params = {'bism_regularization', 'max_stress'}
        if param_name in stress_params:
            self.stress_params = self.parameter_manager.get_stress_parameters()

    def run_action(self):
        self.controller.run()

    def preview_action(self):
        return self.controller.preview_current_frame()

    def cancel_action(self):
        self.controller.cancel()

    def set_force_available_check(self, check) -> None:
        """Wire a predicate reporting whether force is available as this
        stage's input (resident OR done on disk for the active experiment). Injected
        by the shell so Preview/Run enable from the on-disk status, not only when the
        field is resident. See napariTFMWidget._force_available."""
        self._force_available_check = check
        self._update_ui_state()

    def _update_ui_state(self, event=None):
        """Update action button enablement based on available data."""
        has_force = self.data_manager.force_results is not None
        # Also enable from disk: force done for the active experiment is a valid
        # input, pulled into memory on demand when Preview/Run actually fires.
        if not has_force and self._force_available_check is not None:
            try:
                has_force = bool(self._force_available_check())
            except Exception:
                has_force = False
        has_mask = self.data_manager.mask_stack is not None

        self._action_enabled["preview"] = has_force and has_mask
        self._action_enabled["run"] = has_force and has_mask
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def _on_analysis_started(self):
        """Handle analysis start event."""
        self.controller.progress_updated.emit(0, "Analysis started...")

    def _on_analysis_completed(self, results):
        """Handle analysis completion.

        Preview-only: result held in memory and shown in napari;
        nothing written to disk. Batch is the only path to persisted data.
        """
        self.stress_calculated.emit(results)
        self._update_ui_state()

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self.controller.progress_updated.emit(0, f"Analysis failed: {error_msg}")
        self._update_ui_state()
