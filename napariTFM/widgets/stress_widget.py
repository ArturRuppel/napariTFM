from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QApplication, QHBoxLayout, QMessageBox

from napariTFM.backend.parameter_dataclasses import StressParameters
from napariTFM.backend.bism import calculate_bism_stresses
from napariTFM.backend.stress import process_mask_data
from napariTFM.widgets._base_widget import BaseAnalysisController, BaseAnalysisWidget
from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.utilities.visualization_manager import VisualizationManager


class StressController(BaseAnalysisController):
    """Coordinates interactions between UI components, service, and managers."""

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.mask_stack is None:
            QMessageBox.warning(None, "Warning", "No mask loaded. Please load a mask first.")
            return False
        if self.data_manager.force_results is None:
            QMessageBox.warning(None, "Warning", "No force data available. Please calculate forces first.")
            return False
        return True

    def start_analysis(self):
        """Start the stress analysis for all frames."""
        try:
            if not self._validate_prerequisites():
                return

            self.analysis_started.emit()
            self._update_progress(0, "Starting analysis...")

            params = self._get_current_parameters()

            # Get mask and force data
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            force_results = self.data_manager.force_results
            if force_results is None:
                raise ValueError("No force data available")

            # Bind the three stress layers up front so each frame streams in live
            # (preserving the contrast/visibility the user set) and the slider
            # follows it, instead of the whole stack landing only at the end.
            total_frames = force_results.force_field.shape[0]
            self.visualization_manager.begin_stress_stream(
                num_frames=total_frames,
                max_stress=params.max_stress,
                downscale_factor=force_results.parameters.downscale_factor,
            )

            @thread_worker
            def stress_calculation_worker():
                try:
                    stress_generator = calculate_bism_stresses(
                        force_field=force_results.force_field,
                        masks=masks,
                        params=params,
                    )

                    while True:
                        result, current_frame, total = next(stress_generator)
                        # ``stress_tensor`` is the cumulative stack; the newest
                        # frame is its last slice. Hand it off 0-based.
                        yield current_frame - 1, total, result.stress_tensor[-1]

                except StopIteration as e:
                    return e.value

            worker = stress_calculation_worker()
            worker.running = True
            self.active_workers.append(worker)

            def on_yielded(data):
                frame_index, total, stress_tensor_frame = data
                progress = int((frame_index + 1) / max(total, 1) * 100)
                self._update_progress(
                    progress, f"Calculating stress: Frame {frame_index + 1}/{total}"
                )
                self.visualization_manager.stream_stress_frame(frame_index, stress_tensor_frame)

            def on_returned(final_result):
                # The three stress stacks were filled in place as frames arrived
                # and are already on screen; just store the full result for
                # downstream steps. Other layers' visibility is left untouched.
                self.data_manager.set_stress_results(final_result, source="generated", dirty=True)

                r2 = final_result.r2_traction
                r2_text = f"{r2:.4f}" if r2 is not None else "n/a"
                status_msg = (
                    f"Analysis completed successfully (BISM)\n"
                    f"Mean traction-reconstruction R²: {r2_text}"
                )
                self._update_progress(100, status_msg)

                self.analysis_completed.emit(final_result)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            def on_errored(error):
                error_msg = f"Analysis failed: {str(error)}"
                self._update_progress(0, error_msg)
                self.analysis_failed.emit(error_msg)
                QMessageBox.critical(None, "Error", error_msg)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            # Connect worker signals
            worker.yielded.connect(on_yielded)
            worker.returned.connect(on_returned)
            worker.errored.connect(on_errored)
            # Freeze here — after every prerequisite check that can raise, and
            # right before the worker runs — so the header's Run/Preview actions
            # disable during a live BISM run (on_returned/on_errored unfreeze).
            # Every other stage freezes on run; stress used to be the exception.
            self.freeze_ui()
            worker.start()

        except Exception as e:
            error_msg = f"Failed to start analysis: {str(e)}"
            self._update_progress(0, error_msg)
            self.analysis_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def preview_current_frame(self):
        """Calculate and display stress field for current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self._update_progress(0, "Generating stress preview...")

            # Get current frame index
            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            params = self._get_current_parameters()

            # Get mask and force data for current frame
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            force_results = self.data_manager.force_results
            if force_results is None:
                raise ValueError("No force data available")

            # Extract current frame data
            mask = masks[current_frame] if masks.ndim > 2 else masks
            force_field = force_results.force_field[current_frame] if force_results.force_field.ndim > 3 else force_results.force_field

            stress_generator = calculate_bism_stresses(
                force_field=force_field[np.newaxis, ...],
                masks=mask[np.newaxis, ...],
                params=params,
            )

            try:
                # Get stress results
                result, _, _ = next(stress_generator)

                # Update visualization
                self.visualization_manager.visualize_stress_preview(
                    result.stress_tensor,
                    max_stress=params.max_stress,
                    downscale_factor=force_results.parameters.downscale_factor
                )

                # Manage layer visibility and ordering
                xx_layer = None
                yy_layer = None
                avg_layer = None

                # First pass: find the stress layers and disable all others
                for layer in self.viewer.layers:
                    if layer.name == 'Normal Stress XX':
                        xx_layer = layer
                        layer.visible = False
                    elif layer.name == 'Normal Stress YY':
                        yy_layer = layer
                        layer.visible = False
                    elif layer.name == 'Average Normal Stress':
                        avg_layer = layer
                        layer.visible = True
                    elif self.visualization_manager.colorbar_manager.is_colorbar_layer(layer.name):
                        # Keep the scale legend visible alongside the preview.
                        layer.visible = True
                    else:
                        layer.visible = False

                # Second pass: reorder layers
                if xx_layer is not None:
                    current_index = self.viewer.layers.index(xx_layer)
                    if current_index != len(self.viewer.layers) - 3:  # -3 position
                        self.viewer.layers.move(current_index, -3)

                if yy_layer is not None:
                    current_index = self.viewer.layers.index(yy_layer)
                    if current_index != len(self.viewer.layers) - 2:  # -2 position
                        self.viewer.layers.move(current_index, -2)

                if avg_layer is not None:
                    current_index = self.viewer.layers.index(avg_layer)
                    if current_index != len(self.viewer.layers) - 1:  # -1 position (top)
                        self.viewer.layers.move(current_index, -1)

                r2 = result.r2_traction
                r2_text = f"{r2:.4f}" if r2 is not None else "n/a"
                status_msg = (
                    f"Stress preview generated for frame {current_frame} (BISM)\n"
                    f"Traction-reconstruction R²: {r2_text}"
                )

                self._update_progress(100, status_msg)
                return result

            except StopIteration:
                raise ValueError("Stress calculation failed to produce results")

        except Exception as e:
            error_msg = f"Failed to preview stress field: {str(e)}"
            self._update_progress(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def _get_current_parameters(self) -> StressParameters:
        """Get current stress (BISM) parameters from parameter manager."""
        return self.parameter_manager.get_stress_parameters()

    def _update_progress(self, progress: int, status: str):
        """Update progress and emit signal."""
        self.progress_updated.emit(progress, status)

    def cancel_all_operations(self):
        """Cancel all running background operations"""
        for worker in self.active_workers:
            try:
                worker.running = False  # Set cancellation flag
                worker.quit()
                worker.wait(500)  # Wait up to 500ms
                if worker.isRunning():
                    worker.terminate()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()
        # Update UI status
        self.progress_updated.emit(0, "Operations cancelled")
        QApplication.processEvents()
        self.unfreeze_ui()


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
        """Load an external input layer. Only the mask is loadable (ROADMAP §2 —
        masks are an external input); analysis results chain in-memory and are
        never read from disk interactively (ROADMAP §4)."""
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

    def load_mask_from_file(self, mask_path, beads_shape=None) -> bool:
        """Load a mask stack from a TIFF on disk into memory.

        Used to auto-load an experiment's discovered ``masks.tif`` on selection so
        the Stress stage's Run/Preview enable without a manual layer load. Silent
        and best-effort: a missing or unreadable file is a no-op (the user can
        still load a mask manually). Returns True when a mask was loaded.

        ``beads_shape`` is the bead image's ``(height, width)``; when given, the
        mask's visualization layer is scaled so its xy dimensions fit the beads.
        The caller supplies it because the bead stack itself loads asynchronously
        and may not be in memory yet on selection.
        """
        from pathlib import Path

        import tifffile

        mask_path = Path(mask_path)
        if not mask_path.exists():
            return False
        try:
            mask_data = tifffile.imread(str(mask_path))
        except Exception as e:  # pragma: no cover - defensive
            print(f"Could not read mask file {mask_path}: {e}")
            return False
        if not isinstance(mask_data, np.ndarray):
            return False
        self._apply_mask_data(mask_data, warn=False, beads_shape=beads_shape)
        return True

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
        self.controller.start_analysis()

    def preview_action(self):
        self.controller.preview_current_frame()

    def cancel_action(self):
        self.controller.cancel_all_operations()

    def _update_ui_state(self, event=None):
        """Update action button enablement based on available data."""
        has_force = self.data_manager.force_results is not None
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

        Preview-only (ROADMAP §4): result held in memory and shown in napari;
        nothing written to disk. Batch is the only path to persisted data.
        """
        self.stress_calculated.emit(results)
        self._update_ui_state()

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self.controller.progress_updated.emit(0, f"Analysis failed: {error_msg}")
        self._update_ui_state()
