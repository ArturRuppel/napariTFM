from pathlib import Path
from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QTimer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFrame, QCheckBox, QApplication,
    QMessageBox, QSizePolicy
)
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget
)

from napariTFM.widgets._base_widget import BaseAnalysisController, BaseAnalysisWidget
from napariTFM.utilities.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.backend.preprocessing import preprocess_frame, preprocess_stack


def _open_lazy(path):
    """Open a TIFF lazily via memmap; fall back to imread for compressed files.

    memmap returns a real np.ndarray subclass backed by the file in ~1 ms for
    uncompressed TIFFs. imread is only reached if memmap fails (e.g. tiled or
    compressed data), which preserves behaviour for all existing file types.
    """
    import tifffile
    try:
        return tifffile.memmap(str(path))
    except Exception:
        return tifffile.imread(str(path))


class PreprocessingController(BaseAnalysisController):
    """Controller coordinating UI components and data processing.

    Inherits the shared signal set from :class:`BaseAnalysisController`; the
    completion payload (``analysis_completed``) carries a parameters dict.
    """

    # region === Initialization
    def __init__(self, viewer, data_manager, parameter_manager, visualization_manager):
        super().__init__(viewer, data_manager, parameter_manager, visualization_manager)

        self.preview_enabled = False

        # Load-token: incremented on every load_input_files call so that yields
        # from a superseded (stale) load worker are silently dropped.
        self._load_token = 0
        self._load_worker = None

        # Sliders emit valueChanged continuously while dragging, so parameter
        # changes arrive in rapid bursts. Recomputing the preview on each one
        # freezes the UI; instead coalesce a burst into a single recompute
        # once the changes settle.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        # Late-bound through a lambda so tests can patch _update_preview.
        self._preview_timer.timeout.connect(lambda: self._update_preview())

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

            # Pre-allocate the output stacks and bind them to napari layers up
            # front, so each processed frame can stream in live (preserving the
            # contrast/visibility the user has set) and the slider can follow it,
            # instead of the whole result appearing only when the run finishes.
            self._begin_stream()

            worker = self._create_processing_worker()
            self.active_workers.append(worker)
            self.analysis_started.emit()
            self.freeze_ui()

            worker.yielded.connect(self._on_frame_processed)
            worker.returned.connect(self._finish_streaming)
            worker.errored.connect(self._handle_preprocessing_error)
            worker.start()

        except Exception as e:
            self.analysis_failed.emit(str(e))
            QMessageBox.critical(None, "Error", str(e))
            self.unfreeze_ui()

    def _begin_stream(self):
        """Allocate the output stacks, register them, and create live layers.

        The processed images are normalized to [0, 1] floats, so the stacks are
        allocated as float32 zeros and filled frame-by-frame by the worker. They
        are registered with the data manager immediately (dirty, generated) so
        the layers can be backed by the very arrays the run mutates in place.
        """
        bead = self.data_manager.bead_stack
        ref = self.data_manager.reference
        cell = self.data_manager.cell_stack

        n_beads = bead.shape[0] if bead is not None else 0
        n_cells = cell.shape[0] if cell is not None else 0
        self._stream_total = n_beads + (1 if ref is not None else 0) + n_cells
        self._stream_done = 0

        if bead is not None:
            self.data_manager.set_preprocessed_bead_stack(
                np.zeros(bead.shape, dtype=np.float32), source="generated", dirty=True
            )
        if ref is not None:
            self.data_manager.set_preprocessed_reference(
                np.zeros(ref.shape, dtype=np.float32), source="generated", dirty=True
            )
        if cell is not None:
            self.data_manager.set_preprocessed_cell_stack(
                np.zeros(cell.shape, dtype=np.float32), source="generated", dirty=True
            )

        self.visualization_manager.begin_preprocessing_stream()

    @thread_worker
    def _create_processing_worker(self):
        """Process every input frame, yielding each result for live streaming.

        Each yield carries ``(data_type, frame_index, total, processed_image)``;
        the GUI-thread slot writes it straight into the pre-allocated output
        stack and advances the viewer, so the stack fills in on screen as the
        run proceeds rather than all at once at the end.
        """
        params = self.parameter_manager.get_preprocessing_parameters()

        if self.data_manager.bead_stack is not None:
            for result, frame, total in preprocess_stack(
                    image_stack=self.data_manager.bead_stack,
                    params=params,
                    reference_image=self.data_manager.reference
            ):
                yield 'beads', frame, total, result.processed_image

        if self.data_manager.reference is not None:
            result = preprocess_frame(self.data_manager.reference, params)
            yield 'reference', 0, 1, result.processed_image

        if self.data_manager.cell_stack is not None:
            for result, frame, total in preprocess_stack(
                    image_stack=self.data_manager.cell_stack,
                    params=params,
                    is_cell=True
            ):
                yield 'cells', frame, total, result.processed_image

    def _on_frame_processed(self, payload):
        """Stream one freshly processed frame into the viewer (GUI thread).

        ``worker.yielded`` is a cross-thread Qt signal, so this slot runs on the
        main thread and may safely touch napari layers.
        """
        data_type, frame, total, image = payload
        self._stream_done += 1
        progress = int(self._stream_done / max(self._stream_total, 1) * 100)
        if data_type == 'reference':
            message = "Processing reference"
        else:
            message = f"Processing {data_type}: Frame {frame + 1}/{total}"
        self.progress_updated.emit(progress, message)
        self.visualization_manager.stream_preprocessing_frame(data_type, frame, image)

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
            self._preview_timer.start()

    def _on_parameters_reset(self, category: ParameterCategory):
        """Handle parameter reset events."""
        if category == ParameterCategory.PREPROCESSING and self.preview_enabled:
            self._preview_timer.start()

    # endregion === Parameter Handling

    # region === Data Management
    def load_active_layer(self, data_type: str):
        """Load the currently active layer as the specified data type."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(None, "Error", "No active image layer found")
            return

        try:
            self._set_input_data(data_type, active_layer.data)
        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))

    def _set_input_data(self, data_type: str, data: np.ndarray, path=None):
        """Validate, shape, and store one raw input, then announce the update.

        Shared by the active-layer path and the disk-loading path so both reshape
        and notify identically. Emitting ``data_updated`` is what re-enables the
        Preview and Run actions (the shell listens via ``_update_ui_state``).
        """
        if data_type == 'beads':
            # Convert 2D data to 3D with single frame if needed
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim != 3:
                raise ValueError("Bead stack must be 2D or 3D (frames, height, width)")
            self.data_manager.set_bead_stack(data, path=path)

        elif data_type == 'reference':
            if data.ndim != 2:
                raise ValueError("Reference image must be 2D (height, width)")
            self.data_manager.set_reference(data, path=path)

        elif data_type == 'cells':
            # Convert 2D data to 3D with single frame if needed
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim != 3:
                raise ValueError("Cell stack must be 2D or 3D (frames, height, width)")
            self.data_manager.set_cell_stack(data, path=path)
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        # Update UI state and emit signal
        self.data_updated.emit(data_type)
        if self.preview_enabled:
            self._update_preview()

    def peek_input_xy_shape(self, folder, input_files, slot="beads"):
        """Read one input file's ``(height, width)`` from disk without loading it.

        ``_open_lazy`` memmaps the TIFF (~1 ms), so this is cheap to call on the
        UI thread. Lets callers learn the bead image size synchronously even
        though ``load_input_files`` streams the arrays in asynchronously. Returns
        ``None`` when the slot is unnamed, missing, or unreadable.
        """
        name = (input_files or {}).get(slot)
        if not folder or not name:
            return None
        path = Path(folder) / name
        if not path.exists():
            return None
        try:
            arr = _open_lazy(path)
            return tuple(arr.shape[-2:])
        except Exception:
            return None

    def load_input_files(self, folder, input_files):
        """Load an experiment's raw input files lazily and off the UI thread.

        Supersedes any in-flight load (stale yields are silently dropped via a
        monotonic load-token), freezes the UI while loading, then unfreezes on
        completion. Missing or un-named inputs are skipped; a file that fails
        to open pops a warning without aborting the rest.
        """
        if not folder:
            return

        # Supersede any in-flight load from a previous row click. A quit
        # generator-worker may emit neither returned nor errored, so drop it
        # from active_workers here to keep the list from growing on rapid clicks.
        self._load_token += 1
        token = self._load_token
        if self._load_worker is not None:
            try:
                self._load_worker.quit()
            except Exception:
                pass
            try:
                self.active_workers.remove(self._load_worker)
            except ValueError:
                pass
            self._load_worker = None

        folder = Path(folder)
        input_files = input_files or {}

        worker = self._create_input_load_worker(folder, input_files, token)
        self._load_worker = worker
        self.active_workers.append(worker)

        self.progress_updated.emit(0, "Loading inputs...")

        worker.yielded.connect(self._on_input_loaded)
        worker.returned.connect(
            lambda _=None, _token=token: self._finish_input_load(_token)
        )
        worker.errored.connect(
            lambda exc, _token=token: self._handle_input_load_error(exc, _token)
        )
        worker.start()

    def _add_input_layer(self, name: str, data: np.ndarray) -> None:
        """Show a raw input in the viewer, replacing any prior layer of that name.

        Explicit contrast_limits are computed from a single representative frame
        so napari does not scan the full stack on add, which prevents the 3-8 s
        freeze caused by reading a 250 MB stack just to infer the display range.
        """
        add_image = getattr(self.viewer, "add_image", None)
        if add_image is None:
            return
        layers = getattr(self.viewer, "layers", None)
        if layers is not None:
            for layer in list(layers):
                if getattr(layer, "name", None) == name:
                    layers.remove(layer)

        # Compute contrast limits from a single frame to avoid scanning the
        # full stack. Guard against degenerate (flat) data by nudging hi up.
        contrast_limits = None
        try:
            frame = data[0] if data.ndim == 3 else data
            lo, hi = int(frame.min()), int(frame.max())
            contrast_limits = (lo, hi) if lo < hi else (lo, lo + 1)
        except Exception:
            pass

        if contrast_limits is not None:
            add_image(data, name=name, contrast_limits=contrast_limits)
        else:
            add_image(data, name=name)

    @thread_worker
    def _create_input_load_worker(self, folder, input_files, token):
        """Open raw input TIFFs lazily, yielding each array for GUI-thread painting.

        Runs entirely off the UI thread. memmap opens large uncompressed TIFFs
        in ~1 ms; imread is the fallback for compressed files. Each yield
        carries ``(token, data_type, layer_name, path, array)`` so the
        GUI-thread slot can stale-check before touching napari layers.
        """
        for data_type, slot, layer_name in (
            ('beads', 'beads', 'Beads'),
            ('reference', 'reference', 'Reference'),
            ('cells', 'cells', 'Cells'),
        ):
            name = input_files.get(slot)
            if not name:
                continue
            path = folder / name
            if not path.exists():
                continue
            array = _open_lazy(path)
            yield token, data_type, layer_name, path, array

    def _on_input_loaded(self, payload):
        """Paint one loaded input into the viewer (GUI thread).

        Connected to ``worker.yielded``; stale-checks the token before touching
        napari so a superseded load cannot overwrite layers for the current row.
        A failed file pops a warning but does not abort the remaining yields.
        """
        token, data_type, layer_name, path, array = payload
        if token != self._load_token:
            return
        try:
            self._add_input_layer(layer_name, array)
            self._set_input_data(data_type, array, path=path)
        except Exception as exc:
            QMessageBox.warning(None, "Error", f"Could not load {path.name}: {exc}")

    def _finish_input_load(self, token):
        """Finalize an input load run (GUI thread).

        No-ops for stale tokens so a superseded load cannot emit a status
        message that belongs to the current row.
        """
        if token != self._load_token:
            return
        worker = self._load_worker
        self._load_worker = None
        if worker is not None:
            try:
                self.active_workers.remove(worker)
            except ValueError:
                pass
        self.progress_updated.emit(100, "Inputs loaded")

    def _handle_input_load_error(self, exc, token):
        """Handle a fatal error from the input load worker (GUI thread).

        No-ops for stale tokens. Shows the error so the user knows a load
        failed.
        """
        if token != self._load_token:
            return
        worker = self._load_worker
        self._load_worker = None
        if worker is not None:
            try:
                self.active_workers.remove(worker)
            except ValueError:
                pass
        error_msg = str(exc)
        self.progress_updated.emit(0, f"Error loading inputs: {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode."""
        try:
            self.preview_enabled = enabled

            if enabled:
                self._update_preview()
                # Starting the preview isolates the beads channel: hide every
                # other layer (including the reference/cells previews) so the
                # beads are inspected on their own. Later re-renders leave the
                # user's visibility choices alone (see _update_preview).
                if self.preview_enabled:
                    self.visualization_manager.isolate_layers(['Preview Beads'])
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
        self.analysis_failed.emit(error_msg)
        self.progress_updated.emit(0, f"Error: {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)
        self.unfreeze_ui()

    def _finish_streaming(self, _result=None):
        """Finalize a streamed run.

        The output stacks were filled in place as frames arrived and are already
        on screen, so there is nothing left to assemble — just announce
        completion. Visibility of other layers is left untouched (the user's to
        set), and the streamed layers keep whatever contrast/visibility the run
        inherited.

        Preview-only: preprocessed stacks are held in memory and
        shown in napari; nothing is written to disk. Batch is the only path to
        persisted data.
        """
        try:
            self.active_workers.clear()
            current_params = self.parameter_manager.get_preprocessing_parameters()
            self.progress_updated.emit(100, "Preprocessing complete")
            self.analysis_completed.emit({'parameters': current_params.__dict__})

        except Exception as e:
            error_msg = f"Error finalizing preprocessing: {str(e)}"
            self.analysis_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
        finally:
            self.unfreeze_ui()

    # endregion === Data Management


class PreprocessingWidget(BaseAnalysisWidget):
    """Main preprocessing widget integrating all components."""

    preprocessing_completed = Signal(dict)  # Emits processed data

    # region === Initialization
    def __init__(
            self,
            viewer: Viewer,
            data_manager,
            parameter_manager: ParameterManager,
            visualization_manager
    ):
        super().__init__(viewer, data_manager, parameter_manager, visualization_manager)

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

    # endregion

    # region === Signal Handling
    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect preview controls
        self.preview_check.toggled.connect(self._on_preview_toggled)

        # Run/cancel are driven by the header via the action contract
        # (run_action / cancel_action); no body button wiring here.

        # Connect controller signals. Progress is surfaced by the shell's one
        # global status label (P2), so there is no local status slot to wire.
        self.controller.analysis_completed.connect(self._on_preprocessing_completed)
        self.controller.analysis_failed.connect(self._on_preprocessing_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)
        # Parameter-change-driven preview is handled by the controller via
        # ParameterManager.parameter_changed; no panel wiring needed here.

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _on_preview_toggled(self, enabled: bool):
        """Handle preview toggle."""
        self._action_enabled["preview_active"] = enabled
        self.controller.toggle_preview(enabled)
        self.action_states_changed.emit()

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
    def run_action(self):
        self._on_process_clicked()

    def preview_action(self):
        self.preview_check.setChecked(not self.preview_check.isChecked())

    def cancel_action(self):
        self.controller.cancel_all_operations()

    def load_active_layer(self, data_type: str):
        """Delegate input-layer loading to the controller (called by the shell)."""
        self.controller.load_active_layer(data_type)

    def load_input_files(self, folder, input_files):
        """Delegate disk-loading of an experiment's raw inputs to the controller."""
        self.controller.load_input_files(folder, input_files)

    def peek_input_xy_shape(self, folder, input_files, slot="beads"):
        """Delegate a cheap on-disk shape read of an input file to the controller."""
        return self.controller.peek_input_xy_shape(folder, input_files, slot)

    # endregion

    # region === State Management
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
        self._action_enabled["preview_active"] = self.preview_check.isChecked()

        # Update action enablement - now uses _has_required_data()
        self._action_enabled["run"] = self._has_required_data()
        self.action_states_changed.emit()

    def _handle_ui_freeze(self, frozen: bool):
        """Lock the body's preview checkbox alongside the header actions.

        Preprocessing is the only stage with a body control (the Show Preview
        checkbox), so it extends the base freeze to disable it too; the action
        map itself is handled by the base.
        """
        self.preview_check.setEnabled(not frozen)
        super()._handle_ui_freeze(frozen)

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
