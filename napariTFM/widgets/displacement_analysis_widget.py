from pathlib import Path

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


class DisplacementController(VectorStageController):
    """Displacement stage: multi-pass PIV displacement over the raw bead stack.

    A thin :class:`VectorStageController` subclass — the run/cancel lifecycle is
    inherited; this class only supplies the stage's spec (kind, result setter),
    the run hooks, and the synchronous preview.

    As the pipeline's first stage (preprocessing having been removed), it also
    owns loading the experiment's raw bead/reference/cell inputs from disk and
    showing them in the viewer — the machinery the preprocessing stage used to
    carry.
    """

    STAGE_KIND = 'displacement'
    RESULT_SETTER = 'set_displacement_results'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Monotonic token guarding the async raw-input loader against stale
        # yields from a superseded experiment-row click.
        self._load_token = 0
        self._load_worker = None

    # region === Run lifecycle hooks (template lives in the base) ===
    def _validate(self):
        if self.data_manager.reference is None:
            raise ValueError("No reference image loaded")
        if self.data_manager.bead_stack is None:
            raise ValueError("No bead stack loaded")

    def _run_params(self):
        return self.parameter_manager.get_displacement_parameters()

    def _stream_frame_count(self):
        return self.data_manager.bead_stack.shape[0]

    def _vis_params(self, params):
        return {
            'v_max': params.d_max,
            'vector_stride': params.disp_vector_stride,
            'arrow_scale': params.disp_arrow_scale,
            'downscale_factor': params.downscale_factor,
        }

    def _build_worker(self, params):
        return self._run_worker(
            self.data_manager.reference,
            self.data_manager.bead_stack,
            params,
        )

    def _confinement_mask(self, params, frame=None):
        """Raw full-resolution mask for displacement confinement, or ``None``.

        Read from the active experiment's ``masks`` file on disk -- deliberately
        NOT ``data_manager.mask_stack``, which the Stress stage resizes onto the
        downsampled force grid (wrong resolution for the full-res bead images the
        measurement runs on). Only touched when Confine to Mask is on, so the
        (large) file is never read needlessly. ``frame`` loads a single plane for
        the single-frame preview; ``None`` loads the whole stack for a run.
        """
        if not getattr(params, "disp_mask_confine", False):
            return None
        path = self.data_manager.raw_input_path("masks")
        if path is None:
            return None
        import tifffile
        try:
            return tifffile.imread(str(path)) if frame is None else tifffile.imread(str(path), key=frame)
        except Exception:
            return None

    @thread_worker
    def _run_worker(self, reference, bead_stack, params):
        """Process every frame, yielding each displacement field for live streaming."""
        try:
            mask = self._confinement_mask(params)
            gen = calculate_displacement_field(reference, bead_stack, params, mask=mask)
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

    # region === Preview (async single-shot; template lives in the base) ===
    def preview_displacement(self):
        """Preview displacement for the current frame on a worker thread.

        The single-frame solve is a full PIV/iLK/FFD compute whose cost can spike
        far above its usual (GPU contention, a cold kernel); run inline that spell
        freezes the window, so the compute goes off-thread. All GUI/data reads
        happen here on the main thread and cross to the worker as plain arrays;
        the napari visualization returns to the main thread in
        :meth:`_show_displacement_preview`.
        """
        try:
            self._validate()
        except Exception as e:
            QMessageBox.warning(None, "Warning", str(e))
            return

        current_frame = self._current_frame()
        moving = self.data_manager.bead_stack[current_frame]
        reference = self.data_manager.reference
        params = self.parameter_manager.get_displacement_parameters()
        mask = self._confinement_mask(params, frame=current_frame)

        worker = self._preview_worker(reference, moving, params, mask)
        self._start_preview_worker(
            worker,
            lambda result: self._show_displacement_preview(result, params),
            status="Calculating displacement preview...",
        )

    @thread_worker
    def _preview_worker(self, reference, moving, params, mask):
        """Single-frame displacement solve (worker thread; no napari access)."""
        gen = calculate_displacement_field(reference, moving, params, mask=mask)
        final_result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            final_result = e.value
        if final_result is None:
            raise RuntimeError("Preview calculation failed")
        return final_result

    def _show_displacement_preview(self, final_result, params):
        """Paint the previewed displacement field (GUI thread)."""
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

        self.data_updated.emit(data_type)

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
        monotonic load-token), then reports completion. Missing or un-named
        inputs are skipped; a file that fails to open pops a warning without
        aborting the rest.
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

        # Inputs stream in on a worker and each lands on top of napari's stack;
        # re-assert the canonical input/mask order so the arrival sequence (and
        # the async mask racing the cells layer) can't leave them out of order.
        order = getattr(self.visualization_manager, "order_input_layers", None)
        if callable(order):
            order()

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

        No-ops for stale tokens. Shows the error so the user knows a load failed.
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
        has_reference = self.data_manager.reference is not None
        has_beads = self.data_manager.bead_stack is not None

        can_analyze = has_reference and has_beads
        self._action_enabled["preview"] = can_analyze
        self._action_enabled["run"] = can_analyze
        self._action_enabled["cancel"] = True
        self.action_states_changed.emit()

    def load_active_layer(self, data_type: str):
        """Delegate input-layer loading to the controller (called by the shell)."""
        self.controller.load_active_layer(data_type)

    def load_input_files(self, folder, input_files):
        """Delegate raw-input loading to the controller (called by the shell)."""
        self.controller.load_input_files(folder, input_files)

    def peek_input_xy_shape(self, folder, input_files, slot="beads"):
        """Delegate a cheap input-shape peek to the controller (called by the shell)."""
        return self.controller.peek_input_xy_shape(folder, input_files, slot)

    # endregion

    # region === Results Handling

    def _on_analysis_completed(self, results):
        """Handle completed analysis.

        Preview-only: the result is held in memory and shown in
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
