from functools import partial
from typing import Optional

import napari
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMessageBox, QWidget

from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager


class BaseAnalysisController(QObject):
    """Shared base for the per-stage analysis controllers.

    Owns the signal set, manager wiring, worker bookkeeping, and — the point of
    this class — the **one blessed run/cancel lifecycle** every streaming stage
    shares. ``run()`` and ``cancel()`` are *sealed*: a subclass cannot override
    them (see ``__init_subclass__`` and the name-mangled ``__run``), it can only
    fill in the hooks (:meth:`_validate`, :meth:`_run_params`,
    :meth:`_begin_stream`, :meth:`_build_worker`, :meth:`_on_frame_processed`,
    :meth:`_finalize`). This makes the invariants structurally unforgeable — a
    stage can never again forget to freeze the UI on run (the B-6 bug) or invent
    a divergent cancel, because the base owns the sequence and unfreezes on
    *every* terminal path: a run's own ``finished`` signal (success/error), and
    ``cancel()``, which frees the UI inline (see :meth:`cancel`).

    Preview and any one-shot actions stay per-stage and synchronous — they do not
    go through this template (only the full streaming run does).
    """

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)  # Results object
    analysis_failed = Signal(str)  # Error message
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    # Public lifecycle methods a subclass may not redefine; it fills hooks instead.
    _SEALED_METHODS = ("run", "cancel")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in BaseAnalysisController._SEALED_METHODS:
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} may not override the sealed lifecycle method "
                    f"{name!r} — implement the hook methods instead."
                )

    def __init__(self, viewer, data_manager, parameter_manager, visualization_manager):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []

        # Live-streaming progress counters: a run fills the output stacks frame
        # by frame, so progress is "frames done / total".
        self._stream_total = 0
        self._stream_done = 0

    def freeze_ui(self):
        """Signal the owning widget to disable interactive controls (idempotent)."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal the owning widget to re-enable controls (idempotent)."""
        self.ui_frozen.emit(False)

    def _handle_error(self, error_msg: str):
        """Surface an error: blank the progress, announce failure, alert."""
        error_msg = str(error_msg)
        self.progress_updated.emit(0, f"Error: {error_msg}")
        self.analysis_failed.emit(error_msg)
        QMessageBox.critical(None, "Error", error_msg)

    # ------------------------------------------------------------------
    # The sealed run/cancel lifecycle (template method)
    # ------------------------------------------------------------------
    def run(self):
        """Start a full streaming run. Sealed — subclasses fill the hooks."""
        self.__run()

    def __run(self):
        """The invariant sequence. Name-mangled so no subclass can redirect it.

        Freeze happens synchronously; unfreeze is guaranteed on a synchronous
        setup failure (validate/build raises before the worker owns the
        lifecycle — nothing would ever emit ``finished``), on the async terminal
        states of a run (success, error) via the worker's ``finished`` signal,
        and on :meth:`cancel`, which frees the UI inline.
        """
        self.freeze_ui()
        worker = None
        try:
            self._validate()                       # hook — raises on bad prerequisites
            params = self._run_params()            # hook — the stage's parameters
            self._begin_stream(params)             # hook — allocate live layers, set _stream_total
            worker = self._build_worker(params)    # hook — configured, not started
            worker.yielded.connect(self._on_frame_processed)   # hook slot
            worker.returned.connect(self._on_run_returned)
            worker.errored.connect(self._on_run_errored)
            worker.finished.connect(partial(self._on_run_finished, worker))
            self.active_workers.append(worker)
            self.analysis_started.emit()
            worker.start()                         # keep LAST
        except Exception as exc:
            if worker is not None:
                self._forget_worker(worker)
            self._handle_error(str(exc))
            self.unfreeze_ui()

    def _on_run_returned(self, result):
        """Store the final result (GUI thread). A throwing hook must not skip unfreeze."""
        try:
            self._finalize(result)                 # hook
        except Exception as exc:
            self._handle_error(str(exc))

    def _on_run_errored(self, exc):
        self._handle_error(str(exc))

    def _on_run_finished(self, worker):
        """Unfreeze once the last worker of a run has stopped.

        Reached on a run's terminal path (success/error) and on the *late* finish
        of a worker that ``cancel()`` already orphaned. Unfreeze is guarded on no
        workers remaining, so a cancelled run's late finish can't unfreeze the UI
        out from under a fresh run the user has since started.
        """
        self._forget_worker(worker)
        if not self.active_workers:
            self.unfreeze_ui()

    def _forget_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)

    def cancel(self):
        """Cancel a running stage. Sealed. Frees the UI *immediately* — it does
        not wait for the in-flight frame to finish.

        napari's ``@thread_worker`` exposes only cooperative ``quit()`` (no
        ``wait``/``terminate``), and abort is checked between frames, so a heavy
        frame already running cannot be interrupted mid-compute. Rather than leave
        the UI frozen until that frame ends (which reads as "cancel is ignored"),
        we ask the worker to stop, **orphan** it — disconnecting ``yielded`` /
        ``returned`` / ``errored`` so no late frame, result, or error can reach a
        stage we're tearing down — forget it, and unfreeze now. Its one remaining
        frame runs to completion and is discarded in the background; ``finished``
        still fires later but is a guarded no-op (see :meth:`_on_run_finished`).
        """
        for worker in list(self.active_workers):
            for signal_name, slot in (
                ("yielded", self._on_frame_processed),
                ("returned", self._on_run_returned),
                ("errored", self._on_run_errored),
            ):
                try:
                    getattr(worker, signal_name).disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass
            try:
                worker.quit()
            except (RuntimeError, AttributeError):
                pass
            self._forget_worker(worker)
        self._on_cancel_cleanup()
        self.progress_updated.emit(0, "Operation cancelled")
        self.unfreeze_ui()

    # ------------------------------------------------------------------
    # Async single-shot previews (NOT the sealed run template)
    # ------------------------------------------------------------------
    def _start_preview_worker(self, worker, on_result, *, status="Calculating preview..."):
        """Run a single-shot preview off the GUI thread, reusing the run
        lifecycle's terminal-state plumbing.

        A preview is a full single-frame solve; its cost occasionally spikes far
        above the usual (GPU contention, a cold kernel, allocator churn), and run
        inline that spell freezes the whole window. Off-thread it is a moving
        progress bar the user can cancel instead.

        ``worker`` is a configured-but-unstarted ``@thread_worker`` that does ONLY
        the (thread-safe) compute and returns its result — it must not touch napari.
        ``on_result`` is the GUI-thread callback that receives that result and does
        the visualization + stats; it runs via the worker's ``returned`` signal, so
        it is the only place layers may be mutated, and a throw in it is surfaced
        like any run error rather than crashing the thread. Unfreeze and error
        handling reuse the run path's :meth:`_on_run_finished` / :meth:`_on_run_errored`;
        because the worker is registered in ``active_workers``, :meth:`cancel` stops a
        preview too (an already-running compute finishes in the background and is
        discarded, exactly as an in-flight run frame is). Re-entry needs no guard:
        :meth:`freeze_ui` disables the preview button for the worker's lifetime.
        """
        self.freeze_ui()

        def _returned(result, _cb=on_result):
            try:
                _cb(result)
            except Exception as exc:  # a failing visualization must not skip unfreeze
                self._handle_error(str(exc))

        worker.returned.connect(_returned)
        worker.errored.connect(self._on_run_errored)
        worker.finished.connect(partial(self._on_run_finished, worker))
        self.active_workers.append(worker)
        self.progress_updated.emit(0, status)
        worker.start()

    # ------------------------------------------------------------------
    # Hooks — subclasses that use run()/cancel() implement these.
    # ------------------------------------------------------------------
    def _validate(self):
        """Raise (with a user-facing message) if prerequisites are missing."""
        raise NotImplementedError

    def _run_params(self):
        """Return the stage's parameter object for this run."""
        raise NotImplementedError

    def _begin_stream(self, params):
        """Allocate the live output layers and set ``_stream_total``."""
        raise NotImplementedError

    def _build_worker(self, params):
        """Return a configured-but-unstarted napari ``@thread_worker``."""
        raise NotImplementedError

    def _on_frame_processed(self, payload):
        """GUI-thread slot: stream one ``(frame_index, total, field)`` into the viewer."""
        raise NotImplementedError

    def _finalize(self, result):
        """Store the final result object and announce completion."""
        raise NotImplementedError

    def _on_cancel_cleanup(self):
        """Optional stage-specific cleanup, run inline when the stage is cancelled."""


class VectorStageController(BaseAnalysisController):
    """Shared controller for the two vector-field stages (displacement, force).

    These stages differ only in a small spec — which layers they stream, which
    parameters they read, which backend they call, and which result they store —
    so the entire run lifecycle lives here and the concrete controllers
    (:class:`DisplacementController`, :class:`FTTCController`) are thin subclasses
    that set the class attributes and fill a few hooks. This is what
    ``_VECTOR_FIELD_CONFIG`` already did for the rendering layer, applied to the
    controllers.

    Subclasses set:
      ``STAGE_KIND``     — ``'displacement'`` | ``'force'`` (the stream key)
      ``RESULT_SETTER``  — the ``DataManager`` method storing the final result

    and implement: :meth:`_validate`, :meth:`_run_params`, :meth:`_stream_frame_count`,
    :meth:`_vis_params`, :meth:`_build_worker`.
    """

    STAGE_KIND: str = ""
    RESULT_SETTER: str = ""

    def _stream_frame_count(self) -> int:
        """Number of frames this run will produce (drives the progress denominator)."""
        raise NotImplementedError

    def _vis_params(self, params) -> dict:
        """Map stage parameters to the streaming visualization params dict."""
        raise NotImplementedError

    def _begin_stream(self, params):
        """Allocate the live magnitude+vectors layers and reset the frame counters."""
        self._stream_total = self._stream_frame_count()
        self._stream_done = 0
        self.visualization_manager.begin_vector_field_stream(
            self.STAGE_KIND, self._stream_total, self._vis_params(params)
        )

    def _on_frame_processed(self, payload):
        """Stream one freshly computed vector-field frame into the viewer (GUI thread)."""
        frame_index, total, field = payload
        self._stream_done += 1
        progress = int(self._stream_done / max(self._stream_total, 1) * 100)
        self.progress_updated.emit(
            progress, f"Processing frame {frame_index + 1}/{total}"
        )
        self.visualization_manager.stream_vector_field_frame(
            self.STAGE_KIND, frame_index, field
        )

    def _finalize(self, result):
        """Store the streamed result; the live layers already reflect it.

        The magnitude stack and vector cache were filled in place as frames
        arrived, so there is nothing to assemble — just commit the full result for
        downstream steps. Other layers' visibility is left untouched (the user's).
        """
        if result is None:
            raise RuntimeError("Analysis failed to produce results")
        getattr(self.data_manager, self.RESULT_SETTER)(result, dirty=True)
        self.progress_updated.emit(100, "Analysis completed successfully")
        self.analysis_completed.emit(result)

    def _current_frame(self) -> int:
        """The frame a preview targets: 0 for a 2D view, else the slider index."""
        if len(self.viewer.dims.current_step) == 2:
            self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            return 0
        return self.viewer.dims.current_step[0]


class BaseAnalysisWidget(QWidget):
    """Base class for analysis widgets providing shared construction.

    Holds the managers, the header-proxied action enablement contract
    (``action_states`` / ``action_states_changed``), and the default
    freeze/unfreeze and cleanup behaviour. Subclasses set ``_action_enabled``
    with their own action keys and implement ``_update_ui_state``.
    """

    action_states_changed = Signal()

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: Optional["DataManager"] = None,
            parameter_manager: Optional["ParameterManager"] = None,
            visualization_manager: Optional["VisualizationManager"] = None
    ):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        # Per-action enablement consumed by the stage header (StageSection);
        # subclasses populate it with their action keys.
        self._action_enabled = {}

    def action_states(self):
        """Return a copy of the current per-action enablement map."""
        return dict(self._action_enabled)

    def _update_ui_state(self, event=None):
        """Recompute action enablement from current data; subclasses override."""

    def _handle_ui_freeze(self, frozen: bool):
        """Disable every action but cancel while frozen; restore on unfreeze.

        Unfreeze defers to ``_update_ui_state`` so enablement is re-derived from
        the data that now exists, rather than blindly re-enabling everything.
        """
        if frozen:
            for key in self._action_enabled:
                self._action_enabled[key] = False
            self._action_enabled["cancel"] = True
            self.action_states_changed.emit()
        else:
            self._update_ui_state()

    def cleanup(self):
        """Clean up resources before the widget is destroyed.

        Disconnects the frame-change handler if the subclass wired one, so the
        common case needs no override.
        """
        viewer = getattr(self, "viewer", None)
        handler = getattr(self, "_on_frame_changed", None)
        if viewer is not None and handler is not None:
            try:
                viewer.dims.events.current_step.disconnect(handler)
            except (TypeError, RuntimeError):
                pass
