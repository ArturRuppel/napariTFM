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
    *every* terminal path through the worker's ``finished`` signal.

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
        self._cancelling = False

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

        Freeze happens synchronously; unfreeze is guaranteed on *both* a
        synchronous setup failure (validate/build raises before the worker owns
        the lifecycle — nothing would ever emit ``finished``) and every async
        terminal state (success, error, cooperative cancel), all funnelled
        through the worker's ``finished`` signal.
        """
        self.freeze_ui()
        self._cancelling = False
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
        """The single unfreeze chokepoint — reached on every terminal path."""
        self._forget_worker(worker)
        if self._cancelling and not self.active_workers:
            self._on_cancel_cleanup()              # optional hook
            self._cancelling = False
        self.unfreeze_ui()

    def _forget_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)

    def cancel(self):
        """Cancel a running stage. Sealed. Cooperative + async: ask the worker to
        stop and let the existing ``finished`` teardown run — the GUI thread never
        blocks (napari ``@thread_worker`` exposes only cooperative ``quit()``; it
        has no ``wait``/``terminate``). Late frames are disconnected so they can't
        race into a torn-down stage.
        """
        if self.active_workers:
            self._cancelling = True
            for worker in list(self.active_workers):
                try:
                    worker.yielded.disconnect(self._on_frame_processed)
                except (TypeError, RuntimeError, AttributeError):
                    pass
                try:
                    worker.quit()
                except (RuntimeError, AttributeError):
                    pass
            self.progress_updated.emit(0, "Operation cancelled")
            # Teardown + _on_cancel_cleanup + unfreeze run in _on_run_finished.
        else:
            # Nothing running: no `finished` will fire, so finish the job inline.
            self._on_cancel_cleanup()
            self.progress_updated.emit(0, "Operation cancelled")
            self.unfreeze_ui()

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
        """Optional stage-specific cleanup, run after the worker has stopped."""


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
