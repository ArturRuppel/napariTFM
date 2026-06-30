from typing import Optional

import napari
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMessageBox, QWidget

from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.parameter_manager import ParameterManager
from napariTFM.utilities.visualization_manager import VisualizationManager


class BaseAnalysisController(QObject):
    """Shared base for the per-stage analysis controllers.

    Owns the signal set, manager wiring, worker bookkeeping, and the
    freeze/error helpers every stage controller repeats. Stage controllers add
    their own run/preview/cancel logic on top; the shell drives them through
    this uniform surface (``progress_updated`` and ``ui_frozen``).
    """

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)  # Results object
    analysis_failed = Signal(str)  # Error message
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

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
        """Signal the owning widget to disable interactive controls."""
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Signal the owning widget to re-enable controls."""
        self.ui_frozen.emit(False)

    def _handle_error(self, error_msg: str):
        """Surface an error: blank the progress, announce failure, alert."""
        error_msg = str(error_msg)
        self.progress_updated.emit(0, f"Error: {error_msg}")
        self.analysis_failed.emit(error_msg)
        QMessageBox.critical(None, "Error", error_msg)


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
