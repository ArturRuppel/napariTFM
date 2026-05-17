import logging
from typing import Optional

import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget, QTabWidget

from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class BaseAnalysisWidget(QWidget):
    """Base class for analysis widgets with common functionality."""

    # Common signals
    parameters_updated = Signal()
    processing_started = Signal()
    processing_completed = Signal()
    processing_failed = Signal(str)  # Error message

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: Optional["DataManager"] = None,
            visualization_manager: Optional["VisualizationManager"] = None
    ):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self.visualization_manager = visualization_manager
        self._controls = []


    def get_parent_tfm_widget(self) -> Optional["napariTFMWidget"]:
        """Traverse up the widget hierarchy to find the parent napariTFMWidget."""
        current = self.parent()
        while current is not None:
            # First check if this is the napariTFMWidget directly
            if hasattr(current, 'pixel_spin') and hasattr(current, 'frame_spin'):
                return current
            # Then check if this is the tab widget
            if isinstance(current, QTabWidget):
                # The tab widget's parent should be the napariTFMWidget
                parent = current.parent()
                if hasattr(parent, 'pixel_spin') and hasattr(parent, 'frame_spin'):
                    return parent
            current = current.parent()
        return None

    def register_control(self, control):
        """Register a UI control for common operations like enable/disable."""
        self._controls.append(control)

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all registered controls."""
        for control in self._controls:
            control.setEnabled(enabled)

    def _get_active_image_layer(self) -> Optional["napari.layers.Image"]:
        """Get the currently active image layer."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None or not isinstance(active_layer, napari.layers.Image):
            return None
        return active_layer

    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status message and progress bar if available."""
        if hasattr(self, 'status_label'):
            self.status_label.setText(message)
        if hasattr(self, 'progress_bar') and progress is not None:
            self.progress_bar.setValue(progress)

    def _handle_error(self, error):
        """Handle processing errors."""
        error_msg = str(error)
        logger.error(f"Processing error: {error_msg}")
        self._update_status(f"Error: {error_msg}")
        self.processing_failed.emit(error_msg)

    def cleanup(self):
        """Clean up resources before widget is destroyed."""
        pass
