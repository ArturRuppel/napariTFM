from typing import Optional, Tuple

import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from qtpy.QtCore import Signal
import napari
import logging

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

    def create_colorbar_widget(
            self,
            colormap_name: str,
            label: str,
            clim: Tuple[float, float],
            colorbar_manager: "ColorbarManager"
    ) -> QGroupBox:
        """Create a consistent colorbar widget with proper spacing.

        Parameters
        ----------
        colormap_name : str
            Name of the colormap to use (e.g., 'viridis', 'inferno')
        label : str
            Label for the colorbar
        clim : tuple
            Color limits (min, max) for the colorbar
        colorbar_manager : ColorbarManager
            Instance of the ColorbarManager class

        Returns
        -------
        QGroupBox
            A group box containing the colorbar widget
        """
        colorbar_group = QGroupBox()
        colorbar_group.setFixedWidth(100)
        colorbar_group.setFixedHeight(600)

        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(6, 20, 6, 20)  # Add padding top/bottom for tick labels

        colorbar_widget = colorbar_manager.create_colorbar(
            width=100,
            height=500,
            colormap_name=colormap_name,
            label=label,
            clim=clim,
            orientation='right',
            label_color='white',
            border_color='gray',
            border_width=1.0,
            padding=(0.1, 0.1),
            axis_ratio=0.075,
            label_offset=(-16, 0),
            label_rotation=0,
            tick_label_offset=(-49, -45)
        )

        colorbar_layout.addWidget(colorbar_widget)
        colorbar_group.setLayout(colorbar_layout)
        return colorbar_group

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

    @staticmethod
    def _validate_input_data(data):
        """Validate input data format."""
        if data is None:
            return False
        if not hasattr(data, 'shape'):
            return False
        if not (2 <= len(data.shape) <= 3):
            return False
        return True

    @staticmethod
    def _ensure_stack_format(data):
        """Ensure data is in 3D stack format."""
        if data.ndim == 2:
            return data[np.newaxis, ...]
        return data