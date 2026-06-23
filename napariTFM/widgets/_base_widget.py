from typing import Optional

import napari
from qtpy.QtWidgets import QWidget

from napariTFM.utilities.data_manager import DataManager
from napariTFM.utilities.visualization_manager import VisualizationManager


class BaseAnalysisWidget(QWidget):
    """Base class for analysis widgets providing shared construction."""

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

    def cleanup(self):
        """Clean up resources before widget is destroyed."""
        pass
