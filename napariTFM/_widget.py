import logging
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QTabWidget, QSizePolicy
)

from .preprocessing_widget import PreprocessingWidget
from .data_manager import DataManager
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)

class napariTFMWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer

        # Create scroll area for widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create container widget for scroll area
        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        # Add title
        title = QLabel("napariTFM")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        container_layout.addWidget(title)

        # Set size policy for container to prevent vertical stretching
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Initialize managers
        self.data_manager = DataManager()
        self.visualization_manager = VisualizationManager(self.viewer, self.data_manager)

        # Create tab widget for different components
        tabs = QTabWidget()

        # Initialize preprocessing widget
        self.preprocessing_widget = PreprocessingWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )
        self.preprocessing_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Add widgets to tabs
        tabs.addTab(self.preprocessing_widget, "Preprocessing")

        # Add tabs to container
        container_layout.addWidget(tabs)

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Add scroll area to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.connect_signals()

    def connect_signals(self):
        """Connect signals between components"""
        # Connect preprocessing signals
        self.preprocessing_widget.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.preprocessing_widget.processing_failed.connect(self._on_preprocessing_failed)

    def _on_preprocessing_completed(self, processed_stack, preprocessing_info):
        """Handle completion of preprocessing"""
        logger.info("Preprocessing completed successfully")
        self.data_manager.preprocessed_data = processed_stack

    def _on_preprocessing_failed(self, error_msg):
        """Handle preprocessing failure"""
        logger.error(f"Preprocessing failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Preprocessing failed: {error_msg}")