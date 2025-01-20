import logging
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QTabWidget, QSizePolicy, QDoubleSpinBox, QGroupBox, QHBoxLayout, QPushButton
)

from .batch_analysis_widget import BatchAnalysisWidget
from .displacement_analysis_widget import DisplacementAnalysisWidget
from .fttc_widget import FTTCWidget
from .preprocessing_widget import PreprocessingWidget
from .data_manager import DataManager
from .visualization_manager import VisualizationManager
from .msm_widget import MSMWidget

logger = logging.getLogger(__name__)


class napariTFMWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer

        # Set fixed width for entire widget
        self.setFixedWidth(530)

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

        # Create calibration group
        calibration_group = QGroupBox("")
        calibration_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        calibration_layout = QVBoxLayout()  # Main layout is vertical

        # First row: spinboxes
        spinbox_layout = QHBoxLayout()

        # Pixel size controls
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Size (µm):"))
        self.pixel_spin = QDoubleSpinBox()
        self.pixel_spin.setRange(0.001, 100.0)
        self.pixel_spin.setValue(1.0)  # Default value
        self.pixel_spin.setSingleStep(0.1)
        self.pixel_spin.setDecimals(3)
        pixel_layout.addWidget(self.pixel_spin)
        spinbox_layout.addLayout(pixel_layout)

        spinbox_layout.addSpacing(20)  # Add some spacing between controls

        # Frame length controls
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame Length (min):"))
        self.frame_spin = QDoubleSpinBox()
        self.frame_spin.setRange(0.001, 1000.0)
        self.frame_spin.setValue(1.0)  # Default value
        self.frame_spin.setSingleStep(0.1)
        self.frame_spin.setDecimals(3)
        frame_layout.addWidget(self.frame_spin)
        spinbox_layout.addLayout(frame_layout)

        spinbox_layout.addStretch()  # Push spinboxes to the left
        calibration_layout.addLayout(spinbox_layout)

        # Second row: buttons
        button_layout = QHBoxLayout()

        # Save/load buttons and clear data button
        self.save_params_btn = QPushButton("Save Parameters")
        self.load_params_btn = QPushButton("Load Parameters")
        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setStyleSheet("color: red;")

        # Connect signals
        self.save_params_btn.clicked.connect(self._save_parameters)
        self.load_params_btn.clicked.connect(self._load_parameters)
        self.clear_data_btn.clicked.connect(self._clear_all_data)

        # Add buttons to layout with stretch
        button_layout.addWidget(self.save_params_btn, stretch=1)
        button_layout.addWidget(self.load_params_btn, stretch=1)
        button_layout.addWidget(self.clear_data_btn, stretch=1)

        calibration_layout.addLayout(button_layout)

        calibration_group.setLayout(calibration_layout)
        container_layout.addWidget(calibration_group)
        # Create tab widget for different components
        tabs = QTabWidget()

        # Initialize preprocessing widget
        self.preprocessing_widget = PreprocessingWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )
        self.preprocessing_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Initialize displacement widget
        self.displacement_widget = DisplacementAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        # Initialize force calculation widget
        self.force_widget = FTTCWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        # Initialize MSM widget
        self.msm_widget = MSMWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        # Initialize batch analysis widget
        self.batch_widget = BatchAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.visualization_manager
        )

        # Add widgets to tabs
        tabs.addTab(self.preprocessing_widget, "Preprocessing")
        tabs.addTab(self.displacement_widget, "Displacement")
        tabs.addTab(self.force_widget, "Force Analysis")
        tabs.addTab(self.msm_widget, "Stress Analysis")
        tabs.addTab(self.batch_widget, "Batch Analysis")

        # Add tabs to container
        container_layout.addWidget(tabs)

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Add scroll area to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.connect_signals()

    def _save_parameters(self):
        """
        Save current parameter values to a file.
        To be implemented: Save pixel size, frame length, and other relevant parameters.
        """
        logger.info("Save parameters functionality to be implemented")
        # TODO: Implement saving parameters to a file
        QMessageBox.information(self, "Info", "Save parameters functionality will be implemented here")

    def _load_parameters(self):
        """
        Load parameter values from a file.
        To be implemented: Load pixel size, frame length, and other relevant parameters.
        """
        logger.info("Load parameters functionality to be implemented")
        # TODO: Implement loading parameters from a file
        QMessageBox.information(self, "Info", "Load parameters functionality will be implemented here")

    def _clear_all_data(self):
        """
        Clear all data and reset the widget to its initial state.
        This includes clearing the data manager and resetting UI elements.
        """
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Are you sure you want to clear all data? This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Clear data manager
                self.data_manager.clear_all_data()

                # Update UI state in all widgets
                self.preprocessing_widget._update_ui_state()
                self.displacement_widget._update_ui_state()
                self.force_widget._update_ui_state()
                self.msm_widget._update_ui_state()
                self.batch_widget._update_ui_state()

                # # Reset all widget states
                # self.preprocessing_widget._reset_widget()
                # self.displacement_widget._reset_widget()
                # self.force_widget._reset_widget()
                # self.msm_widget._reset_widget()
                # self.batch_widget._reset_widget()

                # # Clear visualizations
                # self.visualization_manager.clear_all_layers()

                logger.info("All data cleared successfully")
                QMessageBox.information(self, "Success", "All data has been cleared successfully")

            except Exception as e:
                logger.error(f"Error clearing data: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to clear data: {str(e)}")

    def connect_signals(self):
        """Connect signals between components"""
        # Existing signal connections
        self.preprocessing_widget.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.preprocessing_widget.processing_failed.connect(self._on_preprocessing_failed)
        self.displacement_widget.displacement_calculated.connect(self._on_displacement_completed)
        self.force_widget.force_calculated.connect(self._on_force_completed)
        self.msm_widget.stress_calculated.connect(self._on_stress_completed)

        # Add new calibration signal connections
        self.pixel_spin.valueChanged.connect(self._on_calibration_changed)
        self.frame_spin.valueChanged.connect(self._on_calibration_changed)

    def _on_calibration_changed(self):
        """Handle changes to calibration values"""
        # Notify all widgets that calibration has changed
        for widget in [
            self.preprocessing_widget,
            self.displacement_widget,
            self.force_widget,
            self.msm_widget,
            self.batch_widget
        ]:
            if hasattr(widget, '_update_calibration'):
                widget._update_calibration()

    def _on_preprocessing_completed(self, results):
        """Handle completion of preprocessing"""
        logger.info("Preprocessing completed successfully")

        # Unpack results and update data manager
        if 'beads' in results:
            processed_data, preprocessing_info = results['beads']
            self.data_manager.preprocessed_bead_stack = processed_data
            self.data_manager.bead_preprocessing_info = preprocessing_info

        if 'reference' in results:
            processed_data, preprocessing_info = results['reference']
            self.data_manager.preprocessed_reference = processed_data
            self.data_manager.reference_preprocessing_info = preprocessing_info

        if 'cells' in results:
            processed_data, preprocessing_info = results['cells']
            self.data_manager.preprocessed_cell_stack = processed_data
            self.data_manager.cell_preprocessing_info = preprocessing_info

        # Update visualization through manager
        self.visualization_manager.update_preprocessing_visualization(results)

    def _on_preprocessing_failed(self, error_msg):
        """Handle preprocessing failure"""
        logger.error(f"Preprocessing failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Preprocessing failed: {error_msg}")

    def _on_displacement_completed(self, results):
        """Handle completion of displacement analysis"""
        logger.info("Displacement analysis completed successfully")
        self.data_manager.displacement_results = results
        self.force_widget._update_ui_state()
        self.msm_widget._update_ui_state()

    def _on_force_completed(self, results):
        """Handle completion of force calculation"""
        logger.info("Force calculation completed successfully")
        self.data_manager.force_results = results
        self.msm_widget._update_ui_state()

    def _on_stress_completed(self, results):
        """Handle completion of stress calculation"""
        logger.info("Stress calculation completed successfully")
        self.data_manager.stress_results = results
