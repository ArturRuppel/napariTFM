import logging
from typing import Any

import napari
from qtpy.QtCore import Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QTabWidget, QSizePolicy, QDoubleSpinBox, QGroupBox, QHBoxLayout, QPushButton, QSpinBox, QComboBox, QFileDialog
)

from .batch_analysis_widget import BatchAnalysisWidget
from .displacement_analysis_widget import DisplacementAnalysisWidget
from .fttc_widget import FTTCWidget
from .parameter_manager import ParameterManager
from .preprocessing_widget import PreprocessingWidget
from .data_manager import DataManager
from .visualization_manager import VisualizationManager
from .msm_widget import MSMWidget

logger = logging.getLogger(__name__)


class SpinBoxEventFilter(QObject):
    def eventFilter(self, obj, event):
        # Check for all spinnable input widgets
        if (isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox)) and
                event.type() == event.Wheel):
            if not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)


class napariTFMWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer

        # Create and install event filter
        self.spinbox_filter = SpinBoxEventFilter(self)

        # Find and filter all spinboxes in the application
        def install_filter_on_inputs():
            for widget in self.window().findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
                widget.installEventFilter(self.spinbox_filter)
                widget.setFocusPolicy(Qt.StrongFocus)

        # Install filters after a short delay to ensure all widgets are created
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, install_filter_on_inputs)

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

        # Set size policy for container
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Initialize managers
        self.data_manager = DataManager()
        self.parameter_manager = ParameterManager()
        self.visualization_manager = VisualizationManager(self.viewer, self.data_manager)

        # Create calibration group
        calibration_group = self._create_general_group()
        container_layout.addWidget(calibration_group)

        # Create tab widget for different components
        tabs = QTabWidget()

        # Initialize all widgets with parameter_manager
        self.preprocessing_widget = PreprocessingWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager,

        )
        self.preprocessing_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.displacement_widget = DisplacementAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        self.force_widget = FTTCWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        self.msm_widget = MSMWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
            self.visualization_manager
        )

        self.batch_widget = BatchAnalysisWidget(
            self.viewer,
            self.data_manager,
            self.parameter_manager,
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

    def _create_general_group(self) -> QGroupBox:
        """Create calibration group box with controls."""
        calibration_group = QGroupBox("")
        calibration_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        calibration_layout = QVBoxLayout()

        # First row: spinboxes
        spinbox_layout = QHBoxLayout()

        # Pixel size controls
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Size (µm):"))
        self.pixel_spin = QDoubleSpinBox()
        self.pixel_spin.setRange(0.001, 100.0)
        self.pixel_spin.setSingleStep(0.1)
        self.pixel_spin.setDecimals(2)
        pixel_layout.addWidget(self.pixel_spin)
        spinbox_layout.addLayout(pixel_layout)

        spinbox_layout.addSpacing(20)

        # Frame length controls
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame Length (min):"))
        self.frame_spin = QDoubleSpinBox()
        self.frame_spin.setRange(0.001, 1000.0)
        self.frame_spin.setSingleStep(0.1)
        self.frame_spin.setDecimals(1)
        frame_layout.addWidget(self.frame_spin)
        spinbox_layout.addLayout(frame_layout)

        # Register callbacks for parameter changes
        self.parameter_manager.register_callback('pixel_size',
                                                 lambda value: self.pixel_spin.setValue(value)
                                                 )
        self.parameter_manager.register_callback('frame_interval',
                                                 lambda value: self.frame_spin.setValue(value)
                                                 )

        # Connect spinbox signals to parameter manager
        self.pixel_spin.valueChanged.connect(
            lambda value: self.parameter_manager.set_value('pixel_size', value)
        )
        self.frame_spin.valueChanged.connect(
            lambda value: self.parameter_manager.set_value('frame_interval', value)
        )

        # Initialize spinbox values from parameter manager
        self.pixel_spin.setValue(self.parameter_manager.get_value('pixel_size'))
        self.frame_spin.setValue(self.parameter_manager.get_value('frame_interval'))

        spinbox_layout.addStretch()
        calibration_layout.addLayout(spinbox_layout)

        # Create 2x2 button grid
        button_grid = QVBoxLayout()

        # First row of buttons
        button_row1 = QHBoxLayout()
        self.save_params_btn = QPushButton("Save Parameters")
        self.load_params_btn = QPushButton("Load Parameters")
        button_row1.addWidget(self.save_params_btn)
        button_row1.addWidget(self.load_params_btn)

        # Second row of buttons
        button_row2 = QHBoxLayout()
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.clear_data_btn = QPushButton("Clear All Data")
        self.clear_data_btn.setStyleSheet("color: red;")
        button_row2.addWidget(self.reset_params_btn)
        button_row2.addWidget(self.clear_data_btn)

        # Add button rows to grid
        button_grid.addLayout(button_row1)
        button_grid.addLayout(button_row2)

        # Connect button signals
        self.save_params_btn.clicked.connect(self._save_parameters)
        self.load_params_btn.clicked.connect(self._load_parameters)
        self.reset_params_btn.clicked.connect(self._reset_parameters)
        self.clear_data_btn.clicked.connect(self._clear_all_data)

        calibration_layout.addLayout(button_grid)
        calibration_group.setLayout(calibration_layout)
        return calibration_group

    def _reset_parameters(self):
        """Reset parameters to default values and notify all widgets."""
        try:
            reply = QMessageBox.question(
                self,
                "Confirm",
                "Are you sure you want to reset all parameters to default values?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Reset parameters
                self.parameter_manager.reset_to_defaults()

                # Update all widget states
                for widget in [
                    self.preprocessing_widget,
                    self.displacement_widget,
                    self.force_widget,
                    self.msm_widget,
                    self.batch_widget
                ]:
                    if hasattr(widget, '_update_ui_state'):
                        widget._update_ui_state()

        except Exception as e:
            logger.error(f"Error resetting parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to reset parameters: {str(e)}")

    def _save_parameters(self):
        """Save parameters using parameter manager."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Parameters",
                "",
                "YAML Files (*.yaml *.yml)"
            )
            if file_path:
                if not file_path.lower().endswith(('.yaml', '.yml')):
                    file_path += '.yaml'
                self.parameter_manager.save_to_file(file_path)
                QMessageBox.information(self, "Success", "Parameters saved successfully!")
        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {str(e)}")

    def _load_parameters(self):
        """Load parameters using parameter manager."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Parameters",
                "",
                "YAML Files (*.yaml *.yml)"
            )
            if file_path:
                self.parameter_manager.load_from_file(file_path)
                QMessageBox.information(self, "Success", "Parameters loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

    def connect_signals(self):
        """Connect signals between components"""
        # Existing signal connections
        self.preprocessing_widget.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.preprocessing_widget.processing_failed.connect(self._on_preprocessing_failed)
        self.displacement_widget.displacement_calculated.connect(self._on_displacement_completed)
        self.force_widget.force_calculated.connect(self._on_force_completed)
        self.msm_widget.stress_calculated.connect(self._on_stress_completed)

        # Connect parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes from parameter manager"""
        # For calibration parameters, we need to update all widgets
        if param_name in ['pixel_size', 'frame_interval']:
            for widget in [
                self.preprocessing_widget,
                self.displacement_widget,
                self.force_widget,
                self.msm_widget,
                self.batch_widget
            ]:
                # Use _update_calibration instead of _update_parameters
                if hasattr(widget, '_update_calibration'):
                    widget._update_calibration()
                # Or just update the UI state if _update_calibration doesn't exist
                elif hasattr(widget, '_update_ui_state'):
                    widget._update_ui_state()

        # Let individual widgets handle their specific parameters if needed
        try:
            if param_name.startswith('preprocessing_'):
                self.preprocessing_widget._update_ui_state()
            elif param_name.startswith('displacement_'):
                self.displacement_widget._update_ui_state()
            elif param_name.startswith('force_'):
                self.force_widget._update_ui_state()
            elif param_name.startswith('stress_'):
                self.msm_widget._update_ui_state()
        except AttributeError:
            pass

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
        self.displacement_widget._update_ui_state()

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
