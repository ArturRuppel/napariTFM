import logging
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QMessageBox, QTabWidget, QSizePolicy
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

    def _connect_parameter_sync(self):
        """Connect parameter synchronization between widgets."""
        # Sync preprocessing parameters
        if hasattr(self.preprocessing_widget, 'pixel_spin'):
            self.preprocessing_widget.pixel_spin.valueChanged.connect(
                lambda val: self._sync_pixel_size(val, source='preprocessing')
            )

        # Sync displacement parameters
        if hasattr(self.displacement_widget, 'pixel_spin'):
            self.displacement_widget.pixel_spin.valueChanged.connect(
                lambda val: self._sync_pixel_size(val, source='displacement')
            )

        # Sync force parameters
        if hasattr(self.force_widget, 'pixel_spin'):
            self.force_widget.pixel_spin.valueChanged.connect(
                lambda val: self._sync_pixel_size(val, source='force')
            )

        # Sync MSM parameters
        if hasattr(self.msm_widget, 'parameter_spins'):
            for param_name, spin in self.msm_widget.parameter_spins.items():
                spin.valueChanged.connect(
                    lambda val, name=param_name: self._sync_parameter(name, val, source='msm')
                )

        # Sync batch widget parameters
        if hasattr(self.batch_widget, 'parameter_spins'):
            for param_name, spin in self.batch_widget.parameter_spins.items():
                spin.valueChanged.connect(
                    lambda val, name=param_name: self._sync_parameter(name, val, source='batch')
                )

    def _sync_pixel_size(self, value: float, source: str):
        """Synchronize pixel size across all widgets."""
        widgets = {
            'preprocessing': self.preprocessing_widget,
            'displacement': self.displacement_widget,
            'force': self.force_widget,
            'msm': self.msm_widget,
            'batch': self.batch_widget
        }

        # Update pixel size in all widgets except source
        for name, widget in widgets.items():
            if name != source and hasattr(widget, 'pixel_spin'):
                widget.pixel_spin.blockSignals(True)
                widget.pixel_spin.setValue(value)
                widget.pixel_spin.blockSignals(False)

            # Special handling for batch widget
            if name != source and name == 'batch' and hasattr(widget, 'parameter_spins'):
                if 'pixelsize' in widget.parameter_spins:
                    widget.parameter_spins['pixelsize'].blockSignals(True)
                    widget.parameter_spins['pixelsize'].setValue(value)
                    widget.parameter_spins['pixelsize'].blockSignals(False)

    def _sync_parameter(self, param_name: str, value: float, source: str):
        """Synchronize parameters between widgets."""
        # Map of parameter names between widgets
        param_mapping = {
            'youngs_modulus': ['young_spin'],
            'poisson_ratio': ['poisson_spin'],
            'target_nodes': ['target_nodes_spin'],
            'boundary_refinement': ['boundary_refinement_spin'],
            # Add more parameter mappings as needed
        }

        # Update parameters in relevant widgets
        widgets = {
            'force': self.force_widget,
            'msm': self.msm_widget,
            'batch': self.batch_widget
        }

        for widget_name, widget in widgets.items():
            if widget_name != source:
                # Handle batch widget parameters
                if widget_name == 'batch' and hasattr(widget, 'parameter_spins'):
                    if param_name in widget.parameter_spins:
                        widget.parameter_spins[param_name].blockSignals(True)
                        widget.parameter_spins[param_name].setValue(value)
                        widget.parameter_spins[param_name].blockSignals(False)

                # Handle other widgets
                elif param_name in param_mapping:
                    for mapped_name in param_mapping[param_name]:
                        if hasattr(widget, mapped_name):
                            getattr(widget, mapped_name).blockSignals(True)
                            getattr(widget, mapped_name).setValue(value)
                            getattr(widget, mapped_name).blockSignals(False)

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

    def connect_signals(self):
        """Connect signals between components"""
        # Connect preprocessing signals
        self.preprocessing_widget.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.preprocessing_widget.processing_failed.connect(self._on_preprocessing_failed)

        # Connect displacement signals
        self.displacement_widget.displacement_calculated.connect(self._on_displacement_completed)

        # Connect force calculation signals
        self.force_widget.force_calculated.connect(self._on_force_completed)

        # Connect MSM signals
        self.msm_widget.stress_calculated.connect(self._on_stress_completed)

