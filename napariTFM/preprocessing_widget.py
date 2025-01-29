import os

import numpy as np
import tifffile
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QObject
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QGroupBox, QCheckBox, QDoubleSpinBox, QComboBox
)
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import (QSizePolicy, QScrollArea, QRadioButton, QFileDialog,
                            QWidget, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
                            QLabel, QPushButton
                            )
from qtrangeslider import QRangeSlider

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.data_manager_old import DataManager
from napariTFM.parameter_manager_old import ParameterManager
from napariTFM.services.preprocessing_service import PreprocessingService, PreprocessingParameters
from napariTFM.visualization_manager import VisualizationManager


class PreprocessingDataPanel(QWidget):
    """Panel for handling data loading and status display."""

    data_loaded = Signal(str)  # Emits data type that was loaded

    def __init__(self, data_manager, viewer):
        super().__init__()
        self.data_manager = data_manager
        self.viewer = viewer
        self.controller = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create data input group
        data_group = QGroupBox("Input Data")
        group_layout = QVBoxLayout()

        # Bead data row
        bead_layout = QHBoxLayout()
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.bead_status = QLabel("Not loaded")
        bead_layout.addWidget(self.load_beads_btn)
        bead_layout.addWidget(self.bead_status)
        group_layout.addLayout(bead_layout)

        # Reference data row
        ref_layout = QHBoxLayout()
        self.load_reference_btn = QPushButton("Load Reference Image")
        self.reference_status = QLabel("Not loaded")
        ref_layout.addWidget(self.load_reference_btn)
        ref_layout.addWidget(self.reference_status)
        group_layout.addLayout(ref_layout)

        # Cell data row
        cell_layout = QHBoxLayout()
        self.load_cells_btn = QPushButton("Load Cell Stack")
        self.cell_status = QLabel("Not loaded")
        cell_layout.addWidget(self.load_cells_btn)
        cell_layout.addWidget(self.cell_status)
        group_layout.addLayout(cell_layout)

        data_group.setLayout(group_layout)
        layout.addWidget(data_group)
        self.setLayout(layout)

    def set_controller(self, controller):
        """Set the controller and connect signals."""
        self.controller = controller
        self._connect_signals()

    def _connect_signals(self):
        """Connect button signals to controller methods."""
        self.load_beads_btn.clicked.connect(
            lambda: self.controller.load_active_layer('beads'))
        self.load_reference_btn.clicked.connect(
            lambda: self.controller.load_active_layer('reference'))
        self.load_cells_btn.clicked.connect(
            lambda: self.controller.load_active_layer('cells'))

    def update_button_states(self, active_layer_exists: bool = False):
        """Update button states based on layer selection."""
        active_layer = self.viewer.layers.selection.active
        has_valid_layer = (active_layer is not None and
                           isinstance(active_layer, Image))

        self.load_beads_btn.setEnabled(has_valid_layer)
        self.load_reference_btn.setEnabled(has_valid_layer)
        self.load_cells_btn.setEnabled(has_valid_layer)

    def update_data_status(self):
        """Update status labels based on loaded data."""
        # Update bead status
        bead_data = self.data_manager.input_bead_stack
        if bead_data is not None:
            self.bead_status.setText(f"Loaded: {bead_data.shape}")
        else:
            self.bead_status.setText("Not loaded")

        # Update reference status
        ref_data = self.data_manager.input_reference
        if ref_data is not None:
            self.reference_status.setText(f"Loaded: {ref_data.shape}")
        else:
            self.reference_status.setText("Not loaded")

        # Update cell status
        cell_data = self.data_manager.input_cell_stack
        if cell_data is not None:
            self.cell_status.setText(f"Loaded: {cell_data.shape}")
        else:
            self.cell_status.setText("Not loaded")


class PreprocessingParameterPanel(QWidget):
    """Panel for handling preprocessing parameter inputs."""

    parameter_changed = Signal()

    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_widgets = {}
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Add parameter groups
        layout.addWidget(self._create_intensity_range_group())
        layout.addWidget(self._create_cell_params_group())
        layout.addWidget(self._create_registration_group())

        # Add reset button
        self.reset_btn = QPushButton("Reset Parameters")
        layout.addWidget(self.reset_btn)

        self.setLayout(layout)

    def _create_intensity_range_group(self):
        """Create the bead/reference parameters group."""
        group = QGroupBox("Bead/Reference Parameters")
        layout = QVBoxLayout()

        # Add intensity slider
        self.intensity_slider = QRangeSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 1000)
        layout.addWidget(self.intensity_slider)

        # Add spinboxes
        spinbox_layout = QHBoxLayout()
        self.min_spinbox = QDoubleSpinBox()
        self.max_spinbox = QDoubleSpinBox()
        for spin in [self.min_spinbox, self.max_spinbox]:
            spin.setRange(0, 100)
            spin.setDecimals(1)
            spin.setSingleStep(0.1)
        spinbox_layout.addWidget(QLabel("Min"))
        spinbox_layout.addWidget(self.min_spinbox)
        spinbox_layout.addStretch()
        spinbox_layout.addWidget(self.max_spinbox)
        spinbox_layout.addWidget(QLabel("Max"))
        layout.addLayout(spinbox_layout)

        # Add gaussian controls
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Blur"))
        self.gaussian_sigma_spin = QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0, 10)
        self.gaussian_sigma_spin.setSingleStep(0.1)
        self.gaussian_sigma_spin.setDecimals(1)
        blur_layout.addWidget(self.gaussian_sigma_spin)
        layout.addLayout(blur_layout)

        group.setLayout(layout)
        return group

    def _create_cell_params_group(self):
        """Create the cell stack parameters group."""
        group = QGroupBox("Cell Stack Parameters")
        layout = QVBoxLayout()

        # Add intensity slider
        self.cell_intensity_slider = QRangeSlider(Qt.Horizontal)
        self.cell_intensity_slider.setRange(0, 1000)
        layout.addWidget(self.cell_intensity_slider)

        # Add spinboxes
        spinbox_layout = QHBoxLayout()
        self.cell_min_spinbox = QDoubleSpinBox()
        self.cell_max_spinbox = QDoubleSpinBox()
        for spin in [self.cell_min_spinbox, self.cell_max_spinbox]:
            spin.setRange(0, 100)
            spin.setDecimals(1)
            spin.setSingleStep(0.1)
        spinbox_layout.addWidget(QLabel("Min"))
        spinbox_layout.addWidget(self.cell_min_spinbox)
        spinbox_layout.addStretch()
        spinbox_layout.addWidget(self.cell_max_spinbox)
        spinbox_layout.addWidget(QLabel("Max"))
        layout.addLayout(spinbox_layout)

        # Add gaussian controls
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Blur"))
        self.cell_gaussian_sigma_spin = QDoubleSpinBox()
        self.cell_gaussian_sigma_spin.setRange(0, 10)
        self.cell_gaussian_sigma_spin.setSingleStep(0.1)
        self.cell_gaussian_sigma_spin.setDecimals(1)
        blur_layout.addWidget(self.cell_gaussian_sigma_spin)
        layout.addLayout(blur_layout)

        group.setLayout(layout)
        return group

    def _create_registration_group(self):
        """Create the registration group."""
        group = QGroupBox("Registration")
        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.addItems(['Translation', 'Rigid', 'No registration'])
        mode_layout.addWidget(self.registration_mode_combo)
        layout.addLayout(mode_layout)

        self.registration_note = QLabel(
            "Note: Registration will be performed relative to the reference image."
        )
        self.registration_note.setWordWrap(True)
        layout.addWidget(self.registration_note)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect widget signals to parameter manager."""
        # Connect intensity controls
        self.intensity_slider.valueChanged.connect(self._update_intensity_spinboxes)
        self.min_spinbox.valueChanged.connect(self._update_intensity_slider)
        self.max_spinbox.valueChanged.connect(self._update_intensity_slider)

        # Connect cell intensity controls
        self.cell_intensity_slider.valueChanged.connect(
            self._update_cell_intensity_spinboxes)
        self.cell_min_spinbox.valueChanged.connect(self._update_cell_intensity_slider)
        self.cell_max_spinbox.valueChanged.connect(self._update_cell_intensity_slider)

        # Connect blur controls
        self.gaussian_sigma_spin.valueChanged.connect(
            lambda v: self.parameter_manager.set_value('gaussian_sigma', v))
        self.cell_gaussian_sigma_spin.valueChanged.connect(
            lambda v: self.parameter_manager.set_value('cell_gaussian_sigma', v))

        # Connect registration mode
        self.registration_mode_combo.currentTextChanged.connect(
            lambda t: self.parameter_manager.set_value('registration_mode', t.lower()))

        # Connect reset button
        self.reset_btn.clicked.connect(self._reset_parameters)

    def _update_intensity_spinboxes(self, values):
        """Update intensity spinboxes from slider values."""
        min_val, max_val = values
        min_percent = float(min_val) / 10.0
        max_percent = float(max_val) / 10.0

        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(min_percent)
        self.max_spinbox.setValue(max_percent)
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)

        self.parameter_manager.set_value('min_intensity', min_percent)
        self.parameter_manager.set_value('max_intensity', max_percent)
        self.parameter_changed.emit()

    def _update_intensity_slider(self):
        """Update intensity slider from spinbox values."""
        min_val = int(self.min_spinbox.value() * 10)
        max_val = int(self.max_spinbox.value() * 10)

        self.intensity_slider.blockSignals(True)
        self.intensity_slider.setValue((min_val, max_val))
        self.intensity_slider.blockSignals(False)

        self.parameter_changed.emit()

    def _update_cell_intensity_spinboxes(self, values):
        """Update cell intensity spinboxes from slider values."""
        min_val, max_val = values
        min_percent = float(min_val) / 10.0
        max_percent = float(max_val) / 10.0

        self.cell_min_spinbox.blockSignals(True)
        self.cell_max_spinbox.blockSignals(True)
        self.cell_min_spinbox.setValue(min_percent)
        self.cell_max_spinbox.setValue(max_percent)
        self.cell_min_spinbox.blockSignals(False)
        self.cell_max_spinbox.blockSignals(False)

        self.parameter_manager.set_value('cell_min_intensity', min_percent)
        self.parameter_manager.set_value('cell_max_intensity', max_percent)
        self.parameter_changed.emit()

    def _update_cell_intensity_slider(self):
        """Update cell intensity slider from spinbox values."""
        min_val = int(self.cell_min_spinbox.value() * 10)
        max_val = int(self.cell_max_spinbox.value() * 10)

        self.cell_intensity_slider.blockSignals(True)
        self.cell_intensity_slider.setValue((min_val, max_val))
        self.cell_intensity_slider.blockSignals(False)

        self.parameter_changed.emit()

    def _reset_parameters(self):
        """Reset parameters to defaults."""
        self.parameter_manager.reset_preprocessing_parameters()
        self._sync_widgets_with_parameters()
        self.parameter_changed.emit()

    def _sync_widgets_with_parameters(self):
        """Sync widget values with current parameter values."""
        # Sync intensity values
        min_intensity = self.parameter_manager.get_value('min_intensity')
        max_intensity = self.parameter_manager.get_value('max_intensity')
        self.intensity_slider.setValue((int(min_intensity * 10), int(max_intensity * 10)))
        self.min_spinbox.setValue(min_intensity)
        self.max_spinbox.setValue(max_intensity)

        # Sync cell intensity values
        cell_min = self.parameter_manager.get_value('cell_min_intensity')
        cell_max = self.parameter_manager.get_value('cell_max_intensity')
        self.cell_intensity_slider.setValue((int(cell_min * 10), int(cell_max * 10)))
        self.cell_min_spinbox.setValue(cell_min)
        self.cell_max_spinbox.setValue(cell_max)

        # Sync blur values
        self.gaussian_sigma_spin.setValue(
            self.parameter_manager.get_value('gaussian_sigma'))
        self.cell_gaussian_sigma_spin.setValue(
            self.parameter_manager.get_value('cell_gaussian_sigma'))

        # Sync registration mode
        mode = self.parameter_manager.get_value('registration_mode')
        self.registration_mode_combo.setCurrentText(mode.title())


class PreprocessingController(QObject):
    """Coordinates interactions between UI components, service, and managers."""

    # Define signals
    progress_updated = Signal(int, str)  # (progress_value, status_message)
    preprocessing_started = Signal()
    preprocessing_completed = Signal(dict)  # Results dictionary
    preprocessing_failed = Signal(str)  # Error message
    preview_updated = Signal()
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    def __init__(self, viewer, service, data_manager, parameter_manager, visualization_manager):
        super().__init__()
        self.viewer = viewer
        self.service = service
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.active_workers = []

        # Initialize panel attributes as None
        self.parameter_panel = None
        self.data_panel = None
        self.preview_enabled = False
        self.current_data_type = 'beads'

    def set_panels(self, parameter_panel, data_panel):
        """Set the parameter and data panels after initialization."""
        self.parameter_panel = parameter_panel
        self.data_panel = data_panel

    def freeze_ui(self):
        """Disable all interactive UI elements"""
        if self.data_panel:
            self.data_panel.freeze_ui(True)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(True)
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Re-enable UI elements and refresh state"""
        if self.data_panel:
            self.data_panel.freeze_ui(False)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(False)
        self.ui_frozen.emit(False)

    def load_active_layer(self, data_type: str):
        """Load the currently active layer as the specified data type."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(None, "Error", "No active image layer found")
            return

        try:
            data = active_layer.data

            # Handle data based on type
            if data_type == 'beads':
                # Convert 2D data to 3D with single frame if needed
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                elif data.ndim != 3:
                    raise ValueError("Bead stack must be 2D or 3D (frames, height, width)")
                self.data_manager.set_input_bead_stack(data)

            elif data_type == 'reference':
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")
                self.data_manager.set_input_reference(data)

            elif data_type == 'cells':
                # Convert 2D data to 3D with single frame if needed
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                elif data.ndim != 3:
                    raise ValueError("Cell stack must be 2D or 3D (frames, height, width)")
                self.data_manager.set_input_cell_stack(data)
            else:
                raise ValueError(f"Invalid data type: {data_type}")

            # Update UI state and emit signal
            self.data_updated.emit(data_type)
            self._update_preview()

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode."""
        try:
            if enabled:
                self._update_preview()
            else:
                self.visualization_manager.handle_preview(
                    frame=None,
                    enable=False
                )
            self.preview_enabled = enabled

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))
            if self.parameter_panel:
                self.parameter_panel.preview_check.setChecked(False)
            self.preview_enabled = False

    def _update_preview(self):
        """Update preview with current frame."""
        if not self.preview_enabled:
            return

        try:
            # Get current data based on type
            if self.current_data_type == 'beads':
                data = self.data_manager.input_bead_stack
            elif self.current_data_type == 'reference':
                data = self.data_manager.input_reference
            else:
                data = self.data_manager.input_cell_stack

            if data is None:
                raise ValueError(f"No {self.current_data_type} data available")

            # Get current frame if data is a stack
            if data.ndim == 3:
                current_step = min(self.viewer.dims.current_step[0], data.shape[0] - 1)
                frame = data[current_step].copy()
            else:
                frame = data.copy()

            # Get parameters and create result
            params = self._get_current_parameters()
            self.service.update_parameters(params)

            # Process frame
            result = self.service.preprocess_frame(
                frame,
                is_cell=(self.current_data_type == 'cells')
            )

            # Update visualization
            self.visualization_manager.handle_preview(
                frame=result.processed_image,
                enable=True,
                layer_name='Preview'
            )

            # Update status with frame information
            info = result.info
            status = (
                f"Preview - Original range: ({info['original_range'][0]:.1f}, {info['original_range'][1]:.1f})\n"
                f"Applied range: {info['intensity_range']}\n"
                f"Mean: {info['final_mean']:.1f}, Std: {info['final_std']:.1f}"
            )
            self.progress_updated.emit(100, status)

        except Exception as e:
            self.progress_updated.emit(0, f"Preview failed: {str(e)}")
            self.visualization_manager.handle_preview(
                frame=None,
                enable=False
            )

    def run_preprocessing(self):
        """Run preprocessing on all available data."""
        try:
            # Disable preview if enabled
            if self.preview_enabled and self.parameter_panel:
                self.parameter_panel.preview_check.setChecked(False)

            self.preprocessing_started.emit()
            params = self._get_current_parameters()
            self.service.update_parameters(params)

            @thread_worker
            def preprocessing_worker():
                generator = self.service.preprocess_stack(
                    bead_stack=self.data_manager.input_bead_stack,
                    reference_image=self.data_manager.input_reference,
                    cell_stack=self.data_manager.input_cell_stack
                )

                try:
                    while True:
                        result, current_frame, total_frames = next(generator)
                        progress = int((current_frame + 1) / total_frames * 100)
                        yield progress, f"Processing frame {current_frame + 1}/{total_frames}"
                except StopIteration as e:
                    return e.value

            worker = preprocessing_worker()
            worker.running = True
            self.active_workers.append(worker)
            self.freeze_ui()

            def on_yielded(progress_data):
                progress, message = progress_data
                self.progress_updated.emit(progress, message)

            def on_returned(results):
                self._handle_preprocessing_results(results)
                self.active_workers.remove(worker)
                self.unfreeze_ui()

            def on_errored(exc):
                error_msg = f"Preprocessing failed: {str(exc)}"
                self.progress_updated.emit(0, error_msg)
                self.preprocessing_failed.emit(error_msg)
                QMessageBox.critical(None, "Error", error_msg)
                self.active_workers.remove(worker)
                self.unfreeze_ui()

            worker.yielded.connect(on_yielded)
            worker.returned.connect(on_returned)
            worker.errored.connect(on_errored)
            worker.start()

        except Exception as e:
            error_msg = f"Failed to start preprocessing: {str(e)}"
            self.progress_updated.emit(0, error_msg)
            self.preprocessing_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)

    def _handle_preprocessing_results(self, results):
        """Handle the results from preprocessing."""
        try:
            # Update data manager with processed results
            self.data_manager.set_preprocessing_results(
                bead_stack=results.get('beads', [None])[0],
                reference=results.get('reference', [None])[0],
                cell_stack=results.get('cells', [None])[0],
                params=self._get_current_parameters().__dict__
            )

            # Update visualization
            self.visualization_manager.update_preprocessing_visualization(results)

            self.progress_updated.emit(100, "Preprocessing complete")
            self.preprocessing_completed.emit(results)

        except Exception as e:
            error_msg = f"Error handling preprocessing results: {str(e)}"
            self.progress_updated.emit(0, error_msg)
            self.preprocessing_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)

    def _get_current_parameters(self) -> PreprocessingParameters:
        """Get current preprocessing parameters from parameter manager."""
        return PreprocessingParameters(
            min_intensity_percentile=self.parameter_manager.get_value('min_intensity') / 100,
            max_intensity_percentile=self.parameter_manager.get_value('max_intensity') / 100,
            enable_gaussian_filter=self.parameter_manager.get_value('gaussian_sigma') > 0,
            gaussian_sigma=self.parameter_manager.get_value('gaussian_sigma'),
            cell_min_intensity_percentile=self.parameter_manager.get_value('cell_min_intensity') / 100,
            cell_max_intensity_percentile=self.parameter_manager.get_value('cell_max_intensity') / 100,
            enable_cell_gaussian_filter=self.parameter_manager.get_value('cell_gaussian_sigma') > 0,
            cell_gaussian_sigma=self.parameter_manager.get_value('cell_gaussian_sigma'),
            registration_mode=self.parameter_manager.get_value('registration_mode')
        )

    def cancel_all_operations(self):
        """Cancel all running background operations."""
        for worker in self.active_workers:
            try:
                worker.running = False
                worker.quit()
                worker.wait(500)
                if worker.isRunning():
                    worker.terminate()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()
        self.progress_updated.emit(0, "Operations cancelled")
        self.unfreeze_ui()


class PreprocessingWidget(BaseAnalysisWidget):
    """Widget for controlling image preprocessing parameters and operations."""

    preprocessing_completed = Signal(dict)  # Emits dict of processed data types

    def __init__(
            self,
            viewer: Viewer,
            data_manager: DataManager,
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        # Initialize base widget first
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize service and panels
        self.service = PreprocessingService()
        self.parameter_panel = PreprocessingParameterPanel(parameter_manager)
        self.data_panel = PreprocessingDataPanel(data_manager, viewer)

        # Initialize controller
        self.controller = PreprocessingController(
            viewer=viewer,
            service=self.service,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager
        )

        # Set panels in controller
        self.controller.set_panels(self.parameter_panel, self.data_panel)
        self.data_panel.set_controller(self.controller)

        # Initialize colorbar manager
        self.colorbar_manager = ColorbarManager()

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left side: Colorbar
        colorbar_container = QWidget()
        colorbar_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(6, 6, 6, 6)

        # Create colorbar
        colorbar_group = self.create_colorbar_widget(
            colormap_name='gray',
            label="Intensity Value",
            clim=(255, 0),  # Reversed for better visualization
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        colorbar_layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        colorbar_layout.addStretch()
        colorbar_container.setLayout(colorbar_layout)

        main_layout.addWidget(colorbar_container)

        # Right side container with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        scroll_area.setFixedWidth(360)

        # Create scroll content widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(6, 6, 6, 6)

        # Add data and parameter panels
        scroll_layout.addWidget(self.data_panel)
        scroll_layout.addWidget(self.parameter_panel)

        # Add preview controls
        preview_frame = self._create_preview_frame()
        scroll_layout.addWidget(preview_frame)

        # Add action buttons
        action_frame = self._create_action_frame()
        scroll_layout.addWidget(action_frame)

        # Add status frame
        status_frame = self._create_status_frame()
        scroll_layout.addWidget(status_frame)

        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        # Initial UI update
        self._update_ui_state()

    def _create_preview_frame(self):
        """Create the preview control frame."""
        preview_frame = QFrame()
        preview_layout = QVBoxLayout()

        # Preview data type selection
        type_group = QGroupBox("Preview Data Type")
        type_layout = QHBoxLayout()

        self.bead_radio = QRadioButton("Bead Stack")
        self.reference_radio = QRadioButton("Reference")
        self.cell_radio = QRadioButton("Cell Stack")
        self.bead_radio.setChecked(True)

        for radio in [self.bead_radio, self.reference_radio, self.cell_radio]:
            type_layout.addWidget(radio)

        type_group.setLayout(type_layout)
        preview_layout.addWidget(type_group)

        # Preview toggle
        preview_toggle = QHBoxLayout()
        self.preview_check = QCheckBox("Show Preview")
        preview_toggle.addWidget(self.preview_check)
        preview_toggle.addStretch()
        preview_layout.addLayout(preview_toggle)

        preview_frame.setLayout(preview_layout)
        return preview_frame

    def _create_action_frame(self):
        """Create the action buttons frame."""
        action_frame = QFrame()
        action_layout = QHBoxLayout()

        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.save_btn = QPushButton("Save Preprocessed Images")
        self.save_btn.setEnabled(False)

        action_layout.addWidget(self.preprocess_btn)
        action_layout.addWidget(self.save_btn)

        action_frame.setLayout(action_layout)
        return action_frame

    def _create_status_frame(self):
        """Create the status display frame."""
        status_frame = QFrame()
        status_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)

        status_frame.setLayout(status_layout)
        return status_frame

    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect preview controls
        self.preview_check.toggled.connect(self.controller.toggle_preview)
        for radio in [self.bead_radio, self.reference_radio, self.cell_radio]:
            radio.toggled.connect(self._on_preview_type_changed)

        # Connect action buttons
        self.preprocess_btn.clicked.connect(self.controller.run_preprocessing)
        self.save_btn.clicked.connect(self._save_preprocessed_data)

        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.controller.preprocessing_failed.connect(self._on_preprocessing_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Connect parameter panel changes
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)

    def _on_preview_type_changed(self):
        """Handle preview type radio button changes."""
        if self.bead_radio.isChecked():
            self.controller.current_data_type = 'beads'
        elif self.reference_radio.isChecked():
            self.controller.current_data_type = 'reference'
        else:
            self.controller.current_data_type = 'cells'

        if self.preview_check.isChecked():
            self.controller._update_preview()

    def _on_parameter_changed(self):
        """Handle parameter changes."""
        if self.preview_check.isChecked():
            self.controller._update_preview()

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _on_preprocessing_completed(self, results):
        """Handle preprocessing completion."""
        self.save_btn.setEnabled(True)
        self.preprocessing_completed.emit(results)

    def _on_preprocessing_failed(self, error_msg: str):
        """Handle preprocessing failure."""
        self.save_btn.setEnabled(False)
        QMessageBox.critical(self, "Error", error_msg)

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update data panel
        self.data_panel.update_button_states()
        self.data_panel.update_data_status()

        # Update preview controls
        has_data = (
                self.data_manager.input_bead_stack is not None or
                self.data_manager.input_reference is not None or
                self.data_manager.input_cell_stack is not None
        )
        self.preview_check.setEnabled(has_data)
        if not has_data and self.preview_check.isChecked():
            self.preview_check.setChecked(False)

        # Update action buttons
        self.preprocess_btn.setEnabled(has_data)
        has_preprocessed = (
                self.data_manager.preprocessed_bead_stack is not None or
                self.data_manager.preprocessed_reference is not None or
                self.data_manager.preprocessed_cell_stack is not None
        )
        self.save_btn.setEnabled(has_preprocessed)

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze."""
        self.preview_check.setEnabled(not frozen)
        self.preprocess_btn.setEnabled(not frozen)
        self.save_btn.setEnabled(not frozen and self.data_manager.preprocessed_bead_stack is not None)

    def _save_preprocessed_data(self):
        """Save preprocessed data to files."""
        try:
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Preprocessed Data",
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )

            if not save_dir:
                return

            # Get calibration values
            pixel_size = self.parameter_manager.get_value('pixel_size')
            frame_interval = self.parameter_manager.get_value('frame_interval')

            files_saved = []

            # Helper function to save TIFF files
            def save_tiff(data, filename):
                if data is None:
                    return False

                filepath = os.path.join(save_dir, filename)

                # Create metadata
                metadata = {
                    'ImageJ': '1.53c',
                    'spacing': pixel_size,
                    'unit': 'um',
                    'frame_interval': frame_interval,
                    'frame_interval_unit': 'minute'
                }

                # Save file with metadata
                tifffile.imwrite(
                    filepath,
                    data,
                    imagej=True,
                    metadata=metadata,
                    resolution=(1 / pixel_size, 1 / pixel_size)
                )
                return True

            # Save each data type if available
            if self.data_manager.preprocessed_bead_stack is not None:
                if save_tiff(self.data_manager.preprocessed_bead_stack, "preprocessed_beads.tif"):
                    files_saved.append("preprocessed_beads.tif")

            if self.data_manager.preprocessed_reference is not None:
                if save_tiff(self.data_manager.preprocessed_reference, "preprocessed_reference.tif"):
                    files_saved.append("preprocessed_reference.tif")

            if self.data_manager.preprocessed_cell_stack is not None:
                if save_tiff(self.data_manager.preprocessed_cell_stack, "preprocessed_cells.tif"):
                    files_saved.append("preprocessed_cells.tif")

            if files_saved:
                self._update_status(
                    100,
                    f"Saved files with calibration:\n"
                    f"pixel size: {pixel_size} µm, frame interval: {frame_interval} min\n" +
                    "\n".join(files_saved)
                )
            else:
                self._update_status(0, "No preprocessed data available to save")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        self.visualization_manager.cleanup()
        super().cleanup()
