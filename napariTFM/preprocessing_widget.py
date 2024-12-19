from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QInputDialog, QScrollArea, QWidget, QSizePolicy,
    QRadioButton, QLabel, QFrame, QProgressBar, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton,
    QComboBox, QMessageBox
)
from qtrangeslider import QRangeSlider
import napari
from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.data_manager import DataManager
from napariTFM.error_handling import ProcessingError
from napariTFM.preprocessing import (
    PreprocessingParameters,
    ImagePreprocessor
)
from napariTFM.visualization_manager import VisualizationManager


class PreprocessingWidget(BaseAnalysisWidget):
    """Widget for controlling image preprocessing parameters"""

    preprocessing_completed = Signal(dict)  # Emits dict of processed data types

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: DataManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize instance variables
        self.preprocessor = ImagePreprocessor()
        self.preview_enabled = False
        self.current_data_type = 'beads'

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode"""
        try:
            if enabled:
                # Get current data type
                if self.bead_radio.isChecked():
                    data = self.data_manager.bead_stack
                elif self.reference_radio.isChecked():
                    data = self.data_manager.reference_image
                else:
                    data = self.data_manager.cell_stack

                if data is None:
                    raise ProcessingError(f"No {self.current_data_type} data available")

                # Get current frame if data is a stack
                if data.ndim == 3:
                    current_step = self.viewer.dims.current_step[0]
                    frame = data[current_step].copy()
                else:
                    frame = data.copy()

                # Process the frame
                processed_frame, frame_info = self.preprocessor.preprocess_frame(
                    frame,
                    is_cell=(self.current_data_type == 'cells')
                )

                # Update visualization through manager
                self.visualization_manager.handle_preview(
                    frame=processed_frame,
                    enable=True,
                    layer_name='Preview'
                )

                # Update status with frame information
                info_text = (
                    f"Preview - Original range: ({frame.min():.1f}, {frame.max():.1f})\n"
                    f"Applied range: {frame_info['intensity_range']}\n"
                    f"Mean: {frame_info['final_mean']:.1f}, Std: {frame_info['final_std']:.1f}"
                )
                self._update_status(info_text)

            else:
                # Disable preview through visualization manager
                self.visualization_manager.handle_preview(
                    frame=None,
                    enable=False
                )

            self.preview_enabled = enabled

        except Exception as e:
            self._handle_error(str(e))
            self.preview_check.setChecked(False)
            self.preview_enabled = False

    def update_preview_frame(self):
        """Update the preview for the current frame"""
        if not self.preview_enabled:
            return

        try:
            # Force update by toggling preview
            self.toggle_preview(True)

        except Exception as e:
            self._handle_error(str(e))

    def _on_preview_type_changed(self):
        """Handle preview data type selection change"""
        if self.bead_radio.isChecked():
            self.current_data_type = 'beads'
        elif self.reference_radio.isChecked():
            self.current_data_type = 'reference'
        else:
            self.current_data_type = 'cells'

        if self.preview_enabled:
            self.update_preview_frame()

    def run_preprocessing(self):
        """Run preprocessing on all available data"""
        if self.preview_enabled:
            self.preview_check.setChecked(False)

        try:
            self._set_controls_enabled(False)
            self._update_status("Starting preprocessing...", 0)

            # Get parameters
            params = PreprocessingParameters(
                min_intensity_percentile=self.min_spinbox.value() / 100,
                max_intensity_percentile=self.max_spinbox.value() / 100,
                enable_gaussian_filter=self.gaussian_check.isChecked(),
                gaussian_sigma=self.gaussian_sigma_spin.value(),
                cell_min_intensity_percentile=self.cell_min_spinbox.value() / 100,
                cell_max_intensity_percentile=self.cell_max_spinbox.value() / 100,
                enable_cell_gaussian_filter=self.cell_gaussian_check.isChecked(),
                cell_gaussian_sigma=self.cell_gaussian_sigma_spin.value(),
                enable_registration=self.registration_check.isChecked(),
                registration_mode=self.registration_mode_combo.currentText().lower()
            )

            self.preprocessor.update_parameters(params)

            # Process all data with progress updates
            results = self.preprocessor.preprocess_all(
                bead_stack=self.data_manager.bead_stack,
                reference_image=self.data_manager.reference_image,
                cell_stack=self.data_manager.cell_stack,
                progress_callback=lambda progress, message: self._update_status(message, int(progress))
            )

            # Update visualization through manager
            self.visualization_manager.update_preprocessing_visualization(results)

            self._update_status("Preprocessing complete", 100)
            self.preprocessing_completed.emit(results)

        except Exception as e:
            self._handle_error(str(e))
            self.processing_failed.emit(str(e))
        finally:
            self._set_controls_enabled(True)

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        removed_layer = event.value

        try:
            if removed_layer == self.original_layer:
                self._update_status("Original layer was removed")
                self.original_layer = None
                self.visualization_manager.handle_preview(
                    frame=None,
                    enable=False
                )
                self.preview_enabled = False
                self.preview_check.setChecked(False)

        except Exception as e:
            self._handle_error(ProcessingError(
                "Error handling layer removal",
                str(e)
            ))

    def cleanup(self):
        """Clean up resources and event connections."""
        # Ensure preview is disabled
        if self.preview_enabled:
            self.visualization_manager.handle_preview(
                frame=None,
                enable=False
            )
        super().cleanup()

    def _create_intensity_range_group(self):
        """Create the intensity range group."""
        intensity_group = QGroupBox("Bead/Reference Intensity Range")
        intensity_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        intensity_layout = QVBoxLayout()

        # Add slider
        self.intensity_slider = QRangeSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 1000)  # Range 0-1000 for 0.1% steps
        self.intensity_slider.setValue((0, 1000))
        intensity_layout.addWidget(self.intensity_slider)

        # Create spinboxes first
        self.min_spinbox = QDoubleSpinBox()
        self.max_spinbox = QDoubleSpinBox()

        # Configure spinboxes with finer step size
        for spinbox in [self.min_spinbox, self.max_spinbox]:
            spinbox.setRange(0, 100)
            spinbox.setDecimals(1)  # Show one decimal place
            spinbox.setSingleStep(0.1)  # 0.1% steps

        self.max_spinbox.setValue(100)

        # Add spinboxes layout
        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(QLabel("Min %:"))
        spinbox_layout.addWidget(self.min_spinbox)
        spinbox_layout.addStretch()
        spinbox_layout.addWidget(QLabel("Max %:"))
        spinbox_layout.addWidget(self.max_spinbox)
        intensity_layout.addLayout(spinbox_layout)

        intensity_group.setLayout(intensity_layout)
        return intensity_group

    def _update_intensity_labels(self, values):
        """Update intensity range spinboxes with slider values"""
        min_val, max_val = values

        # Convert slider values (0-1000) to percentages with one decimal place
        min_percent = float(min_val) / 10.0  # Convert to 0.1% steps
        max_percent = float(max_val) / 10.0

        # Update spinboxes without triggering their signals
        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(min_percent)
        self.max_spinbox.setValue(max_percent)
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)

        self.update_parameters()

    def _update_slider_from_spinbox(self):
        """Update the slider values when spinboxes change"""
        # Convert percentages to slider values (multiply by 10 for 0.1% resolution)
        min_val = int(self.min_spinbox.value() * 10)
        max_val = int(self.max_spinbox.value() * 10)

        # Update slider without triggering its signal
        self.intensity_slider.blockSignals(True)
        self.intensity_slider.setValue((min_val, max_val))
        self.intensity_slider.blockSignals(False)

        self.update_parameters()

    def update_parameters(self):
        """Update preprocessing parameters from UI controls"""
        try:
            # Get bead/reference intensity ranges
            min_percentile = self.min_spinbox.value() / 100.0
            max_percentile = self.max_spinbox.value() / 100.0

            # Get cell intensity ranges
            cell_min_percentile = self.cell_min_spinbox.value() / 100.0
            cell_max_percentile = self.cell_max_spinbox.value() / 100.0

            # Create new parameters including cell-specific ones
            params = PreprocessingParameters(
                # Bead/Reference parameters
                min_intensity_percentile=min_percentile,
                max_intensity_percentile=max_percentile,
                enable_gaussian_filter=self.gaussian_check.isChecked(),
                gaussian_sigma=self.gaussian_sigma_spin.value(),

                # Cell parameters
                cell_min_intensity_percentile=cell_min_percentile,
                cell_max_intensity_percentile=cell_max_percentile,
                enable_cell_gaussian_filter=self.cell_gaussian_check.isChecked(),
                cell_gaussian_sigma=self.cell_gaussian_sigma_spin.value(),

                # Registration parameters
                enable_registration=self.registration_check.isChecked(),
                registration_mode=self.registration_mode_combo.currentText().lower()
            )

            # Validate and update preprocessor
            params.validate()
            self.preprocessor.update_parameters(params)

            # Update preview if enabled
            if self.preview_enabled:
                self.update_preview_frame()

        except Exception as e:
            self._handle_error(e)

    def _create_range_spinboxes(self, min_spinbox, max_spinbox, is_cell=False):
        """Create a layout with min/max range spinboxes."""
        spinbox_layout = QHBoxLayout()

        for spinbox in [min_spinbox, max_spinbox]:
            spinbox.setRange(0, 100)
            spinbox.setDecimals(1)
            # Finer step size for cell controls
            spinbox.setSingleStep(0.1 if is_cell else 1.0)

        max_spinbox.setValue(100)

        spinbox_layout.addWidget(QLabel("Min %:"))
        spinbox_layout.addWidget(min_spinbox)
        spinbox_layout.addStretch()
        spinbox_layout.addWidget(QLabel("Max %:"))
        spinbox_layout.addWidget(max_spinbox)

        return spinbox_layout

    def _create_cell_params_group(self):
        """Create the cell stack parameters group."""
        cell_params_group = QGroupBox("Cell Stack Parameters")
        cell_params_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cell_params_layout = QVBoxLayout()

        # Cell contrast enhancement
        cell_intensity_layout = QVBoxLayout()
        cell_intensity_layout.addWidget(QLabel("Contrast Enhancement"))

        self.cell_intensity_slider = QRangeSlider(Qt.Horizontal)
        self.cell_intensity_slider.setRange(0, 1000)  # Keep 0-1000 range for finer control
        self.cell_intensity_slider.setValue((0, 1000))
        cell_intensity_layout.addWidget(self.cell_intensity_slider)

        # Create cell spinboxes first
        self.cell_min_spinbox = QDoubleSpinBox()
        self.cell_max_spinbox = QDoubleSpinBox()

        # Add cell spinboxes with finer step size
        spinbox_layout = self._create_range_spinboxes(
            self.cell_min_spinbox,
            self.cell_max_spinbox,
            is_cell=True  # Pass flag for cell-specific settings
        )
        cell_intensity_layout.addLayout(spinbox_layout)
        cell_params_layout.addLayout(cell_intensity_layout)

        # Cell gaussian blur
        self.cell_gaussian_check = QCheckBox("Enable Gaussian Filter")
        cell_params_layout.addWidget(self.cell_gaussian_check)

        # Create cell gaussian spinbox first
        self.cell_gaussian_sigma_spin = QDoubleSpinBox()

        sigma_layout = self._create_sigma_control(self.cell_gaussian_sigma_spin)
        cell_params_layout.addLayout(sigma_layout)

        cell_params_group.setLayout(cell_params_layout)
        return cell_params_group

    def _load_active_layer(self, data_type: str):
        """Load the currently active layer as the specified data type"""
        active_layer = self._get_active_image_layer()
        if active_layer is None:
            self._show_warning("No active image layer found")
            return

        try:
            if data_type == 'beads':
                self.data_manager.set_bead_stack(active_layer.data)
            elif data_type == 'reference':
                self.data_manager.set_reference_image(active_layer.data)
            elif data_type == 'cells':
                self.data_manager.set_cell_stack(active_layer.data)
            else:
                raise ValueError(f"Invalid data type: {data_type}")

            self._update_ui_state()
            self._update_status(f"Loaded {data_type} data: {active_layer.data.shape}")

        except Exception as e:
            self._show_warning(str(e))

    def _update_cell_intensity_labels(self, values):
        """Update cell intensity range spinboxes with slider values"""
        min_val, max_val = values
        # Convert from 0-1000 range to 0-100 percentage
        min_percent = min_val / 10.0
        max_percent = max_val / 10.0

        # Update spinboxes without triggering their signals
        self.cell_min_spinbox.blockSignals(True)
        self.cell_max_spinbox.blockSignals(True)
        self.cell_min_spinbox.setValue(min_percent)
        self.cell_max_spinbox.setValue(max_percent)
        self.cell_min_spinbox.blockSignals(False)
        self.cell_max_spinbox.blockSignals(False)

        self.update_parameters()

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        # Reset bead/reference intensity range
        self.intensity_slider.setValue((0, 1000))
        # Don't trigger update twice - let spinboxes do it
        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(0)
        self.max_spinbox.setValue(100)
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)

        # Reset cell intensity range
        self.cell_intensity_slider.setValue((0, 1000))
        self.cell_min_spinbox.blockSignals(True)
        self.cell_max_spinbox.blockSignals(True)
        self.cell_min_spinbox.setValue(0)
        self.cell_max_spinbox.setValue(100)
        self.cell_min_spinbox.blockSignals(False)
        self.cell_max_spinbox.blockSignals(False)

        # Reset bead/reference gaussian filter
        self.gaussian_check.setChecked(False)
        self.gaussian_sigma_spin.setValue(1.0)

        # Reset cell gaussian filter
        self.cell_gaussian_check.setChecked(False)
        self.cell_gaussian_sigma_spin.setValue(1.0)

        # Reset registration
        self.registration_check.setChecked(False)
        self.registration_mode_combo.setCurrentText('Translation')

        self._update_status("Parameters reset to defaults")
        self.update_parameters()

    def _setup_ui(self):
        """Set up the complete user interface for the preprocessing widget."""
        scroll = self._create_scroll_area()
        container = self._create_container()
        main_layout = self._setup_main_layout()
        container.setLayout(main_layout)

        # Add all component groups to main layout
        main_layout.addWidget(self._create_load_group())
        main_layout.addWidget(self._create_preview_selection_group())
        main_layout.addWidget(self._create_intensity_range_group())
        main_layout.addWidget(self._create_filter_group())
        main_layout.addWidget(self._create_cell_params_group())
        main_layout.addWidget(self._create_registration_group())
        main_layout.addWidget(self._create_preview_frame())
        main_layout.addWidget(self._create_action_buttons())
        main_layout.addWidget(self._create_status_frame())

        scroll.setWidget(container)

        # Set the final layout
        final_layout = QVBoxLayout()
        final_layout.setContentsMargins(0, 0, 0, 0)
        final_layout.addWidget(scroll)
        self.setLayout(final_layout)

        self._register_controls()

    def _create_scroll_area(self):
        """Create and configure the main scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _create_container(self):
        """Create the container widget for the scroll area."""
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return container

    def _setup_main_layout(self):
        """Create and configure the main layout."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(6, 6, 6, 6)
        return main_layout

    def _create_load_group(self):
        """Create the data loading group."""
        load_group = QGroupBox("Load Data")
        load_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        load_layout = QVBoxLayout()
        load_layout.setSpacing(4)

        # Initialize buttons and status labels
        self.load_beads_btn = QPushButton("Load Active Layer as Bead Stack")
        self.load_reference_btn = QPushButton("Load Active Layer as Reference")
        self.load_cells_btn = QPushButton("Load Active Layer as Cell Stack")

        self.bead_status = QLabel("Bead Stack: Not loaded")
        self.reference_status = QLabel("Reference Image: Not loaded")
        self.cell_status = QLabel("Cell Stack: Not loaded")

        # Add widgets with their status labels
        buttons_and_labels = [
            (self.load_beads_btn, self.bead_status),
            (self.load_reference_btn, self.reference_status),
            (self.load_cells_btn, self.cell_status)
        ]

        for btn, label in buttons_and_labels:
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(btn)
            btn_layout.addWidget(label)
            load_layout.addLayout(btn_layout)

        load_group.setLayout(load_layout)
        return load_group

    def _create_preview_selection_group(self):
        """Create the preview selection group."""
        preview_select_group = QGroupBox("Preview Data Type")
        preview_select_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        preview_layout = QHBoxLayout()

        self.bead_radio = QRadioButton("Bead Stack")
        self.reference_radio = QRadioButton("Reference Image")
        self.cell_radio = QRadioButton("Cell Stack")
        self.bead_radio.setChecked(True)

        for radio in [self.bead_radio, self.reference_radio, self.cell_radio]:
            preview_layout.addWidget(radio)

        preview_select_group.setLayout(preview_layout)
        return preview_select_group

    def _create_filter_group(self):
        """Create the Gaussian filter group."""
        filter_group = QGroupBox("Bead/Reference Gaussian Filter")
        filter_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        filter_layout = QVBoxLayout()

        self.gaussian_check = QCheckBox("Enable Gaussian Filter")
        filter_layout.addWidget(self.gaussian_check)

        # Create sigma spinbox first
        self.gaussian_sigma_spin = QDoubleSpinBox()

        sigma_layout = self._create_sigma_control(self.gaussian_sigma_spin)
        filter_layout.addLayout(sigma_layout)

        filter_group.setLayout(filter_layout)
        return filter_group

    def _create_sigma_control(self, sigma_spinbox):
        """Create a layout with sigma control for Gaussian filter."""
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))

        sigma_spinbox.setRange(0.1, 10.0)
        sigma_spinbox.setValue(1.0)
        sigma_spinbox.setSingleStep(0.1)
        sigma_spinbox.setEnabled(False)

        sigma_layout.addWidget(sigma_spinbox)
        sigma_layout.addStretch()

        return sigma_layout

    def _create_registration_group(self):
        """Create the registration group."""
        registration_group = QGroupBox("Registration")
        registration_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        registration_layout = QVBoxLayout()

        self.registration_check = QCheckBox("Enable Registration")
        registration_layout.addWidget(self.registration_check)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.addItems(['Translation', 'Rigid'])
        self.registration_mode_combo.setEnabled(False)
        mode_layout.addWidget(self.registration_mode_combo)
        mode_layout.addStretch()
        registration_layout.addLayout(mode_layout)

        self.registration_note = QLabel(
            "Note: Registration requires both reference image and bead stack.\n"
            "Registration will be performed relative to the reference image."
        )
        self.registration_note.setWordWrap(True)
        registration_layout.addWidget(self.registration_note)

        registration_group.setLayout(registration_layout)
        return registration_group

    def _create_preview_frame(self):
        """Create the preview frame."""
        preview_frame = QFrame()
        preview_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        preview_layout = QHBoxLayout()
        self.preview_check = QCheckBox("Show Preview")
        preview_layout.addWidget(self.preview_check)
        preview_layout.addStretch()
        preview_frame.setLayout(preview_layout)
        return preview_frame

    def _create_action_buttons(self):
        """Create the action buttons frame."""
        button_frame = QFrame()
        button_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        button_layout = QHBoxLayout()
        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.reset_btn = QPushButton("Reset Parameters")
        button_layout.addWidget(self.preprocess_btn)
        button_layout.addWidget(self.reset_btn)
        button_frame.setLayout(button_layout)
        return button_frame

    def _create_status_frame(self):
        """Create the status and progress frame."""
        status_frame = QFrame()
        status_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)
        return status_frame

    def _update_ui_state(self):
        """Update UI elements based on available data and current state"""
        # Update data status indicators
        bead_status = "Loaded" if self.data_manager.bead_stack is not None else "Not loaded"
        ref_status = "Loaded" if self.data_manager.reference_image is not None else "Not loaded"
        cell_status = "Loaded" if self.data_manager.cell_stack is not None else "Not loaded"

        self.bead_status.setText(f"Bead Stack: {bead_status}")
        self.reference_status.setText(f"Reference Image: {ref_status}")
        self.cell_status.setText(f"Cell Stack: {cell_status}")

        # Update registration controls
        can_register = (
                self.data_manager.bead_stack is not None and
                self.data_manager.reference_image is not None
        )
        self.registration_check.setEnabled(can_register)
        self.registration_mode_combo.setEnabled(can_register and self.registration_check.isChecked())

        # Show/hide registration note based on data availability
        self.registration_note.setVisible(not can_register)

        # Update preview radio buttons
        self.bead_radio.setEnabled(self.data_manager.bead_stack is not None)
        self.reference_radio.setEnabled(self.data_manager.reference_image is not None)
        self.cell_radio.setEnabled(self.data_manager.cell_stack is not None)

        # Update cell-specific controls
        cell_data_loaded = self.data_manager.cell_stack is not None
        self.cell_intensity_slider.setEnabled(cell_data_loaded)
        self.cell_min_spinbox.setEnabled(cell_data_loaded)
        self.cell_max_spinbox.setEnabled(cell_data_loaded)
        self.cell_gaussian_check.setEnabled(cell_data_loaded)
        self.cell_gaussian_sigma_spin.setEnabled(
            cell_data_loaded and self.cell_gaussian_check.isChecked()
        )

        # Enable preprocessing button if we have any data
        has_data = any([
            self.data_manager.bead_stack is not None,
            self.data_manager.reference_image is not None,
            self.data_manager.cell_stack is not None
        ])
        self.preprocess_btn.setEnabled(has_data)

    def _connect_signals(self):
        """Connect widget signals"""
        # Connect load buttons
        self.load_beads_btn.clicked.connect(lambda: self._load_active_layer('beads'))
        self.load_reference_btn.clicked.connect(lambda: self._load_active_layer('reference'))
        self.load_cells_btn.clicked.connect(lambda: self._load_active_layer('cells'))

        # Preview data type selection
        self.bead_radio.toggled.connect(self._on_preview_type_changed)
        self.reference_radio.toggled.connect(self._on_preview_type_changed)
        self.cell_radio.toggled.connect(self._on_preview_type_changed)

        # Parameter controls - Bead/Reference
        self.intensity_slider.valueChanged.connect(self._update_intensity_labels)
        self.min_spinbox.valueChanged.connect(self._update_slider_from_spinbox)
        self.max_spinbox.valueChanged.connect(self._update_slider_from_spinbox)

        # Gaussian filter - Bead/Reference
        self.gaussian_check.toggled.connect(self.gaussian_sigma_spin.setEnabled)
        self.gaussian_check.toggled.connect(self.update_parameters)
        self.gaussian_sigma_spin.valueChanged.connect(self.update_parameters)

        # Parameter controls - Cell
        self.cell_intensity_slider.valueChanged.connect(self._update_cell_intensity_labels)
        self.cell_min_spinbox.valueChanged.connect(self._update_cell_slider_from_spinbox)
        self.cell_max_spinbox.valueChanged.connect(self._update_cell_slider_from_spinbox)

        # Gaussian filter - Cell
        self.cell_gaussian_check.toggled.connect(self.cell_gaussian_sigma_spin.setEnabled)
        self.cell_gaussian_check.toggled.connect(self.update_parameters)
        self.cell_gaussian_sigma_spin.valueChanged.connect(self.update_parameters)

        # Registration
        self.registration_check.toggled.connect(self._update_registration_controls)
        self.registration_mode_combo.currentTextChanged.connect(self.update_parameters)

        # Preview
        self.preview_check.toggled.connect(self.toggle_preview)

        # Action buttons
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.reset_btn.clicked.connect(self.reset_parameters)

    def _update_cell_slider_from_spinbox(self):
        """Update the cell slider values when spinboxes change"""
        # Convert from percentage to slider range (0-1000)
        min_val = self.cell_min_spinbox.value() * 10
        max_val = self.cell_max_spinbox.value() * 10

        # Update slider without triggering its signal
        self.cell_intensity_slider.blockSignals(True)
        self.cell_intensity_slider.setValue((int(min_val), int(max_val)))
        self.cell_intensity_slider.blockSignals(False)

        self.update_parameters()

    def _update_registration_controls(self, enabled: bool):
        """Update registration controls state"""
        self.registration_mode_combo.setEnabled(enabled)

        # Check if we have required data for registration
        if enabled and (
                self.data_manager.bead_stack is None or
                self.data_manager.reference_image is None
        ):
            self.registration_check.setChecked(False)
            self._show_warning(
                "Registration requires both reference image and bead stack"
            )
            return

        self.update_parameters()

    def _register_controls(self):
        """Register all controls with the base widget"""
        controls = [
            self.intensity_slider,
            self.min_spinbox,
            self.max_spinbox,
            self.gaussian_check,
            self.gaussian_sigma_spin,
            self.registration_check,
            self.registration_mode_combo,
            self.preview_check,
            self.preprocess_btn,
            self.reset_btn,
            self.bead_radio,
            self.reference_radio,
            self.cell_radio,
            self.progress_bar,
            self.status_label,
            self.bead_status,
            self.reference_status,
            self.cell_status,
            self.cell_intensity_slider,
            self.cell_min_spinbox,
            self.cell_max_spinbox,
            self.cell_gaussian_check,
            self.cell_gaussian_sigma_spin,
        ]

        for control in controls:
            self.register_control(control)

    def _get_layer_name(self) -> str:
        """Get appropriate layer name based on current data type"""
        if self.current_data_type == 'beads':
            return 'Preprocessed Beads'
        elif self.current_data_type == 'reference':
            return 'Preprocessed Reference'
        else:
            return 'Preprocessed Cells'

    def _on_data_type_changed(self):
        """Handle data type selection change"""
        if self.bead_radio.isChecked():
            self.current_data_type = 'beads'
        elif self.reference_radio.isChecked():
            self.current_data_type = 'reference'
        else:
            self.current_data_type = 'cells'

        self._update_ui_state()

        if self.preview_enabled:
            self.update_preview_frame()

    def _get_current_data(self) -> Optional[np.ndarray]:
        """Get data for current type"""
        if self.current_data_type == 'beads':
            return self.data_manager.bead_stack
        elif self.current_data_type == 'reference':
            return self.data_manager.reference_image
        else:
            return self.data_manager.cell_stack



