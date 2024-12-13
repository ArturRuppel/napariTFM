from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QInputDialog,
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
        self.load_beads_btn = None
        self.load_reference_btn = None
        self.load_cells_btn = None
        self.bead_status = None
        self.reference_status = None
        self.cell_status = None

        super().__init__(
            viewer=viewer,
            data_manager=data_manager,
            visualization_manager=visualization_manager,
        )

        self.preprocessor = ImagePreprocessor()
        self.preview_enabled = False
        self.original_layer = None
        self.preview_layer = None
        self.current_data_type = 'beads'  # For preview purposes

        self._setup_ui()
        self._connect_signals()

        self.viewer.layers.events.removed.connect(self._handle_layer_removal)
        self._update_ui_state()


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

    def update_parameters(self):
        """Update preprocessing parameters from UI controls"""
        try:
            # Get current intensity range from slider (as percentiles)
            min_percentile = self.intensity_slider.value()[0] / 100
            max_percentile = self.intensity_slider.value()[1] / 100

            # Create new parameters
            params = PreprocessingParameters(
                min_intensity_percentile=min_percentile,
                max_intensity_percentile=max_percentile,
                enable_gaussian_filter=self.gaussian_check.isChecked(),
                gaussian_sigma=self.gaussian_sigma_spin.value(),
                enable_registration=self.registration_check.isChecked(),
                registration_mode=self.registration_mode_combo.currentText().lower(),
                reference_frame=self.ref_frame_spin.value()
            )

            # Validate and update preprocessor
            params.validate()
            self.preprocessor.update_parameters(params)

            # Update preview if enabled
            if self.preview_enabled:
                self.update_preview_frame()

        except Exception as e:
            self._handle_error(ProcessingError(
                "Failed to update parameters",
                str(e)
            ))

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Data Loading Group
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout()

        # Initialize buttons and status labels
        self.load_beads_btn = QPushButton("Load Active Layer as Bead Stack")
        self.load_reference_btn = QPushButton("Load Active Layer as Reference")
        self.load_cells_btn = QPushButton("Load Active Layer as Cell Stack")

        self.bead_status = QLabel("Bead Stack: Not loaded")
        self.reference_status = QLabel("Reference Image: Not loaded")
        self.cell_status = QLabel("Cell Stack: Not loaded")

        # Add widgets to layout
        for btn, label in [
            (self.load_beads_btn, self.bead_status),
            (self.load_reference_btn, self.reference_status),
            (self.load_cells_btn, self.cell_status)
        ]:
            layout = QHBoxLayout()
            layout.addWidget(btn)
            layout.addWidget(label)
            load_layout.addLayout(layout)

        load_group.setLayout(load_layout)
        main_layout.addWidget(load_group)


        # Data selection group
        data_group = QGroupBox("Available Data")
        data_layout = QVBoxLayout()

        # Data status indicators
        self.bead_status = QLabel("Bead Stack: Not loaded")
        self.reference_status = QLabel("Reference Image: Not loaded")
        self.cell_status = QLabel("Cell Stack: Not loaded")

        data_layout.addWidget(self.bead_status)
        data_layout.addWidget(self.reference_status)
        data_layout.addWidget(self.cell_status)

        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)

        # Preview selection (only for checking preprocessing effects)
        preview_select_group = QGroupBox("Preview Data Type")
        preview_layout = QHBoxLayout()
        self.bead_radio = QRadioButton("Bead Stack")
        self.reference_radio = QRadioButton("Reference Image")
        self.cell_radio = QRadioButton("Cell Stack")
        self.bead_radio.setChecked(True)

        preview_layout.addWidget(self.bead_radio)
        preview_layout.addWidget(self.reference_radio)
        preview_layout.addWidget(self.cell_radio)
        preview_select_group.setLayout(preview_layout)
        main_layout.addWidget(preview_select_group)

        # Intensity Range Group
        intensity_group = QGroupBox("Intensity Range")
        intensity_layout = QVBoxLayout()

        self.intensity_slider = QRangeSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue((0, 100))
        intensity_layout.addWidget(self.intensity_slider)

        percentile_layout = QHBoxLayout()
        self.min_percentile_label = QLabel("0%")
        self.max_percentile_label = QLabel("100%")
        percentile_layout.addWidget(self.min_percentile_label)
        percentile_layout.addStretch()
        percentile_layout.addWidget(self.max_percentile_label)
        intensity_layout.addLayout(percentile_layout)

        intensity_group.setLayout(intensity_layout)
        main_layout.addWidget(intensity_group)

        # Gaussian Filter Group
        filter_group = QGroupBox("Gaussian Filter")
        filter_layout = QVBoxLayout()

        self.gaussian_check = QCheckBox("Enable Gaussian Filter")
        filter_layout.addWidget(self.gaussian_check)

        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.gaussian_sigma_spin = QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0.1, 10.0)
        self.gaussian_sigma_spin.setValue(1.0)
        self.gaussian_sigma_spin.setSingleStep(0.1)
        self.gaussian_sigma_spin.setEnabled(False)
        sigma_layout.addWidget(self.gaussian_sigma_spin)

        filter_layout.addLayout(sigma_layout)
        filter_group.setLayout(filter_layout)
        main_layout.addWidget(filter_group)

        # Registration Group
        registration_group = QGroupBox("Registration")
        registration_layout = QVBoxLayout()

        self.registration_check = QCheckBox("Enable Registration")
        registration_layout.addWidget(self.registration_check)

        # Registration requirements note
        self.registration_note = QLabel(
            "Note: Registration requires both reference image and bead stack"
        )
        self.registration_note.setWordWrap(True)
        registration_layout.addWidget(self.registration_note)

        # Add reference frame spinner
        ref_frame_layout = QHBoxLayout()
        ref_frame_layout.addWidget(QLabel("Reference Frame:"))
        self.ref_frame_spin = QSpinBox()
        self.ref_frame_spin.setMinimum(0)
        self.ref_frame_spin.setMaximum(0)  # Will be updated when data is loaded
        self.ref_frame_spin.setEnabled(False)
        ref_frame_layout.addWidget(self.ref_frame_spin)
        registration_layout.addLayout(ref_frame_layout)

        # Registration mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.addItems(['Translation', 'Rigid'])
        self.registration_mode_combo.setEnabled(False)
        mode_layout.addWidget(self.registration_mode_combo)
        registration_layout.addLayout(mode_layout)

        registration_group.setLayout(registration_layout)
        main_layout.addWidget(registration_group)

        # Preview Section
        preview_frame = QFrame()
        preview_layout = QHBoxLayout()
        self.preview_check = QCheckBox("Show Preview")
        preview_layout.addWidget(self.preview_check)
        preview_frame.setLayout(preview_layout)
        main_layout.addWidget(preview_frame)

        # Action Buttons
        button_layout = QHBoxLayout()
        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.reset_btn = QPushButton("Reset Parameters")
        button_layout.addWidget(self.preprocess_btn)
        button_layout.addWidget(self.reset_btn)
        main_layout.addLayout(button_layout)

        # Status and Progress
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self._register_controls()

    def run_preprocessing(self):
        """Run preprocessing on all available data"""
        if self.preview_enabled:
            self.preview_check.setChecked(False)

        try:
            self._set_controls_enabled(False)
            self._update_status("Starting preprocessing...", 0)

            # Get parameters
            params = PreprocessingParameters(
                min_intensity_percentile=self.intensity_slider.value()[0] / 100,
                max_intensity_percentile=self.intensity_slider.value()[1] / 100,
                enable_gaussian_filter=self.gaussian_check.isChecked(),
                gaussian_sigma=self.gaussian_sigma_spin.value(),
                enable_registration=self.registration_check.isChecked(),
                registration_mode=self.registration_mode_combo.currentText().lower(),
                reference_frame=self.ref_frame_spin.value()
            )

            self.preprocessor.update_parameters(params)

            # Process all data
            results = self.preprocessor.preprocess_all(
                bead_stack=self.data_manager.bead_stack,
                reference_image=self.data_manager.reference_image,
                cell_stack=self.data_manager.cell_stack
            )

            # Update visualization
            self._update_visualization(results)

            self._update_status("Preprocessing complete", 100)

            # Emit results dictionary
            self.preprocessing_completed.emit(results)

        except Exception as e:
            error_msg = str(e)
            self._handle_error(error_msg)
            self.processing_failed.emit(error_msg)
        finally:
            self._set_controls_enabled(True)

    def _update_visualization(self, results):
        """Update visualization of processed results"""
        try:
            # Remove existing preprocessed layers
            for layer_name in ['Preprocessed Beads', 'Preprocessed Reference', 'Preprocessed Cells']:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)

            # Add new layers
            if 'beads' in results:
                processed_beads, _ = results['beads']
                self.viewer.add_image(
                    processed_beads,
                    name='Preprocessed Beads',
                    visible=True
                )

            if 'reference' in results:
                processed_ref, _ = results['reference']
                self.viewer.add_image(
                    processed_ref,
                    name='Preprocessed Reference',
                    visible=True
                )

            if 'cells' in results:
                processed_cells, _ = results['cells']
                self.viewer.add_image(
                    processed_cells,
                    name='Preprocessed Cells',
                    visible=True
                )

        except Exception as e:
            self._handle_error(f"Error updating visualization: {str(e)}")
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
        self.ref_frame_spin.setEnabled(can_register and self.registration_check.isChecked())

        # Update reference frame spinner range if bead stack is available
        if self.data_manager.bead_stack is not None:
            self.ref_frame_spin.setMaximum(len(self.data_manager.bead_stack) - 1)
        else:
            self.ref_frame_spin.setMaximum(0)

        self.registration_note.setVisible(not can_register)

        # Update preview radio buttons
        self.bead_radio.setEnabled(self.data_manager.bead_stack is not None)
        self.reference_radio.setEnabled(self.data_manager.reference_image is not None)
        self.cell_radio.setEnabled(self.data_manager.cell_stack is not None)

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

        # Parameter controls
        self.intensity_slider.valueChanged.connect(self._update_intensity_labels)
        self.gaussian_check.toggled.connect(self.gaussian_sigma_spin.setEnabled)
        self.gaussian_check.toggled.connect(self.update_parameters)
        self.gaussian_sigma_spin.valueChanged.connect(self.update_parameters)
        self.registration_check.toggled.connect(self._update_registration_controls)
        self.registration_mode_combo.currentTextChanged.connect(self.update_parameters)

        # Add reference frame spinner connection
        self.ref_frame_spin.valueChanged.connect(self.update_parameters)

        # Preview
        self.preview_check.toggled.connect(self.toggle_preview)

        # Action buttons
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.reset_btn.clicked.connect(self.reset_parameters)

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
            self.min_percentile_label,
            self.max_percentile_label,
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
            self.ref_frame_spin,
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



    def _update_intensity_labels(self, values):
        """Update intensity range labels with percentile values"""
        min_val, max_val = values
        self.min_percentile_label.setText(f"{min_val}%")
        self.max_percentile_label.setText(f"{max_val}%")
        self.update_parameters()


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


    def reset_parameters(self):
        """Reset all parameters to defaults"""
        # Reset intensity range
        self.intensity_slider.setValue((0, 100))

        # Reset gaussian filter
        self.gaussian_check.setChecked(False)
        self.gaussian_sigma_spin.setValue(1.0)

        # Reset registration
        self.registration_check.setChecked(False)
        self.registration_mode_combo.setCurrentText('Translation')
        self.ref_frame_spin.setValue(0)

        self._update_status("Parameters reset to defaults")
        self.update_parameters()


    def _show_warning(self, message: str):
        """Show warning message to user"""
        QMessageBox.warning(self, "Warning", message)


    def toggle_preview(self, enabled: bool):
        """Toggle preview mode"""
        self.preview_enabled = enabled

        try:
            if enabled:
                # Get current layer and data
                if self.original_layer is None:
                    self.original_layer = self._get_active_image_layer()
                    if self.original_layer is None:
                        raise ProcessingError("No image layer found")

                # Create preview layer if it doesn't exist
                if self.preview_layer is None:
                    preview_data = (self.original_layer.data[0] if self.original_layer.data.ndim == 3
                                    else self.original_layer.data)
                    self.preview_layer = self.viewer.add_image(
                        np.zeros_like(preview_data),
                        name='Preview',
                        visible=True
                    )

                self.update_preview_frame()
            else:
                # Remove preview layer if it exists
                if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.preview_layer)
                self.preview_layer = None

        except Exception as e:
            self.preview_check.setChecked(False)
            self.preview_enabled = False
            if self.preview_layer is not None and self.preview_layer in self.viewer.layers:
                self.viewer.layers.remove(self.preview_layer)
            self.preview_layer = None
            raise ProcessingError("Preview failed", str(e))

    def update_preview_frame(self, event: Optional["Event"] = None):
        """Update the preview for the current frame"""
        if not self.preview_enabled or self.original_layer is None:
            return

        try:
            # Get current frame
            if self.original_layer.data.ndim == 3:
                current_step = self.viewer.dims.current_step[0]
                frame = self.original_layer.data[current_step].copy()
            else:
                frame = self.original_layer.data.copy()

            if frame.ndim != 2:
                raise ProcessingError(f"Invalid frame dimensions: {frame.shape}")

            # Process frame
            params = PreprocessingParameters(
                min_intensity_percentile=self.intensity_slider.value()[0] / 100,
                max_intensity_percentile=self.intensity_slider.value()[1] / 100,
                enable_gaussian_filter=self.gaussian_check.isChecked(),
                gaussian_sigma=self.gaussian_sigma_spin.value(),
                enable_registration=self.registration_check.isChecked(),
                registration_mode=self.registration_mode_combo.currentText().lower(),
                reference_frame=self.ref_frame_spin.value()
            )
            self.preprocessor.update_parameters(params)

            processed_frame, frame_info = self.preprocessor.preprocess_frame(frame)

            # Update preview layer
            if self.preview_layer is None:
                self.preview_layer = self.viewer.add_image(
                    processed_frame,
                    name='Preview',
                    visible=True
                )
            else:
                self.preview_layer.data = processed_frame

            # Update status
            info_text = (
                f"Preview - Original range: ({frame.min():.1f}, {frame.max():.1f})\n"
                f"Mean: {frame_info['final_mean']:.1f}, Std: {frame_info['final_std']:.1f}"
            )
            self._update_status(info_text)

        except Exception as e:
            self._handle_error(ProcessingError(
                "Preview failed",
                str(e)
            ))

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        try:
            removed_layer = event.value

            if removed_layer == self.preview_layer:
                self._update_status("Preview layer was removed")
                self.preview_layer = None
                self.preview_enabled = False
                self.preview_check.setChecked(False)

            if removed_layer == self.original_layer:
                self._update_status("Original layer was removed")
                self.original_layer = None
                try:
                    if self.preview_layer is not None:
                        self.viewer.layers.remove(self.preview_layer)
                except Exception as e:
                    self._handle_error(ProcessingError(
                        "Failed to remove preview layer",
                        str(e),
                    ))
                finally:
                    self.preview_layer = None
                    self.preview_enabled = False
                    self.preview_check.setChecked(False)

        except Exception as e:
            self._handle_error(ProcessingError(
                "Error handling layer removal",
                str(e),
            ))


    def _load_bead_stack(self):
        """Load bead stack from napari layer"""
        layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        if not layers:
            self._show_warning("No image layers found in napari viewer")
            return

        layer_names = [layer.name for layer in layers]
        selected_layer, ok = QInputDialog.getItem(
            self,
            "Select Bead Stack",
            "Choose the bead stack layer:",
            layer_names,
            0,
            False
        )

        if ok and selected_layer:
            layer = next(layer for layer in layers if layer.name == selected_layer)
            try:
                self.data_manager.set_bead_stack(layer.data)
                self._update_ui_state()
                self._update_status(f"Loaded bead stack: {layer.data.shape}")
            except ValueError as e:
                self._show_warning(str(e))

    def _load_reference_image(self):
        """Load reference image from napari layer"""
        layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        if not layers:
            self._show_warning("No image layers found in napari viewer")
            return

        layer_names = [layer.name for layer in layers]
        selected_layer, ok = QInputDialog.getItem(
            self,
            "Select Reference Image",
            "Choose the reference image layer:",
            layer_names,
            0,
            False
        )

        if ok and selected_layer:
            layer = next(layer for layer in layers if layer.name == selected_layer)
            try:
                self.data_manager.set_reference_image(layer.data)
                self._update_ui_state()
                self._update_status(f"Loaded reference image: {layer.data.shape}")
            except ValueError as e:
                self._show_warning(str(e))

    def _load_cell_stack(self):
        """Load cell stack from napari layer"""
        layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        if not layers:
            self._show_warning("No image layers found in napari viewer")
            return

        layer_names = [layer.name for layer in layers]
        selected_layer, ok = QInputDialog.getItem(
            self,
            "Select Cell Stack",
            "Choose the cell stack layer:",
            layer_names,
            0,
            False
        )

        if ok and selected_layer:
            layer = next(layer for layer in layers if layer.name == selected_layer)
            try:
                self.data_manager.set_cell_stack(layer.data)
                self._update_ui_state()
                self._update_status(f"Loaded cell stack: {layer.data.shape}")
            except ValueError as e:
                self._show_warning(str(e))
