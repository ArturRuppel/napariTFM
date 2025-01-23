import os
import json
import napari
import numpy as np
import tifffile
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox
)
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea, QWidget, QSizePolicy, QMessageBox,
    QRadioButton, QLabel, QFrame, QProgressBar, QFileDialog, QSlider,
    QDoubleSpinBox, QPushButton,
    QComboBox
)
from qtrangeslider import QRangeSlider

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.data_manager import DataManager
from napariTFM.error_handling import ProcessingError
from napariTFM.parameter_manager import ParameterManager, ParameterCategory
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
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        # Initialize base widget first
        super().__init__(viewer, data_manager, visualization_manager)

        # Store reference to parameter manager
        self.parameter_manager = parameter_manager

        # Initialize instance variables
        self.preprocessor = ImagePreprocessor()
        self.preview_enabled = False
        self.current_data_type = 'beads'
        self.colorbar_manager = ColorbarManager()

        # Block signals during setup
        self.blockSignals(True)
        try:
            self._setup_ui()
            self._connect_signals()
            self._connect_parameters()
            self._sync_widget_with_parameters()
            self._update_button_states()
            self._update_ui_state()

            if hasattr(self.parameter_manager, 'parameter_changed'):
                self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        finally:
            self.blockSignals(False)

        # Connect to viewer layer events
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self.viewer.layers.selection.events.changed.connect(self._on_layer_selection_change)

    def run_preprocessing(self):
        """Run preprocessing on all available data in a separate thread"""
        if self.preview_enabled:
            self.preview_check.setChecked(False)

        try:
            self._set_controls_enabled(False)
            self._update_status("Starting preprocessing...", 0)

            # Get parameters directly from parameter manager
            params = PreprocessingParameters(
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

            self.preprocessor.update_parameters(params)

            # Create and start worker with correct data from data manager
            worker = self.preprocessor.preprocess_all(
                bead_stack=self.data_manager.input_bead_stack,
                reference_image=self.data_manager.input_reference,
                cell_stack=self.data_manager.input_cell_stack,
            )

            # Connect signals
            worker.yielded.connect(self._handle_progress)
            worker.returned.connect(self._handle_results)
            worker.finished.connect(lambda: self._set_controls_enabled(True))
            worker.errored.connect(self._handle_error)

            # Start the worker
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.processing_failed.emit(str(e))
            self._set_controls_enabled(True)

    def toggle_preview(self, enabled: bool):
        """Toggle preview mode"""
        try:
            if enabled:
                # Get current data type
                if self.bead_radio.isChecked():
                    data = self.data_manager.input_bead_stack
                elif self.reference_radio.isChecked():
                    data = self.data_manager.input_reference
                else:
                    data = self.data_manager.input_cell_stack

                if data is None:
                    raise ProcessingError(f"No {self.current_data_type} data available")

                # Get current frame if data is a stack, handling both 2D and 3D cases
                if data.ndim == 3:
                    # For 3D data, ensure we're within bounds
                    current_step = min(self.viewer.dims.current_step[0], data.shape[0] - 1)
                    frame = data[current_step].copy()
                else:
                    # For 2D data, use as is
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

    def _load_active_layer(self, data_type: str):
        """Load the currently active layer as the specified data type"""
        active_layer = self._get_active_image_layer()
        if active_layer is None:
            QMessageBox.warning(self, "Error", "No active image layer found")
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

            self._update_ui_state()
            self._update_status(f"Loaded {data_type} data: {data.shape}")

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _update_ui_state(self):
        """Update UI elements based on available data and current state"""
        # Update data status indicators with shape information
        bead_data = self.data_manager.input_bead_stack
        ref_data = self.data_manager.input_reference
        cell_data = self.data_manager.input_cell_stack

        bead_shape = bead_data.shape if bead_data is not None else None
        ref_shape = ref_data.shape if ref_data is not None else None
        cell_shape = cell_data.shape if cell_data is not None else None

        self.bead_status.setText(f"Loaded: {bead_shape}" if bead_shape else "Not loaded")
        self.reference_status.setText(f"Loaded: {ref_shape}" if ref_shape else "Not loaded")
        self.cell_status.setText(f"Loaded: {cell_shape}" if cell_shape else "Not loaded")

        # Update registration note visibility based on data availability
        can_register = (bead_data is not None and ref_data is not None)
        self.registration_note.setVisible(not can_register)

        # Update preview radio buttons - always enabled
        self.bead_radio.setEnabled(True)
        self.reference_radio.setEnabled(True)
        self.cell_radio.setEnabled(True)

        # Enable preview checkbox only if we have any data
        has_data = any([bead_data is not None, ref_data is not None, cell_data is not None])
        self.preview_check.setEnabled(has_data)
        if not has_data and self.preview_check.isChecked():
            self.preview_check.setChecked(False)

        # Enable preprocessing button if we have any data
        self.preprocess_btn.setEnabled(has_data)

        # Update save button state based on preprocessed data
        has_preprocessed_data = any([
            self.data_manager._preprocessed_bead_stack is not None,
            self.data_manager._preprocessed_reference is not None,
            self.data_manager._preprocessed_cell_stack is not None
        ])
        self.save_btn.setEnabled(has_preprocessed_data)

    def _handle_results(self, results):
        """Handle the final results from the worker"""
        try:
            # Store preprocessing parameters
            params = {
                'min_intensity': self.parameter_manager.get_value('min_intensity'),
                'max_intensity': self.parameter_manager.get_value('max_intensity'),
                'gaussian_sigma': self.parameter_manager.get_value('gaussian_sigma'),
                'cell_min_intensity': self.parameter_manager.get_value('cell_min_intensity'),
                'cell_max_intensity': self.parameter_manager.get_value('cell_max_intensity'),
                'cell_gaussian_sigma': self.parameter_manager.get_value('cell_gaussian_sigma'),
                'registration_mode': self.parameter_manager.get_value('registration_mode')
            }

            # Update data manager with processed results and parameters
            self.data_manager.set_preprocessing_results(
                bead_stack=results.get('beads', [None])[0],
                reference=results.get('reference', [None])[0],
                cell_stack=results.get('cells', [None])[0],
                params=params
            )

            # Update visualization through manager
            self.visualization_manager.update_preprocessing_visualization(results)

            # Use Qt's single shot timer to ensure layers are created
            from qtpy.QtCore import QTimer
            def update_visibility():
                for layer in self.viewer.layers:
                    layer.visible = False
                    if layer.name == 'Bead Overlay':
                        layer.visible = True

            QTimer.singleShot(100, update_visibility)

            self._update_ui_state()
            self._update_status("Preprocessing complete", 100)
            self.preprocessing_completed.emit(results)

        except Exception as e:
            self._handle_error(f"Error handling preprocessing results: {str(e)}")
            self.processing_failed.emit(str(e))
        finally:
            self._set_controls_enabled(True)

    def _on_parameter_changed(self, param_name: str, value: object):
        """Handle parameter changes from the parameter manager"""
        # Only update if the change didn't come from this widget
        if not self.signalsBlocked():
            self._sync_widget_with_parameters()
            if self.preview_enabled:
                self.update_preview_frame()

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager values"""
        if not hasattr(self, 'parameter_manager') or self.parameter_manager is None:
            print("Warning: No parameter manager available for syncing")
            return

        # Block signals temporarily
        self._block_parameter_widgets(True)

        try:
            # Sync intensity ranges
            min_intensity = self.parameter_manager.get_value('min_intensity')
            max_intensity = self.parameter_manager.get_value('max_intensity')

            # Update spinboxes
            self.min_spinbox.setValue(min_intensity)
            self.max_spinbox.setValue(max_intensity)

            # Update slider (values need to be converted to slider range)
            self.intensity_slider.setValue((
                int(min_intensity * 10),
                int(max_intensity * 10)
            ))

            # Sync cell intensity ranges
            cell_min = self.parameter_manager.get_value('cell_min_intensity')
            cell_max = self.parameter_manager.get_value('cell_max_intensity')

            # Update cell spinboxes
            self.cell_min_spinbox.setValue(cell_min)
            self.cell_max_spinbox.setValue(cell_max)

            # Update cell slider
            self.cell_intensity_slider.setValue((
                int(cell_min * 10),
                int(cell_max * 10)
            ))

            # Sync Gaussian parameters
            gaussian_sigma = self.parameter_manager.get_value('gaussian_sigma')
            self.gaussian_sigma_spin.setValue(gaussian_sigma)
            self.gaussian_sigma_slider.setValue(int(gaussian_sigma * 10))

            cell_gaussian_sigma = self.parameter_manager.get_value('cell_gaussian_sigma')
            self.cell_gaussian_sigma_spin.setValue(cell_gaussian_sigma)
            self.cell_gaussian_sigma_slider.setValue(int(cell_gaussian_sigma * 10))

            # Sync registration mode
            reg_mode = self.parameter_manager.get_value('registration_mode')
            if isinstance(reg_mode, str):
                # Find the combo box item that matches when lowercased
                index = -1
                for i in range(self.registration_mode_combo.count()):
                    if self.registration_mode_combo.itemText(i).lower() == reg_mode.lower():
                        index = i
                        break
                if index >= 0:
                    self.registration_mode_combo.setCurrentIndex(index)

        except Exception as e:
            print(f"Error syncing parameters: {str(e)}")

        finally:
            # Restore signal handling
            self._block_parameter_widgets(False)

    def _connect_parameters(self):
        """Connect widget controls to parameter manager."""
        # Block signals during initial setup
        self._block_parameter_widgets(True)

        try:
            # Disconnect any existing connections to avoid duplicates
            if hasattr(self.parameter_manager, 'parameter_changed'):
                try:
                    self.parameter_manager.parameter_changed.disconnect(self._on_parameter_changed)
                except TypeError:
                    pass  # Connection didn't exist

            # Connect intensity range controls
            self.min_spinbox.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('min_intensity', value)
            )
            self.max_spinbox.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('max_intensity', value)
            )
            self.parameter_manager.register_callback(
                'min_intensity',
                lambda value: self._safe_set_value(self.min_spinbox, value)
            )
            self.parameter_manager.register_callback(
                'max_intensity',
                lambda value: self._safe_set_value(self.max_spinbox, value)
            )

            # Connect intensity slider
            self.intensity_slider.valueChanged.connect(self._update_intensity_labels)

            # Connect cell intensity range controls
            self.cell_min_spinbox.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('cell_min_intensity', value)
            )
            self.cell_max_spinbox.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('cell_max_intensity', value)
            )
            self.parameter_manager.register_callback(
                'cell_min_intensity',
                lambda value: self._safe_set_value(self.cell_min_spinbox, value)
            )
            self.parameter_manager.register_callback(
                'cell_max_intensity',
                lambda value: self._safe_set_value(self.cell_max_spinbox, value)
            )

            # Connect cell intensity slider
            self.cell_intensity_slider.valueChanged.connect(self._update_cell_intensity_labels)

            # Connect Gaussian sigma controls
            self.gaussian_sigma_spin.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('gaussian_sigma', value)
            )
            self.parameter_manager.register_callback(
                'gaussian_sigma',
                lambda value: self._safe_set_value(self.gaussian_sigma_spin, value)
            )
            self.gaussian_sigma_slider.valueChanged.connect(self._update_sigma_from_slider)

            # Connect cell Gaussian sigma controls
            self.cell_gaussian_sigma_spin.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('cell_gaussian_sigma', value)
            )
            self.parameter_manager.register_callback(
                'cell_gaussian_sigma',
                lambda value: self._safe_set_value(self.cell_gaussian_sigma_spin, value)
            )
            self.cell_gaussian_sigma_slider.valueChanged.connect(self._update_cell_sigma_from_slider)

            # Connect registration mode
            self.registration_mode_combo.currentTextChanged.connect(
                lambda text: self.parameter_manager.set_value(
                    'registration_mode',
                    text.lower()
                )
            )
            self.parameter_manager.register_callback(
                'registration_mode',
                lambda value: self._safe_set_combo_text(
                    self.registration_mode_combo,
                    value.title() if value else ''
                )
            )

        finally:
            self._block_parameter_widgets(False)

    def _safe_set_value(self, widget, value):
        """Safely set widget value with signal blocking."""
        if value is not None:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _safe_set_combo_text(self, combo, text):
        """Safely set combo box text with signal blocking."""
        combo.blockSignals(True)
        index = combo.findText(text, Qt.MatchFixedString)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _block_parameter_widgets(self, block: bool):
        """Block or unblock signals for all parameter-related widgets"""
        widgets = [
            self.intensity_slider,
            self.min_spinbox,
            self.max_spinbox,
            self.cell_intensity_slider,
            self.cell_min_spinbox,
            self.cell_max_spinbox,
            self.gaussian_sigma_spin,
            self.gaussian_sigma_slider,
            self.cell_gaussian_sigma_spin,
            self.cell_gaussian_sigma_slider,
            self.registration_mode_combo,
            self.preview_check
        ]
        for widget in widgets:
            widget.blockSignals(block)

    def reset_parameters(self):
        """Reset preprocessing-specific parameters to defaults."""
        try:
            # Reset only preprocessing parameters
            self.parameter_manager.reset_category_to_defaults(ParameterCategory.PREPROCESSING)

            # Synchronize widget values with reset parameters
            self._sync_widget_with_parameters()

            # Update preprocessor with new parameters
            params = PreprocessingParameters(
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

            self.preprocessor.update_parameters(params)

            # Update preview if enabled
            if self.preview_enabled:
                self.update_preview_frame()

            self._update_status("Preprocessing parameters reset to defaults")

        except Exception as e:
            self._handle_error(f"Error resetting parameters: {str(e)}")

    def update_parameters(self):
        """Update parameters in the parameter manager"""
        try:
            # Update intensity ranges
            self.parameter_manager.set_value('min_intensity', self.min_spinbox.value())
            self.parameter_manager.set_value('max_intensity', self.max_spinbox.value())

            # Update cell intensity ranges
            self.parameter_manager.set_value('cell_min_intensity', self.cell_min_spinbox.value())
            self.parameter_manager.set_value('cell_max_intensity', self.cell_max_spinbox.value())

            # Update Gaussian parameters
            self.parameter_manager.set_value('gaussian_sigma', self.gaussian_sigma_spin.value())
            self.parameter_manager.set_value('cell_gaussian_sigma', self.cell_gaussian_sigma_spin.value())

            # Update registration mode
            reg_mode = self.registration_mode_combo.currentText()
            self.parameter_manager.set_value(
                'registration_mode',
                reg_mode.lower()
            )

            # Update preprocessor with new parameters
            params = PreprocessingParameters(
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

            self.preprocessor.update_parameters(params)

            # Update preview if enabled
            if self.preview_enabled:
                self.update_preview_frame()

        except Exception as e:
            self._handle_error(str(e))

    def _get_active_image_layer(self):
        """Get currently active image layer"""
        from napari.layers import Image

        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            return None

        active_layer = selected_layers[0]
        if not isinstance(active_layer, Image):
            return None

        return active_layer

    def _validate_layer_for_data_type(self, layer, data_type: str) -> bool:
        """
        Validate if the given layer is suitable for the specified data type.

        Parameters
        ----------
        layer : napari.layers.Image
            The layer to validate
        data_type : str
            The type of data ('beads', 'reference', or 'cells')

        Returns
        -------
        bool
            True if the layer is valid for the data type, False otherwise
        """
        from napari.layers import Image

        if layer is None or not isinstance(layer, Image):
            return False

        data = layer.data

        if data_type == 'reference':
            # Reference image must be 2D
            return data.ndim == 2
        elif data_type in ['beads', 'cells']:
            # Bead and cell stacks must be 3D, or 2D (which can be converted to 3D)
            return data.ndim in [2, 3]

        return False

    def _on_layer_change(self, event=None):
        """Handle layer addition/removal events"""
        self._update_button_states()

    def _on_layer_selection_change(self, event=None):
        """Handle layer selection changes"""
        self._update_button_states()

    def _update_button_states(self):
        """Update the enabled state of load buttons based on available data"""
        active_layer = self._get_active_image_layer()

        # Update each button's enabled state based on layer validation
        self.load_beads_btn.setEnabled(
            self._validate_layer_for_data_type(active_layer, 'beads')
        )
        self.load_reference_btn.setEnabled(
            self._validate_layer_for_data_type(active_layer, 'reference')
        )
        self.load_cells_btn.setEnabled(
            self._validate_layer_for_data_type(active_layer, 'cells')
        )

    def _create_load_group(self):
        """Create the data loading group."""
        load_group = QGroupBox("Input Data")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(4)

        # Initialize buttons and status labels
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.load_beads_btn.setEnabled(False)  # Initially disabled
        self.load_beads_btn.setToolTip("Load a time series of bead images from the active layer in napari")

        self.load_reference_btn = QPushButton("Load Reference Image")
        self.load_reference_btn.setEnabled(False)  # Initially disabled
        self.load_reference_btn.setToolTip("Load a single reference image for registration from the active layer")

        self.load_cells_btn = QPushButton("Load Cell Stack")
        self.load_cells_btn.setEnabled(False)  # Initially disabled
        self.load_cells_btn.setToolTip("Load a time series of cell images from the active layer")

        self.bead_status = QLabel("Not loaded")
        self.reference_status = QLabel("Not loaded")
        self.cell_status = QLabel("Not loaded")

        # Add widgets with their status labels
        for btn, label in [
            (self.load_beads_btn, self.bead_status),
            (self.load_reference_btn, self.reference_status),
            (self.load_cells_btn, self.cell_status)
        ]:
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(btn)
            btn_layout.addWidget(label)
            load_layout.addLayout(btn_layout)

        load_group.setLayout(load_layout)
        return load_group

    def _create_intensity_group(self, title, slider, min_spinbox, max_spinbox, sigma_spinbox, tooltip_prefix=""):
        """Create a parameter group with intensity range and filter controls.

        Args:
            title: Group box title
            slider: Reference to store the range slider
            min_spinbox: Reference to store the minimum spinbox
            max_spinbox: Reference to store the maximum spinbox
            sigma_spinbox: Reference to store the gaussian sigma spinbox
            tooltip_prefix: Optional prefix for tooltip text
        """
        group = QGroupBox(title)
        layout = QVBoxLayout()

        # Create and store the range slider
        setattr(self, slider, QRangeSlider(Qt.Horizontal))
        slider_widget = getattr(self, slider)
        slider_widget.setToolTip(f"Adjust intensity range for {tooltip_prefix}contrast enhancement")
        slider_widget.setRange(0, 1000)
        slider_widget.setValue((0, 1000))
        layout.addWidget(slider_widget)

        # Create and store the spinboxes
        setattr(self, min_spinbox, QDoubleSpinBox())
        setattr(self, max_spinbox, QDoubleSpinBox())
        min_spin = getattr(self, min_spinbox)
        max_spin = getattr(self, max_spinbox)

        for spin in [min_spin, max_spin]:
            spin.setRange(0, 100)
            spin.setDecimals(1)
            spin.setSingleStep(0.1)

        min_spin.setToolTip(f"Set minimum intensity percentile for {tooltip_prefix}(0-100%)")
        max_spin.setToolTip(f"Set maximum intensity percentile for {tooltip_prefix}(0-100%)")
        max_spin.setValue(100)

        # Create spinbox layout
        spinbox_layout = QHBoxLayout()
        min_label = QLabel("Min ")
        min_label.setFixedWidth(40)
        spinbox_layout.addWidget(min_label)
        spinbox_layout.addWidget(min_spin)
        spinbox_layout.addStretch()
        spinbox_layout.addWidget(max_spin)
        spinbox_layout.addWidget(QLabel("Max "))
        layout.addLayout(spinbox_layout)

        # Create and add gaussian controls
        setattr(self, sigma_spinbox, QDoubleSpinBox())
        sigma_spin = getattr(self, sigma_spinbox)
        sigma_spin.setObjectName(sigma_spinbox)  # Set object name for slider reference
        sigma_spin.setToolTip(f"Set Gaussian blur sigma for {tooltip_prefix}(0 = disabled)")
        sigma_layout = self._create_sigma_control(sigma_spin)
        layout.addLayout(sigma_layout)

        group.setLayout(layout)
        return group

    def _create_intensity_range_group(self):
        """Create the bead/reference parameters group."""
        return self._create_intensity_group(
            title="Bead/Reference Parameters",
            slider="intensity_slider",
            min_spinbox="min_spinbox",
            max_spinbox="max_spinbox",
            sigma_spinbox="gaussian_sigma_spin"
        )

    def _create_cell_params_group(self):
        """Create the cell stack parameters group."""
        group = self._create_intensity_group(
            title="Cell Stack Parameters",
            slider="cell_intensity_slider",
            min_spinbox="cell_min_spinbox",
            max_spinbox="cell_max_spinbox",
            sigma_spinbox="cell_gaussian_sigma_spin",
            tooltip_prefix="cell images "
        )
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return group

    def _create_sigma_control(self, sigma_spinbox):
        """Create a layout with sigma control for Gaussian filter."""
        sigma_layout = QHBoxLayout()

        blur_label = QLabel("Blur")
        blur_label.setFixedWidth(40)
        sigma_layout.addWidget(blur_label)

        sigma_spinbox.setRange(0.0, 10.0)
        sigma_spinbox.setValue(0.0)
        sigma_spinbox.setSingleStep(0.1)
        sigma_spinbox.setDecimals(1)
        sigma_spinbox.setButtonSymbols(QDoubleSpinBox.PlusMinus)
        sigma_spinbox.setFixedWidth(105)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)

        sigma_layout.addWidget(sigma_spinbox)
        sigma_layout.addWidget(slider, stretch=1)

        # Determine slider name based on spinbox name
        slider_name = sigma_spinbox.objectName().replace('spin', 'slider')
        setattr(self, slider_name, slider)

        return sigma_layout

    def _create_parameters_group(self):
        """Create a group containing all parameter controls."""
        parameters_group = QGroupBox("Parameters")
        parameters_layout = QVBoxLayout()

        # Add existing parameter groups
        parameters_layout.addWidget(self._create_intensity_range_group())
        parameters_layout.addWidget(self._create_cell_params_group())
        parameters_layout.addWidget(self._create_registration_group())

        # Move reset button to bottom of parameters group
        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.setToolTip("Reset all parameters to default values")
        parameters_layout.addWidget(self.reset_btn)

        parameters_group.setLayout(parameters_layout)
        return parameters_group

    def _setup_ui(self):
        """Set up the complete user interface for the preprocessing widget."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create colorbar container
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

        # Right side container
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Add all component groups
        right_layout.addWidget(self._create_load_group())
        right_layout.addWidget(self._create_preview_selection_group())
        right_layout.addWidget(self._create_parameters_group())  # New grouped parameters
        right_layout.addWidget(self._create_preview_frame())

        # Create action buttons frame
        action_frame = QFrame()
        action_layout = QHBoxLayout()
        self.preprocess_btn = QPushButton("Run Preprocessing")
        self.preprocess_btn.setToolTip("Apply preprocessing to all loaded data")
        self.save_btn = QPushButton("Save Preprocessed Images")
        self.save_btn.setToolTip("Save preprocessed Images as TIFF files with calibration metadata")
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.preprocess_btn)
        action_layout.addWidget(self.save_btn)
        action_frame.setLayout(action_layout)

        right_layout.addWidget(action_frame)
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(360)

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

    def _create_preview_selection_group(self):
        """Create the preview selection group."""
        preview_select_group = QGroupBox("Preview Data Type")
        preview_layout = QHBoxLayout()

        self.bead_radio = QRadioButton("Bead Stack")
        self.bead_radio.setToolTip("Preview preprocessing effects on bead images")

        self.reference_radio = QRadioButton("Reference Image")
        self.reference_radio.setToolTip("Preview preprocessing effects on the reference image")

        self.cell_radio = QRadioButton("Cell Stack")
        self.cell_radio.setToolTip("Preview preprocessing effects on cell images")

        self.bead_radio.setChecked(True)

        for radio in [self.bead_radio, self.reference_radio, self.cell_radio]:
            preview_layout.addWidget(radio)

        preview_select_group.setLayout(preview_layout)
        return preview_select_group

    def _create_registration_group(self):
        """Create the registration group."""
        registration_group = QGroupBox("Registration")
        registration_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        registration_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.setToolTip(
            "Choose registration method:\n"
            "- Translation: Correct for x-y shifts\n"
            "- Rigid: Correct for rotation and translation\n"
            "- No registration: Disable registration"
        )
        self.registration_mode_combo.addItems(['Translation', 'Rigid', 'No registration'])
        self.registration_mode_combo.setCurrentText('Translation')
        mode_layout.addWidget(self.registration_mode_combo)
        mode_layout.addStretch()
        registration_layout.addLayout(mode_layout)

        self.registration_note = QLabel(
            "Note: Registration will be performed relative to the reference image."
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
        self.preview_check.setToolTip("Toggle live preview of preprocessing effects")
        preview_layout.addWidget(self.preview_check)
        preview_layout.addStretch()
        preview_frame.setLayout(preview_layout)
        return preview_frame

    def _handle_progress(self, update_dict):
        """Handle progress updates from the worker"""
        progress = update_dict['progress']
        message = update_dict['message']
        self._update_status(message, int(progress))

    def _update_sigma_from_slider(self):
        """Update sigma spinbox from slider value"""
        slider_value = self.gaussian_sigma_slider.value()
        sigma_value = slider_value / 10.0  # Convert 0-100 range to 0-10.0

        # Update spinbox without triggering its signal
        self.gaussian_sigma_spin.blockSignals(True)
        self.gaussian_sigma_spin.setValue(sigma_value)
        self.gaussian_sigma_spin.blockSignals(False)

        self.update_parameters()

    def _update_slider_from_sigma(self):
        """Update slider from sigma spinbox value"""
        sigma_value = self.gaussian_sigma_spin.value()
        slider_value = int(sigma_value * 10)  # Convert 0-10.0 range to 0-100

        # Update slider without triggering its signal
        self.gaussian_sigma_slider.blockSignals(True)
        self.gaussian_sigma_slider.setValue(slider_value)
        self.gaussian_sigma_slider.blockSignals(False)

        self.update_parameters()

    def _update_cell_sigma_from_slider(self):
        """Update cell sigma spinbox from slider value"""
        slider_value = self.cell_gaussian_sigma_slider.value()
        sigma_value = slider_value / 10.0  # Convert 0-100 range to 0-10.0

        # Update spinbox without triggering its signal
        self.cell_gaussian_sigma_spin.blockSignals(True)
        self.cell_gaussian_sigma_spin.setValue(sigma_value)
        self.cell_gaussian_sigma_spin.blockSignals(False)

        self.update_parameters()

    def _update_cell_slider_from_sigma(self):
        """Update cell slider from sigma spinbox value"""
        sigma_value = self.cell_gaussian_sigma_spin.value()
        slider_value = int(sigma_value * 10)  # Convert 0-10.0 range to 0-100

        # Update slider without triggering its signal
        self.cell_gaussian_sigma_slider.blockSignals(True)
        self.cell_gaussian_sigma_slider.setValue(slider_value)
        self.cell_gaussian_sigma_slider.blockSignals(False)

        self.update_parameters()

    def _update_calibration(self):
        """Update widget when calibration values change"""
        # If we have preprocessed data and save button is enabled, update the status
        # to show new calibration values
        if self.save_btn.isEnabled():
            pixel_size = self.pixel_size
            frame_length = self.frame_length
            self._update_status(
                f"Current calibration: pixel size = {pixel_size:.3f} µm, "
                f"frame length = {frame_length:.3f} min"
            )

    def _connect_signals(self):
        """Connect widget signals"""
        # Load button connections
        self.load_beads_btn.clicked.connect(lambda: self._load_active_layer('beads'))
        self.load_reference_btn.clicked.connect(lambda: self._load_active_layer('reference'))
        self.load_cells_btn.clicked.connect(lambda: self._load_active_layer('cells'))

        # Preview connections
        self.preview_check.toggled.connect(self.toggle_preview)
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.reset_btn.clicked.connect(self.reset_parameters)

        # Parameter change connections
        self.intensity_slider.valueChanged.connect(self._update_intensity_labels)
        self.min_spinbox.valueChanged.connect(self._update_slider_from_spinbox)
        self.max_spinbox.valueChanged.connect(self._update_slider_from_spinbox)

        self.cell_intensity_slider.valueChanged.connect(self._update_cell_intensity_labels)
        self.cell_min_spinbox.valueChanged.connect(self._update_cell_slider_from_spinbox)
        self.cell_max_spinbox.valueChanged.connect(self._update_cell_slider_from_spinbox)

        # Gaussian filter connections
        self.gaussian_sigma_spin.valueChanged.connect(self.update_parameters)
        self.cell_gaussian_sigma_spin.valueChanged.connect(self.update_parameters)

        # Registration connections
        self.registration_mode_combo.currentTextChanged.connect(self.update_parameters)

        # Data type selection connections
        self.bead_radio.toggled.connect(self._on_data_type_changed)
        self.reference_radio.toggled.connect(self._on_data_type_changed)
        self.cell_radio.toggled.connect(self._on_data_type_changed)

        # Save button connection
        self.save_btn.clicked.connect(self._save_preprocessed_data)

        self.gaussian_sigma_slider.valueChanged.connect(self._update_sigma_from_slider)
        self.gaussian_sigma_spin.valueChanged.connect(self._update_slider_from_sigma)

        self.cell_gaussian_sigma_slider.valueChanged.connect(self._update_cell_sigma_from_slider)
        self.cell_gaussian_sigma_spin.valueChanged.connect(self._update_cell_slider_from_sigma)

    def update_preview_frame(self):
        """Update the preview for the current frame"""
        if not self.preview_enabled:
            return

        try:
            # Force update by toggling preview
            self.toggle_preview(True)

        except Exception as e:
            self._handle_error(str(e))

    def cleanup(self):
        """Clean up resources and event connections."""
        # Ensure preview is disabled
        if self.preview_enabled:
            self.visualization_manager.handle_preview(
                frame=None,
                enable=False
            )

        if self.colorbar_manager is not None:
            self.colorbar_manager.cleanup()
            self.colorbar_manager = None

        super().cleanup()

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

    def _save_preprocessed_data(self):
        """Save preprocessed data to user-selected directory with ImageJ-compatible calibration metadata."""
        try:
            # Get directory from user
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Preprocessed Data",
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )

            if not save_dir:  # User cancelled
                return

            files_saved = []

            # Get calibration values
            pixel_size = self.pixel_size  # µm/pixel
            frame_length = self.frame_length  # minutes/frame

            # Function to save a single stack/image
            def save_tiff(data: np.ndarray, filename: str):
                if data is None:
                    return False

                # Convert to 16-bit
                data_normalized = data.astype(float)
                data_normalized = (data_normalized - data_normalized.min()) / (
                        data_normalized.max() - data_normalized.min())
                data_16bit = (data_normalized * 65535).astype(np.uint16)

                filepath = os.path.join(save_dir, filename)

                # Calculate scale (units per pixel)
                scale = pixel_size  # µm/pixel

                # Create ImageJ-compatible metadata
                imagej_metadata = {
                    'ImageJ': '1.53c',
                    'spacing': scale,
                    'unit': 'um',
                    'frame_interval': frame_length,
                    'frame_interval_unit': 'minute'
                }

                # For Z-stacks or time series, specify dimensions
                if data.ndim > 2:
                    imagej_metadata.update({
                        'frames': data.shape[0],
                        'slices': 1,
                        'channels': 1
                    })

                # Create description for ImageJ
                description = json.dumps({
                    'Info': f'Scale: {scale} um/pixel, Frame interval: {frame_length} min',
                    **imagej_metadata
                })

                # Original metadata preserved for compatibility
                metadata = {
                    'PhysicalSizeX': pixel_size,
                    'PhysicalSizeXUnit': 'um',
                    'PhysicalSizeY': pixel_size,
                    'PhysicalSizeYUnit': 'um',
                    'TimeIncrement': frame_length,
                    'TimeIncrementUnit': 'min',
                    **imagej_metadata
                }

                # Save with metadata using tifffile
                tifffile.imwrite(
                    filepath,
                    data_16bit,
                    imagej=True,
                    metadata=metadata,
                    description=description,
                    resolution=(1 / scale, 1 / scale),  # resolution in pixels per unit
                    photometric='minisblack'
                )
                return True

            # Save bead stack if available
            if self.data_manager.preprocessed_bead_stack is not None:
                if save_tiff(self.data_manager.preprocessed_bead_stack, "preprocessed_beads.tif"):
                    files_saved.append("preprocessed_beads.tif")

            # Save reference image if available
            if self.data_manager.preprocessed_reference is not None:
                if save_tiff(self.data_manager.preprocessed_reference, "preprocessed_reference.tif"):
                    files_saved.append("preprocessed_reference.tif")

            # Save cell stack if available
            if self.data_manager.preprocessed_cell_stack is not None:
                if save_tiff(self.data_manager.preprocessed_cell_stack, "preprocessed_cells.tif"):
                    files_saved.append("preprocessed_cells.tif")

            if files_saved:
                self._update_status(
                    f"Saved files with calibration (pixel size: {pixel_size} µm, "
                    f"frame length: {frame_length} min):\n" + "\n".join(files_saved)
                )
            else:
                self._update_status("No preprocessed data available to save")

        except Exception as e:
            error_msg = f"Error saving preprocessed data: {str(e)}"
            self._handle_error(error_msg)
            self.processing_failed.emit(error_msg)
