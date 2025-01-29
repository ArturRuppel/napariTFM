# TODO Gaussian blur doesn't do anything
# TODO add rangesliders
# TODO UI doesn't freeze during processing
# TODO reference button should have different enable/disable logic

from pathlib import Path
from typing import Optional, Any
import numpy as np
from qtpy.QtCore import Qt, Signal, QObject
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget, QSpinBox, QRadioButton, QFileDialog,
    QDoubleSpinBox, QPushButton, QFrame, QScrollArea, QCheckBox,
    QProgressBar, QMessageBox, QComboBox, QSizePolicy
)
import os
import tifffile
from qtrangeslider import QRangeSlider
from napari.layers import Image
from napari.viewer import Viewer
from napari.qt.threading import thread_worker

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.services.preprocessing_service import PreprocessingService


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
        self.load_beads_btn.clicked.connect(lambda: self.controller.load_active_layer('beads'))
        self.load_reference_btn.clicked.connect(lambda: self.controller.load_active_layer('reference'))
        self.load_cells_btn.clicked.connect(lambda: self.controller.load_active_layer('cells'))

    def update_button_states(self, active_layer_exists: bool = False):
        """Update button states based on layer selection."""
        active_layer = self.viewer.layers.selection.active
        has_valid_layer = (active_layer is not None and isinstance(active_layer, Image))

        self.load_beads_btn.setEnabled(has_valid_layer)
        self.load_reference_btn.setEnabled(has_valid_layer)
        self.load_cells_btn.setEnabled(has_valid_layer)

    def update_data_status(self):
        """Update status labels based on loaded data."""
        # Update bead status
        bead_data = self.data_manager.bead_stack
        if bead_data is not None:
            self.bead_status.setText(f"Loaded: {bead_data.shape}")
        else:
            self.bead_status.setText("Not loaded")

        # Update reference status
        ref_data = self.data_manager.reference
        if ref_data is not None:
            self.reference_status.setText(f"Loaded: {ref_data.shape}")
        else:
            self.reference_status.setText("Not loaded")

        # Update cell status
        cell_data = self.data_manager.cell_stack
        if cell_data is not None:
            self.cell_status.setText(f"Loaded: {cell_data.shape}")
        else:
            self.cell_status.setText("Not loaded")

    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        self.load_beads_btn.setEnabled(not frozen)
        self.load_reference_btn.setEnabled(not frozen)
        self.load_cells_btn.setEnabled(not frozen)


class PreprocessingParameterPanel(QWidget):
    """Panel for handling preprocessing parameter inputs."""

    parameter_changed = Signal()

    def __init__(self, parameter_manager: ParameterManager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_spins = {}
        self.parameter_combos = {}
        self._setup_ui()

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
        self._connect_signals()
        self._sync_widget_with_parameters()

    def _create_intensity_range_group(self):
        group = QGroupBox("Bead/Reference Parameters")
        layout = QVBoxLayout()

        # Add intensity parameters
        intensity_params = [
            ("min_intensity_percentile", "Min Intensity (%)", 0, 100, 0.1),
            ("max_intensity_percentile", "Max Intensity (%)", 0, 100, 0.1),
            ("gaussian_sigma", "Gaussian Sigma", 0.0, 10.0, 0.1)
        ]

        for name, label, min_val, max_val, step in intensity_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = self._create_spinbox(min_val, max_val, step)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_cell_params_group(self):
        group = QGroupBox("Cell Stack Parameters")
        layout = QVBoxLayout()

        # Add cell parameters
        cell_params = [
            ("cell_min_intensity_percentile", "Min Intensity (%)", 0, 100, 0.1),
            ("cell_max_intensity_percentile", "Max Intensity (%)", 0, 100, 0.1),
            ("cell_gaussian_sigma", "Gaussian Sigma", 0.0, 10.0, 0.1)
        ]

        for name, label, min_val, max_val, step in cell_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = self._create_spinbox(min_val, max_val, step)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_registration_group(self):
        group = QGroupBox("Registration")
        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.addItems(['Translation', 'Rigid', 'No registration'])
        self.parameter_combos['registration_mode'] = self.registration_mode_combo
        mode_layout.addWidget(self.registration_mode_combo)
        layout.addLayout(mode_layout)

        self.registration_note = QLabel(
            "Note: Registration will be performed relative to the reference image."
        )
        self.registration_note.setWordWrap(True)
        layout.addWidget(self.registration_note)

        group.setLayout(layout)
        return group

    def _create_spinbox(self, min_val: float, max_val: float, step: float, decimals: int = 1) -> QDoubleSpinBox:
        """Create a spinbox with given parameters."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _connect_signals(self):
        """Connect widget signals to parameter manager."""
        # Connect spinboxes
        for name, spin in self.parameter_spins.items():
            spin.valueChanged.connect(
                lambda v, n=name: self.parameter_manager.set_parameter(n, v)
            )

        # Connect comboboxes
        for name, combo in self.parameter_combos.items():
            combo.currentTextChanged.connect(
                lambda t, n=name: self.parameter_manager.set_parameter(n, t.lower())
            )

        # Connect reset button
        self.reset_btn.clicked.connect(self._reset_parameters)

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager values."""
        self._block_widgets(True)
        try:
            # Sync spinboxes
            for name, spin in self.parameter_spins.items():
                value = self.parameter_manager.get_parameter(name)
                self._safe_set_value(spin, value)

            # Sync comboboxes
            for name, combo in self.parameter_combos.items():
                value = self.parameter_manager.get_parameter(name)
                self._safe_set_combo_text(combo, str(value))

        finally:
            self._block_widgets(False)

    def _block_widgets(self, block: bool):
        """Block or unblock signals for all widgets."""
        for widget in list(self.parameter_spins.values()) + list(self.parameter_combos.values()):
            widget.blockSignals(block)

    def _safe_set_value(self, widget, value):
        """Safely set widget value."""
        if value is not None and widget is not None:
            widget.blockSignals(True)
            try:
                value = max(widget.minimum(), min(widget.maximum(), value))
                widget.setValue(value)
            except Exception as e:
                print(f"Error setting widget value: {str(e)}")
            widget.blockSignals(False)

    def _safe_set_combo_text(self, combo, text):
        """Safely set combo box text with case-insensitive matching."""
        if combo is not None and text is not None:
            combo.blockSignals(True)
            try:
                # Try exact match first
                index = combo.findText(str(text), Qt.MatchFixedString)
                if index < 0:
                    # If no exact match, try case-insensitive manual search
                    text_lower = str(text).lower()
                    for i in range(combo.count()):
                        if combo.itemText(i).lower() == text_lower:
                            index = i
                            break
                if index >= 0:
                    combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)
    def _reset_parameters(self):
        """Reset parameters to defaults."""
        self.parameter_manager.reset_preprocessing_parameters()
        self._sync_widget_with_parameters()
        self.parameter_changed.emit()

    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        for spin in self.parameter_spins.values():
            spin.setEnabled(not frozen)
        for combo in self.parameter_combos.values():
            combo.setEnabled(not frozen)
        self.reset_btn.setEnabled(not frozen)


class PreprocessingController(QObject):
    """Controller coordinating UI components and data processing."""

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

        self.parameter_panel = None
        self.data_panel = None
        self.preview_enabled = False
        self.current_data_type = 'beads'

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)

    def set_panels(self, parameter_panel, data_panel):
        """Set the parameter and data panels."""
        self.parameter_panel = parameter_panel
        self.data_panel = data_panel

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        if self.preview_enabled:
            self._update_preview()

    def _on_parameters_reset(self, category: ParameterCategory):
        """Handle parameter reset events."""
        if category == ParameterCategory.PREPROCESSING and self.preview_enabled:
            self._update_preview()

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
                self.data_manager.set_bead_stack(data)

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
            if self.preview_enabled:
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
                data = self.data_manager.bead_stack
            elif self.current_data_type == 'reference':
                data = self.data_manager.reference
            else:
                data = self.data_manager.cell_stack

            if data is None:
                raise ValueError(f"No {self.current_data_type} data available")

            # Get current frame if data is a stack
            if data.ndim == 3:
                current_step = min(self.viewer.dims.current_step[0], data.shape[0] - 1)
                frame = data[current_step].copy()
            else:
                frame = data.copy()

            # Get parameters and update service
            params = self.parameter_manager.get_preprocessing_parameters()
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

    @thread_worker
    def _process_data(self):
        """Process data in a background thread."""
        try:
            # Get parameters and update service
            params = self.parameter_manager.get_preprocessing_parameters()
            self.service.update_parameters(params)

            results = {}
            total_items = sum(1 for x in [self.data_manager.bead_stack,
                                          self.data_manager.reference,
                                          self.data_manager.cell_stack] if x is not None)
            items_processed = 0

            # Process bead stack
            if self.data_manager.bead_stack is not None:
                yield items_processed / total_items * 100, "Processing bead stack..."
                bead_results = list(self.service.preprocess_stack(
                    image_stack=self.data_manager.bead_stack,
                    reference_image=self.data_manager.reference
                ))
                results['beads'] = bead_results
                items_processed += 1

            # Process reference
            if self.data_manager.reference is not None:
                yield items_processed / total_items * 100, "Processing reference image..."
                ref_results = list(self.service.preprocess_stack(
                    image_stack=self.data_manager.reference
                ))
                results['reference'] = ref_results
                items_processed += 1

            # Process cell stack
            if self.data_manager.cell_stack is not None:
                yield items_processed / total_items * 100, "Processing cell stack..."
                cell_results = list(self.service.preprocess_stack(
                    image_stack=self.data_manager.cell_stack,
                    is_cell=True
                ))
                results['cells'] = cell_results
                items_processed += 1

            yield 100, "Processing complete"
            return results

        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")

    def run_preprocessing(self):
        """Execute preprocessing on loaded data using PreprocessingService."""
        try:
            self.preprocessing_started.emit()
            self.freeze_ui()

            # Validate data availability
            if self.data_manager.bead_stack is None:
                raise ValueError("Bead stack must be loaded before preprocessing")
            if self.data_manager.reference is None:
                raise ValueError("Reference image must be loaded before preprocessing")

            # Get parameters and initialize service
            params = self.parameter_manager.get_preprocessing_parameters()
            self.service.update_parameters(params)

            # Process bead stack
            bead_results = []
            for result, frame, total in self.service.preprocess_stack(
                    self.data_manager.bead_stack,
                    self.data_manager.reference
            ):
                bead_results.append(result)
                progress = (frame / total) * 100
                if self.data_manager.cell_stack is None:
                    self.progress_updated.emit(progress, f"Processing beads: Frame {frame}/{total}")
                else:
                    self.progress_updated.emit(progress / 2, f"Processing beads: Frame {frame}/{total}")

            # Process reference
            reference_result = self.service.preprocess_frame(self.data_manager.reference)

            # Process cell stack if available
            cell_results = []
            if self.data_manager.cell_stack is not None:
                for result, frame, total in self.service.preprocess_stack(
                        self.data_manager.cell_stack,
                        reference_image=None,
                        is_cell=True
                ):
                    cell_results.append(result)
                    progress = (frame / total) * 100
                    self.progress_updated.emit(50 + progress / 2, f"Processing cells: Frame {frame}/{total}")

            # Update data manager with processed data
            self.data_manager.set_preprocessed_bead_stack(
                np.stack([r.processed_image for r in bead_results])
            )
            self.data_manager.set_preprocessed_reference(reference_result.processed_image)

            if cell_results:
                self.data_manager.set_preprocessed_cell_stack(
                    np.stack([r.processed_image for r in cell_results])
                )

            # Let visualization manager handle visualization using data from data manager
            self.visualization_manager.update_preprocessing_visualization({
                'beads': (self.data_manager.preprocessed_bead_stack, self.data_manager.reference),
                'reference': (self.data_manager.preprocessed_reference, self.data_manager.reference),
                'parameters': params.__dict__
            })

            self.progress_updated.emit(100, "Preprocessing complete")
            self.preprocessing_completed.emit({
                'parameters': params.__dict__
            })

        except Exception as e:
            self.preprocessing_failed.emit(str(e))
            raise
        finally:
            self.unfreeze_ui()
    def _handle_preprocessing_results(self, results):
        """Handle successful preprocessing results."""
        try:
            # Update data manager with processed results
            if 'beads' in results:
                self.data_manager.set_preprocessed_bead_stack(
                    np.stack([r.processed_image for r in results['beads']])
                )

            if 'reference' in results:
                self.data_manager.set_preprocessed_reference(
                    results['reference'][0].processed_image
                )

            if 'cells' in results:
                self.data_manager.set_preprocessed_cell_stack(
                    np.stack([r.processed_image for r in results['cells']])
                )

            # Update visualization
            self.visualization_manager.update_preprocessing_visualization(results)

            self.progress_updated.emit(100, "Preprocessing complete")
            self.preprocessing_completed.emit(results)

        except Exception as e:
            error_msg = f"Error handling preprocessing results: {str(e)}"
            self.preprocessing_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
        finally:
            self.unfreeze_ui()

    def _handle_preprocessing_error(self, error):
        """Handle preprocessing error."""
        error_msg = str(error)
        self.preprocessing_failed.emit(error_msg)
        self.progress_updated.emit(0, f"Error: {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)
        self.unfreeze_ui()

    def save_preprocessed_data(self):
        """Save preprocessed data to files."""
        try:
            save_dir = QFileDialog.getExistingDirectory(
                None,
                "Select Directory to Save Preprocessed Data",
                str(Path.home()),
                QFileDialog.ShowDirsOnly
            )

            if not save_dir:
                return

            # Get metadata values
            pixel_size = self.parameter_manager.get_parameter('pixel_size')
            frame_interval = self.parameter_manager.get_parameter('frame_interval')

            files_saved = []

            # Helper function to save TIFF files
            def save_tiff(data, filename):
                if data is None:
                    return False

                filepath = os.path.join(save_dir, filename)
                metadata = {
                    'ImageJ': '1.53c',
                    'spacing': pixel_size,
                    'unit': 'um',
                    'frame_interval': frame_interval,
                    'frame_interval_unit': 'minute'
                }

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
                self.progress_updated.emit(
                    100,
                    f"Saved files with calibration:\n"
                    f"pixel size: {pixel_size} µm, frame interval: {frame_interval} min\n" +
                    "\n".join(files_saved)
                )
            else:
                self.progress_updated.emit(0, "No preprocessed data available to save")

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save data: {str(e)}")

    def freeze_ui(self):
        """Disable all interactive UI elements."""
        if self.data_panel:
            self.data_panel.freeze_ui(True)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(True)
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Re-enable UI elements and refresh state."""
        if self.data_panel:
            self.data_panel.freeze_ui(False)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(False)
        self.ui_frozen.emit(False)


class PreprocessingWidget(BaseAnalysisWidget):
    """Main preprocessing widget integrating all components."""

    preprocessing_completed = Signal(dict)  # Emits processed data

    def __init__(
            self,
            viewer: Viewer,
            data_manager,
            parameter_manager: ParameterManager,
            visualization_manager
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize managers and service
        self.parameter_manager = parameter_manager
        self.service = PreprocessingService(parameter_manager.get_preprocessing_parameters())
        self.colorbar_manager = ColorbarManager()

        # Initialize panels
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

        # Set controller in panels
        self.data_panel.set_controller(self.controller)

        # Set up UI and connections
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the main widget UI."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Add colorbar container
        colorbar_container = self._create_colorbar_container()
        main_layout.addWidget(colorbar_container)

        # Add main content
        content_container = self._create_content_container()
        main_layout.addWidget(content_container)

        self.setLayout(main_layout)

    def _create_colorbar_container(self) -> QWidget:
        """Create the colorbar container."""
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)

        colorbar_group = self.create_colorbar_widget(
            colormap_name='gray',
            label="Intensity Value",
            clim=(255, 0),
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        layout.addStretch()

        container.setLayout(layout)
        return container

    def _create_content_container(self) -> QWidget:
        """Create the main content container with scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        scroll.setFixedWidth(360)

        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        # Add components
        layout.addWidget(self.data_panel)
        layout.addWidget(self.parameter_panel)
        layout.addWidget(self._create_preview_frame())
        layout.addWidget(self._create_action_frame())
        layout.addWidget(self._create_status_frame())
        layout.addStretch()

        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll

    def _create_preview_frame(self) -> QFrame:
        """Create preview control frame."""
        frame = QFrame()
        layout = QVBoxLayout()

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
        layout.addWidget(type_group)

        # Preview toggle
        preview_layout = QHBoxLayout()
        self.preview_check = QCheckBox("Show Preview")
        preview_layout.addWidget(self.preview_check)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)

        frame.setLayout(layout)
        return frame

    def _create_action_frame(self) -> QFrame:
        """Create action buttons frame."""
        frame = QFrame()
        layout = QHBoxLayout()

        self.process_btn = QPushButton("Run Preprocessing")
        self.save_btn = QPushButton("Save Preprocessed Images")
        self.save_btn.setEnabled(False)

        layout.addWidget(self.process_btn)
        layout.addWidget(self.save_btn)

        frame.setLayout(layout)
        return frame

    def _create_status_frame(self) -> QFrame:
        """Create status display frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _connect_signals(self):
        """Connect all widget signals."""
        # Set controller in panels
        self.data_panel.set_controller(self.controller)

        # Connect preview controls
        self.preview_check.toggled.connect(self._on_preview_toggled)
        for radio in [self.bead_radio, self.reference_radio, self.cell_radio]:
            radio.toggled.connect(self._on_preview_type_changed)

        # Connect action buttons
        self.process_btn.clicked.connect(self._on_process_clicked)
        self.save_btn.clicked.connect(self._on_save_clicked)

        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.controller.preprocessing_failed.connect(self._on_preprocessing_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Connect parameter panel changes
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _on_preview_toggled(self, enabled: bool):
        """Handle preview toggle."""
        self.controller.toggle_preview(enabled)

    def _on_preview_type_changed(self):
        """Handle preview type selection change."""
        if self.bead_radio.isChecked():
            self.controller.current_data_type = 'beads'
        elif self.reference_radio.isChecked():
            self.controller.current_data_type = 'reference'
        else:
            self.controller.current_data_type = 'cells'

        if self.preview_check.isChecked():
            self.controller._update_preview()

    def _on_process_clicked(self):
        """Handle process button click."""
        try:
            if self.preview_check.isChecked():
                self.preview_check.setChecked(False)
            self.controller.run_preprocessing()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_save_clicked(self):
        """Handle save button click."""
        try:
            self.controller.save_preprocessed_data()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

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
                self.data_manager.bead_stack is not None or
                self.data_manager.reference is not None or
                self.data_manager.cell_stack is not None
        )
        self.preview_check.setEnabled(has_data)
        if not has_data and self.preview_check.isChecked():
            self.preview_check.setChecked(False)

        # Update action buttons
        self.process_btn.setEnabled(has_data)
        has_preprocessed = (
                self.data_manager.preprocessed_bead_stack is not None or
                self.data_manager.preprocessed_reference is not None or
                self.data_manager.preprocessed_cell_stack is not None
        )
        self.save_btn.setEnabled(has_preprocessed)

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze."""
        self.preview_check.setEnabled(not frozen)
        self.process_btn.setEnabled(not frozen)

        # Check if any preprocessed data exists
        has_preprocessed = (
                self.data_manager.preprocessed_bead_stack is not None or
                self.data_manager.preprocessed_reference is not None or
                self.data_manager.preprocessed_cell_stack is not None
        )

        self.save_btn.setEnabled(not frozen and has_preprocessed)

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        self.visualization_manager.cleanup()
        super().cleanup()