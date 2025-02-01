from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QObject
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QRadioButton, QFileDialog, QFrame, QScrollArea, QCheckBox, QApplication,
    QProgressBar, QMessageBox, QSizePolicy, QSpacerItem, QGridLayout
)
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget,
    QDoubleSpinBox, QPushButton, QComboBox, QSlider
)
from qtrangeslider import QRangeSlider

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.services.preprocessing_service import PreprocessingService


# TODO setting bead stack invalidates subsequent steps but setting reference doesn't
# TODO changing intensity in batch changes spinboxes in preprocessing but not sliders
# TODO fix rolling ball UI

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
        self.load_beads_btn.setFixedWidth(150)

        self.bead_status = QLabel("Not loaded")
        bead_layout.addWidget(self.load_beads_btn)
        bead_layout.addWidget(self.bead_status)
        group_layout.addLayout(bead_layout)

        # Reference data row
        ref_layout = QHBoxLayout()
        self.load_reference_btn = QPushButton("Load Reference Image")
        self.load_reference_btn.setFixedWidth(150)
        self.reference_status = QLabel("Not loaded")
        ref_layout.addWidget(self.load_reference_btn)
        ref_layout.addWidget(self.reference_status)
        group_layout.addLayout(ref_layout)

        # Cell data row
        cell_layout = QHBoxLayout()
        self.load_cells_btn = QPushButton("Load Cell Stack")
        self.load_cells_btn.setFixedWidth(150)
        self.cell_status = QLabel("Not loaded")
        cell_layout.addWidget(self.load_cells_btn)
        cell_layout.addWidget(self.cell_status)
        group_layout.addLayout(cell_layout)

        info_label = QLabel(
            "Required: Reference image and bead stack."
        )
        info_label.setWordWrap(True)
        group_layout.addWidget(info_label)

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

    parameter_changed = Signal(str, object)  # (param_name, value)
    parameters_reset = Signal()

    # region === Initialization
    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_spins = {}
        self.parameter_sliders = {}
        self.parameter_range_sliders = {}
        self.parameter_combos = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up the main UI elements."""
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

    # endregion === Initialization

    # region === UI Creation
    def _create_intensity_range_group(self):
        group = QGroupBox("Bead/Reference Parameters")
        layout = QVBoxLayout()

        # Add rolling ball radius control
        radius_layout = QHBoxLayout()
        radius_label = QLabel("Background Subtraction")
        radius_label.setToolTip("Radius for rolling ball background subtraction in pixels. Set to 0 to disable background subtraction.")
        radius_label.setFixedWidth(150)  # Increased width
        radius_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        radius_spin = self._create_spinbox(0, 50, 1)
        radius_spin.setToolTip(
            "Radius for rolling ball background subtraction.\n"
            "Larger values remove broader background variations.\n"
            "Set to 0 to disable background subtraction."
        )
        radius_spin.setFixedWidth(99)
        self.parameter_spins['rolling_ball_radius'] = radius_spin
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(radius_spin)
        layout.addLayout(radius_layout)

        # Create range slider
        intensity_slider = QRangeSlider(Qt.Horizontal)
        intensity_slider.setRange(0, 1000)  # 0-100.0 with 0.1 precision
        intensity_slider.setToolTip(
            "Intensity percentile range for bead detection.\n"
            "Adjust to include relevant beads while excluding noise and background."
        )
        self.parameter_range_sliders['intensity'] = intensity_slider
        layout.addWidget(intensity_slider)

        # Create spinboxes
        spinbox_layout = QHBoxLayout()
        min_label = QLabel("Min")
        min_label.setFixedWidth(40)

        min_spin = self._create_spinbox(0, 100, 0.1)
        min_spin.setToolTip(
            "Minimum intensity threshold percentile for bead detection.\n"
            "Lower values include dimmer beads but may increase noise."
        )
        max_spin = self._create_spinbox(0, 100, 0.1)
        max_spin.setToolTip(
            "Maximum intensity threshold percentile for bead detection.\n"
            "Higher values include brighter beads but may exclude valid data."
        )
        self.parameter_spins['min_intensity_percentile'] = min_spin
        self.parameter_spins['max_intensity_percentile'] = max_spin

        spinbox_layout.addWidget(min_label)
        spinbox_layout.addWidget(min_spin)
        spinbox_layout.addStretch()
        spinbox_layout.addWidget(max_spin)
        spinbox_layout.addWidget(QLabel("Max"))
        layout.addLayout(spinbox_layout)

        # Create Gaussian sigma controls
        sigma_layout = QHBoxLayout()
        blur_label = QLabel("Blur")
        blur_label.setFixedWidth(40)

        sigma_spin = self._create_spinbox(0, 10, 0.1)
        sigma_spin.setToolTip(
            "Standard deviation for Gaussian smoothing of bead images.\n"
            "Higher values reduce noise but may blur bead features."
        )
        sigma_spin.setFixedWidth(99)
        self.parameter_spins['gaussian_sigma'] = sigma_spin

        sigma_slider = QSlider(Qt.Horizontal)
        sigma_slider.setRange(0, 100)  # 0-10.0 with 0.1 precision
        self.parameter_sliders['gaussian_sigma'] = sigma_slider

        sigma_layout.addWidget(blur_label)
        sigma_layout.addWidget(sigma_spin)
        sigma_layout.addWidget(sigma_slider)
        layout.addLayout(sigma_layout)

        group.setLayout(layout)
        return group

    def _create_cell_params_group(self):
        group = QGroupBox("Cell Stack Parameters")
        layout = QVBoxLayout()

        # Create range slider
        cell_intensity_slider = QRangeSlider(Qt.Horizontal)
        cell_intensity_slider.setToolTip(
            "Intensity percentile range for cell detection.\n"
            "Adjust to clearly capture cell boundaries while excluding background."
        )
        cell_intensity_slider.setRange(0, 1000)  # 0-100.0 with 0.1 precision
        self.parameter_range_sliders['cell_intensity'] = cell_intensity_slider
        layout.addWidget(cell_intensity_slider)

        # Create spinboxes
        spinbox_layout = QHBoxLayout()
        min_label = QLabel("Min")
        min_label.setFixedWidth(40)

        min_spin = self._create_spinbox(0, 100, 0.1)
        min_spin.setToolTip(
            "Minimum intensity threshold percentile for cell detection.\n"
            "Lower values include dimmer cell regions but may increase noise."
        )
        max_spin = self._create_spinbox(0, 100, 0.1)
        max_spin.setToolTip(
            "Maximum intensity threshold percentile for cell detection.\n"
            "Higher values include brighter cell regions but may exclude valid data."
        )
        self.parameter_spins['cell_min_intensity_percentile'] = min_spin
        self.parameter_spins['cell_max_intensity_percentile'] = max_spin

        spinbox_layout.addWidget(min_label)
        spinbox_layout.addWidget(min_spin)
        spinbox_layout.addStretch()
        spinbox_layout.addWidget(max_spin)
        spinbox_layout.addWidget(QLabel("Max"))
        layout.addLayout(spinbox_layout)

        # Create Gaussian sigma controls
        sigma_layout = QHBoxLayout()
        blur_label = QLabel("Blur")
        blur_label.setFixedWidth(40)

        sigma_spin = self._create_spinbox(0, 10, 0.1)
        sigma_spin.setToolTip(
            "Standard deviation for Gaussian smoothing of cell images.\n"
            "Higher values reduce noise but may blur cell boundaries."
        )
        sigma_spin.setFixedWidth(99)
        self.parameter_spins['cell_gaussian_sigma'] = sigma_spin

        sigma_slider = QSlider(Qt.Horizontal)
        sigma_slider.setRange(0, 100)  # 0-10.0 with 0.1 precision
        self.parameter_sliders['cell_gaussian_sigma'] = sigma_slider

        sigma_layout.addWidget(blur_label)
        sigma_layout.addWidget(sigma_spin)
        sigma_layout.addWidget(sigma_slider)
        layout.addLayout(sigma_layout)

        group.setLayout(layout)
        return group

    def _create_registration_group(self):
        """Create the registration control group."""
        group = QGroupBox("Registration")
        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.setToolTip(
            "Method for aligning image sequences:\n"
            "- Translation: Corrects x-y drift\n"
            "- Rigid: Corrects drift and rotation\n"
            "- No registration: Uses raw images"
        )
        # Ensure exact same strings as BatchAnalysisWidget
        self.registration_mode_combo.addItems(['Translation', 'Rigid', 'No registration'])
        self.parameter_combos['registration_mode'] = self.registration_mode_combo
        mode_layout.addWidget(self.registration_mode_combo)
        layout.addLayout(mode_layout)

        # Connect signal with case handling
        self.registration_mode_combo.currentTextChanged.connect(
            lambda text: self.parameter_manager.set_parameter(
                'registration_mode',
                text.lower()  # Convert to lowercase to match batch widget
            )
        )

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

    # endregion === UI Creation

    # region === Signal Handling
    def _connect_signals(self):
        """Connect all parameter control signals."""
        # Connect rolling ball radius controls
        radius_spin = self.parameter_spins['rolling_ball_radius']
        radius_spin.valueChanged.connect(lambda v: self._direct_parameter_update('rolling_ball_radius', v))

        # Connect intensity range controls
        intensity_slider = self.parameter_range_sliders['intensity']
        intensity_slider.valueChanged.connect(self._update_intensity_from_slider)

        min_spin = self.parameter_spins['min_intensity_percentile']
        max_spin = self.parameter_spins['max_intensity_percentile']
        min_spin.valueChanged.connect(self._update_intensity_from_spinbox)
        max_spin.valueChanged.connect(self._update_intensity_from_spinbox)

        # Connect cell intensity range controls
        cell_slider = self.parameter_range_sliders['cell_intensity']
        cell_slider.valueChanged.connect(self._update_cell_intensity_from_slider)

        cell_min_spin = self.parameter_spins['cell_min_intensity_percentile']
        cell_max_spin = self.parameter_spins['cell_max_intensity_percentile']
        cell_min_spin.valueChanged.connect(self._update_cell_intensity_from_spinbox)
        cell_max_spin.valueChanged.connect(self._update_cell_intensity_from_spinbox)

        # Connect Gaussian sigma controls
        for param in ['gaussian_sigma', 'cell_gaussian_sigma']:
            slider = self.parameter_sliders[param]
            spin = self.parameter_spins[param]

            slider.valueChanged.connect(
                lambda v, p=param: self._update_sigma_from_slider(p, v)
            )
            spin.valueChanged.connect(
                lambda v, p=param: self._update_sigma_from_spinbox(p, v)
            )

        # Connect registration mode
        self.registration_mode_combo.currentTextChanged.connect(
            lambda text: self.parameter_manager.set_parameter(
                'registration_mode',
                text.lower()
            )
        )

        # Connect reset button
        self.reset_btn.clicked.connect(self._reset_parameters)

    def _direct_parameter_update(self, param_name: str, value: float):
        """Directly update parameter manager and emit change signal."""
        self.parameter_manager.set_parameter(param_name, value)
        self.parameter_changed.emit(param_name, value)

    def _update_intensity_from_slider(self, values):
        """Update intensity spinboxes from range slider."""
        min_val, max_val = values
        min_percent = min_val / 10.0
        max_percent = max_val / 10.0

        # Update spinboxes without triggering their signals
        min_spin = self.parameter_spins['min_intensity_percentile']
        max_spin = self.parameter_spins['max_intensity_percentile']

        self._block_widgets(True)
        min_spin.setValue(min_percent)
        max_spin.setValue(max_percent)
        self._block_widgets(False)

        # Update parameters
        self.parameter_manager.set_parameter('min_intensity_percentile', min_percent)
        self.parameter_manager.set_parameter('max_intensity_percentile', max_percent)
        # Update these two lines to include parameters
        self.parameter_changed.emit('min_intensity_percentile', min_percent)
        self.parameter_changed.emit('max_intensity_percentile', max_percent)

    def _update_intensity_from_spinbox(self):
        """Update intensity slider from spinboxes."""
        min_spin = self.parameter_spins['min_intensity_percentile']
        max_spin = self.parameter_spins['max_intensity_percentile']
        slider = self.parameter_range_sliders['intensity']

        min_val = int(min_spin.value() * 10)
        max_val = int(max_spin.value() * 10)

        # Update slider without triggering its signal
        slider.blockSignals(True)
        slider.setValue((min_val, max_val))
        slider.blockSignals(False)

        # Update parameters
        self.parameter_manager.set_parameter('min_intensity_percentile', min_spin.value())
        self.parameter_manager.set_parameter('max_intensity_percentile', max_spin.value())
        # Update these two lines to include parameters
        self.parameter_changed.emit('min_intensity_percentile', min_spin.value())
        self.parameter_changed.emit('max_intensity_percentile', max_spin.value())

    def _update_sigma_from_slider(self, param: str, value: int):
        """Update sigma spinbox from slider."""
        sigma_value = value / 10.0
        spin = self.parameter_spins[param]

        spin.blockSignals(True)
        spin.setValue(sigma_value)
        spin.blockSignals(False)

        self.parameter_manager.set_parameter(param, sigma_value)
        # Update this line to include parameter
        self.parameter_changed.emit(param, sigma_value)

    def _update_sigma_from_spinbox(self, param: str, value: float):
        """Update sigma slider from spinbox."""
        slider_value = int(value * 10)
        slider = self.parameter_sliders[param]

        slider.blockSignals(True)
        slider.setValue(slider_value)
        slider.blockSignals(False)

        self.parameter_manager.set_parameter(param, value)
        # Update this line to include parameter
        self.parameter_changed.emit(param, value)

    def _update_cell_intensity_from_slider(self, values):
        """Update cell intensity spinboxes from range slider."""
        min_val, max_val = values
        min_percent = min_val / 10.0
        max_percent = max_val / 10.0

        # Update spinboxes without triggering their signals
        min_spin = self.parameter_spins['cell_min_intensity_percentile']
        max_spin = self.parameter_spins['cell_max_intensity_percentile']

        self._block_widgets(True)
        min_spin.setValue(min_percent)
        max_spin.setValue(max_percent)
        self._block_widgets(False)

        # Update parameters
        self.parameter_manager.set_parameter('cell_min_intensity_percentile', min_percent)
        self.parameter_manager.set_parameter('cell_max_intensity_percentile', max_percent)
        # Update these two lines to include parameters
        self.parameter_changed.emit('cell_min_intensity_percentile', min_percent)
        self.parameter_changed.emit('cell_max_intensity_percentile', max_percent)

    def _update_cell_intensity_from_spinbox(self):
        """Update cell intensity slider from spinboxes."""
        min_spin = self.parameter_spins['cell_min_intensity_percentile']
        max_spin = self.parameter_spins['cell_max_intensity_percentile']
        slider = self.parameter_range_sliders['cell_intensity']

        min_val = int(min_spin.value() * 10)
        max_val = int(max_spin.value() * 10)

        # Update slider without triggering its signal
        slider.blockSignals(True)
        slider.setValue((min_val, max_val))
        slider.blockSignals(False)

        # Update parameters
        self.parameter_manager.set_parameter('cell_min_intensity_percentile', min_spin.value())
        self.parameter_manager.set_parameter('cell_max_intensity_percentile', max_spin.value())
        # Update these two lines to include parameters
        self.parameter_changed.emit('cell_min_intensity_percentile', min_spin.value())
        self.parameter_changed.emit('cell_max_intensity_percentile', max_spin.value())

    # endregion === Signal Handling

    # region === Parameter Management
    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager values."""
        # Existing code for registration mode...

        # Sync spin boxes
        for name, spin in self.parameter_spins.items():
            value = self.parameter_manager.get_parameter(name)
            if name == 'young_modulus':
                value = value / 1000  # Convert Pa to kPa for display
            elif name == 'gel_height' and value is None:
                value = 0
            if isinstance(spin, tuple):
                # Handle special cases
                spin_widget, slider = spin
                self._safe_set_value(spin_widget, value)
                self._safe_set_value(slider, value)
            else:
                self._safe_set_value(spin, value)

        # Sync bead intensity range slider
        min_intensity = self.parameter_manager.get_parameter('min_intensity_percentile')
        max_intensity = self.parameter_manager.get_parameter('max_intensity_percentile')
        min_slider_val = int(min_intensity * 10)  # Convert 0.0-100.0 → 0-1000
        max_slider_val = int(max_intensity * 10)
        self.parameter_range_sliders['intensity'].setValue((min_slider_val, max_slider_val))

        # Sync cell intensity range slider
        cell_min = self.parameter_manager.get_parameter('cell_min_intensity_percentile')
        cell_max = self.parameter_manager.get_parameter('cell_max_intensity_percentile')
        cell_min_slider = int(cell_min * 10)
        cell_max_slider = int(cell_max * 10)
        self.parameter_range_sliders['cell_intensity'].setValue((cell_min_slider, cell_max_slider))

        # Sync Gaussian sigma sliders
        sigma = self.parameter_manager.get_parameter('gaussian_sigma')
        cell_sigma = self.parameter_manager.get_parameter('cell_gaussian_sigma')
        self.parameter_sliders['gaussian_sigma'].setValue(int(sigma * 10))  # 0.0-10.0 → 0-100
        self.parameter_sliders['cell_gaussian_sigma'].setValue(int(cell_sigma * 10))

    def _reset_parameters(self):
        """Reset all parameters to their default values."""
        self.parameter_manager.reset_preprocessing_parameters()
        self._sync_widget_with_parameters()
        self.parameters_reset.emit()

    def update_parameter(self, name: str, value: Any):
        """Update a single parameter value."""
        try:
            if name in self.parameter_spins:
                self._safe_set_value(self.parameter_spins[name], value)
            elif name == 'registration_mode' and value is not None:
                self._safe_set_combo_text(self.registration_mode_combo, str(value))
            elif name in self.parameter_combos:
                self._safe_set_combo_text(self.parameter_combos[name], str(value))
        except Exception as e:
            print(f"Error updating parameter {name}: {str(e)}")

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

    # endregion === Parameter Management

    # region === State Management
    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        for widget_dict in [
            self.parameter_spins,
            self.parameter_sliders,
            self.parameter_range_sliders,
            self.parameter_combos
        ]:
            for widget in widget_dict.values():
                widget.setEnabled(not frozen)

        self.reset_btn.setEnabled(not frozen)

    def _block_widgets(self, block: bool):
        """Block or unfreeze all widget signals."""
        for widgets in [
            self.parameter_spins.values(),
            self.parameter_sliders.values(),
            self.parameter_range_sliders.values(),
            self.parameter_combos.values()
        ]:
            for widget in widgets:
                widget.blockSignals(block)

    # endregion === State Management


class PreprocessingController(QObject):
    """Controller coordinating UI components and data processing."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    preprocessing_started = Signal()
    preprocessing_completed = Signal(dict)  # Results dictionary
    preprocessing_failed = Signal(str)  # Error message
    preview_updated = Signal()
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    # region === Initialization
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

        # Connect to viewer events for frame changes
        self.connect_viewer_events()

    def set_panels(self, parameter_panel, data_panel):
        """Set the parameter and data panels."""
        self.parameter_panel = parameter_panel
        self.data_panel = data_panel

    def connect_viewer_events(self):
        """Connect to viewer dimension events for frame changes."""
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

    def _on_frame_changed(self, event=None):
        """Handle frame change events from the viewer."""
        if self.preview_enabled:
            self._update_preview()

    # endregion === Initialization

    # region === Processing Execution
    def run_preprocessing(self):
        """Execute preprocessing on loaded data."""
        try:
            if self.data_manager.bead_stack is None:
                raise ValueError("Bead stack must be loaded before preprocessing")
            if self.data_manager.reference is None:
                raise ValueError("Reference image must be loaded before preprocessing")

            worker = self._create_processing_worker()
            self.active_workers.append(worker)
            self.preprocessing_started.emit()
            self.freeze_ui()

            worker.yielded.connect(lambda x: self.progress_updated.emit(*x))
            worker.returned.connect(self._handle_preprocessing_results)
            worker.errored.connect(self._handle_preprocessing_error)
            worker.start()

        except Exception as e:
            self.preprocessing_failed.emit(str(e))
            QMessageBox.critical(None, "Error", str(e))
            self.unfreeze_ui()

    @thread_worker
    def _create_processing_worker(self):
        """Create worker for processing data."""
        params = self.parameter_manager.get_preprocessing_parameters()
        self.service.update_parameters(params)

        results = []

        # Process bead stack with generator
        if self.data_manager.bead_stack is not None:
            for result, frame, total in self.service.preprocess_stack(
                    image_stack=self.data_manager.bead_stack,
                    reference_image=self.data_manager.reference
            ):
                results.append(result)
                yield frame / total * 100, f"Processing beads: Frame {frame + 1}/{total}"

        # Process reference image
        if self.data_manager.reference is not None:
            results.append(self.service.preprocess_frame(self.data_manager.reference))

        # Process cell stack if available
        if self.data_manager.cell_stack is not None:
            start_progress = len(results)
            for result, frame, total in self.service.preprocess_stack(
                    image_stack=self.data_manager.cell_stack,
                    is_cell=True
            ):
                results.append(result)
                yield start_progress + frame / total * 100, f"Processing cells: Frame {frame + 1}/{total}"

        return results

    def cancel_all_operations(self):
        """Cancel all running background operations."""
        if not self.active_workers:
            # No active workers, just update status
            self.progress_updated.emit(0, "No active operations to cancel")
            return

        for worker in self.active_workers:
            try:
                worker.quit()  # This should be sufficient for napari workers
            except Exception as e:
                print(f"Warning: Could not quit worker cleanly: {str(e)}")

        self.active_workers.clear()

        # Update UI status and ensure responsiveness
        self.progress_updated.emit(0, "Operations cancelled")
        QApplication.processEvents()
        self.unfreeze_ui()

    # endregion === Processing Execution

    # region === Parameter Handling
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        if self.preview_enabled:
            self._update_preview()

    def _on_parameters_reset(self, category: ParameterCategory):
        """Handle parameter reset events."""
        if category == ParameterCategory.PREPROCESSING and self.preview_enabled:
            self._update_preview()

    # endregion === Parameter Handling

    # region === Data Management
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
                self.data_manager.set_reference(data)

            elif data_type == 'cells':
                # Convert 2D data to 3D with single frame if needed
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                elif data.ndim != 3:
                    raise ValueError("Cell stack must be 2D or 3D (frames, height, width)")
                self.data_manager.set_cell_stack(data)
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
            self.preview_enabled = enabled

            if enabled:
                self._update_preview()
            else:
                self.visualization_manager.handle_preview(
                    frame=None,
                    enable=False
                )

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
                # Get the current frame from viewer dimensions
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

            # Update status with frame information and current frame number
            info = result.info
            frame_info = f" (Frame {current_step + 1}/{data.shape[0]})" if data.ndim == 3 else ""
            status = (
                f"Preview{frame_info}\n"
                f"Original range: ({info['original_range'][0]:.1f}, {info['original_range'][1]:.1f})\n"
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

    def _handle_preprocessing_error(self, error):
        """Handle preprocessing error."""
        error_msg = str(error)
        self.preprocessing_failed.emit(error_msg)
        self.progress_updated.emit(0, f"Error: {error_msg}")
        QMessageBox.critical(None, "Error", error_msg)
        self.unfreeze_ui()

    def _handle_preprocessing_results(self, results):
        """Handle successful preprocessing results."""
        try:
            if results is None:
                return

            # Process all results at once
            processed_images = [r.processed_image for r in results]

            # Update data manager
            if self.data_manager.bead_stack is not None:
                n_beads = self.data_manager.bead_stack.shape[0]
                self.data_manager.set_preprocessed_bead_stack(np.stack(processed_images[:n_beads]))
                processed_images = processed_images[n_beads:]

            if self.data_manager.reference is not None:
                self.data_manager.set_preprocessed_reference(processed_images[0])
                processed_images = processed_images[1:]

            if self.data_manager.cell_stack is not None:
                n_cells = self.data_manager.cell_stack.shape[0]
                self.data_manager.set_preprocessed_cell_stack(np.stack(processed_images[:n_cells]))

            # Update visualization
            self.visualization_manager.update_preprocessing_visualization()

            # Manage layer visibility
            bead_overlay_layer = None

            # First pass: find the bead overlay layer and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Bead Overlay':
                    bead_overlay_layer = layer
                    layer.visible = True
                else:
                    layer.visible = False

            # If bead overlay exists, move it to the top (index 0)
            if bead_overlay_layer is not None:
                current_index = self.viewer.layers.index(bead_overlay_layer)
                # Move to index 0 (top-most position)
                if current_index > 0:
                    self.viewer.layers.move(current_index, -1)

            # Get current parameters for the completion signal
            current_params = self.parameter_manager.get_preprocessing_parameters()

            self.progress_updated.emit(100, "Preprocessing complete")
            self.preprocessing_completed.emit({'parameters': current_params.__dict__})

        except Exception as e:
            error_msg = f"Error handling preprocessing results: {str(e)}"
            self.preprocessing_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
        finally:
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

            save_dir = Path(save_dir)
            files_saved = []

            # Get calibration parameters
            pixel_size = self.parameter_manager.get_parameter('pixel_size')
            frame_interval = self.parameter_manager.get_parameter('frame_interval')

            # Helper function to save TIFF files
            def _save_calibrated_tiff(data: np.ndarray, filepath: Path) -> bool:
                """
                Save data as calibrated TIFF file with ImageJ-compatible metadata.

                Args:
                    data: numpy array to save
                    filepath: path where to save the file

                Returns:
                    bool: True if save was successful
                """
                if data is None:
                    return False

                # Convert to 16-bit
                data_normalized = data.astype(float)
                data_normalized = (data_normalized - data_normalized.min()) / (
                        data_normalized.max() - data_normalized.min())
                data_16bit = (data_normalized * 65535).astype(np.uint16)

                # Create ImageJ-compatible metadata
                imagej_metadata = {
                    'ImageJ': '1.53c',
                    'spacing': pixel_size,
                    'unit': 'um',
                    'frame_interval': frame_interval,
                    'frame_interval_unit': 'minute'
                }

                # For Z-stacks or time series, specify dimensions
                if data.ndim > 2:
                    imagej_metadata.update({
                        'frames': data.shape[0],
                        'slices': 1,
                        'channels': 1
                    })

                # Combine metadata for compatibility
                metadata = {
                    'PhysicalSizeX': pixel_size,
                    'PhysicalSizeXUnit': 'um',
                    'PhysicalSizeY': pixel_size,
                    'PhysicalSizeYUnit': 'um',
                    'TimeIncrement': frame_interval,
                    'TimeIncrementUnit': 'min',
                    **imagej_metadata
                }

                # Save with metadata using tifffile
                tifffile.imwrite(
                    str(filepath),
                    data_16bit,
                    imagej=True,
                    metadata=metadata,
                    resolution=(1 / pixel_size, 1 / pixel_size),  # resolution in pixels per unit
                    photometric='minisblack'
                )

                return True

            # Save each data type if available
            if self.data_manager.preprocessed_bead_stack is not None:
                bead_path = save_dir / "preprocessed_beads.tif"
                if _save_calibrated_tiff(self.data_manager.preprocessed_bead_stack, bead_path):
                    files_saved.append("preprocessed_beads.tif")

            if self.data_manager.preprocessed_reference is not None:
                ref_path = save_dir / "preprocessed_reference.tif"
                if _save_calibrated_tiff(self.data_manager.preprocessed_reference, ref_path):
                    files_saved.append("preprocessed_reference.tif")

            if self.data_manager.preprocessed_cell_stack is not None:
                cell_path = save_dir / "preprocessed_cells.tif"
                if _save_calibrated_tiff(self.data_manager.preprocessed_cell_stack, cell_path):
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

    # endregion === Data Management

    # region === State Management
    def freeze_ui(self):
        """Disable all interactive UI elements except cancel button."""
        if self.data_panel is not None:
            self.data_panel.freeze_ui(True)
        if self.parameter_panel is not None:
            self.parameter_panel.freeze_ui(True)
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Re-enable UI elements and refresh state."""
        if self.data_panel is not None:
            self.data_panel.freeze_ui(False)
        if self.parameter_panel is not None:
            self.parameter_panel.freeze_ui(False)
        self.ui_frozen.emit(False)

    # endregion === State Management


class PreprocessingWidget(BaseAnalysisWidget):
    """Main preprocessing widget integrating all components."""

    preprocessing_completed = Signal(dict)  # Emits processed data

    # region === Initialization
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

        self.controller.set_panels(self.parameter_panel, self.data_panel)

        # Set up UI and connections
        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    # endregion

    # region === UI Creation
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        # Left: Colorbar
        colorbar_container = self._create_colorbar_container()
        colorbar_container.setFixedWidth(100)
        main_layout.addWidget(colorbar_container)

        # Right: Scrollable content
        content_container = self._create_content_container()
        main_layout.addWidget(content_container)

        self.setLayout(main_layout)

    def _create_colorbar_container(self) -> QWidget:
        """Create the colorbar container."""
        container = QWidget()
        layout = QVBoxLayout()

        colorbar_group = self.create_colorbar_widget(
            colormap_name='gray',
            label="Relative Intensity Value",
            clim=(1, 0),
            colorbar_manager=self.colorbar_manager
        )
        layout.addWidget(colorbar_group, alignment=Qt.AlignTop)

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

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add components
        layout.addWidget(self.data_panel)
        layout.addItem(QSpacerItem(0, -12, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self.parameter_panel)
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_preview_frame())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_action_frame())
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
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
        layout = QVBoxLayout()  # Changed to VBoxLayout for stacked buttons

        # Create button row
        button_layout = QHBoxLayout()
        self.process_btn = QPushButton("Run Preprocessing")
        self.save_btn = QPushButton("Save Result Images")
        self.process_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.save_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.save_btn)
        layout.addLayout(button_layout)

        # Add cancel button (full width)
        self.cancel_btn = QPushButton("Cancel Operation")
        layout.addWidget(self.cancel_btn)

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

    # endregion

    # region === Signal Handling
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
        self.cancel_btn.clicked.connect(self.controller.cancel_all_operations)

        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.preprocessing_completed.connect(self._on_preprocessing_completed)
        self.controller.preprocessing_failed.connect(self._on_preprocessing_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)
        # Connect parameter panel changes
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)
        self.parameter_panel.parameters_reset.connect(self._on_parameters_reset)

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _sync_parameter(self, param_name: str, value: Any):
        """Sync a single parameter change from parameter manager"""
        if self.parameter_panel:
            self.parameter_panel.update_parameter(param_name, value)

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

    def _on_parameters_reset(self):
        """Handle parameter reset and update status."""
        self._update_status(0, "Preprocessing parameters reset to default values.")

    # endregion

    # region === State Management
    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _has_required_data(self) -> bool:
        """Check if required data for processing is loaded."""
        return (self.data_manager.bead_stack is not None and
                self.data_manager.reference is not None)

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update data panel
        self.data_panel.update_button_states()
        self.data_panel.update_data_status()

        # Update preview controls
        has_any_data = (
                self.data_manager.bead_stack is not None or
                self.data_manager.reference is not None or
                self.data_manager.cell_stack is not None
        )
        self.preview_check.setEnabled(has_any_data)

        # Update radio button availability based on loaded data
        self.bead_radio.setEnabled(self.data_manager.bead_stack is not None)
        self.reference_radio.setEnabled(self.data_manager.reference is not None)
        self.cell_radio.setEnabled(self.data_manager.cell_stack is not None)

        # Uncheck preview if no data
        if not has_any_data and self.preview_check.isChecked():
            self.preview_check.setChecked(False)

        # Update action buttons - now uses _has_required_data()
        self.process_btn.setEnabled(self._has_required_data())

        # Update save button based on preprocessed data
        has_preprocessed = (
                self.data_manager.preprocessed_bead_stack is not None or
                self.data_manager.preprocessed_reference is not None or
                self.data_manager.preprocessed_cell_stack is not None
        )
        self.save_btn.setEnabled(has_preprocessed)

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze."""
        # Disable preview and process buttons during processing
        self.preview_check.setEnabled(not frozen)
        self.process_btn.setEnabled(not frozen and self._has_required_data())

        # Cancel button is always enabled
        self.cancel_btn.setEnabled(True)

        # Disable radio buttons during processing
        self.bead_radio.setEnabled(not frozen)
        self.reference_radio.setEnabled(not frozen)
        self.cell_radio.setEnabled(not frozen)

        # Update save button based on preprocessed data availability
        has_preprocessed = (
                self.data_manager.preprocessed_bead_stack is not None or
                self.data_manager.preprocessed_reference is not None or
                self.data_manager.preprocessed_cell_stack is not None
        )
        self.save_btn.setEnabled(not frozen and has_preprocessed)

    def _check_preprocessed_data(self) -> bool:
        """Check availability of preprocessed data."""
        return (self.data_manager.preprocessed_bead_stack is not None or
                self.data_manager.preprocessed_reference is not None or
                self.data_manager.preprocessed_cell_stack is not None)

    # endregion

    # region === Results Handling
    def _on_preprocessing_completed(self, results):
        """Handle preprocessing completion."""
        self.save_btn.setEnabled(True)
        self.preprocessing_completed.emit(results)

    def _on_preprocessing_failed(self, error_msg: str):
        """Handle preprocessing failure."""
        self.save_btn.setEnabled(False)
        QMessageBox.critical(self, "Error", error_msg)

    # endregion

    # region === Cleanup
    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        self.visualization_manager.cleanup()
        super().cleanup()

    # endregion
