from pathlib import Path
from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtCore import QObject
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (QFileDialog, QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
                            QSizePolicy, QProgressBar, QLabel, QFrame, QSpacerItem)

from utilities.colorbar import ColorbarManager
from utilities.data_manager import DataManager
from utilities.parameter_manager import ParameterCategory, ParameterManager
from utilities.visualization_manager import VisualizationManager

from services.fttc_service import FTTCService, FTTCResult

from widgets._base_widget import BaseAnalysisWidget

# TODO layer visibility after calculations (preview and full)
# TODO load displacement data should trigger visualization
# TODO load displacment button disables when trying to load the wrong file
# TODO review button disable/enable logic
# TODO preview button should clear vector cache
# TODO auto-gcv puts a 0 in the UI
# TODO test in all widgets whether or not loading external data updates params

class FTTCDataPanel(QWidget):
    """Panel for handling FTTC data loading and status display."""

    data_loaded = Signal(str)  # Emits data type that was loaded
    displacement_loaded = Signal(object)

    def __init__(self, data_manager, viewer):
        super().__init__()
        self.data_manager = data_manager
        self.viewer = viewer
        self.controller = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Create data input group
        data_group = QGroupBox("Input Data")
        data_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        group_layout = QVBoxLayout()

        # Displacement data row
        displacement_layout = QHBoxLayout()
        self.load_displacement_btn = QPushButton("Load Displacement Data")
        self.load_displacement_btn.setFixedWidth(150)
        self.load_displacement_btn.setFixedHeight(25)
        self.load_displacement_btn.setToolTip("Load displacement data from file")
        self.displacement_status = QLabel("Not loaded")
        self.displacement_status.setWordWrap(True)
        displacement_layout.addWidget(self.load_displacement_btn)
        displacement_layout.addWidget(self.displacement_status)
        group_layout.addLayout(displacement_layout)

        # Add description label for required data
        info_label = QLabel(
            "Required: Displacement field data."
        )
        info_label.setWordWrap(True)
        group_layout.addWidget(info_label)
        data_group.setLayout(group_layout)
        layout.addWidget(data_group)
        self.setLayout(layout)

    def set_controller(self, controller):
        """Set the controller and connect signals."""
        self.controller = controller
        if self.controller:
            self.load_displacement_btn.clicked.connect(
                lambda: self.controller.load_displacement_data()
            )

    def update_data_status(self):
        """Update status labels based on loaded data."""
        displacement_field = self.data_manager.displacement_results
        if displacement_field is not None:
            self.displacement_status.setText(
                f"Loaded: {displacement_field.displacement_field.shape}"
            )
        else:
            self.displacement_status.setText("Not loaded")

    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        self.load_displacement_btn.setEnabled(not frozen)


class FTTCParameterPanel(QWidget):
    """Panel for handling all FTTC parameter inputs."""

    parameter_changed = Signal(str, object)  # (param_name, value)
    parameters_reset = Signal()

    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_spins = {}
        self._setup_ui()
        self._connect_signals()

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Add parameter groups
        layout.addWidget(self._create_material_parameters())
        layout.addWidget(self._create_regularization_parameters())
        layout.addWidget(self._create_visualization_parameters())

        # Add reset button
        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.setToolTip("Reset all FTTC parameters to their default values")
        layout.addWidget(self.reset_btn)

        self.setLayout(layout)
        self._sync_widget_with_parameters()

    def _sync_parameter(self, param_name: str, value: Any):
        """Sync a single parameter from parameter manager."""
        if param_name in self.parameter_spins:
            if param_name == 'young_modulus':
                value = value / 1000  # Convert Pa to kPa
            elif param_name == 'regularization':
                value = np.log10(value)  # Convert to log10 for display
            self._safe_set_value(self.parameter_spins[param_name], value)
        elif param_name == 'auto_gcv':
            self.auto_gcv_checkbox.setChecked(bool(value))

    def _on_value_changed(self, param_name: str, value: object):
        """Handle parameter value changes."""
        if param_name == 'young_modulus':
            value = value * 1000  # Convert kPa to Pa
        elif param_name == 'regularization':
            value = 10 ** value  # Convert from log10 to actual value

        # Update parameter manager
        self.parameter_manager.set_parameter(param_name, value)
        # Emit our own signal
        self.parameter_changed.emit(param_name, value)

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager."""
        self._block_widgets(True)
        try:
            for name, spin in self.parameter_spins.items():
                value = self.parameter_manager.get_parameter(name)
                if value is not None:
                    if name == 'young_modulus':
                        # Convert Pa to kPa for display
                        value = value / 1000
                    elif name == 'regularization':
                        # Convert to log10 for display
                        value = np.log10(value)
                    elif name == 'gel_height':
                        # Handle infinity case
                        if value == float('inf'):
                            value = 0  # Will display as "∞"
                    self._safe_set_value(spin, value)

            # Sync GCV checkbox
            auto_gcv = self.parameter_manager.get_parameter('auto_gcv')
            self.auto_gcv_checkbox.setChecked(bool(auto_gcv))
            self.parameter_spins['regularization'].setEnabled(not auto_gcv)
            self.gcv_button.setEnabled(not auto_gcv)

        finally:
            self._block_widgets(False)

    def _create_material_parameters(self) -> QGroupBox:
        """Create material parameter group."""
        group = QGroupBox("Material Parameters")
        layout = QVBoxLayout()

        # Define parameters with tooltips
        params = [
            ("young_modulus", "Young's Modulus (kPa):", 0.1, 1000.0, 0.1,
             "Elastic modulus of the gel substrate in kilopascals (kPa)"),
            ("poisson_ratio_substrate", "Poisson Ratio:", 0, 0.5, 0.01,
             "Poisson's ratio of the gel substrate (typically 0.45-0.5 for hydrogels)"),
            ("gel_height", "Gel Height (μm):", 0, 1000, 10,
             "Thickness of the gel substrate in micrometers. Set to 0 for infinite thickness"),
            ("lanczos_exp", "Lanczos Exponent:", 0, 5, 1,
             "Exponent for Lanczos interpolation. Higher values increase smoothing")
        ]

        for name, label_text, min_val, max_val, step, tooltip in params:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            row.addWidget(label)

            if isinstance(step, int):
                spin = QSpinBox()
                spin.setFixedWidth(135)

            else:
                spin = QDoubleSpinBox()
                spin.setFixedWidth(135)
                spin.setDecimals(2)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setToolTip(tooltip)
            if name == "gel_height":
                spin.setSpecialValueText("∞")

            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_regularization_parameters(self) -> QGroupBox:
        """Create regularization parameter group."""
        group = QGroupBox("Regularization Parameters")
        layout = QVBoxLayout()

        # Regularization value spinbox
        reg_layout = QHBoxLayout()
        reg_label = QLabel("Parameter (10^x):")
        reg_layout.addWidget(reg_label)

        reg_spin = QDoubleSpinBox()
        reg_spin.setFixedWidth(135)
        reg_spin.setRange(-21, 0)
        reg_spin.setValue(-4)
        reg_spin.setSingleStep(0.5)
        reg_spin.setDecimals(1)
        reg_spin.setToolTip(
            "Tikhonov regularization parameter as a power of 10.\n"
            "Lower values give more detailed but potentially noisier results"
        )
        self.parameter_spins['regularization'] = reg_spin
        reg_layout.addWidget(reg_spin)
        layout.addLayout(reg_layout)

        # Auto GCV checkbox and button
        gcv_layout = QHBoxLayout()
        self.auto_gcv_checkbox = QCheckBox("Auto-GCV per frame")
        self.auto_gcv_checkbox.setToolTip(
            "Automatically optimize regularization parameter for each frame\n"
            "using Generalized Cross-Validation"
        )
        gcv_layout.addWidget(self.auto_gcv_checkbox)

        self.gcv_button = QPushButton("Auto-select (GCV)")
        self.gcv_button.setToolTip(
            "Calculate optimal regularization parameter for current frame\n"
            "using Generalized Cross-Validation"
        )
        self.gcv_button.setFixedWidth(135)

        gcv_layout.addWidget(self.gcv_button)
        layout.addLayout(gcv_layout)

        group.setLayout(layout)
        return group

    def _create_visualization_parameters(self) -> QGroupBox:
        """Create visualization parameter group."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        params = [
            ("force_vector_stride", "Vector Stride:", 1, 100, 1,
             "Display every nth vector. Higher values show fewer vectors but improve clarity"),
            ("force_arrow_scale", "Arrow Scale:", 0.1, 10.0, 0.1,
             "Scale factor for force vectors. Adjust to make vectors more visible"),
            ("f_max", "Max Force (Pa):", 0.1, 10000.0, 1.0,
             "Maximum force value for color scaling")
        ]

        for name, label_text, min_val, max_val, step, tooltip in params:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            row.addWidget(label)

            if isinstance(step, int):
                spin = QSpinBox()
                spin.setFixedWidth(135)
            else:
                spin = QDoubleSpinBox()
                spin.setFixedWidth(135)
                if name == "force_arrow_scale":
                    spin.setDecimals(1)
                else:
                    spin.setDecimals(1)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setToolTip(tooltip)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect parameter spinboxes
        for name, spin in self.parameter_spins.items():
            spin.valueChanged.connect(
                lambda value, n=name: self._on_value_changed(n, value)
            )

        # Connect GCV controls
        self.auto_gcv_checkbox.stateChanged.connect(self._on_auto_gcv_changed)
        self.gcv_button.clicked.connect(self._on_gcv_clicked)

        # Connect reset button
        self.reset_btn.clicked.connect(self._reset_parameters)

    def _on_auto_gcv_changed(self, state):
        """Handle auto GCV checkbox state changes."""
        is_checked = state == Qt.Checked
        self.parameter_spins['regularization'].setEnabled(not is_checked)
        self.gcv_button.setEnabled(not is_checked)
        self.parameter_manager.set_parameter('auto_gcv', is_checked)

    def _on_gcv_clicked(self):
        """Handle GCV button clicks."""
        if self.controller:
            self.controller.calculate_optimal_regularization()

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        if category == ParameterCategory.FORCE:
            self._sync_widget_with_parameters()

    def _reset_parameters(self):
        """Reset parameters to defaults."""
        self.parameter_manager.reset_force_parameters()
        self.parameters_reset.emit()

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

    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        for spin in self.parameter_spins.values():
            spin.setEnabled(not frozen)
        self.auto_gcv_checkbox.setEnabled(not frozen)
        self.gcv_button.setEnabled(not frozen and not self.auto_gcv_checkbox.isChecked())
        self.reset_btn.setEnabled(not frozen)

    def _block_widgets(self, block: bool):
        """Block or unblock all widget signals."""
        for widget in self.parameter_spins.values():
            widget.blockSignals(block)
        self.auto_gcv_checkbox.blockSignals(block)
        self.gcv_button.blockSignals(block)

    def set_controller(self, controller):
        """Set the controller reference."""
        self.controller = controller


class FTTCActionPanel(QWidget):
    """Panel for FTTC analysis actions."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Create button pairs in rows
        # Row 1: Preview and Calculate buttons
        row1_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.preview_btn.setToolTip(
            "Calculate and visualize forces for the current frame only"
        )
        self.calculate_btn = QPushButton("Calculate Forces")
        self.calculate_btn.setToolTip(
            "Calculate forces for all frames in the dataset"
        )
        row1_layout.addWidget(self.preview_btn)
        row1_layout.addWidget(self.calculate_btn)
        layout.addLayout(row1_layout)

        # Row 2: Save and Load buttons
        row2_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Forces")
        self.save_btn.setToolTip(
            "Save force calculation results to file"
        )
        self.load_btn = QPushButton("Load Forces")
        self.load_btn.setToolTip(
            "Load previously saved force calculation results"
        )
        row2_layout.addWidget(self.save_btn)
        row2_layout.addWidget(self.load_btn)
        layout.addLayout(row2_layout)

        # Cancel button in its own row
        self.cancel_btn = QPushButton("Cancel Operation")
        self.cancel_btn.setToolTip(
            "Cancel the current operation"
        )
        layout.addWidget(self.cancel_btn)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect action buttons to controller methods."""
        self.preview_btn.clicked.connect(self.controller.preview_force)
        self.calculate_btn.clicked.connect(self.controller.calculate_forces)
        self.save_btn.clicked.connect(self.controller.save_results)
        self.load_btn.clicked.connect(self.controller.load_results)
        self.cancel_btn.clicked.connect(self.controller.cancel_operation)

    def freeze_ui(self, freeze: bool = True):
        """Disable/enable action buttons during processing."""
        # Disable all buttons except cancel during processing
        action_buttons = [
            self.preview_btn,
            self.calculate_btn,
            self.save_btn,
            self.load_btn
        ]
        for btn in action_buttons:
            btn.setEnabled(not freeze)

        # Cancel button is enabled only during processing
        self.cancel_btn.setEnabled(freeze)

    def update_button_states(self,
                             has_displacement: bool = False,
                             has_results: bool = False):
        """Update button states based on data availability."""
        # Analysis buttons need displacement data
        self.preview_btn.setEnabled(has_displacement)
        self.calculate_btn.setEnabled(has_displacement)

        # Save button needs results
        self.save_btn.setEnabled(has_results)

        # Load button is always enabled
        self.load_btn.setEnabled(True)

        # Cancel button is disabled by default (enabled during processing)
        self.cancel_btn.setEnabled(False)


class FTTCController(QObject):
    """Controller coordinating FTTC analysis components."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(FTTCResult)
    analysis_failed = Signal(str)
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    def __init__(self, viewer, service, data_manager, parameter_manager,
                 visualization_manager, data_panel):
        super().__init__()
        self.viewer = viewer
        self.service = service
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.data_panel = data_panel
        self.active_workers = []

        # Initialize panel attributes
        self.parameter_panel = None
        self.action_panel = None

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)

    def set_panels(self, parameter_panel, action_panel):
        """Set the parameter and action panels."""
        self.parameter_panel = parameter_panel
        self.action_panel = action_panel
        self.parameter_panel.set_controller(self)

    def preview_force(self):
        """Preview force calculation for current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Calculating force preview...")

            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]

            # Create and configure service
            params = self.parameter_manager.get_fttc_parameters()
            self.service.update_parameters(params)

            # Create worker for processing
            worker = self._create_preview_worker(displacement_field, params)
            worker.returned.connect(self._handle_preview_results)
            worker.errored.connect(self._handle_error)
            worker.finished.connect(lambda: self.unfreeze_ui())

            self.active_workers.append(worker)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def calculate_forces(self):
        """Calculate forces for all frames."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Starting force calculation...")

            # Get parameters and update service
            params = self.parameter_manager.get_fttc_parameters()
            self.service.update_parameters(params)

            # Create worker for processing
            displacement_field = self.data_manager.displacement_results.displacement_field
            worker = self._create_force_worker(displacement_field, params)

            self.active_workers.append(worker)
            self.analysis_started.emit()

            worker.yielded.connect(self._handle_progress)
            worker.returned.connect(self._handle_analysis_results)
            worker.errored.connect(self._handle_error)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def calculate_optimal_regularization(self):
        """Calculate optimal regularization parameter using GCV."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Calculating optimal regularization...")

            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            displacement_field = self.data_manager.displacement_results.displacement_field[current_frame]

            # Create worker for GCV calculation
            worker = self._create_gcv_worker(displacement_field)

            worker.returned.connect(self._handle_gcv_results)
            worker.errored.connect(self._handle_error)
            worker.finished.connect(lambda: self.unfreeze_ui())

            self.active_workers.append(worker)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def load_displacement_data(self):
        """Load displacement data from active layer or file."""
        try:
            # If no data in manager, try to load from file
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Displacement Data",
                str(Path.home()),
                "NumPy Files (*.npy)"
            )

            if file_path:
                displacement_data = np.load(file_path, allow_pickle=True).item()
                self.data_manager.set_displacement_results(displacement_data)

                if displacement_data is not None:
                    self.data_updated.emit('displacement')
                    self.progress_updated.emit(
                        100,
                        f"Displacement data loaded: {displacement_data.displacement_field.shape}"
                    )
            else:
                self.progress_updated.emit(0, "No displacement data loaded")

        except Exception as e:
            self._handle_error(f"Failed to load displacement data: {str(e)}")

    def save_results(self):
        """Save force calculation results."""
        try:
            if self.data_manager.force_results is None:
                raise ValueError("No force results to save")

            save_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Force Results",
                str(Path.home()),
                "NumPy Files (*.npy)"
            )

            if save_path:
                results = self.data_manager.force_results
                np.save(save_path, results)
                self.progress_updated.emit(100, f"Results saved to {save_path}")

        except Exception as e:
            self._handle_error(f"Failed to save results: {str(e)}")

    def load_results(self):
        """Load previously saved force results."""
        try:
            load_path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Force Results",
                str(Path.home()),
                "NumPy Files (*.npy)"
            )

            if load_path:
                # Load data
                results = np.load(load_path, allow_pickle=True).item()

                # Update parameters if they exist in the results
                if hasattr(results, 'parameters'):
                    # Block parameter change signals temporarily
                    if self.parameter_panel:
                        self.parameter_panel._block_widgets(True)
                    try:
                        # Update parameter manager with loaded parameters
                        params = results.parameters
                        for param_name, value in vars(params).items():
                            if param_name != '_sa_instance_state':  # Skip SQLAlchemy state
                                if param_name == 'young_modulus':
                                    # Convert Pa to kPa for UI display
                                    self.parameter_manager.set_parameter(param_name, value)
                                elif param_name == 'regularization':
                                    # Store actual value, UI will convert to log10
                                    self.parameter_manager.set_parameter(param_name, value)
                                else:
                                    self.parameter_manager.set_parameter(param_name, value)

                        # Sync UI with new parameters
                        if self.parameter_panel:
                            self.parameter_panel._sync_widget_with_parameters()
                    finally:
                        if self.parameter_panel:
                            self.parameter_panel._block_widgets(False)

                # Update data manager and visualization
                self.data_manager.set_force_results(results)
                self.visualization_manager.visualize_force_results()
                self.progress_updated.emit(100, f"Results and parameters loaded from {load_path}")
                self.analysis_completed.emit(results)

        except Exception as e:
            self._handle_error(f"Failed to load results: {str(e)}")

    def _sync_parameters_with_results(self, result):
        """Sync parameters from loaded results."""
        if not hasattr(result, 'parameters'):
            return

        params = result.parameters
        for param_name, value in vars(params).items():
            if param_name != '_sa_instance_state':  # Skip SQLAlchemy state
                if param_name == 'young_modulus':
                    # Store in Pa, UI will convert to kPa
                    self.parameter_manager.set_parameter(param_name, value)
                elif param_name == 'regularization':
                    # Store actual value, UI will convert to log10
                    self.parameter_manager.set_parameter(param_name, value)
                elif param_name == 'gel_height':
                    # Handle infinity case
                    if value == 0:
                        value = float('inf')
                    self.parameter_manager.set_parameter(param_name, value)
                else:
                    self.parameter_manager.set_parameter(param_name, value)
    def cancel_operation(self):
        """Cancel any running operations."""
        for worker in self.active_workers:
            try:
                worker.quit()
                worker.wait()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()
        self.progress_updated.emit(0, "Operation cancelled")
        self.unfreeze_ui()

    @thread_worker
    def _create_preview_worker(self, displacement_field, params):
        """Create worker for preview calculation."""
        try:
            result = self.service.calculate_forces(displacement_field[np.newaxis, ...])
            # Process generator to get result
            try:
                while True:
                    next(result)
            except StopIteration as e:
                return e.value

        except Exception as e:
            raise ValueError(f"Preview calculation failed: {str(e)}")

    @thread_worker
    def _create_force_worker(self, displacement_field, params):
        """Create worker for full force calculation."""
        try:
            # Get the generator from the service
            force_generator = self.service.calculate_forces(displacement_field)

            # Process all frames through the generator
            try:
                while True:
                    force_field, frame, total = next(force_generator)
                    yield {
                        'progress': (frame + 1) / total * 100,
                        'message': f"Processing frame {frame + 1}/{total}"
                    }
            except StopIteration as e:
                # Return the final result from the generator
                return e.value

        except Exception as e:
            raise ValueError(f"Force calculation failed: {str(e)}")

    @thread_worker
    def _create_gcv_worker(self, displacement_field):
        """Create worker for GCV calculation."""
        try:
            optimal_reg = self.service.find_optimal_regularization(displacement_field)
            return optimal_reg
        except Exception as e:
            raise ValueError(f"GCV calculation failed: {str(e)}")

    def _handle_preview_results(self, result: FTTCResult):
        """Handle preview calculation results."""
        try:
            if result is None:
                raise RuntimeError("Preview calculation failed to produce results")

            # Update visualization for preview
            self.visualization_manager.visualize_force_preview(
                result.force_field[0],
                result.parameters.f_max,
                result.parameters.force_vector_stride,
                result.parameters.force_arrow_scale,
                downscale_factor=result.parameters.downscale_factor
            )

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the force layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Force Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Force Magnitude':
                    magnitude_layer = layer
                    layer.visible = True
                else:
                    layer.visible = False

            # Move layers to desired positions if they exist
            if magnitude_layer is not None:
                current_index = self.viewer.layers.index(magnitude_layer)
                # Move magnitude layer to second from top (-2)
                if current_index != -2:
                    self.viewer.layers.move(current_index, -2)

            if vector_layer is not None:
                current_index = self.viewer.layers.index(vector_layer)
                # Move vector layer to top (-1)
                if current_index != -1:
                    self.viewer.layers.move(current_index, -1)

            # Calculate and show statistics
            magnitude = np.sqrt(np.sum(result.force_field[0] ** 2, axis=-1))
            self.analysis_completed.emit(result)
            self.progress_updated.emit(
                100,
                f"Preview statistics:\n"
                f"Max force: {np.max(magnitude):.2f} Pa\n"
                f"Mean force: {np.mean(magnitude):.2f} Pa"
            )

        except Exception as e:
            self._handle_error(str(e))

    def _handle_analysis_results(self, result: FTTCResult):
        """Handle complete analysis results."""
        try:
            if result is None:
                raise RuntimeError("Analysis failed to produce results")

            # Update data manager
            self.data_manager.set_force_results(result)

            # Update visualization
            self.visualization_manager.visualize_force_results()

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the force layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Force Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Force Magnitude':
                    magnitude_layer = layer
                    layer.visible = True
                else:
                    layer.visible = False

            # Move layers to desired positions if they exist
            if magnitude_layer is not None:
                current_index = self.viewer.layers.index(magnitude_layer)
                # Move magnitude layer to second from top (-2)
                if current_index != -2:
                    self.viewer.layers.move(current_index, -2)

            if vector_layer is not None:
                current_index = self.viewer.layers.index(vector_layer)
                # Move vector layer to top (-1)
                if current_index != -1:
                    self.viewer.layers.move(current_index, -1)

            self.progress_updated.emit(100, "Analysis completed successfully")
            self.analysis_completed.emit(result)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.active_workers.clear()
            self.unfreeze_ui()

    def _handle_gcv_results(self, regularization: float):
        """Handle GCV calculation results."""
        if regularization is not None:
            # Update parameter in log scale
            log_reg = np.log10(regularization)
            self.parameter_manager.set_parameter('regularization', regularization)
            self.progress_updated.emit(
                100,
                f"Optimal regularization parameter: {regularization:.2e}"
            )
        else:
            self._handle_error("GCV calculation failed")

    def _handle_progress(self, progress_info: dict):
        """Handle progress updates."""
        self.progress_updated.emit(
            progress_info['progress'],
            progress_info['message']
        )

    def _handle_error(self, error_msg: str):
        """Handle errors during processing."""
        self.progress_updated.emit(0, f"Error: {error_msg}")
        self.analysis_failed.emit(error_msg)
        QMessageBox.critical(None, "Error", error_msg)

    def _validate_prerequisites(self) -> bool:
        """Check if required data and parameters are available."""
        if self.data_manager.displacement_results is None:
            QMessageBox.warning(None, "Warning", "No displacement data loaded")
            return False
        return True

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes from parameter manager."""
        pass

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        pass

    def freeze_ui(self):
        """Disable all interactive UI elements."""
        if self.data_panel:
            self.data_panel.freeze_ui(True)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(True)
        if self.action_panel:
            self.action_panel.freeze_ui(True)
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Re-enable UI elements and refresh state."""
        if self.data_panel:
            self.data_panel.freeze_ui(False)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(False)
        if self.action_panel:
            self.action_panel.freeze_ui(False)
        self.ui_frozen.emit(False)


class FTTCWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method."""

    force_calculated = Signal(FTTCResult)

    def __init__(
            self,
            viewer: Viewer,
            data_manager: DataManager,
            parameter_manager: ParameterManager,
            visualization_manager: VisualizationManager
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Store managers and create service
        self.parameter_manager = parameter_manager
        self.service = FTTCService(parameter_manager.get_fttc_parameters())
        self.colorbar_manager = ColorbarManager()

        # Initialize panels
        self.data_panel = FTTCDataPanel(data_manager, viewer)
        self.parameter_panel = FTTCParameterPanel(parameter_manager)

        # Initialize controller
        self.controller = FTTCController(
            viewer=viewer,
            service=self.service,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
            data_panel=self.data_panel
        )

        self.data_panel.set_controller(self.controller)

        # Initialize action panel with controller
        self.action_panel = FTTCActionPanel(self.controller)

        # Connect controller
        self.controller.set_panels(self.parameter_panel, self.action_panel)

        # Set up UI
        self._setup_ui()

        # Connect signals
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

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
            colormap_name='inferno',
            label="Force (Pa)",
            clim=(0, self.parameter_manager.get_parameter('f_max')),
            colorbar_manager=self.colorbar_manager
        )
        layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        container.setLayout(layout)
        return container

    def _create_content_container(self) -> QWidget:
        """Create the main content container."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add panels
        layout.addWidget(self.data_panel)
        layout.addItem(QSpacerItem(0, -12, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self.parameter_panel)
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self.action_panel)
        layout.addItem(QSpacerItem(0, -10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll

    def _create_status_frame(self) -> QFrame:
        """Create the status display frame."""
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
        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)

        # Connect parameter panel signals
        self.parameter_panel.parameters_reset.connect(self._on_parameters_reset)

        # Connect to layer selection changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update data panel
        self.data_panel.update_data_status()

        # Update action panel
        has_displacement = self.data_manager.displacement_results is not None
        has_results = self.data_manager.force_results is not None
        self.action_panel.update_button_states(
            has_displacement=has_displacement,
            has_results=has_results
        )

    def _on_parameters_reset(self):
        """Handle parameter reset."""
        self._update_status(0, "Force parameters reset to default values")

    def _on_analysis_completed(self, results: FTTCResult):
        """Handle completed analysis."""
        # Update colorbar
        if hasattr(results, 'parameters'):
            self.colorbar_manager.update_limits(0, results.parameters.f_max)
        # Emit results
        self.force_calculated.emit(results)

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_status(0, f"Error: {error_msg}")

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.force_results is not None:
            self.visualization_manager.update_force_frame(
                self.viewer.dims.current_step[0]
            )

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()
