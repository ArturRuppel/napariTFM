from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal, Qt, QObject
from qtpy.QtWidgets import (
    QGroupBox, QLabel, QCheckBox, QSizePolicy, QFrame, QScrollArea, QSpinBox, QDoubleSpinBox, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QMessageBox, QProgressBar
)

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.data_manager_old import DataManager
from napariTFM.parameter_manager_old import ParameterManager
from napariTFM.services.fttc_service import FTTCService, FTTCParameters
from napariTFM.visualization_manager import VisualizationManager


# TODO colorbar doesn't update
# TODO button enabling/disabling logic is not on point
# TODO Live update of force preview and results with vis params only works partially
class FTTCParameterPanel(QWidget):
    """Panel for handling all FTTC parameter inputs."""

    parameter_changed = Signal()
    parameter_value_changed = Signal(str, object)


    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_widgets = {}
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create parameter groups
        layout.addWidget(self._create_material_parameters())
        layout.addWidget(self._create_regularization_parameters())
        layout.addWidget(self._create_visualization_parameters())

        self.setLayout(layout)

    def _create_material_parameters(self) -> QGroupBox:
        """Create material parameter group."""
        group = QGroupBox("Material Parameters")
        layout = QVBoxLayout()

        # Note: young_modulus is stored in Pa but displayed in kPa
        params = [
            ("young_modulus", "Young's Modulus (kPa):", 0.1, 1000, 0.1, 10),  # Display in kPa
            ("poisson_ratio_substrate", "Poisson Ratio:", 0, 0.5, 0.01, 0.49),
            ("gel_height", "Gel Height (μm):", 0, 1000, 10, 0),  # Special handling for 0/None
            ("lanczos_exp", "Lanczos Exponent:", 0, 5, 1, 1)
        ]

        for name, label, min_val, max_val, step, default in params:
            widget = self._create_parameter_widget(name, label, min_val, max_val, step, default)
            layout.addLayout(widget)

            # Special handling for gel height
            if name == "gel_height":
                spin_widget = self.parameter_widgets[name]
                spin_widget.setSpecialValueText("∞")  # Show infinity symbol when value is 0

        group.setLayout(layout)
        return group

    def _on_value_changed(self, param_name: str, value: Any):
        """Handle parameter value changes."""
        # Convert kPa to Pa before storing in parameter manager
        if param_name == "young_modulus":
            value = value * 1000  # Convert kPa to Pa
        elif param_name == "regularization":
            value = 10 ** value  # Convert from log10 scale
        elif param_name == "gel_height":
            value = None if value == 0 else value  # Convert 0 to None for infinite height

        self.parameter_manager.set_value(param_name, value)
        self.parameter_value_changed.emit(param_name, value)
        self.parameter_changed.emit()

        # Update visualization if f_max changes
        if param_name in ["f_max", "force_vector_stride", "force_arrow_scale"] and hasattr(self, 'controller'):
            self.controller.update_force_visualization()

    def _update_widget_value(self, param_name: str, value: any):
        """Update widget when parameter changes externally."""
        if param_name in self.parameter_widgets:
            widget = self.parameter_widgets[param_name]
            widget.blockSignals(True)

            if param_name == "young_modulus":
                # Convert Pa to kPa for display
                widget.setValue(value / 1000)
            elif param_name == "regularization":
                widget.setValue(np.log10(value))
            elif param_name == "gel_height":
                # Convert None to 0 for display
                widget.setValue(0 if value is None else value)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)

            widget.blockSignals(False)

    def freeze_ui(self, freeze=True):
        """Disable/enable interactive elements in parameter panel"""
        for widget in self.parameter_widgets.values():
            widget.setEnabled(not freeze)

    def _create_regularization_parameters(self) -> QGroupBox:
        """Create regularization parameter group."""
        group = QGroupBox("Regularization Parameters")
        layout = QVBoxLayout()

        # Regularization value control
        reg_layout = QHBoxLayout()
        reg_layout.addWidget(QLabel("Parameter (10^x):"))
        reg_spin = QDoubleSpinBox()
        reg_spin.setRange(-21, 0)
        reg_spin.setValue(-4)
        reg_spin.setSingleStep(0.5)
        reg_spin.setDecimals(1)
        self.parameter_widgets["regularization"] = reg_spin
        reg_layout.addWidget(reg_spin)
        layout.addLayout(reg_layout)

        # GCV controls
        gcv_layout = QHBoxLayout()
        auto_gcv = QCheckBox("Auto-GCV per frame")
        self.parameter_widgets["auto_gcv"] = auto_gcv
        gcv_layout.addWidget(auto_gcv)

        gcv_button = QPushButton("Auto-select (GCV)")
        self.parameter_widgets["gcv_button"] = gcv_button
        gcv_layout.addWidget(gcv_button)
        layout.addLayout(gcv_layout)

        group.setLayout(layout)
        return group

    def _create_visualization_parameters(self) -> QGroupBox:
        """Create visualization parameter group."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        params = [
            ("force_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("force_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("f_max", "Max Force (Pa):", 0.1, 10000.0, 1, 500.0)  # Changed default to 500
        ]

        for name, label, min_val, max_val, step, default in params:
            widget = self._create_parameter_widget(name, label, min_val, max_val, step, default)
            layout.addLayout(widget)

        group.setLayout(layout)
        return group

    def _create_parameter_widget(self, name: str, label: str,
                                 min_val: float, max_val: float,
                                 step: float, default: float) -> QHBoxLayout:
        """Create a parameter widget with label and input."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))

        if isinstance(step, int):
            spin = QSpinBox()
        else:
            spin = QDoubleSpinBox()
            if name == "arrow_scale":
                spin.setDecimals(2)
            else:
                spin.setDecimals(1)

        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(default)

        self.parameter_widgets[name] = spin
        layout.addWidget(spin)

        return layout

    def set_controller(self, controller):
        """Set the controller reference and connect GCV button."""
        self.controller = controller
        self.parameter_widgets["gcv_button"].clicked.connect(controller.auto_select_gcv)

    def _connect_signals(self):
        """Connect widget signals to parameter manager."""
        for name, widget in self.parameter_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(
                    lambda value, n=name: self._on_value_changed(n, value)
                )
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(
                    lambda state, n=name: self._on_value_changed(n, bool(state))
                )

        self.parameter_manager.parameter_changed.connect(self._update_widget_value)
        if hasattr(self, 'controller'):
            self.parameter_widgets["gcv_button"].clicked.connect(self.controller.auto_select_gcv)


class FTTCDataPanel(QWidget):
    """Panel for handling data loading and status display."""

    data_loaded = Signal(str)  # Emits data type that was loaded
    displacement_data_loaded = Signal(object)
    data_load_failed = Signal(str)

    def __init__(self, data_manager, viewer):
        super().__init__()
        self.data_manager = data_manager
        self.viewer = viewer
        self.controller = None
        self._setup_ui()

    def freeze_ui(self, freeze=True):
        """Disable/enable interactive elements in data panel"""
        self.load_displacement_btn.setEnabled(not freeze)

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create data input group
        data_group = QGroupBox("Data Input")
        group_layout = QVBoxLayout()

        # Displacement data row
        displacement_layout = QHBoxLayout()
        self.load_displacement_btn = QPushButton("Load Displacements")
        self.displacement_status = QLabel("Not loaded")
        displacement_layout.addWidget(self.load_displacement_btn)
        displacement_layout.addWidget(self.displacement_status)
        group_layout.addLayout(displacement_layout)

        data_group.setLayout(group_layout)
        layout.addWidget(data_group)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect UI signals to controller methods."""
        self.load_displacement_btn.clicked.connect(self._load_displacement_data)

    def set_controller(self, controller):
        """Set the controller and connect signals."""
        self.controller = controller
        self._connect_signals()

    def _load_displacement_data(self):
        """Load displacement data from files."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Displacement Data File", "", "NumPy Files (*.npy)"
            )

            if file_path:
                displacement_data = np.load(file_path, allow_pickle=True).item()

                if 'flows' not in displacement_data:
                    raise ValueError("Invalid displacement data format")

                flows = displacement_data['flows']
                parameters = displacement_data.get('parameters', {})

                self.data_manager.set_displacement_results(flows, parameters)
                self.displacement_status.setText(f"Loaded: {flows.shape}")
                self.displacement_data_loaded.emit(displacement_data)
                self.data_loaded.emit('displacement')

        except Exception as e:
            error_msg = f"Failed to load displacement data: {str(e)}"
            self.displacement_status.setText("Error loading")
            self.data_load_failed.emit(error_msg)
            QMessageBox.critical(self, "Error", error_msg)


class FTTCActionPanel(QWidget):
    """Panel for analysis actions and progress display."""

    def __init__(self, fttc_controller):
        super().__init__()
        self.controller = fttc_controller
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Button grid
        button_layout = QVBoxLayout()

        # Row 1: Preview and Calculate
        row1_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.calculate_btn = QPushButton("Calculate Forces")
        row1_layout.addWidget(self.preview_btn)
        row1_layout.addWidget(self.calculate_btn)
        button_layout.addLayout(row1_layout)

        # Row 2: Save and Load
        row2_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Force Data")
        self.load_force_btn = QPushButton("Load Force Data")
        row2_layout.addWidget(self.save_btn)
        row2_layout.addWidget(self.load_force_btn)
        button_layout.addLayout(row2_layout)

        layout.addLayout(button_layout)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel Operation")
        layout.addWidget(self.cancel_btn)

        self.setLayout(layout)

    def freeze_ui(self, freeze=True):
        """Disable/enable action buttons (keep cancel enabled)"""
        buttons = [
            self.preview_btn, self.calculate_btn,
            self.save_btn, self.load_force_btn
        ]
        for btn in buttons:
            btn.setEnabled(not freeze)
        self.cancel_btn.setEnabled(True)

    def update_button_states(self, displacement_data: bool = False,
                             force_data: bool = False):
        """Update button states based on current data availability."""
        self.preview_btn.setEnabled(displacement_data)
        self.calculate_btn.setEnabled(displacement_data)
        self.save_btn.setEnabled(force_data)

    def _connect_signals(self):
        """Connect action panel buttons to controller methods."""
        self.preview_btn.clicked.connect(self.controller.preview_force)
        self.calculate_btn.clicked.connect(self.controller.start_analysis)
        self.save_btn.clicked.connect(self.controller.save_results)
        self.load_force_btn.clicked.connect(self.controller.load_force_data)
        self.cancel_btn.clicked.connect(self.controller.cancel_all_operations)


class FTTCController(QObject):
    """Controller coordinating FTTC widget components."""

    data_updated = Signal(str)
    progress_updated = Signal(int, str)
    analysis_started = Signal()
    analysis_completed = Signal(object)
    analysis_failed = Signal(str)
    ui_frozen = Signal(bool)

    def __init__(self, viewer: Viewer, service: FTTCService,
                 data_manager: DataManager, parameter_manager: ParameterManager,
                 visualization_manager: VisualizationManager, data_panel: FTTCDataPanel):
        super().__init__()
        self.viewer = viewer
        self.service = service
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.data_panel = data_panel
        self.active_workers = []

        # Initialize panel attributes as None
        self.parameter_panel = None
        self.action_panel = None


    def set_panels(self, parameter_panel: 'FTTCParameterPanel',
                   action_panel: 'FTTCActionPanel'):
        """Set the parameter and action panels after initialization."""
        self.parameter_panel = parameter_panel
        self.action_panel = action_panel

    def freeze_ui(self):
        """Disable all interactive UI elements"""
        if self.data_panel:
            self.data_panel.freeze_ui(True)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(True)
        if self.action_panel:
            self.action_panel.freeze_ui(True)
        self.ui_frozen.emit(True)

    def unfreeze_ui(self):
        """Re-enable UI elements and refresh state"""
        if self.data_panel:
            self.data_panel.freeze_ui(False)
        if self.parameter_panel:
            self.parameter_panel.freeze_ui(False)
        if self.action_panel:
            self.action_panel.freeze_ui(False)
        self.ui_frozen.emit(False)

    def start_analysis(self):
        """Start the force calculation process."""
        try:
            if not self._validate_prerequisites():
                return

            self.analysis_started.emit()
            self._update_progress(0, "Starting force calculation...")

            # Get current parameters
            params = self._get_current_parameters()

            # Initialize service
            self.service.initialize_calculator(params)

            # Get displacement data
            displacement_field = self.data_manager.displacement_field
            pixel_size = self.data_manager.displacement_params['pixel_size']
            downscale_factor = self.data_manager.displacement_params.get('downscale_factor', 1)

            @thread_worker
            def force_worker():
                # Start force calculation generator
                force_generator = self.service.calculate_force_stack(
                    displacement_field=displacement_field,
                    pixel_size=pixel_size,
                    downscale_factor=downscale_factor,
                    regularization=None if params.auto_gcv else params.regularization,
                    use_gcv=params.auto_gcv
                )

                try:
                    while True:
                        progress_info, current_frame, total_frames = next(force_generator)
                        progress = int((current_frame) / total_frames * 100)

                        # Format progress message
                        message = (
                            f"Processing frame {current_frame}/{total_frames}\n"
                            f"Mean force: {progress_info['mean_force']:.2f} Pa\n"
                            f"Max force: {progress_info['max_force']:.2f} Pa"
                        )

                        yield progress, message, progress_info

                except StopIteration as e:
                    return e.value

            # Create and configure worker
            worker = force_worker()
            worker.started.connect(self.freeze_ui)
            worker.yielded.connect(self._handle_force_progress)
            worker.returned.connect(self._handle_force_results)
            worker.errored.connect(self._handle_error)
            worker.finished.connect(self.unfreeze_ui)

            self.active_workers.append(worker)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))

    def preview_force(self):
        """Preview force calculation for current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self._update_progress(0, "Calculating forces for preview...")

            # Get current parameters and initialize service
            params = self._get_current_parameters()
            self.service.initialize_calculator(params)

            # Get current frame data
            current_frame = self.viewer.dims.current_step[0]
            displacement_field = self.data_manager.displacement_field[current_frame]
            pixel_size = self.data_manager.displacement_params['pixel_size']
            downscale_factor = self.data_manager.displacement_params.get('downscale_factor', 1)

            @thread_worker
            def preview_worker():
                tx, ty = self.service.calculate_forces(
                    displacement_field=displacement_field,
                    pixel_size=pixel_size,
                    downscale_factor=downscale_factor,
                    regularization=None if params.auto_gcv else params.regularization,
                    use_gcv=params.auto_gcv
                )
                return tx, ty, params  # Also return current parameters

            # Create and configure worker
            worker = preview_worker()
            worker.started.connect(self.freeze_ui)
            worker.returned.connect(self._handle_preview_results)
            worker.errored.connect(self._handle_error)
            worker.finished.connect(self.unfreeze_ui)

            self.active_workers.append(worker)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))


    def auto_select_gcv(self):
        """Calculate optimal regularization parameter for current frame using GCV."""
        try:
            if not self._validate_prerequisites():
                return

            self._update_progress(0, "Calculating optimal regularization parameter...")
            self.freeze_ui()

            # Get current frame data
            current_frame = self.viewer.dims.current_step[0]
            displacement_field = self.data_manager.displacement_field[current_frame]
            pixel_size = self.data_manager.displacement_params['pixel_size']
            downscale_factor = self.data_manager.displacement_params.get('downscale_factor', 1)

            @thread_worker
            def gcv_worker():
                # Initialize calculator with current parameters
                params = self._get_current_parameters()
                self.service.initialize_calculator(params)

                # Calculate optimal regularization using GCV
                shape = displacement_field.shape[:-1]
                pos = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy'))
                vec = np.array([displacement_field[..., 0], displacement_field[..., 1]])

                return self.service.find_optimal_regularization(
                    displacement_field=displacement_field,
                    pixel_size=pixel_size,
                    downscale_factor=downscale_factor
                )

            # Create and configure worker
            worker = gcv_worker()
            worker.returned.connect(self._handle_gcv_result)
            worker.errored.connect(self._handle_error)
            worker.finished.connect(self.unfreeze_ui)

            self.active_workers.append(worker)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def _handle_gcv_result(self, reg_param):
        """Handle the result from GCV calculation."""
        try:
            # Update parameter manager with new regularization value
            self.parameter_manager.set_value('regularization', reg_param)

            # Update status with the result
            self._update_progress(100, f"Optimal regularization parameter: {reg_param:.2e}")

        except Exception as e:
            self._handle_error(str(e))

    def _create_visualization_parameters(self) -> QGroupBox:
        """Create visualization parameter group."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        params = [
            ("force_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("force_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("f_max", "Max Force (Pa):", 0.1, 10000.0, 1, 500.0)  # Changed default to 500
        ]

        for name, label, min_val, max_val, step, default in params:
            widget = self._create_parameter_widget(name, label, min_val, max_val, step, default)
            layout.addLayout(widget)

        group.setLayout(layout)
        return group


    def _handle_preview_results(self, results):
        """Handle preview calculation results."""
        try:
            tx, ty, params = results  # Unpack the results including parameters

            # Get visualization parameters
            downscale_factor = self.data_manager.displacement_params.get('downscale_factor', 1)

            # Update visualization with current parameters
            self.visualization_manager.visualize_force_preview(
                tx, ty,
                f_max=params.f_max,
                vector_stride=params.force_vector_stride,
                arrow_scale=params.force_arrow_scale,
                downscale_factor=downscale_factor
            )

            # Calculate force magnitude for statistics
            force_magnitude = np.sqrt(tx ** 2 + ty ** 2)

            # Calculate and show statistics
            stats_message = (
                f"Preview statistics:\n"
                f"Max force: {np.max(force_magnitude):.2f} Pa\n"
                f"Mean force: {np.mean(force_magnitude):.2f} Pa\n"
                f"Median force: {np.median(force_magnitude):.2f} Pa"
            )
            self._update_progress(100, stats_message)

        except Exception as e:
            self._handle_error(str(e))

    def _handle_force_results(self, results):
        """Handle completed force calculation results."""
        try:
            # Get parameters for results packaging
            params = self._get_current_parameters()
            pixel_size = self.data_manager.displacement_params['pixel_size']
            downscale_factor = self.data_manager.displacement_params.get('downscale_factor', 1)

            # Create force field array
            force_field = np.stack([results['tx'], results['ty']], axis=-1)

            # Calculate force magnitude for colorbar
            force_magnitude = np.sqrt(results['tx'] ** 2 + results['ty'] ** 2)

            # Update colorbar with force magnitude range
            max_force = min(params.f_max, np.max(force_magnitude))  # Use the smaller of f_max or actual max
            if hasattr(self.visualization_manager, 'widget'):
                if hasattr(self.visualization_manager.widget, 'colorbar_manager'):
                    self.visualization_manager.widget.colorbar_manager.update_limits(0, max_force)

            # Package parameters
            force_params = {
                'young_modulus': params.young_modulus,
                'poisson_ratio_substrate': params.poisson_ratio_substrate,
                'gel_height': params.gel_height,
                'pixel_size': pixel_size,
                'frame_interval': params.frame_interval,
                'regularization': params.regularization,
                'lanczos_exp': params.lanczos_exp,
                'downscale_factor': downscale_factor,
                'visualization': {
                    'force_vector_stride': params.force_vector_stride,
                    'force_arrow_scale': params.force_arrow_scale,
                    'f_max': params.f_max
                }
            }

            # Update data manager
            self.data_manager.set_force_results(force_field, force_params)

            # Update visualization
            self.visualization_manager.visualize_force_results(
                results,
                downscale_factor=downscale_factor
            )

            self._update_progress(100, "Force calculation completed successfully")
            self.analysis_completed.emit(results)

        except Exception as e:
            self._handle_error(str(e))

    def _handle_force_progress(self, progress_data):
        """Handle progress updates during force calculation."""
        progress, message, _ = progress_data
        self._update_progress(progress, message)

    def save_results(self):
        """Save force calculation results to file."""
        try:
            if self.data_manager.force_field is None:
                raise ValueError("No force data to save")

            file_path, _ = QFileDialog.getSaveFileName(
                None, "Save Force Data", "", "NumPy Files (*.npy)"
            )

            if file_path:
                if not file_path.endswith('.npy'):
                    file_path += '.npy'

                force_results = {
                    'force_field': self.data_manager.force_field,
                    'parameters': self.data_manager.force_params
                }

                np.save(file_path, force_results)
                self._update_progress(100, f"Results saved to:\n{file_path}")

        except Exception as e:
            self._handle_error(str(e))

    def update_force_visualization(self):
        """Update force visualization with current parameters."""
        try:
            if not self.data_manager.force_field is None:
                # Get current parameters
                params = self._get_current_parameters()

                # Update visualization with new parameters
                results = {
                    'tx': self.data_manager.force_field[..., 0],
                    'ty': self.data_manager.force_field[..., 1],
                    'parameters': {
                        'visualization': {
                            'f_max': params.f_max,
                            'force_vector_stride': params.force_vector_stride,
                            'force_arrow_scale': params.force_arrow_scale
                        }
                    }
                }

                self.visualization_manager.visualize_force_results(
                    results,
                    downscale_factor=self.data_manager.force_params.get('downscale_factor', 1)
                )

                # Update colorbar
                if hasattr(self.visualization_manager, 'widget'):
                    if hasattr(self.visualization_manager.widget, 'colorbar_manager'):
                        self.visualization_manager.widget.colorbar_manager.update_limits(0, params.f_max)

        except Exception as e:
            self._handle_error(str(e))

    def load_force_data(self):
        """Load force data from file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                None, "Load Force Data", "", "NumPy Files (*.npy)"
            )

            if file_path:
                force_data = np.load(file_path, allow_pickle=True).item()
                force_field, parameters, warnings = self.service.process_force_data(force_data)

                # Show any warnings
                for warning in warnings:
                    QMessageBox.warning(None, "Warning", warning)

                # Update data manager
                self.data_manager.set_force_results(force_field, parameters)

                # Update visualization
                self.visualization_manager.visualize_force_results(
                    {'tx': force_field[..., 0], 'ty': force_field[..., 1], 'parameters': parameters},
                    downscale_factor=parameters.get('downscale_factor', 1)
                )

                self._update_progress(100, f"Force data loaded from:\n{file_path}")
                self.data_updated.emit('force')

        except Exception as e:
            self._handle_error(str(e))

    def cancel_all_operations(self):
        """Cancel all running background operations."""
        for worker in self.active_workers:
            try:
                worker.quit()
                worker.wait()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()
        self._update_progress(0, "Operations cancelled")
        self.unfreeze_ui()

    def _get_current_parameters(self) -> FTTCParameters:
        """Get current FTTC parameters from parameter manager."""
        return FTTCParameters(
            young_modulus=self.parameter_manager.get_value('young_modulus'),  # convert to kPa
            poisson_ratio_substrate=self.parameter_manager.get_value('poisson_ratio_substrate'),
            gel_height=self.parameter_manager.get_value('gel_height'),
            lanczos_exp=self.parameter_manager.get_value('lanczos_exp'),
            regularization=self.parameter_manager.get_value('regularization'),
            auto_gcv=self.parameter_manager.get_value('auto_gcv'),
            force_vector_stride=self.parameter_manager.get_value('force_vector_stride'),
            force_arrow_scale=self.parameter_manager.get_value('force_arrow_scale'),
            f_max=self.parameter_manager.get_value('f_max'),
            frame_interval=self.parameter_manager.get_value('frame_interval')
        )

    def _validate_prerequisites(self) -> bool:
        """Check if required data and parameters are available."""
        if self.data_manager.displacement_field is None:
            QMessageBox.warning(None, "Warning", "No displacement data loaded")
            return False

        params = self._get_current_parameters()
        is_valid, error_message = self.service.validate_parameters(params)
        if not is_valid:
            QMessageBox.warning(None, "Invalid Parameters", error_message)
            return False

        return True

    def _handle_error(self, error_message: str):
        """Handle errors during operations."""
        self._update_progress(0, f"Error: {error_message}")
        self.analysis_failed.emit(error_message)
        QMessageBox.critical(None, "Error", error_message)
        self.unfreeze_ui()

    def _update_progress(self, progress: int, message: str):
        """Update progress information."""
        self.progress_updated.emit(progress, message)


class FTTCWidget(BaseAnalysisWidget):
    """Main widget for FTTC force calculation."""

    force_calculated = Signal(object)

    def __init__(self, viewer: Viewer, data_manager: DataManager,
                 parameter_manager: ParameterManager,
                 visualization_manager: VisualizationManager):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize service and managers
        self.service = FTTCService()
        self.parameter_manager = parameter_manager

        # Initialize controller first
        self.controller = FTTCController(
            viewer=viewer,
            service=self.service,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
            data_panel=None  # Will be set later
        )

        # Initialize panels with controller reference
        self.parameter_panel = FTTCParameterPanel(parameter_manager)
        self.parameter_panel.set_controller(self.controller)  # Set controller reference
        self.data_panel = FTTCDataPanel(data_manager, viewer)

        # Initialize action panel
        self.action_panel = FTTCActionPanel(self.controller)

        # Set panels in controller
        self.controller.set_panels(self.parameter_panel, self.action_panel)
        self.controller.data_panel = self.data_panel

        # Set controller in data panel
        self.data_panel.set_controller(self.controller)

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left side: Colorbar
        colorbar_container = QWidget()
        colorbar_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(0, 0, 0, 0)

        self.colorbar_manager = ColorbarManager()
        colorbar_group = self.create_colorbar_widget(
            colormap_name='inferno',
            label="Force (Pa)",
            clim=(0, 500.0),  # Changed default to 500
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        colorbar_layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        colorbar_container.setLayout(colorbar_layout)

        main_layout.addWidget(colorbar_container)

        # Create a scroll area for the right side
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        scroll_area.setFixedWidth(360)

        # Create a widget to hold all the content in the scroll area
        scroll_content = QWidget()
        scroll_content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(0)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Add panels to the scroll area
        scroll_layout.addWidget(self.data_panel)
        scroll_layout.addWidget(self.parameter_panel)
        scroll_layout.addWidget(self.action_panel)

        # Add status frame
        status_frame = QFrame()
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.progress_bar)
        # Add status frame
        status_frame = QFrame()
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)
        scroll_layout.addWidget(status_frame)

        # Set the layout for the scroll content
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self._update_button_states()

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Check data availability
        has_displacement = self.data_manager.displacement_field is not None
        has_force = self.data_manager.force_field is not None

        # Update displacement status
        if has_displacement:
            try:
                self.data_panel.displacement_status.setText(
                    f"Loaded: {self.data_manager.displacement_field.shape}"
                )
            except Exception as e:
                self.data_panel.displacement_status.setText(f"Error ({str(e)})")
        else:
            self.data_panel.displacement_status.setText("Not loaded")

        # Update button states
        if hasattr(self, 'controller'):
            analysis_running = (hasattr(self.controller, 'active_workers') and
                                len(self.controller.active_workers) > 0)

            # Update action panel button states
            self.action_panel.update_button_states(
                displacement_data=has_displacement and not analysis_running,
                force_data=has_force and not analysis_running
            )

            # Update GCV button state based on auto GCV checkbox
            if hasattr(self.parameter_panel, 'parameter_widgets'):
                auto_gcv = self.parameter_panel.parameter_widgets["auto_gcv"].isChecked()
                self.parameter_panel.parameter_widgets["gcv_button"].setEnabled(
                    has_displacement and not analysis_running and not auto_gcv
                )

            # Update status message
            if analysis_running:
                self.status_label.setText("Analysis in progress...")
            elif has_displacement:
                self.status_label.setText("Ready for force calculation")
            else:
                self.status_label.setText("Missing required displacement data")

    def _connect_signals(self):
        """Connect all widget signals."""
        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.data_updated.connect(self._on_data_updated)

        # Connect analysis signals
        self.controller.analysis_started.connect(self._on_analysis_started)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)

        # Connect data panel signals
        self.data_panel.data_loaded.connect(self._on_data_loaded)
        self.data_panel.displacement_data_loaded.connect(self._on_displacement_loaded)

        # Connect parameter panel changes
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)

        # Connect UI freeze signals
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Connect auto GCV checkbox to button state
        self.parameter_panel.parameter_widgets["auto_gcv"].stateChanged.connect(
            lambda state: self.parameter_panel.parameter_widgets["gcv_button"].setEnabled(not bool(state))
        )
        self.parameter_panel.parameter_value_changed.connect(self._on_parameter_changed)


    def _on_displacement_loaded(self, displacement_data: dict):
        """Handle displacement data loading."""
        try:
            # Update visualization
            self.visualization_manager.visualize_displacement_results(
                displacement_data,
                downscale_factor=displacement_data.get('parameters', {}).get('downscale_factor', 1)
            )

            # Update colorbar with default force range
            self.colorbar_manager.update_limits(0, 500.0)  # Changed default to 500

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process displacement data: {str(e)}")

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _on_data_updated(self, data_type: str):
        """Handle data updates."""
        self._update_button_states()

    def _on_data_loaded(self, data_type: str):
        """Handle data loading completion."""
        self._update_button_states()

    def _on_analysis_started(self):
        """Handle analysis start."""
        self._update_button_states(analysis_running=True)

    def _on_analysis_completed(self, results):
        """Handle analysis completion."""
        self._update_button_states()
        self.force_calculated.emit(results)

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_button_states()
        self._update_status(0, f"Analysis failed: {error_msg}")

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        if param_name == "f_max":
            # Update colorbar immediately
            self.colorbar_manager.update_limits(0, value)
    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze."""
        if not frozen:
            self._update_button_states()

    def _update_button_states(self, analysis_running: bool = False):
        """Update button states based on current state."""
        has_displacement = self.data_manager.displacement_field is not None
        has_force = self.data_manager.force_field is not None

        self.action_panel.update_button_states(
            displacement_data=has_displacement and not analysis_running,
            force_data=has_force and not analysis_running
        )

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()

        # Cancel any running operations
        self.controller.cancel_all_operations()

        # Clean up visualizations
        self.visualization_manager.cleanup()

        super().cleanup()
