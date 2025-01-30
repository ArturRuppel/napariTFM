from pathlib import Path
from typing import Any

import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import Signal, Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QFileDialog, QScrollArea, QCheckBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox,
    QSizePolicy, QFrame, QProgressBar
)
from napari.layers import Image

from napari.qt.threading import thread_worker

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.data_manager import DataManager
from napariTFM.parameter_manager import ParameterManager, ParameterCategory
from napariTFM.visualization_manager import VisualizationManager
from napariTFM.services.displacement_service import DisplacementService, DisplacementParameters, DisplacementResult


class DisplacementDataPanel(QWidget):
    """Panel for handling data loading and status display."""

    data_loaded = Signal(str)  # Emits data type that was loaded
    reference_loaded = Signal(object)
    beads_loaded = Signal(object)
    # region === Initialization
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

        data_group.setLayout(group_layout)
        layout.addWidget(data_group)
        self.setLayout(layout)
    def _connect_signals(self):
        """Connect UI signals to controller methods."""
        self.load_beads_btn.clicked.connect(
            lambda: self._load_data('beads')
        )
        self.load_reference_btn.clicked.connect(
            lambda: self._load_data('reference')
        )

    # endregion === Initialization

    # region === Controller Setup
    def set_controller(self, controller):
        """Set the controller and connect signals."""
        self.controller = controller
        self.load_beads_btn.clicked.connect(lambda: self.controller.load_active_layer('beads'))
        self.load_reference_btn.clicked.connect(lambda: self.controller.load_active_layer('reference'))

    def _load_data(self, data_type: str):
        """Load data from active layer."""
        active_layer = self._get_active_layer()
        if active_layer is None:
            return

        try:
            data = active_layer.data

            if data_type == 'beads':
                self.data_manager.set_preprocessing_results(bead_stack=data)
                self.bead_status.setText(f"Loaded: {data.shape}")
                self.beads_loaded.emit(data)
            else:  # reference
                self.data_manager.set_preprocessing_results(reference=data)
                self.reference_status.setText(f"Loaded: {data.shape}")
                self.reference_loaded.emit(data)

            self.data_loaded.emit(data_type)

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _get_active_layer(self):
        """Get the currently active napari layer."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(
                self,
                "No Layer Selected",
                "Please select an image layer first."
            )
        return active_layer

    # endregion === Controller Setup

    # region === State Management
    def update_button_states(self, active_layer_exists: bool = False):
        """Update button states based on layer selection."""
        active_layer = self.viewer.layers.selection.active
        has_valid_layer = active_layer is not None and isinstance(active_layer, Image)

        self.load_beads_btn.setEnabled(has_valid_layer)
        self.load_reference_btn.setEnabled(has_valid_layer)

    def update_data_status(self):
        """Update status labels based on loaded data."""
        # Update reference status
        ref_data = self.data_manager.preprocessed_reference
        if ref_data is not None:
            self.reference_status.setText(f"Loaded: {ref_data.shape}")
        else:
            self.reference_status.setText("Not loaded")

        # Update bead status
        bead_data = self.data_manager.preprocessed_bead_stack
        if bead_data is not None:
            self.bead_status.setText(f"Loaded: {bead_data.shape}")
        else:
            self.bead_status.setText("Not loaded")

    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        self.load_beads_btn.setEnabled(not frozen)
        self.load_reference_btn.setEnabled(not frozen)

    # endregion === State Management




class DisplacementParameterPanel(QWidget):
    """Panel for handling all displacement parameter inputs."""

    parameter_changed = Signal(str, object)  # (param_name, value)
    parameters_reset = Signal()

    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_spins = {}
        self.parameter_combos = {}
        self._setup_ui()
        self._connect_signals()

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)

    def _connect_signals(self):
        """Connect widget signals."""
        # Connect all spinboxes
        for name, spin in self.parameter_spins.items():
            spin.valueChanged.connect(
                lambda value, n=name: self._on_value_changed(n, value)
            )

        # Connect reset button
        self.reset_btn.clicked.connect(self._reset_parameters)

    def _on_value_changed(self, param_name: str, value: object):
        """Handle parameter value changes."""
        # Update parameter manager
        self.parameter_manager.set_parameter(param_name, value)
        # Emit our own signal
        self.parameter_changed.emit(param_name, value)

    def _sync_parameter(self, param_name: str, value: Any):
        """Sync a single parameter from parameter manager."""
        if param_name in self.parameter_spins:
            self._safe_set_value(self.parameter_spins[param_name], value)
        elif param_name in self.parameter_combos:
            self._safe_set_combo_text(self.parameter_combos[param_name], value)

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        if category == ParameterCategory.DISPLACEMENT:
            self._sync_widget_with_parameters()

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager."""
        self._block_widgets(True)
        try:
            for name, spin in self.parameter_spins.items():
                value = self.parameter_manager.get_parameter(name)
                if value is not None:
                    self._safe_set_value(spin, value)

            for name, combo in self.parameter_combos.items():
                value = self.parameter_manager.get_parameter(name)
                if value is not None:
                    self._safe_set_combo_text(combo, value)
        finally:
            self._block_widgets(False)

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
        """Safely set combo box text."""
        if combo is not None and text is not None:
            combo.blockSignals(True)
            try:
                index = combo.findText(str(text), Qt.MatchFixedString)
                if index >= 0:
                    combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)

    def _block_widgets(self, block: bool):
        """Block or unblock all widget signals."""
        for widget in self.parameter_spins.values():
            widget.blockSignals(block)
        for widget in self.parameter_combos.values():
            widget.blockSignals(block)

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Add parameter groups
        layout.addWidget(self._create_flow_parameters())
        layout.addWidget(self._create_analysis_parameters())
        layout.addWidget(self._create_visualization_parameters())

        # Add reset button
        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.setToolTip("Reset all displacement parameters to their default values")
        layout.addWidget(self.reset_btn)

        self.setLayout(layout)
        self._connect_signals()
        self._sync_widget_with_parameters()

    def _create_flow_parameters(self) -> QGroupBox:
        """Create optical flow parameter group."""
        group = QGroupBox("Optical Flow Parameters")
        layout = QVBoxLayout()

        # Define parameters with tooltips
        params = [
            ("tau", "Tau:", 0.01, 1.0, 0.01,
             "Time step for optical flow computation. Lower values give more accurate but slower results"),
            ("lambda_", "Lambda:", 0.01, 1.0, 0.01,
             "Regularization parameter. Higher values produce smoother flow fields"),
            ("theta", "Theta:", 0.1, 1.0, 0.1,
             "Weight parameter for the divergence term. Controls flow field smoothness"),
            ("nscales", "Pyramid Scales:", 1, 10, 1,
             "Number of pyramid levels. More levels handle larger displacements but increase computation time"),
            ("warps", "Warps:", 1, 10, 1,
             "Number of warping steps per scale. More warps increase accuracy for large displacements"),
            ("epsilon", "Epsilon:", 0.001, 0.1, 0.001,
             "Stopping criterion threshold. Lower values give more precise results but longer computation times"),
            ("inner_iterations", "Inner Iterations:", 1, 50, 1,
             "Maximum number of inner iterations. More iterations improve accuracy but increase computation time"),
            ("outer_iterations", "Outer Iterations:", 1, 20, 1,
             "Maximum number of outer iterations. More iterations improve accuracy but increase computation time"),
            ("scale_step", "Scale Step:", 0.1, 0.99, 0.01,
             "Scale factor between pyramid levels. Lower values create more pyramid levels"),
            ("median_filtering", "Median Filter:", 1, 9, 2,
             "Size of median filter for post-processing. Larger values remove more noise but may lose detail"),
        ]

        for name, label, min_val, max_val, step, tooltip in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            if isinstance(step, int):
                spin = QSpinBox()
                spin.setFixedWidth(135)
                spin.setRange(min_val, max_val)
                spin.setSingleStep(step)
            else:
                spin = QDoubleSpinBox()
                spin.setFixedWidth(135)

                spin.setRange(min_val, max_val)
                spin.setSingleStep(step)
                if name == "epsilon":
                    spin.setDecimals(3)
                else:
                    spin.setDecimals(2)

            spin.setToolTip(tooltip)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_analysis_parameters(self) -> QGroupBox:
        """Create analysis parameter group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Downscale factor
        downscale_layout = QHBoxLayout()
        downscale_layout.addWidget(QLabel("Downscale Factor:"))
        downscale_spin = QSpinBox()
        downscale_spin.setFixedWidth(135)

        downscale_spin.setRange(1, 10)
        downscale_spin.setToolTip(
            "Factor for spatial averaging of displacement field.\n"
            "1 = no averaging (full resolution)\n"
            "Higher values reduce resolution but improve signal-to-noise ratio"
        )
        self.parameter_spins['downscale_factor'] = downscale_spin
        downscale_layout.addWidget(downscale_spin)
        layout.addLayout(downscale_layout)

        group.setLayout(layout)
        return group

    def _create_visualization_parameters(self) -> QGroupBox:
        """Create visualization parameter group."""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        params = [
            ("disp_vector_stride", "Vector Stride:", 1, 100, 1,
             "Display every nth vector. Higher values show fewer vectors but improve clarity"),
            ("disp_arrow_scale", "Arrow Scale:", 0.1, 10.0, 0.1,
             "Scale factor for displacement vectors. Adjust to make vectors more visible"),
            ("d_max", "Max Displacement (µm):", 0.1, 200.0, 0.1,
             "Maximum displacement value for color scaling"),
        ]

        for name, label, min_val, max_val, step, tooltip in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            if isinstance(step, int):
                spin = QSpinBox()
                spin.setFixedWidth(135)

            else:
                spin = QDoubleSpinBox()
                spin.setFixedWidth(135)
                spin.setDecimals(1)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setToolTip(tooltip)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _reset_parameters(self):
        """Reset parameters to defaults."""
        self.parameter_manager.reset_displacement_parameters()
        self.parameters_reset.emit()

    def update_parameter(self, name: str, value: Any):
        """Update a single parameter value."""
        try:
            if name in self.parameter_spins:
                self._safe_set_value(self.parameter_spins[name], value)
            elif name in self.parameter_combos:
                self._safe_set_combo_text(self.parameter_combos[name], value)
        except Exception as e:
            print(f"Error updating parameter {name}: {str(e)}")

    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        for spin in self.parameter_spins.values():
            spin.setEnabled(not frozen)
        self.reset_btn.setEnabled(not frozen)

    def _update_widget_value(self, param_name: str, value: object):
        """Update widget when parameter changes externally."""
        if param_name in self.parameter_widgets:
            widget = self.parameter_widgets[param_name]
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)


class DisplacementActionPanel(QWidget):
    """Panel for displacement analysis actions and progress display."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create grid of button pairs
        button_layout = QVBoxLayout()

        # Row 1: Preview and Calculate
        row1_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.analyze_btn = QPushButton("Calculate All Frames")
        row1_layout.addWidget(self.preview_btn)
        row1_layout.addWidget(self.analyze_btn)
        button_layout.addLayout(row1_layout)

        # Row 2: Save and Load
        row2_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Displacements")
        self.load_btn = QPushButton("Load Displacements")
        row2_layout.addWidget(self.save_btn)
        row2_layout.addWidget(self.load_btn)
        button_layout.addLayout(row2_layout)

        layout.addLayout(button_layout)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel Operation")
        layout.addWidget(self.cancel_btn)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect action panel buttons to controller methods."""
        self.preview_btn.clicked.connect(self.controller.preview_displacement)
        self.analyze_btn.clicked.connect(self.controller.calculate_all_frames)
        self.save_btn.clicked.connect(self.controller.save_results)
        self.load_btn.clicked.connect(self.controller.load_results)
        self.cancel_btn.clicked.connect(self.controller.cancel_operation)

    def freeze_ui(self, freeze=True):
        """Disable/enable action buttons (keep cancel enabled)"""
        buttons = [
            self.preview_btn, self.analyze_btn,
            self.save_btn, self.load_btn
        ]
        for btn in buttons:
            btn.setEnabled(not freeze)
        self.cancel_btn.setEnabled(True)

    def update_button_states(self, has_reference: bool = False,
                             has_beads: bool = False,
                             has_results: bool = False):
        """Update button states based on data availability."""
        self.preview_btn.setEnabled(has_reference and has_beads)
        self.analyze_btn.setEnabled(has_reference and has_beads)
        self.save_btn.setEnabled(has_results)
        self.load_btn.setEnabled(True)


class DisplacementController(QObject):
    """Controller coordinating displacement analysis components."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(DisplacementResult)  # Results object
    analysis_failed = Signal(str)  # Error message
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
        self.preview_enabled = False

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)

    def set_panels(self, parameter_panel, action_panel):
        """Set the parameter and action panels."""
        self.parameter_panel = parameter_panel
        self.action_panel = action_panel

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
                self.data_manager.set_preprocessed_bead_stack(data)

            elif data_type == 'reference':
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")
                self.data_manager.set_preprocessed_reference(data)
            else:
                raise ValueError(f"Invalid data type: {data_type}")

            # Update UI state and emit signal
            self.data_updated.emit(data_type)
            if self.preview_enabled:
                self._update_preview()

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))

    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Calculating displacement preview...")

            # Get current frame data
            current_frame = self.viewer.dims.current_step[0]
            moving = self.data_manager.preprocessed_bead_stack[current_frame]
            reference = self.data_manager.preprocessed_reference

            # Get parameters and update service
            params = self.parameter_manager.get_displacement_parameters()
            self.service.update_parameters(params)

            # Calculate displacement field for single frame
            result = self.service.calculate_displacement_field(reference, moving)

            # Process generator to get the result
            try:
                while True:
                    displacement_field, frame, total = next(result)
                    self.progress_updated.emit(
                        int((frame + 1) / total * 100),
                        f"Processing preview frame..."
                    )
            except StopIteration as e:
                # Get final result from generator
                final_result = e.value

            if final_result is None:
                raise RuntimeError("Preview calculation failed")

            self.analysis_completed.emit(final_result)

            # Update visualization with preview result
            self.visualization_manager.visualize_displacement_preview(
                final_result.displacement_field[0],  # Single frame result
                params.d_max,
                params.disp_vector_stride,
                params.disp_arrow_scale,
                downscale_factor=params.downscale_factor
            )

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the displacement layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Displacement Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Displacement Magnitude':
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

            # Update status with statistics
            stats = self.visualization_manager.get_displacement_statistics(final_result.displacement_field[0])
            self.progress_updated.emit(
                100,
                f"Maximum displacement: {stats['max']:.2f} µm\n"
                f"Mean displacement: {stats['mean']:.2f} µm"
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.unfreeze_ui()

    @thread_worker
    def _create_displacement_worker(self, reference, bead_stack, params):
        """Create worker for processing data."""
        try:
            # Get the generator from the service
            displacement_generator = self.service.calculate_displacement_field(reference, bead_stack)

            # Process all frames through the generator
            try:
                while True:
                    displacement_field, frame, total = next(displacement_generator)
                    yield {
                        'progress': (frame + 1) / total * 100,
                        'message': f"Processing frame {frame + 1}/{total}"
                    }
            except StopIteration as e:
                # Return the final result from the generator
                return e.value

        except Exception as e:
            raise ValueError(f"Displacement calculation failed: {str(e)}")

    def _handle_analysis_results(self, result: DisplacementResult):
        """Handle completed analysis results."""
        try:
            if result is None:
                raise RuntimeError("Analysis failed to produce results")

            # Update data manager
            self.data_manager.set_displacement_results(result)

            # Update visualization
            self.visualization_manager.visualize_displacement_results()

            # Manage layer visibility and order
            vector_layer = None
            magnitude_layer = None

            # First pass: find the displacement layers and disable all others
            for layer in self.viewer.layers:
                if layer.name == 'Displacement Vectors':
                    vector_layer = layer
                    layer.visible = True
                elif layer.name == 'Displacement Magnitude':
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

    def calculate_all_frames(self):
        """Calculate displacements for all frames."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Starting displacement analysis...")

            # Get parameters and update service
            params = self.parameter_manager.get_displacement_parameters()
            self.service.update_parameters(params)

            # Create worker for processing
            worker = self._create_displacement_worker(
                self.data_manager.preprocessed_reference,
                self.data_manager.preprocessed_bead_stack,
                params
            )

            self.active_workers.append(worker)
            self.analysis_started.emit()

            worker.yielded.connect(self._handle_progress)
            worker.returned.connect(self._handle_analysis_results)
            worker.errored.connect(self._handle_error)
            worker.start()

        except Exception as e:
            self._handle_error(str(e))
            self.unfreeze_ui()

    def save_results(self):
        """Save displacement results to file."""
        try:
            if self.data_manager.displacement_results is None:
                raise ValueError("No displacement results to save")

            save_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Displacement Results",
                str(Path.home()),
                "NumPy Files (*.npy)"
            )

            if save_path:
                # Get results and save
                results = self.data_manager.displacement_results
                np.save(save_path, results)
                self.progress_updated.emit(100, f"Results saved to {save_path}")

        except Exception as e:
            self._handle_error(f"Failed to save results: {str(e)}")

    def load_results(self):
        """Load displacement results from file."""
        try:
            load_path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Displacement Results",
                str(Path.home()),
                "NumPy Files (*.npy)"
            )

            if load_path:
                # Load data
                result = np.load(load_path, allow_pickle=True).item()

                # Update data manager and visualization
                self.data_manager.set_displacement_results(result)
                self.visualization_manager.visualize_displacement_results()

                self.progress_updated.emit(100, f"Results loaded from {load_path}")
                self.analysis_completed.emit(result)

        except Exception as e:
            self._handle_error(f"Failed to load results: {str(e)}")

    def cancel_operation(self):
        """Cancel all running operations."""
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

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.preprocessed_reference is None:
            QMessageBox.warning(None, "Warning", "No reference image loaded")
            return False
        if self.data_manager.preprocessed_bead_stack is None:
            QMessageBox.warning(None, "Warning", "No bead stack loaded")
            return False
        return True

    def _handle_progress(self, progress_info: dict):
        """Handle progress updates."""
        self.progress_updated.emit(
            progress_info['progress'],
            progress_info['message']
        )

    def _handle_error(self, error_msg: str):
        """Handle errors."""
        self.progress_updated.emit(0, f"Error: {error_msg}")
        self.analysis_failed.emit(error_msg)
        QMessageBox.critical(None, "Error", error_msg)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        if self.preview_enabled:
            self._update_preview()

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        if self.preview_enabled:
            self._update_preview()

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

    def _get_current_parameters(self) -> DisplacementParameters:
        """Get current parameters from parameter manager."""
        return DisplacementParameters(
            tau=self.parameter_manager.get_value('tau'),
            lambda_=self.parameter_manager.get_value('lambda_'),
            theta=self.parameter_manager.get_value('theta'),
            nscales=self.parameter_manager.get_value('nscales'),
            warps=self.parameter_manager.get_value('warps'),
            epsilon=self.parameter_manager.get_value('epsilon'),
            inner_iterations=self.parameter_manager.get_value('inner_iterations'),
            outer_iterations=self.parameter_manager.get_value('outer_iterations'),
            scale_step=self.parameter_manager.get_value('scale_step'),
            median_filtering=self.parameter_manager.get_value('median_filtering'),
            downscale_factor=self.parameter_manager.get_value('downscale_factor'),
            pixel_size=self.parameter_manager.get_value('pixel_size'),
            frame_interval=self.parameter_manager.get_value('frame_interval'),
            d_max=self.parameter_manager.get_value('d_max'),
            disp_vector_stride=self.parameter_manager.get_value('disp_vector_stride'),
            disp_arrow_scale=self.parameter_manager.get_value('disp_arrow_scale')
        )

    def _update_progress(self, progress: int, message: str):
        """Update progress information."""
        self.progress_updated.emit(progress, message)


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements using optical flow."""

    displacement_calculated = Signal(object)

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
        self.service = DisplacementService(parameter_manager.get_displacement_parameters())
        self.colorbar_manager = ColorbarManager()

        # Initialize panels
        self.parameter_panel = DisplacementParameterPanel(parameter_manager)
        self.data_panel = DisplacementDataPanel(data_manager, viewer)

        # Initialize controller
        self.controller = DisplacementController(
            viewer=viewer,
            service=self.service,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
            data_panel=self.data_panel
        )

        # Initialize action panel with controller
        self.action_panel = DisplacementActionPanel(self.controller)

        # Set controller in panels
        self.data_panel.set_controller(self.controller)
        self.controller.set_panels(self.parameter_panel, self.action_panel)

        # Set up the UI
        self._setup_ui()

        # Connect signals after UI is fully set up
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

    def _setup_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left: Colorbar (same as preprocessing)
        colorbar_container = self._create_colorbar_container()
        main_layout.addWidget(colorbar_container)

        # Right: Scrollable content
        content_container = self._create_content_container()
        main_layout.addWidget(content_container)

        self.setLayout(main_layout)

    def _create_content_container(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)  # Fixed width for the scroll area

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)  # Constrain width
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        # Match preprocessing's component order
        layout.addWidget(self.data_panel)
        layout.addWidget(self.parameter_panel)
        layout.addWidget(self._create_action_frame())  # Ensure this is called
        layout.addWidget(self._create_status_frame())

        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll

    def _create_action_frame(self):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Constrain height
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)  # Minimize margins

        # Main action row
        action_layout = QHBoxLayout()
        action_layout.setSpacing(6)
        action_layout.setContentsMargins(0, 0, 0, 0)  # Minimize margins

        self.preview_btn = QPushButton("Preview Current Frame")
        self.process_btn = QPushButton("Run Displacement Analysis")

        # Set size policies for buttons to prevent expansion
        self.preview_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.process_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        action_layout.addWidget(self.preview_btn)
        action_layout.addWidget(self.process_btn)
        layout.addLayout(action_layout)

        # Data buttons
        data_layout = QHBoxLayout()
        data_layout.setSpacing(6)
        data_layout.setContentsMargins(0, 0, 0, 0)  # Minimize margins
        self.save_btn = QPushButton("Save Displacements")
        self.load_btn = QPushButton("Load Displacements")


        # Set size policies for additional buttons
        self.save_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.load_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        data_layout.addWidget(self.save_btn)
        data_layout.addWidget(self.load_btn)

        layout.addLayout(data_layout)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel All Operations")
        self.cancel_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addWidget(self.cancel_btn)

        frame.setLayout(layout)
        return frame

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update data panel
        self.data_panel.update_button_states()
        self.data_panel.update_data_status()

        # Update button states
        has_data = (
                self.data_manager.preprocessed_reference is not None and
                self.data_manager.preprocessed_bead_stack is not None
        )
        has_results = self.data_manager.displacement_results is not None

        # Check if preview_btn exists before using it
        if hasattr(self, 'preview_btn') and self.preview_btn is not None:
            self.preview_btn.setEnabled(has_data)
        if hasattr(self, 'process_btn') and self.process_btn is not None:
            self.process_btn.setEnabled(has_data)
        if hasattr(self, 'save_btn') and self.save_btn is not None:
            self.save_btn.setEnabled(has_results)

    def _create_colorbar_container(self):
        # Identical to preprocessing's implementation
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)

        colorbar_group = self.create_colorbar_widget(
            colormap_name='viridis',
            label="Displacement (µm)",
            clim=(0, self.parameter_manager.get_parameter('d_max')),
            colorbar_manager=self.colorbar_manager
        )
        layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        layout.addStretch()
        container.setLayout(layout)
        return container

    def _connect_signals(self):
        """Connect all widget signals."""
        if hasattr(self, 'preview_btn') and self.preview_btn is not None:
            self.preview_btn.clicked.connect(self.controller.preview_displacement)
        if hasattr(self, 'process_btn') and self.process_btn is not None:
            self.process_btn.clicked.connect(self.controller.calculate_all_frames)
        if hasattr(self, 'save_btn') and self.save_btn is not None:
            self.save_btn.clicked.connect(self.controller.save_results)
        if hasattr(self, 'load_btn') and self.load_btn is not None:
            self.load_btn.clicked.connect(self.controller.load_results)
        if hasattr(self, 'cancel_btn') and self.cancel_btn is not None:
            self.cancel_btn.clicked.connect(self.controller.cancel_operation)

        # Connect controller signals
        self.controller.progress_updated.connect(self._update_status)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)
        self.controller.data_updated.connect(self._update_ui_state)
        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Connect parameter panel signals
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)
        self.parameter_panel.parameters_reset.connect(self._on_parameters_reset)

        # Add layer selection monitoring
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _create_status_frame(self):
        # Identical implementation to preprocessing
        frame = QFrame()
        layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        frame.setLayout(layout)
        return frame

    def _on_analysis_completed(self, results):
        """Handle completed analysis."""
        self.save_btn.setEnabled(True)
        # Update colorbar before visualization
        if hasattr(results, 'parameters'):
            d_max = results.parameters.d_max
            self.colorbar_manager.update_limits(0, d_max)
        self.displacement_calculated.emit(results)

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self.save_btn.setEnabled(False)
        QMessageBox.critical(self, "Error", error_msg)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        pass

    def _on_parameters_reset(self):
        """Handle parameter reset and update status."""
        self._update_status(0, "Displacement parameters reset to default values.")

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.displacement_results is not None:
            self.visualization_manager.update_displacement_frame(
                self.viewer.dims.current_step[0]
            )

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze."""
        self.preview_btn.setEnabled(not frozen and self._has_required_data())
        self.process_btn.setEnabled(not frozen and self._has_required_data())
        self.save_btn.setEnabled(not frozen and self.data_manager.displacement_results is not None)
        self.load_btn.setEnabled(not frozen)
        self.cancel_btn.setEnabled(frozen)

    def _has_required_data(self) -> bool:
        """Check if required data is available."""
        return (
                self.data_manager.preprocessed_reference is not None and
                self.data_manager.preprocessed_bead_stack is not None
        )

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()

    def _on_data_loaded(self, data_type: str):
        """Handle data loading events."""
        self.data_panel.update_data_status()
        self.action_panel.update_button_states(
            has_reference=self.data_manager.preprocessed_reference is not None,
            has_beads=self.data_manager.preprocessed_bead_stack is not None,
            has_results=self.data_manager.displacement_field is not None
        )

    def _create_preview_frame(self) -> QFrame:
        """Create preview control frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        # Create preview toggle
        preview_layout = QHBoxLayout()
        self.preview_check = QCheckBox("Show Preview")
        self.preview_check.setToolTip("Show live preview of displacement calculation for current frame")
        preview_layout.addWidget(self.preview_check)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)

        frame.setLayout(layout)
        return frame

    def _on_preview_toggled(self, enabled: bool):
        """Handle preview toggle."""
        if hasattr(self.controller, 'preview_enabled'):
            self.controller.preview_enabled = enabled

        if enabled:
            self.controller.preview_displacement()
        else:
            self.visualization_manager.handle_preview(
                frame=None,
                enable=False
            )
