from pathlib import Path
from typing import Any

import numpy as np
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal, Qt, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QFileDialog, QScrollArea, QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox, QSpacerItem,
    QSizePolicy, QFrame, QProgressBar
)

from widgets._base_widget import BaseAnalysisWidget
from utilities.colorbar import ColorbarManager
from utilities.data_manager import DataManager
from utilities.parameter_manager import ParameterManager, ParameterCategory
from services.displacement_service import DisplacementService, DisplacementResult
from utilities.visualization_manager import VisualizationManager

# TODO load displacement throws error

class DisplacementDataPanel(QWidget):
    """Panel for handling data loading and status display."""

    data_loaded = Signal(str)  # Emits data type that was loaded
    reference_loaded = Signal(object)
    beads_loaded = Signal(object)

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
        self.load_beads_btn.setFixedHeight(25)
        self.load_beads_btn.setToolTip("Load bead stack data from active layer")
        self.bead_status = QLabel("Not loaded")
        self.bead_status.setWordWrap(True)

        bead_layout.addWidget(self.load_beads_btn)
        bead_layout.addWidget(self.bead_status)
        group_layout.addLayout(bead_layout)

        # Reference data row
        ref_layout = QHBoxLayout()
        self.load_reference_btn = QPushButton("Load Reference Image")
        self.load_reference_btn.setFixedWidth(150)
        self.load_reference_btn.setFixedHeight(25)
        self.load_reference_btn.setToolTip("Load reference image from active layer")
        self.reference_status = QLabel("Not loaded")
        self.reference_status.setWordWrap(True)
        ref_layout.addWidget(self.load_reference_btn)
        ref_layout.addWidget(self.reference_status)
        group_layout.addLayout(ref_layout)

        # Add description label for required data
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


class DisplacementParameterPanel(QWidget):
    """Panel for handling all displacement parameter inputs."""

    parameter_changed = Signal(str, object)  # (param_name, value)
    parameters_reset = Signal()

    # region === Initialization
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

    # endregion === Initialization

    # region === UI Creation

    def _create_flow_parameters(self) -> QGroupBox:
        """Create optical flow parameter group."""
        group = QGroupBox("Optical Flow Parameters")
        layout = QVBoxLayout()

        # Basic parameter (lambda)
        lambda_layout = QHBoxLayout()
        lambda_layout.addWidget(QLabel("Lambda:"))
        lambda_spin = QDoubleSpinBox()
        lambda_spin.setFixedWidth(135)
        lambda_spin.setRange(0.01, 1.0)
        lambda_spin.setSingleStep(0.01)
        lambda_spin.setDecimals(2)
        lambda_spin.setToolTip("Weight parameter for the data term. Smaller values produce smoother solutions.")
        self.parameter_spins['lambda_'] = lambda_spin
        lambda_layout.addWidget(lambda_spin)
        layout.addLayout(lambda_layout)

        # Advanced parameters section
        advanced_container = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(5)

        # Create a custom label-style toggle
        toggle_container = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)

        self.arrow_label = QLabel("▶")
        self.text_label = QLabel("Advanced Parameters")

        toggle_layout.addWidget(self.arrow_label)
        toggle_layout.addWidget(self.text_label)
        toggle_layout.addStretch()  # Push widgets to the left

        toggle_container.setLayout(toggle_layout)
        toggle_container.setCursor(Qt.PointingHandCursor)  # Show pointer cursor on hover

        # Install event filter for click handling
        toggle_container.mousePressEvent = self._toggle_advanced_parameters

        layout.addWidget(toggle_container)

        # Container for advanced parameters
        self.advanced_widget = QWidget()
        self.advanced_widget.setVisible(False)
        advanced_params_layout = QVBoxLayout()
        advanced_params_layout.setContentsMargins(10, 0, 0, 0)  # Add left indent

        # Define advanced parameters with tooltips
        params = [
            ("tau", "Tau:", 0.1, 1.0, 0.01,
             "Time step of the numerical scheme. Smaller values may improve accuracy but increase computation time."),
            ("theta", "Theta:", 0.01, 1.0, 0.01,
             "Weight parameter that balances between matching image intensities (data term) and ensuring smooth transitions between neighboring flow vectors. Lower values recommended."),
            ("nscales", "Pyramid Scales:", 1, 10, 1,
             "Number of image pyramid levels. More levels allow detection of larger displacements but increase computation time. Reduce if small displacements expected."),
            ("warps", "Warps:", 1, 10, 1,
             "Number of warpings per scale. More warps increase accuracy but increase computation time."),
            ("epsilon", "Epsilon:", 0.001, 0.1, 0.001,
             "Stopping criterion threshold. Lower values give more precise results but increase computation time."),
            ("inner_iterations", "Inner Iterations:", 1, 50, 1,
             "Inner iterations between outlier filtering. More iterations may improve accuracy but increase computation time."),
            ("outer_iterations", "Outer Iterations:", 1, 20, 1,
             "Outer iterations (number of inner loops). More iterations may improve accuracy but increase computation time."),
            ("scale_step", "Scale Step:", 0.1, 0.99, 0.01,
             "Scale factor between pyramid levels. For a 1000x1000 image with scale_step=0.5: 1000→500→250→125. With scale_step=0.8: 1000→800→640→512"),
            ("median_filtering", "Median Filter:", 1, 5, 2,
             "Median filter kernel size (1 = no filter) (3 or 5)"),
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
            advanced_params_layout.addLayout(row)

        self.advanced_widget.setLayout(advanced_params_layout)
        advanced_layout.addWidget(self.advanced_widget)
        advanced_container.setLayout(advanced_layout)
        layout.addWidget(advanced_container)

        group.setLayout(layout)
        return group

    def _toggle_advanced_parameters(self, event):
        """Toggle visibility of advanced parameters."""
        self.advanced_widget.setVisible(not self.advanced_widget.isVisible())
        self.arrow_label.setText("▼" if self.advanced_widget.isVisible() else "▶")
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

    # endregion === UI Creation

    # region === Signal Handling
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

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        if category == ParameterCategory.DISPLACEMENT:
            self._sync_widget_with_parameters()

    # endregion === Signal Handling

    # region === Parameter Management
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

    def _sync_parameter(self, param_name: str, value: Any):
        """Sync a single parameter from parameter manager."""
        if param_name in self.parameter_spins:
            self._safe_set_value(self.parameter_spins[param_name], value)
        elif param_name in self.parameter_combos:
            self._safe_set_combo_text(self.parameter_combos[param_name], value)

    def _reset_parameters(self):
        """Reset parameters to defaults."""
        self.parameter_manager.reset_displacement_parameters()
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

    # endregion === Parameter Management

    # region === State Management
    def freeze_ui(self, frozen: bool):
        """Freeze or unfreeze UI elements."""
        for spin in self.parameter_spins.values():
            spin.setEnabled(not frozen)
        self.reset_btn.setEnabled(not frozen)

    def _block_widgets(self, block: bool):
        """Block or unblock all widget signals."""
        for widget in self.parameter_spins.values():
            widget.blockSignals(block)
        for widget in self.parameter_combos.values():
            widget.blockSignals(block)

    # endregion === State Management


class DisplacementActionPanel(QWidget):
    """Panel for displacement analysis actions and progress display."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Create button pairs in rows
        # Row 1: Preview and Calculate buttons
        row1_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Current Frame")
        self.preview_btn.setToolTip(
            "Calculate and visualize displacement for the current frame only"
        )
        self.calculate_btn = QPushButton("Calculate All Frames")
        self.calculate_btn.setToolTip(
            "Calculate displacements for all frames in the dataset"
        )
        self.preview_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.calculate_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        row1_layout.addWidget(self.preview_btn)
        row1_layout.addWidget(self.calculate_btn)
        layout.addLayout(row1_layout)

        # Row 2: Save and Load buttons
        row2_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Displacements")
        self.save_btn.setToolTip(
            "Save displacement calculation results to file"
        )
        self.load_btn = QPushButton("Load Displacements")
        self.load_btn.setToolTip(
            "Load previously saved displacement results"
        )
        self.save_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.load_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        row2_layout.addWidget(self.save_btn)
        row2_layout.addWidget(self.load_btn)
        layout.addLayout(row2_layout)

        # Cancel button (full width)
        self.cancel_btn = QPushButton("Cancel Operation")
        self.cancel_btn.setToolTip(
            "Cancel the current operation"
        )
        layout.addWidget(self.cancel_btn)

        # Connect signals
        self.preview_btn.clicked.connect(self.controller.preview_displacement)
        self.calculate_btn.clicked.connect(self.controller.calculate_all_frames)
        self.save_btn.clicked.connect(self.controller.save_results)
        self.load_btn.clicked.connect(self.controller.load_results)
        self.cancel_btn.clicked.connect(self.controller.cancel_operation)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect action panel buttons to controller methods."""
        self.preview_btn.clicked.connect(self.controller.preview_displacement)
        self.calculate_btn.clicked.connect(self.controller.calculate_all_frames)
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
                             has_reference: bool = False,
                             has_beads: bool = False,
                             has_results: bool = False):
        """Update button states based on data availability."""
        self.preview_btn.setEnabled(has_reference and has_beads)
        self.calculate_btn.setEnabled(has_reference and has_beads)
        self.save_btn.setEnabled(has_results)
        self.load_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)


class DisplacementController(QObject):
    """Controller coordinating displacement analysis components."""

    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(DisplacementResult)  # Results object
    analysis_failed = Signal(str)  # Error message
    data_updated = Signal(str)  # Data type that was updated
    ui_frozen = Signal(bool)

    # region === Initialization
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

    # endregion === Initialization

    # region === Processing Execution
    def get_displacement_statistics(self, flow: np.ndarray) -> dict:
        """Calculate displacement statistics."""
        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
        return {
            'max': magnitude.max(),
            'mean': magnitude.mean(),
            'std': magnitude.std(),
            'median': np.median(magnitude)
        }
    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self.freeze_ui()
            self.progress_updated.emit(0, "Calculating displacement preview...")

            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
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
            stats = self.get_displacement_statistics(final_result.displacement_field[0])
            self.progress_updated.emit(
                100,
                f"Maximum displacement: {stats['max']:.2f} µm\n"
                f"Mean displacement: {stats['mean']:.2f} µm"
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
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

    def cancel_operation(self):
        """Cancel all running operations."""
        for worker in self.active_workers:
            try:
                worker.quit()
                worker.wait(500)
                if worker.isRunning():
                    worker.terminate()
                worker.deleteLater()
            except Exception:
                pass
        self.active_workers.clear()

        # Clear any partial results when canceling
        self.data_manager.set_displacement_results(None)

        # Update UI
        self.progress_updated.emit(0, "Operations cancelled")
        self.unfreeze_ui()

        # Force UI state update to reflect cleared results
        self.data_updated.emit('displacement')
    # endregion === Processing Execution

    # region === Parameter Handling
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        if self.preview_enabled:
            self._update_preview()

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        if self.preview_enabled:
            self._update_preview()

    def _sync_parameters_with_results(self, result):
        """Sync parameters from loaded results."""
        if not hasattr(result, 'parameters'):
            return

        params = result.parameters
        for param_name, value in vars(params).items():
            if param_name != '_sa_instance_state':  # Skip SQLAlchemy state
                self.parameter_manager.set_parameter(param_name, value)

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

                # Update parameters if they exist in the results
                if hasattr(result, 'parameters'):
                    # Block parameter change signals temporarily
                    if self.parameter_panel:
                        self.parameter_panel._block_widgets(True)
                    try:
                        # Update parameter manager with loaded parameters
                        params = result.parameters
                        for param_name, value in vars(params).items():
                            if param_name != '_sa_instance_state':  # Skip SQLAlchemy state
                                self.parameter_manager.set_parameter(param_name, value)

                        # Sync UI with new parameters
                        if self.parameter_panel:
                            self.parameter_panel._sync_widget_with_parameters()
                    finally:
                        if self.parameter_panel:
                            self.parameter_panel._block_widgets(False)

                # Update data manager and visualization
                self.data_manager.set_displacement_results(result)
                self.visualization_manager.visualize_displacement_results()

                # Manage layer visibility after loading
                for layer in self.viewer.layers:
                    if layer.name in ['Displacement Vectors', 'Displacement Magnitude']:
                        layer.visible = True
                        # Move displacement layers to top
                        self.viewer.layers.move(self.viewer.layers.index(layer), -1)
                    else:
                        layer.visible = False

                self.progress_updated.emit(100, f"Results and parameters loaded from {load_path}")
                self.analysis_completed.emit(result)

        except Exception as e:
            self._handle_error(f"Failed to load results: {str(e)}")

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.preprocessed_reference is None:
            QMessageBox.warning(None, "Warning", "No reference image loaded")
            return False
        if self.data_manager.preprocessed_bead_stack is None:
            QMessageBox.warning(None, "Warning", "No bead stack loaded")
            return False
        return True

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

    # endregion === Data Management

    # region === State Management
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

    # endregion === State Management


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements using optical flow."""

    displacement_calculated = Signal(object)

    # region === Initialization
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

        # Initialize UI state
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
            colormap_name='viridis',
            label="Displacement (µm)",
            clim=(0, self.parameter_manager.get_parameter('d_max')),
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
        self.status_label.setWordWrap(True)  # Enable text wrapping
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _create_action_frame(self):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Constrain height
        layout = QVBoxLayout()

        # Main action row
        action_layout = QHBoxLayout()

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
        self.cancel_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.cancel_btn)

        frame.setLayout(layout)
        return frame

    # endregion

    # region === Signal Handling
    def _connect_signals(self):
        """Connect all widget signals."""
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

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        pass

    def _on_parameters_reset(self):
        """Handle parameter reset and update status."""
        self._update_status(0, "Displacement parameters reset to default values.")

    # endregion

    # region === State Management
    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        # Update data panel
        self.data_panel.update_button_states()
        self.data_panel.update_data_status()

        # Get current data state
        has_reference = self.data_manager.preprocessed_reference is not None
        has_beads = self.data_manager.preprocessed_bead_stack is not None
        has_results = self.data_manager.displacement_results is not None  # Full results, not preview

        # Update action panel button states based on data availability
        if hasattr(self, 'action_panel'):
            # Analysis buttons require both reference and beads
            can_analyze = has_reference and has_beads
            self.action_panel.preview_btn.setEnabled(can_analyze)
            self.action_panel.calculate_btn.setEnabled(can_analyze)

            # Save requires full results (not just preview)
            self.action_panel.save_btn.setEnabled(has_results)

            # Load is always enabled as it's independent of current state
            self.action_panel.load_btn.setEnabled(True)

            # Cancel is always enabled
            self.action_panel.cancel_btn.setEnabled(True)

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze during processing."""
        if hasattr(self, 'data_panel'):
            self.data_panel.freeze_ui(frozen)

        if hasattr(self, 'parameter_panel'):
            self.parameter_panel.freeze_ui(frozen)

        if hasattr(self, 'action_panel'):
            # During processing, disable all buttons except cancel
            self.action_panel.preview_btn.setEnabled(not frozen)
            self.action_panel.calculate_btn.setEnabled(not frozen)
            self.action_panel.save_btn.setEnabled(not frozen)
            self.action_panel.load_btn.setEnabled(not frozen)
            # Cancel button always enabled
            self.action_panel.cancel_btn.setEnabled(True)

    def _has_required_data(self) -> bool:
        """Check if required data for processing is available."""
        return (
                self.data_manager.preprocessed_reference is not None and
                self.data_manager.preprocessed_bead_stack is not None
        )

    # endregion

    # region === Results Handling

    def _on_analysis_completed(self, results):
        """Handle completed analysis."""
        # Update action panel button states
        if self.action_panel:
            self.action_panel.update_button_states(
                has_reference=self.data_manager.preprocessed_reference is not None,
                has_beads=self.data_manager.preprocessed_bead_stack is not None,
                has_results=True
            )

        # Update colorbar
        if hasattr(results, 'parameters'):
            d_max = results.parameters.d_max
            self.colorbar_manager.update_limits(0, d_max)
        self.displacement_calculated.emit(results)

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        if self.action_panel:
            self.action_panel.update_button_states(
                has_reference=self.data_manager.preprocessed_reference is not None,
                has_beads=self.data_manager.preprocessed_bead_stack is not None,
                has_results=False
            )
        QMessageBox.critical(self, "Error", error_msg)

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.displacement_results is not None:
            self.visualization_manager.update_displacement_frame(
                self.viewer.dims.current_step[0]
            )

    # endregion

    # region === Cleanup
    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()

    # endregion
