from pathlib import Path
from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import Signal, Qt, QObject
from qtpy.QtWidgets import (
    QGroupBox, QLabel, QCheckBox, QSizePolicy, QFrame, QScrollArea, QApplication, QSpacerItem,
    QSpinBox, QDoubleSpinBox, QPushButton, QComboBox, QProgressBar, QFileDialog
)
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMessageBox

from backend.parameter_dataclasses import MSMParameters
from widgets._base_widget import BaseAnalysisWidget
from utilities.colorbar import ColorbarManager
from utilities.data_manager import DataManager
from utilities.parameter_manager import ParameterManager, ParameterCategory
from services.msm_service import MSMService, MSMResult
from utilities.visualization_manager import VisualizationManager


# TODO spinboxes should be initialized with default values from parameter manager
# TODO reimplement mask preview
# TODO make preview current frame run in seperate thread
# TODO layer visibility (only sigma_xx after calculations)
# TODO verify parameter synching
# TODO test in all widgets whether or not loading external data updates params
# TODO image layer should stay activated when creating several mask stacks in a row

def _is_valid_image_layer(layer) -> bool:
    """Check if layer is valid for mask creation/loading."""
    # Check if layer exists and has data
    if layer is None or not hasattr(layer, 'data'):
        return False

    # Check if it's a Vectors layer
    if layer.__class__.__name__ == 'Vectors':
        return False

    # Must be a numpy array
    return isinstance(layer.data, np.ndarray)


class MSMParameterPanel(QWidget):
    """Panel for handling all MSM parameter inputs."""

    parameter_changed = Signal()
    parameter_value_changed = Signal(str, object)

    MESH_ALGORITHMS = {
        "Frontal-Del.": "frontal-del",
        "Delaunay": "delaunay",
        "MeshAdapt": "meshadapt",
        "BAMG": "bamg",
        "FD Quads": "fd quads",
        "Para. Pack": "para pack"
    }

    def __init__(self, parameter_manager):
        super().__init__()
        self.parameter_manager = parameter_manager
        self.parameter_widgets = {}
        self._setup_ui()
        self._connect_signals()

        # Connect to parameter manager signals
        self.parameter_manager.parameter_changed.connect(self._sync_parameter)
        self.parameter_manager.parameters_reset.connect(self._on_parameters_reset)

        # Initial sync with parameter manager
        self._sync_widget_with_parameters()

    def _sync_parameter(self, param_name: str, value: Any):
        """Sync a single parameter from parameter manager."""
        if param_name in self.parameter_widgets:
            self._safe_set_value(self.parameter_widgets[param_name], value)

    def _safe_set_value(self, widget, value):
        """Safely set widget value."""
        if value is not None and widget is not None:
            widget.blockSignals(True)
            try:
                if isinstance(widget, QComboBox):
                    # Handle mesh algorithm combo box
                    for display_name, internal_name in self.MESH_ALGORITHMS.items():
                        if internal_name == value:
                            widget.setCurrentText(display_name)
                            break
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                else:  # SpinBox cases
                    value = max(widget.minimum(), min(widget.maximum(), value))
                    widget.setValue(value)
            except Exception as e:
                print(f"Error setting widget value: {str(e)}")
            widget.blockSignals(False)

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager."""
        self._block_widgets(True)
        try:
            for name, widget in self.parameter_widgets.items():
                value = self.parameter_manager.get_parameter(name)
                if value is not None:
                    self._safe_set_value(widget, value)
        finally:
            self._block_widgets(False)

    def _on_parameters_reset(self, category):
        """Handle parameter reset events."""
        if category == ParameterCategory.STRESS:
            self._sync_widget_with_parameters()

    def _block_widgets(self, block: bool):
        """Block or unblock all widget signals."""
        for widget in self.parameter_widgets.values():
            widget.blockSignals(block)


    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create parameter groups
        layout.addWidget(self._create_mask_parameters())
        layout.addWidget(self._create_mesh_parameters())
        layout.addWidget(self._create_material_parameters())
        layout.addWidget(self._create_visualization_parameters())

        self.setLayout(layout)

    def freeze_ui(self, freeze=True):
        """Disable/enable interactive elements in parameter panel"""
        for widget in self.parameter_widgets.values():
            widget.setEnabled(not freeze)

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
            if name == "density_factor":
                spin.setDecimals(3)
            else:
                spin.setDecimals(2)

        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(default)

        self.parameter_widgets[name] = spin
        layout.addWidget(spin)

        return layout

    def _create_mask_parameters(self) -> QGroupBox:
        """Create mask parameter group."""
        group = QGroupBox("Mask Parameters")
        layout = QVBoxLayout()

        params = [
            ("threshold", "Threshold:", 0, 100, 1, 0),
            ("dilation", "Dilation (px):", 0, 50, 1, 10),
            ("smoothing_sigma", "Smoothing:", 0, 40, 0.1, 10),
        ]

        for name, label, min_val, max_val, step, default in params:
            widget = self._create_parameter_widget(name, label, min_val, max_val, step, default)
            layout.addLayout(widget)

        group.setLayout(layout)
        return group

    def _create_mesh_parameters(self) -> QGroupBox:
        """Create mesh parameter group."""
        group = QGroupBox("Mesh Parameters")
        layout = QVBoxLayout()

        # Density factor
        density_layout = self._create_parameter_widget(
            "density_factor", "Density Factor:", 0.005, 0.05, 0.001, 0.025
        )
        layout.addLayout(density_layout)

        # Algorithm selector
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm:"))
        algo_combo = QComboBox()
        algo_combo.addItems(self.MESH_ALGORITHMS.keys())
        self.parameter_widgets["mesh_algorithm"] = algo_combo
        algo_layout.addWidget(algo_combo)
        layout.addLayout(algo_layout)

        # Optimization checkbox
        opt_layout = QHBoxLayout()
        opt_check = QCheckBox("Use Optimization")
        opt_check.setChecked(True)
        self.parameter_widgets["use_optimization"] = opt_check
        opt_layout.addWidget(opt_check)
        layout.addLayout(opt_layout)

        group.setLayout(layout)
        return group

    def _create_material_parameters(self) -> QGroupBox:
        """Create material parameter group."""
        group = QGroupBox("Material Parameters")
        layout = QVBoxLayout()

        poisson_layout = self._create_parameter_widget(
            "poisson_ratio_cells", "Poisson's Ratio:", 0, 0.5, 0.01, 0.5
        )
        layout.addLayout(poisson_layout)

        group.setLayout(layout)
        return group

    def _create_visualization_parameters(self) -> QGroupBox:
        """Create visualization parameter group."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        stress_layout = self._create_parameter_widget(
            "max_stress", "Max Stress (mN/m):", 0.01, 1000, 0.1, 1.0
        )
        layout.addLayout(stress_layout)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect widget signals to parameter manager."""
        for name, widget in self.parameter_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(
                    lambda value, n=name: self._on_value_changed(n, value)
                )
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(
                    lambda text, n=name: self._on_value_changed(
                        n, self.MESH_ALGORITHMS[text]
                    )
                )
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(
                    lambda state, n=name: self._on_value_changed(n, bool(state))
                )

        # Connect parameter manager changes back to widgets
        self.parameter_manager.parameter_changed.connect(self._update_widget_value)

    def _on_value_changed(self, param_name: str, value: Any):
        """Handle parameter value changes."""
        self.parameter_manager.set_parameter(param_name, value)
        self.parameter_value_changed.emit(param_name, value)
        self.parameter_changed.emit()

    def _update_widget_value(self, param_name: str, value: any):
        """Update widget when parameter changes externally."""
        if param_name in self.parameter_widgets:
            widget = self.parameter_widgets[param_name]
            widget.blockSignals(True)
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QComboBox):
                for display_name, internal_name in self.MESH_ALGORITHMS.items():
                    if internal_name == value:
                        widget.setCurrentText(display_name)
                        break
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            widget.blockSignals(False)


class MSMDataPanel(QWidget):
    """Panel for handling data loading and status display in a unified group."""

    data_loaded = Signal(str)  # Emits data type that was loaded ('force' or 'mask')
    force_data_loaded = Signal(object)
    mask_data_loaded = Signal(object)
    data_load_failed = Signal(str)

    def __init__(self, data_manager, viewer):
        super().__init__()
        self.data_manager = data_manager
        self.viewer = viewer
        self.controller = None
        self._setup_ui()

    def freeze_ui(self, freeze=True):
        """Disable/enable interactive elements in data panel"""
        self.load_force_btn.setEnabled(not freeze)
        self.load_mask_btn.setEnabled(not freeze)

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create unified data input group
        data_group = QGroupBox("Data Input")
        group_layout = QVBoxLayout()
        data_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Force data row
        force_layout = QHBoxLayout()
        self.load_force_btn = QPushButton("Load Forces")
        self.load_force_btn.setFixedWidth(150)
        self.load_force_btn.setFixedHeight(25)
        self.force_status = QLabel("Not loaded")
        force_layout.addWidget(self.load_force_btn)
        force_layout.addWidget(self.force_status)
        group_layout.addLayout(force_layout)

        # Mask data row
        mask_layout = QHBoxLayout()
        self.load_mask_btn = QPushButton("Load Masks")
        self.load_mask_btn.setFixedWidth(150)
        self.load_mask_btn.setFixedHeight(25)
        self.mask_status = QLabel("Not loaded")
        mask_layout.addWidget(self.load_mask_btn)
        mask_layout.addWidget(self.mask_status)
        group_layout.addLayout(mask_layout)

        data_group.setLayout(group_layout)

        # Add description label for required data
        info_label = QLabel(
            "Required: Force field data and tissue mask."
        )
        info_label.setWordWrap(True)
        group_layout.addWidget(info_label)

        layout.addWidget(data_group)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect UI signals to controller methods."""
        self.load_force_btn.clicked.connect(self._load_force_data)
        self.load_mask_btn.clicked.connect(self._handle_load_mask_button)

    def update_button_states(self, active_layer):
        """Update button states based on current conditions."""
        # Update Load Masks button
        is_valid_layer = _is_valid_image_layer(active_layer)
        self.load_mask_btn.setEnabled(is_valid_layer)

        if active_layer is None:
            self.load_mask_btn.setToolTip("Select an image layer first")
        elif not is_valid_layer:
            if active_layer.__class__.__name__ == 'Vectors':
                self.load_mask_btn.setToolTip("Cannot create masks from vector layers")
            else:
                self.load_mask_btn.setToolTip("Selected layer is not a valid image layer")
        else:
            self.load_mask_btn.setToolTip("Load masks from selected layer")

        # Force button always enabled with appropriate tooltip
        self.load_force_btn.setEnabled(True)
        self.load_force_btn.setToolTip("Load force data from file")

    def update_data_status(self):
        """Update both force and mask status labels based on data manager state."""
        # Update force status
        force_results = self.data_manager.force_results
        if force_results is not None:
            force_shape = force_results.force_field.shape
            self.force_status.setText(f"Loaded: {force_shape}")
        else:
            self.force_status.setText("Not loaded")

        # Update mask status - check mask_stack only
        masks = self.data_manager.mask_stack
        if masks is not None:
            mask_shape = masks.shape if isinstance(masks, np.ndarray) else "unknown"
            self.mask_status.setText(f"Loaded: {mask_shape}")
        else:
            self.mask_status.setText("Not loaded")

    def update_mask_status(self, status_text: str):
        """Update the mask status label."""
        self.mask_status.setText(status_text)

    def _get_active_layer(self):
        """Get the currently active napari layer."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(
                self,
                "No Layer Selected",
                "Please select an image layer to create masks from."
            )
            return None
        return active_layer

    def _get_active_layer_data(self):
        """Get the data from currently active napari layer."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(
                self,
                "No Layer Selected",
                "Please select an image layer to create masks from."
            )
            return None

        # Get the actual numpy array from the layer
        if hasattr(active_layer, 'data') and isinstance(active_layer.data, np.ndarray):
            return active_layer.data
        return None

    def _handle_load_mask_button(self):
        """Handle the Load Masks button click by getting data from active layer."""
        mask_data = self._get_active_layer_data()
        if mask_data is not None:
            self._load_mask_data(mask_data)
            # Let the controller handle visualization
            if self.controller:
                self.controller.mask_creation_completed.emit()

    def set_controller(self, controller):
        """Set the controller and connect signals."""
        self.controller = controller
        self._connect_signals()

    def _load_force_data(self):
        """Load force data from files."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Force Data File", "", "NumPy Files (*.npy)"
            )

            if file_path:
                # Load force data
                force_data = np.load(file_path, allow_pickle=True).item()

                # Update parameters if they exist in the loaded data
                if hasattr(force_data, 'parameters'):
                    # Block parameter change signals temporarily
                    if self.controller.parameter_panel:
                        self.controller.parameter_panel._block_widgets(True)
                    try:
                        # Update parameter manager with loaded parameters
                        params = force_data.parameters
                        for param_name, value in vars(params).items():
                            if param_name != '_sa_instance_state':  # Skip SQLAlchemy state
                                if param_name == 'gel_height':
                                    # Handle infinity case
                                    if value == 0:
                                        value = float('inf')
                                self.controller.parameter_manager.set_parameter(param_name, value)

                        # Sync UI with new parameters
                        if self.controller.parameter_panel:
                            self.controller.parameter_panel._sync_widget_with_parameters()
                    finally:
                        if self.controller.parameter_panel:
                            self.controller.parameter_panel._block_widgets(False)

                # Update data manager
                self.data_manager.set_force_results(force_data)

                # Update visualization using visualization manager
                self.controller.visualization_manager.visualize_force_results()

                # Manage layer visibility
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

                self.force_status.setText(f"Loaded: {force_data.force_field.shape}")
                self.data_loaded.emit('force')
                self.force_data_loaded.emit(force_data)

        except Exception as e:
            error_msg = f"Failed to load force data: {str(e)}"
            self.force_status.setText("Error loading")
            QMessageBox.critical(self, "Error", error_msg)

    def load_mask_data(self, mask_data):
        """Public method to load mask data directly."""
        self._load_mask_data(mask_data=mask_data)

    def _load_mask_data(self, mask_data=None):
        """Handle mask loading from either active layer or provided data."""
        try:
            if mask_data is None:
                # Get active layer if no mask_data provided
                active_layer = self._get_active_layer()
                if active_layer is None:
                    return
                mask_data = active_layer.data
            else:
                # Validate provided mask_data
                if not isinstance(mask_data, np.ndarray):
                    raise ValueError("Provided mask data is not a numpy array")

            # Check for force data presence
            force_results = self.data_manager.force_results
            force_field = None
            if force_results is not None:
                force_field = force_results.force_field
            if force_field is None:
                QMessageBox.warning(
                    self,
                    "No Force Data",
                    "No force data loaded. Masks may need resizing later."
                )

            # Process masks (resizing if needed)
            processed_masks, warnings = self.controller.service.process_mask_data(
                mask_data,
                force_field
            )

            # Show processing warnings
            for warning in warnings:
                QMessageBox.warning(self, "Warning", warning)

            # Update mask data in data manager
            self.data_manager.set_mask_stack(processed_masks)

            # Update status immediately
            self.update_data_status()

            # Emit signal that mask data was loaded
            self.data_loaded.emit('mask')
            self.mask_data_loaded.emit(processed_masks)

        except Exception as e:
            error_msg = f"Failed to load mask data: {str(e)}"
            self.mask_status.setText("Error loading")
            QMessageBox.critical(self, "Error", error_msg)
            self.data_load_failed.emit(error_msg)


class MSMActionPanel(QWidget):
    """Panel for analysis actions and progress display."""

    def __init__(self, msm_controller):
        super().__init__()
        self.controller = msm_controller
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create grid of button pairs
        button_layout = QVBoxLayout()

        # Row 1: Create Masks and Preview Mesh
        row1_layout = QHBoxLayout()
        self.create_mask_btn = QPushButton("Create Masks from Image")
        self.preview_mesh_btn = QPushButton("Preview Mesh")
        row1_layout.addWidget(self.create_mask_btn)
        row1_layout.addWidget(self.preview_mesh_btn)
        button_layout.addLayout(row1_layout)

        # Row 2: Preview Frame and Calculate Stress
        row2_layout = QHBoxLayout()
        self.preview_frame_btn = QPushButton("Preview Current Frame")
        self.analyze_btn = QPushButton("Calculate Stress Tensors")
        row2_layout.addWidget(self.preview_frame_btn)
        row2_layout.addWidget(self.analyze_btn)
        button_layout.addLayout(row2_layout)

        # Row 3: Save and Load Stress
        row3_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Stress Tensors")
        self.load_stress_btn = QPushButton("Load Stress Tensors")
        row3_layout.addWidget(self.save_btn)
        row3_layout.addWidget(self.load_stress_btn)
        button_layout.addLayout(row3_layout)

        layout.addLayout(button_layout)

        # Cancel button (full width)
        self.cancel_btn = QPushButton("Cancel All Operations")
        layout.addWidget(self.cancel_btn)

        self.setLayout(layout)

    def freeze_ui(self, freeze=True):
        """Disable/enable action buttons (keep cancel enabled)"""
        buttons = [
            self.preview_mesh_btn, self.preview_frame_btn,
            self.analyze_btn, self.save_btn, self.create_mask_btn,
            self.load_stress_btn
        ]
        for btn in buttons:
            btn.setEnabled(not freeze)
        self.cancel_btn.setEnabled(True)  # Always keep cancel enabled

    def update_button_states(self, active_layer: bool = None, force_data: bool = False,
                             mask_data: bool = False, stress_data: bool = False):
        """Update button states based on current data availability."""
        # Ensure all parameters are boolean, defaulting to False if None
        force_data = bool(force_data)
        mask_data = bool(mask_data)
        stress_data = bool(stress_data)

        # Create Masks button - check if layer is valid for mask creation
        is_valid_layer = _is_valid_image_layer(active_layer)
        self.create_mask_btn.setEnabled(is_valid_layer)

        if active_layer is None:
            self.create_mask_btn.setToolTip("Select an image layer first")
        elif not is_valid_layer:
            if active_layer.__class__.__name__ == 'Vectors':
                self.create_mask_btn.setToolTip("Cannot create masks from vector layers")
            else:
                self.create_mask_btn.setToolTip("Selected layer is not a valid image layer")
        else:
            self.create_mask_btn.setToolTip("Create masks from selected image")

        # Preview Mesh button
        self.preview_mesh_btn.setEnabled(mask_data)
        if not mask_data:
            self.preview_mesh_btn.setToolTip("Load masks first")
        else:
            self.preview_mesh_btn.setToolTip(
                "Preview mesh (Warning: No force data present)" if not force_data
                else "Preview mesh"
            )

        # Preview Frame and Analysis buttons
        for btn in [self.preview_frame_btn, self.analyze_btn]:
            btn.setEnabled(mask_data and force_data)
            btn.setToolTip(
                "Load both mask and force data first" if not (mask_data and force_data)
                else btn.text()
            )

        # Load Stress button is always enabled
        self.load_stress_btn.setEnabled(True)
        self.load_stress_btn.setToolTip("Load pre-calculated stress tensor results")

        # Save Results button
        self.save_btn.setEnabled(stress_data)
        self.save_btn.setToolTip(
            "Calculate stress tensors first" if not stress_data
            else "Save results"
        )

    def _connect_signals(self):
        """Connect action panel buttons to controller methods."""
        # Connect buttons to controller methods
        self.preview_mesh_btn.clicked.connect(self.controller.preview_mesh)
        self.preview_frame_btn.clicked.connect(self.controller.preview_current_frame)
        self.analyze_btn.clicked.connect(self._handle_analyze_click)
        self.save_btn.clicked.connect(self.controller.save_results)
        self.load_stress_btn.clicked.connect(self.controller.load_results)
        self.create_mask_btn.clicked.connect(self.controller.create_masks_from_images)
        self.cancel_btn.clicked.connect(self.controller.cancel_all_operations)

        # Listen to controller signals for data updates
        self.controller.data_updated.connect(self._update_button_states)

    def _update_button_states(self, data_type=None):
        """Update button states based on data availability."""
        has_force_data = self.controller.data_manager.force_field is not None
        has_mask_data = self.controller.data_manager.masks is not None
        has_stress_data = self.controller.data_manager.stress_tensor is not None

        # Enable/disable buttons based on data availability
        self.preview_mesh_btn.setEnabled(has_force_data and has_mask_data)
        self.preview_frame_btn.setEnabled(has_force_data and has_mask_data)
        self.analyze_btn.setEnabled(has_force_data and has_mask_data)
        self.save_btn.setEnabled(has_stress_data)

    def _handle_analyze_click(self):
        """Handle analyze button click by disabling buttons and starting analysis."""
        self.set_buttons_enabled(False)
        self.controller.start_analysis()

    def update_progress(self, progress: int, status: str):
        """Update progress bar and status label."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)

    def set_buttons_enabled(self, enabled: bool):
        """Enable/disable all action buttons."""
        for btn in [self.preview_mesh_btn, self.preview_frame_btn,
                    self.analyze_btn, self.save_btn]:
            btn.setEnabled(enabled)


class MSMController(QObject):
    """Coordinates interactions between UI components, service, and managers."""

    # Define signals as class attributes
    data_updated = Signal(str)
    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)  # Results object
    analysis_failed = Signal(str)  # Error message
    mask_creation_progress = Signal(int, str)  # (progress, message)
    mask_creation_completed = Signal()
    mask_creation_failed = Signal(str)
    ui_frozen = Signal(bool)

    def __init__(self, viewer: Viewer, service: MSMService,
                 data_manager: DataManager, parameter_manager: ParameterManager,
                 visualization_manager: VisualizationManager, data_panel: MSMDataPanel):
        super().__init__()
        self.viewer = viewer
        self.service = service
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager
        self.data_panel = data_panel
        self.active_workers = []

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.mask_stack is None:
            QMessageBox.warning(None, "Warning", "No mask loaded. Please load a mask first.")
            return False
        if self.data_manager.force_results is None:
            QMessageBox.warning(None, "Warning", "No force data available. Please calculate forces first.")
            return False
        return True

    def start_analysis(self):
        """Start the stress analysis for all frames."""
        try:
            if not self._validate_prerequisites():
                return

            self.analysis_started.emit()
            self._update_progress(0, "Starting analysis...")

            # Get current parameters and update service
            params = self._get_current_parameters()
            self.service.update_parameters(params)

            # Get mask and force data
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            force_results = self.data_manager.force_results
            if force_results is None:
                raise ValueError("No force data available")

            # First generate all meshes in the main thread
            print("Generating meshes for all frames...")
            mesh_generator = self.service.generate_mesh_stack(masks)
            mesh_data = []

            # Process mesh generation results in main thread
            try:
                while True:
                    nodes, elements, quality_metrics, frame, total = next(mesh_generator)
                    mesh_data.append((nodes, elements, quality_metrics))
                    self._update_progress(
                        int((frame + 1) / total * 45),  # Use first 45% for mesh generation
                        f"Generating mesh: Frame {frame + 1}/{total}"
                    )
                    # Keep UI responsive
                    QApplication.processEvents()
            except StopIteration as e:
                # Get the final mesh results if returned
                final_mesh_results = e.value
                if final_mesh_results:
                    mesh_data = final_mesh_results

            # Update status for numba compilation
            self._update_progress(45, "Compiling numerical methods (may take a moment on first run)...")
            QApplication.processEvents()

            # Now calculate stress using the pre-generated meshes
            @thread_worker
            def stress_calculation_worker():
                try:
                    # Get stress generator with pre-generated meshes
                    stress_generator = self.service.calculate_stresses(
                        force_field=force_results.force_field,
                        masks=masks,
                        mesh_data=mesh_data
                    )

                    while True:
                        result, current_frame, total_frames = next(stress_generator)
                        progress = 50 + int((current_frame + 1) / total_frames * 50)  # Start from 50%
                        yield (progress, f"Calculating stress: Frame {current_frame + 1}/{total_frames}", result)

                except StopIteration as e:
                    return e.value

            worker = stress_calculation_worker()
            worker.running = True
            self.active_workers.append(worker)

            def on_yielded(data):
                progress, status, result = data
                self._update_progress(progress, status)
                if progress == 51:  # First frame after mesh generation
                    # Update visualization for first frame
                    self.visualization_manager.visualize_stress_preview(
                        result.stress_tensor,
                        max_stress=params.max_stress,
                        downscale_factor=force_results.parameters.downscale_factor
                    )

            def on_returned(final_result):
                # Store the complete results
                self.data_manager.set_stress_results(final_result)

                # Update visualization with all frames
                self.visualization_manager.visualize_stress_results()

                # Update status with final metrics
                status_msg = (
                    f"Analysis completed successfully\n"
                    f"Average condition number: {final_result.condition_number:.1e}\n"
                    f"Average residual: {final_result.residual:.1e}"
                )
                self._update_progress(100, status_msg)

                self.analysis_completed.emit(final_result)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            def on_errored(error):
                error_msg = f"Analysis failed: {str(error)}"
                self._update_progress(0, error_msg)
                self.analysis_failed.emit(error_msg)
                QMessageBox.critical(None, "Error", error_msg)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            # Connect worker signals
            worker.yielded.connect(on_yielded)
            worker.returned.connect(on_returned)
            worker.errored.connect(on_errored)
            worker.start()

        except Exception as e:
            error_msg = f"Failed to start analysis: {str(e)}"
            self._update_progress(0, error_msg)
            self.analysis_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def preview_current_frame(self):
        """Calculate and display stress field for current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self._update_progress(0, "Generating stress preview...")

            # Get current frame index
            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            # Get current parameters and update service
            params = self._get_current_parameters()
            self.service.update_parameters(params)

            # Get mask and force data for current frame
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            force_results = self.data_manager.force_results
            if force_results is None:
                raise ValueError("No force data available")

            # Extract current frame data
            mask = masks[current_frame] if masks.ndim > 2 else masks
            force_field = force_results.force_field[current_frame] if force_results.force_field.ndim > 3 else force_results.force_field

            # Calculate stress field for current frame
            stress_generator = self.service.calculate_stresses(
                force_field=force_field[np.newaxis, ...],
                masks=mask[np.newaxis, ...],
            )

            try:
                # Get stress results
                result, _, _ = next(stress_generator)

                # Update visualization
                self.visualization_manager.visualize_stress_preview(
                    result.stress_tensor,
                    max_stress=params.max_stress,
                    downscale_factor=force_results.parameters.downscale_factor
                )

                # Create status message with key metrics
                status_msg = (
                    f"Stress preview generated for frame {current_frame}\n"
                    f"Condition number: {result.condition_number:.1e}\n"
                    f"Residual: {result.residual:.1e}"
                )

                self._update_progress(100, status_msg)
                return result

            except StopIteration:
                raise ValueError("Stress calculation failed to produce results")

        except Exception as e:
            error_msg = f"Failed to preview stress field: {str(e)}"
            self._update_progress(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def _get_current_parameters(self) -> MSMParameters:
        """Get current MSM parameters from parameter manager."""
        return self.parameter_manager.get_msm_parameters()

    def _start_stress_calculation(self, mesh_results, params):
        """Start thread worker for stress calculation."""

        @thread_worker
        def stress_calculation_worker():
            # Get masks and force field from displacement results
            masks = self.data_manager.masks
            force_results = self.data_manager.force_results
            if force_results is None:
                raise ValueError("No force results available")

            force_field = force_results.force_field
            total_frames = force_field.shape[0]

            # Initialize stress generator with explicit mask passing
            stress_generator = self.service.calculate_stresses(
                force_field=force_field,
                masks=masks,
                mesh_data=mesh_results
            )

            try:
                while True:
                    result, current_frame, total_frames = next(stress_generator)
                    progress = int((current_frame + 1) / total_frames * 100)
                    yield (progress, f"Calculating stress: Frame {current_frame + 1}/{total_frames}")
            except StopIteration as e:
                return e.value

        worker = stress_calculation_worker()
        worker.running = True
        self.active_workers.append(worker)

        def on_returned(results):
            # Create MSMResult object
            stress_result = MSMResult(
                stress_tensor=results.stress_tensor,
                nodes=results.nodes,
                elements=results.elements,
                parameters=params,
                condition_number=results.condition_number,
                residual=results.residual
            )

            # Update data manager with results
            self.data_manager.set_stress_results(stress_result)

            # Update visualization
            self.visualization_manager.visualize_stress_results(
                stress_result,
                max_stress=params.max_stress
            )

            self._update_progress(100, "Analysis completed successfully")
            self.analysis_completed.emit(results)
            self.active_workers.remove(worker)
            if not self.active_workers:
                self.unfreeze_ui()

        # Connect worker signals...
        worker.yielded.connect(lambda x: self._update_progress(*x))
        worker.returned.connect(on_returned)
        worker.errored.connect(self._handle_worker_error)
        worker.start()

    def save_results(self):
        """Save analysis results to file."""
        try:
            stress_results = self.data_manager.stress_results
            if stress_results is None:
                raise ValueError("No stress tensor data to save")

            file_path, _ = QFileDialog.getSaveFileName(
                None, "Save Stress Tensor Data", "", "NumPy Files (*.npy)"
            )

            if file_path:
                if not file_path.endswith('.npy'):
                    file_path += '.npy'

                # Save the complete MSMResult object
                np.save(file_path, stress_results)
                return True

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save results: {str(e)}")
        return False

    def load_results(self):
        """Load previously saved stress results."""
        try:
            load_path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Stress Results",
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
                            if param_name != '_sa_instance_state' and param_name != "young_modulus":  # Skip SQLAlchemy state
                                self.parameter_manager.set_parameter(param_name, value)

                        # Sync UI with new parameters
                        if self.parameter_panel:
                            self.parameter_panel._sync_widget_with_parameters()
                    finally:
                        if self.parameter_panel:
                            self.parameter_panel._block_widgets(False)

                # Update data manager and visualization
                self.data_manager.set_stress_results(results)
                self.visualization_manager.visualize_stress_results()

                # First pass: find the stress layer and disable all others
                for layer in self.viewer.layers:
                    if layer.name == 'Average Normal Stress':
                        layer.visible = True
                    else:
                        layer.visible = False

                # Move stress layers to top if they exist
                for layer_name in ['Normal Stress XX', 'Normal Stress YY', 'Average Normal Stress']:
                    layer = next((layer for layer in self.viewer.layers if layer.name == layer_name), None)
                    if layer is not None:
                        current_index = self.viewer.layers.index(layer)
                        # Keep XX and YY below Average
                        if layer_name == 'Average Normal Stress':
                            self.viewer.layers.move(current_index, -1)
                        else:
                            self.viewer.layers.move(current_index, -2)

                self.progress_updated.emit(100, f"Results and parameters loaded from {load_path}")
                self.analysis_completed.emit(results)

        except Exception as e:
            error_msg = f"Failed to load results: {str(e)}"
            self.progress_updated.emit(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)

    def set_panels(self, parameter_panel: 'MSMParameterPanel', action_panel: 'MSMActionPanel'):
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

    def _update_progress(self, progress: int, status: str):
        """Update progress and emit signal."""
        self.progress_updated.emit(progress, status)

    def create_masks_from_images(self):
        """Handle mask creation from the active image layer."""
        try:
            active_layer = self.viewer.layers.selection.active
            if not active_layer or not isinstance(active_layer.data, np.ndarray):
                raise ValueError("No valid image layer selected.")

            image_data = active_layer.data
            params = self._get_current_parameters()

            @thread_worker
            def mask_creation_worker():
                mask_generator = self.service.create_mask_stack(image_data, params)
                total_frames = image_data.shape[0] if image_data.ndim > 2 else 1

                try:
                    while True:
                        _, current_frame, total_frames = next(mask_generator)
                        progress = int((current_frame + 1) / total_frames * 100)
                        yield (progress, f"Creating masks: Frame {current_frame + 1}/{total_frames}")
                except StopIteration as e:
                    analysis_stack = e.value
                    return analysis_stack

            worker = mask_creation_worker()
            worker.running = True
            self.active_workers.append(worker)
            if len(self.active_workers) >= 1:
                self.freeze_ui()

            def on_yielded(progress_data):
                progress, message = progress_data
                self.mask_creation_progress.emit(progress, message)

            def on_returned(analysis_stack):
                # Pass raw masks to data panel for processing
                self.data_panel.load_mask_data(analysis_stack)
                self.mask_creation_completed.emit()
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            def on_errored(exc):
                error_msg = f"Mask creation failed: {str(exc)}"
                self.mask_creation_progress.emit(0, error_msg)
                self.mask_creation_failed.emit(error_msg)
                QMessageBox.critical(None, "Error", error_msg)
                self.active_workers.remove(worker)
                if not self.active_workers:
                    self.unfreeze_ui()

            worker.yielded.connect(on_yielded)
            worker.returned.connect(on_returned)
            worker.errored.connect(on_errored)
            worker.start()

        except Exception as e:
            error_msg = f"Failed to start mask creation: {str(e)}"
            self.mask_creation_progress.emit(0, error_msg)
            self.mask_creation_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)

    def _generate_mesh_stack(self, masks, params):
        """Generate mesh stack in the main thread."""
        try:
            mesh_generator = self.service.generate_mesh_stack(masks, params)
            total_frames = masks.shape[0]
            mesh_results = []

            while True:
                try:
                    # Process next mesh
                    mesh_result, _, _, current_frame, total_frames = next(mesh_generator)
                    mesh_results.append(mesh_result)

                    # Update progress
                    progress = int((current_frame + 1) / total_frames * 100)
                    self._update_progress(progress, f"Generating mesh: Frame {current_frame + 1}/{total_frames}")

                    # Process UI events to keep the interface responsive
                    QApplication.processEvents()

                except StopIteration as e:
                    # Generator completed, return final results
                    return e.value

        except Exception as e:
            raise Exception(f"Mesh generation failed: {str(e)}")

    def _handle_progress(self, current: int, total: int, status: str):
        """Handle progress updates during analysis."""
        progress = int((current + 1) / total * 100)
        self._update_progress(progress, status)

    def preview_mesh(self):
        """Generate and display mesh preview for the current frame."""
        try:
            self._update_progress(0, "Generating mesh preview...")

            # Get current frame index
            if len(self.viewer.dims.current_step) == 2:
                current_frame = 0
                self.progress_updated.emit(0, "No image stack found, previewing frame 0")
            else:
                current_frame = self.viewer.dims.current_step[0]

            # Get current parameters and update service
            params = self._get_current_parameters()
            self.service.update_parameters(params)

            # Get mask for current frame
            masks = self.data_manager.mask_stack
            if masks is None:
                raise ValueError("No mask data available")

            mask = masks[current_frame] if masks.ndim > 2 else masks

            # Generate mesh using service
            mesh_generator = self.service.generate_mesh_stack(mask)

            try:
                # Get first mesh result
                nodes, elements, quality_metrics, _, _ = next(mesh_generator)

                # Get downscale factor from force results if available
                if self.data_manager.force_results is not None:
                    downscale_factor = self.data_manager.force_results.parameters.downscale_factor
                else:
                    downscale_factor = 1

                # Update visualization
                self.visualization_manager.visualize_mesh(
                    nodes=nodes,
                    elements=elements,
                    downscale_factor=downscale_factor
                )

                # Simplified quality metrics message
                num_elements = len(elements)
                mean_quality = quality_metrics.get('mean_quality', 0.0)
                status_msg = (
                    f"Mesh preview generated for frame {current_frame}\n"
                    f"Elements: {num_elements}\n"
                    f"Mean quality: {mean_quality:.3f}"
                )

                self._update_progress(100, status_msg)

                return nodes, elements, quality_metrics

            except StopIteration:
                raise ValueError("Mesh generation failed to produce results")

        except Exception as e:
            error_msg = f"Failed to preview mesh: {str(e)}"
            self._update_progress(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def cancel_all_operations(self):
        """Cancel all running background operations"""
        for worker in self.active_workers:
            try:
                worker.running = False  # Set cancellation flag
                worker.quit()
                worker.wait(500)  # Wait up to 500ms
                if worker.isRunning():
                    worker.terminate()
                worker.deleteLater()
            except Exception as e:
                pass
        self.active_workers.clear()
        # Update UI status
        self.mask_creation_progress.emit(0, "Operations cancelled")
        self.progress_updated.emit(0, "Operations cancelled")
        QApplication.processEvents()
        self.unfreeze_ui()


class MSMWidget(BaseAnalysisWidget):
    """Widget for Monolayer Stress Microscopy analysis."""
    stress_calculated = Signal(object)  # Emits stress analysis results

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

        # Get initial parameters from parameter manager
        self.msm_params = parameter_manager.get_msm_parameters()

        # Initialize service with parameters
        self.service = MSMService(self.msm_params)
        self.colorbar_manager = ColorbarManager()

        # Initialize panels
        self.data_panel = MSMDataPanel(data_manager, viewer)
        self.parameter_panel = MSMParameterPanel(parameter_manager)

        # Initialize controller
        self.controller = MSMController(
            viewer=viewer,
            service=self.service,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager,
            data_panel=self.data_panel
        )

        # Initialize action panel with controller
        self.action_panel = MSMActionPanel(self.controller)

        # Set panels in controller
        self.controller.set_panels(self.parameter_panel, self.action_panel)
        self.data_panel.set_controller(self.controller)

        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()

        # Monitor frame changes
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

        # Connect parameter manager to update service parameters when they change
        parameter_manager.parameters_reset.connect(self._update_service_parameters)
        parameter_manager.parameter_changed.connect(self._handle_parameter_change)

        self.controller.unfreeze_ui()

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
            colormap_name='seismic',
            label="Stress (mN/m)",
            clim=(-self.parameter_manager.get_parameter('max_stress'),
                  self.parameter_manager.get_parameter('max_stress')),
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
        # Connect progress updates from controller to status panel
        self.controller.progress_updated.connect(self._update_status)

        # Connect data panel signals
        self.data_panel.data_loaded.connect(self._on_data_loaded)

        # Connect mask loading/creation signals
        self.controller.mask_creation_progress.connect(self._update_status)
        self.controller.mask_creation_completed.connect(self._on_mask_creation_completed)
        self.controller.mask_creation_failed.connect(self._on_mask_creation_failed)

        # Connect parameter panel changes
        self.parameter_panel.parameter_changed.connect(self._on_parameter_changed)

        # Connect controller signals
        self.controller.analysis_started.connect(self._on_analysis_started)
        self.controller.analysis_completed.connect(self._on_analysis_completed)
        self.controller.analysis_failed.connect(self._on_analysis_failed)

        self.controller.ui_frozen.connect(self._handle_ui_freeze)

        # Connect to layer selection changes
        self.viewer.layers.selection.events.active.connect(self._update_ui_state)

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if self.data_manager.stress_results is not None:
            self.visualization_manager.update_stress_frame(
                self.viewer.dims.current_step[0]
            )

    def _update_service_parameters(self, category: ParameterCategory):
        """Update service parameters when parameters are reset."""
        if category == ParameterCategory.STRESS:
            self.msm_params = self.parameter_manager.get_msm_parameters()
            self.service.update_parameters(self.msm_params)

    def _handle_parameter_change(self, param_name: str, value: Any):
        """Update service parameters when individual parameters change."""
        stress_params = {
            'threshold', 'dilation', 'smoothing_sigma', 'density_factor',
            'mesh_algorithm', 'use_optimization', 'poisson_ratio_cells',
            'max_stress'
        }
        if param_name in stress_params:
            self.msm_params = self.parameter_manager.get_msm_parameters()
            self.service.update_parameters(self.msm_params)

    def _update_ui_state(self, event=None):
        """Update UI state based on current data and selection."""
        active_layer = self.viewer.layers.selection.active

        has_force = self.data_manager.force_results is not None
        has_mask = self.data_manager.mask_stack is not None
        has_stress = self.data_manager.stress_results is not None

        self.data_panel.update_button_states(active_layer)
        self.data_panel.update_data_status()
        self.action_panel.update_button_states(
            active_layer=active_layer,
            force_data=has_force,
            mask_data=has_mask,
            stress_data=has_stress
        )

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        super().cleanup()

    def _handle_ui_freeze(self, frozen: bool):
        """Handle UI freeze/unfreeze events."""
        if not frozen:
            self._update_ui_state()

    def _on_data_loaded(self, data_type: str):
        """Handle data loading events."""
        self._update_ui_state()

    def _on_parameter_changed(self):
        """Handle parameter change events."""
        if hasattr(self, 'preview_active') and self.preview_active:
            self.controller.preview_current_frame()

    def _on_analysis_started(self):
        """Handle analysis start event."""
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis started...")

    def _on_analysis_completed(self, results):
        """Handle analysis completion."""
        self.stress_calculated.emit(results)
        self._update_ui_state()

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._update_status(0, f"Analysis failed: {error_msg}")
        self._update_ui_state()

    def _on_mask_creation_completed(self):
        """Handle mask creation completion."""
        masks = self.data_manager.mask_stack
        if masks is not None:
            self.data_panel.update_mask_status(f"Loaded: {masks.shape}")

            downscale_factor = 1
            if self.data_manager.force_results is not None:
                downscale_factor = self.data_manager.force_results.parameters.downscale_factor

            self.visualization_manager.visualize_masks(
                masks=masks,
                downscale_factor=downscale_factor
            )
        self._update_ui_state()

    def _on_mask_creation_failed(self, error_msg: str):
        """Handle mask creation failure."""
        self.data_panel.update_mask_status("Mask creation failed")
        QMessageBox.critical(self, "Error", error_msg)
        self._update_ui_state()
