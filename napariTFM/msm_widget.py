from typing import Any

import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import Signal, Qt, QObject
from qtpy.QtWidgets import (
    QGroupBox, QLabel, QCheckBox, QSizePolicy, QFrame, QScrollArea,
    QSpinBox, QDoubleSpinBox, QPushButton, QComboBox, QProgressBar, QFileDialog
)
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMessageBox

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.data_manager import DataManager
from napariTFM.parameter_manager import ParameterManager
from napariTFM.services.msm_service import MSMParameters, MSMService
from napariTFM.visualization_manager import VisualizationManager


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

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Create parameter groups
        layout.addWidget(self._create_mask_parameters())
        layout.addWidget(self._create_mesh_parameters())
        layout.addWidget(self._create_material_parameters())
        layout.addWidget(self._create_visualization_parameters())

        layout.addStretch()
        self.setLayout(layout)

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
        self.parameter_widgets["algorithm"] = algo_combo
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
        self.parameter_manager.set_value(param_name, value)
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
    """Panel for handling data loading and status display."""

    # Define signals
    data_loaded = Signal(str)  # Emits data type that was loaded ('force' or 'mask')
    force_data_loaded = Signal(object)  # Emits force data
    mask_data_loaded = Signal(object)  # Emits mask data
    data_load_failed = Signal(str)  # Emits error message

    def __init__(self, data_manager, viewer):
        super().__init__()
        self.data_manager = data_manager
        self.viewer = viewer
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Force data section
        force_group = QGroupBox("Force Data")
        force_layout = QHBoxLayout()
        self.load_force_btn = QPushButton("Load Forces")
        self.force_status = QLabel("Not loaded")
        force_layout.addWidget(self.load_force_btn)
        force_layout.addWidget(self.force_status)
        force_group.setLayout(force_layout)

        # Mask data section
        mask_group = QGroupBox("Mask Data")
        mask_layout = QVBoxLayout()

        # Top row for buttons
        button_layout = QHBoxLayout()
        self.load_mask_btn = QPushButton("Load Masks")
        button_layout.addWidget(self.load_mask_btn)

        # Bottom row for status
        status_layout = QHBoxLayout()
        self.mask_status = QLabel("Not loaded")
        status_layout.addWidget(self.mask_status)

        mask_layout.addLayout(button_layout)
        mask_layout.addLayout(status_layout)
        mask_group.setLayout(mask_layout)

        layout.addWidget(force_group)
        layout.addWidget(mask_group)
        self.setLayout(layout)

    def _connect_signals(self):
        self.load_force_btn.clicked.connect(self._load_force_data)
        self.load_mask_btn.clicked.connect(self._load_mask_data)

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

    def _load_force_data(self):
        """Load force data from files."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Force Data File", "", "NumPy Files (*.npy)"
            )
            if file_path:
                force_data = np.load(file_path, allow_pickle=True).item()
                self.data_manager.set_force_results(
                    force_data["force_field"],
                    force_data["parameters"]
                )
                self.force_status.setText(f"Loaded: {force_data['force_field'].shape}")
                self.data_loaded.emit('force')
                self.force_data_loaded.emit(force_data)
        except Exception as e:
            error_msg = f"Failed to load force data: {str(e)}"
            self.force_status.setText("Error loading")
            self.data_load_failed.emit(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _load_mask_data(self):
        """Load mask data from active layer."""
        try:
            # Implementation of mask loading
            active_layer = self._get_active_layer()
            if active_layer is not None:
                mask_data = self._process_mask_layer(active_layer)
                if mask_data is not None:
                    self.data_manager.set_masks(mask_data)
                    self.mask_status.setText(f"Loaded: {mask_data.shape}")
                    self.data_loaded.emit('mask')
                    self.mask_data_loaded.emit(mask_data)
        except Exception as e:
            error_msg = f"Failed to load mask data: {str(e)}"
            self.mask_status.setText("Error loading")
            self.data_load_failed.emit(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _process_mask_layer(self, layer):
        """Process a layer into mask data."""
        if layer.data is None:
            raise ValueError("Selected layer contains no data")

        # Convert layer data to binary mask
        mask_data = layer.data.astype(bool)

        # If this is a single mask, add time dimension
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, ...]

        # Ensure we have a 3D array (time, height, width)
        if mask_data.ndim != 3:
            raise ValueError(
                f"Mask data must be 2D or 3D, got shape {mask_data.shape}"
            )

        return mask_data


class MSMActionPanel(QWidget):
    """Panel for analysis actions and progress display."""

    def __init__(self, msm_controller):
        super().__init__()
        self.controller = msm_controller
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Action buttons
        self.create_mask_btn = QPushButton("Create Masks from Image")
        layout.addWidget(self.create_mask_btn)

        button_layout = QHBoxLayout()
        # Left column
        left_col = QVBoxLayout()
        self.preview_mesh_btn = QPushButton("Preview Mesh")
        self.preview_frame_btn = QPushButton("Preview Current Frame")
        left_col.addWidget(self.preview_mesh_btn)
        left_col.addWidget(self.preview_frame_btn)

        # Right column
        right_col = QVBoxLayout()
        self.analyze_btn = QPushButton("Calculate Stress Tensors")
        self.save_btn = QPushButton("Save Results")
        right_col.addWidget(self.analyze_btn)
        right_col.addWidget(self.save_btn)

        button_layout.addLayout(left_col)
        button_layout.addLayout(right_col)
        layout.addLayout(button_layout)

        # Progress section
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect action panel buttons to controller methods."""
        # Existing connections
        self.preview_mesh_btn.clicked.connect(self.controller.preview_mesh)
        self.preview_frame_btn.clicked.connect(self.controller.preview_current_frame)
        self.analyze_btn.clicked.connect(self._handle_analyze_click)
        self.save_btn.clicked.connect(self.controller.save_results)

        # New connection for mask creation
        self.create_mask_btn.clicked.connect(self.controller.create_masks_from_images)

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
    progress_updated = Signal(int, str)  # (progress_value, status_message)
    analysis_started = Signal()
    analysis_completed = Signal(object)  # Results object
    analysis_failed = Signal(str)  # Error message
    mask_creation_progress = Signal(int, str)  # (progress, message)
    mask_creation_completed = Signal()
    mask_creation_failed = Signal(str)


    def __init__(self, viewer: Viewer, service: MSMService,
                 data_manager: DataManager, parameter_manager: ParameterManager,
                 visualization_manager: VisualizationManager):
        super().__init__()  # Initialize QObject
        self.viewer = viewer
        self.service = service
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        self.visualization_manager = visualization_manager

    def _update_progress(self, progress: int, status: str):
        """Update progress and emit signal."""
        self.progress_updated.emit(progress, status)

    def start_analysis(self):
        """Start the stress analysis."""
        try:
            if not self._validate_prerequisites():
                return

            self.analysis_started.emit()
            self._update_progress(0, "Starting analysis...")

            params = self._get_current_parameters()

            # Get all required data
            masks = self.data_manager.masks
            force_field = self.data_manager.force_field
            pixel_size = self.data_manager.force_params['pixel_size']
            downscale_factor = self.data_manager.force_params['downscale_factor']

            # Calculate stress fields
            results = self.service.calculate_stress_stack(
                masks=masks,
                force_field=force_field,
                params=params,
                progress_callback=self._handle_progress
            )

            # Process results
            stress_tensors = [r.stress_tensor for r in results]
            stress_tensor_stack = np.stack(stress_tensors) * pixel_size * downscale_factor * 1e-6

            # Update data manager
            analysis_params = {
                'pixel_size': pixel_size,
                'downscale_factor': downscale_factor,
                'density_factor': params.density_factor,
                'poisson_ratio_cells': params.poisson_ratio_cells,
                'algorithm': params.algorithm,
                'use_optimization': params.use_optimization,
                'max_stress': params.max_stress,
            }
            self.data_manager.set_stress_results(stress_tensor_stack, analysis_params)

            # Update visualization
            self.visualization_manager.visualize_stress_results(
                stress_tensor_stack,
                params.max_stress,
                downscale_factor=downscale_factor
            )

            self._update_progress(100, "Analysis completed successfully")
            self.analysis_completed.emit(results)
            return results

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self._update_progress(0, error_msg)
            self.analysis_failed.emit(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def create_masks_from_images(self):
        """Handle mask creation from the active image layer."""
        try:
            active_layer = self.viewer.layers.selection.active
            if not active_layer or not isinstance(active_layer.data, np.ndarray):
                raise ValueError("No valid image layer selected.")

            image_data = active_layer.data
            params = self._get_current_parameters()  # Reuse parameter fetch logic

            # Generate masks using the service
            masks, vis_masks = self.service.create_mask_stack(image_data, params)

            # Update DataManager without triggering external signals
            self.data_manager.set_masks(masks)

            # Update UI status via the action panel
            self.progress_updated.emit(100, "Masks created successfully.")

        except Exception as e:
            self.progress_updated.emit(0, f"Mask creation failed: {str(e)}")
            QMessageBox.critical(None, "Error", str(e))

    def _handle_progress(self, current: int, total: int, status: str):
        """Handle progress updates during analysis."""
        progress = int((current + 1) / total * 100)
        self._update_progress(progress, status)

    def preview_mesh(self):
        """Generate and display mesh preview."""
        try:
            if not self._validate_prerequisites():
                return

            self._update_progress(0, "Generating mesh preview...")
            current_frame = self.viewer.dims.current_step[0]
            params = self._get_current_parameters()

            # Get mask and resize if needed
            mask = self.data_manager.masks[current_frame]
            if self.data_manager.force_field is not None:
                target_shape = self.data_manager.force_field[current_frame, ..., 0].shape
                mask = self.service.resize_mask_to_forces(mask, target_shape)

            # Generate mesh preview
            preview_result = self.service.generate_mesh_preview(mask, params)

            # Get visualization scale factor
            downscale_factor = self.data_manager.force_params.get('downscale_factor', 1)

            # Update visualization
            self.visualization_manager.visualize_mesh(
                nodes=preview_result.nodes,
                elements=preview_result.elements,
                downscale_factor=downscale_factor
            )

            self._update_progress(100, "Mesh preview generated successfully")
            return preview_result

        except Exception as e:
            error_msg = f"Failed to preview mesh: {str(e)}"
            self._update_progress(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def preview_current_frame(self):
        """Calculate and display stress field for current frame."""
        try:
            if not self._validate_prerequisites():
                return

            self._update_progress(0, "Generating stress preview...")
            current_frame = self.viewer.dims.current_step[0]
            params = self._get_current_parameters()

            # Get current frame data
            mask = self.data_manager.masks[current_frame]
            tx = self.data_manager.force_field[current_frame, ..., 0]
            ty = self.data_manager.force_field[current_frame, ..., 1]

            # Ensure mask matches force field shape
            mask = self.service.resize_mask_to_forces(mask, tx.shape)

            # Calculate stress field
            result = self.service.calculate_stress_field(
                mask=mask,
                traction_x=tx,
                traction_y=ty,
                params=params
            )

            # Update visualization
            pixel_size = self.data_manager.force_params['pixel_size']
            downscale_factor = self.data_manager.force_params['downscale_factor']

            stress_tensor = result.stress_tensor * pixel_size * downscale_factor * 1e-6  # Convert to N/m
            self.visualization_manager.visualize_stress_preview(
                stress_tensor,
                params.max_stress,
                downscale_factor=downscale_factor
            )

            status = (
                f"Preview generated for frame {current_frame}\n"
                f"Condition number: {result.condition_number:.1e}\n"
                f"Residual: {result.residual:.1e}"
            )
            self._update_progress(100, status)
            return result

        except Exception as e:
            error_msg = f"Failed to preview stress field: {str(e)}"
            self._update_progress(0, error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            return None

    def _get_current_parameters(self) -> MSMParameters:
        """Get current MSM parameters from parameter manager."""
        return MSMParameters(
            density_factor=self.parameter_manager.get_value('density_factor'),
            algorithm=self.parameter_manager.get_value('mesh_algorithm'),
            use_optimization=self.parameter_manager.get_value('use_optimization'),
            poisson_ratio_cells=self.parameter_manager.get_value('poisson_ratio_cells'),
            young_modulus=1.0,  # Fixed value as per original implementation
            threshold=self.parameter_manager.get_value('threshold'),
            dilation=self.parameter_manager.get_value('dilation'),
            smoothing_sigma=self.parameter_manager.get_value('smoothing_sigma'),
            max_stress=self.parameter_manager.get_value('max_stress')
        )

    def save_results(self):
        """Save analysis results to file."""
        try:
            if self.data_manager.stress_tensor is None:
                raise ValueError("No stress tensor data to save")

            file_path, _ = QFileDialog.getSaveFileName(
                None, "Save Stress Tensor Data", "", "NumPy Files (*.npy)"
            )

            if file_path:
                if not file_path.endswith('.npy'):
                    file_path += '.npy'

                stress_results = {
                    'stress_tensor': self.data_manager.stress_tensor,
                    'parameters': self.data_manager.stress_params
                }
                np.save(file_path, stress_results)
                return True

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save results: {str(e)}")
        return False

    def _validate_prerequisites(self) -> bool:
        """Check if required data is available."""
        if self.data_manager.masks is None:
            QMessageBox.warning(None, "Warning", "No mask loaded. Please load a mask first.")
            return False
        if self.data_manager.force_field is None:
            QMessageBox.warning(None, "Warning", "No force data available. Please calculate forces first.")
            return False
        return True


class MSMWidget(BaseAnalysisWidget):
    """Widget for Monolayer Stress Microscopy analysis."""
    stress_calculated = Signal(dict)  # Emits stress analysis results

    def __init__(self, viewer: Viewer, data_manager: DataManager,
                 parameter_manager: ParameterManager, visualization_manager: VisualizationManager):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize managers and service
        self.parameter_manager = parameter_manager
        self.service = MSMService()
        self.parameter_panel = MSMParameterPanel(parameter_manager)
        self.data_panel = MSMDataPanel(data_manager, viewer)
        self.controller = MSMController(
            viewer=viewer,
            service=self.service,
            data_manager=data_manager,
            parameter_manager=parameter_manager,
            visualization_manager=visualization_manager
        )
        self.action_panel = MSMActionPanel(self.controller)

        # Setup UI
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left side: Colorbar (from BaseAnalysisWidget)
        colorbar_container = QWidget()
        colorbar_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(6, 6, 6, 6)

        self.colorbar_manager = ColorbarManager()
        colorbar_group = self.create_colorbar_widget(
            colormap_name='seismic',
            label="Stress (mN/m)",
            clim=(-1, 1),
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        colorbar_layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        colorbar_layout.addStretch()
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
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(6, 6, 6, 6)

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
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)
        scroll_layout.addWidget(status_frame)

        # Set the layout for the scroll content
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        main_layout.addWidget(scroll_area)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

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

    def _update_status(self, progress: int, message: str):
        """Update status display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def cleanup(self):
        """Clean up resources."""
        if self.colorbar_manager:
            self.colorbar_manager.cleanup()
        self.visualization_manager.cleanup()
        super().cleanup()

    # Event handlers
    def _on_data_loaded(self, data_type: str):
        self._update_ui_state()

    def _on_parameter_changed(self):
        if hasattr(self, 'preview_active') and self.preview_active:
            self.controller.preview_current_frame()

    def _on_analysis_started(self):
        self._set_controls_enabled(False)

    def _on_analysis_completed(self, results):
        self._set_controls_enabled(True)
        self.stress_calculated.emit(results)

    def _on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        self._set_controls_enabled(True)
        self._update_status(0, f"Analysis failed: {error_msg}")

    def _on_mask_creation_completed(self):
        """Handle successful mask creation."""
        self.data_panel.update_mask_status("Masks created successfully")

    def _on_mask_creation_failed(self, error_msg: str):
        """Handle mask creation failure."""
        self.data_panel.update_mask_status("Mask creation failed")
        QMessageBox.critical(self, "Error", error_msg)

    def _update_ui_state(self):
        """Update UI element states based on current data availability."""
        has_force_data = self.data_manager.force_field is not None
        has_mask_data = self.data_manager.masks is not None
        has_stress_data = self.data_manager.stress_tensor is not None

        self.action_panel.set_buttons_enabled(has_force_data and has_mask_data)
        self.action_panel.save_btn.setEnabled(has_stress_data)
