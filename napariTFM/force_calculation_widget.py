from typing import Dict

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QPushButton, QFrame,
    QProgressBar, QMessageBox, QWidget, QScrollArea,
    QSizePolicy
)

import force_calculation as tfm_functions
from .base_widget import BaseAnalysisWidget
from .colorbar import ColorbarManager
from .data_manager import DataManager
from .visualization_manager import VisualizationManager


class ForceCalculationWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method."""

    force_calculated = Signal(dict)  # Emits force calculation results
    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize parameters
        self.young_modulus = 10000  # Pa
        self.poisson_ratio = 0.49
        self.gel_height = None  # μm (None means infinite)
        self.pixel_size = 0.1  # μm/pixel
        self.spatial_filter = "gaussian"
        self.filter_size = 6  # μm

        # Initialize colorbar
        self.colorbar_manager = ColorbarManager()
        self.colorbar_widget = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create and set up colorbar container
        colorbar_container = QWidget()
        colorbar_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        colorbar_layout = QVBoxLayout()
        colorbar_layout.setContentsMargins(6, 6, 6, 6)

        # Create colorbar using base class method
        colorbar_group = self.create_colorbar_widget(
            colormap_name='inferno',
            label="Force (Pa)",
            clim=(1000, 0),
            colorbar_manager=self.colorbar_manager
        )
        colorbar_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Add colorbar to its container with alignment
        colorbar_layout.addWidget(colorbar_group, alignment=Qt.AlignTop)
        colorbar_layout.addStretch()
        colorbar_container.setLayout(colorbar_layout)

        main_layout.addWidget(colorbar_container)

        # Container for the right side content
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Add all component groups
        right_layout.addWidget(self._create_material_params_group())
        right_layout.addWidget(self._create_filter_params_group())
        right_layout.addWidget(self._create_action_buttons())
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(300)  # Fixed width for the right side

        # Add right container to main layout
        main_layout.addWidget(right_container)
        main_layout.addStretch(1)  # Add stretch after the container

        self.setLayout(main_layout)
    def _create_material_params_group(self) -> QGroupBox:
        """Create the material parameters group."""
        group = QGroupBox("Material Parameters")
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        # Create spin boxes first
        self.young_spin = QDoubleSpinBox()
        self.poisson_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        self.pixel_spin = QDoubleSpinBox()

        params = [
            ("Young's Modulus (Pa):", self.young_spin, 100, 1000000, 100, self.young_modulus),
            ("Poisson Ratio:", self.poisson_spin, 0, 0.5, 0.01, self.poisson_ratio),
            ("Gel Height (μm):", self.height_spin, 0, 1000, 10, 0),
            ("Pixel Size (μm):", self.pixel_spin, 0.001, 10, 0.1, self.pixel_size)
        ]

        for label_text, spin, min_val, max_val, step, default in params:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(120)  # Fixed width for labels
            row.addWidget(label)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if label_text.startswith("Gel Height"):
                spin.setSpecialValueText("∞")
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group
    def _create_filter_params_group(self) -> QGroupBox:
        """Create the filter parameters group."""
        group = QGroupBox("Filter Parameters")
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        row = QHBoxLayout()
        label = QLabel("Filter Size (μm):")
        label.setFixedWidth(120)  # Fixed width for consistency
        row.addWidget(label)

        self.filter_size_spin = QDoubleSpinBox()
        self.filter_size_spin.setRange(0.1, 50)
        self.filter_size_spin.setValue(self.filter_size)
        self.filter_size_spin.setSingleStep(0.5)
        row.addWidget(self.filter_size_spin)

        layout.addLayout(row)
        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout()

        self.calculate_btn = QPushButton("Calculate Forces")
        self.reset_btn = QPushButton("Reset Parameters")

        layout.addWidget(self.calculate_btn)
        layout.addWidget(self.reset_btn)

        frame.setLayout(layout)
        return frame

    def _create_status_frame(self) -> QFrame:
        """Create the status and progress frame."""
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)  # Allow text wrapping

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _update_colorbar(self, vmin: float, vmax: float):
        """Update the colorbar limits and appearance."""
        if self.colorbar_manager is not None:
            self.colorbar_manager.update_limits(vmin, vmax)

    def _update_visualization(self, results: Dict):
        """Update visualization of force calculation results with colorbar."""
        try:
            # Remove existing force layers
            self.visualization_manager._clear_layers(['Force Magnitude', 'Force Vectors'])

            # Calculate magnitude stack
            magnitude_stack = np.sqrt(results['tx'] ** 2 + results['ty'] ** 2)

            # Add magnitude layer
            with self.viewer.events.blocker_all():
                magnitude_layer = self.viewer.add_image(
                    magnitude_stack,
                    name='Force Magnitude',
                    colormap='inferno',
                    blending='additive'
                )

            # Update colorbar
            vmax = magnitude_stack.max()
            self._update_colorbar(0, vmax)

            # Update status with statistics
            mean_force = np.mean(magnitude_stack)
            max_force = np.max(magnitude_stack)

            stats_text = (
                f"Mean force magnitude: {mean_force:.2f} Pa\n"
                f"Max force magnitude: {max_force:.2f} Pa"
            )
            self._update_status(stats_text, 100)

        except Exception as e:
            self._handle_error(f"Failed to update visualization: {str(e)}")

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.colorbar_manager is not None:
                self.colorbar_manager.cleanup()
                self.colorbar_manager = None
                self.colorbar_widget = None
        except Exception:
            pass

        super().cleanup()

    def calculate_forces(self):

        """Calculate traction forces using FTTC method."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating forces...", 0)

            # Get displacement data
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            num_frames = len(flows)

            # Initialize result arrays
            force_results = {
                'tx': [],
                'ty': []
            }

            # Process each frame
            for i, flow in enumerate(flows):
                progress = (i + 1) / num_frames * 100
                self._update_status(f"Processing frame {i + 1}/{num_frames}...", progress)

                # Extract u and v components
                u = flow[..., 0]
                v = flow[..., 1]

                # Calculate forces using TFM functions
                tx, ty = tfm_functions.TFM_tractions(
                    u=u,
                    v=v,
                    pixelsize1=self.pixel_size,
                    pixelsize2=self.pixel_size,
                    h="infinite" if self.gel_height is None else self.gel_height,
                    young=self.young_modulus,
                    sigma=self.poisson_ratio,
                    spatial_filter=self.spatial_filter,
                    fs=self.filter_size
                )

                # Store results
                force_results['tx'].append(tx)
                force_results['ty'].append(ty)

            # Convert lists to arrays
            force_results['tx'] = np.stack(force_results['tx'])
            force_results['ty'] = np.stack(force_results['ty'])
            force_results['parameters'] = {
                'young_modulus': self.young_modulus,
                'poisson_ratio': self.poisson_ratio,
                'gel_height': self.gel_height,
                'pixel_size': self.pixel_size,
                'spatial_filter': self.spatial_filter,
                'filter_size': self.filter_size
            }

            # Store results in data manager
            self.data_manager.force_results = force_results

            # Update visualization through manager
            stats = self.visualization_manager.update_force_visualization(force_results)

            # Update status with statistics
            stats_text = (
                f"Mean force magnitude: {stats['mean_force']:.2f} Pa\n"
                f"Max force magnitude: {stats['max_force']:.2f} Pa"
            )
            self._update_status(stats_text, 100)

            # Emit results
            self.force_calculated.emit(force_results)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)
    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
            self.young_spin,
            self.poisson_spin,
            self.height_spin,
            self.pixel_spin,
            self.filter_size_spin,
            self.calculate_btn,
            self.reset_btn,
            self.progress_bar,
            self.status_label
        ]

        for control in controls:
            self.register_control(control)

    def _connect_signals(self):
        """Connect widget signals."""
        # Parameter updates
        self.young_spin.valueChanged.connect(self._update_parameters)
        self.poisson_spin.valueChanged.connect(self._update_parameters)
        self.height_spin.valueChanged.connect(self._update_parameters)
        self.pixel_spin.valueChanged.connect(self._update_parameters)
        self.filter_size_spin.valueChanged.connect(self._update_parameters)

        # Action buttons
        self.calculate_btn.clicked.connect(self.calculate_forces)
        self.reset_btn.clicked.connect(self.reset_parameters)

    def _update_parameters(self):
        """Update parameters from UI controls."""
        try:
            self.young_modulus = self.young_spin.value()
            self.poisson_ratio = self.poisson_spin.value()
            self.gel_height = None if self.height_spin.value() == 0 else self.height_spin.value()
            self.pixel_size = self.pixel_spin.value()
            self.filter_size = self.filter_size_spin.value()

        except Exception as e:
            self._handle_error(str(e))

    def reset_parameters(self):
        """Reset all parameters to defaults."""
        self.young_spin.setValue(10000)
        self.poisson_spin.setValue(0.49)
        self.height_spin.setValue(0)
        self.pixel_spin.setValue(0.1)
        self.filter_size_spin.setValue(6)

        self._update_status("Parameters reset to defaults")

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        if not hasattr(self.data_manager, 'displacement_results'):
            QMessageBox.warning(self, "Error", "Displacement analysis required")
            return False

        if 'flows' not in self.data_manager.displacement_results:
            QMessageBox.warning(self, "Error", "Invalid displacement data")
            return False

        return True

