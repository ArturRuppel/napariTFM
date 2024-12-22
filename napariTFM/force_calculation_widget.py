from typing import Dict

import napari
import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QPushButton, QFrame,
    QProgressBar, QMessageBox, QWidget,
    QSizePolicy, QSpinBox, QComboBox
)

from .base_widget import BaseAnalysisWidget
from .colorbar import ColorbarManager
from .force_calculation import TractionForceCalculator, CalculationMethod
from .data_manager import DataManager
from .visualization_manager import VisualizationManager


class ForceCalculationWidget(BaseAnalysisWidget):
    """Widget for calculating traction forces using FTTC method with finite thickness corrections."""

    force_calculated = Signal(dict)  # Emits force calculation results

    def __init__(
            self,
            viewer: "napari.Viewer",
            data_manager: "DataManager",
            visualization_manager: "VisualizationManager"
    ):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize parameters first
        self.young_modulus = 10000  # Pa
        self.poisson_ratio = 0.49
        self.gel_height = None  # μm (None means infinite)
        self.pixel_size = 0.1  # μm/pixel
        self.regularization = 1e-6
        self.filter_sigma = 2.0  # pixels
        self.characteristic_length = 50  # μm (typical cell size)
        self.calculation_method = CalculationMethod.FFTC

        # Initialize other attributes
        self.calculator = None
        self.colorbar_manager = ColorbarManager()
        self.visualization_params = {}

        # Setup UI and connections
        self._setup_ui()
        self._connect_signals()
        self._register_controls()
        self._update_ui_state()

    def _setup_ui(self):
        """Set up the user interface."""
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
            colormap_name='inferno',
            label="Force (Pa)",
            clim=(1000, 0),
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

        # Add parameter groups - remove correction_params_group
        right_layout.addWidget(self._create_material_params_group())
        right_layout.addWidget(self._create_calculation_params_group())
        right_layout.addWidget(self._create_visualization_parameters_group())
        right_layout.addWidget(self._create_action_buttons())
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(350)

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect widget signals."""
        # Parameter updates
        self.young_spin.valueChanged.connect(self._update_parameters)
        self.poisson_spin.valueChanged.connect(self._update_parameters)
        self.height_spin.valueChanged.connect(self._update_parameters)
        self.pixel_spin.valueChanged.connect(self._update_parameters)
        self.regularization_spin.valueChanged.connect(self._update_parameters)
        self.filter_sigma_spin.valueChanged.connect(self._update_parameters)
        self.calculation_method_combo.currentTextChanged.connect(self._update_parameters)

        # Action buttons
        self.calculate_btn.clicked.connect(self.calculate_forces)
        self.reset_btn.clicked.connect(self.reset_parameters)

    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
                       self.young_spin,
                       self.poisson_spin,
                       self.height_spin,
                       self.pixel_spin,
                       self.regularization_spin,
                       self.filter_sigma_spin,
                       self.calculation_method_combo,
                       self.calculate_btn,
                       self.reset_btn,
                       self.progress_bar,
                       self.status_label
                   ] + list(self.visualization_params.values())

        for control in controls:
            self.register_control(control)

    def _create_calculation_params_group(self) -> QGroupBox:
        """Create the calculation parameters group."""
        group = QGroupBox("Calculation Parameters")
        layout = QVBoxLayout()

        # Method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.calculation_method_combo = QComboBox()
        self.calculation_method_combo.addItems([
            "FFTC",
            "Pure Shear"
        ])
        method_layout.addWidget(self.calculation_method_combo)
        layout.addLayout(method_layout)

        # Gel height (optional for FFTC, required for Pure Shear)
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Gel Height (μm):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0, 1000)
        self.height_spin.setSpecialValueText("∞")  # Only meaningful for FFTC
        self.height_spin.setValue(0)
        height_layout.addWidget(self.height_spin)
        layout.addLayout(height_layout)

        # Regularization parameter
        reg_layout = QHBoxLayout()
        reg_layout.addWidget(QLabel("Regularization:"))
        self.regularization_spin = QSpinBox()
        self.regularization_spin.setRange(1, 100000)
        self.regularization_spin.setValue(1000)
        self.regularization_spin.setToolTip(
            "Regularization parameter (will be multiplied by 10⁻²¹)\n"
            "Typical values: 1000-10000"
        )
        reg_layout.addWidget(self.regularization_spin)
        layout.addLayout(reg_layout)

        # Filter sigma
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Sigma:"))
        self.filter_sigma_spin = QDoubleSpinBox()
        self.filter_sigma_spin.setRange(0, 10)
        self.filter_sigma_spin.setValue(2.0)
        self.filter_sigma_spin.setSingleStep(0.1)
        filter_layout.addWidget(self.filter_sigma_spin)
        layout.addLayout(filter_layout)

        group.setLayout(layout)
        return group

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Check if required data is available using DataManager method
        has_required_data = self.data_manager.has_required_force_data()

        # Update button states
        self.calculate_btn.setEnabled(has_required_data)

        # Update status message
        if not has_required_data:
            self._update_status("Required displacement data not available")
        else:
            self._update_status("Ready for force calculation")

        # Update gel height field based on method
        is_pure_shear = self.calculation_method == CalculationMethod.PURE_SHEAR
        if is_pure_shear:
            self.height_spin.setSpecialValueText("")  # Remove infinity symbol
            if self.height_spin.value() == 0:
                self.height_spin.setValue(100)  # Set default height for pure shear
        else:
            self.height_spin.setSpecialValueText("∞")  # Show infinity symbol for FFTC

        # Update tooltip
        self.height_spin.setToolTip(
            "Required for Pure Shear calculation\n"
            "Optional for FFTC (enables finite thickness correction)" if not is_pure_shear
            else "Required gel height for Pure Shear calculation"
        )

        # Update parameters based on current results
        if hasattr(self.data_manager, 'force_results') and self.data_manager.force_results:
            results = self.data_manager.force_results
            if 'parameters' in results and 'visualization' in results['parameters']:
                vis_params = results['parameters']['visualization']
                self.visualization_params['vector_stride'].setValue(vis_params.get('vector_stride', 20))
                self.visualization_params['arrow_scale'].setValue(vis_params.get('arrow_scale', 1.0))
                self.visualization_params['f_max'].setValue(vis_params.get('f_max', 1000.0))
    def calculate_forces(self):
        """Calculate traction forces using the TractionForceCalculator."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating forces...", 0)

            # Get displacement data
            displacement_results = self.data_manager.displacement_results
            flows = displacement_results['flows']
            num_frames = len(flows)

            # Initialize calculator with current parameters
            self._initialize_calculator()

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

                # Calculate forces
                tx, ty = self.calculator.calculate_forces(u, v)
                force_results['tx'].append(tx)
                force_results['ty'].append(ty)

            # Convert lists to arrays
            force_results['tx'] = np.stack(force_results['tx'])
            force_results['ty'] = np.stack(force_results['ty'])

            # Get visualization parameters
            visualization_params = {
                'vector_stride': self.visualization_params['vector_stride'].value(),
                'arrow_scale': self.visualization_params['arrow_scale'].value(),
                'f_max': self.visualization_params['f_max'].value()
            }

            # Store parameters
            force_results['parameters'] = {
                'young_modulus': self.young_modulus,
                'poisson_ratio': self.poisson_ratio,
                'gel_height': self.gel_height,
                'pixel_size': self.pixel_size,
                'regularization': self.regularization,
                'filter_sigma': self.filter_sigma,
                'calculation_method': self.calculation_method.value,
                'characteristic_length': self.characteristic_length,
                'visualization': visualization_params
            }

            # Update data manager
            self.data_manager.force_results = force_results

            # Update visualization through visualization manager
            self.visualization_manager.update_force_visualization(force_results, visualization_params)

            # Update colorbar
            f_max = visualization_params['f_max']
            if f_max is not None:
                self.colorbar_manager.update_limits(0, f_max)

            # Get and display statistics
            stats = self.visualization_manager.get_force_statistics(force_results)
            if stats:
                stats_text = (
                    f"Mean force: {stats['mean_force']:.2f} Pa\n"
                    f"Max force: {stats['max_force']:.2f} Pa\n"
                    f"Median force: {stats['median_force']:.2f} Pa"
                )
                self._update_status(stats_text, 100)

            # Emit results
            self.force_calculated.emit(force_results)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)


    def _create_material_params_group(self) -> QGroupBox:
        """Create the material parameters group."""
        group = QGroupBox("Material Parameters")
        layout = QVBoxLayout()

        # Create spinboxes
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
            label.setFixedWidth(120)
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

    def _create_correction_params_group(self) -> QGroupBox:
        """Create the correction method parameters group."""
        group = QGroupBox("Correction Method")
        layout = QVBoxLayout()

        # Correction method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.correction_method_combo = QComboBox()
        self.correction_method_combo.addItems([
            "None (Infinite)",
            "Finite Thickness",
            "Pure Shear"
        ])
        method_layout.addWidget(self.correction_method_combo)
        layout.addLayout(method_layout)

        # Characteristic length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Char. Length (μm):"))
        self.char_length_spin = QDoubleSpinBox()
        self.char_length_spin.setRange(1, 1000)
        self.char_length_spin.setValue(self.characteristic_length)
        self.char_length_spin.setSingleStep(5)
        length_layout.addWidget(self.char_length_spin)
        layout.addLayout(length_layout)

        group.setLayout(layout)
        return group

    def reset_parameters(self):
        """Reset all parameters to defaults."""
        self.young_spin.setValue(10000)
        self.poisson_spin.setValue(0.49)
        self.height_spin.setValue(0)
        self.pixel_spin.setValue(0.1)
        self.regularization_spin.setValue(1000)
        self.filter_sigma_spin.setValue(2.0)
        self.calculation_method_combo.setCurrentText("FFTC")

        self.visualization_params['vector_stride'].setValue(20)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['f_max'].setValue(1000.0)

        self._update_status("Parameters reset to defaults")

    def _create_visualization_parameters_group(self) -> QGroupBox:
        """Create the visualization parameters group."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        # Vector stride
        stride_layout = QHBoxLayout()
        stride_layout.addWidget(QLabel("Vector Stride:"))
        self.visualization_params['vector_stride'] = QSpinBox()
        self.visualization_params['vector_stride'].setRange(1, 100)
        self.visualization_params['vector_stride'].setValue(20)
        stride_layout.addWidget(self.visualization_params['vector_stride'])
        layout.addLayout(stride_layout)

        # Arrow scale
        arrow_layout = QHBoxLayout()
        arrow_layout.addWidget(QLabel("Arrow Scale:"))
        self.visualization_params['arrow_scale'] = QDoubleSpinBox()
        self.visualization_params['arrow_scale'].setRange(0.1, 50.0)
        self.visualization_params['arrow_scale'].setValue(1.0)
        arrow_layout.addWidget(self.visualization_params['arrow_scale'])
        layout.addLayout(arrow_layout)

        # Maximum force
        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Force (Pa):"))
        self.visualization_params['f_max'] = QDoubleSpinBox()
        self.visualization_params['f_max'].setRange(0.1, 10000.0)
        self.visualization_params['f_max'].setValue(1000.0)
        fmax_layout.addWidget(self.visualization_params['f_max'])
        layout.addLayout(fmax_layout)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
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
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        if not self.data_manager.displacement_results:
            QMessageBox.warning(self, "Error", "Displacement analysis required before force calculation")
            return False

        if 'flows' not in self.data_manager.displacement_results:
            QMessageBox.warning(self, "Error", "Invalid displacement data format")
            return False

        # Validate gel height for pure shear method
        if self.calculation_method == CalculationMethod.PURE_SHEAR and (self.gel_height is None or self.gel_height == 0):
            QMessageBox.warning(self, "Error", "Gel height must be specified for pure shear calculation")
            return False

        return True

    def _initialize_calculator(self):
        """Initialize the TractionForceCalculator with current parameters."""
        # Update parameters from UI
        self._update_parameters()

        # Create calculator instance
        self.calculator = TractionForceCalculator(
            young_modulus=self.young_modulus,
            pixel_size=self.pixel_size * 1e-6,  # Convert to meters
            poisson_ratio=self.poisson_ratio,
            regularization=self.regularization,
            gel_height=None if self.gel_height is None else self.gel_height * 1e-6,  # Convert to meters
            calculation_method=self.calculation_method,
            filter_sigma=self.filter_sigma
        )

    def _update_parameters(self):
        """Update parameters from UI controls."""
        # Update basic parameters
        self.young_modulus = self.young_spin.value()
        self.poisson_ratio = self.poisson_spin.value()
        self.pixel_size = self.pixel_spin.value()
        self.regularization = self.regularization_spin.value() * 1e-21
        self.filter_sigma = self.filter_sigma_spin.value()

        # Update calculation method
        method_text = self.calculation_method_combo.currentText()
        self.calculation_method = (
            CalculationMethod.PURE_SHEAR if method_text == "Pure Shear"
            else CalculationMethod.FFTC
        )

        # Handle gel height
        height_value = self.height_spin.value()
        self.gel_height = None if height_value == 0 else height_value

        # Update UI state based on method
        self._update_ui_state()

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.colorbar_manager is not None:
                self.colorbar_manager.cleanup()
                self.colorbar_manager = None

            self._clear_force_layers()
            self.calculator = None
        except Exception as e:
            self._handle_error(f"Cleanup failed: {str(e)}")

        super().cleanup()
