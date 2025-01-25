import os
import sys
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget, QGridLayout,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame, QScrollArea,
    QProgressBar, QMessageBox, QListWidget, QCheckBox,
    QFileDialog, QComboBox
)

from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.parameter_manager import ParameterManager, ParameterCategory


class BatchAnalysisWidget(BaseAnalysisWidget):
    """Widget for running batch analysis on multiple folders."""

    batch_completed = Signal(dict)  # Emits results when batch processing completes

    MESH_ALGORITHMS = {
        "Frontal-Del.": 6,
        "Delaunay": 5,
        "MeshAdapt": 1,
        "BAMG": 7,
        "FD Quads": 8,
        "Para. Pack": 9
    }

    def __init__(self, viewer, data_manager, parameter_manager: ParameterManager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        self.parameter_manager = parameter_manager
        self.parameter_spins = {}
        self.parameter_combos = {}
        self.parameter_checks = {}
        self.analysis_checkboxes = {}
        self.visualization_checkboxes = {}
        self.folder_list = []

        # Block signals during setup
        self.blockSignals(True)
        try:
            self._setup_ui()
            self._connect_signals()

            # Ensure parameters are connected before syncing
            self._connect_parameters()

            # Force an initial sync with parameter manager
            self._sync_widget_with_parameters()

            self._update_ui_state()

            # Connect to parameter manager signals after everything is set up
            if hasattr(self.parameter_manager, 'parameter_changed'):
                self.parameter_manager.parameter_changed.connect(self._on_parameter_changed)
        finally:
            self.blockSignals(False)

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager values"""
        if not hasattr(self, 'parameter_manager') or self.parameter_manager is None:
            print("Warning: No parameter manager available for syncing")
            return

        self._block_parameter_widgets(True)
        try:
            # Sync threshold - handle both tuple and single widget cases
            if 'threshold' in self.parameter_spins:
                threshold_value = self.parameter_manager.get_value('threshold')
                threshold_widget = self.parameter_spins['threshold']
                if isinstance(threshold_widget, tuple):
                    # If it's a tuple of (spinbox, slider)
                    threshold_spin, threshold_slider = threshold_widget
                    threshold_spin.setValue(threshold_value)
                    threshold_slider.setValue(threshold_value)
                else:
                    # If it's just a single widget
                    threshold_widget.setValue(threshold_value)

            # Sync other numeric parameters
            for param_name in ['dilation', 'smoothing_sigma', 'density_factor', 'max_stress']:
                if param_name in self.parameter_spins:
                    value = self.parameter_manager.get_value(param_name)
                    self._safe_set_value(self.parameter_spins[param_name], value)

            # Sync mesh algorithm
            algo_value = self.parameter_manager.get_value('mesh_algorithm')
            if algo_value and 'algorithm' in self.parameter_combos:
                self._safe_set_combo_text(
                    self.parameter_combos['algorithm'],
                    algo_value.replace('_', '-').title()
                )

            # Sync optimization checkbox
            if 'use_optimization' in self.parameter_checks:
                self._safe_set_checked(
                    self.parameter_checks['use_optimization'],
                    bool(self.parameter_manager.get_value('use_optimization'))
                )

            # Sync Poisson ratio (poisson_ratio)
            if 'poisson_ratio' in self.parameter_spins:
                value = self.parameter_manager.get_value('poisson_ratio')
                self._safe_set_value(self.parameter_spins['poisson_ratio'], value)

        except Exception as e:
            print(f"Error syncing parameters: {str(e)}")
        finally:
            self._block_parameter_widgets(False)

    def _block_parameter_widgets(self, block: bool):
        """Block or unblock signals for all parameter widgets."""
        widgets = []

        # Add all spinboxes
        for spin in self.parameter_spins.values():
            if isinstance(spin, tuple):  # Handle special cases like threshold
                widgets.extend(spin)
            else:
                widgets.append(spin)

        # Add all comboboxes
        for combo in self.parameter_combos.values():
            widgets.append(combo)

        # Add all checkboxes
        for checkbox in self.parameter_checks.values():
            widgets.append(checkbox)

        # Add visualization checkboxes
        for checkbox in self.visualization_checkboxes.values():
            widgets.append(checkbox)

        for widget in widgets:
            widget.blockSignals(block)

    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes from parameter manager."""
        # Only update if the change didn't come from this widget
        if not self.signalsBlocked():
            if param_name in ['pixel_size', 'frame_interval']:
                self._update_calibration()
            else:
                self._sync_widget_with_parameters()

    def _connect_parameters(self):
        """Connect widget controls to parameter manager."""
        # Block signals during initial setup
        self._block_parameter_widgets(True)

        try:
            # Disconnect any existing connections to avoid duplicates
            if hasattr(self.parameter_manager, 'parameter_changed'):
                try:
                    self.parameter_manager.parameter_changed.disconnect(self._on_parameter_changed)
                except TypeError:
                    pass  # Connection didn't exist

            # Connect basic parameters
            basic_params = {
                'pixel_size': self.parameter_spins['pixel_size'],
                'frame_interval': self.parameter_spins['frame_interval']
            }

            for name, spin in basic_params.items():
                # Connect widget to parameter manager
                spin.valueChanged.connect(
                    lambda value, name=name: self.parameter_manager.set_value(name, value)
                )
                # Connect parameter manager to widget
                self.parameter_manager.register_callback(
                    name,
                    lambda value, spin=spin: self._safe_set_value(spin, value)
                )
                # Set initial value
                value = self.parameter_manager.get_value(name)
                self._safe_set_value(spin, value)

            preprocessing_params = [
                'min_intensity', 'max_intensity', 'gaussian_sigma',
                'cell_min_intensity', 'cell_max_intensity', 'cell_gaussian_sigma',
                'registration_mode'
            ]

            for name in preprocessing_params:
                if name in self.parameter_spins or name in self.parameter_combos:
                    if name == 'registration_mode':  # Handle combobox
                        combo = self.parameter_combos[name]
                        combo.currentTextChanged.connect(
                            lambda text, name=name: self.parameter_manager.set_value(name, text)
                        )
                        self.parameter_manager.register_callback(
                            name,
                            lambda value, combo=combo: self._safe_set_combo_text(combo, value)
                        )
                    else:  # Handle spinboxes
                        spin = self.parameter_spins[name]
                        spin.valueChanged.connect(
                            lambda value, name=name: self.parameter_manager.set_value(name, float(value))
                        )
                        self.parameter_manager.register_callback(
                            name,
                            lambda value, spin=spin: self._safe_set_value(spin, float(value))
                        )
                    # Set initial value
                    try:
                        value = self.parameter_manager.get_value(name)
                        if name == 'registration_mode':
                            self._safe_set_combo_text(combo, value)
                        else:
                            self._safe_set_value(spin, value)
                    except KeyError:
                        print(f"Warning: Preprocessing parameter {name} not found in parameter manager")

            # Special handling for Young's modulus (convert Pa to kPa for display)
            young_spin = self.parameter_spins['young_modulus']
            young_spin.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('young_modulus', value * 1000)
            )
            self.parameter_manager.register_callback(
                'young_modulus',
                lambda value: self._safe_set_value(young_spin, value / 1000 if value is not None else 0)
            )
            try:
                value = self.parameter_manager.get_value('young_modulus')
                self._safe_set_value(young_spin, value / 1000 if value is not None else 0)
            except KeyError:
                print("Warning: Parameter young_modulus not found in parameter manager")

            # Handle gel height with special "infinity" case
            gel_height_spin = self.parameter_spins['gel_height']
            gel_height_spin.valueChanged.connect(
                lambda value: self.parameter_manager.set_value(
                    'gel_height',
                    None if value == 0 else value
                )
            )
            self.parameter_manager.register_callback(
                'gel_height',
                lambda value: self._safe_set_value(gel_height_spin, 0 if value is None else value)
            )
            try:
                value = self.parameter_manager.get_value('gel_height')
                self._safe_set_value(gel_height_spin, 0 if value is None else value)
            except KeyError:
                print("Warning: Parameter gel_height not found in parameter manager")

            # Handle regularization parameter (stored as actual value, displayed as log10)
            reg_spin = self.parameter_spins['regularization']
            reg_spin.valueChanged.connect(
                lambda value: self.parameter_manager.set_value('regularization', 10 ** value)
            )
            self.parameter_manager.register_callback(
                'regularization',
                lambda value: self._safe_set_value(reg_spin, np.log10(value) if value and value > 0 else -4)
            )
            try:
                value = self.parameter_manager.get_value('regularization')
                self._safe_set_value(reg_spin, np.log10(value) if value and value > 0 else -4)
            except KeyError:
                print("Warning: Parameter regularization not found in parameter manager")

            # Handle integer-based parameters
            int_params = ['lanczos_exp', 'force_vector_stride', 'disp_vector_stride']
            for name in int_params:
                if name in self.parameter_spins:
                    spin = self.parameter_spins[name]
                    spin.valueChanged.connect(
                        lambda value, name=name: self.parameter_manager.set_value(name, int(value))
                    )
                    self.parameter_manager.register_callback(
                        name,
                        lambda value, spin=spin: self._safe_set_value(spin, int(value) if value is not None else 0)
                    )
                    try:
                        value = self.parameter_manager.get_value(name)
                        self._safe_set_value(spin, int(value) if value is not None else 0)
                    except KeyError:
                        print(f"Warning: Parameter {name} not found in parameter manager")

            # Handle float-based parameters
            float_params = [
                'poisson_ratio', 'force_arrow_scale', 'f_max',
                'disp_arrow_scale', 'd_max', 'tau', 'lambda_',
                'theta', 'epsilon', 'scale_step', 'threshold', 'dilation',
                'smoothing_sigma', 'density_factor', 'max_stress'
            ]
            for name in float_params:
                if name in self.parameter_spins:
                    spin = self.parameter_spins[name]
                    spin.valueChanged.connect(
                        lambda value, name=name: self.parameter_manager.set_value(name, float(value))
                    )
                    self.parameter_manager.register_callback(
                        name,
                        lambda value, spin=spin: self._safe_set_value(spin, float(value) if value is not None else 0)
                    )
                    try:
                        value = self.parameter_manager.get_value(name)
                        self._safe_set_value(spin, float(value) if value is not None else 0)
                    except KeyError:
                        print(f"Warning: Parameter {name} not found in parameter manager")

            # Handle auto-GCV checkbox and its interaction with regularization
            if 'auto_gcv' in self.parameter_checks:
                auto_gcv = self.parameter_checks['auto_gcv']
                reg_spin = self.parameter_spins['regularization']

                def on_auto_gcv_changed(state):
                    is_checked = state == Qt.Checked
                    self.parameter_manager.set_value('auto_gcv', is_checked)
                    reg_spin.setEnabled(not is_checked)

                auto_gcv.stateChanged.connect(on_auto_gcv_changed)

                def update_auto_gcv(value):
                    auto_gcv.setChecked(bool(value))
                    reg_spin.setEnabled(not bool(value))

                self.parameter_manager.register_callback('auto_gcv', update_auto_gcv)

                try:
                    value = self.parameter_manager.get_value('auto_gcv')
                    auto_gcv.setChecked(bool(value))
                    reg_spin.setEnabled(not bool(value))
                except KeyError:
                    print("Warning: Parameter auto_gcv not found in parameter manager")

            if 'use_optimization' in self.parameter_checks:
                checkbox = self.parameter_checks['use_optimization']
                checkbox.stateChanged.connect(
                    lambda state: self.parameter_manager.set_value('use_optimization', state == Qt.Checked)
                )
                self.parameter_manager.register_callback(
                    'use_optimization',
                    lambda value: self._safe_set_checked(checkbox, bool(value))
                )
                try:
                    value = self.parameter_manager.get_value('use_optimization')
                    checkbox.setChecked(bool(value))
                except KeyError:
                    print("Warning: Parameter use_optimization not found in parameter manager")

            # Handle comboboxes
            for name, combo in self.parameter_combos.items():
                combo.currentTextChanged.connect(
                    lambda text, name=name: self.parameter_manager.set_value(name, text)
                )

                self.parameter_manager.register_callback(
                    name,
                    lambda value, combo=combo: self._safe_set_combo_text(
                        combo, value.replace('_', '-').title() if value else ''
                    )
                )
                try:
                    value = self.parameter_manager.get_value(name)
                    if value:
                        self._safe_set_combo_text(combo, value.replace('_', '-').title())
                except KeyError:
                    print(f"Warning: Parameter {name} not found in parameter manager")

            # Handle visualization checkboxes
            for viz_name, checkbox in self.visualization_checkboxes.items():
                param_name = f'save_{viz_name}'
                checkbox.stateChanged.connect(
                    lambda state, name=param_name: self.parameter_manager.set_value(
                        name, state == Qt.Checked
                    )
                )
                self.parameter_manager.register_callback(
                    param_name,
                    lambda value, cb=checkbox: self._safe_set_checked(cb, bool(value))
                )
                try:
                    value = self.parameter_manager.get_value(param_name)
                    self._safe_set_checked(checkbox, bool(value))
                except KeyError:
                    self.parameter_manager.set_value(param_name, False)
                    self._safe_set_checked(checkbox, False)

        finally:
            # Restore signal handling
            self._block_parameter_widgets(False)

    def _safe_set_value(self, widget, value):
        """Safely set widget value with signal blocking."""
        if value is not None:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _safe_set_combo_text(self, combo, text):
        """Safely set combo box text with signal blocking."""
        combo.blockSignals(True)
        index = combo.findText(text, Qt.MatchFixedString)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _safe_set_checked(self, checkbox, checked):
        """Safely set checkbox state with signal blocking."""
        checkbox.blockSignals(True)
        checkbox.setChecked(checked)
        checkbox.blockSignals(False)

    def _update_calibration(self):
        """Update widget based on calibration parameters."""
        try:
            self._block_parameter_widgets(True)

            # Get values from parameter manager
            pixel_size = self.parameter_manager.get_value('pixel_size')
            frame_interval = self.parameter_manager.get_value('frame_interval')

            # Update spinboxes
            self._safe_set_value(self.parameter_spins['pixel_size'], pixel_size)
            self._safe_set_value(self.parameter_spins['frame_interval'], frame_interval)

        except KeyError as e:
            print(f"Warning: Calibration parameter not found: {str(e)}")
        finally:
            self._block_parameter_widgets(False)

    def _create_general_params_group(self) -> QGroupBox:
        """Create general parameters group."""
        group = QGroupBox("General Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Pixel size
        pixel_row = QHBoxLayout()
        pixel_row.addWidget(QLabel("Pixel Size (µm):"))
        pixel_spin = QDoubleSpinBox()
        pixel_spin.setRange(0.01, 10.0)
        pixel_spin.setSingleStep(0.01)
        pixel_spin.setDecimals(2)
        pixel_spin.setToolTip("Physical size of each pixel in micrometers")
        self.parameter_spins['pixel_size'] = pixel_spin
        pixel_row.addWidget(pixel_spin)
        layout.addLayout(pixel_row)

        # Frame interval
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame Length (min):"))
        frame_spin = QDoubleSpinBox()
        frame_spin.setRange(0.001, 1000.0)
        frame_spin.setSingleStep(0.1)
        frame_spin.setDecimals(1)
        frame_spin.setToolTip("Time between consecutive frames in minutes")
        self.parameter_spins['frame_interval'] = frame_spin
        frame_row.addWidget(frame_spin)
        layout.addLayout(frame_row)

        group.setLayout(layout)
        return group

    def _create_preprocessing_params_group(self) -> QGroupBox:
        """Create preprocessing parameters group."""
        group = QGroupBox("Preprocessing Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Bead/Reference parameters
        bead_params = [
            ("min_intensity", "Min Intensity (%)", 0, 100, 0.1, 0),
            ("max_intensity", "Max Intensity (%)", 0, 100, 0.1, 100),
            ("gaussian_sigma", "Gaussian Sigma", 0.0, 10.0, 0.1, 0.0)
        ]

        for name, label, min_val, max_val, step, default in bead_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setDecimals(1)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Cell parameters
        cell_params = [
            ("cell_min_intensity", "Cell Min Intensity (%)", 0, 100, 0.1, 0),
            ("cell_max_intensity", "Cell Max Intensity (%)", 0, 100, 0.1, 100),
            ("cell_gaussian_sigma", "Cell Gaussian Sigma", 0.0, 10.0, 0.1, 0.0)
        ]

        for name, label, min_val, max_val, step, default in cell_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setDecimals(1)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Registration mode
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Registration Mode:"))
        reg_combo = QComboBox()
        reg_combo.addItems(['Translation', 'Rigid', 'No registration'])
        reg_combo.setToolTip("Choose registration method")
        self.parameter_combos['registration_mode'] = reg_combo
        reg_row.addWidget(reg_combo)
        layout.addLayout(reg_row)

        group.setLayout(layout)
        return group

    def _create_displacement_params_group(self) -> QGroupBox:
        """Create displacement analysis parameters group."""
        group = QGroupBox("Displacement Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Optical flow parameters
        flow_params = [
            ("tau", "Tau:", 0.01, 1.0, 0.01, 0.25),
            ("lambda_", "Lambda:", 0.01, 1.0, 0.01, 0.4),
            ("theta", "Theta:", 0.1, 1.0, 0.1, 0.3),
            ("nscales", "Pyramid Scales:", 1, 10, 1, 3),
            ("warps", "Warps:", 1, 10, 1, 3),
            ("epsilon", "Epsilon:", 0.001, 0.1, 0.001, 0.01),
            ("inner_iterations", "Inner Iterations:", 1, 50, 1, 15),
            ("outer_iterations", "Outer Iterations:", 1, 20, 1, 5),
            ("scale_step", "Scale Step:", 0.1, 0.99, 0.01, 0.5),
            ("median_filtering", "Median Filter:", 1, 9, 2, 5),
            ("downscale_factor", "Downscale Factor:", 1, 10, 1, 1)
        ]

        for name, label, min_val, max_val, step, default in flow_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            if isinstance(step, int):
                spin = QSpinBox()
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(2)
            if name == "epsilon":
                spin.setDecimals(3)
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Visualization parameters
        vis_params = [
            ("disp_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("disp_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("d_max", "Max Displacement (µm):", 0.1, 200.0, 0.1, 5.0)
        ]

        for name, label, min_val, max_val, step, default in vis_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_force_params_group(self) -> QGroupBox:
        """Create force analysis parameters group."""
        group = QGroupBox("Force Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Material parameters setup
        material_params = [
            ("young_modulus", "Young's Modulus (kPa):", 0.1, 1000, 0.1, 10),
            ("poisson_ratio", "Poisson's Ratio:", 0, 0.5, 0.01, 0.49),
            ("gel_height", "Gel Height (µm):", 0, 1000, 10, 0)
        ]

        for name, label, min_val, max_val, step, default in material_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if name == "gel_height":
                spin.setSpecialValueText("∞")
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Lanczos exponent (integer spinbox)
        lanczos_row = QHBoxLayout()
        lanczos_row.addWidget(QLabel("Lanczos Exponent:"))
        lanczos_spin = QSpinBox()  # Changed to QSpinBox
        lanczos_spin.setRange(0, 5)
        lanczos_spin.setValue(1)
        self.parameter_spins['lanczos_exp'] = lanczos_spin
        lanczos_row.addWidget(lanczos_spin)
        layout.addLayout(lanczos_row)

        # Regularization parameter (as log10)
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Regularization (10^x):"))
        reg_spin = QDoubleSpinBox()
        reg_spin.setRange(-21, 0)
        reg_spin.setSingleStep(0.5)
        reg_spin.setValue(-4)
        reg_spin.setDecimals(1)
        self.parameter_spins['regularization'] = reg_spin
        reg_row.addWidget(reg_spin)
        layout.addLayout(reg_row)

        # Auto-GCV checkbox
        auto_gcv = QCheckBox("Auto-GCV per frame")
        self.parameter_checks['auto_gcv'] = auto_gcv
        layout.addWidget(auto_gcv)

        # Connect auto-GCV checkbox to enable/disable regularization spinbox
        def toggle_reg_spin(state):
            reg_spin.setEnabled(not state)

        auto_gcv.stateChanged.connect(toggle_reg_spin)

        # Add visualization parameters
        vis_params = [
            ("force_vector_stride", "Vector Stride:", 1, 100, 1, 20),
            ("force_arrow_scale", "Arrow Scale:", 0.1, 50.0, 0.1, 1.0),
            ("f_max", "Max Force (Pa):", 0.1, 10000.0, 1, 1000.0)
        ]

        for name, label, min_val, max_val, step, default in vis_params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_stress_params_group(self) -> QGroupBox:
        """Create stress parameters group with proper parameter manager integration."""
        group = QGroupBox("Stress Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Mask parameters
        mask_params = [
            ("threshold", "Threshold Percentile:", 0, 100, 0.1, 0,
             "Percentile threshold for cell mask generation"),
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 10,
             "Number of pixels to dilate the mask"),
            ("smoothing_sigma", "Boundary Smoothing:", 0.0, 40.0, 0.1, 10,
             "Sigma for Gaussian smoothing of mask boundary")
        ]

        for name, label, min_val, max_val, step, default, tooltip in mask_params:
            row = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setToolTip(tooltip)
            row.addWidget(label_widget)

            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            if isinstance(spin, QDoubleSpinBox):
                spin.setDecimals(1)

            self.parameter_spins[name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        # Mesh parameters
        mesh_params = [
            ("density_factor", "Density Factor:", 0.001, 0.1, 0.001, 0.025,
             "Controls mesh density. Lower values create finer meshes."),
            ("mesh_algorithm", "Mesh Algorithm:", None, None, None, "Frontal-Del.",
             "Algorithm used for mesh generation"),
        ]

        for name, label, min_val, max_val, step, default, tooltip in mesh_params:
            row = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setToolTip(tooltip)
            row.addWidget(label_widget)

            if name == "mesh_algorithm":
                combo = QComboBox()
                combo.addItems(self.MESH_ALGORITHMS.keys())
                combo.setCurrentText(default)
                self.parameter_combos[name] = combo
                row.addWidget(combo)
            else:
                spin = QDoubleSpinBox()
                spin.setRange(min_val, max_val)
                spin.setSingleStep(step)
                spin.setValue(default)
                spin.setDecimals(3)
                self.parameter_spins[name] = spin
                row.addWidget(spin)

            layout.addLayout(row)

        # Add optimization checkbox
        self.parameter_checks['use_optimization'] = QCheckBox("Mesh Optimization")
        self.parameter_checks['use_optimization'].setChecked(True)
        self.parameter_checks['use_optimization'].setToolTip(
            "Enable mesh quality optimization after generation"
        )
        layout.addWidget(self.parameter_checks['use_optimization'])

        # Add max stress visualization parameter
        poisson_row = QHBoxLayout()
        poisson_row.addWidget(QLabel("Poisson's Ratio:"))
        poisson_spin = QDoubleSpinBox()
        poisson_spin.setRange(0, 1.0)
        poisson_spin.setSingleStep(0.01)
        poisson_spin.setValue(1.0)
        poisson_spin.setDecimals(2)
        self.parameter_spins['poisson_ratio'] = poisson_spin
        poisson_row.addWidget(poisson_spin)
        layout.addLayout(poisson_row)

        stress_row = QHBoxLayout()
        stress_row.addWidget(QLabel("Max Stress (mN/m):"))
        max_stress_spin = QDoubleSpinBox()
        max_stress_spin.setRange(0.01, 1000.0)
        max_stress_spin.setSingleStep(0.1)
        max_stress_spin.setValue(1.0)
        max_stress_spin.setDecimals(2)
        self.parameter_spins['max_stress'] = max_stress_spin
        stress_row.addWidget(max_stress_spin)
        layout.addLayout(stress_row)

        group.setLayout(layout)
        return group

    def _get_parameter_dict(self) -> dict:
        """Get dictionary of current parameter values."""
        params = {}
        params.update(self.parameter_manager.get_category_parameters(ParameterCategory.GENERAL))
        params.update(self.parameter_manager.get_category_parameters(ParameterCategory.PREPROCESSING))
        params.update(self.parameter_manager.get_category_parameters(ParameterCategory.DISPLACEMENT))
        params.update(self.parameter_manager.get_category_parameters(ParameterCategory.FORCE))
        params.update(self.parameter_manager.get_category_parameters(ParameterCategory.STRESS))
        params.update(self.parameter_manager.get_category_parameters(ParameterCategory.VISUALIZATION))

        return params

    def _create_analysis_steps_group(self) -> QGroupBox:
        """Create analysis steps group with checkboxes."""
        group = QGroupBox("Analysis Steps")
        layout = QVBoxLayout()

        steps = [
            "preprocessing", "displacement", "force", "stress"
        ]

        for step in steps:
            checkbox = QCheckBox(step.capitalize())
            checkbox.setChecked(True)
            self.analysis_checkboxes[step] = checkbox
            layout.addWidget(checkbox)

        group.setLayout(layout)
        return group

    def _create_visualization_group(self) -> QGroupBox:
        """Create visualization options group."""
        group = QGroupBox("Save Visualizations")
        layout = QVBoxLayout()

        viz_options = [
            "bead_overlay",
            "displacement_map",
            "force_map",
            "force_cell_overlay",
            "sigma_xx",
            "sigma_yy",
            "shear",
            "normal_stress"
        ]

        for viz in viz_options:
            checkbox = QCheckBox(viz.replace("_", " ").title())
            self.visualization_checkboxes[viz] = checkbox
            layout.addWidget(checkbox)

        group.setLayout(layout)
        return group

    def _create_folder_management_group(self) -> QGroupBox:
        """Create folder management group."""
        group = QGroupBox("Folder Management")
        layout = QVBoxLayout()

        # Folder list first
        self.folder_list_widget = QListWidget()
        layout.addWidget(self.folder_list_widget)

        # Create grid layout for buttons
        button_layout = QGridLayout()

        # Add folder management and run buttons (bottom row)
        self.add_folder_btn = QPushButton("Add Folder")
        self.clear_folders_btn = QPushButton("Clear Folders")
        self.run_analysis_btn = QPushButton("Run Analysis")
        button_layout.addWidget(self.add_folder_btn, 1, 0)
        button_layout.addWidget(self.clear_folders_btn, 1, 1)
        button_layout.addWidget(self.run_analysis_btn, 1, 2)

        layout.addLayout(button_layout)
        group.setLayout(layout)
        return group

    def _setup_ui(self):
        """Set up the user interface."""
        # Keep existing scroll area setup
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Main container
        container = QWidget()
        main_layout = QVBoxLayout()

        # Add widget groups
        main_layout.addWidget(self._create_general_params_group())
        main_layout.addWidget(self._create_preprocessing_params_group())
        main_layout.addWidget(self._create_displacement_params_group())
        main_layout.addWidget(self._create_force_params_group())
        main_layout.addWidget(self._create_stress_params_group())
        main_layout.addWidget(self._create_analysis_steps_group())
        main_layout.addWidget(self._create_visualization_group())
        main_layout.addWidget(self._create_folder_management_group())
        main_layout.addWidget(self._create_status_frame())

        container.setLayout(main_layout)
        scroll.setWidget(container)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect widget signals."""
        # Keep existing signal connections
        self.add_folder_btn.clicked.connect(self._add_folder)
        self.clear_folders_btn.clicked.connect(self._clear_folders)
        self.run_analysis_btn.clicked.connect(self._run_batch_analysis)

    def _create_status_frame(self) -> QFrame:
        """Create status frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def _run_batch_analysis(self):
        """Run batch analysis by launching a new Python console."""
        pass

    def _add_folder(self):
        """Add folder to analysis queue."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            os.path.expanduser("~")
        )

        if folder:
            # Validate folder contents
            if self._validate_folder(folder):
                self.folder_list.append(folder)
                self.folder_list_widget.addItem(folder)
                self._update_ui_state()
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Folder",
                    "Folder must contain at least beads.tif and reference.tif"
                )

    def _clear_folders(self):
        """Clear folder list."""
        self.folder_list.clear()
        self.folder_list_widget.clear()
        self._update_ui_state()

    def _validate_folder(self, folder: str) -> bool:
        """Validate folder contains required files."""
        required_files = ['beads.tif', 'reference.tif']
        return all(os.path.exists(os.path.join(folder, f)) for f in required_files)

    def _update_ui_state(self):
        """Update UI element states."""
        has_folders = len(self.folder_list) > 0
        self.run_analysis_btn.setEnabled(has_folders)
        self.clear_folders_btn.setEnabled(has_folders)

    def _handle_error(self, error_message: str):
        """Handle error by showing message box."""
        QMessageBox.critical(self, "Error", error_message)
