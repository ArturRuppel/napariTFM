import os

import numpy as np
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget, QSizePolicy, QCheckBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame, QComboBox, QSlider,
    QProgressBar, QMessageBox, QFileDialog
)
from scipy import ndimage
from skimage.transform import resize
from napari.qt.threading import thread_worker

from napariTFM.backend.mesh_generator import MeshParameters, MeshGenerator
from napariTFM.backend.msm import MonolayerStressMicroscopy
from napariTFM.base_widget import BaseAnalysisWidget
from napariTFM.colorbar import ColorbarManager
from napariTFM.parameter_manager import ParameterCategory


class MSMWidget(BaseAnalysisWidget):
    """Widget for Monolayer Stress Microscopy analysis."""

    stress_calculated = Signal(dict)  # Emits stress analysis results

    # Define algorithm choices as a class variable
    MESH_ALGORITHMS = {
        "Frontal-Del.": 6,  # Shortened from "Frontal-Delaunay"
        "Delaunay": 5,
        "MeshAdapt": 1,
        "BAMG": 7,  # Shortened from "BAMG (experimental)"
        "FD Quads": 8,  # Shortened from "FD for Quads (experimental)"
        "Para. Pack": 9  # Shortened from "Parallelogram Packing (experimental)"
    }

    # Create a mapping for full names to show in tooltips
    ALGORITHM_FULL_NAMES = {
        "Frontal-Del.": "Frontal-Delaunay",
        "Delaunay": "Delaunay",
        "MeshAdapt": "MeshAdapt",
        "BAMG": "BAMG (experimental)",
        "FD Quads": "FD for Quads (experimental)",
        "Para. Pack": "Parallelogram Packing (experimental)"
    }

    def __init__(self, viewer, data_manager, parameter_manager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        # Store reference to parameter manager
        self.parameter_manager = parameter_manager

        # Initialize UI-specific attributes
        self.parameter_spins = {}
        self.parameter_combos = {}
        self.parameter_checks = {}
        self.colorbar_manager = ColorbarManager()
        self.analyzer = None
        self.mesh_generator = None

        # Block signals during setup
        self.blockSignals(True)
        try:
            self._setup_ui()
            self._connect_signals()

            self._update_parameters()

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

    def start_analysis(self):
        """Start the stress analysis with proper thread handling for GMSH"""
        try:
            if not self._validate_prerequisites():
                return

            self._set_controls_enabled(False)
            self._update_status("Preparing meshes...", 0)

            # Pre-generate meshes in main thread (GMSH requires main thread)
            mask_stack = self.data_manager.masks
            self.mesh_data = self._pre_generate_meshes(mask_stack)

            # Start analysis in background thread
            self._update_status("Starting stress analysis...", 0)
            worker = self.analyze_all_frames_worker()
            worker.yielded.connect(self._handle_analysis_progress)
            worker.returned.connect(self._handle_analysis_result)
            worker.errored.connect(self._handle_analysis_error)
            worker.start()

        except Exception as e:
            self._handle_error(f"Failed to start analysis: {str(e)}")
            self._set_controls_enabled(True)

    def _pre_generate_meshes(self, mask_stack):
        """Generate all meshes in the main thread before analysis"""
        meshes = []
        density_factor = self.parameter_spins['density_factor'].value()
        algorithm = self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()]
        use_optimization = self.parameter_checks['use_optimization'].isChecked()

        for frame in range(mask_stack.shape[0]):
            mesh_params = MeshParameters(
                mask=mask_stack[frame],
                density_factor=density_factor,
                algorithm=algorithm,
                use_optimization=use_optimization
            )
            generator = MeshGenerator(mesh_params)
            nodes, elements = generator.generate_mesh(mask_stack[frame])
            meshes.append((nodes, elements))

        return meshes

    @thread_worker
    def analyze_all_frames_worker(self):
        """Thread worker using pre-generated meshes"""
        tx = self.data_manager.force_field[..., 0]
        ty = self.data_manager.force_field[..., 1]
        params = self.data_manager.force_params
        num_frames = tx.shape[0]

        stress_results = []
        condition_numbers = []
        residuals = []

        for frame in range(num_frames):
            current_tx = tx[frame]
            current_ty = ty[frame]
            nodes, elements = self.mesh_data[frame]

            # Use pre-generated mesh
            analyzer = MonolayerStressMicroscopy(
                mask=self.data_manager.masks[frame],  # Not used for mesh generation
                density_factor=self.parameter_spins['density_factor'].value(),
                algorithm=self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()],
                use_optimization=self.parameter_checks['use_optimization'].isChecked(),
                poisson_ratio=self.parameter_spins['poisson_ratio_cells'].value(),
                nodes=nodes,
                elements=elements
            )

            stress_tensor, cond_num, residual = analyzer.calculate_stress_field(current_tx, current_ty)
            stress_tensor *= params['pixel_size'] * 1e-6  # Convert to N/m

            stress_results.append(stress_tensor)
            condition_numbers.append(cond_num)
            residuals.append(residual)

            progress = int((frame + 1) / num_frames * 100)
            yield {"progress": progress, "current_frame": frame + 1}

        return {
            "stress_tensor_stack": np.stack(stress_results, axis=0),
            "condition_numbers": condition_numbers,
            "residuals": residuals
        }

    def _handle_analysis_progress(self, progress_data):
        """Update progress during analysis."""
        self.progress_bar.setValue(progress_data["progress"])
        self.status_label.setText(
            f"Processing frame {progress_data['current_frame']}..."
        )

    def _handle_analysis_result(self, results):
        """Handle successful completion of analysis."""
        stress_tensor_stack = results["stress_tensor_stack"]
        params = {
            'pixel_size': self.data_manager.force_params['pixel_size'],
            'downscale_factor': self.data_manager.force_params['downscale_factor'],
            'frame_interval': self.parameter_manager.get_value('frame_interval'),
            'density_factor': self.parameter_manager.get_value('density_factor'),
            'poisson_ratio_cells': self.parameter_manager.get_value('poisson_ratio_cells'),
            'algorithm': self.parameter_manager.get_value('mesh_algorithm'),
            'use_optimization': self.parameter_manager.get_value('use_optimization'),
            'max_stress': self.parameter_manager.get_value('max_stress'),

        }
        stress_results = {
            'stress_tensor': stress_tensor_stack,
            'parameters': params
        }
        self.data_manager.set_stress_results(stress_tensor_stack, params)
        self.visualization_manager.visualize_stress_results(
            stress_results,
            max_stress=params['max_stress']
        )

        stats_text = (f"Stress analysis completed for {stress_tensor_stack.shape[0]} frames\n"
                      f"Mean condition number: {np.mean(results['condition_numbers']):.1e}\n"
                      f"Mean residual: {np.mean(results['residuals']):.1e}")
        self._update_status(stats_text, 100)
        self._handle_visualization_layers()
        self._set_controls_enabled(True)

    def _handle_analysis_error(self, exc):
        """Handle errors during analysis."""
        self._handle_error(f"Analysis error: {str(exc)}")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        """Enable/disable analysis controls."""
        self.analyze_btn.setEnabled(enabled)
        self.preview_mesh_btn.setEnabled(enabled)
        self.preview_frame_btn.setEnabled(enabled)
        self.save_stress_btn.setEnabled(enabled)

    def _validate_prerequisites(self):
        """Check required data is loaded."""
        if self.data_manager.masks is None:
            self._handle_error("No mask loaded. Please load a mask first.")
            return False
        if self.data_manager.force_field is None:
            self._handle_error("No force data available. Please calculate forces first.")
            return False
        return True

    def _connect_signals(self):
        """Connect widget signals."""
        # Existing connections
        self.load_force_btn.clicked.connect(self._load_force_data)
        self.load_mask_btn.clicked.connect(lambda _: self._load_masks())
        self.create_mask_btn.clicked.connect(self._create_mask_from_images)
        self.preview_mesh_btn.clicked.connect(self.preview_mesh)
        self.preview_frame_btn.clicked.connect(self.preview_current_frame)
        self.analyze_btn.clicked.connect(self.start_analysis)
        # self.analyze_btn.clicked.connect(self.analyze_all_frames)
        self.save_stress_btn.clicked.connect(self._save_stress_tensor)
        self.load_stress_btn.clicked.connect(self._load_stress_tensor)

        # Connect parameter changes to preview update
        self.parameter_spins['dilation'].valueChanged.connect(self._update_mask_preview)
        self.parameter_spins['smoothing_sigma'].valueChanged.connect(self._update_mask_preview)

        # Connect both threshold spinbox and slider
        threshold_spin, threshold_slider = self.parameter_spins['threshold']
        threshold_spin.valueChanged.connect(self._update_mask_preview)
        threshold_slider.valueChanged.connect(self._update_mask_preview)

        # Connect preview checkbox with immediate update
        self.parameter_spins['show_preview'].stateChanged.connect(self._handle_preview_toggle)

        # Connect frame change event from viewer
        self.viewer.dims.events.current_step.connect(self._handle_frame_change)

        self.viewer.layers.selection.events.active.connect(self._update_ui_state)
        self.viewer.layers.events.inserted.connect(self._update_ui_state)
        self.viewer.layers.events.removed.connect(self._update_ui_state)

        # Parameter connections
        self.parameter_spins['algorithm'].currentTextChanged.connect(
            lambda text: self.parameter_manager.set_value('mesh_algorithm', text)
        )

    def _create_mask_from_images(self):
        """Create masks from the cell stack using MSM mask creation."""
        try:
            # Get active layer
            active_layer = self._get_active_image_layer() or self._get_active_labels_layer()
            if active_layer is None:
                raise ValueError(...)

            # Update status
            self.status_label.setText("Creating masks...")
            self.progress_bar.setValue(20)

            # Get parameters
            dilation = self.parameter_spins['dilation'].value()
            smoothing_sigma = self.parameter_spins['smoothing_sigma'].value()
            threshold_spin, _ = self.parameter_spins['threshold']
            threshold = threshold_spin.value()

            # Convert layer data to image data
            cell_stack = (active_layer.data > 0).astype(np.uint8)

            # Get target shape and downscale factor if force data exists
            target_shape = None
            downscale_factor = 1

            # Check for force data and show warning if not available
            if self.data_manager.force_field is not None:
                tx = self.data_manager.force_field[..., 0]
                target_shape = tx.shape[1:]
                downscale_factor = self.data_manager.force_params['downscale_factor']
            else:
                QMessageBox.warning(
                    self,
                    "No Force Data",
                    "No force data loaded.\n"
                    "Mask shapes will not match force shapes.\n"
                    "Mesh preview will not correspond to mesh during stress calculations."
                )

            # Ensure proper dimensionality for cell stack
            if cell_stack.ndim == 2:
                cell_stack = cell_stack[np.newaxis, ...]

            # Create mask stacks
            analysis_mask_stack = np.zeros_like(cell_stack, dtype=bool)

            # Process each frame
            for frame in range(cell_stack.shape[0]):
                analysis_mask_stack[frame] = MonolayerStressMicroscopy.create_mask_from_image(
                    cell_stack[frame],
                    threshold_percentile=threshold,
                    dilation=dilation,
                    smoothing_sigma=smoothing_sigma
                )

            # Handle resizing for analysis and visualization
            vis_mask_stack = analysis_mask_stack

            if target_shape is not None and analysis_mask_stack.shape[1:] != target_shape:
                analysis_mask_stack = resize(
                    analysis_mask_stack.astype(float),
                    (analysis_mask_stack.shape[0], *target_shape),
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5

            if downscale_factor > 1:
                vis_shape = (
                    analysis_mask_stack.shape[0],
                    analysis_mask_stack.shape[1] * downscale_factor,
                    analysis_mask_stack.shape[2] * downscale_factor
                )
                vis_mask_stack = resize(
                    analysis_mask_stack.astype(float),
                    vis_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5

            # Store analysis masks and update visualization
            self.data_manager.set_masks(analysis_mask_stack)
            self._update_mask_visualization(vis_mask_stack)

            # Update UI state and parameters
            self._update_ui_state()
            self._update_parameters()

            # Update status
            num_frames = analysis_mask_stack.shape[0]
            shape_info = f"{analysis_mask_stack.shape[1]}x{analysis_mask_stack.shape[2]}"
            vis_shape_info = f"{vis_mask_stack.shape[1]}x{vis_mask_stack.shape[2]}"
            self.status_label.setText(
                f"Successfully created {num_frames} masks\n"
                f"Data resolution: {shape_info}\n"
                f"Display resolution: {vis_shape_info}"
            )
            self.progress_bar.setValue(100)

        except Exception as e:
            self._handle_error(f"Failed to create mask from images: {str(e)}")
            self.progress_bar.setValue(0)

    def _update_mask_preview(self):
        """Update the mask preview based on current parameters."""
        try:
            # Check if preview is enabled
            if not self.parameter_spins['show_preview'].isChecked():
                if 'Mask Preview' in self.viewer.layers:
                    self.viewer.layers.remove('Mask Preview')
                return

            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                self.status_label.setText("No active image layer found")
                return

            # Get current frame data
            current_frame = self.viewer.dims.current_step[0]
            image = active_layer.data
            if image.ndim == 3:
                image = image[current_frame]
            elif image.ndim != 2:
                self.status_label.setText("Image must be 2D or 3D")
                return

            # Get current parameters
            dilation = self.parameter_spins['dilation'].value()
            smoothing_sigma = self.parameter_spins['smoothing_sigma'].value()
            threshold_spin, _ = self.parameter_spins['threshold']
            threshold = threshold_spin.value()

            # Get target shape and downscale factor if force data exists
            target_shape = None
            downscale_factor = 1
            if self.data_manager.force_field is not None:
                tx = self.data_manager.force_field[0, :, :, 0]
                target_shape = tx.shape
                downscale_factor = self.data_manager.force_params['downscale_factor']

            # Create preview mask using MSM class methods
            _, preview_mask = MonolayerStressMicroscopy.create_mask_stack(
                image,
                threshold_percentile=threshold,
                dilation=dilation,
                smoothing_sigma=smoothing_sigma,
                target_shape=target_shape,
                downscale_factor=downscale_factor
            )

            # Ensure we get a 2D mask (create_mask_stack returns 3D)
            preview_mask = preview_mask[0] if preview_mask.ndim == 3 else preview_mask

            # Update or create preview layer
            if 'Mask Preview' in self.viewer.layers:
                self.viewer.layers['Mask Preview'].data = preview_mask
            else:
                # Create a color mapping for the labels
                colors = {
                    0: 'transparent',
                    1: [1, 1, 0, 0.5]  # Yellow with 0.5 opacity
                }

                self.viewer.add_labels(
                    data=preview_mask.astype(np.uint8),
                    name='Mask Preview'
                )
                # Set the colors after creation
                self.viewer.layers['Mask Preview'].opacity = 0.5
                self.viewer.layers['Mask Preview'].color = colors

            self.status_label.setText(f"Preview updated (Frame {current_frame})")

        except Exception as e:
            self.status_label.setText(f"Preview error: {str(e)}")
            # Remove preview layer if there's an error
            if 'Mask Preview' in self.viewer.layers:
                self.viewer.layers.remove('Mask Preview')
    def _load_masks(self, mask_stack=None):
        """
        Load and process mask data, handling shape matching and data management.

        Parameters
        ----------
        mask_stack : np.ndarray, optional
            If provided, uses this mask stack instead of loading from active layer
        """
        try:
            # If no mask_stack provided, load from active layer
            if mask_stack is None:
                active_layer = self.viewer.layers.selection.active
                if active_layer is None:
                    raise ValueError("No active layer selected")

                data = active_layer.data
                if data is None:
                    raise ValueError("Selected layer contains no data")

                # Convert data to numpy array
                data = np.array(data)

                # Handle different layer types
                if active_layer._type_string == 'labels':
                    # Handle multiple labels
                    unique_labels = np.unique(data)
                    unique_labels = unique_labels[unique_labels != 0]

                    if len(unique_labels) > 1:
                        QMessageBox.warning(
                            self,
                            "Multiple Labels Detected",
                            f"Found {len(unique_labels)} different labels. "
                            "All non-zero labels will be converted to 1."
                        )

                    mask_stack = (data > 0).astype(np.uint8)

                elif active_layer._type_string == 'image':
                    # Handle multiple intensity values
                    unique_values = np.unique(data)
                    unique_values = unique_values[unique_values != 0]

                    if len(unique_values) > 1:
                        QMessageBox.warning(
                            self,
                            "Multiple Intensity Values",
                            f"Found {len(unique_values)} different non-zero intensity values. "
                            "All non-zero values will be converted to 1."
                        )

                    mask_stack = (data > 0).astype(np.uint8)

                else:
                    raise ValueError(f"Unsupported layer type: {active_layer._type_string}")

            # Ensure proper dimensionality
            if mask_stack.ndim == 2:
                mask_stack = mask_stack[np.newaxis, ...]

            # Check if force data is available and handle shape matching
            analysis_mask_stack = mask_stack
            vis_mask_stack = mask_stack

            if self.data_manager.force_field is not None:
                tx = self.data_manager.force_field[..., 0]
                target_shape = tx.shape
                downscale_factor = self.data_manager.force_params['downscale_factor']

                # Resize mask if needed
                if target_shape is not None and mask_stack.shape[1:] != target_shape:
                    analysis_mask_stack = resize(mask_stack.astype(float), target_shape, order=0, preserve_range=True, anti_aliasing=False) > 0.5

                # Create visualization mask (potentially upscaled)
                if downscale_factor > 1:
                    vis_shape = (analysis_mask_stack.shape[0],
                                 analysis_mask_stack.shape[1] * downscale_factor,
                                 analysis_mask_stack.shape[2] * downscale_factor)
                    vis_mask_stack = resize(mask_stack.astype(float), vis_shape, order=0, preserve_range=True, anti_aliasing=False) > 0.5
            else:
                QMessageBox.warning(
                    self,
                    "No Force Data",
                    "No force data loaded.\n"
                    "Mask shapes will not match force shapes.\n"
                    "Mesh preview will not correspond to mesh during stress calculations.\n",

                )

            # Store the analysis mask in the data manager
            self.data_manager.set_masks(analysis_mask_stack)

            # Update visualization
            self._update_mask_visualization(vis_mask_stack)

            # Update UI state
            self._update_ui_state()
            self._update_parameters()

            # Update status
            num_frames = analysis_mask_stack.shape[0]
            shape_info = f"{analysis_mask_stack.shape[1]}x{analysis_mask_stack.shape[2]}"
            vis_shape_info = f"{vis_mask_stack.shape[1]}x{vis_mask_stack.shape[2]}"
            self.status_label.setText(
                f"Successfully loaded {num_frames} masks\n"
                f"Data resolution: {shape_info}\n"
                f"Display resolution: {vis_shape_info}"
            )
            self.progress_bar.setValue(100)

        except Exception as e:
            self._handle_error(f"Failed to load masks: {str(e)}")
            self.progress_bar.setValue(0)

    def _handle_preview_toggle(self, state):
        """Handle preview checkbox state changes."""
        # Store currently active layer
        active_layer = self.viewer.layers.selection.active

        if not state:  # If unchecked
            if 'Mask Preview' in self.viewer.layers:
                self.viewer.layers.remove('Mask Preview')
        else:  # If checked
            self._update_mask_preview()

        # Restore the previously active layer
        if active_layer is not None:
            self.viewer.layers.selection.active = active_layer

    def _handle_frame_change(self, event=None):
        """Handle frame change events."""
        if self.parameter_spins['show_preview'].isChecked():
            self._update_mask_preview()

    def _on_frame_changed(self, event):
        """Handle frame change events."""
        if self.parameter_spins['show_preview'].isChecked():
            self._update_mask_preview()

    def _connect_parameters(self):
        """Connect widget controls to parameter manager."""
        # Block signals during initial setup
        self._block_parameter_widgets(True)

        try:
            # Connect basic parameters
            basic_params = ['dilation', 'smoothing_sigma', 'density_factor', 'max_stress']
            for name in basic_params:
                if name in self.parameter_spins:
                    spin = self.parameter_spins[name]
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
                    try:
                        value = self.parameter_manager.get_value(name)
                        self._safe_set_value(spin, value if value is not None else 0)
                    except KeyError:
                        print(f"Warning: Parameter {name} not found in parameter manager")

            # Handle threshold parameter
            if 'threshold' in self.parameter_spins:
                threshold_widget = self.parameter_spins['threshold']
                if isinstance(threshold_widget, tuple):
                    # Handle tuple case (spinbox, slider)
                    threshold_spin, threshold_slider = threshold_widget
                    threshold_spin.valueChanged.connect(
                        lambda value: self.parameter_manager.set_value('threshold', value)
                    )
                    threshold_slider.valueChanged.connect(
                        lambda value: self.parameter_manager.set_value('threshold', value)
                    )
                else:
                    # Handle single widget case
                    threshold_widget.valueChanged.connect(
                        lambda value: self.parameter_manager.set_value('threshold', value)
                    )
                # Connect parameter manager to widget(s)
                self.parameter_manager.register_callback(
                    'threshold',
                    lambda value: self._safe_set_threshold(value)
                )
                # Set initial value
                try:
                    value = self.parameter_manager.get_value('threshold')
                    self._safe_set_threshold(value if value is not None else 0)
                except KeyError:
                    print("Warning: Parameter threshold not found in parameter manager")

            # Connect mesh algorithm combo box
            if 'algorithm' in self.parameter_combos:
                combo = self.parameter_combos['algorithm']
                combo.currentTextChanged.connect(
                    lambda text: self.parameter_manager.set_value(
                        'mesh_algorithm',
                        text.lower().replace('-', '_')
                    )
                )
                self.parameter_manager.register_callback(
                    'mesh_algorithm',
                    lambda value: self._safe_set_combo_text(
                        combo,
                        value.replace('_', '-').title() if value else ''
                    )
                )
                try:
                    value = self.parameter_manager.get_value('mesh_algorithm')
                    if value:
                        self._safe_set_combo_text(combo, value.replace('_', '-').title())
                except KeyError:
                    print("Warning: Parameter mesh_algorithm not found in parameter manager")

            # Connect optimization checkbox
            if 'use_optimization' in self.parameter_checks:
                checkbox = self.parameter_checks['use_optimization']
                checkbox.stateChanged.connect(
                    lambda state: self.parameter_manager.set_value(
                        'use_optimization',
                        state == Qt.Checked
                    )
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

            # Connect poisson_ratio_cells (Poisson ratio) spinbox
            if 'poisson_ratio_cells' in self.parameter_spins:
                spin = self.parameter_spins['poisson_ratio_cells']
                spin.valueChanged.connect(
                    lambda value: self.parameter_manager.set_value('poisson_ratio_cells', value)
                )
                self.parameter_manager.register_callback(
                    'poisson_ratio_cells',
                    lambda value: self._safe_set_value(spin, value if value is not None else 0.5)
                )
                try:
                    value = self.parameter_manager.get_value('poisson_ratio_cells')
                    self._safe_set_value(spin, value if value is not None else 0.5)
                except KeyError:
                    print("Warning: Parameter poisson_ratio_cells not found in parameter manager")

        finally:
            # Restore signal handling
            self._block_parameter_widgets(False)

    def _safe_set_threshold(self, value):
        """Safely set threshold value for both spinbox and slider or single widget."""
        if value is not None:
            threshold_widget = self.parameter_spins['threshold']
            if isinstance(threshold_widget, tuple):
                threshold_spin, threshold_slider = threshold_widget
                threshold_spin.blockSignals(True)
                threshold_slider.blockSignals(True)
                threshold_spin.setValue(value)
                threshold_slider.setValue(value)
                threshold_spin.blockSignals(False)
                threshold_slider.blockSignals(False)
            else:
                threshold_widget.blockSignals(True)
                threshold_widget.setValue(value)
                threshold_widget.blockSignals(False)

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

    def _sync_widget_with_parameters(self):
        """Sync widget values with parameter manager values"""
        if not hasattr(self, 'parameter_manager') or self.parameter_manager is None:
            print("Warning: No parameter manager available for syncing")
            return

        self._block_parameter_widgets(True)
        try:
            # Threshold
            threshold_spin, threshold_slider = self.parameter_spins['threshold']
            threshold_value = self.parameter_manager.get_value('threshold')
            threshold_spin.setValue(threshold_value)
            threshold_slider.setValue(threshold_value)

            # Algorithm
            algo_value = self.parameter_manager.get_value('mesh_algorithm')
            if algo_value not in self.MESH_ALGORITHMS:
                algo_value = 'Frontal-Del.'  # Default to valid value
                self.parameter_manager.set_value('mesh_algorithm', algo_value)
            self.parameter_spins['algorithm'].setCurrentText(algo_value)

            # Other parameters
            self.parameter_spins['dilation'].setValue(
                self.parameter_manager.get_value('dilation'))
            self.parameter_spins['smoothing_sigma'].setValue(
                self.parameter_manager.get_value('smoothing_sigma'))
            self.parameter_spins['density_factor'].setValue(
                self.parameter_manager.get_value('density_factor'))
            self.parameter_checks['use_optimization'].setChecked(
                self.parameter_manager.get_value('use_optimization'))
            self.parameter_spins['poisson_ratio_cells'].setValue(
                self.parameter_manager.get_value('poisson_ratio_cells'))
            self.parameter_spins['max_stress'].setValue(
                self.parameter_manager.get_value('max_stress'))

        except Exception as e:
            print(f"Error syncing parameters: {str(e)}")
        finally:
            self._block_parameter_widgets(False)

    def _update_parameters(self):
        """Update parameters in the parameter manager"""
        try:
            # Block signals temporarily
            self.blockSignals(True)

            # Update threshold
            threshold_spin, _ = self.parameter_spins['threshold']
            self.parameter_manager.set_value('threshold', threshold_spin.value())

            # Update other parameters
            self.parameter_manager.set_value('dilation',
                                             self.parameter_spins['dilation'].value())
            self.parameter_manager.set_value('smoothing_sigma',
                                             self.parameter_spins['smoothing_sigma'].value())
            self.parameter_manager.set_value('density_factor',
                                             self.parameter_spins['density_factor'].value())
            self.parameter_manager.set_value('mesh_algorithm',
                                             self.parameter_spins['algorithm'].currentText())
            self.parameter_manager.set_value('use_optimization',
                                             self.parameter_checks['use_optimization'].isChecked())
            self.parameter_manager.set_value('poisson_ratio_cells',
                                             self.parameter_spins['poisson_ratio_cells'].value())
            self.parameter_manager.set_value('max_stress',
                                             self.parameter_spins['max_stress'].value())

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self.blockSignals(False)

    def _on_parameter_changed(self, param_name: str, value: object):
        """Handle parameter changes from the parameter manager"""
        # Only update if the change didn't come from this widget
        if not self.signalsBlocked():
            self._sync_widget_with_parameters()

    def _block_parameter_widgets(self, block: bool):
        """Block or unblock signals for all parameter-related widgets"""
        widgets = [
            *self.parameter_spins['threshold'],  # Unpack tuple of spin and slider
            self.parameter_spins['dilation'],
            self.parameter_spins['smoothing_sigma'],
            self.parameter_spins['density_factor'],
            self.parameter_spins['algorithm'],
            self.parameter_checks['use_optimization'],
            self.parameter_spins['poisson_ratio_cells'],
            self.parameter_spins['max_stress']
        ]
        for widget in widgets:
            widget.blockSignals(block)

    def _get_active_image_layer(self):
        """Get the currently active image layer."""
        # First try to get the selected layer
        active_layer = self.viewer.layers.selection.active
        if active_layer is not None and active_layer._type_string == 'image':
            return active_layer

        # If no layer is selected or selected layer is not an image,
        # try to find the first image layer
        for layer in self.viewer.layers:
            if layer._type_string == 'image':
                return layer

        return None

    def _get_active_labels_layer(self):
        """Get the currently active labels layer."""
        # First try to get the selected layer
        active_layer = self.viewer.layers.selection.active
        if active_layer is not None and active_layer._type_string == 'labels':
            return active_layer

        # If no layer is selected or selected layer is not a labels layer,
        # try to find the first labels layer
        for layer in self.viewer.layers:
            if layer._type_string == 'labels':
                return layer

        return None

    def _load_force_data(self):
        """Load force data from files."""
        try:
            # Get file path
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Force Data File",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if file_path:
                # Load the force data
                force_data = np.load(file_path, allow_pickle=True).item()
                self.data_manager.set_force_results(force_data["force_field"], force_data["parameters"])

                # Update UI state
                self._update_ui_state()

                # Update status
                force_map_shape = force_data["force_field"].shape[0:3]
                shape_info = f"{force_map_shape}"
                self._update_status(
                    f"Force data successfully loaded from:\n{file_path}\n"
                    f"Force map shape: {shape_info}\n",
                    100
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load force data: {str(e)}"
            )
            # Print the full error for debugging
            import traceback
            traceback.print_exc()

    def _process_single_mask(self, image: np.ndarray, dilation: int,
                             smoothing_sigma: float) -> np.ndarray:
        """
        Process a single frame to create a mask.

        Parameters
        ----------
        image : np.ndarray
            Input image frame
        dilation : int
            Number of pixels to dilate the mask
        smoothing_sigma : float
            Sigma value for Gaussian smoothing

        Returns
        -------
        np.ndarray
            Binary mask
        """
        # Get threshold percentile value
        threshold_spin, _ = self.parameter_spins['threshold']
        percentile = threshold_spin.value()

        # Convert image to float for consistent processing
        image_float = image.astype(float)

        # Calculate threshold value based on percentile
        # Ignore zero values when calculating percentile
        if percentile > 0:
            nonzero_mask = image_float > 0
            if np.any(nonzero_mask):
                threshold_value = np.percentile(image_float[nonzero_mask], percentile)
                thresholded_image = np.where(image_float > threshold_value, image_float, 0)
            else:
                thresholded_image = image_float
        else:
            thresholded_image = image_float

        # Basic thresholding to create binary mask
        mask = thresholded_image > 0

        # Fill holes
        filled_mask = ndimage.binary_fill_holes(mask)

        # Smoothing
        if smoothing_sigma > 0:
            float_mask = filled_mask.astype(float)
            smoothed = ndimage.gaussian_filter(float_mask, sigma=smoothing_sigma)
            smoothed_mask = smoothed > 0.5
            smoothed_mask = ndimage.binary_fill_holes(smoothed_mask)
        else:
            smoothed_mask = filled_mask

        # Dilation
        if dilation > 0:
            struct = ndimage.generate_binary_structure(2, 2)
            dilated_mask = ndimage.binary_dilation(
                smoothed_mask,
                structure=struct,
                iterations=dilation
            )
        else:
            dilated_mask = smoothed_mask

        # Get largest connected component
        labels, num_features = ndimage.label(dilated_mask)
        if num_features > 0:
            sizes = ndimage.sum(dilated_mask, labels, range(1, num_features + 1))
            largest_feature = np.argmax(sizes) + 1
            final_mask = labels == largest_feature
        else:
            final_mask = dilated_mask

        return final_mask

    def reset_parameters(self):
        """Reset MSM-specific parameters to defaults."""
        try:
            # Reset only stress parameters
            self.parameter_manager.reset_category_to_defaults(ParameterCategory.STRESS)

            # Synchronize widget values with reset parameters
            self._sync_widget_with_parameters()

            # Update mask preview with new parameters
            self._update_mask_preview()

            # Update UI state in case any changes affect button states
            self._update_ui_state()

            self._update_status("Stress parameters reset to defaults", 100)

        except Exception as e:
            self._handle_error(f"Error resetting parameters: {str(e)}")

    def _update_mask_visualization(self, vis_mask_stack: np.ndarray):
        """Update the mask visualization in napari."""
        if 'Masks' in self.viewer.layers:
            self.viewer.layers.remove('Masks')

        self.viewer.add_labels(
            vis_mask_stack.astype(np.uint8),
            name='Masks',
            visible=True,
            opacity=0.5,
        )

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

        # Right side container
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Add main UI elements
        right_layout.addWidget(self._create_data_loading_group())
        right_layout.addWidget(self._create_parameters_group())
        right_layout.addWidget(self._create_action_buttons())
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(360)

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

    def _update_ui_state(self):
        """Update UI element states based on current data availability."""
        # First update mask-related button states
        # Check for valid layers
        has_valid_layer = False
        active_layer = self.viewer.layers.selection.active

        if active_layer is not None:
            # Check if active layer is an image or labels layer
            if active_layer._type_string in ['image', 'labels']:
                has_valid_layer = True

        # If no active layer, check if there are any valid layers
        if not has_valid_layer:
            for layer in self.viewer.layers:
                if layer._type_string in ['image', 'labels']:
                    has_valid_layer = True
                    break

        # Update button states
        if hasattr(self, 'create_mask_btn'):
            self.create_mask_btn.setEnabled(has_valid_layer)
        if hasattr(self, 'load_mask_btn'):
            self.load_mask_btn.setEnabled(has_valid_layer)

        # Update force data status
        has_force_data = False
        if self.data_manager.force_field is not None:
            force_field = self.data_manager.force_field
            tx = force_field[..., 0]
            self.force_status.setText(f"Loaded: {tx.shape}")
            has_force_data = True
        else:
            self.force_status.setText("Not loaded")

        # Update mask status
        has_mask = False
        if self.data_manager.masks is not None:
            mask_stack = self.data_manager.masks

            mask_shape = mask_stack.shape
            self.mask_status.setText(f"Loaded: {mask_shape}")
            has_mask = True

        else:
            self.mask_status.setText("Not loaded")

        # Update analysis button states
        self.preview_mesh_btn.setEnabled(has_mask)
        self.preview_frame_btn.setEnabled(has_mask and has_force_data)
        self.analyze_btn.setEnabled(has_mask and has_force_data)
        self.save_stress_btn.setEnabled(self.data_manager.stress_tensor is not None)

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group."""
        group = QGroupBox("Input Data")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Force data row
        force_layout = QHBoxLayout()
        self.load_force_btn = QPushButton("Load Forces")
        self.load_force_btn.setToolTip("Load force data from file")
        self.force_status = QLabel("Not loaded")
        force_layout.addWidget(self.load_force_btn)
        force_layout.addWidget(self.force_status)

        # Input mask row
        mask_layout = QHBoxLayout()
        self.load_mask_btn = QPushButton("Load Masks")
        self.load_mask_btn.setToolTip("Load mask from active layer")
        self.load_mask_btn.setEnabled(False)  # Initially disabled
        self.mask_status = QLabel("Not loaded")
        mask_layout.addWidget(self.load_mask_btn)
        mask_layout.addWidget(self.mask_status)

        layout.addLayout(force_layout)
        layout.addLayout(mask_layout)

        group.setLayout(layout)
        return group

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Parameters")
        layout = QVBoxLayout()
        layout.setSpacing(8)  # Match FTTC spacing
        layout.setContentsMargins(6, 6, 6, 6)  # Match FTTC margins

        # Mask parameters
        mask_params = [
            ("threshold", "Threshold Percentile:", 0, 100, 1, 0,
             "Clip intensity values below this percentile before creating the mask."),
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 10,
             "Number of pixels to dilate the mask. Higher values create a larger boundary around the cell."),
            ("smoothing_sigma", "Boundary Smoothing:", 0.0, 40.0, 0.1, 10,
             "Gaussian smoothing sigma for the mask boundary. Higher values create smoother boundaries."),
            ("show_preview", "Show Preview:", None, None, None, False,
             "Show mask preview in real-time as parameters are adjusted"),
        ]

        # Rest of the parameters remain the same
        mesh_params = [
            ("density_factor", "Density Factor:", 0.005, 0.05, 0.001, 0.025,
             "Controls mesh density. Lower values create finer meshes with more elements."),
            ("algorithm", "Mesh Algorithm:", None, None, None, "Frontal-Del.",
             "Algorithm used for mesh generation. Frontal-Delaunay is recommended for most cases."),
            ("use_optimization", "Mesh Optimization", None, None, None, True,
             "Optimize mesh quality after generation. Check mesh quality, does not always improve results."),
        ]

        material_params = [
            ("poisson_ratio_cells", "Poisson's Ratio:", 0.0, 1.0, 0.01, 0.5,
             "Material's Poisson ratio. Typical value is 0.5 for incompressible materials."),
        ]

        vis_params = [
            ("max_stress", "Max Stress (mN/m):", 0.01, 1000.0, 0.1, 1.0,
             "Maximum stress value for colormap scaling. Adjust to optimize visualization."),
        ]

        sections = [
            ("Mask Parameters", mask_params),
            ("Mesh Parameters", mesh_params),
            ("Material Parameters", material_params),
            ("Visualization Parameters", vis_params)
        ]

        for section_title, params in sections:
            section_group = QGroupBox(section_title)
            section_layout = QVBoxLayout()
            section_layout.setSpacing(4)
            section_layout.setContentsMargins(6, 6, 6, 6)  # Match FTTC margins

            for param_info in params:
                param_name, label, min_val, max_val, step, default, tooltip = param_info
                param_layout = QHBoxLayout()
                param_layout.setSpacing(10)

                # Create label with fixed width
                label_widget = QLabel(label)
                label_widget.setToolTip(tooltip)
                label_widget.setFixedWidth(150)
                param_layout.addWidget(label_widget)

                if param_name == "show_preview":
                    spin = QCheckBox()
                    spin.setChecked(default)
                    spin.setToolTip(tooltip)
                    self.parameter_spins[param_name] = spin
                    param_layout.addWidget(spin)
                    param_layout.addStretch()

                elif param_name == "threshold":
                    # Create both spinbox and slider for threshold
                    spin = QDoubleSpinBox()
                    spin.setRange(min_val, max_val)
                    spin.setDecimals(1)
                    spin.setSingleStep(step)
                    spin.setValue(default)
                    spin.setToolTip(tooltip)
                    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(min_val, max_val)
                    slider.setSingleStep(step)
                    slider.setValue(default)
                    slider.setToolTip(tooltip)
                    slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                    # Connect signals for synchronization
                    spin.valueChanged.connect(slider.setValue)
                    slider.valueChanged.connect(spin.setValue)

                    self.parameter_spins[param_name] = (spin, slider)

                    # Add to layout with correct proportions
                    param_layout.addWidget(spin)
                    param_layout.addWidget(slider)

                elif param_name == "use_optimization":
                    spin = QCheckBox()
                    spin.setChecked(default)
                    self.parameter_checks[param_name] = spin
                    param_layout.addWidget(spin)

                elif param_name == "algorithm":
                    spin = QComboBox()
                    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                    for algo_name in self.MESH_ALGORITHMS.keys():
                        spin.addItem(algo_name)
                    spin.setCurrentText(default)
                    full_names_tooltip = "Available algorithms:\n" + \
                                         "\n".join(f"• {short}: {full}"
                                                   for short, full in self.ALGORITHM_FULL_NAMES.items())
                    spin.setToolTip(tooltip + "\n\n" + full_names_tooltip)
                    self.parameter_spins[param_name] = spin
                    param_layout.addWidget(spin)

                else:
                    if isinstance(step, int):
                        spin = QSpinBox()
                    else:
                        spin = QDoubleSpinBox()
                        if param_name == "smoothing_sigma":
                            spin.setDecimals(1)
                        elif param_name == "density_factor":
                            spin.setDecimals(3)
                        else:
                            spin.setDecimals(2)
                    spin.setRange(min_val, max_val)
                    spin.setSingleStep(step)
                    spin.setValue(default)
                    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                    spin.setToolTip(tooltip)
                    self.parameter_spins[param_name] = spin
                    param_layout.addWidget(spin)

                section_layout.addLayout(param_layout)

            section_group.setLayout(section_layout)
            layout.addWidget(section_group)

        # Add reset parameters button
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.reset_params_btn.setToolTip("Reset all parameters to their default values")
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        layout.addWidget(self.reset_params_btn)

        layout.addStretch(1)  # Add stretch at the end only
        group.setLayout(layout)

        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        button_grid = QHBoxLayout()

        # Create left column (Create Mask and Preview)
        left_column = QVBoxLayout()
        self.create_mask_btn = QPushButton("Create Masks from Images")
        self.create_mask_btn.setToolTip("Generate a stack of masks from the active image layer using current intensity thresholding, dilation and smoothing settings")
        self.preview_frame_btn = QPushButton("Preview Current Frame")
        self.preview_frame_btn.setToolTip("Calculate and visualize stress field for the current frame")
        left_column.addWidget(self.create_mask_btn)
        left_column.addWidget(self.preview_frame_btn)

        # Create right column (Preview Mesh and Analyze)
        right_column = QVBoxLayout()
        self.preview_mesh_btn = QPushButton("Preview Mesh")
        self.preview_mesh_btn.setToolTip("Generate and display the finite element mesh for the current frame")
        self.analyze_btn = QPushButton("Calculate Stress Tensors")
        self.analyze_btn.setToolTip("Calculate stress fields for all frames in the dataset")
        right_column.addWidget(self.preview_mesh_btn)
        right_column.addWidget(self.analyze_btn)

        button_grid.addLayout(left_column)
        button_grid.addLayout(right_column)

        # Add save/load buttons in a row
        save_load_layout = QHBoxLayout()
        self.save_stress_btn = QPushButton("Save Stress Tensor")
        self.save_stress_btn.setToolTip("Save calculated stress tensor data to file")
        self.load_stress_btn = QPushButton("Load Stress Tensor")
        self.load_stress_btn.setToolTip("Load previously saved stress tensor data")
        save_load_layout.addWidget(self.save_stress_btn)
        save_load_layout.addWidget(self.load_stress_btn)

        layout.addLayout(button_grid)
        layout.addLayout(save_load_layout)

        frame.setLayout(layout)
        return frame

    def _create_status_frame(self) -> QFrame:
        """Create the status and progress frame."""
        frame = QFrame()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        frame.setLayout(layout)
        return frame

    def preview_current_frame(self):
        """Preview stress calculation for the current frame."""
        try:
            # Check prerequisites
            mask_stack = self.data_manager.masks
            if mask_stack is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            if self.data_manager.force_field is None:
                raise ValueError("No force data available. Please calculate forces first.")

            # Get current frame index and data
            current_frame = self.viewer.dims.current_step[0]
            tx = self.data_manager.force_field[..., 0][current_frame]
            ty = self.data_manager.force_field[..., 1][current_frame]

            # Get downscale factor from force results
            params = self.data_manager.force_params
            downscale_factor = params['downscale_factor']

            # Ensure mask matches force data shape
            current_mask = mask_stack[current_frame]
            if current_mask.shape != tx.shape:
                current_mask = resize(
                    current_mask.astype(float),
                    tx.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5

            # Create new analyzer instance with current frame's mask
            pixel_size = params['pixel_size']
            if pixel_size is None:
                raise ValueError("Pixel size not available in force results")

            self.analyzer = MonolayerStressMicroscopy(mask=current_mask, density_factor=self.parameter_spins['density_factor'].value(),
                                                      algorithm=self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()],
                                                      use_optimization=self.parameter_checks['use_optimization'].isChecked(), poisson_ratio=self.parameter_spins['poisson_ratio_cells'].value(),
                                                      young_modulus=1.0)

            self._update_status("Calculating stress field...", 20)

            # Calculate stress tensor
            stress_tensor, condition_number, residual = self.analyzer.calculate_stress_field(tx, ty)
            stress_tensor = stress_tensor * params['pixel_size'] * params['downscale_factor'] * 1e-6 # convert to N/m

            self._update_status("Updating visualization...", 80)

            # Get max stress parameter and visualization parameters
            max_stress = self.parameter_spins['max_stress'].value()
            self.visualization_manager.visualize_stress_preview(
                stress_tensor,  # Get just the tensor from the tuple
                max_stress,
                downscale_factor=downscale_factor
            )

            # Update status with condition number and residual if available
            status_text = f"Stress preview generated for frame {current_frame}"
            status_text += f"\nCondition number: {condition_number:.1e}"
            status_text += f"\nResidual: {residual:.1e}"

            self._handle_visualization_layers()

            self._update_status(status_text, 100)

        except Exception as e:
            self._handle_error(f"Failed to preview stress field: {str(e)}")
            self.progress_bar.setValue(0)

    def analyze_all_frames(self):
        """Run stress analysis for all frames."""
        try:
            # Validate prerequisites
            mask_stack = self.data_manager.masks
            if mask_stack is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            if self.data_manager.force_field is None:
                raise ValueError("No force data available. Please calculate forces first.")

            # Get force data and parameters
            tx = self.data_manager.force_field[..., 0]
            ty = self.data_manager.force_field[..., 1]
            params = self.data_manager.force_params
            downscale_factor = params['downscale_factor']
            pixel_size = params['pixel_size']

            num_frames = tx.shape[0]

            # Initialize results storage
            stress_results = []
            condition_numbers = []
            residuals = []

            # Process each frame
            for frame in range(num_frames):
                self._update_status(
                    f"Processing frame {frame + 1}/{num_frames}...",
                    int((frame / num_frames) * 100)
                )

                # Get current frame's data
                current_tx = tx[frame]
                current_ty = ty[frame]

                # Ensure mask matches force data shape
                current_mask = mask_stack[frame]
                if current_mask.shape != current_tx.shape:
                    current_mask = resize(
                        current_mask.astype(float),
                        current_tx.shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False
                    ) > 0.5

                # Update analyzer with current frame's mask
                self.analyzer = MonolayerStressMicroscopy(mask=current_mask, density_factor=self.parameter_spins['density_factor'].value(),
                                                          algorithm=self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()],
                                                          use_optimization=self.parameter_checks['use_optimization'].isChecked(), poisson_ratio=self.parameter_spins['poisson_ratio_cells'].value(),
                                                          young_modulus=1.0)

                # Calculate stress tensor
                stress_tensor, condition_number, residual = self.analyzer.calculate_stress_field(current_tx, current_ty)

                # Store results
                stress_tensor = stress_tensor * params['pixel_size'] * 1e-6 # convert to N/m
                stress_results.append(stress_tensor)

                condition_numbers.append(condition_number)
                residuals.append(residual)

            # Convert results to numpy array
            stress_tensor_stack = np.stack(stress_results, axis=0)

            # Store results in data manager with all parameters
            max_stress = self.parameter_manager.get_value('max_stress')
            params = {
                    'pixel_size': pixel_size,
                    'downscale_factor': downscale_factor,
                    'frame_interval': self.parameter_manager.get_value('frame_interval'),
                    'young_modulus': self.analyzer.E,
                    'poisson_ratio_cells': self.analyzer.poisson_ratio,
                    'density_factor': self.parameter_manager.get_value('density_factor'),
                    'algorithm': self.parameter_manager.get_value('mesh_algorithm'),
                    'use_optimization': self.parameter_manager.get_value('use_optimization'),
                    'max_stress': max_stress,

                }

            self.data_manager.set_stress_results(stress_tensor_stack, params)


            stress_results = {
                'stress_tensor': stress_tensor_stack,
                # 'condition_numbers': np.array(condition_numbers) if condition_numbers else None,
                # 'residuals': np.array(residuals) if residuals else None,
                'parameters': params
            }

            # Emit results
            self.stress_calculated.emit(stress_results)

            # Update visualization
            self._update_status("Updating visualization...", 90)
            self.visualization_manager.visualize_stress_results(
                stress_results,
                max_stress=max_stress
            )

            # Final status update with metrics if available
            status_text = f"Stress analysis completed for {num_frames} frames"
            if condition_numbers:
                avg_condition = np.mean(condition_numbers)
                avg_residual = np.mean(residuals)
                status_text += f"\nMean condition number: {avg_condition:.1e}"
                status_text += f"\nMean residual: {avg_residual:.1e}"

            self._handle_visualization_layers()

            self._update_status(status_text, 100)

        except Exception as e:
            self._handle_error(f"Failed to analyze frames: {str(e)}")
            self.progress_bar.setValue(0)

    def preview_mesh(self):
        """Generate and display preview of the triangular mesh for the current frame."""
        try:
            # Get mask from data manager
            mask_stack = self.data_manager.masks
            if mask_stack is None:
                raise ValueError("No mask loaded. Please create or load a mask first.")

            # Get current frame and corresponding mask
            current_frame = self.viewer.dims.current_step[0]
            current_mask = mask_stack[current_frame]

            self._update_parameters()

            downscale_factor = 1
            forces = self.data_manager.force_field
            if forces is not None:
                downscale_factor = self.data_manager.force_params['downscale_factor']
                target_shape = forces[current_frame, :, :, 0].shape
                current_mask = resize(current_mask.astype(float), target_shape, order=0) > 0.5

            # Initialize mesh parameters
            mesh_params = MeshParameters(
                mask=current_mask,
                density_factor=self.parameter_spins['density_factor'].value(),
                algorithm=self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()],
                use_optimization=self.parameter_checks['use_optimization'].isChecked()
            )

            # Initialize mesh generator
            self.mesh_generator = MeshGenerator(mesh_params)

            self._update_status("Generating mesh...", 20)

            # Generate mesh
            nodes, elements = self.mesh_generator.generate_mesh(current_mask)

            self._update_status("Creating visualization...", 50)

            # Scale up the node coordinates
            nodes_scaled = nodes * downscale_factor

            # Remove existing mesh layers
            for layer_name in ['Mesh Edges', 'Mesh Nodes']:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)

            # Swap x and y coordinates for display
            nodes_display = np.column_stack((nodes_scaled[:, 1], nodes_scaled[:, 0]))

            # Create edge data for visualization
            num_elements = len(elements)
            edge_data = np.zeros((num_elements * 3, 2, 2))

            for i, element in enumerate(elements):
                # Get node coordinates for triangle vertices
                v1 = nodes_display[element[0]]
                v2 = nodes_display[element[1]]
                v3 = nodes_display[element[2]]

                # Add three edges (v1-v2, v2-v3, v3-v1)
                edge_data[i * 3] = np.array([v1, v2])
                edge_data[i * 3 + 1] = np.array([v2, v3])
                edge_data[i * 3 + 2] = np.array([v3, v1])

            # Add visualization layers
            self.viewer.add_shapes(
                edge_data,
                shape_type='line',
                edge_color='yellow',
                edge_width=1,
                opacity=0.6,
                name='Mesh Edges'
            )

            self.viewer.add_points(
                nodes_display,
                size=4,
                face_color='red',
                opacity=0.7,
                name='Mesh Nodes'
            )

            # Calculate and display mesh quality metrics
            quality_metrics = self.mesh_generator.analyze_mesh_quality(nodes, elements)

            status_text = (
                f"Mesh generated for frame {current_frame}:\n"
                f"{quality_metrics['n_elements']} elements\n"
                f"Min angle: {quality_metrics['min_angle']:.1f}°\n"
                f"Mean quality: {quality_metrics['mean_quality']:.3f}"
            )
            self._update_status(status_text, 100)

            # Enable preview frame button if we have force data
            self.preview_frame_btn.setEnabled(self.data_manager.force_field is not None)

        except Exception as e:
            self._handle_error(f"Failed to preview mesh: {str(e)}")
            self.progress_bar.setValue(0)

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.colorbar_manager is not None:
                self.colorbar_manager.cleanup()
                self.colorbar_manager = None
        except Exception:
            pass

        super().cleanup()

    def _save_stress_tensor(self):
        """Save stress tensor data to files."""
        if self.data_manager.stress_tensor is None:
            QMessageBox.warning(self, "Warning", "No stress tensor data to save.")
            return

        try:
            # Get file path to save
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Stress Tensor Data",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if file_path:
                # Add .npy extension if not present
                if not file_path.endswith('.npy'):
                    file_path += '.npy'

                # Package results with all necessary parameters
                stress_results = {
                    'stress_tensor': self.data_manager.stress_tensor,
                    'parameters': self.data_manager.stress_params
                    }

                # Save file
                np.save(file_path, stress_results)

                self._update_status(f"Stress tensor data successfully saved to:\n{file_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save stress tensor data: {str(e)}"
            )

    def _load_stress_tensor(self):
        """Load stress tensor data from file."""
        try:
            # Get file to load
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Stress Tensor Data File",
                os.path.expanduser("~"),
                "NumPy Files (*.npy)"
            )

            if file_path:
                # Load the stress tensor data
                stress_data = np.load(file_path, allow_pickle=True).item()

                # Validate the loaded data structure
                required_fields = ['stress_tensor', 'parameters']
                if not all(field in stress_data for field in required_fields):
                    raise ValueError("Invalid stress tensor data format")

                parameters = stress_data['parameters']

                # Update UI
                self.parameter_spins['density_factor'].setValue(parameters['density_factor'])
                self.parameter_spins['poisson_ratio_cells'].setValue(parameters['poisson_ratio_cells'])
                self.parameter_spins['max_stress'].setValue(parameters['max_stress'])
                self.parameter_spins['algorithm'].setCurrentText(parameters['algorithm'])
                self.parameter_checks['use_optimization'].setChecked(parameters['use_optimization'])
                self._update_parent_calibration(parameters['pixel_size'], parameters.get('frame_interval', 1.0))



                # Update data manager and visualization
                self.data_manager.set_stress_results(stress_data['stress_tensor'], stress_data['parameters'])
                self.visualization_manager.visualize_stress_results(
                    stress_data,
                    max_stress=parameters['max_stress']
                )

                # Update colorbar with loaded max_stress
                self.colorbar_manager.update_limits(-parameters['max_stress'], parameters['max_stress'])

                # Enable save button
                self.save_stress_btn.setEnabled(True)

                # Emit the stress_calculated signal with the results
                self.stress_calculated.emit(stress_data)

                self._handle_visualization_layers()

                self._update_status(f"Stress tensor data successfully loaded from:\n{file_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load stress tensor data: {str(e)}"
            )

    def _handle_visualization_layers(self):
        """Handle layer visibility and ordering for stress visualization."""
        from qtpy.QtCore import QTimer

        def update_visibility():
            # Track indices of stress layers
            avg_normal_index = None
            xx_index = None
            yy_index = None

            # First pass: collect indices and set visibility
            for i, layer in enumerate(self.viewer.layers):
                # Default all layers to invisible
                layer.visible = False

                # Keep track of indices and set desired visibility
                if layer.name == 'Average Normal Stress':
                    layer.visible = True
                    avg_normal_index = i
                elif layer.name == 'Normal Stress XX':
                    xx_index = i
                elif layer.name == 'Normal Stress YY':
                    yy_index = i

            # Move layers to desired order
            if avg_normal_index is not None:
                self.viewer.layers.move(avg_normal_index, -1)  # Move to top
            if xx_index is not None:
                self.viewer.layers.move(xx_index, -2)  # Move second from top
            if yy_index is not None:
                self.viewer.layers.move(yy_index, -3)  # Move third from top

        # Wait a brief moment for layers to be created
        QTimer.singleShot(10, update_visibility)

    def _update_parent_calibration(self, pixel_size: float, frame_interval: float):
        """Update calibration values in parent widget."""
        try:
            # Find parent widget instance (napariTFMWidget)
            parent = self
            while parent is not None:
                if hasattr(parent, 'pixel_spin') and hasattr(parent, 'frame_spin'):
                    break
                parent = parent.parent()

            if parent is not None:
                # Update calibration values
                parent.pixel_spin.setValue(pixel_size)
                parent.frame_spin.setValue(frame_interval)
            else:
                self._update_status("Warning: Could not update calibration in parent widget", 100)

        except Exception as e:
            self._handle_error(f"Failed to update calibration: {str(e)}")