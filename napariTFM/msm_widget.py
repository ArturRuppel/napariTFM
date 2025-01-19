import os

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame, QComboBox,
    QProgressBar, QMessageBox, QFileDialog
)
from qtpy.QtCore import Signal, Qt
import numpy as np

from .base_widget import BaseAnalysisWidget
from .colorbar import ColorbarManager
from .mesh_generator import MeshParameters, MeshGenerator
from .msm import MonolayerStressMicroscopy


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

    def __init__(self, viewer, data_manager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize parameters
        self.parameter_spins = {}
        self._pixel_size = None  # Will be set from data manager
        self._downscale_factor = None  # Will be set from data manager
        self.current_mask = None
        self.mesh = None
        self.colorbar_manager = ColorbarManager()

        # Initialize analyzer with default parameters (will be updated when needed)
        self.analyzer = None
        self.mesh_generator = None

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def _create_mask_from_images(self):
        """Create masks from the cell stack using dilation and smoothing."""
        try:
            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                raise ValueError("No active image layer selected")

            # Check if force data is available and get target shape
            target_shape = None
            downscale_factor = 1
            if self.data_manager.force_results is not None:
                tx = self.data_manager.force_results['tx'][0]  # Use first frame
                target_shape = tx.shape
                downscale_factor = self.data_manager.force_results.get('parameters', {}).get('downscale_factor', 1)
            else:
                # Show warning dialog
                from qtpy.QtWidgets import QMessageBox
                response = QMessageBox.warning(
                    self,
                    "No Force Data",
                    "Creating masks without loading force data may lead to inconsistent behavior. "
                    "The masks may need to be regenerated after loading force data. "
                    "Do you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if response == QMessageBox.No:
                    return

            # Update status
            self.status_label.setText("Creating masks...")
            self.progress_bar.setValue(20)

            # Get parameters
            dilation = self.parameter_spins['dilation'].value()
            smoothing_sigma = self.parameter_spins['smoothing_sigma'].value()

            # Process the image stack
            from scipy import ndimage
            from skimage.transform import resize

            cell_stack = active_layer.data
            if cell_stack.ndim == 2:
                cell_stack = cell_stack[np.newaxis, ...]

            # Create mask stack at original size first
            mask_stack = np.zeros_like(cell_stack, dtype=bool)

            for frame in range(cell_stack.shape[0]):
                # Basic thresholding
                mask = cell_stack[frame] > 0

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

                # Get largest connected component last
                labels, num_features = ndimage.label(dilated_mask)
                if num_features > 0:
                    sizes = ndimage.sum(dilated_mask, labels, range(1, num_features + 1))
                    largest_feature = np.argmax(sizes) + 1
                    final_mask = labels == largest_feature
                else:
                    final_mask = dilated_mask

                mask_stack[frame] = final_mask

            # First resize to match force data resolution if target shape exists
            if target_shape is not None and mask_stack.shape[1:] != target_shape:
                resized_stack = np.zeros((mask_stack.shape[0], *target_shape), dtype=bool)
                for frame in range(mask_stack.shape[0]):
                    resized_stack[frame] = resize(
                        mask_stack[frame].astype(float),
                        target_shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False
                    ) > 0.5
                mask_stack = resized_stack

            # Store the mask at the force data resolution
            self.current_mask = mask_stack

            # Create visualization mask (potentially upscaled)
            vis_mask = mask_stack.copy()
            if downscale_factor > 1:
                vis_shape = (mask_stack.shape[1] * downscale_factor,
                             mask_stack.shape[2] * downscale_factor)
                upscaled_stack = np.zeros((mask_stack.shape[0], *vis_shape), dtype=bool)
                for frame in range(mask_stack.shape[0]):
                    upscaled_stack[frame] = resize(
                        mask_stack[frame].astype(float),
                        vis_shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False
                    ) > 0.5
                vis_mask = upscaled_stack

            # Add visualization layer
            if 'Cell Mask' in self.viewer.layers:
                self.viewer.layers.remove('Cell Mask')

            self.viewer.add_labels(
                vis_mask.astype(np.uint8),
                name='Cell Mask',
                visible=True,
                opacity=0.5,
            )

            # Update status
            num_frames = self.current_mask.shape[0]
            shape_info = f"{self.current_mask.shape[1]}x{self.current_mask.shape[2]}"
            vis_shape_info = f"{vis_mask.shape[1]}x{vis_mask.shape[2]}"
            self.status_label.setText(
                f"Successfully created masks for {num_frames} frames\n"
                f"Data resolution: {shape_info}\n"
                f"Display resolution: {vis_shape_info}"
            )
            self.progress_bar.setValue(100)

            # Update UI state
            self._update_ui_state()
            self._update_parameters()  # Initialize analyzer with new mask

        except Exception as e:
            self._handle_error(f"Failed to create mask from images:\n{str(e)}")
            self.progress_bar.setValue(0)

    def preview_current_frame(self):
        """Preview stress calculation for the current frame."""
        try:
            # Check prerequisites
            if self.current_mask is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            if self.data_manager.force_results is None:
                raise ValueError("No force data available. Please calculate forces first.")

            # Get current frame index and data
            current_frame = self.viewer.dims.current_step[0]
            tx = self.data_manager.force_results['tx'][current_frame]
            ty = self.data_manager.force_results['ty'][current_frame]

            # Get downscale factor from force results
            params = self.data_manager.force_results.get('parameters', {})
            downscale_factor = params.get('downscale_factor', 1)

            # Ensure mask matches force data shape
            current_mask = self.current_mask[current_frame]
            if current_mask.shape != tx.shape:
                from skimage.transform import resize
                current_mask = resize(
                    current_mask.astype(float),
                    tx.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5

            # Create new analyzer instance with current frame's mask
            pixel_size = params.get('pixel_size', self._pixelsize)
            if pixel_size is None:
                raise ValueError("Pixel size not available in force results")

            self.analyzer = MonolayerStressMicroscopy(
                mask=current_mask,  # Use resized mask
                pixelsize=pixel_size * downscale_factor * 1e-6,  # Convert to meters
                sigma=self.parameter_spins['sigma'].value(),
                youngs_modulus=1.0,
                density_factor=self.parameter_spins['density_factor'].value(),
                algorithm=self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()],
                use_optimization=self.parameter_spins['use_optimization'].isChecked()
            )

            self._update_status("Calculating stress field...", 20)

            # Calculate stress tensor
            stress_tensor = self.analyzer.calculate_stress_field(tx, ty)

            self._update_status("Updating visualization...", 80)

            # Get max stress parameter and visualization parameters
            max_stress = self.parameter_spins['max_stress'].value()
            self.visualization_manager.visualize_stress_preview(
                stress_tensor[0],  # Get just the tensor from the tuple
                max_stress,
                downscale_factor=downscale_factor
            )

            # Update status with condition number and residual if available
            status_text = f"Stress preview generated for frame {current_frame}"
            if len(stress_tensor) == 3:
                condition_number = stress_tensor[1]
                residual = stress_tensor[2]
                status_text += f"\nCondition number: {condition_number:.1e}"
                status_text += f"\nResidual: {residual:.1e}"

            self._update_status(status_text, 100)

        except Exception as e:
            self._handle_error(f"Failed to preview stress field: {str(e)}")
            self.progress_bar.setValue(0)

    def analyze_all_frames(self):
        """Run stress analysis for all frames."""
        try:
            # Validate prerequisites
            if self.current_mask is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            if self.data_manager.force_results is None:
                raise ValueError("No force data available. Please calculate forces first.")

            # Get force data and parameters
            tx = self.data_manager.force_results['tx']
            ty = self.data_manager.force_results['ty']
            params = self.data_manager.force_results.get('parameters', {})
            downscale_factor = params.get('downscale_factor', 1)
            pixel_size = params.get('pixel_size')

            if pixel_size is None:
                raise ValueError("Pixel size not available in force results")

            num_frames = len(tx)

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
                current_mask = self.current_mask[frame]
                if current_mask.shape != current_tx.shape:
                    from skimage.transform import resize
                    current_mask = resize(
                        current_mask.astype(float),
                        current_tx.shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False
                    ) > 0.5

                # Update analyzer with current frame's mask
                self.analyzer = MonolayerStressMicroscopy(
                    mask=current_mask,
                    pixelsize=pixel_size * downscale_factor * 1e-6,  # Convert to meters
                    sigma=self.parameter_spins['sigma'].value(),
                    youngs_modulus=1.0,
                    density_factor=self.parameter_spins['density_factor'].value(),
                    algorithm=self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()],
                    use_optimization=self.parameter_spins['use_optimization'].isChecked()
                )

                # Calculate stress tensor
                result = self.analyzer.calculate_stress_field(current_tx, current_ty)

                # Store results
                stress_tensor = result[0]
                stress_results.append(stress_tensor)

                if len(result) == 3:  # If metrics are available
                    condition_numbers.append(result[1])
                    residuals.append(result[2])

            # Convert results to numpy array
            stress_tensor_stack = np.stack(stress_results, axis=0)

            # Store results in data manager with all parameters
            max_stress = self.parameter_spins['max_stress'].value()
            self.data_manager.stress_results = {
                'stress_tensor': stress_tensor_stack,
                'condition_numbers': np.array(condition_numbers) if condition_numbers else None,
                'residuals': np.array(residuals) if residuals else None,
                'parameters': {
                    'pixel_size': pixel_size,
                    'youngs_modulus': self.analyzer.E,
                    'poisson_ratio': self.analyzer.sigma,
                    'density_factor': self.parameter_spins['density_factor'].value(),
                    'algorithm': self.parameter_spins['algorithm'].currentText(),
                    'use_optimization': self.parameter_spins['use_optimization'].isChecked(),
                    'max_stress': max_stress,
                    'downscale_factor': downscale_factor
                }
            }

            # Emit results
            self.stress_calculated.emit(self.data_manager.stress_results)

            # Update visualization
            self._update_status("Updating visualization...", 90)
            self.visualization_manager.visualize_stress_results(
                self.data_manager.stress_results,
                max_stress=max_stress
            )

            # Final status update with metrics if available
            status_text = f"Stress analysis completed for {num_frames} frames"
            if condition_numbers:
                avg_condition = np.mean(condition_numbers)
                avg_residual = np.mean(residuals)
                status_text += f"\nMean condition number: {avg_condition:.1e}"
                status_text += f"\nMean residual: {avg_residual:.1e}"

            self._update_status(status_text, 100)

        except Exception as e:
            self._handle_error(f"Failed to analyze frames: {str(e)}")
            self.progress_bar.setValue(0)

    def _update_parameters(self):
        """Update analysis parameters."""
        try:
            # Get current parameters from UI
            sigma = self.parameter_spins['sigma'].value()
            density_factor = self.parameter_spins['density_factor'].value()
            algorithm_name = self.parameter_spins['algorithm'].currentText()
            algorithm = self.MESH_ALGORITHMS[algorithm_name]
            use_optimization = self.parameter_spins['use_optimization'].isChecked()

            # Get pixelsize and downscale_factor from force results if available
            self._pixelsize = None
            self._downscale_factor = None
            if self.data_manager.force_results is not None:
                params = self.data_manager.force_results.get('parameters', {})
                self._pixelsize = params.get("pixel_size")
                self._downscale_factor = params.get("downscale_factor")

            # Create mesh parameters if we have a mask and force data
            if self.current_mask is not None and self._pixelsize is not None:
                # Get force data shape
                if self.data_manager.force_results is not None:
                    tx = self.data_manager.force_results['tx'][0]  # Use first frame
                    force_shape = tx.shape

                    # Check if mask needs resizing
                    if self.current_mask[0].shape != force_shape:
                        from skimage.transform import resize
                        current_mask = resize(
                            self.current_mask[0].astype(float),
                            force_shape,
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False
                        ) > 0.5
                    else:
                        current_mask = self.current_mask[0]

                else:
                    current_mask = self.current_mask[0]

                # Initialize analyzer with current parameters
                self.analyzer = MonolayerStressMicroscopy(
                    mask=current_mask,
                    pixelsize=self._pixelsize * self._downscale_factor * 1e-6,
                    sigma=sigma,
                    youngs_modulus=1.0,
                    density_factor=density_factor,
                    algorithm=algorithm,
                    use_optimization=use_optimization
                )

                # Update colorbar limits based on max_stress parameter
                max_stress = self.parameter_spins['max_stress'].value()
                self.colorbar_manager.update_limits(-max_stress, max_stress)

        except Exception as e:
            self._handle_error(f"Failed to update parameters: {str(e)}")

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Mesh generation parameters
        mesh_params = [
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 10,
             "Number of pixels to dilate the mask. Higher values create a larger boundary around the cell."),
            ("smoothing_sigma", "Boundary Smoothing:", 0.0, 40.0, 0.1, 10,
             "Gaussian smoothing sigma for the mask boundary. Higher values create smoother boundaries."),
            ("density_factor", "Density Factor:", 0.001, 0.1, 0.001, 0.025,
             "Controls mesh density. Lower values create finer meshes with more elements."),
            ("algorithm", "Mesh Algorithm:", None, None, None, "Frontal-Del.",
             "Algorithm used for mesh generation. Frontal-Delaunay is recommended for most cases."),
            ("use_optimization", "Mesh Optimization", None, None, None, True,
             "Optimize mesh quality after generation. Check mesh quality, does not always improve results."),
        ]

        # Material parameters
        material_params = [
            ("sigma", "Poisson's Ratio:", 0.0, 1.0, 0.01, 0.5,
             "Material's Poisson ratio. Typical value is 0.5 for incompressible materials."),
        ]

        # Visualization parameters
        vis_params = [
            ("max_stress", "Max Stress (mN/m):", 0.01, 1000.0, 0.1, 1.0,
             "Maximum stress value for colormap scaling. Adjust to optimize visualization."),
        ]

        # Create parameter sections with headers
        sections = [
            ("Mesh Parameters", mesh_params),
            ("Material Parameters", material_params),
            ("Visualization", vis_params)
        ]

        for section_title, params in sections:
            header = QLabel(f"<b>{section_title}</b>")
            layout.addWidget(header)

            for param_info in params:
                param_name, label, min_val, max_val, step, default, tooltip = param_info
                param_layout = QHBoxLayout()
                param_layout.setSpacing(10)

                # Create label with fixed width to align all inputs
                label_widget = QLabel(label)
                label_widget.setToolTip(tooltip)
                label_widget.setFixedWidth(150)  # Fixed width for consistent alignment
                param_layout.addWidget(label_widget)

                if param_name == "use_optimization":
                    from qtpy.QtWidgets import QCheckBox
                    spin = QCheckBox()
                    spin.setChecked(default)
                elif param_name == "algorithm":
                    spin = QComboBox()
                    # Set size policy to expand horizontally
                    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                    # Add items with shortened names
                    for algo_name in self.MESH_ALGORITHMS.keys():
                        spin.addItem(algo_name)

                    # Create detailed tooltip that shows full algorithm names
                    full_names_tooltip = "Available algorithms:\n" + \
                                         "\n".join(f"• {short}: {full}"
                                                   for short, full in self.ALGORITHM_FULL_NAMES.items())
                    spin.setToolTip(tooltip + "\n\n" + full_names_tooltip)

                    spin.setCurrentText(default)
                else:
                    if isinstance(step, int):
                        spin = QSpinBox()
                    else:
                        spin = QDoubleSpinBox()
                        spin.setDecimals(3 if param_name == "density_factor" else 2)

                    spin.setRange(min_val, max_val)
                    spin.setSingleStep(step)
                    spin.setValue(default)
                    # Set size policy to expand horizontally
                    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                if param_name != "algorithm":  # Algorithm already has a custom tooltip
                    spin.setToolTip(tooltip)

                self.parameter_spins[param_name] = spin
                param_layout.addWidget(spin)
                layout.addLayout(param_layout)

            layout.addSpacing(10)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame with compact layout."""
        frame = QFrame()
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Create buttons with tooltips
        self.create_mask_btn = QPushButton("Create Mask from Images")
        self.create_mask_btn.setToolTip("Generate a mask from the active image layer using current dilation and smoothing settings")

        self.preview_mesh_btn = QPushButton("Preview Mesh")
        self.preview_mesh_btn.setToolTip("Generate and display the finite element mesh for the current frame")

        self.preview_frame_btn = QPushButton("Preview Current Frame")
        self.preview_frame_btn.setToolTip("Calculate and visualize stress field for the current frame")

        self.analyze_btn = QPushButton("Analyze All Frames")
        self.analyze_btn.setToolTip("Calculate stress fields for all frames in the dataset")

        save_load_layout = QHBoxLayout()
        self.save_stress_btn = QPushButton("Save Stress Tensor")
        self.save_stress_btn.setToolTip("Save calculated stress tensor data to file")

        self.load_stress_btn = QPushButton("Load Stress Tensor")
        self.load_stress_btn.setToolTip("Load previously saved stress tensor data")

        save_load_layout.addWidget(self.save_stress_btn)
        save_load_layout.addWidget(self.load_stress_btn)

        layout.addWidget(self.create_mask_btn)
        layout.addWidget(self.preview_mesh_btn)
        layout.addWidget(self.preview_frame_btn)
        layout.addWidget(self.analyze_btn)
        layout.addLayout(save_load_layout)

        frame.setLayout(layout)
        return frame

    def _connect_signals(self):
        """Connect widget signals."""
        # Buttons
        self.create_mask_btn.clicked.connect(self._create_mask_from_images)
        self.preview_mesh_btn.clicked.connect(self.preview_mesh)
        self.preview_frame_btn.clicked.connect(self.preview_current_frame)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)
        self.save_stress_btn.clicked.connect(self._save_stress_tensor)
        self.load_stress_btn.clicked.connect(self._load_stress_tensor)

        # Parameters - need to handle QComboBox differently from other widgets
        for name, widget in self.parameter_spins.items():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._update_parameters)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._update_parameters)
            else:  # QCheckBox
                widget.stateChanged.connect(self._update_parameters)

    def preview_mesh(self):
        """Generate and display preview of the triangular mesh for the current frame."""
        try:
            if self.current_mask is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            current_frame = self.viewer.dims.current_step[0]
            current_mask = self.current_mask[current_frame]

            self._update_parameters()

            # Get force data shape and downscale factor
            target_shape = None
            downscale_factor = 1

            # Get downscale factor from class variables
            if self._downscale_factor is not None:
                downscale_factor = self._downscale_factor

            # Get target shape from force results if available
            if self.data_manager.force_results is not None:
                tx = self.data_manager.force_results['tx'][current_frame]
                target_shape = tx.shape

            # Resize mask if needed
            if target_shape is not None and current_mask.shape != target_shape:
                from skimage.transform import resize
                current_mask = resize(current_mask.astype(float), target_shape, order=0) > 0.5

            # Initialize mesh parameters
            mesh_params = MeshParameters(
                mask=current_mask,
                density_factor=self.parameter_spins['density_factor'].value(),
                algorithm=self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()],
                use_optimization=self.parameter_spins['use_optimization'].isChecked()
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

            # Store mesh for later use
            self.mesh = {
                'nodes': nodes,  # Store unscaled coordinates
                'elements': elements,
                'frame': current_frame,
                'scale_factor': downscale_factor
            }

            # Calculate and display mesh quality metrics
            quality_metrics = self.mesh_generator.analyze_mesh_quality(nodes, elements)

            status_text = (
                f"Mesh generated: {quality_metrics['n_elements']} elements\n"
                f"Min angle: {quality_metrics['min_angle']:.1f}°\n"
                f"Mean quality: {quality_metrics['mean_quality']:.3f}"
            )
            self._update_status(status_text, 100)

            # Enable preview frame button if we have force data
            self.preview_frame_btn.setEnabled(self.data_manager.force_results is not None)

        except Exception as e:
            self._handle_error(f"Failed to preview mesh: {str(e)}")
            self.progress_bar.setValue(0)

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create colorbar widget
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

        # Container for right side content
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Add main UI elements
        right_layout.addWidget(self._create_parameters_group())
        right_layout.addWidget(self._create_action_buttons())
        right_layout.addWidget(self._create_status_frame())
        right_layout.addStretch()

        right_container.setLayout(right_layout)
        right_container.setFixedWidth(350)

        main_layout.addWidget(right_container)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.colorbar_manager is not None:
                self.colorbar_manager.cleanup()
                self.colorbar_manager = None
        except Exception:
            pass

        super().cleanup()

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

    def _save_stress_tensor(self):
        """Save stress tensor data to files."""
        if not hasattr(self.data_manager, 'stress_results') or not self.data_manager.stress_results:
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

                # Get the algorithm name from its ID
                algorithm_id = self.MESH_ALGORITHMS[self.parameter_spins['algorithm'].currentText()]

                # Get Young's modulus safely - default to 1.0 if analyzer is not initialized
                youngs_modulus = getattr(self.analyzer, 'E', 1.0)

                # Package results with all necessary parameters
                stress_results = {
                    'stress_tensor': self.data_manager.stress_results['stress_tensor'],
                    'condition_numbers': self.data_manager.stress_results.get('condition_numbers'),
                    'residuals': self.data_manager.stress_results.get('residuals'),
                    'parameters': {
                        'pixel_size': self._pixelsize,
                        'downscale_factor': self._downscale_factor,
                        'youngs_modulus': youngs_modulus,
                        'sigma': self.parameter_spins['sigma'].value(),
                        'density_factor': self.parameter_spins['density_factor'].value(),
                        'algorithm': algorithm_id,
                        'algorithm_name': self.parameter_spins['algorithm'].currentText(),
                        'use_optimization': self.parameter_spins['use_optimization'].isChecked(),
                        'max_stress': self.parameter_spins['max_stress'].value(),
                        'dilation': self.parameter_spins['dilation'].value(),
                        'smoothing_sigma': self.parameter_spins['smoothing_sigma'].value()
                    }
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

                # Extract algorithm name from ID if saved in old format
                algorithm_name = parameters.get('algorithm_name')
                if algorithm_name is None:
                    # Find algorithm name by ID
                    algorithm_id = parameters['algorithm']
                    algorithm_name = next(
                        (name for name, id in self.MESH_ALGORITHMS.items() if id == algorithm_id),
                        'Frontal-Del.'  # Default if not found
                    )

                # Update UI parameters with loaded values
                parameter_mapping = {
                    'sigma': 'sigma',
                    'density_factor': 'density_factor',
                    'max_stress': 'max_stress',
                    'dilation': 'dilation',
                    'smoothing_sigma': 'smoothing_sigma'
                }

                for param_name, spin_name in parameter_mapping.items():
                    if param_name in parameters:
                        self.parameter_spins[spin_name].setValue(parameters[param_name])

                # Update algorithm combobox
                self.parameter_spins['algorithm'].setCurrentText(algorithm_name)

                # Update optimization checkbox
                if 'use_optimization' in parameters:
                    self.parameter_spins['use_optimization'].setChecked(parameters['use_optimization'])

                # Store pixel size and downscale factor
                self._pixelsize = parameters['pixel_size']
                self._downscale_factor = parameters['downscale_factor']

                # Update analyzer with loaded parameters
                if self.current_mask is not None:
                    self._update_parameters()

                # Update data manager and visualization
                self.data_manager.stress_results = stress_data
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

                self._update_status(f"Stress tensor data successfully loaded from:\n{file_path}", 100)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load stress tensor data: {str(e)}"
            )

    def _update_ui_state(self):
        """Update UI element states based on current data availability."""
        has_mask = self.current_mask is not None
        has_force_data = self.data_manager.force_results is not None

        self.preview_mesh_btn.setEnabled(has_mask)
        self.preview_frame_btn.setEnabled(has_mask and has_force_data)
        self.analyze_btn.setEnabled(has_mask and has_force_data)
