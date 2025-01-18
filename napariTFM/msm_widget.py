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
        "Frontal-Delaunay": 6,
        "Delaunay": 5,
        "MeshAdapt": 1,
        "BAMG (experimental)": 7,
        "FD for Quads (experimental)": 8,
        "Parallelogram Packing (experimental)": 9
    }

    def __init__(self, viewer, data_manager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize parameters
        self.parameter_spins = {}
        self.current_mask = None
        self.mesh = None
        self.colorbar_manager = ColorbarManager()

        # Initialize analyzer with default parameters (will be updated when needed)
        self.analyzer = None
        self.mesh_generator = None

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Mesh generation parameters
        mesh_params = [
            ("density_factor", "Density Factor:", 0.001, 0.1, 0.001, 0.025),  # Direct density factor control
            ("algorithm", "Mesh Algorithm:", None, None, None, "Frontal-Delaunay"),  # Dropdown for algorithm selection
            ("use_optimization", "Enable Mesh Optimization", None, None, None, True),  # Checkbox
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 0),
            ("smoothing_sigma", "Boundary Smoothing:", 0.0, 40.0, 0.1, 1.0),
        ]

        # Material parameters
        material_params = [
            ("sigma", "Poisson's Ratio:", 0.0, 1.0, 0.01, 0.5),
            ("pixelsize", "Pixel Size (µm):", 0.01, 10.0, 0.01, 1.0),
        ]

        # Visualization parameters
        vis_params = [
            ("max_stress", "Max Stress (mN/m):", 0.1, 1000.0, 0.1, 10.0),
        ]

        # Create parameter sections with headers
        sections = [
            ("Mesh Parameters", mesh_params),
            ("Material Parameters", material_params),
            ("Visualization", vis_params)
        ]

        for section_title, params in sections:
            # Add section header
            header = QLabel(f"<b>{section_title}</b>")
            layout.addWidget(header)

            # Add parameters for this section
            for param_name, label, min_val, max_val, step, default in params:
                param_layout = QHBoxLayout()
                param_layout.addWidget(QLabel(label))

                if param_name == "use_optimization":
                    # Create checkbox for optimization
                    from qtpy.QtWidgets import QCheckBox
                    spin = QCheckBox()
                    spin.setChecked(default)
                elif param_name == "algorithm":
                    # Create dropdown for algorithm selection
                    spin = QComboBox()
                    for algo_name in self.MESH_ALGORITHMS.keys():
                        spin.addItem(algo_name)
                    spin.setCurrentText(default)
                elif isinstance(step, int):
                    spin = QSpinBox()
                    spin.setRange(min_val, max_val)
                    spin.setSingleStep(step)
                    spin.setValue(default)
                else:
                    spin = QDoubleSpinBox()
                    spin.setDecimals(3 if param_name == "density_factor" else 2)
                    spin.setRange(min_val, max_val)
                    spin.setSingleStep(step)
                    spin.setValue(default)

                self.parameter_spins[param_name] = spin
                param_layout.addWidget(spin)
                layout.addLayout(param_layout)

            # Add spacing between sections
            layout.addSpacing(10)

        group.setLayout(layout)
        return group

    def _update_parameters(self):
        """Update analysis parameters."""
        try:
            # Get current parameters from UI
            pixelsize = self.parameter_spins['pixelsize'].value()
            sigma = self.parameter_spins['sigma'].value()
            density_factor = self.parameter_spins['density_factor'].value()
            algorithm_name = self.parameter_spins['algorithm'].currentText()
            algorithm = self.MESH_ALGORITHMS[algorithm_name]
            use_optimization = self.parameter_spins['use_optimization'].isChecked()

            # Create mesh parameters if we have a mask
            if self.current_mask is not None:
                mesh_params = MeshParameters(
                    mask=self.current_mask[0],  # Use first frame's mask
                    density_factor=density_factor,
                    algorithm=algorithm,
                    use_optimization=use_optimization
                )

                # Update mesh generator
                self.mesh_generator = MeshGenerator(mesh_params)

                # Create new analyzer with current parameters
                self.analyzer = MonolayerStressMicroscopy(
                    mask=self.current_mask[0],
                    pixelsize=pixelsize,
                    sigma=sigma,
                    youngs_modulus=1.0  # Fixed value as per new API
                )

            # Update colorbar limits based on max_stress parameter
            max_stress = self.parameter_spins['max_stress'].value()
            self.colorbar_manager.update_limits(-max_stress, max_stress)

        except Exception as e:
            self._handle_error(f"Failed to update parameters: {str(e)}")

    def _connect_signals(self):
        """Connect widget signals."""
        # Buttons
        self.create_mask_btn.clicked.connect(self._create_mask_from_cells)
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

    def _create_mask_from_cells(self):
        """Create masks from the cell stack using dilation and smoothing."""
        try:
            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                raise ValueError("No active image layer selected")

            # Update status
            self.status_label.setText("Creating masks...")
            self.progress_bar.setValue(20)

            # Get parameters
            dilation = self.parameter_spins['dilation'].value()
            smoothing_sigma = self.parameter_spins['smoothing_sigma'].value()

            # Process the image stack
            from scipy import ndimage

            cell_stack = active_layer.data
            if cell_stack.ndim == 2:
                cell_stack = cell_stack[np.newaxis, ...]

            mask_stack = np.zeros_like(cell_stack, dtype=bool)

            for frame in range(cell_stack.shape[0]):
                # Basic thresholding
                mask = cell_stack[frame] > 0

                # Fill holes
                filled_mask = ndimage.binary_fill_holes(mask)

                # Get largest connected component
                labels, num_features = ndimage.label(filled_mask)
                if num_features > 0:
                    sizes = ndimage.sum(filled_mask, labels, range(1, num_features + 1))
                    largest_feature = np.argmax(sizes) + 1
                    filled_mask = labels == largest_feature

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
                    final_mask = ndimage.binary_dilation(
                        smoothed_mask,
                        structure=struct,
                        iterations=dilation
                    )
                else:
                    final_mask = smoothed_mask

                mask_stack[frame] = final_mask

            # Store the mask
            self.current_mask = mask_stack

            # Add visualization layer
            if 'Cell Mask' in self.viewer.layers:
                self.viewer.layers.remove('Cell Mask')

            self.viewer.add_labels(
                self.current_mask.astype(np.uint8),
                name='Cell Mask',
                visible=True,
                opacity=0.5,
            )

            # Update status
            num_frames = self.current_mask.shape[0]
            self.status_label.setText(f"Successfully created masks for {num_frames} frames")
            self.progress_bar.setValue(100)

            # Update UI state
            self._update_ui_state()
            self._update_parameters()  # Initialize analyzer with new mask

        except Exception as e:
            self._handle_error(f"Failed to create mask from cells:\n{str(e)}")
            self.progress_bar.setValue(0)

    def preview_mesh(self):
        """Generate and display preview of the triangular mesh for the current frame."""
        try:
            if self.current_mask is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            current_frame = self.viewer.dims.current_step[0]
            current_mask = self.current_mask[current_frame]

            # Get force data shape and downscale factor
            target_shape = None
            downscale_factor = 1

            # Get downscale factor from displacement results if available
            if self.data_manager.displacement_results is not None:
                downscale_factor = self.data_manager.displacement_results.get('parameters', {}).get('downscale_factor', 1)

            # Get target shape from force results if available
            if self.data_manager.force_results is not None:
                tx = self.data_manager.force_results['tx'][current_frame]
                target_shape = tx.shape

            # Resize mask if needed
            if target_shape is not None and current_mask.shape != target_shape:
                from skimage.transform import resize
                current_mask = resize(current_mask.astype(float), target_shape, order=0) > 0.5

            # Update parameters to ensure mesh generator is initialized
            self._update_parameters()

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

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame with compact layout."""
        frame = QFrame()
        layout = QVBoxLayout()
        layout.setSpacing(6)  # Reduce spacing between buttons

        # Data preparation
        self.create_mask_btn = QPushButton("Create Mask from Cells")

        # Analysis operations
        self.preview_mesh_btn = QPushButton("Preview Mesh")
        self.preview_frame_btn = QPushButton("Preview Current Frame")
        self.analyze_btn = QPushButton("Analyze All Frames")

        # Save/Load operations
        save_load_layout = QHBoxLayout()
        self.save_stress_btn = QPushButton("Save Stress Tensor")
        self.load_stress_btn = QPushButton("Load Stress Tensor")
        save_load_layout.addWidget(self.save_stress_btn)
        save_load_layout.addWidget(self.load_stress_btn)

        # Add all buttons to layout
        layout.addWidget(self.create_mask_btn)
        layout.addWidget(self.preview_mesh_btn)
        layout.addWidget(self.preview_frame_btn)
        layout.addWidget(self.analyze_btn)
        layout.addLayout(save_load_layout)

        frame.setLayout(layout)
        return frame

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

    def preview_current_frame(self):
        """Preview stress calculation for the current frame."""
        try:
            # Check prerequisites
            if self.current_mask is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            if self.data_manager.force_results is None:
                raise ValueError("No force data available. Please calculate forces first.")

            # Get current frame index
            current_frame = self.viewer.dims.current_step[0]

            # Get force data for current frame
            tx = self.data_manager.force_results['tx'][current_frame]
            ty = self.data_manager.force_results['ty'][current_frame]

            # Get current frame's mask
            current_mask = self.current_mask[current_frame]

            # Update status
            self._update_status("Calculating stress field...", 20)

            # Update parameters and initialize analyzer
            self._update_parameters()

            # Calculate stress tensor
            stress_tensor = self.analyzer.calculate_stress_field(tx, ty, current_mask)

            # Convert stress units from N/pixel² to mN/m
            pixelsize = self.parameter_spins['pixelsize'].value()
            stress_tensor = stress_tensor / (pixelsize * 1e-6)

            self._update_status("Updating visualization...", 80)

            # Get max stress parameter
            max_stress = self.parameter_spins['max_stress'].value()

            # Update visualization
            self.visualization_manager.visualize_stress_preview(
                stress_tensor,
                max_stress
            )

            # Update status
            self._update_status(
                f"Stress preview generated for frame {current_frame}",
                100
            )

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

            # Get force data
            tx = self.data_manager.force_results['tx']
            ty = self.data_manager.force_results['ty']
            num_frames = len(tx)

            # Update parameters and initialize analyzer
            self._update_parameters()

            # Initialize results storage
            stress_results = []
            pixelsize = self.parameter_spins['pixelsize'].value()

            # Process each frame
            for frame in range(num_frames):
                self._update_status(
                    f"Processing frame {frame + 1}/{num_frames}...",
                    int((frame / num_frames) * 100)
                )

                # Get current frame's data
                current_tx = tx[frame]
                current_ty = ty[frame]
                current_mask = self.current_mask[frame]

                # Calculate stress tensor for current frame
                stress_tensor = self.analyzer.calculate_stress_field(
                    current_tx,
                    current_ty,
                    current_mask
                )

                # Convert units from N/pixel² to mN/m
                stress_tensor = stress_tensor / (pixelsize * 1e-6)
                stress_results.append(stress_tensor)

            # Convert results to numpy array
            stress_tensor_stack = np.stack(stress_results, axis=0)

            # Store results in data manager with all parameters
            self.data_manager.stress_results = {
                'stress_tensor': stress_tensor_stack,
                'parameters': {
                    'pixelsize': pixelsize,
                    'youngs_modulus': self.analyzer.E,
                    'poisson_ratio': self.analyzer.sigma,
                    'target_nodes': self.parameter_spins['target_nodes'].value(),
                    'boundary_refinement': self.parameter_spins['boundary_refinement'].value(),
                    'gradient_refinement': self.parameter_spins['gradient_refinement'].value(),
                    'max_stress': self.parameter_spins['max_stress'].value()  # Added max_stress to parameters
                }
            }

            # Emit results
            self.stress_calculated.emit(self.data_manager.stress_results)

            # Update visualization using the new method
            self._update_status("Updating visualization...", 90)
            max_stress = self.parameter_spins['max_stress'].value()
            self.visualization_manager.visualize_stress_results(
                self.data_manager.stress_results,
                max_stress=max_stress
            )

            self._update_status(
                f"Stress analysis completed for {num_frames} frames",
                100
            )

        except Exception as e:
            self._handle_error(f"Failed to analyze frames: {str(e)}")
            self.progress_bar.setValue(0)

    def _save_stress_tensor(self):
        """Save stress tensor data to files."""
        if not hasattr(self.data_manager, 'stress_results') or not self.data_manager.stress_results:
            QMessageBox.warning(self, "Warning", "No stress tensor data to save.")
            return

        try:
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Stress Tensor Data",
                os.path.expanduser("~")
            )

            if save_dir:
                # Package results with all necessary parameters - matching batch analyzer structure
                stress_results = {
                    'stress_tensor': self.data_manager.stress_results['stress_tensor'],
                    'parameters': {
                        'pixelsize': self.parameter_spins['pixelsize'].value(),
                        'youngs_modulus': self.analyzer.E,
                        'poisson_ratio': self.parameter_spins['sigma'].value(),  # Use poisson_ratio as key
                        'target_nodes': self.parameter_spins['target_nodes'].value(),
                        'boundary_refinement': self.parameter_spins['boundary_refinement'].value(),
                        'gradient_refinement': self.parameter_spins['gradient_refinement'].value(),
                        'max_stress': self.parameter_spins['max_stress'].value(),
                        'dilation': self.parameter_spins['dilation'].value(),
                        'smoothing_sigma': self.parameter_spins['smoothing_sigma'].value()
                    }
                }

                # Save file
                np.save(os.path.join(save_dir, 'stress_tensor.npy'), stress_results)

                self._update_status(f"Stress tensor data successfully saved to:\n{save_dir}", 100)

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

                # Extract poisson ratio - expecting 'poisson_ratio' from batch analyzer
                if 'poisson_ratio' not in parameters:
                    raise ValueError("Required parameter 'poisson_ratio' not found in file")

                poisson_ratio = parameters['poisson_ratio']

                # Update UI parameters with loaded values
                parameter_mapping = {
                    'pixelsize': 'pixelsize',
                    'target_nodes': 'target_nodes',
                    'boundary_refinement': 'boundary_refinement',
                    'gradient_refinement': 'gradient_refinement',
                    'max_stress': 'max_stress',
                    'dilation': 'dilation',
                    'smoothing_sigma': 'smoothing_sigma'
                }

                for param_name, spin_name in parameter_mapping.items():
                    if param_name in parameters:
                        self.parameter_spins[spin_name].setValue(parameters[param_name])

                # Update sigma spinbox with the loaded Poisson ratio
                self.parameter_spins['sigma'].setValue(poisson_ratio)

                # Update analyzer with loaded parameters
                self.analyzer = MonolayerStressMicroscopy(
                    pixelsize=parameters['pixelsize'],
                    sigma=poisson_ratio,
                    youngs_modulus=parameters['youngs_modulus'],
                    target_nodes=parameters['target_nodes'],
                    boundary_refinement=parameters['boundary_refinement'],
                    gradient_refinement=parameters['gradient_refinement']
                )

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
