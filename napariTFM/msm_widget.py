import os

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame,
    QProgressBar, QMessageBox, QFileDialog
)
from qtpy.QtCore import Signal, Qt
import numpy as np

from .base_widget import BaseAnalysisWidget
from .colorbar import ColorbarManager
from .msm import MonolayerStressMicroscopy


class MSMWidget(BaseAnalysisWidget):
    """Widget for Monolayer Stress Microscopy analysis."""

    stress_calculated = Signal(dict)  # Emits stress analysis results

    def __init__(self, viewer, data_manager, visualization_manager):
        super().__init__(viewer, data_manager, visualization_manager)

        # Initialize MSM analyzer with default parameters
        self.pixelsize = 1.0  # Default pixelsize, will be updated from parameters
        self.analyzer = MonolayerStressMicroscopy(pixelsize=self.pixelsize)

        self.parameter_spins = {}
        self.current_mask = None
        self.mesh = None
        self.colorbar_manager = ColorbarManager()

        self._setup_ui()
        self._connect_signals()
        self._update_parameters()  # Initialize analyzer with UI parameters
        self._update_ui_state()

    def _update_parameters(self):
        """Update analysis parameters."""
        try:
            # Get parameters from UI
            pixelsize = self.parameter_spins['pixelsize'].value()
            sigma = self.parameter_spins['sigma'].value()
            target_nodes = self.parameter_spins['target_nodes'].value()
            boundary_refinement = self.parameter_spins['boundary_refinement'].value()
            gradient_refinement = self.parameter_spins['gradient_refinement'].value()

            # Create new analyzer instance with current parameters
            self.analyzer = MonolayerStressMicroscopy(
                pixelsize=pixelsize,
                sigma=sigma,
                target_nodes=target_nodes,
                boundary_refinement=boundary_refinement,
                gradient_refinement=gradient_refinement
            )

            # Update colorbar limits based on max_stress parameter
            max_stress = self.parameter_spins['max_stress'].value()
            self.colorbar_manager.update_limits(-max_stress, max_stress)

        except Exception as e:
            self._handle_error(f"Failed to update parameters: {str(e)}")

    def _create_mask_from_cells(self):
        """Create masks from the cell stack using MSM analyzer."""
        try:
            # Ensure analyzer is initialized
            if self.analyzer is None:
                self._update_parameters()
                if self.analyzer is None:
                    raise ValueError("Failed to initialize MSM analyzer")

            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                raise ValueError("No active image layer selected")

            # Update status
            self.status_label.setText("Creating masks...")
            self.progress_bar.setValue(20)

            # Create masks using MSM analyzer
            self.current_mask = self.analyzer.create_mask_from_cells(
                active_layer.data,
                dilation_pixels=self.parameter_spins['dilation'].value(),
                smoothing_sigma=self.parameter_spins['smoothing_sigma'].value()
            )

            # Update visualization
            if 'Cell Mask' in self.viewer.layers:
                self.viewer.layers.remove('Cell Mask')

            # Add as labels layer
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

        except Exception as e:
            self._handle_error(f"Failed to create mask from cells:\n{str(e)}")
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

    def _connect_signals(self):
        """Connect widget signals."""
        # Buttons
        self.create_mask_btn.clicked.connect(self._create_mask_from_cells)
        self.preview_mesh_btn.clicked.connect(self.preview_mesh)
        self.preview_frame_btn.clicked.connect(self.preview_current_frame)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)
        self.save_stress_btn.clicked.connect(self._save_stress_tensor)
        self.load_stress_btn.clicked.connect(self._load_stress_tensor)

        # Parameters
        for spin in self.parameter_spins.values():
            spin.valueChanged.connect(self._update_parameters)

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

            # Resize mask if needed
            from skimage.transform import resize
            if current_mask.shape != tx.shape:
                current_mask = resize(current_mask.astype(float), tx.shape, order=0) > 0.5

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

                # Resize mask if needed
                if current_mask.shape != current_tx.shape:
                    from skimage.transform import resize
                    current_mask = resize(
                        current_mask.astype(float),
                        current_tx.shape,
                        order=0
                    ) > 0.5

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

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Mesh parameters
        mesh_params = [
            ("target_nodes", "Target Nodes:", 100, 10000, 100, 1000),
            ("boundary_refinement", "Boundary Refinement:", 1.0, 5.0, 0.1, 2.0),
            ("gradient_refinement", "Gradient Refinement:", 1.0, 5.0, 0.1, 1.5),
        ]

        # Material parameters
        material_params = [
            ("sigma", "Poisson's Ratio:", 0.0, 1.0, 0.01, 0.5),
            ("pixelsize", "Pixel Size (µm):", 0.01, 10.0, 0.01, 1.0),
        ]

        # Mask parameters
        mask_params = [
            ("dilation", "Mask Dilation (px):", 0, 50, 1, 0),
            ("smoothing_sigma", "Boundary Smoothing:", 0.0, 40.0, 0.1, 1.0),
        ]

        # Visualization parameters
        vis_params = [
            ("max_stress", "Max Stress (mN/m):", 0.1, 1000.0, 0.1, 10.0),
        ]

        # Add all parameters to layout
        for param_name, label, min_val, max_val, step, default in (
                mesh_params + material_params + mask_params + vis_params
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            if isinstance(step, int):
                spin = QSpinBox()
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(2)

            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)

            self.parameter_spins[param_name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def preview_mesh(self):
        """Generate and display preview of the triangular mesh for the current frame with proper scaling."""
        try:
            if self.current_mask is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            self._update_parameters()
            current_frame = self.viewer.dims.current_step[0]
            current_mask = self.current_mask[current_frame]

            # Get force data shape and downscale factor
            target_shape = None
            downscale_factor = 1  # Default scale factor

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

            # Get traction forces for the current frame if available
            tx = ty = None
            if self.data_manager.force_results is not None:
                tx = self.data_manager.force_results['tx'][current_frame]
                ty = self.data_manager.force_results['ty'][current_frame]

            self._update_status("Generating mesh...", 20)

            # Generate mesh with target nodes
            nodes_xy, elements = self.analyzer.mesh_generator.generate_mesh(
                current_mask,
                tx,
                ty
            )

            self._update_status("Creating visualization...", 50)

            # Remove existing mesh layers
            for layer_name in ['Mesh Edges', 'Mesh Nodes']:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)

            # Scale up the node coordinates
            nodes_xy_scaled = nodes_xy * downscale_factor

            # Create edge data for visualization
            num_elements = len(elements)
            edge_data = np.zeros((num_elements * 3, 2, 2))

            # Swap x and y coordinates and apply scaling
            nodes_rotated = np.column_stack((nodes_xy_scaled[:, 1], nodes_xy_scaled[:, 0]))

            for i, element in enumerate(elements):
                # Get node coordinates for triangle vertices
                v1 = nodes_rotated[element[0]]
                v2 = nodes_rotated[element[1]]
                v3 = nodes_rotated[element[2]]

                # Add three edges (v1-v2, v2-v3, v3-v1)
                edge_data[i * 3] = np.array([v1, v2])
                edge_data[i * 3 + 1] = np.array([v2, v3])
                edge_data[i * 3 + 2] = np.array([v3, v1])

            self._update_status("Adding visualization layers...", 80)

            # Add edges as shapes layer
            self.viewer.add_shapes(
                edge_data,
                shape_type='line',
                edge_color='yellow',
                edge_width=1,
                opacity=0.6,
                name='Mesh Edges'
            )

            # Add nodes as points layer
            self.viewer.add_points(
                nodes_rotated,
                size=4,
                face_color='red',
                opacity=0.7,
                name='Mesh Nodes'
            )

            # Store original (unrotated) mesh for later use
            self.mesh = {
                'nodes': nodes_xy,  # Store unscaled coordinates
                'elements': elements,
                'frame': current_frame,
                'scale_factor': downscale_factor
            }

            # Update status with actual node count
            num_nodes = len(nodes_xy)
            num_elements = len(elements)
            self._update_status(
                f"Mesh preview generated: \n{num_nodes} nodes ({num_nodes / self.parameter_spins['target_nodes'].value():.1%} of target), {num_elements} elements",
                100
            )

            # Enable preview frame button if we have force data
            self.preview_frame_btn.setEnabled(self.data_manager.force_results is not None)

        except Exception as e:
            self._handle_error(f"Failed to preview mesh: {str(e)}")
            self.progress_bar.setValue(0)

    def _save_stress_tensor(self):
        """Save stress tensor data to file."""
        if not hasattr(self.data_manager, 'stress_results') or not self.data_manager.stress_results:
            QMessageBox.warning(self, "Warning", "No stress tensor data to save.")
            return

        try:
            # Get directory to save file
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Stress Tensor Data",
                os.path.expanduser("~")
            )

            if save_dir:
                stress_tensor = self.data_manager.stress_results['stress_tensor']

                # Save file
                np.save(os.path.join(save_dir, 'stress_tensor.npy'), stress_tensor)

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
            # Get directory containing the file
            load_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory Containing Stress Tensor Data",
                os.path.expanduser("~")
            )

            if load_dir:
                # Check if file exists
                stress_tensor_path = os.path.join(load_dir, 'stress_tensor.npy')

                if not os.path.exists(stress_tensor_path):
                    raise FileNotFoundError("Could not find stress_tensor.npy in selected directory")

                # Load the stress tensor
                stress_tensor = np.load(stress_tensor_path)

                # Create results dictionary with current parameters
                results = {
                    'stress_tensor': stress_tensor,
                    'parameters': {
                        'pixelsize': self.parameter_spins['pixelsize'].value(),
                        'youngs_modulus': self.analyzer.E,
                        'poisson_ratio': self.analyzer.sigma,
                        'target_nodes': self.parameter_spins['target_nodes'].value(),
                        'boundary_refinement': self.parameter_spins['boundary_refinement'].value(),
                        'gradient_refinement': self.parameter_spins['gradient_refinement'].value(),
                        'max_stress': self.parameter_spins['max_stress'].value()
                    }
                }

                # Update data manager and visualization
                self.data_manager.stress_results = results
                self.visualization_manager.visualize_stress_results(
                    results,
                    max_stress=self.parameter_spins['max_stress'].value()
                )

                # Enable save button
                self.save_stress_btn.setEnabled(True)

                # Emit the stress_calculated signal with the results
                self.stress_calculated.emit(results)

                self._update_status(f"Stress tensor data successfully loaded from:\n{load_dir}", 100)

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