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

        # Initialize MSM analyzer
        self.analyzer = None  # Will be initialized with parameters
        self.parameter_spins = {}
        self.current_mask = None
        self.mesh = None
        self.colorbar_manager = ColorbarManager()

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

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
            colormap_name='cividis',
            label="Stress (mN/m)",
            clim=(0, 1),  # Will be updated based on max_stress parameter
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
        right_layout.addWidget(self._create_data_loading_group())
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

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group with mask loading functionality."""
        group = QGroupBox("Data")
        layout = QVBoxLayout()

        # Add mask loading button and status
        self.load_mask_btn = QPushButton("Load Mask")
        self.mask_status = QLabel("Not loaded")

        row = QHBoxLayout()
        row.addWidget(self.load_mask_btn)
        row.addWidget(self.mask_status)
        layout.addLayout(row)

        group.setLayout(layout)
        return group

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
        self.load_mask_btn.clicked.connect(self._load_mask)
        self.create_mask_btn.clicked.connect(self._create_mask_from_cells)
        self.preview_mesh_btn.clicked.connect(self.preview_mesh)
        self.preview_frame_btn.clicked.connect(self.preview_current_frame)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)
        self.save_stress_btn.clicked.connect(self._save_stress_tensor)
        self.load_stress_btn.clicked.connect(self._load_stress_tensor)

        # Parameters
        for spin in self.parameter_spins.values():
            spin.valueChanged.connect(self._update_parameters)

    def _create_mask_from_cells(self):
        """Create masks from the cell stack by thresholding and morphological operations."""
        try:
            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                raise ValueError("No active image layer selected")

            # Get image data
            cell_stack = active_layer.data

            # Ensure 3D stack
            if cell_stack.ndim == 2:
                cell_stack = cell_stack[np.newaxis, ...]

            # Get number of frames
            num_frames = cell_stack.shape[0]
            mask_stack = np.zeros_like(cell_stack, dtype=bool)

            # Update status
            self.status_label.setText("Creating masks...")

            # Process each frame
            for frame in range(num_frames):
                # Get cell image for current frame
                cell_image = cell_stack[frame]

                # Create initial binary mask (threshold > 0)
                mask = cell_image > 0

                # Fill holes in the mask
                from scipy import ndimage
                filled_mask = ndimage.binary_fill_holes(mask)

                # Label connected components
                labels, num_features = ndimage.label(filled_mask)

                if num_features > 0:
                    # Find sizes of all features
                    sizes = ndimage.sum(filled_mask, labels, range(1, num_features + 1))

                    # Keep only the largest component
                    largest_feature = np.argmax(sizes) + 1
                    filled_mask = labels == largest_feature

                # Smooth edges
                # Convert to float for Gaussian filter
                float_mask = filled_mask.astype(float)
                smoothed = ndimage.gaussian_filter(float_mask, sigma=1.0)
                # Re-threshold to get binary mask (threshold at 0.5)
                smoothed_mask = smoothed > 0.5

                # Apply dilation if specified
                dilation_pixels = self.parameter_spins['dilation'].value()
                if dilation_pixels > 0:
                    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
                    dilated_mask = ndimage.binary_dilation(
                        smoothed_mask,
                        structure=struct,
                        iterations=dilation_pixels
                    )
                else:
                    dilated_mask = smoothed_mask

                # Add to mask stack
                mask_stack[frame] = dilated_mask

                # Update progress
                self.progress_bar.setValue(int((frame + 1) / num_frames * 100))

            # Store the mask stack
            self.current_mask = mask_stack

            # Update visualization
            if 'Cell Mask' in self.viewer.layers:
                self.viewer.layers.remove('Cell Mask')

            # Add as labels layer
            self.viewer.add_labels(
                mask_stack.astype(np.uint8),
                name='Cell Mask',
                visible=True,
                opacity=0.5,
            )

            # Update mask status
            self.mask_status.setText(f"Created from cells ({num_frames} frames)")
            self.status_label.setText(f"Successfully created masks for {num_frames} frames\n")
            self.progress_bar.setValue(100)

            # Update UI state
            self._update_ui_state()

        except Exception as e:
            self._handle_error(f"Failed to create mask from cells:\n{str(e)}")
            self.progress_bar.setValue(0)
    def preview_current_frame(self):
        """Preview stress calculation for the current frame."""
        # TODO: Implement preview for current frame
        # - Get current frame index
        # - Calculate stress for single frame
        # - Update visualization
        pass

    def analyze_all_frames(self):
        """Run stress analysis for all frames."""
        # TODO: Implement full analysis for all frames
        # - Process each frame
        # - Update progress bar
        # - Store results
        # - Update visualization
        pass

    def _load_mask(self):
        """Load and validate mask from file or active layer."""
        try:
            # Get active layer
            active_layer = self._get_active_image_layer()
            if active_layer is None:
                raise ValueError("No active image layer selected")

            # Get mask data
            mask_data = active_layer.data

            # Input validation
            if not np.issubdtype(mask_data.dtype, np.bool_) and not np.issubdtype(mask_data.dtype, np.integer):
                raise ValueError("Mask must be boolean or integer type")

            # Convert to boolean if integer
            if np.issubdtype(mask_data.dtype, np.integer):
                mask_data = mask_data > 0

            # Ensure 3D
            if mask_data.ndim == 2:
                mask_data = mask_data[np.newaxis, ...]

            # Basic validation
            if not np.any(mask_data):
                raise ValueError("Mask is empty (all False)")

            # Store mask
            self.current_mask = mask_data

            # Update mask status
            num_frames = mask_data.shape[0]
            self.mask_status.setText(f"Loaded ({num_frames} frames)")

            # Enable relevant buttons
            self.preview_mesh_btn.setEnabled(True)

            # Update status
            self._update_status(f"Successfully loaded mask with {num_frames} frames", 100)

        except Exception as e:
            self._handle_error(f"Failed to load mask: {str(e)}")
            self.mask_status.setText("Not loaded")
            self.preview_mesh_btn.setEnabled(False)
            self.current_mask = None

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

    def _update_parameters(self):
        """Update analysis parameters."""
        try:
            # Update MSM analyzer with current parameters
            self.analyzer = MonolayerStressMicroscopy(
                pixelsize=self.parameter_spins['pixelsize'].value(),
                sigma=self.parameter_spins['sigma'].value(),
                target_nodes=self.parameter_spins['target_nodes'].value(),
                boundary_refinement=self.parameter_spins['boundary_refinement'].value(),
                gradient_refinement=self.parameter_spins['gradient_refinement'].value()
            )

            # Update colorbar limits based on max_stress parameter
            max_stress = self.parameter_spins['max_stress'].value()
            self.colorbar_manager.update_limits(0, max_stress)

        except Exception as e:
            self._handle_error(str(e))

    def preview_mesh(self):
        """Generate and display preview of the triangular mesh for the current frame."""
        try:
            # Check for mask
            if self.current_mask is None:
                raise ValueError("No mask loaded. Please load a mask first.")

            # Update parameters
            self._update_parameters()

            # Get current frame index
            current_frame = self.viewer.dims.current_step[0]

            # Get current frame's mask
            current_mask = self.current_mask[current_frame]

            # Get traction forces for the current frame if available
            tx = ty = None
            if self.data_manager.force_results is not None:
                tx = self.data_manager.force_results['tx'][current_frame]
                ty = self.data_manager.force_results['ty'][current_frame]

            # Update status and progress
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

            # Create edge data for visualization
            num_elements = len(elements)
            edge_data = np.zeros((num_elements * 3, 2, 2))

            # Swap x and y coordinates to fix rotation
            nodes_rotated = np.column_stack((nodes_xy[:, 1], nodes_xy[:, 0]))

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
                edge_width=0.2,
                opacity=0.6,
                name='Mesh Edges'
            )

            # Add nodes as points layer
            self.viewer.add_points(
                nodes_rotated,
                size=2,
                face_color='red',
                opacity=0.7,
                name='Mesh Nodes'
            )

            # Store original (unrotated) mesh for later use
            self.mesh = {
                'nodes': nodes_xy,
                'elements': elements,
                'frame': current_frame
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
    def run_analysis(self):
        """Run the MSM analysis."""
        # TODO: Implement full MSM analysis workflow
        pass

    def _save_stress_tensor(self):
        """Save stress tensor results to file."""
        # TODO: Implement stress tensor saving
        pass

    def _load_stress_tensor(self):
        """Load stress tensor results from file."""
        # TODO: Implement stress tensor loading
        pass

    def _update_ui_state(self):
        """Update UI element states based on current data availability."""
        # TODO: Implement UI state updates
        pass
