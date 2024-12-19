from typing import Optional, Dict
import numpy as np
from matplotlib import pyplot as plt
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QWidget,
    QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
    QCheckBox, QFrame, QSizePolicy, QScrollArea,
    QProgressBar, QMessageBox
)
import napari

from .base_widget import BaseAnalysisWidget, logger
from .data_manager import DataManager
from .displacement_analysis import DisplacementAnalyzer, TVL1Parameters
from .visualization_manager import VisualizationManager
from .colorbar import ColorbarManager


class DisplacementAnalysisWidget(BaseAnalysisWidget):
    """Widget for analyzing bead displacements using TV-L1 optical flow."""

    displacement_calculated = Signal(dict)  # Emits displacement results

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager",
                 visualization_manager: "VisualizationManager"):
        super().__init__(viewer, data_manager, visualization_manager)

        self.analyzer = DisplacementAnalyzer()
        self.colorbar_manager = ColorbarManager()
        self.colorbar_widget = None
        self.current_flow = None
        self.parameter_spins = {}
        self.visualization_params = {}

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def _setup_ui(self):
        """Set up the user interface."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.setSpacing(8)
        horizontal_layout.setContentsMargins(6, 6, 6, 6)

        # Create colorbar group
        colorbar_group = QGroupBox("Displacement Magnitude")
        colorbar_layout = QVBoxLayout()

        self.colorbar_widget = self.colorbar_manager.create_colorbar(
            width=80,
            height=200,
            colormap_name='viridis',
            label="Displacement (pixels)",
            clim=(10, 0),
            orientation='right',
            label_color='white',
            border_color='gray',
            border_width=1.0,
            padding=(0.1, 0.1),
            axis_ratio=0.05,
            # label_offset=(-15, 0),
            label_offset=(0, 0),
            label_rotation=0,
            # tick_label_offset=(-40, -45),
            tick_label_offset = (0, 0)
        )

        colorbar_group.setFixedWidth(120)
        colorbar_layout.addWidget(self.colorbar_widget)
        colorbar_group.setLayout(colorbar_layout)
        horizontal_layout.addWidget(colorbar_group)

        # Main content layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(self._create_data_loading_group())
        main_layout.addWidget(self._create_parameters_group())
        main_layout.addWidget(self._create_visualization_parameters_group())
        main_layout.addWidget(self._create_action_buttons())
        main_layout.addWidget(self._create_status_frame())

        main_content = QWidget()
        main_content.setLayout(main_layout)
        horizontal_layout.addWidget(main_content)

        container.setLayout(horizontal_layout)
        scroll.setWidget(container)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setLayout(layout)

        self._register_controls()

    def _connect_signals(self):
        """Connect all widget signals."""
        self.load_beads_btn.clicked.connect(lambda: self._load_data('beads'))
        self.load_reference_btn.clicked.connect(lambda: self._load_data('reference'))
        self.load_cells_btn.clicked.connect(lambda: self._load_data('cells'))

        for spin in self.parameter_spins.values():
            spin.valueChanged.connect(self.update_parameters)

        for param in self.visualization_params.values():
            param.valueChanged.connect(self._update_visualization_params)

        self.preview_btn.clicked.connect(self.preview_displacement)
        self.analyze_btn.clicked.connect(self.analyze_all_frames)

        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

    def _update_colorbar(self, vmin: float, vmax: float):
        """Update the colorbar limits and appearance."""
        if self.colorbar_manager is not None:
            self.colorbar_manager.update_limits(vmax, vmin)

            # if hasattr(self, 'visualization_params'):
            #     if 'd_max' in self.visualization_params:
            #         max_val = self.visualization_params['d_max'].value()
                    # if max_val is not None:
                    #     # Adjust label position based on value range
                    #     if max_val > 100:
                    #         self.colorbar_manager.update_label_position(offset=(0.2, 0))
                    #     elif max_val > 50:
                    #         self.colorbar_manager.update_label_position(offset=(0.15, 0))
                    #     else:
                    #         self.colorbar_manager.update_label_position(offset=(0.1, 0))

    def _update_visualization_params(self):
        """Update visualization when parameters change."""
        if self.current_flow is not None:
            reference = (self.data_manager.preprocessed_reference or
                         self.data_manager.reference_image)
            bead_stack = (self.data_manager.preprocessed_bead_stack or
                          self.data_manager.bead_stack)
            current_frame = self.viewer.dims.current_step[0]
            moving = bead_stack[current_frame]

            d_max = self.visualization_params['d_max'].value()
            vector_stride = self.visualization_params['vector_stride'].value()
            arrow_scale = self.visualization_params['arrow_scale'].value()

            # Update vector layer
            flow_scaled = self.current_flow * arrow_scale
            vectors, colors = self.visualization_manager._create_vector_visualization(
                flow_scaled, self.current_flow, vector_stride, d_max)

            # Update layers
            with self.viewer.events.blocker_all():
                # Update magnitude layer
                magnitude = np.sqrt(np.sum(self.current_flow ** 2, axis=-1))
                if d_max is not None:
                    magnitude = np.clip(magnitude, 0, d_max)

                mag_layer = None
                for layer in self.viewer.layers:
                    if layer.name == 'Displacement Magnitude':
                        mag_layer = layer
                        break

                if mag_layer is not None:
                    mag_layer.data = magnitude
                else:
                    self.viewer.add_image(
                        magnitude,
                        name='Displacement Magnitude',
                        colormap='viridis',
                        blending='additive'
                    )

                # Update vector layer
                vector_layer = None
                for layer in self.viewer.layers:
                    if layer.name == 'Flow Vectors':
                        vector_layer = layer
                        break

                if vector_layer is not None:
                    vector_layer.data = vectors
                    vector_layer.edge_color = colors
                else:
                    self.viewer.add_shapes(
                        vectors,
                        shape_type='line',
                        name='Flow Vectors',
                        edge_color=colors,
                        edge_width=2,
                        blending='additive'
                    )

            # Update colorbar
            self._update_colorbar(0, d_max if d_max is not None else magnitude.max())

    def _on_frame_changed(self, event=None):
        """Handle frame change events."""
        if hasattr(self.data_manager, 'displacement_results'):
            results = self.data_manager.displacement_results
            if results and 'vector_cache' in results:
                current_frame = self.viewer.dims.current_step[0]
                if current_frame < len(results['vector_cache']['data']):
                    with self.viewer.events.blocker_all():
                        vector_layer = None
                        for layer in self.viewer.layers:
                            if layer.name == 'Flow Vectors':
                                vector_layer = layer
                                break

                        if vector_layer is not None:
                            vector_layer.data = results['vector_cache']['data'][current_frame]
                            vector_layer.edge_color = results['vector_cache']['colors'][current_frame]

    def cleanup(self):
        """Clean up resources."""
        try:
            self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)

            if self.colorbar_manager is not None:
                self.colorbar_manager.cleanup()
                self.colorbar_manager = None
                self.colorbar_widget = None

        except Exception:
            pass

        super().cleanup()

    def preview_displacement(self):
        """Preview displacement calculation on current frame."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Calculating displacement...", 0)

            # Get current frame data
            current_frame = self.viewer.dims.current_step[0]
            reference = (self.data_manager.preprocessed_reference or
                         self.data_manager.reference_image)
            bead_stack = (self.data_manager.preprocessed_bead_stack or
                          self.data_manager.bead_stack)
            moving = bead_stack[current_frame]

            # Calculate flow
            self.current_flow = self.analyzer.calculate_flow(reference, moving)

            # Get visualization parameters
            d_max = self.visualization_params['d_max'].value()
            vector_stride = self.visualization_params['vector_stride'].value()
            arrow_scale = self.visualization_params['arrow_scale'].value()

            # Get cell data if available
            cells = None
            if self.data_manager.preprocessed_cell_stack is not None:
                cells = self.data_manager.preprocessed_cell_stack[current_frame]
            elif self.data_manager.cell_stack is not None:
                cells = self.data_manager.cell_stack[current_frame]

            # Clear existing layers
            self.visualization_manager._clear_layers([
                'Displacement Overlay',
                'Displacement Magnitude',
                'Flow Vectors',
                'Cell Overlay'
            ])

            # Create vector data
            flow_scaled = self.current_flow * arrow_scale
            vectors, colors = self.visualization_manager._create_vector_visualization(
                flow_scaled,
                self.current_flow,
                vector_stride,
                d_max
            )

            # Create and add visualization layers
            with self.viewer.events.blocker_all():
                # Add overlay
                overlay = self.visualization_manager._create_overlay(reference, moving)
                self.viewer.add_image(
                    overlay,
                    name='Displacement Overlay',
                    rgb=True,
                    blending='additive'
                )

                # Add magnitude
                magnitude = np.sqrt(np.sum(self.current_flow ** 2, axis=-1))
                if d_max is not None:
                    magnitude = np.clip(magnitude, 0, d_max)

                self.viewer.add_image(
                    magnitude,
                    name='Displacement Magnitude',
                    colormap='viridis',
                    blending='additive'
                )

                # Add vectors
                if len(vectors) > 0:
                    self.viewer.add_shapes(
                        vectors,
                        shape_type='line',
                        name='Flow Vectors',
                        edge_color=colors,
                        edge_width=2,
                        blending='additive'
                    )

                # Add cell overlay if available
                if cells is not None:
                    self.viewer.add_image(
                        cells,
                        name='Cell Overlay',
                        colormap='gray',
                        opacity=0.5,
                        blending='additive'
                    )

            # Update colorbar
            vmax = d_max if d_max is not None else magnitude.max()
            self._update_colorbar(0, vmax)

            # Update status with displacement statistics
            stats = self.visualization_manager.get_displacement_statistics(self.current_flow)
            self._update_status(
                f"Max displacement: {stats['max']:.2f} pixels\n"
                f"Mean displacement: {stats['mean']:.2f} pixels",
                100
            )

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _validate_input_data(self) -> bool:
        """Validate required input data is available."""
        reference = (self.data_manager.preprocessed_reference or
                     self.data_manager.reference_image)
        bead_stack = (self.data_manager.preprocessed_bead_stack or
                      self.data_manager.bead_stack)

        if reference is None:
            QMessageBox.warning(self, "Error", "Reference image required")
            return False

        if bead_stack is None:
            QMessageBox.warning(self, "Error", "Bead stack required")
            return False

        return True

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Check for either preprocessed or raw data
        reference = (self.data_manager.preprocessed_reference or
                     self.data_manager.reference_image)
        bead_stack = (self.data_manager.preprocessed_bead_stack or
                      self.data_manager.bead_stack)

        has_reference = reference is not None
        has_beads = bead_stack is not None

        # Update status labels
        self.reference_status.setText("Loaded: " + str(reference.shape) if has_reference else "Not loaded")
        self.bead_status.setText("Loaded: " + str(bead_stack.shape) if has_beads else "Not loaded")

        # Both conditions must be met to enable analyze and preview buttons
        can_analyze = has_beads and has_reference

        # Update button states
        self.analyze_btn.setEnabled(can_analyze)
        self.preview_btn.setEnabled(can_analyze)

        # Update status message if data is missing
        if not can_analyze:
            missing = []
            if not has_beads:
                missing.append("bead stack")
            if not has_reference:
                missing.append("reference image")
            if missing:
                self.status_label.setText(f"Missing required data: {', '.join(missing)}")
        else:
            self.status_label.setText("Ready for analysis")

    def _load_data(self, data_type: str):
        """Load data from active layer."""
        active_layer = self._get_active_image_layer()
        if active_layer is None:
            QMessageBox.warning(self, "Warning", "No active image layer")
            return

        try:
            data = active_layer.data

            # Ensure 3D data for stacks
            if data_type in ['beads', 'cells']:
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                if data.ndim != 3:
                    raise ValueError(f"{data_type} stack must be 3D (frames, height, width)")
            else:  # reference
                if data.ndim != 2:
                    raise ValueError("Reference image must be 2D (height, width)")

            # Set data in manager
            if data_type == 'beads':
                self.data_manager.bead_stack = data
            elif data_type == 'reference':
                self.data_manager.reference_image = data
            elif data_type == 'cells':
                self.data_manager.cell_stack = data

            # Update UI state after any data load
            self._update_ui_state()

        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def analyze_all_frames(self):
        """Analyze displacement for all frames and create visualization stacks."""
        try:
            if not self._validate_input_data():
                return

            self._set_controls_enabled(False)
            self._update_status("Starting analysis...", 0)

            # Get input data
            reference = (self.data_manager.preprocessed_reference or
                         self.data_manager.reference_image)
            bead_stack = (self.data_manager.preprocessed_bead_stack or
                          self.data_manager.bead_stack)
            num_frames = len(bead_stack)

            # Initialize result arrays
            flows = []
            magnitudes = np.zeros((num_frames, *bead_stack.shape[1:]))
            overlay_stack = np.zeros((num_frames, *bead_stack.shape[1:], 3))

            # Get visualization parameters
            d_max = self.visualization_params['d_max'].value()
            vector_stride = self.visualization_params['vector_stride'].value()
            arrow_scale = self.visualization_params['arrow_scale'].value()

            # Pre-calculate vector data for all frames
            vector_data_cache = []
            vector_colors_cache = []

            # Process each frame
            for i in range(num_frames):
                progress = (i + 1) / num_frames * 100
                self._update_status(f"Processing frame {i + 1}/{num_frames}...", progress)

                # Calculate flow
                flow = self.analyzer.calculate_flow(reference, bead_stack[i])
                flows.append(flow)

                # Calculate magnitude
                magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
                magnitudes[i] = magnitude

                # Create overlay
                overlay_stack[i] = self.visualization_manager._create_overlay(
                    reference, bead_stack[i]
                )

                # Calculate vector data for this frame
                flow_scaled = flow * arrow_scale
                vectors, colors = self.visualization_manager._create_vector_visualization(
                    flow_scaled,
                    flow,
                    vector_stride,
                    d_max
                )
                vector_data_cache.append(vectors)
                vector_colors_cache.append(colors)

            # Store results
            results = {
                'flows': flows,
                'magnitudes': magnitudes,
                'overlay_stack': overlay_stack,
                'parameters': self.analyzer.params,
                'vector_cache': {
                    'data': vector_data_cache,
                    'colors': vector_colors_cache,
                    'parameters': {
                        'd_max': d_max,
                        'stride': vector_stride,
                        'arrow_scale': arrow_scale
                    }
                }
            }

            # Clear existing layers
            self.visualization_manager._clear_layers([
                'Displacement Overlay',
                'Displacement Magnitude',
                'Flow Vectors',
                'Cell Overlay'
            ])

            # Add visualization layers
            with self.viewer.events.blocker_all():
                # Add overlay stack
                self.viewer.add_image(
                    overlay_stack,
                    name='Displacement Overlay',
                    rgb=True,
                    blending='additive'
                )

                # Add magnitude stack
                if d_max is not None:
                    magnitudes = np.clip(magnitudes, 0, d_max)
                magnitude_layer = self.viewer.add_image(
                    magnitudes,
                    name='Displacement Magnitude',
                    colormap='viridis',
                    blending='additive'
                )

                # Add initial vector layer
                if len(vector_data_cache[0]) > 0:
                    self.viewer.add_shapes(
                        vector_data_cache[0],
                        shape_type='line',
                        name='Flow Vectors',
                        edge_color=vector_colors_cache[0],
                        edge_width=2,
                        blending='additive'
                    )

                # Add cell stack if available
                if self.data_manager.preprocessed_cell_stack is not None:
                    self.viewer.add_image(
                        self.data_manager.preprocessed_cell_stack,
                        name='Cell Overlay',
                        colormap='gray',
                        opacity=0.5,
                        blending='additive'
                    )

            # Update colorbar limits
            vmax = d_max if d_max is not None else magnitudes.max()
            self._update_colorbar(0, vmax)

            # Store results in data manager
            self.data_manager.displacement_results = results

            # Emit results
            self.displacement_calculated.emit(results)
            self._update_status("Analysis complete", 100)

        except Exception as e:
            self._handle_error(str(e))
        finally:
            self._set_controls_enabled(True)

    def _update_vector_layer(self, event=None):
        """Update vector layer using cached data."""
        try:
            if not hasattr(self.data_manager, 'displacement_results'):
                return

            results = self.data_manager.displacement_results
            if not results or 'vector_cache' not in results:
                return

            current_frame = self.viewer.dims.current_step[0]
            cache = results['vector_cache']

            if current_frame >= len(cache['data']):
                return

            # Check if visualization parameters have changed
            current_params = {
                'd_max': self.visualization_params['d_max'].value(),
                'stride': self.visualization_params['vector_stride'].value(),
                'arrow_scale': self.visualization_params['arrow_scale'].value()
            }

            # If parameters changed, need to recalculate for current frame
            if current_params != cache['parameters']:
                flow = results['flows'][current_frame]
                flow_scaled = flow * current_params['arrow_scale']
                vector_data = self._create_vector_data(flow_scaled, current_params['stride'])

                if len(vector_data) > 0:
                    orig_magnitudes = np.sqrt(np.sum(flow ** 2, axis=-1))
                    max_mag = current_params['d_max'] if current_params['d_max'] is not None else orig_magnitudes.max()

                    y_indices = vector_data[:, 0, 0].astype(int)
                    x_indices = vector_data[:, 0, 1].astype(int)
                    vector_magnitudes = orig_magnitudes[y_indices, x_indices]
                    colors = plt.cm.viridis(vector_magnitudes / max_mag)
                else:
                    vector_data = np.zeros((0, 2, 2))
                    colors = np.zeros((0, 4))
            else:
                # Use cached data
                vector_data = cache['data'][current_frame]
                colors = cache['colors'][current_frame]

            # Update or create vector layer
            vector_layer = None
            for layer in self.viewer.layers:
                if layer.name == 'Flow Vectors':
                    vector_layer = layer
                    break

            with self.viewer.events.blocker_all():
                if vector_layer is not None:
                    # Store current state
                    visible = vector_layer.visible

                    # Update data and colors
                    vector_layer.data = vector_data
                    vector_layer.edge_color = colors

                    # Restore state
                    vector_layer.visible = visible
                else:
                    # Create new vector layer if none exists
                    self.viewer.add_shapes(
                        vector_data,
                        shape_type='line',
                        name='Flow Vectors',
                        edge_color=colors,
                        edge_width=2,
                        blending='additive'
                    )

        except Exception as e:
            self._handle_error(f"Failed to update vector layer: {str(e)}")

    def _create_visualization_layers(self, results):
        """Create stack-based visualization layers."""
        try:
            # Clear existing layers
            self.visualization_manager._clear_displacement_layers()

            d_max = self.visualization_params['d_max'].value()

            with self.viewer.events.blocker_all():
                # Add 3D stack layers
                self.viewer.add_image(
                    results['overlay_stack'],
                    name='Displacement Overlay',
                    rgb=True,
                    blending='additive'
                )

                self.viewer.add_image(
                    results['magnitudes'],
                    name='Displacement Magnitude',
                    colormap='viridis',
                    blending='additive',
                    contrast_limits=[0, d_max if d_max is not None else results['magnitudes'].max()]
                )

                # Add cell stack if available
                if self.data_manager.preprocessed_cell_stack is not None:
                    self.viewer.add_image(
                        self.data_manager.preprocessed_cell_stack,
                        name='Cell Overlay',
                        colormap='gray',
                        opacity=0.5,
                        blending='additive'
                    )

        except Exception as e:
            self._handle_error(f"Failed to create visualization layers: {str(e)}")

    def _create_parameters_group(self) -> QGroupBox:
        """Create the analysis parameters group."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # TV-L1 parameters (removed d_max)
        params = [
            ("tau", "Tau:", 0.01, 1.0, 0.01, 0.25),
            ("lambda_", "Lambda:", 0.01, 1.0, 0.01, 0.4),
            ("theta", "Theta:", 0.1, 1.0, 0.1, 0.3),
            ("nscales", "Pyramid Scales:", 1, 10, 1, 3),
            ("warps", "Warps:", 1, 10, 1, 3),
            ("epsilon", "Epsilon:", 0.001, 0.1, 0.001, 0.01),
            ("inner_iterations", "Inner Iterations:", 1, 50, 1, 15),
            ("outer_iterations", "Outer Iterations:", 1, 20, 1, 5),
            ("scale_step", "Scale Step:", 0.1, 0.99, 0.01, 0.5),
            ("median_filtering", "Median Filter Size:", 1, 9, 2, 5)
        ]

        # Add all parameters
        for param_name, label, min_val, max_val, step, default in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))

            spin = QDoubleSpinBox() if isinstance(default, float) else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)

            self.parameter_spins[param_name] = spin
            row.addWidget(spin)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _create_visualization_parameters_group(self) -> QGroupBox:
        """Create the visualization parameters group with expanded ranges."""
        group = QGroupBox("Visualization Parameters")
        layout = QVBoxLayout()

        # Vector stride control
        stride_layout = QHBoxLayout()
        stride_layout.addWidget(QLabel("Vector Stride:"))
        self.visualization_params['vector_stride'] = QSpinBox()
        self.visualization_params['vector_stride'].setRange(1, 100)  # Increased range
        self.visualization_params['vector_stride'].setValue(20)
        self.visualization_params['vector_stride'].setToolTip("Display every nth vector")
        stride_layout.addWidget(self.visualization_params['vector_stride'])
        layout.addLayout(stride_layout)

        # Arrow scale control with increased range
        arrow_layout = QHBoxLayout()
        arrow_layout.addWidget(QLabel("Arrow Scale:"))
        self.visualization_params['arrow_scale'] = QDoubleSpinBox()
        self.visualization_params['arrow_scale'].setRange(0.1, 50.0)  # Increased range
        self.visualization_params['arrow_scale'].setSingleStep(0.5)
        self.visualization_params['arrow_scale'].setValue(1.0)
        self.visualization_params['arrow_scale'].setToolTip("Scale factor for arrow length")
        arrow_layout.addWidget(self.visualization_params['arrow_scale'])
        layout.addLayout(arrow_layout)

        # Maximum displacement visualization control
        dmax_layout = QHBoxLayout()
        dmax_layout.addWidget(QLabel("Max Displacement:"))
        self.visualization_params['d_max'] = QDoubleSpinBox()
        self.visualization_params['d_max'].setRange(0.1, 200.0)  # Increased range
        self.visualization_params['d_max'].setSingleStep(1.0)
        self.visualization_params['d_max'].setValue(10.0)
        self.visualization_params['d_max'].setToolTip("Maximum displacement for color scaling")
        dmax_layout.addWidget(self.visualization_params['d_max'])
        layout.addLayout(dmax_layout)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the action buttons frame."""
        frame = QFrame()
        layout = QHBoxLayout()

        self.preview_btn = QPushButton("Preview Current Frame")
        self.analyze_btn = QPushButton("Analyze All Frames")

        layout.addWidget(self.preview_btn)
        layout.addWidget(self.analyze_btn)

        frame.setLayout(layout)
        return frame

    def update_parameters(self):
        """Update analysis and visualization parameters."""
        try:
            params = TVL1Parameters(
                tau=self.parameter_spins['tau'].value(),
                lambda_=self.parameter_spins['lambda_'].value(),
                theta=self.parameter_spins['theta'].value(),
                nscales=self.parameter_spins['nscales'].value(),
                warps=self.parameter_spins['warps'].value(),
                epsilon=self.parameter_spins['epsilon'].value(),
                inner_iterations=self.parameter_spins['inner_iterations'].value(),
                outer_iterations=self.parameter_spins['outer_iterations'].value(),
                scale_step=self.parameter_spins['scale_step'].value(),
                median_filtering=self.parameter_spins['median_filtering'].value()
            )
            self.analyzer = DisplacementAnalyzer(params)

            # Update d_max in visualization manager
            self.visualization_manager.set_d_max(self.parameter_spins['d_max'].value())

        except ValueError as e:
            self._handle_error(str(e))

    def _create_data_loading_group(self) -> QGroupBox:
        """Create the data loading group."""
        group = QGroupBox("Data")
        layout = QVBoxLayout()

        # Load buttons
        self.load_beads_btn = QPushButton("Load Bead Stack")
        self.load_reference_btn = QPushButton("Load Reference Image")
        self.load_cells_btn = QPushButton("Load Cell Stack (Optional)")

        # Status labels
        self.bead_status = QLabel("Not loaded")
        self.reference_status = QLabel("Not loaded")
        self.cell_status = QLabel("Not loaded")

        # Add with status labels
        for btn, status in [
            (self.load_beads_btn, self.bead_status),
            (self.load_reference_btn, self.reference_status),
            (self.load_cells_btn, self.cell_status)
        ]:
            row = QHBoxLayout()
            row.addWidget(btn)
            row.addWidget(status)
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

    def _create_vector_data(self, flow: np.ndarray, stride: int = 20) -> np.ndarray:
        """Create vector data for visualization."""
        h, w = flow.shape[:2]
        stride = max(1, stride)  # Ensure stride is at least 1

        # Calculate grid points
        y_points = np.arange(stride // 2, h - stride // 2, stride)
        x_points = np.arange(stride // 2, w - stride // 2, stride)
        y, x = np.meshgrid(y_points, x_points, indexing='ij')

        # Get flow components
        u = flow[y, x, 0]  # x-component
        v = flow[y, x, 1]  # y-component

        # Calculate magnitudes
        magnitudes = np.sqrt(u ** 2 + v ** 2)

        # Create mask for significant displacements
        d_max = self.visualization_params['d_max'].value()
        if d_max is not None:
            threshold = d_max * 0.05  # 5% of max displacement
        else:
            threshold = magnitudes.max() * 0.05

        mask = magnitudes > threshold

        # Create vectors array
        vectors = []
        for i in range(len(y.flat)):
            if mask.flat[i]:
                start_x, start_y = x.flat[i], y.flat[i]
                end_x = start_x + u.flat[i]
                end_y = start_y + v.flat[i]

                # Check if endpoint is within image bounds
                if 0 <= end_x < w and 0 <= end_y < h:
                    vectors.append([[start_x, start_y], [end_x, end_y]])

        return np.array(vectors) if vectors else np.zeros((0, 2, 2))

    def _register_controls(self):
        """Register all controls with the base widget."""
        controls = [
            # Data loading controls
            self.load_beads_btn,
            self.load_reference_btn,
            self.load_cells_btn,

            # Analysis parameters
            *self.parameter_spins.values(),

            # Action buttons
            self.analyze_btn,
            self.preview_btn,

            # Status elements
            self.progress_bar,
            self.status_label,
            self.bead_status,
            self.reference_status,
            self.cell_status
        ]

        for control in controls:
            self.register_control(control)
