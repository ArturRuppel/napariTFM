import logging

from matplotlib import pyplot as plt
from napari.layers import Layer
from qtpy import QtWidgets

from .error_handling import ErrorSeverity, ErrorHandlingMixin

logger = logging.getLogger(__name__)

from threading import Lock
from typing import Optional, Dict

import napari
import numpy as np
from napari.layers import Layer


class VisualizationManager(ErrorHandlingMixin):
    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager

        # Initialize attributes
        self._displacement_layers = {}
        self._d_max = None
        self._updating = False
        self._layer_lock = Lock()
        self._current_dims = None

        # Initialize colorbar attributes
        self._colorbar_widget = None
        self._active_magnitude_layer = None

        # Connect to viewer events
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

    def _update_visualization(self, results: Dict):
        """Update visualization of force calculation results with integrated colorbar."""
        try:
            # Remove existing force layers
            for layer_name in ['Force Magnitude', 'Force Vectors']:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)

            # Calculate magnitude stack with integrated colorbar
            magnitude_stack = np.sqrt(results['tx'] ** 2 + results['ty'] ** 2)
            magnitude_with_colorbar = self._create_magnitude_with_colorbar(magnitude_stack)

            # Add magnitude layer
            self.viewer.add_image(
                magnitude_with_colorbar,
                name='Force Magnitude',
                colormap='inferno',
                blending='additive'
            )

            # Update status with statistics
            mean_force = np.mean(magnitude_stack)
            max_force = np.max(magnitude_stack)
            mean_contractile = np.mean(results['contractile_force'])

            stats_text = (
                f"Mean force magnitude: {mean_force:.2f} Pa\n"
                f"Max force magnitude: {max_force:.2f} Pa\n"
                f"Mean contractile force: {mean_contractile:.2e} N"
            )
            self._update_status(stats_text)

        except Exception as e:
            self._handle_error(f"Failed to update visualization: {str(e)}")

    def update_displacement_visualization(
            self,
            reference: Optional[np.ndarray] = None,
            moving: Optional[np.ndarray] = None,
            flow: Optional[np.ndarray] = None,
            flow_scaled: Optional[np.ndarray] = None,
            cells: Optional[np.ndarray] = None,
            show_overlay: bool = True,
            show_vectors: bool = True,
            show_magnitude: bool = True,
            vector_stride: int = 20,
            d_max: Optional[float] = None
    ) -> None:
        """Update displacement visualization with integrated colorbar."""
        if self._updating:
            return

        try:
            self._updating = True
            self._d_max = d_max
            logger.debug("Starting displacement visualization update")

            with self._layer_lock:
                self._clear_displacement_layers()

                if flow is None:
                    raise ValueError("No displacement data available")

                with self.viewer.events.blocker_all():
                    # Image overlay
                    if show_overlay and reference is not None and moving is not None:
                        logger.debug("Adding displacement overlay")
                        overlay = self._create_overlay(reference, moving)
                        self._displacement_layers['overlay'] = self.viewer.add_image(
                            overlay,
                            name='Displacement Overlay',
                            rgb=True,
                            blending='additive'
                        )

                    # Magnitude display with integrated colorbar
                    if show_magnitude:
                        logger.debug("Adding magnitude layer with colorbar")
                        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
                        if d_max is not None:
                            magnitude = np.clip(magnitude, 0, d_max)

                        magnitude_with_colorbar = self._create_magnitude_with_colorbar(magnitude)
                        magnitude_layer = self.viewer.add_image(
                            magnitude_with_colorbar,
                            name='Displacement Magnitude',
                            colormap='viridis',
                            blending='additive'
                        )
                        self._displacement_layers['magnitude'] = magnitude_layer

                    # Vector display
                    if show_vectors and flow_scaled is not None:
                        logger.debug("Adding vector layer")
                        vector_data = self._create_vector_data(flow_scaled, vector_stride)
                        if len(vector_data) > 0:
                            orig_magnitudes = np.sqrt(np.sum(flow ** 2, axis=-1))
                            max_mag = self._d_max if self._d_max is not None else orig_magnitudes.max()

                            y_indices = vector_data[:, 0, 0].astype(int)
                            x_indices = vector_data[:, 0, 1].astype(int)
                            vector_magnitudes = orig_magnitudes[y_indices, x_indices]
                            colors = plt.cm.viridis(vector_magnitudes / max_mag)

                            vectors_layer = self.viewer.add_shapes(
                                vector_data,
                                shape_type='line',
                                name='Flow Vectors',
                                edge_color=colors,
                                edge_width=2,
                                blending='additive'
                            )
                            self._displacement_layers['vectors'] = vectors_layer

                    if cells is not None:
                        logger.debug("Adding cell overlay")
                        self._displacement_layers['cells'] = self.viewer.add_image(
                            cells,
                            name='Cell Overlay',
                            colormap='gray',
                            opacity=0.5,
                            blending='additive'
                        )

        except Exception as e:
            logger.error(f"Failed to update displacement visualization: {str(e)}")
            self.handle_error(self.create_error(
                message="Failed to update displacement visualization",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check input data and parameters"
            ))
        finally:
            self._updating = False

    def _refresh_colorbar(self, title: str = "Magnitude"):
        """Refresh the colorbar after changes to the layer."""
        if self._active_magnitude_layer is not None and self._colorbar_widget is not None:
            self._update_colorbar(self._active_magnitude_layer, title)

    def _on_layer_removed(self, event):
        """Handle layer removal events."""
        layer = event.value
        if layer == self._active_magnitude_layer:
            if self._colorbar_widget is not None:
                self.viewer.window.remove_dock_widget(self._colorbar_widget)
                self._colorbar_widget = None
                self._active_magnitude_layer = None

    def _update_colorbar(self, layer: "napari.layers.Image", title: str = "Magnitude"):
        """Update or create colorbar for the given layer."""
        logger.debug(f"Updating colorbar for {title}")
        try:
            # Remove existing colorbar if it exists
            if self._colorbar_widget is not None:
                logger.debug("Removing existing colorbar")
                try:
                    self.viewer.window.remove_dock_widget(self._colorbar_widget)
                except Exception:
                    logger.debug("Could not remove existing colorbar widget")
                    pass

            # Create new colorbar
            logger.debug("Creating new colorbar")
            widget_tuple = self._create_colorbar_widget(layer, title)
            if isinstance(widget_tuple, tuple):
                self._colorbar_widget, canvas = widget_tuple
            else:
                self._colorbar_widget = widget_tuple
                canvas = None

            self._active_magnitude_layer = layer

            # Add the colorbar widget to the right side of the main viewer
            logger.debug("Adding colorbar to viewer")
            self.viewer.window.add_dock_widget(
                self._colorbar_widget,
                name=f"{title} Colorbar",
                area='right',
                allowed_areas=['right'],  # Only allow right docking
                add_vertical_stretch=False
            )

            # Set size constraints
            self._colorbar_widget.setFixedWidth(100)
            if hasattr(self._colorbar_widget, 'setMaximumHeight'):
                self._colorbar_widget.setMaximumHeight(400)

            # Connect to layer events if layer is valid
            logger.debug("Connecting layer events")
            if hasattr(layer, 'events') and hasattr(layer.events, 'contrast_limits'):
                layer.events.contrast_limits.connect(lambda _: self._refresh_colorbar(title))
            logger.debug("Colorbar update complete")

        except Exception as e:
            logger.error(f"Failed to update colorbar: {str(e)}")
            raise

    def _create_colorbar_widget(self, layer: "napari.layers.Image", title: str = "Magnitude"):
        """Create a colorbar widget with optimized size and appearance."""
        from qtpy.QtWidgets import QWidget, QVBoxLayout
        from qtpy.QtCore import Qt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        # Create figure with adjusted size ratio
        fig = Figure(figsize=(1.0, 4))
        fig.patch.set_facecolor('#262930')

        # Create canvas
        canvas = FigureCanvasQTAgg(fig)
        canvas.setStyleSheet("background-color: #262930;")

        # Create colorbar axes with adjusted position
        ax = fig.add_axes([0.35, 0.03, 0.3, 0.94])
        ax.patch.set_alpha(0)

        mappable = plt.cm.ScalarMappable(
            norm=plt.Normalize(
                vmin=layer.contrast_limits[0],
                vmax=layer.contrast_limits[1]
            ),
            cmap='viridis'
        )

        colorbar = fig.colorbar(mappable, cax=ax, label=title)
        colorbar.ax.yaxis.label.set_color('white')
        colorbar.ax.tick_params(colors='white')

        # Create widget with specific sizing policy
        widget = QWidget()
        widget.setStyleSheet("background-color: #262930; color: white;")
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.MinimumExpanding
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(canvas)
        widget.setLayout(layout)

        return widget, canvas

    def _create_overlay(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Create colored overlay of two images with blue channel."""
        # Normalize images
        img1_norm = img1.astype(float) / img1.max()
        img2_norm = img2.astype(float) / img2.max()

        # Create RGB overlay
        overlay = np.zeros((*img1.shape, 3))
        overlay[..., 0] = img1_norm * 0.7  # Red channel for first image
        overlay[..., 1] = img2_norm * 0.7  # Green channel for second image
        overlay[..., 2] = (img1_norm + img2_norm) * 0.3  # Blue channel mixed from both

        return np.clip(overlay, 0, 1)

    def _create_vector_data(self, flow: np.ndarray, stride: int = 20) -> np.ndarray:
        """Create vector data for napari visualization with bounds checking."""
        h, w = flow.shape[:2]

        # Ensure stride is at least 1
        stride = max(1, stride)

        # Calculate grid points with bounds checking
        y_points = np.arange(stride // 2, h - stride // 2, stride)
        x_points = np.arange(stride // 2, w - stride // 2, stride)
        y, x = np.meshgrid(y_points, x_points, indexing='ij')

        # Get flow components for valid points
        u = flow[y, x, 0]  # x-component
        v = flow[y, x, 1]  # y-component

        # Calculate magnitudes
        magnitudes = np.sqrt(u ** 2 + v ** 2)

        # Create mask for significant displacements
        if self._d_max is not None:
            threshold = self._d_max * 0.05  # 5% of max displacement
        else:
            threshold = magnitudes.max() * 0.05

        mask = magnitudes > threshold

        # Create vectors array with bounds checking
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

    def _clear_displacement_layers(self):
        """Remove all displacement-related layers."""
        layers_to_remove = []
        for layer in self.viewer.layers:
            if isinstance(layer, Layer) and layer.name in [
                'Displacement Overlay',
                'Displacement Magnitude',
                'Flow Vectors',
                'Cell Overlay'
            ]:
                layers_to_remove.append(layer)

        for layer in layers_to_remove:
            self.viewer.layers.remove(layer)

        self._displacement_layers.clear()

    def _calculate_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """Calculate displacement magnitude with optional scaling."""
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        if self._d_max is not None:
            magnitude = np.clip(magnitude / self._d_max, 0, 1)

        return magnitude

    def get_displacement_statistics(self, flow: np.ndarray) -> dict:
        """Calculate displacement statistics."""
        magnitude = self._calculate_magnitude(flow)
        return {
            'max': magnitude.max(),
            'mean': magnitude.mean(),
            'std': magnitude.std(),
            'median': np.median(magnitude)
        }





    def set_data_manager(self, data_manager: "DataManager"):
        """Allow setting the data manager after initialization."""
        self.data_manager = data_manager



    def cleanup(self) -> None:
        """Clean up resources when closing."""
        self.clear_visualization()
        self.viewer = None



    def _on_frame_changed(self, event=None):
        """Handle frame change events by updating vector layer if needed."""
        # Skip if no displacement results available
        if not hasattr(self.data_manager, 'displacement_results'):
            return

        results = self.data_manager.displacement_results
        if not results or 'flows' not in results:
            return

        current_frame = self.viewer.dims.current_step[0]
        if current_frame >= len(results['flows']):
            return

        # Only update vector layer if it exists
        if 'vectors' in self._displacement_layers:
            vector_layer = self._displacement_layers['vectors']
            if vector_layer in self.viewer.layers:
                # Get cached vector data for current frame
                if 'vector_cache' in results and current_frame < len(results['vector_cache']['data']):
                    vector_data = results['vector_cache']['data'][current_frame]
                    colors = results['vector_cache']['colors'][current_frame]

                    # Update vector layer
                    with self.viewer.events.blocker_all():
                        vector_layer.data = vector_data
                        vector_layer.edge_color = colors



