import colorsys
import logging
from threading import Lock
from typing import Optional, Dict, List
from typing import Tuple, Union

import napari
import numpy as np
from matplotlib import pyplot as plt
from napari.layers import Layer, Labels, Points
from napari.utils.transforms import Affine
from qtpy import QtWidgets

from .error_handling import ErrorSeverity, ErrorHandlingMixin, ApplicationError

logger = logging.getLogger(__name__)

import colorsys
import logging
from threading import Lock
from typing import Optional, Dict, List, Tuple, Union

import napari
import numpy as np
from napari.layers import Layer, Labels, Points
from napari.utils.transforms import Affine


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

    def _create_magnitude_with_colorbar(self, magnitude_data: np.ndarray) -> np.ndarray:
        """Create magnitude data with integrated colorbar."""
        if magnitude_data.ndim == 2:
            return self._create_single_magnitude_with_colorbar(magnitude_data)
        elif magnitude_data.ndim == 3:
            # Handle stack of magnitude data
            frames = []
            for frame in magnitude_data:
                frames.append(self._create_single_magnitude_with_colorbar(frame))
            return np.stack(frames)
        else:
            raise ValueError(f"Invalid magnitude data dimensions: {magnitude_data.ndim}")

    def _create_single_magnitude_with_colorbar(self, magnitude: np.ndarray) -> np.ndarray:
        """Create a single frame with integrated colorbar."""
        # Calculate dimensions
        data_height, data_width = magnitude.shape
        colorbar_width = max(50, data_width // 20)  # Adaptive width, min 50 pixels
        padding = max(10, data_width // 100)  # Adaptive padding

        # Create combined image
        total_width = data_width + colorbar_width + padding
        combined = np.zeros((data_height, total_width))

        # Add magnitude data
        combined[:, :data_width] = magnitude

        # Create colorbar
        vmin, vmax = np.nanmin(magnitude), np.nanmax(magnitude)
        if vmin == vmax:
            vmax = vmin + 1  # Prevent division by zero

        colorbar_values = np.linspace(vmin, vmax, data_height)
        colorbar = np.tile(colorbar_values[::-1], (colorbar_width, 1)).T

        # Add colorbar
        combined[:, -colorbar_width:] = colorbar

        # Add text labels (optional)
        # We could add value labels here using cv2.putText if needed

        return combined

    def _update_colormap_range(self, layer: "napari.layers.Image"):
        """Update colormap range for the magnitude layer."""
        if layer.data.ndim == 3:
            # For image stack
            data_portion = layer.data[:, :, :-self.colorbar_width]
        else:
            # For single image
            data_portion = layer.data[:, :-self.colorbar_width]

        vmin = np.nanmin(data_portion)
        vmax = np.nanmax(data_portion)
        layer.contrast_limits = (vmin, vmax)

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

    def _add_magnitude_layer(self, magnitude_data: np.ndarray, title: str = "Magnitude"):
        """Create a magnitude layer with integrated colorbar."""
        # Calculate dimensions for the combined image
        data_height, data_width = magnitude_data.shape
        colorbar_width = 50  # Width in pixels for colorbar
        padding = 10  # Padding between data and colorbar

        # Create combined image with space for colorbar
        combined = np.zeros((data_height, data_width + colorbar_width + padding))

        # Add magnitude data to main portion
        combined[:, :data_width] = magnitude_data

        # Create colorbar data
        colorbar = np.linspace(magnitude_data.min(), magnitude_data.max(), data_height)
        colorbar = np.tile(colorbar[::-1], (colorbar_width, 1)).T

        # Add colorbar to the right portion
        combined[:, -colorbar_width:] = colorbar

        # Add to viewer
        layer = self.viewer.add_image(
            combined,
            name=title,
            colormap='viridis',
            blending='additive'
        )

        return layer

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

    def _generate_distinct_color(self) -> np.ndarray:
        """Generate a distinct color using golden ratio."""
        golden_ratio = 0.618033988749895
        hue = self._color_cycle.random()

        # Keep generating until we get a distinct color
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            hue += golden_ratio
            hue %= 1.0

            # Convert to RGB with good saturation and value
            hsv = np.array([hue, 0.8, 0.95])
            rgb = np.array(colorsys.hsv_to_rgb(*hsv))
            color = np.append(rgb, 1.0)  # Add alpha channel

            # Convert to tuple for set comparison
            color_tuple = tuple(color)

            if color_tuple not in self._used_colors:
                self._used_colors.add(color_tuple)
                return color

            attempts += 1

        # If we couldn't find a distinct color, return a random one
        return np.append(self._color_cycle.random(3), 1.0)

    def _reset_color_cycle(self):
        """Reset the color generation state."""
        self._color_cycle = np.random.RandomState(42)
        self._used_colors.clear()

    def update_tracking_visualization(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> None:
        """Update tracking visualization with layer preservation and error handling."""
        if self._updating:
            logger.debug("VisualizationManager: Update cancelled - already updating")
            return

        try:
            self._updating = True
            with self._layer_lock:
                update_data = self._prepare_update_data(data)

                with self.viewer.events.blocker_all():
                    if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                        self.tracking_layer.data = update_data
                        self.tracking_layer.refresh()
                    else:
                        self.tracking_layer = self.viewer.add_labels(
                            update_data,
                            name='Segmentation',
                            opacity=0.5,
                            visible=True
                        )

        except ValueError as e:
            self.handle_error(self.create_error(
                message="Invalid data format",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check data dimensions and format"
            ))
        except Exception as e:
            self.handle_error(self.create_error(
                message="Failed to update visualization",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                original_error=e
            ))
        finally:
            self._updating = False

    def _prepare_update_data(self, data: Union[np.ndarray, Tuple[np.ndarray, int]]) -> np.ndarray:
        """Prepare data for visualization update with validation."""
        try:
            if isinstance(data, tuple):
                frame_data, frame_index = data
                if not isinstance(frame_data, np.ndarray):
                    raise ValueError("Frame data must be a numpy array")

                num_frames = self.data_manager._num_frames
                if frame_index >= num_frames:
                    raise ValueError(f"Frame index {frame_index} exceeds stack size {num_frames}")

                if self.tracking_layer is None:
                    update_data = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
                    update_data[frame_index] = frame_data
                else:
                    update_data = self._update_existing_stack(frame_data, frame_index, num_frames)
            else:
                if not isinstance(data, np.ndarray):
                    raise ValueError("Data must be a numpy array")
                update_data = self._prepare_full_stack(data)

            return update_data

        except ValueError as e:
            raise ApplicationError(
                message="Invalid data format",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check input data format and dimensions"
            )
        except Exception as e:
            raise ApplicationError(
                message="Failed to prepare visualization data",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                original_error=e
            )

    def clear_visualization(self):
        """Remove all visualization layers with error handling."""
        if self._updating:
            return

        try:
            self._updating = True
            self._remove_layers_safely()
            self._current_dims = None
            logger.debug("Cleared all visualization layers")

        except Exception as e:
            self.handle_error(self.create_error(
                message="Failed to clear visualization",
                details=str(e),
                severity=ErrorSeverity.WARNING,
                recovery_hint="Some layers may need to be removed manually"
            ))
        finally:
            self._updating = False

    def _remove_layers_safely(self):
        """Safely remove layers with individual error handling."""
        for layer_name in ['tracking_layer', 'overlay_layer']:
            layer = getattr(self, layer_name, None)
            if layer is not None:
                try:
                    if layer in self.viewer.layers:
                        self.viewer.layers.remove(layer)
                    setattr(self, layer_name, None)
                except Exception as e:
                    self.handle_error(self.create_error(
                        message=f"Failed to remove {layer_name}",
                        details=str(e),
                        severity=ErrorSeverity.WARNING
                    ))

    def validate_stack_consistency(self) -> bool:
        """Validate visualization state consistency with detailed error reporting."""
        try:
            if self.tracking_layer is None:
                return True

            if self._current_dims is not None:
                if self.tracking_layer.data.shape != self._current_dims:
                    self.handle_error(self.create_error(
                        message="Visualization shape mismatch",
                        details=f"Expected {self._current_dims}, got {self.tracking_layer.data.shape}",
                        severity=ErrorSeverity.WARNING,
                        recovery_hint="Data may not be displayed correctly"
                    ))
                    return False

            if self.data_manager is not None and self.data_manager.segmentation_data is not None:
                stack_shape = self.data_manager.segmentation_data.shape
                visualization_shape = self.tracking_layer.data.shape

                if stack_shape != visualization_shape:
                    self.handle_error(self.create_error(
                        message="Stack shape mismatch",
                        details=f"DataManager={stack_shape}, Visualization={visualization_shape}",
                        severity=ErrorSeverity.WARNING,
                        recovery_hint="Visualization may be out of sync with data"
                    ))
                    return False

            return True

        except Exception as e:
            self.handle_error(self.create_error(
                message="Failed to validate stack consistency",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                original_error=e
            ))
            return False

    def _update_existing_stack(self, frame_data: np.ndarray, frame_index: int, num_frames: int) -> np.ndarray:
        """Update existing stack with new frame data."""
        if self.tracking_layer.data.shape[0] < num_frames:
            new_data = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
            new_data[:self.tracking_layer.data.shape[0]] = self.tracking_layer.data
            update_data = new_data
        else:
            update_data = self.tracking_layer.data.copy()
        update_data[frame_index] = frame_data
        return update_data

    def _prepare_full_stack(self, data: np.ndarray) -> np.ndarray:
        """Prepare full stack data for update."""
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.shape[0] < self.data_manager._num_frames:
            new_data = np.zeros((self.data_manager._num_frames, *data.shape[1:]), dtype=data.dtype)
            new_data[:data.shape[0]] = data
            return new_data
        return data.copy()

    def _handle_layer_removal(self, event):
        """Handle layer removal events safely."""
        layer = event.value
        if layer == self.tracking_layer:
            logger.debug("VisualizationManager: Tracking layer was removed")
            with self._layer_lock:
                if layer not in self.viewer.layers:
                    self.tracking_layer = None
                    self._current_dims = None

    def _create_tracking_layer(self, data: np.ndarray) -> "napari.layers.Labels":
        """Create or update tracking layer safely."""
        try:
            with self.viewer.events.blocker_all():
                with self._layer_lock:
                    # Always try to update existing layer first
                    if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                        logger.debug("VisualizationManager: Updating existing tracking layer")
                        self.tracking_layer.data = data
                        self.tracking_layer.refresh()
                        return self.tracking_layer

                    # Create new layer only if necessary
                    logger.debug("VisualizationManager: Creating new tracking layer")
                    layer = self.viewer.add_labels(
                        data,
                        name='Segmentation',
                        opacity=0.5,
                        visible=True
                    )
                    self.tracking_layer = layer
                    return layer
        except Exception as e:
            logger.error(f"Error creating tracking layer: {e}")
            raise

    def get_current_tracking_layer(self) -> Optional["napari.layers.Labels"]:
        """Get current tracking layer with validation."""
        with self._layer_lock:
            if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                return self.tracking_layer
            return None

    def _update_full_stack(self, stack_data: np.ndarray) -> None:
        """Update full stack while preserving layer."""
        logger.debug(f"Updating full stack with shape {stack_data.shape}")

        # Handle 2D data
        if stack_data.ndim == 2:
            stack_data = stack_data[np.newaxis, ...]

        # Validate dimensions
        if stack_data.ndim != 3:
            raise ValueError(f"Stack data must be 3D, got shape {stack_data.shape}")

        logger.debug(f"VisualizationManager: Starting full stack update")
        logger.debug(f"VisualizationManager: Input data shape: {stack_data.shape}")
        logger.debug(f"VisualizationManager: Input data unique values: {np.unique(stack_data)}")

        if self.tracking_layer is not None:
            logger.debug(f"VisualizationManager: Current tracking layer: {self.tracking_layer.name}")
            logger.debug(f"VisualizationManager: Tracking layer in viewer: {self.tracking_layer in self.viewer.layers}")
            if self.tracking_layer in self.viewer.layers:
                logger.debug(f"VisualizationManager: Current tracking data shape: {self.tracking_layer.data.shape}")
                logger.debug(f"VisualizationManager: Current tracking unique values: {np.unique(self.tracking_layer.data)}")

        # Log all current layers
        logger.debug("VisualizationManager: Current viewer layers:")
        for layer in self.viewer.layers:
            logger.debug(f"  - {layer.name} ({type(layer)})")

        try:
            # Update existing layer if possible
            if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                self.tracking_layer.data = stack_data
                self.tracking_layer.refresh()
            else:
                # Create new layer only if necessary
                self.tracking_layer = self.viewer.add_labels(
                    stack_data,
                    name='Segmentation',
                    opacity=0.5,
                    visible=True
                )

            self._current_dims = stack_data.shape

            logger.debug("VisualizationManager: After update:")
            logger.debug(f"VisualizationManager: Tracking layer still exists: {self.tracking_layer is not None}")
            if self.tracking_layer is not None:
                logger.debug(f"VisualizationManager: Tracking layer in viewer: {self.tracking_layer in self.viewer.layers}")
                logger.debug(f"VisualizationManager: Updated data shape: {self.tracking_layer.data.shape}")
                logger.debug(f"VisualizationManager: Updated unique values: {np.unique(self.tracking_layer.data)}")


        except Exception as e:
            logger.error(f"Error updating full stack: {e}")
            raise

    def _update_single_frame(self, frame_data: np.ndarray, frame_index: int) -> None:
        """Update a single frame in the visualization."""
        if frame_data.ndim != 2:
            logger.debug(f"Invalid frame data shape: {frame_data.shape}")
            raise ValueError(f"Frame data must be 2D, got shape {frame_data.shape}")

        logger.debug(f"Updating frame {frame_index}")

        if self.tracking_layer is None:
            # Initialize with proper dimensions
            if self._current_dims is None:
                num_frames = int(self.viewer.dims.range[0][1] + 1)
                empty_stack = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
                empty_stack[frame_index] = frame_data
                self.tracking_layer = self._create_tracking_layer(empty_stack)
                self._current_dims = empty_stack.shape
            else:
                # Use existing dimensions
                empty_stack = np.zeros(self._current_dims, dtype=frame_data.dtype)
                empty_stack[frame_index] = frame_data
                self.tracking_layer = self._create_tracking_layer(empty_stack)
        else:
            # Update existing layer
            current_data = self.tracking_layer.data.copy()
            if frame_data.shape != current_data.shape[1:]:
                raise ValueError(
                    f"Frame shape {frame_data.shape} doesn't match existing data shape "
                    f"{current_data.shape[1:]}"
                )

            current_data[frame_index] = frame_data
            self.tracking_layer.data = current_data

    def set_data_manager(self, data_manager: "DataManager"):
        """Allow setting the data manager after initialization."""
        self.data_manager = data_manager

    def _setup_layer_transforms(self, layer: "napari.layers.Layer", data_shape: Tuple[int, ...]) -> None:
        """Set up proper transforms for the layer based on data dimensions."""
        ndim = len(data_shape)
        scale = np.ones(ndim)
        translate = np.zeros(ndim)

        affine_matrix = np.eye(ndim + 1)
        affine_matrix[:-1, :-1] = np.diag(scale)
        affine_matrix[:-1, -1] = translate

        transform = Affine(affine_matrix=affine_matrix)
        layer.affine = transform

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the data for the current frame."""
        if self.tracking_layer is None:
            return None

        current_step = int(self.viewer.dims.point[0])
        return self.tracking_layer.data[current_step]

    def set_layer_visibility(self, visible: bool) -> None:
        """Set the visibility of the tracking layer."""
        if self.tracking_layer is not None:
            self.tracking_layer.visible = visible

    def cleanup(self) -> None:
        """Clean up resources when closing."""
        self.clear_visualization()
        self.viewer = None

    def clear_edge_layers(self) -> None:
        """Remove all edge-related layers"""
        layer_names = ['Edge Analysis', 'Intercalation Events', 'Cell Edges']
        for name in layer_names:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def clear_all_layers(self) -> None:
        """Remove all managed layers from the viewer"""
        if self._tracked_layer is not None and self._tracked_layer in self.viewer.layers:
            self.viewer.layers.remove(self._tracked_layer)
        self._tracked_layer = None

        self.clear_edge_layers()
        self._edge_layer = None
        self._intercalation_layer = None
        self._analysis_layer = None

    def generate_distinct_colors(self, n: int) -> List[tuple]:
        """Generate n visually distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((*rgb, 1.0))  # Add alpha channel
        return colors

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

    @property
    def tracked_layer(self) -> Optional[Labels]:
        """The current tracked cells layer"""
        return self._tracked_layer

    @property
    def edge_layer(self) -> Optional[Points]:
        """The current edge detection layer"""
        return self._edge_layer

    @property
    def analysis_layer(self) -> Optional[Points]:
        """The current edge analysis layer"""
        return self._analysis_layer

    def _create_image_overlay(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Create colored overlay of two images."""
        # Normalize images
        img1_norm = img1.astype(float) / img1.max()
        img2_norm = img2.astype(float) / img2.max()

        # Create RGB overlay
        overlay = np.zeros((*img1.shape, 3))
        overlay[..., 0] += img1_norm * 0.9  # Red channel for first image
        overlay[..., 1] += img2_norm * 0.9  # Green channel for second image

        return np.clip(overlay, 0, 1)

