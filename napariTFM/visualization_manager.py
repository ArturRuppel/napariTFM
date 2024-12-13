import colorsys
import logging
from threading import Lock
from typing import Optional, Dict, List
from typing import Tuple, Union

import napari
import numpy as np
from napari.layers import Layer, Labels, Points
from napari.utils.transforms import Affine

from .error_handling import ErrorSeverity, ErrorHandlingMixin, ApplicationError

logger = logging.getLogger(__name__)


class VisualizationManager(ErrorHandlingMixin):
    """Manages visualization of cell tracking results in napari."""

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        # Initialize ErrorHandlingMixin
        super().__init__()

        # Store main components
        self.viewer = viewer
        self.data_manager = data_manager

        # Initialize layer attributes
        self.tracking_layer = None
        self._edge_layer = None
        self._intercalation_layer = None
        self._analysis_layer = None

        # Initialize state variables
        self._updating = False
        self._layer_lock = Lock()
        self._current_dims = None

        # Initialize color generation
        self._color_cycle = np.random.RandomState(42)  # For reproducible colors
        self._used_colors = set()

        # Connect to layer removal event with error handling
        try:
            self.viewer.layers.events.removed.connect(self._handle_layer_removal)
        except Exception as e:
            self.handle_error(self.create_error(
                message="Failed to connect layer events",
                details=str(e),
                severity=ErrorSeverity.WARNING,
                recovery_hint="Layer removal events may not be handled properly"
            ))

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

    def _generate_distinct_color(self) -> np.ndarray:
        """Generate a distinct color using golden ratio"""
        golden_ratio = 0.618033988749895
        hue = self._color_cycle.random()
        hue += golden_ratio
        hue %= 1
        # Convert to RGB
        hsv = np.array([hue, 0.8, 0.95])
        rgb = np.array(colorsys.hsv_to_rgb(*hsv))
        return np.append(rgb, 1.0)  # Add alpha channel

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


