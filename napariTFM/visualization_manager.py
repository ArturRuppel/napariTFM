import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import napari
import numpy as np
from matplotlib import pyplot as plt
from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.layers import Layer

from .error_handling import ErrorSeverity, ErrorHandlingMixin

logger = logging.getLogger(__name__)


@dataclass
class PreviewConfig:
    """Configuration for preview visualization"""
    enabled: bool = False
    current_data_type: str = 'beads'  # 'beads', 'reference', or 'cells'
    original_layer: Optional["napari.layers.Layer"] = None
    preview_layer: Optional["napari.layers.Layer"] = None


class VisualizationManager(ErrorHandlingMixin):
    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager

        # Remove colorbar-related state
        self._layers: Dict[str, Layer] = {}
        self._d_max: Optional[float] = None
        self._active_magnitude_layer: Optional["napari.layers.Image"] = None
        self._preview_config = PreviewConfig()

        # Connect to viewer events
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

    def update_displacement_visualization(
            self,
            flow: np.ndarray,
            reference: Optional[np.ndarray] = None,
            moving: Optional[np.ndarray] = None,
            cells: Optional[np.ndarray] = None,
            vector_stride: int = 20,
            arrow_scale: float = 1.0,
            d_max: Optional[float] = None
    ) -> None:
        """Update displacement visualization with all components"""
        try:
            logger.debug("Starting displacement visualization update")
            self._d_max = d_max
            self._clear_layers(['Displacement Overlay', 'Displacement Magnitude', 'Flow Vectors', 'Cell Overlay'])

            with self.viewer.events.blocker_all():
                # Create overlay if reference and moving images are provided
                if reference is not None and moving is not None:
                    overlay = self._create_overlay(reference, moving)
                    self._layers['overlay'] = self.viewer.add_image(
                        overlay,
                        name='Displacement Overlay',
                        rgb=True,
                        blending='additive'
                    )

                # Create magnitude layer with colorbar
                magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
                if d_max is not None:
                    magnitude = np.clip(magnitude, 0, d_max)

                magnitude_layer = self.viewer.add_image(
                    magnitude,
                    name='Displacement Magnitude',
                    colormap='viridis',
                    blending='additive',
                    colorbar=True,  # Enable colorbar
                    colorbar_label='Displacement (pixels)'  # Add label
                )
                self._layers['magnitude'] = magnitude_layer

                # Create vector visualization
                flow_scaled = flow * arrow_scale
                vector_data, colors = self._create_vector_visualization(
                    flow_scaled,
                    flow,
                    vector_stride,
                    d_max
                )

                if len(vector_data) > 0:
                    self._layers['vectors'] = self.viewer.add_shapes(
                        vector_data,
                        shape_type='line',
                        name='Flow Vectors',
                        edge_color=colors,
                        edge_width=2,
                        blending='additive'
                    )

                # Add cell overlay if provided
                if cells is not None:
                    self._layers['cells'] = self.viewer.add_image(
                        cells,
                        name='Cell Overlay',
                        colormap='gray',
                        opacity=0.5,
                        blending='additive'
                    )

        except Exception as e:
            logger.error(f"Failed to update displacement visualization: {str(e)}")
            self.handle_error(self.create_error(
                message="Visualization update failed",
                details=str(e),
                severity=ErrorSeverity.ERROR
            ))

    def update_force_visualization(self, results: Dict) -> Dict[str, float]:
        """Update force calculation visualization with both magnitude and vector representation"""
        try:
            self._clear_layers(['Force Magnitude', 'Force Vectors'])

            # Calculate magnitude stack
            magnitude_stack = np.sqrt(results['tx'] ** 2 + results['ty'] ** 2)

            # Add magnitude layer
            magnitude_layer = self.viewer.add_image(
                magnitude_stack,
                name='Force Magnitude',
                colormap='inferno',
                blending='additive'
            )
            self._layers['force_magnitude'] = magnitude_layer

            # Create vector visualization
            vector_stride = 20  # Can be made configurable if needed
            arrow_scale = 1.0  # Can be made configurable if needed
            d_max = magnitude_stack.max()  # Use maximum force for scaling

            num_frames = len(results['tx'])
            vector_data_cache = []
            vector_colors_cache = []

            for frame in range(num_frames):
                # Combine force components into flow-like array
                force_field = np.stack(
                    (results['tx'][frame], results['ty'][frame]),
                    axis=-1
                )

                # Scale the forces for visualization
                force_field_scaled = force_field * arrow_scale

                # Create vector visualization
                vectors, colors = self._create_vector_visualization(
                    force_field_scaled,
                    force_field,
                    vector_stride,
                    d_max
                )

                vector_data_cache.append(vectors)
                vector_colors_cache.append(colors)

            # Add initial vector layer
            if len(vector_data_cache[0]) > 0:
                vector_layer = self.viewer.add_shapes(
                    vector_data_cache[0],
                    shape_type='line',
                    name='Force Vectors',
                    edge_color=vector_colors_cache[0],
                    edge_width=2,
                    blending='additive'
                )
                self._layers['force_vectors'] = vector_layer

            # Store vector cache for frame changes
            if not hasattr(self.data_manager, 'force_vector_cache'):
                self.data_manager.force_vector_cache = {}

            self.data_manager.force_vector_cache = {
                'data': vector_data_cache,
                'colors': vector_colors_cache,
                'parameters': {
                    'stride': vector_stride,
                    'arrow_scale': arrow_scale,
                    'd_max': d_max
                }
            }

            # Return basic statistics
            stats = {
                'mean_force': float(np.mean(magnitude_stack)),
                'max_force': float(np.max(magnitude_stack))
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to update force visualization: {str(e)}")
            raise

    def _on_frame_changed(self, event=None) -> None:
        """Handle frame change events for both displacement and force visualizations"""
        current_frame = self.viewer.dims.current_step[0]

        # Handle displacement vectors
        if hasattr(self.data_manager, 'displacement_results'):
            results = self.data_manager.displacement_results
            if results and 'vector_cache' in results:
                if current_frame < len(results['vector_cache']['data']):
                    if 'vectors' in self._layers:
                        vector_layer = self._layers['vectors']
                        vector_layer.data = results['vector_cache']['data'][current_frame]
                        vector_layer.edge_color = results['vector_cache']['colors'][current_frame]

        # Handle force vectors
        if hasattr(self.data_manager, 'force_vector_cache'):
            cache = self.data_manager.force_vector_cache
            if current_frame < len(cache['data']):
                if 'force_vectors' in self._layers:
                    vector_layer = self._layers['force_vectors']
                    vector_layer.data = cache['data'][current_frame]
                    vector_layer.edge_color = cache['colors'][current_frame]

    def _on_layer_removed(self, event) -> None:
        """Handle layer removal events"""
        layer = event.value
        if layer == self._active_magnitude_layer:
            self._active_magnitude_layer = None

    def cleanup(self) -> None:
        """Clean up resources"""
        self._clear_layers([name for name in self._layers])
        self._layers.clear()
        self.viewer = None


    def update_preprocessing_visualization(self, results: Dict[str, Tuple[np.ndarray, List[Dict]]]) -> None:
        """Update preprocessing visualization"""
        try:
            # Remove existing preprocessed layers
            self._clear_layers(['Preprocessed Beads', 'Preprocessed Reference', 'Preprocessed Cells'])

            # Add new layers
            if 'beads' in results:
                processed_beads, _ = results['beads']
                self._layers['preprocessed_beads'] = self.viewer.add_image(
                    processed_beads,
                    name='Preprocessed Beads',
                    visible=True
                )

            if 'reference' in results:
                processed_ref, _ = results['reference']
                self._layers['preprocessed_ref'] = self.viewer.add_image(
                    processed_ref,
                    name='Preprocessed Reference',
                    visible=True
                )

            if 'cells' in results:
                processed_cells, _ = results['cells']
                self._layers['preprocessed_cells'] = self.viewer.add_image(
                    processed_cells,
                    name='Preprocessed Cells',
                    visible=True
                )

        except Exception as e:
            logger.error(f"Failed to update preprocessing visualization: {str(e)}")
            raise

    def handle_preview(self, frame: np.ndarray, enable: bool = True, layer_name: str = 'Preview') -> None:
        """Handle preview visualization"""
        try:
            if enable:
                if self._preview_config.preview_layer is None:
                    self._preview_config.preview_layer = self.viewer.add_image(
                        frame,
                        name=layer_name,
                        visible=True
                    )
                else:
                    self._preview_config.preview_layer.data = frame
            else:
                if self._preview_config.preview_layer is not None:
                    self.viewer.layers.remove(self._preview_config.preview_layer)
                    self._preview_config.preview_layer = None

        except Exception as e:
            logger.error(f"Preview handling failed: {str(e)}")
            raise

    def _create_vector_visualization(
            self,
            flow_scaled: np.ndarray,
            original_flow: np.ndarray,
            stride: int,
            d_max: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create vector data and colors for visualization"""
        h, w = flow_scaled.shape[:2]
        stride = max(1, stride)

        # Calculate grid points
        y_points = np.arange(stride // 2, h - stride // 2, stride)
        x_points = np.arange(stride // 2, w - stride // 2, stride)
        y, x = np.meshgrid(y_points, x_points, indexing='ij')

        # Get flow components
        u = flow_scaled[y, x, 0]
        v = flow_scaled[y, x, 1]

        # Calculate magnitudes
        orig_magnitudes = np.sqrt(
            original_flow[y, x, 0] ** 2 +
            original_flow[y, x, 1] ** 2
        )

        # Create mask for significant displacements
        threshold = d_max * 0.05 if d_max is not None else orig_magnitudes.max() * 0.05
        mask = orig_magnitudes > threshold

        # Create vectors array
        vectors = []
        valid_magnitudes = []

        for i in range(len(y.flat)):
            if mask.flat[i]:
                start_x, start_y = x.flat[i], y.flat[i]
                end_x = start_x + u.flat[i]
                end_y = start_y + v.flat[i]

                if 0 <= end_x < w and 0 <= end_y < h:
                    vectors.append([[start_x, start_y], [end_x, end_y]])
                    valid_magnitudes.append(orig_magnitudes.flat[i])

        if vectors:
            vectors = np.array(vectors)
            max_mag = d_max if d_max is not None else max(valid_magnitudes)
            colors = plt.cm.viridis(np.array(valid_magnitudes) / max_mag)
        else:
            vectors = np.zeros((0, 2, 2))
            colors = np.zeros((0, 4))

        return vectors, colors

    def _create_overlay(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Create colored overlay of two images"""
        img1_norm = img1.astype(float) / img1.max()
        img2_norm = img2.astype(float) / img2.max()

        overlay = np.zeros((*img1.shape, 3))
        overlay[..., 0] = img1_norm * 0.7  # Red channel
        overlay[..., 1] = img2_norm * 0.7  # Green channel
        overlay[..., 2] = (img1_norm + img2_norm) * 0.3  # Blue channel

        return np.clip(overlay, 0, 1)

    def _clear_layers(self, layer_names: List[str]) -> None:
        """Remove specified layers from viewer"""
        for name in layer_names:
            for layer in list(self.viewer.layers):
                if layer.name == name:
                    self.viewer.layers.remove(layer)

    def get_displacement_statistics(self, flow: np.ndarray) -> dict:
        """Calculate displacement statistics."""
        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
        return {
            'max': magnitude.max(),
            'mean': magnitude.mean(),
            'std': magnitude.std(),
            'median': np.median(magnitude)
        }


