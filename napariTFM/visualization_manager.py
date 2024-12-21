import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import napari
import numpy as np
from matplotlib import pyplot as plt
from napari.layers import Layer

from .error_handling import ErrorHandlingMixin

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
        self._layers: Dict[str, Any] = {}
        # Properly initialize the preview config
        self._preview_config = PreviewConfig()

        # Connect to viewer events
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

    def handle_preview(self, frame: Optional[np.ndarray], enable: bool = True, layer_name: str = 'Preview') -> None:
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

            # Update preview config state
            self._preview_config.enabled = enable

        except Exception as e:
            logger.error(f"Preview handling failed: {str(e)}")
            raise

    def visualize_displacement_preview(
            self,
            flow: np.ndarray,
            reference: np.ndarray,
            moving: np.ndarray,
            d_max: float,
            vector_stride: int,
            arrow_scale: float
    ) -> None:
        """Visualize displacement preview for a single frame."""
        try:
            # Create overlay
            overlay = self._create_overlay(reference, moving)

            # Scale flow for visualization
            flow_scaled = flow * arrow_scale

            # Create vector data
            vectors, colors = self._create_vector_visualization(
                flow_scaled,
                flow,
                vector_stride,
                d_max
            )

            # Add or update visualization layers
            with self.viewer.events.blocker_all():
                # Update or create overlay layer
                if 'overlay' in self._layers and self._layers['overlay'] is not None:
                    self._layers['overlay'].data = overlay
                else:
                    self._layers['overlay'] = self.viewer.add_image(
                        overlay,
                        name='Displacement Overlay',
                        rgb=True,
                        blending='additive'
                    )

                # Add magnitude
                magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
                if d_max is not None:
                    magnitude = np.clip(magnitude, 0, d_max)

                if 'magnitude' in self._layers and self._layers['magnitude'] is not None:
                    self._layers['magnitude'].data = magnitude
                else:
                    self._layers['magnitude'] = self.viewer.add_image(
                        magnitude,
                        name='Displacement Magnitude',
                        colormap='viridis',
                        blending='additive'
                    )

                # Update or create vector layer
                if len(vectors) > 0:
                    if 'vectors' in self._layers and self._layers['vectors'] is not None:
                        self._layers['vectors'].data = vectors
                        self._layers['vectors'].edge_color = colors
                    else:
                        self._layers['vectors'] = self.viewer.add_vectors(
                            vectors,
                            name='Displacement Vectors',  # Fixed name
                            edge_color=colors,
                            edge_width=2,
                            vector_style='arrow',
                            blending='additive',
                            length=1
                        )

        except Exception as e:
            logger.error(f"Failed to visualize displacement preview: {str(e)}")
            raise

    def visualize_displacement_results(self, results: Dict) -> None:
        """Visualize displacement results for all frames."""
        try:
            flows = results['flows']
            vis_params = results['visualization_params']
            reference = self.data_manager.displacement_reference_image
            bead_stack = self.data_manager.displacement_bead_stack

            # Create visualization stacks
            num_frames = len(flows)
            magnitudes = np.zeros((num_frames, *flows[0].shape[:2]))
            overlay_stack = np.zeros((num_frames, *flows[0].shape[:2], 3))
            vector_data_cache = []
            vector_colors_cache = []

            # Process each frame
            for i in range(num_frames):
                # Calculate magnitude
                magnitude = np.sqrt(np.sum(flows[i] ** 2, axis=-1))
                if vis_params['d_max'] is not None:
                    magnitude = np.clip(magnitude, 0, vis_params['d_max'])
                magnitudes[i] = magnitude

                # Create overlay
                overlay_stack[i] = self._create_overlay(reference, bead_stack[i])

                # Calculate vector data
                flow_scaled = flows[i] * vis_params['arrow_scale']
                vectors, colors = self._create_vector_visualization(
                    flow_scaled,
                    flows[i],
                    vis_params['vector_stride'],
                    vis_params['d_max']
                )
                vector_data_cache.append(vectors)
                vector_colors_cache.append(colors)

            # Store vector cache in results
            results['vector_cache'] = {
                'data': vector_data_cache,
                'colors': vector_colors_cache
            }

            # Add or update visualization layers
            with self.viewer.events.blocker_all():
                # Update or create overlay layer
                if 'overlay' in self._layers and self._layers['overlay'] is not None:
                    self._layers['overlay'].data = overlay_stack
                else:
                    self._layers['overlay'] = self.viewer.add_image(
                        overlay_stack,
                        name='Displacement Overlay',
                        rgb=True,
                        blending='additive'
                    )

                # Update or create magnitude layer
                if 'magnitude' in self._layers and self._layers['magnitude'] is not None:
                    self._layers['magnitude'].data = magnitudes
                else:
                    self._layers['magnitude'] = self.viewer.add_image(
                        magnitudes,
                        name='Displacement Magnitude',
                        colormap='viridis',
                        blending='additive'
                    )

                # Update or create vector layer
                current_frame = self.viewer.dims.current_step[0]
                if len(vector_data_cache[current_frame]) > 0:
                    if 'vectors' in self._layers and self._layers['vectors'] is not None:
                        self._layers['vectors'].data = vector_data_cache[current_frame]
                        self._layers['vectors'].edge_color = vector_colors_cache[current_frame]
                    else:
                        self._layers['vectors'] = self.viewer.add_vectors(
                            vector_data_cache[current_frame],
                            name='Displacement Vectors',  # Fixed name
                            edge_color=vector_colors_cache[current_frame],
                            edge_width=2,
                            vector_style='arrow',
                            blending='additive',
                            length=1
                        )

        except Exception as e:
            logger.error(f"Failed to visualize displacement results: {str(e)}")
            raise

    def update_force_visualization(self, results: Dict[str, Any], visualization_params: Dict[str, Any]) -> None:
        """Update force visualization with current results and parameters."""
        try:
            # Clear existing force layers
            self._clear_layers(['Force Magnitude', 'Force Vectors'])

            # Calculate magnitude stack
            magnitude_stack = np.sqrt(results['tx'] ** 2 + results['ty'] ** 2)

            # Get visualization parameters
            f_max = visualization_params['f_max']
            vector_stride = visualization_params['vector_stride']
            arrow_scale = visualization_params['arrow_scale']

            # Create vector cache for all frames
            vector_cache = {
                'data': [],
                'colors': [],
                'parameters': visualization_params.copy()
            }

            # Process each frame
            for frame_idx in range(len(results['tx'])):
                force_vectors = np.stack([
                    results['tx'][frame_idx],
                    results['ty'][frame_idx]
                ], axis=-1)

                vectors, colors = self._create_vector_visualization(
                    force_vectors * arrow_scale,
                    force_vectors,
                    vector_stride,
                    f_max
                )
                vector_cache['data'].append(vectors)
                vector_cache['colors'].append(colors)

            # Store vector cache in data manager
            self.data_manager.force_vector_cache = vector_cache

            # Add visualization layers
            with self.viewer.events.blocker_all():
                # Add magnitude layer with clipping
                if f_max is not None:
                    magnitude_stack = np.clip(magnitude_stack, 0, f_max)

                magnitude_layer = self.viewer.add_image(
                    magnitude_stack,
                    name='Force Magnitude',
                    colormap='inferno',
                    blending='additive'
                )
                self._layers['force_magnitude'] = magnitude_layer

                # Add initial vector layer
                current_frame = self.viewer.dims.current_step[0]
                if len(vector_cache['data'][current_frame]) > 0:
                    vector_layer = self.viewer.add_vectors(
                        vector_cache['data'][current_frame],
                        edge_color=vector_cache['colors'][current_frame],
                        edge_width=2,
                        name='Force Vectors',
                        blending='additive'
                    )
                    self._layers['force_vectors'] = vector_layer

        except Exception as e:
            self.handle_error(f"Failed to update force visualization: {str(e)}")
            raise

    def _on_frame_changed(self, event=None) -> None:
        """Handle frame change events for both displacement and force visualizations."""
        try:
            current_frame = self.viewer.dims.current_step[0]

            # Handle displacement vectors
            if hasattr(self.data_manager, 'displacement_results'):
                results = self.data_manager.displacement_results
                if results and 'vector_cache' in results:
                    cache = results['vector_cache']
                    if current_frame < len(cache['data']):
                        vector_layer = self._layers.get('vectors')
                        if vector_layer is not None:
                            with self.viewer.events.blocker_all():
                                vector_layer.data = cache['data'][current_frame]
                                vector_layer.edge_color = cache['colors'][current_frame]

            # Handle force vectors
            if hasattr(self.data_manager, 'force_vector_cache'):
                cache = self.data_manager.force_vector_cache
                if current_frame < len(cache['data']):
                    vector_layer = self._layers.get('force_vectors')
                    if vector_layer is not None:
                        with self.viewer.events.blocker_all():
                            vector_layer.data = cache['data'][current_frame]
                            vector_layer.edge_color = cache['colors'][current_frame]

        except Exception as e:
            self.handle_error(f"Failed to update frame visualization: {str(e)}")

    def get_force_statistics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate force statistics."""
        try:
            # Calculate magnitudes
            tx = results['tx']
            ty = results['ty']
            magnitudes = np.sqrt(tx ** 2 + ty ** 2)

            # Calculate statistics
            stats = {
                'mean_force': float(np.mean(magnitudes)),
                'max_force': float(np.max(magnitudes)),
                'median_force': float(np.median(magnitudes)),
                'std_force': float(np.std(magnitudes))
            }

            return stats

        except Exception as e:
            self.handle_error(f"Failed to calculate force statistics: {str(e)}")
            return {}

    def _on_layer_removed(self, event) -> None:
        """Handle layer removal events."""
        layer = event.value
        # Remove from tracked layers if present
        self._layers = {name: layer_obj for name, layer_obj in self._layers.items()
                        if layer_obj != layer}

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Disconnect events
            if self.viewer is not None:
                self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
                self.viewer.layers.events.removed.disconnect(self._on_layer_removed)

            # Clear layers
            self._clear_layers([name for name in self._layers])
            self._layers.clear()
            self.viewer = None

        except Exception as e:
            self.handle_error(f"Failed to cleanup visualization manager: {str(e)}")

    def update_displacement_frame(self, frame_index: int) -> None:
        """Update vector visualization for the current frame."""
        if not hasattr(self.data_manager, 'displacement_results'):
            return

        results = self.data_manager.displacement_results
        if not results or 'vector_cache' not in results:
            return

        cache = results['vector_cache']
        if frame_index >= len(cache['data']):
            return

        # Update vectors using stored layer reference
        if 'vectors' in self._layers and self._layers['vectors'] is not None:
            with self.viewer.events.blocker_all():
                # Update vectors and colors for current frame
                self._layers['vectors'].data = cache['data'][frame_index]
                self._layers['vectors'].edge_color = cache['colors'][frame_index]

    def _clear_layers(self, layer_names: List[str]) -> None:
        """Remove specified layers from viewer"""
        for name in layer_names:
            for layer in list(self.viewer.layers):
                if layer.name == name:
                    self.viewer.layers.remove(layer)
                    # Also remove from our layer dictionary if present
                    for key, stored_layer in list(self._layers.items()):
                        if stored_layer == layer:
                            del self._layers[key]

    def _create_vector_visualization(
            self,
            flow_scaled: np.ndarray,
            original_flow: np.ndarray,
            stride: int,
            d_max: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create vector data and colors for visualization."""
        h, w = flow_scaled.shape[:2]
        stride = max(1, stride)

        # Create regular grid of positions
        y_points = np.arange(stride // 2, h - stride // 2, stride)
        x_points = np.arange(stride // 2, w - stride // 2, stride)
        Y, X = np.meshgrid(y_points, x_points, indexing='ij')

        # Get flow components
        U = flow_scaled[Y, X, 0]  # x-component
        V = flow_scaled[Y, X, 1]  # y-component

        # Calculate original magnitudes for coloring
        orig_u = original_flow[Y, X, 0]
        orig_v = original_flow[Y, X, 1]
        magnitudes = np.sqrt(orig_u ** 2 + orig_v ** 2)

        # Filter out small displacements
        if d_max is not None:
            threshold = d_max * 0.01
        else:
            threshold = magnitudes.max() * 0.01

        mask = magnitudes > threshold

        # Create vectors array in correct format (N, 2, D)
        Y_flat = Y[mask]
        X_flat = X[mask]
        U_flat = U[mask]
        V_flat = V[mask]

        N = len(Y_flat)
        vectors = np.zeros((N, 2, 2))  # (N, 2, 2) for N vectors with start/end points in 2D

        # Start points
        vectors[:, 0, 1] = X_flat  # x coordinates
        vectors[:, 0, 0] = Y_flat  # y coordinates

        # End points
        vectors[:, 1, 1] = U_flat  # x + dx
        vectors[:, 1, 0] = V_flat  # y + dy

        # Create colors for filtered vectors
        max_mag = d_max if d_max is not None else magnitudes.max()
        if max_mag > 0:
            colors = plt.cm.viridis(magnitudes[mask] / max_mag)
        else:
            colors = plt.cm.viridis(np.zeros(N))

        return vectors, colors


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

    def _create_overlay(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Create colored overlay of two images"""
        img1_norm = img1.astype(float) / img1.max()
        img2_norm = img2.astype(float) / img2.max()

        overlay = np.zeros((*img1.shape, 3))
        overlay[..., 0] = img1_norm * 0.7  # Red channel
        overlay[..., 1] = img2_norm * 0.7  # Green channel
        overlay[..., 2] = (img1_norm + img2_norm) * 0.3  # Blue channel

        return np.clip(overlay, 0, 1)

    def get_displacement_statistics(self, flow: np.ndarray) -> dict:
        """Calculate displacement statistics."""
        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
        return {
            'max': magnitude.max(),
            'mean': magnitude.mean(),
            'std': magnitude.std(),
            'median': np.median(magnitude)
        }
