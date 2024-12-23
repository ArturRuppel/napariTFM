import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import cv2
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

    def update_preprocessing_visualization(self, results: Dict) -> None:
        """
        Update visualization after preprocessing.

        Parameters
        ----------
        results : dict
            Dictionary containing preprocessing results
        """
        if 'beads' in results and 'reference' in results:
            processed_beads, _ = results['beads']
            processed_reference, _ = results['reference']
            if processed_beads is not None and processed_reference is not None:
                # Create the bead-reference overlay
                self.create_bead_overlay(processed_beads, processed_reference)

        # Keep all existing visualization code exactly the same
        if 'beads' in results:
            processed_beads, _ = results['beads']
            if processed_beads is not None:
                if 'Preprocessed Beads' in self.viewer.layers:
                    self.viewer.layers.remove('Preprocessed Beads')
                self.viewer.add_image(
                    processed_beads,
                    name='Preprocessed Beads',
                    colormap='green',
                    visible=True
                )

        if 'reference' in results:
            processed_reference, _ = results['reference']
            if processed_reference is not None:
                if 'Preprocessed Reference' in self.viewer.layers:
                    self.viewer.layers.remove('Preprocessed Reference')
                self.viewer.add_image(
                    processed_reference,
                    name='Preprocessed Reference',
                    colormap='magenta',
                    visible=True
                )

        if 'cells' in results:
            processed_cells, _ = results['cells']
            if processed_cells is not None:
                if 'Preprocessed Cells' in self.viewer.layers:
                    self.viewer.layers.remove('Preprocessed Cells')
                self.viewer.add_image(
                    processed_cells,
                    name='Preprocessed Cells',
                    colormap='gray',
                    visible=True
                )

    def create_bead_overlay(self, bead_stack: np.ndarray, reference_image: np.ndarray) -> None:
        """Create combined bead-reference overlay layer.

        Parameters
        ----------
        bead_stack : np.ndarray
            Preprocessed bead stack
        reference_image : np.ndarray
            Preprocessed reference image
        """
        # Remove existing overlay if present
        if 'Bead Overlay' in self.viewer.layers:
            self.viewer.layers.remove('Bead Overlay')

        # Create RGB overlay stack
        overlay_stack = self._create_overlay_stack(bead_stack, reference_image)

        # Add overlay layer
        self.viewer.add_image(
            overlay_stack,
            name='Bead Overlay',
            visible=True,
            rgb=True
        )

    def _create_overlay_stack(self, bead_stack: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
        """Create RGB overlay stack combining beads (green) and reference (magenta).

        Parameters
        ----------
        bead_stack : np.ndarray
            Bead image stack
        reference_image : np.ndarray
            Reference image

        Returns
        -------
        np.ndarray
            RGB overlay stack
        """
        # Get dimensions
        num_frames = len(bead_stack)
        height, width = bead_stack.shape[1:]

        # Create RGB stack
        overlay_stack = np.zeros((num_frames, height, width, 3), dtype=float)

        # Normalize reference image
        reference = reference_image.astype(float)
        ref_min = reference.min()
        ref_max = reference.max()
        if ref_max > ref_min:
            reference = (reference - ref_min) / (ref_max - ref_min)

        # Process each frame
        for i in range(num_frames):
            # Normalize bead frame
            bead_frame = bead_stack[i].astype(float)
            bead_min = bead_frame.min()
            bead_max = bead_frame.max()
            if bead_max > bead_min:
                bead_frame = (bead_frame - bead_min) / (bead_max - bead_min)

            # Combine into RGB (magenta reference, green beads)
            overlay_stack[i, :, :, 0] = reference  # Red channel (for magenta)
            overlay_stack[i, :, :, 1] = bead_frame  # Green channel
            overlay_stack[i, :, :, 2] = reference  # Blue channel (for magenta)

        return overlay_stack

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
            d_max: float,
            vector_stride: int,
            arrow_scale: float,
            downscale_factor: int = 1
    ) -> None:
        """Visualize displacement preview for a single frame."""
        try:
            # If downscaled, upscale flow for visualization only
            display_flow = cv2.resize(
                flow,
                (flow.shape[1] * downscale_factor, flow.shape[0] * downscale_factor),
                interpolation=cv2.INTER_LINEAR
            )


            # Scale flow for visualization
            flow_scaled = display_flow * arrow_scale

            # Create vector data
            vectors, colors = self._create_vector_visualization(
                flow_scaled,
                display_flow,
                vector_stride,
                d_max
            )

            # Add or update visualization layers
            with self.viewer.events.blocker_all():
                # Add magnitude
                magnitude = np.sqrt(np.sum(display_flow ** 2, axis=-1))

                if 'magnitude' in self._layers and self._layers['magnitude'] is not None:
                    self._layers['magnitude'].data = magnitude
                    self._layers['magnitude'].contrast_limits = (0, d_max)
                else:
                    self._layers['magnitude'] = self.viewer.add_image(
                        magnitude,
                        name='Displacement Magnitude',
                        colormap='viridis',
                        blending='additive',
                        contrast_limits=(0, d_max)
                    )

                # Update or create vector layer
                if len(vectors) > 0:
                    if 'vectors' in self._layers and self._layers['vectors'] is not None:
                        self._layers['vectors'].data = vectors
                        self._layers['vectors'].edge_color = colors
                    else:
                        self._layers['vectors'] = self.viewer.add_vectors(
                            vectors,
                            name='Displacement Vectors',
                            edge_color=colors,
                            edge_width=2,
                            vector_style='arrow',
                            blending='additive',
                            length=1
                        )

        except Exception as e:
            logger.error(f"Failed to visualize displacement preview: {str(e)}")
            raise

    def visualize_displacement_results(self, results: Dict, downscale_factor: int = 1) -> None:
        """Visualize displacement results for all frames."""
        try:
            flows = results['flows']
            vis_params = results['visualization_params']

            # Create visualization stacks - use original flow shape before upscaling
            num_frames = len(flows)
            # Initialize magnitudes with the upscaled shape
            if downscale_factor > 1:
                upscaled_shape = (
                    flows[0].shape[0] * downscale_factor,
                    flows[0].shape[1] * downscale_factor
                )
                magnitudes = np.zeros((num_frames, *upscaled_shape))
            else:
                magnitudes = np.zeros((num_frames, *flows[0].shape[:2]))

            vector_data_cache = []
            vector_colors_cache = []

            for i in range(num_frames):
                # If downscaled, upscale flow for visualization
                if downscale_factor > 1:
                    display_flow = cv2.resize(
                        flows[i],
                        (flows[i].shape[1] * downscale_factor, flows[i].shape[0] * downscale_factor),
                        interpolation=cv2.INTER_LINEAR
                    )
                    # Scale the vectors to account for resolution change
                    display_flow *= downscale_factor
                else:
                    display_flow = flows[i]

                # Calculate magnitude from display flow - now matches the upscaled size
                magnitude = np.sqrt(np.sum(display_flow ** 2, axis=-1))
                magnitudes[i] = magnitude  # This should now match in size

                # Calculate vector data using display flow
                flow_scaled = display_flow * vis_params['arrow_scale']
                vectors, colors = self._create_vector_visualization(
                    flow_scaled,
                    display_flow,
                    vis_params['vector_stride'],
                    vis_params['d_max']
                )
                vector_data_cache.append(vectors)
                vector_colors_cache.append(colors)

            # Store vector cache in results
            vector_cache = {
                'data': vector_data_cache,
                'colors': vector_colors_cache
            }
            results['vector_cache'] = vector_cache

            # Add or update visualization layers
            with self.viewer.events.blocker_all():
                # Update or create magnitude layer
                if 'magnitude' in self._layers and self._layers['magnitude'] is not None:
                    self._layers['magnitude'].data = magnitudes
                    self._layers['magnitude'].contrast_limits = (0, vis_params['d_max'])
                else:
                    self._layers['magnitude'] = self.viewer.add_image(
                        magnitudes,
                        name='Displacement Magnitude',
                        colormap='viridis',
                        blending='additive',
                        contrast_limits=(0, vis_params['d_max'])
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
                            name='Displacement Vectors',
                            edge_color=vector_colors_cache[current_frame],
                            edge_width=2,
                            vector_style='arrow',
                            blending='additive',
                            length=1
                        )

        except Exception as e:
            logger.error(f"Failed to visualize displacement results: {str(e)}")
            raise

    def visualize_force_preview(
            self,
            force_x: np.ndarray,
            force_y: np.ndarray,
            f_max: float,
            vector_stride: int,
            arrow_scale: float
    ) -> None:
        """Visualize force preview for a single frame."""
        try:
            # Create combined force field
            force_field = np.stack([force_x, force_y], axis=-1)

            # Create vector data and colors
            vectors, colors = self._create_vector_visualization(
                force_field * arrow_scale,
                force_field,
                vector_stride,
                f_max,
                colormap='inferno'
            )

            # Add or update visualization layers
            with self.viewer.events.blocker_all():
                # Add magnitude
                magnitude = np.sqrt(np.sum(force_field ** 2, axis=-1))
                magnitude = np.clip(magnitude, 0, f_max)

                if 'force_magnitude' in self._layers:
                    self._layers['force_magnitude'].data = magnitude
                    self._layers['force_magnitude'].contrast_limits = (0, f_max)
                else:
                    self._layers['force_magnitude'] = self.viewer.add_image(
                        magnitude,
                        name='Force Magnitude',
                        colormap='inferno',
                        blending='additive',
                        contrast_limits=(0, f_max)
                    )

                # Update or create vector layer
                if len(vectors) > 0:
                    if 'force_vectors' in self._layers:
                        self._layers['force_vectors'].data = vectors
                        self._layers['force_vectors'].edge_color = colors
                    else:
                        self._layers['force_vectors'] = self.viewer.add_vectors(
                            vectors,
                            name='Force Vectors',
                            edge_color=colors,
                            edge_width=2,
                            vector_style='arrow',
                            blending='additive',
                            length=1
                        )

        except Exception as e:
            logger.error(f"Failed to visualize force preview: {str(e)}")
            raise

    def visualize_force_results(self, results: Dict[str, Any], visualization_params: Dict[str, Any]) -> None:
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
                    f_max,
                    colormap='inferno'
                )
                vector_cache['data'].append(vectors)
                vector_cache['colors'].append(colors)

            # Store vector cache in results instead of data manager for preview case
            if 'parameters' in results:  # Full analysis case
                self.data_manager.force_vector_cache = vector_cache
            else:  # Preview case
                results['vector_cache'] = vector_cache

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
                        vector_style='arrow',
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
            d_max: Optional[float],
            colormap: str = 'viridis'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create vector data and colors for visualization.

        Parameters
        ----------
        flow_scaled : np.ndarray
            Scaled flow field for vector display
        original_flow : np.ndarray
            Original flow field for magnitude calculation
        stride : int
            Spacing between vectors
        d_max : Optional[float]
            Maximum value for color normalization
        colormap : str
            Name of the matplotlib colormap to use (default: 'viridis')

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Vector data and colors arrays
        """
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

        # Create colors for filtered vectors using specified colormap
        max_mag = d_max if d_max is not None else magnitudes.max()
        if max_mag > 0:
            colors = plt.cm.get_cmap(colormap)(magnitudes[mask] / max_mag)
        else:
            colors = plt.cm.get_cmap(colormap)(np.zeros(N))

        return vectors, colors

    def get_displacement_statistics(self, flow: np.ndarray) -> dict:
        """Calculate displacement statistics."""
        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
        return {
            'max': magnitude.max(),
            'mean': magnitude.mean(),
            'std': magnitude.std(),
            'median': np.median(magnitude)
        }
