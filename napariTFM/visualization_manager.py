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

    def _create_vector_visualization(
            self,
            flow_scaled: np.ndarray,
            original_flow: np.ndarray,
            stride: int,
            d_max: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create vector data and colors for visualization using napari's vector layer format.

        Returns [N, 2, 2] array containing N vectors with start and end points in 2D,
        and corresponding colors."""
        h, w = flow_scaled.shape[:2]
        stride = max(1, stride)

        # Create regular grid of positions
        y_points = np.arange(stride // 2, h - stride // 2, stride)
        x_points = np.arange(stride // 2, w - stride // 2, stride)
        Y, X = np.meshgrid(y_points, x_points, indexing='ij')

        # Get flow components at these positions
        U = flow_scaled[Y, X, 0]  # x-component
        V = flow_scaled[Y, X, 1]  # y-component

        # Calculate original magnitudes for coloring
        orig_u = original_flow[Y, X, 0]
        orig_v = original_flow[Y, X, 1]
        magnitudes = np.sqrt(orig_u ** 2 + orig_v ** 2)

        # Create vectors array with start and end points
        n_vectors = len(Y.flatten())
        vectors = np.zeros((n_vectors, 2, 2))  # [N, 2, 2] array

        # Start points
        vectors[:, 0, 0] = Y.flatten()  # y coordinates
        vectors[:, 0, 1] = X.flatten()  # x coordinates

        # End points
        vectors[:, 1, 0] = V.flatten()  # y + dy
        vectors[:, 1, 1] = U.flatten()  # x + dx

        # Create colors
        max_mag = d_max if d_max is not None else magnitudes.max()
        if max_mag > 0:
            colors = plt.cm.viridis(magnitudes.flatten() / max_mag)
        else:
            colors = plt.cm.viridis(np.zeros(n_vectors))

        return vectors, colors

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


