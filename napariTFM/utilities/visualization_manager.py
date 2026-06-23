import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import cv2
import napari
import numpy as np
from matplotlib import pyplot as plt

from napariTFM.utilities.error_handling import ErrorSeverity, ErrorHandlingMixin
from napariTFM.utilities.viewer_colorbar import ViewerColorbarManager

logger = logging.getLogger(__name__)



@dataclass
class PreviewConfig:
    """Configuration for preview visualization"""
    enabled: bool = False
    current_data_type: str = 'beads'  # 'beads', 'reference', or 'cells'
    original_layer: Optional["napari.layers.Layer"] = None
    preview_layer: Optional["napari.layers.Layer"] = None


class VisualizationManager(ErrorHandlingMixin):
    # region === Initialization
    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self._layers: Dict[str, Any] = {}
        self.colorbar_manager = ViewerColorbarManager(viewer)
        self._preview_config = PreviewConfig()

        # Connect to viewer events
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.colorbar_manager.clear()

            # Disconnect other events
            if self.viewer is not None:
                self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
                self.viewer.layers.events.removed.disconnect(self._on_layer_removed)

            # Clear layers
            self._clear_layers([name for name in self._layers])
            self._layers.clear()
            self.viewer = None

        except Exception as e:
            self.handle_error(self.create_error(
                "Failed to cleanup visualization manager",
                details=str(e),
                original_error=e,
            ))

    # endregion

    # region === Event Handlers
    def _on_frame_changed(self, event=None) -> None:
        """Handle frame change events for both displacement and force visualizations."""
        try:
            current_frame = self.viewer.dims.current_step[0]

            # Handle displacement vectors
            if (hasattr(self.data_manager, 'displacement_vector_cache') and
                    self.data_manager.displacement_vector_cache is not None):
                cache = self.data_manager.displacement_vector_cache
                if ('data' in cache and current_frame < len(cache['data']) and
                        'displacement_vectors' in self._layers and
                        self._layers['displacement_vectors'] is not None):
                    with self.viewer.events.blocker_all():
                        self._layers['displacement_vectors'].data = cache['data'][current_frame]
                        self._layers['displacement_vectors'].edge_color = cache['colors'][current_frame]

            # Handle force vectors
            if (hasattr(self.data_manager, 'force_vector_cache') and
                    self.data_manager.force_vector_cache is not None):
                cache = self.data_manager.force_vector_cache
                if ('data' in cache and current_frame < len(cache['data']) and
                        'force_vectors' in self._layers and
                        self._layers['force_vectors'] is not None):
                    with self.viewer.events.blocker_all():
                        self._layers['force_vectors'].data = cache['data'][current_frame]
                        self._layers['force_vectors'].edge_color = cache['colors'][current_frame]

        except Exception as e:
            error = self.create_error(
                message="Failed to update frame visualization",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check vector cache and layer consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)

    def _on_layer_removed(self, event) -> None:
        """Handle layer removal events."""
        layer = event.value
        # Remove from tracked layers if present
        self._layers = {name: layer_obj for name, layer_obj in self._layers.items()
                        if layer_obj != layer}

    # endregion

    # region === Layer Management
    def _clear_layers(self, display_names: List[str]) -> None:
        """Clear specified layers from the viewer."""
        for name in display_names:
            for layer in list(self.viewer.layers):
                if layer.name == name:
                    self.viewer.layers.remove(layer)
                    # Also clear from tracking dict if present
                    if name in self._layers:
                        self._layers[name] = None

    def _upscale_field(self, field: np.ndarray, downscale_factor: int) -> np.ndarray:
        """Upscale a vector field for visualization."""
        if downscale_factor <= 1:
            return field

        return cv2.resize(
            field,
            (field.shape[1] * downscale_factor, field.shape[0] * downscale_factor),
            interpolation=cv2.INTER_LINEAR
        )

    def clear_disp_vector_cache(self) -> None:
        """Clear displacement vector cache from data manager."""
        if hasattr(self.data_manager, 'displacement_vector_cache'):
            self.data_manager.displacement_vector_cache = None

    def clear_force_vector_cache(self) -> None:
        """Clear force vector cache from data manager."""
        if hasattr(self.data_manager, 'force_vector_cache'):
            self.data_manager.force_vector_cache = None

    def _validate_frame_index(self, frame_index: int, num_frames: int) -> int:
        """
        Validate and adjust frame index to be within bounds.

        Parameters
        ----------
        frame_index : int
            Current frame index from viewer
        num_frames : int
            Total number of available frames

        Returns
        -------
        int
            Valid frame index within bounds
        """
        if num_frames == 0:
            raise ValueError("No frames available in data")

        # Ensure frame_index is within valid range
        valid_frame = max(0, min(frame_index, num_frames - 1))

        # Log warning if frame was adjusted
        if valid_frame != frame_index:
            logger.warning(
                f"Frame index {frame_index} out of range for {num_frames} frames. "
                f"Adjusted to frame {valid_frame}."
            )

        return valid_frame

    # endregion

    # region === Vector Visualization
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

        # Flatten the coordinate arrays
        Y_flat = Y.flatten()
        X_flat = X.flatten()
        U_flat = U.flatten()
        V_flat = V.flatten()

        # Create vectors array in correct format (N, 2, 2)
        N = len(Y_flat)
        vectors = np.zeros((N, 2, 2))  # (N, 2, 2) for N vectors with start/end points in 2D

        # Start points
        vectors[:, 0, 1] = X_flat  # x coordinates
        vectors[:, 0, 0] = Y_flat  # y coordinates

        vectors[:, 1, 1] = U_flat
        vectors[:, 1, 0] = V_flat

        # Create colors based on magnitudes
        max_mag = d_max if d_max is not None else magnitudes.max()
        if max_mag > 0:
            normalized_magnitudes = np.clip(magnitudes.flatten() / max_mag, 0, 1)
            colors = plt.cm.get_cmap(colormap)(normalized_magnitudes)
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

    # endregion

    # region === Displacement Visualization
    def visualize_displacement_results(self) -> None:
        """Visualize displacement results for all frames using data from data manager."""
        try:
            # Check if displacement results exist
            if self.data_manager.displacement_results is None:
                raise ValueError("No displacement results available in data manager")

            # Get results from data manager
            results = self.data_manager.displacement_results
            flows = results.displacement_field
            params = results.parameters

            # Validate number of frames
            num_frames = len(flows)
            if num_frames == 0:
                raise ValueError("Displacement field contains no frames")

            downscale_factor = params.downscale_factor

            # Clear existing layers
            self._clear_layers(['Displacement Magnitude', 'Displacement Vectors'])

            # Create visualization stacks
            upscaled_shape = (
                flows[0].shape[0] * downscale_factor,
                flows[0].shape[1] * downscale_factor
            )
            magnitudes = np.zeros((num_frames, *upscaled_shape))

            # Get visualization parameters
            vis_params = {
                'd_max': params.d_max,
                'vector_stride': params.disp_vector_stride,
                'arrow_scale': params.disp_arrow_scale
            }

            # Create vector cache
            vector_cache = {
                'data': [],
                'colors': [],
                'parameters': vis_params.copy(),
                'original_resolution': flows[0].shape[:2],
                'num_frames': num_frames  # Add frame count to cache
            }

            # Process each frame
            for i in range(num_frames):
                display_flow = self._upscale_field(flows[i], downscale_factor)

                # Calculate magnitude
                magnitude = np.sqrt(np.sum(display_flow ** 2, axis=-1))
                magnitudes[i] = magnitude

                # Calculate vector data
                flow_scaled = display_flow * vis_params['arrow_scale'] / vis_params['d_max'] * 50
                vectors, colors = self._create_vector_visualization(
                    flow_scaled,
                    display_flow,
                    vis_params['vector_stride'],
                    vis_params['d_max']
                )
                vector_cache['data'].append(vectors)
                vector_cache['colors'].append(colors)

            # Store vector cache in data manager
            self.data_manager.displacement_vector_cache = vector_cache

            # Add visualization layers
            with self.viewer.events.blocker_all():
                self._layers['displacement_magnitude'] = self.viewer.add_image(
                    magnitudes,
                    name='Displacement Magnitude',
                    colormap='viridis',
                    blending='additive',
                    contrast_limits=(0, vis_params['d_max']),
                    visible=True
                )
                self.colorbar_manager.show_for_layer(
                    self._layers['displacement_magnitude'],
                    colormap_name='viridis',
                    label='Displacement (µm)'
                )

                # Get valid current frame index
                current_frame = self._validate_frame_index(
                    self.viewer.dims.current_step[0],
                    num_frames
                )

                # Add initial vector layer
                if len(vector_cache['data'][current_frame]) > 0:
                    self._layers['displacement_vectors'] = self.viewer.add_vectors(
                        vector_cache['data'][current_frame],
                        name='Displacement Vectors',
                        edge_color=vector_cache['colors'][current_frame],
                        edge_width=2,
                        vector_style='arrow',
                        blending='additive',
                        length=1
                    )

                # Update viewer dimensions to match data
                self.viewer.dims.set_point(0, current_frame)

        except Exception as e:
            error = self.create_error(
                message="Failed to visualize displacement results",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Try adjusting visualization parameters or check data consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)
            raise

    def update_displacement_frame(self, frame_index: int) -> None:
        """Update vector visualization for the current frame."""
        try:
            # Check if we have displacement results and vector cache
            if not hasattr(self.data_manager, 'displacement_vector_cache'):
                return

            cache = self.data_manager.displacement_vector_cache
            if cache is None or 'data' not in cache:
                return

            # Get number of frames from cache
            num_frames = cache.get('num_frames', len(cache['data']))

            # Validate frame index
            valid_frame = self._validate_frame_index(frame_index, num_frames)

            # Update vectors using stored layer reference
            if 'displacement_vectors' in self._layers and self._layers['displacement_vectors'] is not None:
                with self.viewer.events.blocker_all():
                    self._layers['displacement_vectors'].data = cache['data'][valid_frame]
                    self._layers['displacement_vectors'].edge_color = cache['colors'][valid_frame]

        except Exception as e:
            error = self.create_error(
                message="Failed to update displacement frame",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check displacement results and vector cache consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)

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
            # Clear vector cache first
            self.clear_disp_vector_cache()

            # Clear existing layers
            self._clear_layers(['Displacement Magnitude', 'Displacement Vectors'])

            # Upscale flow for visualization
            display_flow = self._upscale_field(flow, downscale_factor)

            # Scale flow for visualization
            flow_scaled = display_flow * arrow_scale / d_max * 50

            # Add visualization layers
            with self.viewer.events.blocker_all():
                # Add magnitude
                magnitude = np.sqrt(np.sum(display_flow ** 2, axis=-1))
                self._layers['displacement_magnitude'] = self.viewer.add_image(
                    magnitude,
                    name='Displacement Magnitude',
                    colormap='viridis',
                    blending='additive',
                    contrast_limits=(0, d_max),
                    visible=True
                )
                self.colorbar_manager.show_for_layer(
                    self._layers['displacement_magnitude'],
                    colormap_name='viridis',
                    label='Displacement (µm)'
                )

                # Create vector data and add layer
                vectors, colors = self._create_vector_visualization(
                    flow_scaled,
                    display_flow,
                    vector_stride,
                    d_max
                )

                if len(vectors) > 0:
                    self._layers['displacement_vectors'] = self.viewer.add_vectors(
                        vectors,
                        name='Displacement Vectors',
                        edge_color=colors,
                        edge_width=2,
                        vector_style='arrow',
                        blending='additive',
                        length=1
                    )

        except Exception as e:
            error = self.create_error(
                message="Failed to visualize displacement preview",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Try adjusting visualization parameters or check input data",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)
            raise

    # endregion

    # region === Force Visualization
    def visualize_force_results(self) -> None:
        """Visualize force results for all frames using data from data manager."""
        try:
            # Check if force results exist
            if self.data_manager.force_results is None:
                raise ValueError("No force results available in data manager")

            # Get results from data manager
            results = self.data_manager.force_results
            force_fields = results.force_field
            params = results.parameters

            # Validate number of frames
            num_frames = len(force_fields)
            if num_frames == 0:
                raise ValueError("Force field contains no frames")

            downscale_factor = params.downscale_factor

            # Clear existing layers
            self._clear_layers(['Force Magnitude', 'Force Vectors'])

            # Create visualization stacks
            upscaled_shape = (
                force_fields[0].shape[0] * downscale_factor,
                force_fields[0].shape[1] * downscale_factor
            )
            magnitudes = np.zeros((num_frames, *upscaled_shape))

            # Get visualization parameters
            vis_params = {
                'f_max': params.f_max,
                'vector_stride': params.force_vector_stride,
                'arrow_scale': params.force_arrow_scale
            }

            # Create vector cache
            vector_cache = {
                'data': [],
                'colors': [],
                'parameters': vis_params.copy(),
                'original_resolution': force_fields[0].shape[:2],
                'num_frames': num_frames  # Add frame count to cache
            }

            # Process each frame
            for i in range(num_frames):
                display_force = self._upscale_field(force_fields[i], downscale_factor)

                # Calculate magnitude
                magnitude = np.sqrt(np.sum(display_force ** 2, axis=-1))
                magnitudes[i] = magnitude

                # Calculate vector data
                force_scaled = display_force * vis_params['arrow_scale'] / vis_params['f_max'] * 50
                vectors, colors = self._create_vector_visualization(
                    force_scaled,
                    display_force,
                    vis_params['vector_stride'],
                    vis_params['f_max'],
                    colormap='inferno'
                )
                vector_cache['data'].append(vectors)
                vector_cache['colors'].append(colors)

            # Store vector cache in data manager
            self.data_manager.force_vector_cache = vector_cache

            # Add visualization layers
            with self.viewer.events.blocker_all():
                self._layers['force_magnitude'] = self.viewer.add_image(
                    magnitudes,
                    name='Force Magnitude',
                    colormap='inferno',
                    blending='additive',
                    contrast_limits=(0, vis_params['f_max']),
                    visible=True
                )
                self.colorbar_manager.show_for_layer(
                    self._layers['force_magnitude'],
                    colormap_name='inferno',
                    label='Force (Pa)'
                )

                # Get valid current frame index
                current_frame = self._validate_frame_index(
                    self.viewer.dims.current_step[0],
                    num_frames
                )

                # Add initial vector layer
                if len(vector_cache['data'][current_frame]) > 0:
                    self._layers['force_vectors'] = self.viewer.add_vectors(
                        vector_cache['data'][current_frame],
                        name='Force Vectors',
                        edge_color=vector_cache['colors'][current_frame],
                        edge_width=2,
                        vector_style='arrow',
                        blending='additive',
                        length=1
                    )

                # Update viewer dimensions to match data
                self.viewer.dims.set_point(0, current_frame)

        except Exception as e:
            error = self.create_error(
                message="Failed to visualize force results",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Try adjusting visualization parameters or check data consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)
            raise

    def update_force_frame(self, frame_index: int) -> None:
        """Update force vector visualization for the current frame."""
        try:
            # Check if we have force results and vector cache
            if not hasattr(self.data_manager, 'force_vector_cache'):
                return

            cache = self.data_manager.force_vector_cache
            if cache is None or 'data' not in cache:
                return

            # Get number of frames from cache
            num_frames = cache.get('num_frames', len(cache['data']))

            # Validate frame index
            valid_frame = self._validate_frame_index(frame_index, num_frames)

            # Update vectors using stored layer reference
            if 'force_vectors' in self._layers and self._layers['force_vectors'] is not None:
                with self.viewer.events.blocker_all():
                    self._layers['force_vectors'].data = cache['data'][valid_frame]
                    self._layers['force_vectors'].edge_color = cache['colors'][valid_frame]

        except Exception as e:
            error = self.create_error(
                message="Failed to update force frame",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check force results and vector cache consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)

    def visualize_force_preview(
            self,
            force_field: np.ndarray,
            f_max: float,
            vector_stride: int,
            arrow_scale: float,
            downscale_factor: int = 1
    ) -> None:
        """Visualize force preview for a single frame."""
        try:
            # Clear vector cache first
            self.clear_force_vector_cache()

            # Clear existing layers
            self._clear_layers(['Force Magnitude', 'Force Vectors'])
            display_force = self._upscale_field(force_field, downscale_factor)

            # Add visualization layers
            with self.viewer.events.blocker_all():
                # Add magnitude
                magnitude = np.sqrt(np.sum(display_force ** 2, axis=-1))
                magnitude = np.clip(magnitude, 0, f_max)

                self._layers['force_magnitude'] = self.viewer.add_image(
                    magnitude,
                    name='Force Magnitude',
                    colormap='inferno',
                    blending='additive',
                    contrast_limits=(0, f_max)
                )
                self.colorbar_manager.show_for_layer(
                    self._layers['force_magnitude'],
                    colormap_name='inferno',
                    label='Force (Pa)'
                )

                # Create vector data and add layer
                force_scaled = display_force * arrow_scale / f_max * 50
                vectors, colors = self._create_vector_visualization(
                    force_scaled,
                    display_force,
                    vector_stride,
                    f_max,
                    colormap='inferno'
                )

                if len(vectors) > 0:
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
            error = self.create_error(
                message="Failed to visualize force preview",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Try adjusting visualization parameters or check input data",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)
            raise

    # endregion

    # region === Stress Visualization
    def visualize_stress_results(self) -> None:
        """Visualize stress results for all frames using data from data manager."""
        try:
            # Check if stress results exist
            if self.data_manager.stress_results is None:
                raise ValueError("No stress results available in data manager")

            # Get results and parameters from data manager
            results = self.data_manager.stress_results
            stress_tensors = results.stress_tensor
            params = results.parameters

            # Validate number of frames
            num_frames = len(stress_tensors)
            if num_frames == 0:
                raise ValueError("Stress tensor contains no frames")

            # Get visualization parameters
            max_stress = params.max_stress
            downscale_factor = params.downscale_factor

            # Clear existing layers
            self._clear_layers([
                'Normal Stress XX',
                'Normal Stress YY',
                'Average Normal Stress'
            ])

            def upscale_component(component):
                if downscale_factor > 1:
                    if component.ndim == 3:  # Multiple frames
                        return np.stack([
                            cv2.resize(
                                frame,
                                (frame.shape[1] * downscale_factor,
                                 frame.shape[0] * downscale_factor),
                                interpolation=cv2.INTER_LINEAR
                            )
                            for frame in component
                        ])
                    else:  # Single frame
                        return cv2.resize(
                            component,
                            (component.shape[1] * downscale_factor,
                             component.shape[0] * downscale_factor),
                            interpolation=cv2.INTER_LINEAR
                        )
                return component

            # Extract and upscale stress components
            sigma_xx = upscale_component(stress_tensors[..., 0, 0])
            sigma_yy = upscale_component(stress_tensors[..., 1, 1])

            # Calculate average normal stress after upscaling
            sigma_normal = (sigma_xx + sigma_yy) / 2

            # Store stress data in manager for frame updates
            self.data_manager.stress_components = {
                'sigma_xx': sigma_xx,
                'sigma_yy': sigma_yy,
                'sigma_normal': sigma_normal,
                'num_frames': num_frames,
                'max_stress': max_stress
            }

            # Add visualization layers
            with self.viewer.events.blocker_all():
                # Get valid current frame index
                current_frame = self._validate_frame_index(
                    self.viewer.dims.current_step[0],
                    num_frames
                )

                # Normal stress XX
                self._layers['stress_xx'] = self.viewer.add_image(
                    sigma_xx,
                    name='Normal Stress XX',
                    colormap='seismic',
                    blending='additive',
                    contrast_limits=(-max_stress, max_stress)
                )

                # Normal stress YY
                self._layers['stress_yy'] = self.viewer.add_image(
                    sigma_yy,
                    name='Normal Stress YY',
                    colormap='seismic',
                    blending='additive',
                    contrast_limits=(-max_stress, max_stress)
                )

                # Average normal stress
                self._layers['stress_normal'] = self.viewer.add_image(
                    sigma_normal,
                    name='Average Normal Stress',
                    colormap='seismic',
                    blending='additive',
                    contrast_limits=(-max_stress, max_stress)
                )
                self.colorbar_manager.show_for_layer(
                    self._layers['stress_normal'],
                    colormap_name='seismic',
                    label='Stress (mN/m)'
                )

                # Update viewer dimensions to match data
                self.viewer.dims.set_point(0, current_frame)

        except Exception as e:
            error = self.create_error(
                message="Failed to visualize stress results",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Try adjusting visualization parameters or check data consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)
            raise

    def update_stress_frame(self, frame_index: int) -> None:
        """Update stress tensor visualization for the current frame."""
        try:
            # Check if stress components exist
            if not hasattr(self.data_manager, 'stress_components'):
                return

            components = self.data_manager.stress_components
            if components is None:
                return

            # Get number of frames and validate frame index
            num_frames = components.get('num_frames', 0)
            if num_frames == 0:
                return

            # Validate frame index
            valid_frame = self._validate_frame_index(frame_index, num_frames)

            # Update stress layers if they exist
            if 'stress_xx' in self._layers and self._layers['stress_xx'] is not None:
                with self.viewer.events.blocker_all():
                    # Update normal stress components
                    self._layers['stress_xx'].data = components['sigma_xx'][valid_frame]
                    self._layers['stress_yy'].data = components['sigma_yy'][valid_frame]
                    self._layers['stress_normal'].data = components['sigma_normal'][valid_frame]

        except Exception as e:
            error = self.create_error(
                message="Failed to update stress frame",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check stress results and layer consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)

    def visualize_stress_preview(
            self,
            stress_tensor: np.ndarray,
            max_stress: float,
            downscale_factor: int = 1
    ) -> None:
        """Visualize stress tensor components for a single frame."""
        try:
            # Clear existing layers
            self._clear_layers([
                'Normal Stress XX',
                'Normal Stress YY',
                'Average Normal Stress'
            ])

            # Function to upscale stress components
            def upscale_component(component):
                if downscale_factor > 1:
                    return cv2.resize(
                        component,
                        (component.shape[1] * downscale_factor,
                         component.shape[0] * downscale_factor),
                        interpolation=cv2.INTER_LINEAR
                    )
                return component

            # Extract and upscale stress components
            sigma_xx = upscale_component(np.squeeze(stress_tensor[..., 0, 0]))
            sigma_yy = upscale_component(np.squeeze(stress_tensor[..., 1, 1]))

            # Calculate average normal stress after upscaling
            sigma_normal = (sigma_xx + sigma_yy) / 2

            # Add visualization layers
            with self.viewer.events.blocker_all():
                # Normal stress XX
                self._layers['stress_xx'] = self.viewer.add_image(
                    sigma_xx,
                    name='Normal Stress XX',
                    colormap='seismic',
                    blending='additive',
                    contrast_limits=(-max_stress, max_stress)
                )

                # Normal stress YY
                self._layers['stress_yy'] = self.viewer.add_image(
                    sigma_yy,
                    name='Normal Stress YY',
                    colormap='seismic',
                    blending='additive',
                    contrast_limits=(-max_stress, max_stress)
                )

                # Average normal stress
                self._layers['stress_normal'] = self.viewer.add_image(
                    sigma_normal,
                    name='Average Normal Stress',
                    colormap='seismic',
                    blending='additive',
                    contrast_limits=(-max_stress, max_stress)
                )
                self.colorbar_manager.show_for_layer(
                    self._layers['stress_normal'],
                    colormap_name='seismic',
                    label='Stress (mN/m)'
                )

        except Exception as e:
            error = self.create_error(
                message="Failed to visualize stress preview",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Try adjusting visualization parameters or check input data",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)
            raise

    # endregion

    # region === Preprocessing Visualization
    def update_preprocessing_visualization(self) -> None:
        """Update visualization after preprocessing."""
        try:
            self.colorbar_manager.clear()

            for layer_name in [
                'Preprocessed Beads',
                'Preprocessed Reference',
                'Preprocessed Cells',
                'Bead Overlay',
            ]:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)

            if self.data_manager.preprocessed_cell_stack is not None:
                self.viewer.add_image(
                    self.data_manager.preprocessed_cell_stack,
                    name='Preprocessed Cells',
                    colormap='gray',
                    blending='additive',
                    visible=True
                )

            if self.data_manager.preprocessed_reference is not None:
                self.viewer.add_image(
                    self.data_manager.preprocessed_reference,
                    name='Preprocessed Reference',
                    colormap='magenta',
                    blending='additive',
                    visible=True
                )

            if self.data_manager.preprocessed_bead_stack is not None:
                self.viewer.add_image(
                    self.data_manager.preprocessed_bead_stack,
                    name='Preprocessed Beads',
                    colormap='green',
                    blending='additive',
                    visible=True
                )

        except Exception as e:
            error = self.create_error(
                message="Failed to update preprocessing visualization",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check data availability and consistency",
                original_error=e,
                source="visualization"
            )
            self.handle_error(error)

    def handle_preprocessing_preview(self, frames: Dict[str, np.ndarray], enable: bool = True) -> None:
        """Render preprocessing preview inputs as separate additive layers."""
        layer_specs = {
            'cells': ('Preview Cells', 'gray'),
            'reference': ('Preview Reference', 'magenta'),
            'beads': ('Preview Beads', 'green'),
        }

        try:
            for data_type, (layer_name, colormap) in layer_specs.items():
                if not enable or data_type not in frames:
                    if layer_name in self.viewer.layers:
                        self.viewer.layers.remove(layer_name)
                    continue

                frame = frames[data_type]
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    layer.data = frame
                    layer.visible = True
                    layer.colormap = colormap
                    layer.blending = 'additive'
                else:
                    self.viewer.add_image(
                        frame,
                        name=layer_name,
                        colormap=colormap,
                        blending='additive',
                        visible=True,
                    )

            self._preview_config.enabled = enable

        except Exception as e:
            logger.error(f"Preprocessing preview handling failed: {str(e)}")
            for layer_name, _ in layer_specs.values():
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
            self._preview_config.enabled = False
            raise

    def handle_preview(self, frame: Optional[np.ndarray], enable: bool = True, layer_name: str = 'Preview') -> None:
        """Handle preview visualization"""
        try:
            # First check if preview layer still exists in viewer
            preview_exists = False
            if self._preview_config.preview_layer is not None:
                for layer in self.viewer.layers:
                    if layer == self._preview_config.preview_layer:
                        preview_exists = True
                        break

            # Clear invalid reference if layer doesn't exist
            if not preview_exists:
                self._preview_config.preview_layer = None

            # Handle layer based on enable state
            if enable:
                if self._preview_config.preview_layer is None:
                    # Create new layer
                    self._preview_config.preview_layer = self.viewer.add_image(
                        frame,
                        name=layer_name,
                        visible=True
                    )
                else:
                    # Update existing layer
                    self._preview_config.preview_layer.data = frame
            else:
                # Try to remove layer if it exists and is valid
                if preview_exists:
                    self.viewer.layers.remove(self._preview_config.preview_layer)
                self._preview_config.preview_layer = None

            # Update preview config state
            self._preview_config.enabled = enable

        except Exception as e:
            logger.error(f"Preview handling failed: {str(e)}")
            # Make sure to clean up reference on error
            self._preview_config.preview_layer = None
            self._preview_config.enabled = False
            raise

    # endregion

    # region === Mesh Visualization
    def visualize_mesh(self, nodes: np.ndarray, elements: np.ndarray,
                       downscale_factor: float = 1.0, layer_prefix: str = ''):
        """
        Visualize mesh nodes and edges in napari viewer.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates (N x 2)
        elements : np.ndarray
            Element connectivity (M x 3)
        downscale_factor : float
            Factor to scale coordinates by
        layer_prefix : str
            Prefix for layer names (for multiple visualizations)
        """
        # Remove existing mesh layers
        edge_layer_name = f"{layer_prefix}Mesh Edges"
        node_layer_name = f"{layer_prefix}Mesh Nodes"

        for layer_name in [edge_layer_name, node_layer_name]:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        # Scale node coordinates
        nodes_scaled = nodes * downscale_factor

        # Swap x and y coordinates for display
        nodes_display = np.column_stack((nodes_scaled[:, 1], nodes_scaled[:, 0]))

        # Create edge data for visualization
        num_elements = len(elements)
        edge_data = np.zeros((num_elements * 3, 2, 2))

        for i, element in enumerate(elements):
            # Get node coordinates for triangle vertices
            v1 = nodes_display[element[0]]
            v2 = nodes_display[element[1]]
            v3 = nodes_display[element[2]]

            # Add three edges (v1-v2, v2-v3, v3-v1)
            edge_data[i * 3] = np.array([v1, v2])
            edge_data[i * 3 + 1] = np.array([v2, v3])
            edge_data[i * 3 + 2] = np.array([v3, v1])

        # Add visualization layers
        self.viewer.add_shapes(
            edge_data,
            shape_type='line',
            edge_color='yellow',
            edge_width=1,
            opacity=0.6,
            name=edge_layer_name
        )

        self.viewer.add_points(
            nodes_display,
            size=4,
            face_color='red',
            opacity=0.7,
            name=node_layer_name
        )

    def visualize_masks(self, masks: np.ndarray, downscale_factor: int = 1, name: str = 'Masks', opacity: float = 0.5):
        """
        Visualize masks with proper scaling.

        Parameters
        ----------
        masks : np.ndarray
            Binary mask array to visualize
        downscale_factor : int
            Factor by which to upscale the masks for visualization
        name : str
            Name of the layer in napari viewer
        opacity : float
            Opacity of the mask layer (0-1)
        """
        # Remove existing mask layer if it exists
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)

        # Upscale masks if needed
        if downscale_factor > 1:
            upscaled_masks = np.repeat(
                np.repeat(masks, downscale_factor, axis=-2),
                downscale_factor, axis=-1
            )
        else:
            upscaled_masks = masks

        # Add the mask layer
        self.viewer.add_labels(
            upscaled_masks.astype(np.uint8),
            name=name,
            visible=True,
            opacity=opacity
        )

    # endregion
