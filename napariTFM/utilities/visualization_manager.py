import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import cv2
import napari
import numpy as np

from napariTFM.utilities.error_handling import ErrorSeverity, ErrorHandlingMixin
from napariTFM.utilities.vector_field import build_frame_vectors, upscale_field
from napariTFM.utilities.viewer_colorbar import ViewerColorbarManager

logger = logging.getLogger(__name__)



class VisualizationManager(ErrorHandlingMixin):
    # region === Initialization
    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager"):
        super().__init__()
        self.viewer = viewer
        self.data_manager = data_manager
        self._layers: Dict[str, Any] = {}
        self.colorbar_manager = ViewerColorbarManager(viewer)

        # The original input (bead-image) ``(H, W)`` that every analysis-grid field
        # is displayed at. Set per selection from the raw input on disk; when known,
        # fields resize to *exactly* this instead of ``grid × downscale_factor``. The
        # latter silently oversizes whenever the display's downscale dial disagrees
        # with the grid the data was actually computed on (e.g. previewing force with
        # downscale=4 over a displacement loaded from disk at downscale=2). None ⇒
        # fall back to the downscale-factor behaviour (ad-hoc data, no selection).
        self._display_reference_xy: Optional[tuple] = None

        # Connect to viewer events
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.colorbar_manager.clear()

            # Disconnect other events
            if self.viewer is not None:
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

    def isolate_layers(self, keep_names) -> None:
        """Show only ``keep_names``; hide every other layer in the viewer.

        A stage preview calls this when it starts so its output is inspected on
        its own rather than blended with the previous stage's layers. Re-renders
        (e.g. a live preview reacting to a parameter change) must not call this:
        once isolated, the user's own visibility tweaks are theirs to keep.

        The active colorbar legend always rides along with the kept layers, so a
        preview shows the same scale legend the committed/result view does.
        """
        keep = set(keep_names) | set(self.colorbar_manager.layer_names)
        for layer in self.viewer.layers:
            layer.visible = layer.name in keep

    def hide_other_layers(self, keep_names) -> None:
        """Hide every layer *not* in ``keep_names``; leave the kept ones alone.

        The gentler sibling of :meth:`isolate_layers`. A streamed stage run uses
        this to take the viewer over — hiding the previous stage and the raw
        inputs so its output isn't blended (worklist §4) — without forcing its
        *own* layers visible. That preserves the per-layer visibility the user
        dialled in across a re-run (e.g. a magnitude layer they hid to inspect
        the vectors alone stays hidden), which a full isolate would clobber.

        The active colorbar legend always rides along with the kept layers.
        """
        keep = set(keep_names) | set(self.colorbar_manager.layer_names)
        for layer in self.viewer.layers:
            if layer.name not in keep:
                layer.visible = False

    def bring_layers_to_front(self, layers) -> None:
        """Stack the named preview layers on top, hiding everything else.

        ``layers`` is an ordered list of ``(name, visible)`` from bottom-most to
        top-most among the front slots. Each named layer is set to its given
        visibility and moved into the top ``len(layers)`` positions in that
        order; the active colorbar legend stays visible; every other layer is
        hidden. This is the shared body of the displacement / force / stress
        preview layer-management (which differ only in the names and per-layer
        visibility) — previously copy-pasted three times.
        """
        spec = list(layers)
        visible_by_name = dict(spec)
        found = {}
        for layer in self.viewer.layers:
            if layer.name in visible_by_name:
                layer.visible = visible_by_name[layer.name]
                found[layer.name] = layer
            elif self.colorbar_manager.is_colorbar_layer(layer.name):
                layer.visible = True
            else:
                layer.visible = False

        n = len(spec)
        for i, (name, _visible) in enumerate(spec):
            layer = found.get(name)
            if layer is None:
                continue
            current = self.viewer.layers.index(layer)
            target = -(n - i)  # top-most slot is -1, next -2, ...
            if current != target:
                self.viewer.layers.move(current, target)

    # Canonical bottom-to-top z-order for the raw-input + mask layers. The inputs
    # and mask stream in asynchronously (each on its own worker), so napari's
    # arrival-order stacking is non-deterministic — the mask, in particular,
    # races the cells layer and visibly jumps above it when its read lands late.
    # Pinning the order and re-asserting it on every add makes the final stack
    # deterministic (Masks on top, then Cells, Reference, Beads) with no flip.
    _INPUT_LAYER_ORDER = ("Beads", "Reference", "Cells", "Masks")

    def order_input_layers(self) -> None:
        """Force the raw-input + mask layers into :data:`_INPUT_LAYER_ORDER`.

        Idempotent: moves each present input/mask layer into the lowest slots in
        canonical order (so they sit as a block beneath any result layers), skip-
        ping any that are already in place. Called after each input/mask layer is
        added so no intermediate state ever shows the layers out of order.
        """
        layers = getattr(self.viewer, "layers", None)
        move = getattr(layers, "move", None)
        if layers is None or not callable(move):
            return
        present = [name for name in self._INPUT_LAYER_ORDER if name in layers]
        for target, name in enumerate(present):
            try:
                current = layers.index(layers[name])
            except (KeyError, ValueError):
                continue
            if current != target:
                move(current, target)

    def capture_layer_visibility(self) -> dict:
        """Snapshot ``{layer_name: visible}`` for every layer in the viewer.

        Paired with :meth:`restore_layer_visibility` to make the streaming
        sink's per-stage :meth:`isolate_layers` takeover reversible (worklist
        §4): the shell snapshots before a run-all and restores after.
        """
        return {layer.name: layer.visible for layer in self.viewer.layers}

    def restore_layer_visibility(self, snapshot: dict) -> None:
        """Restore the visibility recorded by :meth:`capture_layer_visibility`.

        Only layers present in *snapshot* are touched, so a layer the run
        created (e.g. the final stage's result) keeps whatever visibility the
        last :meth:`isolate_layers` left it with — the user's original layers
        come back exactly as they were, and the run's last stage stays shown.
        """
        for layer in self.viewer.layers:
            if layer.name in snapshot:
                layer.visible = snapshot[layer.name]

    def set_display_reference_shape(self, shape) -> None:
        """Set the original input ``(H, W)`` that analysis-grid fields display at.

        Call on selection with the raw bead-image xy size; pass ``None`` to clear
        (revert to the ``grid × downscale_factor`` behaviour). See
        ``_display_reference_xy``.
        """
        self._display_reference_xy = None if shape is None else tuple(int(s) for s in shape[-2:])

    def _to_display_resolution(self, arr: np.ndarray, downscale_factor: int) -> np.ndarray:
        """Resize an analysis-grid array to the display resolution.

        When a display reference shape is set, resize to *exactly* that (the true
        original input size), which is robust to a stale ``downscale_factor``.
        Otherwise fall back to ``grid × downscale_factor``. Handles both a 2D image
        and an ``(H, W, C)`` field (cv2 resizes per channel). A no-op when the array
        is already at the target size.
        """
        ref = getattr(self, "_display_reference_xy", None)
        if ref is not None:
            h, w = ref
            if arr.shape[0] == h and arr.shape[1] == w:
                return arr
            return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return upscale_field(arr, downscale_factor)

    def _upscale_field(self, field: np.ndarray, downscale_factor: int) -> np.ndarray:
        """Upscale a vector field to display resolution (shared with headless export)."""
        return self._to_display_resolution(field, downscale_factor)

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
        return build_frame_vectors(flow_scaled, original_flow, stride, d_max, colormap)

    # region === Displacement Visualization
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

            # Lazy display: a frame the circle-click load didn't build yet is
            # rendered on demand now (which also sets the layer), so scrubbing
            # into unseen frames fills them one at a time instead of up front.
            if cache['data'][valid_frame] is None:
                self._ensure_vector_frame('displacement', cache, valid_frame)
                return

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

            self.isolate_layers(['Displacement Magnitude', 'Displacement Vectors'])

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

            # Lazy display: build a not-yet-rendered frame on demand (see
            # update_displacement_frame).
            if cache['data'][valid_frame] is None:
                self._ensure_vector_frame('force', cache, valid_frame)
                return

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

            self.isolate_layers(['Force Magnitude', 'Force Vectors'])

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

            # Upscale stress components to the display resolution (see
            # _to_display_resolution: the original input size when known).
            def upscale_component(component):
                return self._to_display_resolution(component, downscale_factor)

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

            self.isolate_layers([
                'Normal Stress XX',
                'Normal Stress YY',
                'Average Normal Stress',
            ])

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

    # region === Image-layer streaming helpers (shared by all streamed stages)
    def _rebind_image_layer(self, layer, data: np.ndarray):
        """Swap an image layer's backing array without disturbing its settings.

        Setting ``.data`` to a freshly zeroed array can collapse the contrast
        range, so the prior contrast limits (and their range) are captured and
        restored — that is what "preserve the user's settings across runs" means.
        Used by every streamed step that re-runs into an existing image layer.

        Returns the layer to keep using. When the new array's dimensionality
        differs from the layer's current one (e.g. a 2D preview layer being
        rebound to a 3D stream stack), the layer is *recreated* rather than
        swapped in place: napari never refreshes a vispy layer's cached
        ``_world_to_layer_units_scale`` on an ndim change, so assigning ``.data``
        with a different ndim leaves a stale, too-short scale tuple and the next
        slice raises ``IndexError`` in vispy. A fresh layer gets a correctly
        sized scale tuple, side-stepping that napari bug.
        """
        clim = getattr(layer, 'contrast_limits', None)
        clim_range = getattr(layer, 'contrast_limits_range', None)

        if np.ndim(data) != getattr(layer, 'ndim', np.ndim(data)):
            return self._recreate_image_layer(layer, data, clim, clim_range)

        with self.viewer.events.blocker_all():
            layer.data = data
            if clim_range is not None:
                try:
                    layer.contrast_limits_range = clim_range
                except Exception:
                    pass
            if clim is not None:
                try:
                    layer.contrast_limits = clim
                except Exception:
                    pass
        return layer

    def _recreate_image_layer(self, layer, data: np.ndarray, clim, clim_range):
        """Replace *layer* with a fresh image layer backed by *data*.

        Preserves the visible settings that survive a re-run (name, colormap,
        blending, contrast, visibility, opacity, gamma, scale, translate) and
        restores the layer's position in the stack. Used when the data's ndim
        changes (see :meth:`_rebind_image_layer`).
        """
        name = layer.name
        kwargs = {'name': name}
        for attr in ('colormap', 'blending', 'visible', 'opacity', 'gamma',
                     'scale', 'translate'):
            value = getattr(layer, attr, None)
            if value is not None:
                kwargs[attr] = value
        if clim is not None:
            kwargs['contrast_limits'] = clim

        with self.viewer.events.blocker_all():
            try:
                index = self.viewer.layers.index(layer)
            except (ValueError, KeyError):
                index = None
            self.viewer.layers.remove(layer)
            new_layer = self.viewer.add_image(data, **kwargs)
            if index is not None:
                try:
                    self.viewer.layers.move(self.viewer.layers.index(new_layer), index)
                except (AttributeError, ValueError, KeyError):
                    pass
            if clim_range is not None:
                try:
                    new_layer.contrast_limits_range = clim_range
                except Exception:
                    pass
            if clim is not None:
                try:
                    new_layer.contrast_limits = clim
                except Exception:
                    pass
        return new_layer

    def _advance_to_frame(self, frame_index: int) -> None:
        """Move the time slider to *frame_index* on axis 0 (best-effort)."""
        try:
            self.viewer.dims.set_current_step(0, frame_index)
        except Exception:
            pass

    # --- vector-field streaming (displacement / force) --------------------
    _VECTOR_FIELD_CONFIG = {
        'displacement': {
            'magnitude_layer': 'Displacement Magnitude',
            'vectors_layer': 'Displacement Vectors',
            'magnitude_key': 'displacement_magnitude',
            'vectors_key': 'displacement_vectors',
            'cache_attr': 'displacement_vector_cache',
            'magnitude_colormap': 'viridis',
            'vector_colormap': 'viridis',
            'colorbar_label': 'Displacement (µm)',
        },
        'force': {
            'magnitude_layer': 'Force Magnitude',
            'vectors_layer': 'Force Vectors',
            'magnitude_key': 'force_magnitude',
            'vectors_key': 'force_vectors',
            'cache_attr': 'force_vector_cache',
            'magnitude_colormap': 'inferno',
            'vector_colormap': 'inferno',
            'colorbar_label': 'Force (Pa)',
        },
    }

    def begin_vector_field_stream(self, kind: str, num_frames: int, vis_params: dict) -> None:
        """Prepare the magnitude + vectors layers so *kind* can stream frame by frame.

        The magnitude image stack and the per-frame vector cache are allocated
        lazily on the first frame (when the upscaled resolution is known); here
        we just reset the cache and drop any stale vectors layer, remembering
        whether the user had hidden it. The magnitude image layer is reused on a
        re-run so its contrast/visibility survive (see ``_rebind_image_layer``).

        ``vis_params`` carries ``v_max`` (d_max/f_max), ``vector_stride``,
        ``arrow_scale`` and ``downscale_factor``.
        """
        try:
            cfg = self._VECTOR_FIELD_CONFIG[kind]

            vec_name = cfg['vectors_layer']
            vec_visible = True
            if vec_name in self.viewer.layers:
                vec_visible = getattr(self.viewer.layers[vec_name], 'visible', True)
                self.viewer.layers.remove(vec_name)
            self._layers[cfg['vectors_key']] = None

            cache = {
                'data': [None] * num_frames,
                'colors': [None] * num_frames,
                'parameters': dict(vis_params),
                'original_resolution': None,
                'num_frames': num_frames,
                'vectors_visible': vec_visible,
                'magnitude_ready': False,
                # Set by display_vector_field for the lazy circle-click path, so
                # scrubbing can build unbuilt frames on demand. None for the live
                # stream, which fills every frame as it computes.
                'source_field': None,
            }
            setattr(self.data_manager, cfg['cache_attr'], cache)

            # Take the viewer over for this stage (worklist §4): hide every other
            # layer so a streamed run shows only its own output. The
            # magnitude/vectors layers are created lazily on the first frame and
            # come up visible as they stream in; hiding (not isolating) preserves
            # any per-layer visibility the user kept across a re-run.
            self.hide_other_layers([cfg['magnitude_layer'], cfg['vectors_layer']])

        except Exception as e:
            error = self.create_error(
                message="Failed to begin vector-field stream",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check data availability and consistency",
                original_error=e,
                source="visualization",
            )
            self.handle_error(error)

    def stream_vector_field_frame(self, kind: str, frame_index: int, field_frame: np.ndarray) -> None:
        """Render one freshly computed displacement/force frame and follow it live.

        The frame is upscaled, reduced to a magnitude written in place into the
        pre-allocated stack, turned into a vector overlay (cached for later
        scrubbing), and the slider auto-advances to it. Used by the live run /
        preview streaming, which fills every frame as it computes.
        """
        try:
            cfg = self._VECTOR_FIELD_CONFIG[kind]
            cache = getattr(self.data_manager, cfg['cache_attr'], None)
            if cache is None:
                return
            self._render_vector_frame(cfg, cache, field_frame, frame_index)
            self._advance_to_frame(frame_index)

        except Exception as e:
            error = self.create_error(
                message="Failed to stream vector-field frame",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check layer/array consistency",
                original_error=e,
                source="visualization",
            )
            self.handle_error(error)

    def _render_vector_frame(self, cfg, cache, field_frame: np.ndarray, frame_index: int) -> None:
        """Build and show one vector-field frame: magnitude slice + vector overlay.

        The per-frame work shared by the live stream (which advances the slider
        after) and the lazy circle-click display (which builds only the current
        frame up front and lets this fill the rest on demand as the user
        scrubs). Does not touch the slider itself.
        """
        vis = cache['parameters']
        downscale = vis.get('downscale_factor', 1)
        vmax = vis['v_max']

        display_field = self._upscale_field(field_frame, downscale)
        magnitude = np.sqrt(np.sum(display_field ** 2, axis=-1))

        if not cache['magnitude_ready']:
            self._allocate_vector_field_magnitude(cfg, cache, magnitude.shape, vmax)
            cache['original_resolution'] = field_frame.shape[:2]
            cache['magnitude_ready'] = True

        mag_layer = self._layers.get(cfg['magnitude_key'])
        if mag_layer is None:
            return
        magnitudes = mag_layer.data
        if frame_index < 0 or frame_index >= magnitudes.shape[0]:
            return
        magnitudes[frame_index] = magnitude

        field_scaled = display_field * vis['arrow_scale'] / vmax * 50
        vectors, colors = self._create_vector_visualization(
            field_scaled, display_field, vis['vector_stride'], vmax,
            colormap=cfg['vector_colormap'],
        )
        cache['data'][frame_index] = vectors
        cache['colors'][frame_index] = colors

        with self.viewer.events.blocker_all():
            self._set_streamed_vectors(cfg, vectors, colors, cache['vectors_visible'])
        mag_layer.refresh()

    def display_vector_field(self, kind: str, field: np.ndarray, vis_params: dict) -> None:
        """Show a persisted vector field *lazily* — the circle-click display path.

        Unlike the live stream (which fills every frame as it computes), this
        builds only the frame currently under the slider and remembers the
        source ``field`` on the cache, so each other frame is rendered on demand
        the first time the user scrubs to it (see ``update_*_frame``). That keeps
        a click on a stage with many frames near-instant instead of upscaling and
        glyphing the whole stack up front.
        """
        try:
            num_frames = int(field.shape[0])
            self.begin_vector_field_stream(kind, num_frames, vis_params)
            cfg = self._VECTOR_FIELD_CONFIG[kind]
            cache = getattr(self.data_manager, cfg['cache_attr'], None)
            if cache is None:
                return
            # Retain the source so scrubbing can build unbuilt frames on demand.
            cache['source_field'] = field
            current = self._current_frame_index(num_frames)
            self._render_vector_frame(cfg, cache, field[current], current)
            self._advance_to_frame(current)
        except Exception as e:
            error = self.create_error(
                message="Failed to display vector field",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check layer/array consistency",
                original_error=e,
                source="visualization",
            )
            self.handle_error(error)

    def _current_frame_index(self, num_frames: int) -> int:
        """The slider's current frame, clamped to ``[0, num_frames - 1]`` (0 if unknown)."""
        try:
            step = int(self.viewer.dims.current_step[0])
        except Exception:
            step = 0
        return max(0, min(step, num_frames - 1))

    def _ensure_vector_frame(self, kind: str, cache, frame_index: int) -> bool:
        """Build ``frame_index`` from the cached source field if not yet built.

        Returns ``True`` when the frame's vectors are now available in the cache
        (either already built, or built just now). The lazy-display counterpart
        of the live stream having filled every frame up front.
        """
        if cache['data'][frame_index] is not None:
            return True
        source = cache.get('source_field')
        if source is None or frame_index >= source.shape[0]:
            return False
        cfg = self._VECTOR_FIELD_CONFIG[kind]
        # Only build for a stage the user is actually looking at: if another
        # stage has isolated the viewer (this stage's magnitude layer hidden),
        # scrubbing must not spend CPU rendering frames nobody can see. A
        # re-click rebuilds from scratch, so nothing is lost.
        mag_layer = self._layers.get(cfg['magnitude_key'])
        if mag_layer is None or not getattr(mag_layer, 'visible', True):
            return False
        self._render_vector_frame(cfg, cache, source[frame_index], frame_index)
        return cache['data'][frame_index] is not None

    def _allocate_vector_field_magnitude(self, cfg, cache, upscaled_shape, vmax) -> None:
        """Allocate the magnitude stack and (re)bind its image layer + colorbar."""
        magnitudes = np.zeros((cache['num_frames'], *upscaled_shape), dtype=np.float32)
        mag_name = cfg['magnitude_layer']
        with self.viewer.events.blocker_all():
            if mag_name in self.viewer.layers:
                mag_layer = self.viewer.layers[mag_name]
                mag_layer = self._rebind_image_layer(mag_layer, magnitudes)
            else:
                mag_layer = self.viewer.add_image(
                    magnitudes, name=mag_name,
                    colormap=cfg['magnitude_colormap'], blending='additive',
                    contrast_limits=(0, vmax), visible=True,
                )
        self._layers[cfg['magnitude_key']] = mag_layer
        self.colorbar_manager.show_for_layer(
            mag_layer, colormap_name=cfg['magnitude_colormap'],
            label=cfg['colorbar_label'],
        )

    def _set_streamed_vectors(self, cfg, vectors, colors, visible: bool) -> None:
        """Show this frame's vectors, creating the layer on the first non-empty frame."""
        vec_layer = self._layers.get(cfg['vectors_key'])
        if vec_layer is not None and cfg['vectors_layer'] in self.viewer.layers:
            if len(vectors) > 0:
                vec_layer.data = vectors
                vec_layer.edge_color = colors
            return
        if len(vectors) == 0:
            return
        vec_layer = self.viewer.add_vectors(
            vectors, name=cfg['vectors_layer'], edge_color=colors,
            edge_width=2, vector_style='arrow', blending='additive', length=1,
            visible=visible,
        )
        self._layers[cfg['vectors_key']] = vec_layer

    # --- stress streaming --------------------------------------------------
    def begin_stress_stream(self, num_frames: int, max_stress: float, downscale_factor: int) -> None:
        """Prepare the three stress image layers so BISM can stream frame by frame.

        The XX / YY / average-normal stacks are allocated lazily on the first
        frame; existing layers are reused so their contrast/visibility survive a
        re-run.
        """
        self._stress_stream = {
            'num_frames': num_frames,
            'max_stress': max_stress,
            'downscale': downscale_factor,
            'ready': False,
        }
        # Hide every non-stress layer for the duration of the run (worklist §4);
        # the stress stacks are created on the first frame and come up visible as
        # they stream in.
        self.hide_other_layers(['Normal Stress XX', 'Normal Stress YY', 'Average Normal Stress'])

    def stream_stress_frame(self, frame_index: int, stress_tensor_frame: np.ndarray) -> None:
        """Render one freshly computed stress frame (XX, YY, average) and follow it live."""
        try:
            cfg = getattr(self, '_stress_stream', None)
            if cfg is None:
                return

            downscale = cfg['downscale']
            max_stress = cfg['max_stress']

            def upscale(component):
                return self._to_display_resolution(component, downscale)

            sigma_xx = upscale(np.squeeze(stress_tensor_frame[..., 0, 0]))
            sigma_yy = upscale(np.squeeze(stress_tensor_frame[..., 1, 1]))
            sigma_normal = (sigma_xx + sigma_yy) / 2

            if not cfg['ready']:
                self._allocate_stress_stacks(cfg, sigma_xx.shape, max_stress)
                cfg['ready'] = True

            components = self.data_manager.stress_components
            if components is None:
                return
            if frame_index < 0 or frame_index >= components['num_frames']:
                return
            components['sigma_xx'][frame_index] = sigma_xx
            components['sigma_yy'][frame_index] = sigma_yy
            components['sigma_normal'][frame_index] = sigma_normal

            for key in ('stress_xx', 'stress_yy', 'stress_normal'):
                layer = self._layers.get(key)
                if layer is not None:
                    layer.refresh()
            self._advance_to_frame(frame_index)

        except Exception as e:
            error = self.create_error(
                message="Failed to stream stress frame",
                details=str(e),
                severity=ErrorSeverity.ERROR,
                recovery_hint="Check layer/array consistency",
                original_error=e,
                source="visualization",
            )
            self.handle_error(error)

    def _allocate_stress_stacks(self, cfg, upscaled_shape, max_stress) -> None:
        """Allocate the XX/YY/normal stacks and (re)bind their seismic layers."""
        num_frames = cfg['num_frames']
        sigma_xx = np.zeros((num_frames, *upscaled_shape), dtype=np.float32)
        sigma_yy = np.zeros((num_frames, *upscaled_shape), dtype=np.float32)
        sigma_normal = np.zeros((num_frames, *upscaled_shape), dtype=np.float32)

        self.data_manager.stress_components = {
            'sigma_xx': sigma_xx,
            'sigma_yy': sigma_yy,
            'sigma_normal': sigma_normal,
            'num_frames': num_frames,
            'max_stress': max_stress,
        }

        specs = [
            ('stress_xx', 'Normal Stress XX', sigma_xx),
            ('stress_yy', 'Normal Stress YY', sigma_yy),
            ('stress_normal', 'Average Normal Stress', sigma_normal),
        ]
        with self.viewer.events.blocker_all():
            for key, name, data in specs:
                if name in self.viewer.layers:
                    layer = self.viewer.layers[name]
                    self._rebind_image_layer(layer, data)
                else:
                    layer = self.viewer.add_image(
                        data, name=name, colormap='seismic', blending='additive',
                        contrast_limits=(-max_stress, max_stress), visible=True,
                    )
                self._layers[key] = layer
        self.colorbar_manager.show_for_layer(
            self._layers['stress_normal'], colormap_name='seismic',
            label='Stress (mN/m)',
        )

    # endregion

    # region === Mask Visualization
    def visualize_masks(self, masks: np.ndarray, downscale_factor: int = 1, name: str = 'Masks', opacity: float = 0.5, scale=None):
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
        scale : sequence of float, optional
            Per-axis world scale for the labels layer. Masks live on the
            downsampled analysis grid; passing the input/mask size ratio here
            renders them at the same size as the full-resolution input layers
            without inflating the array.
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
        add_labels_kwargs = dict(name=name, visible=True, opacity=opacity)
        if scale is not None:
            add_labels_kwargs['scale'] = scale
        self.viewer.add_labels(
            upscaled_masks.astype(np.uint8),
            **add_labels_kwargs,
        )
        # The mask read is async and may land before or after the cells layer;
        # re-assert the canonical stack so it never briefly sits below cells.
        self.order_input_layers()

    # endregion
