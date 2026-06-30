from typing import Iterable, Tuple

import numpy as np
from matplotlib import pyplot as plt


def make_vertical_colorbar_image(
        colormap_name: str,
        height: int,
        width: int
) -> np.ndarray:
    """Create an RGBA vertical colorbar image with high values at the top."""
    height = max(2, int(height))
    width = max(1, int(width))

    values = np.linspace(1.0, 0.0, height, dtype=np.float32)
    colors = plt.get_cmap(colormap_name)(values).astype(np.float32)
    return np.repeat(colors[:, np.newaxis, :], width, axis=1)


def _last_two(values: Iterable[float], default: Tuple[float, float]) -> Tuple[float, float]:
    if values is None:
        return default

    values = tuple(values)
    if len(values) < 2:
        return default
    return float(values[-2]), float(values[-1])


def _display_shape(data: np.ndarray) -> Tuple[int, int]:
    shape = np.asarray(data).shape
    if len(shape) < 2:
        raise ValueError("Colorbar reference layer data must be at least 2D")
    return int(shape[-2]), int(shape[-1])


# Bar spans this fraction of the image height, centred vertically.
COLORBAR_HEIGHT_FRACTION = 1.0

# How far the endpoint numbers ("1.00" / "0") are inset from the bar's ends,
# as a fraction of the bar height. The numbers are center-anchored (napari's
# top/bottom text anchors use font-wide ascender/descender metrics inflated by
# glyphs digits never contain, so they overshoot); this inset is what pulls a
# center-anchored number in by half its own height so its outer edge lands
# flush with the bar's end. Raise it to pull the two numbers closer together
# vertically without resizing the bar.
LABEL_INSET_FRACTION = 0.012


def _colorbar_dimensions(image_height: int, image_width: int) -> Tuple[int, int, int, int]:
    bar_height = max(2, int(round(image_height * COLORBAR_HEIGHT_FRACTION)))
    bar_width = max(4, int(round(image_width * 0.035)))
    # Snug the bar up against the image; only a hairline of clear space.
    gap = max(2, int(round(image_width * 0.012)))
    label_gap = max(12, int(round(image_width * 0.04)))
    return bar_height, bar_width, gap, label_gap


def _value_range(reference_layer) -> Tuple[float, float]:
    """Best-effort (min, max) for the colorbar scale from the reference layer."""
    limits = getattr(reference_layer, "contrast_limits", None)
    if limits is not None and len(limits) >= 2:
        return float(limits[0]), float(limits[1])

    data = np.asarray(reference_layer.data)
    if data.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(data)), float(np.nanmax(data))


def format_scale_value(value: float) -> str:
    """Format a colorbar endpoint compactly with sensible precision."""
    value = float(value)
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


class ViewerColorbarManager:
    """Render simple colorbar legends as napari viewer layers."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._layer_names = []

    @property
    def layer_names(self) -> Tuple[str, ...]:
        """Names of the colorbar layers (legend + label) currently in the viewer."""
        return tuple(self._layer_names)

    def is_colorbar_layer(self, name: str) -> bool:
        """True if ``name`` is one of this manager's rendered legend layers."""
        return name in self._layer_names

    def clear(self) -> None:
        if self.viewer is None:
            return

        for layer in list(self.viewer.layers):
            if layer.name in self._layer_names:
                self.viewer.layers.remove(layer)
        self._layer_names = []

    def show_for_layer(self, reference_layer, colormap_name: str, label: str) -> None:
        self.clear()

        image_height, image_width = _display_shape(reference_layer.data)
        bar_height, bar_width, gap, label_gap = _colorbar_dimensions(
            image_height,
            image_width
        )
        scale_y, scale_x = _last_two(getattr(reference_layer, "scale", None), (1.0, 1.0))
        translate_y, translate_x = _last_two(
            getattr(reference_layer, "translate", None),
            (0.0, 0.0)
        )

        colorbar_name = f"{label} Colorbar"
        label_name = f"{label} Colorbar Label"
        max_name = f"{label} Colorbar Max"
        min_name = f"{label} Colorbar Min"
        colorbar_x = image_width + gap
        bar_center_x = colorbar_x + bar_width / 2.0
        # Numbers sit just off the right edge of the bar, flush with its ends.
        number_x = colorbar_x + bar_width + max(3.0, bar_width * 0.5)
        # Centre the (shorter-than-image) bar vertically over the image.
        bar_top = (image_height - bar_height) / 2.0
        bar_bottom = bar_top + bar_height
        # Pull the endpoint numbers in from the bar ends to tighten their
        # vertical spacing without resizing the bar. 0.0 == flush with the ends.
        inset = bar_height * LABEL_INSET_FRACTION
        number_top_y = bar_top + inset
        number_bottom_y = bar_bottom - inset
        vmin, vmax = _value_range(reference_layer)

        colorbar_layer = self.viewer.add_image(
            make_vertical_colorbar_image(colormap_name, bar_height, bar_width),
            name=colorbar_name,
            rgb=True,
            scale=(scale_y, scale_x),
            translate=(translate_y + bar_top * scale_y, translate_x + colorbar_x * scale_x),
            blending="translucent",
            visible=True,
        )

        label_position = np.array(
            [[image_height / 2.0, bar_center_x]],
            dtype=float
        )
        label_layer = self.viewer.add_points(
            label_position,
            name=label_name,
            size=0.01,
            face_color=[0.0, 0.0, 0.0, 0.0],
            border_color=[0.0, 0.0, 0.0, 0.0],
            scale=(scale_y, scale_x),
            translate=(translate_y, translate_x),
            blending="translucent",
            text={
                "string": [label],
                "size": 14,
                "color": "white",
                "anchor": "center",
                "rotation": -90,
            },
        )

        # Endpoint numbers: each sits just right of the bar, center-anchored
        # and inset by LABEL_INSET_FRACTION so its own top/bottom edge lands
        # flush with the bar's top/bottom edge. A separate layer per number is
        # needed because napari's text anchor is per-layer, not per-point.
        max_layer = self._add_scale_number(
            max_name, format_scale_value(vmax), number_top_y, number_x,
            "center", (scale_y, scale_x), (translate_y, translate_x),
        )
        min_layer = self._add_scale_number(
            min_name, format_scale_value(vmin), number_bottom_y, number_x,
            "center", (scale_y, scale_x), (translate_y, translate_x),
        )

        self._layer_names = [
            colorbar_layer.name,
            label_layer.name,
            max_layer.name,
            min_layer.name,
        ]

    def _add_scale_number(self, name, text, y, x, anchor, scale, translate):
        """Add one flush colorbar endpoint number as a text-only points layer."""
        return self.viewer.add_points(
            np.array([[y, x]], dtype=float),
            name=name,
            size=0.01,
            face_color=[0.0, 0.0, 0.0, 0.0],
            border_color=[0.0, 0.0, 0.0, 0.0],
            scale=scale,
            translate=translate,
            blending="translucent",
            text={
                "string": [text],
                "size": 12,
                "color": "white",
                "anchor": anchor,
                "rotation": 0,
            },
        )
