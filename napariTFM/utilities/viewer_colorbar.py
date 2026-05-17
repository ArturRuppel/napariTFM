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


def _colorbar_dimensions(image_height: int, image_width: int) -> Tuple[int, int, int, int]:
    bar_width = max(4, int(round(image_width * 0.035)))
    gap = max(4, int(round(image_width * 0.04)))
    label_gap = max(12, int(round(image_width * 0.04)))
    return image_height, bar_width, gap, label_gap


class ViewerColorbarManager:
    """Render simple colorbar legends as napari viewer layers."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._layer_names = []

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
        colorbar_x = image_width + gap

        colorbar_layer = self.viewer.add_image(
            make_vertical_colorbar_image(colormap_name, bar_height, bar_width),
            name=colorbar_name,
            rgb=True,
            scale=(scale_y, scale_x),
            translate=(translate_y, translate_x + colorbar_x * scale_x),
            blending="translucent",
            visible=True,
        )

        label_position = np.array(
            [[image_height / 2.0, colorbar_x + bar_width + label_gap]],
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
            },
        )

        self._layer_names = [colorbar_layer.name, label_layer.name]
