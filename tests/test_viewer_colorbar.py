import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "napariTFM" / "utilities" / "viewer_colorbar.py"
SPEC = importlib.util.spec_from_file_location("viewer_colorbar", MODULE_PATH)
viewer_colorbar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer_colorbar)

ViewerColorbarManager = viewer_colorbar.ViewerColorbarManager
make_vertical_colorbar_image = viewer_colorbar.make_vertical_colorbar_image


class _FakeLayer:
    def __init__(self, name, data, scale=(1.0, 1.0), translate=(0.0, 0.0)):
        self.name = name
        self.data = data
        self.scale = scale
        self.translate = translate


class _FakeLayers(list):
    def remove(self, layer):
        super().remove(layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayers()
        self.added_images = []
        self.added_points = []

    def add_image(self, data, **kwargs):
        layer = _FakeLayer(kwargs["name"], data)
        layer.kwargs = kwargs
        self.layers.append(layer)
        self.added_images.append((data, kwargs))
        return layer

    def add_points(self, data, **kwargs):
        layer = _FakeLayer(kwargs["name"], data)
        layer.kwargs = kwargs
        self.layers.append(layer)
        self.added_points.append((data, kwargs))
        return layer


class _Napari07LikeViewer(_FakeViewer):
    def add_points(
        self,
        data,
        *,
        name,
        size,
        face_color,
        border_color,
        scale,
        translate,
        blending,
        text,
    ):
        kwargs = {
            "name": name,
            "size": size,
            "face_color": face_color,
            "border_color": border_color,
            "scale": scale,
            "translate": translate,
            "blending": blending,
            "text": text,
        }
        layer = _FakeLayer(name, data)
        layer.kwargs = kwargs
        self.layers.append(layer)
        self.added_points.append((data, kwargs))
        return layer


def test_make_vertical_colorbar_image_returns_rgba_gradient():
    image = make_vertical_colorbar_image("viridis", height=12, width=4)

    assert image.shape == (12, 4, 4)
    assert image.dtype == np.float32
    assert np.allclose(image[:, 0, :], image[:, -1, :])
    assert not np.allclose(image[0, 0, :3], image[-1, 0, :3])
    assert np.allclose(image[..., 3], 1.0)


def test_viewer_colorbar_is_added_to_right_of_reference_layer():
    viewer = _FakeViewer()
    reference = _FakeLayer(
        name="Displacement Magnitude",
        data=np.zeros((2, 40, 80), dtype=float),
        scale=(2.0, 3.0),
        translate=(10.0, 20.0),
    )
    manager = ViewerColorbarManager(viewer)

    manager.show_for_layer(reference, colormap_name="viridis", label="Displacement (um)")

    assert len(viewer.added_images) == 1
    colorbar_data, colorbar_kwargs = viewer.added_images[0]
    assert colorbar_data.shape[0] == 40
    assert colorbar_data.shape[2] == 4
    assert colorbar_kwargs["name"] == "Displacement (um) Colorbar"
    assert colorbar_kwargs["rgb"] is True
    assert colorbar_kwargs["scale"] == (2.0, 3.0)
    assert colorbar_kwargs["translate"][0] == 10.0
    assert colorbar_kwargs["translate"][1] > 20.0 + 80 * 3.0

    assert len(viewer.added_points) == 1
    label_data, label_kwargs = viewer.added_points[0]
    assert label_data.shape == (1, 2)
    assert label_data[0, 0] == 20
    assert label_data[0, 1] > 80
    assert label_kwargs["name"] == "Displacement (um) Colorbar Label"
    assert label_kwargs["text"]["string"] == ["Displacement (um)"]


def test_viewer_colorbar_replaces_existing_legend_layers():
    viewer = _FakeViewer()
    reference = _FakeLayer(name="Force Magnitude", data=np.zeros((30, 60), dtype=float))
    manager = ViewerColorbarManager(viewer)

    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")
    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")

    names = [layer.name for layer in viewer.layers]
    assert names == ["Force (Pa) Colorbar", "Force (Pa) Colorbar Label"]


def test_viewer_colorbar_uses_napari_07_border_color_for_label_points():
    viewer = _Napari07LikeViewer()
    reference = _FakeLayer(name="Force Magnitude", data=np.zeros((30, 60), dtype=float))
    manager = ViewerColorbarManager(viewer)

    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")

    _, label_kwargs = viewer.added_points[0]
    assert label_kwargs["border_color"] == [0.0, 0.0, 0.0, 0.0]
    assert label_kwargs["text"]["string"] == ["Force (Pa)"]
