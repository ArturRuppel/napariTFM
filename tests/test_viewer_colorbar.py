import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "napariTFM" / "utilities" / "viewer_colorbar.py"
SPEC = importlib.util.spec_from_file_location("viewer_colorbar", MODULE_PATH)
viewer_colorbar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer_colorbar)

ViewerColorbarManager = viewer_colorbar.ViewerColorbarManager
make_vertical_colorbar_image = viewer_colorbar.make_vertical_colorbar_image
format_scale_value = viewer_colorbar.format_scale_value


class _FakeLayer:
    def __init__(self, name, data, scale=(1.0, 1.0), translate=(0.0, 0.0),
                 contrast_limits=None):
        self.name = name
        self.data = data
        self.scale = scale
        self.translate = translate
        if contrast_limits is not None:
            self.contrast_limits = contrast_limits


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


def test_format_scale_value_picks_compact_precision():
    assert format_scale_value(0) == "0"
    assert format_scale_value(1500) == "1500"      # large forces: no decimals
    assert format_scale_value(-250) == "-250"      # signed stress endpoint
    assert format_scale_value(42.0) == "42.0"
    assert format_scale_value(1.5) == "1.50"
    assert format_scale_value(0.0123) == "0.0123"  # small displacement (um)


def test_viewer_colorbar_uses_data_range_when_contrast_limits_absent():
    viewer = _FakeViewer()
    data = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=float)
    reference = _FakeLayer(name="Force Magnitude", data=data)  # no contrast_limits
    manager = ViewerColorbarManager(viewer)

    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")

    _, max_kwargs = viewer.added_points[1]
    _, min_kwargs = viewer.added_points[2]
    assert max_kwargs["text"]["string"] == ["8.00"]
    assert min_kwargs["text"]["string"] == ["0"]


def test_viewer_colorbar_is_added_to_right_of_reference_layer():
    viewer = _FakeViewer()
    reference = _FakeLayer(
        name="Displacement Magnitude",
        data=np.zeros((2, 40, 80), dtype=float),
        scale=(2.0, 3.0),
        translate=(10.0, 20.0),
        contrast_limits=(0.0, 1.5),
    )
    manager = ViewerColorbarManager(viewer)

    manager.show_for_layer(reference, colormap_name="viridis", label="Displacement (um)")

    assert len(viewer.added_images) == 1
    colorbar_data, colorbar_kwargs = viewer.added_images[0]
    # Bar spans the full image height (40) and starts flush with the image top,
    # so its translate matches the reference's translate_y (10).
    assert colorbar_data.shape[0] == 40
    assert colorbar_data.shape[2] == 4
    assert colorbar_kwargs["name"] == "Displacement (um) Colorbar"
    assert colorbar_kwargs["rgb"] is True
    assert colorbar_kwargs["scale"] == (2.0, 3.0)
    assert colorbar_kwargs["translate"][0] == 10.0
    assert colorbar_kwargs["translate"][1] > 20.0 + 80 * 3.0

    assert len(viewer.added_points) == 3
    label_data, label_kwargs = viewer.added_points[0]
    assert label_data.shape == (1, 2)
    # Label is rotated -90 degrees and centred over the colorbar column so it
    # reads vertically along the bar without the glyphs appearing mirrored on
    # napari's y-down canvas.
    assert label_data[0, 0] == 20
    assert label_data[0, 1] > 80
    assert label_kwargs["name"] == "Displacement (um) Colorbar Label"
    assert label_kwargs["text"]["string"] == ["Displacement (um)"]
    assert label_kwargs["text"]["rotation"] == -90

    # Scale endpoints: each is its own layer, sitting just right of the bar and
    # inset from one end by LABEL_INSET_FRACTION * bar_height. Both are
    # center-anchored — napari's "upper_left"/"lower_left" text anchors use
    # font-wide ascender/descender metrics inflated by glyphs digits never
    # contain (e.g. accents, descenders), which overshoots and visually
    # centers the text on the anchor point instead of sitting flush below/
    # above it. Center-anchoring plus a half-text-height inset is what
    # actually lands the number's own edge flush with the bar's edge. The bar
    # is full height (40), so the un-inset ends would be at y == 0 and y == 40.
    max_data, max_kwargs = viewer.added_points[1]
    min_data, min_kwargs = viewer.added_points[2]
    assert max_data.shape == (1, 2)
    assert min_data.shape == (1, 2)
    assert max_kwargs["name"] == "Displacement (um) Colorbar Max"
    assert min_kwargs["name"] == "Displacement (um) Colorbar Min"
    assert 0 < max_data[0, 0] < 20        # inset down from the bar's top
    assert 20 < min_data[0, 0] < 40       # inset up from the bar's bottom
    assert max_data[0, 1] > label_data[0, 1]   # off to the right of the bar
    assert min_data[0, 1] == max_data[0, 1]    # same column for both numbers
    assert max_kwargs["text"]["string"] == ["1.50"]
    assert max_kwargs["text"]["anchor"] == "center"
    assert min_kwargs["text"]["string"] == ["0"]
    assert min_kwargs["text"]["anchor"] == "center"
    assert max_kwargs["text"]["rotation"] == 0


def test_colorbar_layer_names_track_rendered_legend_layers():
    viewer = _FakeViewer()
    reference = _FakeLayer(name="Force Magnitude", data=np.zeros((30, 60), dtype=float))
    manager = ViewerColorbarManager(viewer)

    assert manager.layer_names == ()
    assert manager.is_colorbar_layer("Force (Pa) Colorbar") is False

    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")

    assert manager.layer_names == (
        "Force (Pa) Colorbar",
        "Force (Pa) Colorbar Label",
        "Force (Pa) Colorbar Max",
        "Force (Pa) Colorbar Min",
    )
    assert manager.is_colorbar_layer("Force (Pa) Colorbar") is True
    assert manager.is_colorbar_layer("Force (Pa) Colorbar Label") is True
    assert manager.is_colorbar_layer("Force (Pa) Colorbar Max") is True
    assert manager.is_colorbar_layer("Force (Pa) Colorbar Min") is True
    assert manager.is_colorbar_layer("Force Magnitude") is False

    manager.clear()
    assert manager.layer_names == ()
    assert manager.is_colorbar_layer("Force (Pa) Colorbar") is False


def test_viewer_colorbar_replaces_existing_legend_layers():
    viewer = _FakeViewer()
    reference = _FakeLayer(name="Force Magnitude", data=np.zeros((30, 60), dtype=float))
    manager = ViewerColorbarManager(viewer)

    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")
    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")

    names = [layer.name for layer in viewer.layers]
    assert names == [
        "Force (Pa) Colorbar",
        "Force (Pa) Colorbar Label",
        "Force (Pa) Colorbar Max",
        "Force (Pa) Colorbar Min",
    ]


def test_viewer_colorbar_uses_napari_07_border_color_for_label_points():
    viewer = _Napari07LikeViewer()
    reference = _FakeLayer(name="Force Magnitude", data=np.zeros((30, 60), dtype=float))
    manager = ViewerColorbarManager(viewer)

    manager.show_for_layer(reference, colormap_name="inferno", label="Force (Pa)")

    _, label_kwargs = viewer.added_points[0]
    assert label_kwargs["border_color"] == [0.0, 0.0, 0.0, 0.0]
    assert label_kwargs["text"]["string"] == ["Force (Pa)"]
