"""Tests for VisualizationManager.bring_layers_to_front — the shared preview
layer-management the displacement / force / stress previews all use (previously
copy-pasted three times, untested)."""

from napariTFM.utilities.visualization_manager import VisualizationManager


class _Layer:
    def __init__(self, name, visible=True):
        self.name = name
        self.visible = visible


class _Layers(list):
    def __init__(self, items):
        super().__init__(items)
        self.moves = []

    def move(self, src, dest):
        layer = self[src]
        self.moves.append((layer.name, dest))
        self.pop(src)
        self.insert(dest if dest >= 0 else len(self) + 1 + dest, layer)


class _Viewer:
    def __init__(self, layers):
        self.layers = _Layers(layers)


class _Colorbar:
    def __init__(self, names=()):
        self._names = set(names)

    def is_colorbar_layer(self, name):
        return name in self._names


def _manager(layers, colorbar_names=()):
    manager = VisualizationManager.__new__(VisualizationManager)
    manager.viewer = _Viewer(layers)
    manager.colorbar_manager = _Colorbar(colorbar_names)
    return manager


def test_shows_named_layers_and_colorbar_hides_the_rest():
    layers = [
        _Layer("Other"),
        _Layer("Colorbar"),
        _Layer("Force Magnitude", visible=False),
        _Layer("Force Vectors", visible=False),
    ]
    manager = _manager(layers, colorbar_names={"Colorbar"})

    manager.bring_layers_to_front([
        ("Force Magnitude", True),
        ("Force Vectors", True),
    ])

    by_name = {layer.name: layer for layer in manager.viewer.layers}
    assert by_name["Force Magnitude"].visible
    assert by_name["Force Vectors"].visible
    assert by_name["Colorbar"].visible            # legend rides along
    assert not by_name["Other"].visible           # everything else hidden
    # Each named layer is moved into a front slot (magnitude below vectors).
    assert ("Force Magnitude", -2) in manager.viewer.layers.moves
    assert ("Force Vectors", -1) in manager.viewer.layers.moves


def test_named_layers_can_be_stacked_but_hidden():
    # Stress preview: XX/YY stay loaded (for scrubbing) but hidden beneath the
    # visible average-normal-stress layer.
    layers = [
        _Layer("Normal Stress XX"),
        _Layer("Normal Stress YY"),
        _Layer("Average Normal Stress", visible=False),
        _Layer("Other"),
    ]
    manager = _manager(layers)

    manager.bring_layers_to_front([
        ("Normal Stress XX", False),
        ("Normal Stress YY", False),
        ("Average Normal Stress", True),
    ])

    by_name = {layer.name: layer for layer in manager.viewer.layers}
    assert not by_name["Normal Stress XX"].visible
    assert not by_name["Normal Stress YY"].visible
    assert by_name["Average Normal Stress"].visible
    assert not by_name["Other"].visible
    assert ("Average Normal Stress", -1) in manager.viewer.layers.moves


def test_absent_named_layer_is_skipped():
    layers = [_Layer("Force Magnitude", visible=False)]
    manager = _manager(layers)

    # "Force Vectors" isn't present — must not raise.
    manager.bring_layers_to_front([
        ("Force Magnitude", True),
        ("Force Vectors", True),
    ])

    assert manager.viewer.layers[0].visible
    assert [name for name, _ in manager.viewer.layers.moves] == ["Force Magnitude"]
