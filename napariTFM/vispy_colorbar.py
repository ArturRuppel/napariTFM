import numpy as np
from vispy import scene
from vispy.scene import widgets as vp_widgets
from vispy import color as vp_color
from qtpy.QtWidgets import QWidget
from qtpy.QtCore import Qt


class VispyColorbarManager:
    """Manages Vispy-based colorbars that integrate directly with napari's rendering."""

    def __init__(self):
        self._canvas = None
        self._colorbar = None
        self._colormap = None
        self._current_clim = (0, 1)

    def create_colorbar(self, width=300, height=50,
                        colormap_name='viridis',
                        label="",
                        clim=(0, 1),
                        orientation='bottom',
                        label_color='white',
                        border_color='gray',
                        border_width=1.0,
                        padding=(0.2, 0.2),
                        axis_ratio=0.05):
        """
        Create a Vispy colorbar that integrates with napari's scene.

        Parameters
        ----------
        width : int
            Width of the colorbar in pixels
        height : int
            Height of the colorbar in pixels
        colormap_name : str
            Name of the colormap to use
        label : str
            Label for the colorbar
        clim : tuple
            Color limits (min, max)
        orientation : str
            Orientation of colorbar ('top', 'bottom', 'left', 'right')
        label_color : str
            Color of the label text
        border_color : str
            Color of the border
        border_width : float
            Width of the border
        padding : tuple
            Padding around the colorbar (x, y)
        axis_ratio : float
            Ratio of axis size to colorbar size
        """
        # Create canvas sized for the colorbar
        self._canvas = scene.SceneCanvas(size=(width, height), bgcolor='transparent')

        # Get the colormap
        try:
            self._colormap = vp_color.get_colormap(colormap_name)
        except KeyError:
            print(f"Colormap {colormap_name} not found, using viridis")
            self._colormap = vp_color.get_colormap('viridis')

        # Create the colorbar widget
        self._colorbar = vp_widgets.ColorBarWidget(
            self._colormap,
            orientation,
            label,
            label_color,
            clim=clim,
            border_width=border_width,
            border_color=border_color,
            padding=padding,
            axis_ratio=axis_ratio
        )

        # Add colorbar to canvas
        self._canvas.central_widget.add_widget(self._colorbar)
        self._current_clim = clim

        # Return the native Qt widget
        return self._canvas.native

    def update_limits(self, vmin, vmax):
        """Update the colorbar limits."""
        if self._colorbar is not None:
            self._colorbar.clim = (vmin, vmax)
            self._current_clim = (vmin, vmax)

    def update_colormap(self, colormap_name):
        """Update the colormap."""
        if self._colorbar is not None:
            try:
                new_cmap = vp_color.get_colormap(colormap_name)
                self._colormap = new_cmap
                self._colorbar.cmap = new_cmap
            except KeyError:
                print(f"Colormap {colormap_name} not found, keeping current colormap")

    def create_custom_discrete_colormap(self, colors, interpolation='zero'):
        """
        Create a custom discrete colormap.

        Parameters
        ----------
        colors : array-like
            List of colors to use
        interpolation : str
            Interpolation method ('zero', 'linear')
        """
        if isinstance(colors, (list, np.ndarray)):
            self._colormap = vp_color.Colormap(colors, interpolation=interpolation)
            if self._colorbar is not None:
                self._colorbar.cmap = self._colormap

    def update_label(self, label):
        """Update the colorbar label."""
        if self._colorbar is not None:
            self._colorbar.label = label

    def get_native_widget(self):
        """Get the native Qt widget for the colorbar."""
        return self._canvas.native if self._canvas is not None else None

    def cleanup(self):
        """Clean up resources."""
        if self._canvas is not None:
            self._canvas.close()
            self._canvas = None
        self._colorbar = None
        self._colormap = None