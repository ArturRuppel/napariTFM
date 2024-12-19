import numpy as np
from vispy import scene
from vispy.scene import widgets as vp_widgets
from vispy import color as vp_color
from qtpy.QtWidgets import QWidget
from qtpy.QtCore import Qt
from vispy.scene.widgets import Widget
from vispy.visuals import ColorBarVisual


class EnhancedColorBarWidget(Widget):
    """Enhanced ColorBar Widget with fine-grained control over appearance and positioning

    Parameters
    ----------
    cmap : str | vispy.color.ColorMap
        Either the name of the ColorMap or a custom ColorMap object
    orientation : {'left', 'right', 'top', 'bottom'}
        The orientation of the colorbar
    label : str
        The label to be drawn with the colorbar
    label_color : str | vispy.color.Color
        The color of labels
    clim : tuple (min, max)
        The minimum and maximum values for the colorbar scale
    border_width : float (in px)
        The width of the border
    border_color : str | vispy.color.Color
        The color of the border
    padding : tuple (major_axis, minor_axis) [0, 1]
        padding with respect to the major and minor axis
    axis_ratio : float
        ratio of minor axis to major axis
    label_offset : tuple (x, y)
        Offset of the label from its default position (-1 to 1)
    label_rotation : float
        Rotation angle of the label in degrees
    """

    def __init__(self, cmap, orientation,
                 label="", label_color='black', clim=("", ""),
                 border_width=0.0, border_color="black",
                 padding=(0.2, 0.2), axis_ratio=0.05,
                 label_offset=(0, 0), label_rotation=0,
                 **kwargs):

        # Get tick_label_offset from kwargs with default value
        self._tick_label_offset = kwargs.pop('tick_label_offset', (0, 0))

        # Initialize widget WITHOUT calling super yet
        self.unfreeze()

        # Store parameters as instance attributes
        self._major_axis_padding = padding[0]
        self._minor_axis_padding = padding[1]
        self._minor_axis_ratio = axis_ratio
        self._label_offset = label_offset
        self._label_rotation = label_rotation
        self._orientation = orientation

        # Invert the clim values for the colorbar visual
        vmin, vmax = clim
        inverted_clim = (vmax, vmin)

        # Create the ColorBarVisual with inverted limits
        self._colorbar = ColorBarVisual(
            size=(1, 1),  # dummy size
            cmap=cmap,
            orientation=orientation,
            label=label,
            clim=inverted_clim,  # Use inverted limits here
            label_color=label_color,
            border_width=border_width,
            border_color=border_color,
            **kwargs
        )
        self._colorbar.unfreeze()

        # Store original clim for reference
        self._original_clim = clim

        # NOW initialize the Widget base class
        Widget.__init__(self)

        # Add the colorbar visual to the widget
        self.add_subvisual(self._colorbar)

        # Update the colorbar
        self._update_colorbar()

        # Refreeze after setup
        self.freeze()

    def _update_tick_positions(self):
        """Update the tick label positions with offset"""
        if not hasattr(self._colorbar, '_ticks') or not self._colorbar._ticks:
            return

        # Get the base position based on orientation
        width, height = self.rect.size
        center_x, center_y = self.rect.center

        # Calculate base positions for both ticks based on orientation
        if self._orientation == 'left':
            base_x = center_x - width * 0.3
            base_y1 = center_y - height * 0.4
            base_y2 = center_y + height * 0.4
            final_positions = [
                (base_x + self._tick_label_offset[0],
                 base_y1 + self._tick_label_offset[1]),
                (base_x + self._tick_label_offset[0],
                 base_y2 - self._tick_label_offset[1])
            ]
        elif self._orientation == 'right':
            base_x = center_x + width * 0.3
            base_y1 = center_y - height * 0.4 + 5  # Microadjustments
            base_y2 = center_y + height * 0.4
            final_positions = [
                (base_x + self._tick_label_offset[0],
                 base_y1 + self._tick_label_offset[1]),
                (base_x + self._tick_label_offset[0],
                 base_y2 - self._tick_label_offset[1])
            ]
        elif self._orientation == 'top':
            base_y = center_y + height * 0.3
            base_x1 = center_x - width * 0.4  # Left tick
            base_x2 = center_x + width * 0.4  # Right tick
            final_positions = [
                (base_x1 + self._tick_label_offset[0],
                 base_y + self._tick_label_offset[1]),
                (base_x2 + self._tick_label_offset[0],
                 base_y + self._tick_label_offset[1])
            ]
        else:  # bottom
            base_y = center_y - height * 0.3
            base_x1 = center_x - width * 0.4  # Left tick
            base_x2 = center_x + width * 0.4  # Right tick
            final_positions = [
                (base_x1 + self._tick_label_offset[0],
                 base_y + self._tick_label_offset[1]),
                (base_x2 + self._tick_label_offset[0],
                 base_y + self._tick_label_offset[1])
            ]

        # Update positions
        for tick, pos in zip(self._colorbar._ticks, final_positions):
            tick.pos = pos

    def _update_positions(self):
        """Update all positions"""
        # Original label position update
        self._update_label_position()
        # New tick position update
        self._update_tick_positions()

    def _update_colorbar(self):
        """Update colorbar position and size"""
        self._colorbar.pos = self.rect.center
        self._colorbar.size = self._calc_size()
        self._update_label_position()
        self._update_tick_positions()
        self.update()

    def _update_label_position(self):
        """Update the label position with offset and rotation"""
        if not hasattr(self._colorbar, '_label') or self._colorbar._label is None:
            return

        # Get the base position based on orientation
        width, height = self.rect.size
        center_x, center_y = self.rect.center

        # Calculate base position
        if self._orientation == 'left':
            base_x = center_x - width * 0.3
            base_y = center_y
        elif self._orientation == 'right':
            base_x = center_x + width * 0.3
            base_y = center_y
        elif self._orientation == 'top':
            base_x = center_x
            base_y = center_y + height * 0.3
        else:  # bottom
            base_x = center_x
            base_y = center_y - height * 0.3

        # Apply offset
        final_x = base_x + self._label_offset[0]
        final_y = base_y + self._label_offset[1]

        # Update label position directly
        self._colorbar._label.pos = final_x, final_y

        # Apply rotation
        if self._label_rotation != 0:
            self._colorbar._label.rotation = self._label_rotation

    def on_resize(self, event):
        """Resize event handler"""
        self._update_colorbar()

    def _calc_size(self):
        """Calculate colorbar size based on container dimensions"""
        total_halfx, total_halfy = self.rect.right, self.rect.top
        if self._orientation in ["bottom", "top"]:
            total_major_axis, total_minor_axis = total_halfx, total_halfy
        else:
            total_major_axis, total_minor_axis = total_halfy, total_halfx

        major_axis = total_major_axis * (1.0 - self._major_axis_padding)
        minor_axis = major_axis * self._minor_axis_ratio
        minor_axis = np.minimum(minor_axis,
                                total_minor_axis * (1.0 - self._minor_axis_padding))

        return (major_axis, minor_axis)

    def _modify_label_transform(self):
        """Apply custom offset and rotation to label transform"""
        if not hasattr(self._colorbar, '_label') or self._colorbar._label is None:
            return

        # Get base transform from original method
        base_transform = self._original_label_transform()

        # Calculate pixel offsets based on widget size
        offset_x = self._label_offset[0]
        offset_y = self._label_offset[1]

        # Create translation transform for offset
        from vispy.visuals.transforms import STTransform
        offset_transform = STTransform(translate=[offset_x, offset_y, 0])

        # Combine transforms
        self._colorbar._label.transform = base_transform * offset_transform

        # Apply rotation if specified
        if self._label_rotation != 0:
            self._colorbar._label.rotation = self._label_rotation

    @property
    def border_color(self):
        return self._colorbar.border_color

    @border_color.setter
    def border_color(self, color):
        self._colorbar.border_color = color

    @property
    def label_offset(self):
        return self._label_offset

    @label_offset.setter
    def label_offset(self, offset):
        self._label_offset = offset
        self._update_label_position()

    @property
    def label_rotation(self):
        return self._label_rotation

    @label_rotation.setter
    def label_rotation(self, angle):
        self._label_rotation = angle
        self._update_label_position()

    @label_offset.setter
    def label_offset(self, offset):
        self._label_offset = offset
        self._update_label_position()

    @label_rotation.setter
    def label_rotation(self, angle):
        self._label_rotation = angle
        self._update_label_position()

    @property
    def tick_label_offset(self):
        """Get the current tick label offset"""
        return self._tick_label_offset

    @tick_label_offset.setter
    def tick_label_offset(self, offset):
        """Set the tick label offset and update positions"""
        self._tick_label_offset = offset
        self._update_tick_positions()

    @label_offset.setter
    def label_offset(self, offset):
        self._label_offset = offset
        self._modify_label_transform()

    @label_rotation.setter
    def label_rotation(self, angle):
        self._label_rotation = angle
        self._modify_label_transform()

    def _get_base_label_position(self):
        """Calculate the base position for the label"""
        center_x, center_y = self.rect.center
        width, height = self.rect.size

        if self._colorbar.orientation == 'left':
            return (center_x - width * 0.4, center_y)
        elif self._colorbar.orientation == 'right':
            return (center_x + width * 0.4, center_y)
        elif self._colorbar.orientation == 'top':
            return (center_x, center_y + height * 0.4)
        else:  # bottom
            return (center_x, center_y - height * 0.4)

    @property
    def cmap(self):
        return self._colorbar.cmap

    @cmap.setter
    def cmap(self, cmap):
        self._colorbar.cmap = cmap

    @property
    def label(self):
        return self._colorbar.label

    @label.setter
    def label(self, label):
        self._colorbar.label = label

    @property
    def ticks(self):
        return self._colorbar.ticks

    @ticks.setter
    def ticks(self, ticks):
        self._colorbar.ticks = ticks

    @property
    def clim(self):
        return self._original_clim

    @clim.setter
    def clim(self, clim):
        vmin, vmax = clim
        self._original_clim = (vmin, vmax)
        self._colorbar.clim = (vmax, vmin)  # Invert for the visual

    @border_color.setter
    def border_color(self, color):
        self._colorbar.border_color = color

    @property
    def border_width(self):
        return self._colorbar.border_width

    @border_width.setter
    def border_width(self, width):
        self._colorbar.border_width = width

    @property
    def orientation(self):
        return self._colorbar.orientation

    @label_offset.setter
    def label_offset(self, offset):
        self._label_offset = offset
        self._update_label_position()

    @label_rotation.setter
    def label_rotation(self, angle):
        self._label_rotation = angle
        self._update_label_position()


class ColorbarManager:
    """Manager class for creating and controlling Vispy colorbars"""

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
                        axis_ratio=0.05,
                        label_offset=(0, 0),
                        label_rotation=0,
                        tick_label_offset=(-10, 0)):
        """Create a new colorbar with the specified parameters"""

        self._canvas = scene.SceneCanvas(size=(width, height), bgcolor='transparent')

        try:
            self._colormap = vp_color.get_colormap(colormap_name)
        except KeyError:
            print(f"Colormap {colormap_name} not found, using viridis")
            self._colormap = vp_color.get_colormap('viridis')

        vmin, vmax = min(clim), max(clim)
        self._colorbar = EnhancedColorBarWidget(
            self._colormap,
            orientation,
            label=label,
            label_color=label_color,
            clim=(vmin, vmax),  # Pass in consistent order
            border_width=border_width,
            border_color=border_color,
            padding=padding,
            axis_ratio=axis_ratio,
            label_offset=label_offset,
            label_rotation=label_rotation,
            tick_label_offset=tick_label_offset
        )

        self._canvas.central_widget.add_widget(self._colorbar)
        self._current_clim = (vmin, vmax)
        return self._canvas.native

    def update_tick_label_offset(self, offset):
        """Update the tick label offset"""
        if self._colorbar is not None:
            self._colorbar.tick_label_offset = offset

    def update_limits(self, vmin, vmax):
        """Update the colorbar limits"""
        if self._colorbar is not None:
            vmin, vmax = min(vmin, vmax), max(vmin, vmax)
            self._colorbar.clim = (vmin, vmax)
            self._current_clim = (vmin, vmax)

    def update_colormap(self, colormap_name):
        """Update the colormap"""
        if self._colorbar is not None:
            try:
                new_cmap = vp_color.get_colormap(colormap_name)
                self._colormap = new_cmap
                self._colorbar.cmap = new_cmap
            except KeyError:
                print(f"Colormap {colormap_name} not found, keeping current colormap")

    def create_custom_discrete_colormap(self, colors, interpolation='zero'):
        """Create a custom discrete colormap"""
        if isinstance(colors, (list, np.ndarray)):
            self._colormap = vp_color.Colormap(colors, interpolation=interpolation)
            if self._colorbar is not None:
                self._colorbar.cmap = self._colormap

    def update_label(self, label):
        """Update the colorbar label"""
        if self._colorbar is not None:
            self._colorbar.label = label

    def update_label_position(self, offset=None, rotation=None):
        """Update the label position and rotation"""
        if self._colorbar is not None:
            if offset is not None:
                self._colorbar.label_offset = offset
            if rotation is not None:
                self._colorbar.label_rotation = rotation

    def get_native_widget(self):
        """Get the native Qt widget"""
        return self._canvas.native if self._canvas is not None else None

    def cleanup(self):
        """Clean up resources"""
        if self._canvas is not None:
            self._canvas.close()
            self._canvas = None
        self._colorbar = None
        self._colormap = None
