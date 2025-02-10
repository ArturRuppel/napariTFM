import napari
import numpy as np
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
import imageio.v2 as imageio
import time


def export_napari_to_video(viewer, output_path, fps=20, scale_factor=1):
    """
    Export a napari viewer state to an MP4 video using FFMPEG backend.

    Parameters:
    -----------
    viewer : napari.Viewer
        The napari viewer containing the visualization
    output_path : str or Path
        Path where the output MP4 should be saved
    fps : int
        Frames per second for the output video
    scale_factor : int
        Factor by which to reduce the image size (default=2 means half size)
    """
    output_path = Path(output_path)
    canvas = viewer.window._qt_viewer.canvas
    frames = []

    # Get the number of timepoints
    n_timepoints = 0
    for dim_index in range(viewer.dims.ndim):
        dim_range = viewer.dims.range[dim_index]
        if dim_range[1] > 2:
            n_timepoints = int(dim_range[1])
            time_dim = dim_index
            break

    if n_timepoints == 0:
        raise ValueError("Could not find time dimension with more than 2 points")

    print(f"Found {n_timepoints} timepoints in dimension {time_dim}")

    # Collect frames
    for t in range(n_timepoints):
        viewer.dims.set_point(time_dim, t)
        viewer.window.qt_viewer.repaint()
        QApplication.processEvents()
        time.sleep(0.1)

        qimage = canvas.native.grabFramebuffer()

        if scale_factor != 1:
            new_width = qimage.width() // scale_factor
            new_height = qimage.height() // scale_factor
            qimage = qimage.scaled(new_width, new_height,
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)

        # Convert to numpy array
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)

        # Convert BGRA to RGB
        frame = arr[:, :, [2, 1, 0]].copy()  # Make contiguous array
        frames.append(frame)

        print(f"Processing frame {t + 1}/{n_timepoints}", end='\r')

    print("\nSaving video...")

    # Save using imageio's FFMPEG writer
    writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264',
                                quality=8, pixelformat='yuv420p')

    for frame in frames:
        writer.append_data(frame)

    writer.close()
    print(f"Video saved to {output_path}")



# Usage example:
viewer = napari.current_viewer()
export_napari_to_video(viewer, 'animation.mp4', scale_factor=1)
