"""Batch visualizations rendered with matplotlib — portable and windowless (§7).

Decision (2026-06-30): the **export** renders with matplotlib, not napari.
napari stays the live interactive viewer, but the exported movies must ship to
end users on every OS — ``pip install`` and it works, no OpenGL, no virtual
display, no window — and matplotlib is the better publication artifact anyway:
the arrows sit *on* the colormap instead of additively brightening it (napari's
GL blending oversaturates the magnitude it is meant to encode), and matplotlib
can emit vector-grade output.

Consistency with the viewer is kept by **sharing the geometry, not the
renderer**: the arrows come from the same
:func:`~napariTFM.utilities.vector_field.build_frame_vectors` the live
:class:`VisualizationManager` uses, with the same colormaps, contrast limits and
arrow-scale convention. Only the final raster is matplotlib's instead of GL.

Outputs are ``.mp4`` (libx264 via ``imageio-ffmpeg``, which bundles ffmpeg on
all platforms). Rendering uses matplotlib's ``Agg`` backend, so it needs no
display and opens no window. :class:`BatchVisualizationSaver` keeps the old
per-stage ``save_*`` surface and writes to the experiment's ``figures/`` folder.
The FE-mesh product was dropped (MSM-only diagnostic, superseded by mesh-free
BISM).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import matplotlib
import numpy as np
from skimage.transform import resize

from napariTFM.backend.displacement_analysis import DisplacementResult
from napariTFM.backend.fttc import FTTCResult
from napariTFM.backend.stress import StressResult
from napariTFM.utilities.vector_field import build_frame_vectors, upscale_field

matplotlib.use("Agg")  # headless, windowless raster backend
from matplotlib import pyplot as plt  # noqa: E402  (must follow use("Agg"))


def _even(n: int) -> int:
    """Round up to the nearest even integer (yuv420p needs even dimensions)."""
    n = int(n)
    return n if n % 2 == 0 else n + 1


def _figure_for(height: int, width: int):
    """A full-bleed figure+axes sized to the data aspect, no chrome.

    Width is pinned to 6 in at 150 dpi (900 px); height follows the aspect.
    Returns ``(fig, ax)`` with the image axes filling the whole canvas.
    """
    dpi = 150
    fig_w = 6.0
    fig_h = fig_w * height / width
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # origin upper, matches imshow
    return fig, ax


def _add_inline_colorbar(fig, cmap: str, vmin: float, vmax: float, label: str) -> None:
    """Overlay a thin vertical colorbar on the right edge (napari-viewer style)."""
    cax = fig.add_axes([0.99, 0.04, 0.022, 0.92])
    gradient = plt.get_cmap(cmap)(np.linspace(1, 0, 256))[:, None, :]
    cax.imshow(gradient, aspect="auto", extent=[0, 1, 0, 1])
    cax.set_axis_off()
    cax.text(0.5, 1.015, _fmt(vmax), color="white", ha="center", va="bottom", fontsize=8)
    cax.text(0.5, -0.015, _fmt(vmin), color="white", ha="center", va="top", fontsize=8)
    cax.text(2.6, 0.5, label, color="white", ha="center", va="center",
             rotation=-90, fontsize=8, transform=cax.transAxes)


def _fmt(value: float) -> str:
    """Compact endpoint formatting, mirroring the viewer colorbar's precision."""
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


def _fig_to_rgb(fig) -> np.ndarray:
    """Rasterize a figure to an ``(H, W, 3)`` uint8 array with even dimensions."""
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    h, w = rgb.shape[:2]
    eh, ew = _even(h), _even(w)
    if (eh, ew) != (h, w):  # pad one px so libx264/yuv420p accepts it
        padded = np.zeros((eh, ew, 3), dtype=rgb.dtype)
        padded[:h, :w] = rgb
        rgb = padded
    return rgb


_EDGE_WIDTH = 2.0  # napari's hardcoded Vectors edge_width (visualization_manager.py);
# X/Y/dx/dy here live in that same data-pixel space (the upscaled field grid), so
# pinning quiver's shaft width to this value in 'xy' units — instead of the
# default axes-fraction units — makes thickness scale with arrow length exactly
# as the live viewer's Vectors layer does.


def _quiver_args(field2d: np.ndarray, downscale: int, v_max: float,
                 arrow_scale: float, stride: int, cmap: str):
    """Sample one frame's field as quiver arrays, identical geometry to napari.

    Reuses :func:`build_frame_vectors` (the live viewer's sampler) and unpacks
    its napari ``(N, 2, 2)`` layout into matplotlib quiver components. Returns
    ``(magnitude, X, Y, dx, dy, colors)``; arrow lengths are already in display
    pixels, so quiver uses ``scale=1, scale_units='xy', angles='xy'``.
    """
    disp = upscale_field(field2d, downscale)
    magnitude = np.sqrt(np.sum(disp ** 2, axis=-1))
    scaled = disp * arrow_scale / v_max * 50
    vectors, colors = build_frame_vectors(scaled, disp, stride, v_max, colormap=cmap)
    X = vectors[:, 0, 1]
    Y = vectors[:, 0, 0]
    dx = vectors[:, 1, 1]  # column direction (U)
    dy = vectors[:, 1, 0]  # row direction (V); +dy points down under inverted ylim
    return magnitude, X, Y, dx, dy, colors


class BatchVisualizationSaver:
    """Render batch visualization movies with matplotlib (headless, .mp4)."""

    def __init__(self, base_folder: str):
        self.base_folder = Path(base_folder)
        self.viz_folder = self.base_folder / "figures"
        self.viz_folder.mkdir(parents=True, exist_ok=True)

    # region === Public API (per-stage) ====================================
    def save_displacement_visualization(self, displacement_results: DisplacementResult,
                                        fps: int = 10) -> None:
        """Displacement magnitude map (viridis) with white vectors."""
        params = displacement_results.parameters
        self._render_field_movie(
            np.asarray(displacement_results.displacement_field),
            downscale=params.downscale_factor, v_max=params.d_max,
            arrow_scale=params.disp_arrow_scale, stride=params.disp_vector_stride,
            cmap="viridis", label="Displacement (µm)", arrow_color="white",
            filename="displacement_map.mp4", fps=fps,
        )

    def save_force_visualization(self, force_results: FTTCResult, fps: int = 10) -> None:
        """Traction force magnitude map (inferno) with white vectors."""
        params = force_results.parameters
        self._render_field_movie(
            np.asarray(force_results.force_field),
            downscale=params.downscale_factor, v_max=params.f_max,
            arrow_scale=params.force_arrow_scale, stride=params.force_vector_stride,
            cmap="inferno", label="Force (Pa)", arrow_color="white",
            filename="force_map.mp4", fps=fps,
        )

    def save_force_cell_overlay(self, force_results: FTTCResult, cell_images: np.ndarray,
                                fps: int = 10) -> None:
        """Force vectors (inferno, coloured by magnitude) over grayscale cells."""
        params = force_results.parameters
        self._render_field_movie(
            np.asarray(force_results.force_field),
            downscale=params.downscale_factor, v_max=params.f_max,
            arrow_scale=params.force_arrow_scale, stride=params.force_vector_stride,
            cmap="inferno", label="Force (Pa)", arrow_color="magnitude",
            filename="force_cell_overlay.mp4", fps=fps,
            cell_images=np.asarray(cell_images),
        )

    def save_stress_visualization(self, stress_results: StressResult,
                                  plot_sigma_xx: bool = True, plot_sigma_yy: bool = True,
                                  plot_normal_stress: bool = True, fps: int = 10) -> None:
        """Selected stress-tensor components (seismic, centred at 0)."""
        stress_tensor = np.asarray(stress_results.stress_tensor)
        max_stress = float(stress_results.parameters.max_stress)

        sigma_xx = stress_tensor[..., 0, 0]
        sigma_yy = stress_tensor[..., 1, 1]
        components = [
            ("sigma_xx", sigma_xx, plot_sigma_xx),
            ("sigma_yy", sigma_yy, plot_sigma_yy),
            ("normal_stress", (sigma_xx + sigma_yy) * 0.5, plot_normal_stress),
        ]
        for name, data, enabled in components:
            if not enabled:
                continue
            self._render_image_movie(
                np.asarray(data), cmap="seismic",
                vmin=-max_stress, vmax=max_stress, label="Stress (mN/m)",
                filename=f"{name}.mp4", fps=fps,
            )

    # endregion

    # region === Rendering =================================================
    def _render_field_movie(self, fields: np.ndarray, *, downscale: int, v_max: float,
                            arrow_scale: float, stride: int, cmap: str, label: str,
                            arrow_color: str, filename: str, fps: int,
                            cell_images: Optional[np.ndarray] = None) -> None:
        """Magnitude (or cell) background + per-frame quiver → mp4."""
        if fields.ndim == 3:  # single frame (y, x, 2) -> (1, y, x, 2)
            fields = fields[None]

        with imageio.get_writer(
            str(self.viz_folder / filename), fps=fps, codec="libx264",
            quality=8, pixelformat="yuv420p", macro_block_size=1,
        ) as writer:
            for t in range(fields.shape[0]):
                magnitude, X, Y, dx, dy, colors = _quiver_args(
                    fields[t], downscale, v_max, arrow_scale, stride, cmap)
                h, w = magnitude.shape
                fig, ax = _figure_for(h, w)

                if cell_images is not None:
                    # Inverted grayscale cells as the backdrop (paper style); the
                    # colorbar still encodes the force scale.
                    cell = resize(np.asarray(cell_images[t], dtype=float), (h, w),
                                  order=3, anti_aliasing=True)
                    ax.imshow(1.0 - cell, cmap="gray", origin="upper",
                              extent=[0, w, h, 0])
                else:
                    ax.imshow(magnitude, cmap=cmap, vmin=0, vmax=v_max,
                              interpolation="bilinear", origin="upper",
                              extent=[0, w, h, 0])

                if len(X):
                    color = "white" if arrow_color == "white" else colors
                    ax.quiver(X, Y, dx, dy, color=color, angles="xy",
                              scale_units="xy", scale=1.0,
                              units="xy", width=_EDGE_WIDTH,
                              headwidth=3, headlength=4.5, headaxislength=4,
                              pivot="tail")

                _add_inline_colorbar(fig, cmap, 0.0, v_max, label)
                writer.append_data(_fig_to_rgb(fig))
                plt.close(fig)

    def _render_image_movie(self, stack: np.ndarray, *, cmap: str, vmin: float,
                            vmax: float, label: str, filename: str, fps: int) -> None:
        """Single-image-per-frame movie (stress components) → mp4."""
        if stack.ndim == 2:  # single frame (y, x) -> (1, y, x)
            stack = stack[None]

        with imageio.get_writer(
            str(self.viz_folder / filename), fps=fps, codec="libx264",
            quality=8, pixelformat="yuv420p", macro_block_size=1,
        ) as writer:
            for frame in stack:
                h, w = frame.shape
                fig, ax = _figure_for(h, w)
                ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation="bilinear", origin="upper",
                          extent=[0, w, h, 0])
                _add_inline_colorbar(fig, cmap, vmin, vmax, label)
                writer.append_data(_fig_to_rgb(fig))
                plt.close(fig)

    # endregion
