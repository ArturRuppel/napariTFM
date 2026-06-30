from pathlib import Path

import imageio
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

from napariTFM.backend.displacement_analysis import DisplacementResult
from napariTFM.backend.fttc import FTTCResult
from napariTFM.backend.msm import MSMResult


class BatchVisualizationSaver:
    """Handles saving visualizations for batch analysis results."""

    def __init__(self, base_folder: str):
        """
        Initialize visualization saver.

        Parameters
        ----------
        base_folder : str
            Base folder where data is located
        """
        self.base_folder = Path(base_folder)
        self.viz_folder = self.base_folder / "figures"
        self.viz_folder.mkdir(parents=True, exist_ok=True)

    def save_displacement_visualization(self, displacement_results: DisplacementResult, fps: int = 10) -> None:
        """
        Create a GIF visualizing displacement fields with magnitude map and vectors.

        Shows displacement magnitude as a color map with overlaid vectors indicating
        direction and magnitude. Vector stride and scale are determined by parameters.

        Parameters
        ----------
        displacement_results : DisplacementResult
            Displacement field data and parameters
        fps : int, optional
            Frames per second for output GIF, default 10
        """
        displacement_fields = displacement_results.displacement_field
        params = displacement_results.parameters

        downscale_factor = params.downscale_factor
        d_max = params.d_max
        arrow_scale = params.disp_arrow_scale
        vector_stride = params.disp_vector_stride

        vector_scale = arrow_scale / d_max * 50 / downscale_factor
        vector_stride_scaled = vector_stride // downscale_factor

        frames = []
        for displacement_field in displacement_fields:
            fig, (ax_map, ax_cbar) = plt.subplots(2, 1, figsize=(8, 10),
                                                  gridspec_kw={'height_ratios': [20, 1]})

            plt.rcParams.update({'font.size': 18, 'text.color': 'black'})

            # Calculate and display magnitude
            magnitude = np.sqrt(np.sum(displacement_field ** 2, axis=-1))
            im = ax_map.imshow(magnitude, cmap='viridis', vmin=0, vmax=d_max)

            # Sample points for vectors
            h, w = displacement_field.shape[:2]
            y_points = np.arange(vector_stride_scaled // 2, h - vector_stride_scaled // 2, vector_stride_scaled)
            x_points = np.arange(vector_stride_scaled // 2, w - vector_stride_scaled // 2, vector_stride_scaled)
            Y, X = np.meshgrid(y_points, x_points, indexing='ij')

            # Get displacement_field components and scale them
            U = displacement_field[Y, X, 0] * vector_scale
            V = displacement_field[Y, X, 1] * vector_scale

            # Plot vectors with constant gray color
            ax_map.quiver(X, Y, U, V,
                          color='gray',
                          scale=1.0, scale_units='xy',
                          angles='xy', width=0.003)

            cbar = plt.colorbar(im, cax=ax_cbar, orientation='horizontal',
                                label='Displacement (µm)')

            cbar.ax.tick_params(labelsize=16, labelcolor='black')
            cbar.set_label('Displacement (µm)', size=20, color='black')

            ax_map.set_xticks([])
            ax_map.set_yticks([])
            plt.tight_layout()

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames.append(frame)

            plt.close(fig)

        output_path = self.viz_folder / 'displacement_map.gif'
        imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, palettesize=256, loop=0)

    def save_force_visualization(self, force_results: FTTCResult, fps: int = 10) -> None:
        """
        Create a GIF visualizing traction force fields with magnitude map and vectors.

        Shows force magnitude as a color map with overlaid vectors indicating
        direction and magnitude. Vectors below 1% of max force are filtered out.

        Parameters
        ----------
        force_results : FTTCResult
            Force field data and parameters
        fps : int, optional
            Frames per second for output GIF, default 10
        """
        tx = force_results.force_field[..., 0]
        ty = force_results.force_field[..., 1]
        params = force_results.parameters

        downscale_factor = params.downscale_factor
        f_max = params.f_max
        arrow_scale = params.force_arrow_scale
        vector_stride = params.force_vector_stride

        vector_scale = arrow_scale / f_max * 50 / downscale_factor
        vector_stride_scaled = vector_stride // downscale_factor

        frames = []
        for frame_idx in range(len(tx)):
            fig, (ax_map, ax_cbar) = plt.subplots(2, 1, figsize=(8, 10),
                                                  gridspec_kw={'height_ratios': [20, 1]})

            plt.rcParams.update({'font.size': 18, 'text.color': 'black'})

            force_magnitude = np.sqrt(tx[frame_idx] ** 2 + ty[frame_idx] ** 2)
            im = ax_map.imshow(force_magnitude, cmap='inferno', vmin=0, vmax=f_max)

            h, w = tx[frame_idx].shape
            y_points = np.arange(vector_stride_scaled // 2, h - vector_stride_scaled // 2, vector_stride_scaled)
            x_points = np.arange(vector_stride_scaled // 2, w - vector_stride_scaled // 2, vector_stride_scaled)
            Y, X = np.meshgrid(y_points, x_points, indexing='ij')

            U = tx[frame_idx][Y, X] * vector_scale
            V = ty[frame_idx][Y, X] * vector_scale

            # Use constant gray color for arrows
            ax_map.quiver(X, Y, U, V,
                          color='gray',
                          scale=1.0, scale_units='xy',
                          angles='xy', width=0.003)

            cbar = plt.colorbar(im, cax=ax_cbar, orientation='horizontal',
                                label='Force (Pa)')

            cbar.ax.tick_params(labelsize=16, labelcolor='black')
            cbar.set_label('Force (Pa)', size=20, color='black')

            ax_map.set_xticks([])
            ax_map.set_yticks([])
            plt.tight_layout()

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames.append(frame)

            plt.close(fig)

        output_path = self.viz_folder / 'force_map.gif'
        imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, loop=0)

    def save_force_cell_overlay(self, force_results: FTTCResult, cell_images: np.ndarray, fps: int = 10) -> None:
        """
        Create a GIF overlaying force vectors on phase contrast cell images.

        Shows cell images in grayscale with colored force vectors. Cell images
        are resized to match force field dimensions.

        Parameters
        ----------
        force_results : FTTCResult
            Force field data and parameters
        cell_images : np.ndarray
            Stack of cell phase contrast images
        fps : int, optional
            Frames per second for output GIF, default 10
        """
        tx = force_results.force_field[..., 0]
        ty = force_results.force_field[..., 1]
        params = force_results.parameters

        downscale_factor = params.downscale_factor
        f_max = params.f_max
        arrow_scale = params.force_arrow_scale
        vector_stride = params.force_vector_stride

        vector_scale = arrow_scale / f_max * 50 / downscale_factor
        vector_stride_scaled = vector_stride // downscale_factor

        frames = []
        for frame_idx in range(len(tx)):
            fig, (ax_map, ax_cbar) = plt.subplots(2, 1, figsize=(8, 10),
                                                  gridspec_kw={'height_ratios': [20, 1]})

            plt.rcParams.update({'font.size': 18, 'text.color': 'black'})

            # Resize cell image to match force map dimensions
            cell_img = cell_images[frame_idx].astype(float)
            h, w = tx[frame_idx].shape
            resized_cell = resize(cell_img, (h, w), order=3, anti_aliasing=True)

            # Normalize cell image
            resized_cell = 1 - resized_cell
            ax_map.imshow(resized_cell, cmap='gray')

            force_magnitude = np.sqrt(tx[frame_idx] ** 2 + ty[frame_idx] ** 2)

            y_points = np.arange(vector_stride_scaled // 2, h - vector_stride_scaled // 2, vector_stride_scaled)
            x_points = np.arange(vector_stride_scaled // 2, w - vector_stride_scaled // 2, vector_stride_scaled)
            Y, X = np.meshgrid(y_points, x_points, indexing='ij')

            sampled_magnitude = force_magnitude[Y, X]

            U = tx[frame_idx][Y, X] * vector_scale
            V = ty[frame_idx][Y, X] * vector_scale

            colors = plt.cm.inferno(sampled_magnitude / f_max).reshape(-1, 4)

            ax_map.quiver(X, Y, U, V,
                          color=colors,
                          scale=1.0, scale_units='xy',
                          angles='xy', width=0.003)

            dummy_mappable = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=f_max))
            cbar = plt.colorbar(dummy_mappable, cax=ax_cbar, orientation='horizontal',
                                label='Force (Pa)')

            cbar.ax.tick_params(labelsize=16, labelcolor='black')
            cbar.set_label('Force (Pa)', size=20, color='black')

            ax_map.set_xticks([])
            ax_map.set_yticks([])
            plt.tight_layout()

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames.append(frame)

            plt.close(fig)

        output_path = self.viz_folder / 'force_cell_overlay.gif'
        imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, loop=0)

    def save_stress_visualization(self, stress_results: MSMResult, plot_sigma_xx: bool = True,
                                  plot_sigma_yy: bool = True, plot_normal_stress: bool = True,
                                  fps: int = 10) -> None:
        """
        Create GIFs visualizing stress tensor components.

        Generates separate GIFs for selected stress components using a diverging
        colormap centered at zero.

        Parameters
        ----------
        stress_results : MSMResult
            Stress tensor data and parameters
        plot_sigma_xx : bool
            Whether to generate XX normal stress visualization
        plot_sigma_yy : bool
            Whether to generate YY normal stress visualization
        plot_normal_stress : bool
            Whether to generate average normal stress visualization
        fps : int, optional
            Frames per second for output GIFs, default 10
        """
        # Extract stress tensor and parameters
        stress_tensor = stress_results.stress_tensor
        params = stress_results.parameters
        max_stress = params.max_stress  # mN/m

        # Define components and their corresponding visualization parameters
        components = {
            'sigma_xx': {
                'data': stress_tensor[..., 0, 0],
                'label': 'Normal Stress XX (mN/m)',
                'enabled': plot_sigma_xx
            },
            'sigma_yy': {
                'data': stress_tensor[..., 1, 1],
                'label': 'Normal Stress YY (mN/m)',
                'enabled': plot_sigma_yy
            },
            'normal_stress': {
                'data': (stress_tensor[..., 0, 0] + stress_tensor[..., 1, 1]) * 0.5,  # Convert to mN/m
                'label': 'Average Normal Stress (mN/m)',
                'enabled': plot_normal_stress
            }
        }

        # Process each component if enabled
        for component_name, info in components.items():
            if not info['enabled']:
                continue

            frames = []
            for frame_idx in range(len(info['data'])):
                # Create figure with two subplots - one for stress map, one for colorbar
                fig, (ax_map, ax_cbar) = plt.subplots(2, 1, figsize=(8, 10),
                                                      gridspec_kw={'height_ratios': [20, 1]})

                # Set font properties for better visibility
                plt.rcParams.update({'font.size': 18, 'text.color': 'black'})

                # Plot stress component with seismic colormap
                im = ax_map.imshow(info['data'][frame_idx],
                                   cmap='seismic',
                                   vmin=-max_stress,
                                   vmax=max_stress)

                # Add colorbar with improved styling
                cbar = plt.colorbar(im, cax=ax_cbar, orientation='horizontal',
                                    label=info['label'])

                # Style the colorbar
                cbar.ax.tick_params(labelsize=16, labelcolor='black')
                cbar.set_label(info['label'], size=20, color='black')

                # Remove axes and adjust layout
                ax_map.set_xticks([])
                ax_map.set_yticks([])
                plt.tight_layout()

                # Convert figure to image
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                frames.append(frame)

                plt.close(fig)

            if frames:  # Only save if we have frames
                # Save as GIF
                output_path = self.viz_folder / f'{component_name}.gif'
                imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, loop=0)

    def save_mesh_visualization(self, stress_results: MSMResult, fps: int = 10) -> None:
        """
        Create a GIF showing the finite element mesh for each frame.

        Visualizes mesh elements as blue lines and nodes as red points, maintaining
        the correct aspect ratio of the original data.

        Parameters
        ----------
        stress_results : MSMResult
            Mesh data including nodes and elements for each frame
        fps : int, optional
            Frames per second for output GIF, default 10
        """

        frames = []
        stress_shape = stress_results.stress_shape
        mask_height, mask_width = stress_shape
        aspect_ratio = mask_width / mask_height

        # Calculate figure size to maintain aspect ratio with higher resolution
        base_size = 6
        if aspect_ratio > 1:
            figsize = (base_size, base_size / aspect_ratio)
        else:
            figsize = (base_size * aspect_ratio, base_size)

        total_frames = len(stress_results.nodes)

        for frame_idx in range(total_frames):
            points = np.array(stress_results.nodes[frame_idx])
            triangles = np.array(stress_results.elements[frame_idx], dtype=np.int32)

            print(f"Processing mesh frame {frame_idx + 1}/{total_frames}")

            # Create figure with high DPI and no margins
            fig = plt.figure(figsize=figsize, dpi=300)
            ax = fig.add_axes([0, 0, 1, 1])  # No padding

            try:
                # Create mesh visualization
                edges = []
                for triangle in triangles:
                    i0, i1, i2 = triangle[0], triangle[1], triangle[2]
                    edges.append(np.vstack([points[i0], points[i1]]))
                    edges.append(np.vstack([points[i1], points[i2]]))
                    edges.append(np.vstack([points[i2], points[i0]]))

                lc = plt.matplotlib.collections.LineCollection(edges, colors='b', alpha=1.0, linewidth=0.4)
                ax.add_collection(lc)

                # Plot nodes with smaller markers
                ax.plot(points[:, 0], points[:, 1], 'r.', markersize=0.6, alpha=1.0)

                # Configure axes to exactly match mask dimensions
                ax.set_xlim(0, mask_width)
                ax.set_ylim(0, mask_height)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)

                # Convert figure to image with high resolution
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                frames.append(frame)

            except Exception as e:
                print(f"Warning: Failed to generate mesh visualization for frame {frame_idx + 1}: {str(e)}")

            plt.close(fig)

        if frames:
            output_path = self.viz_folder / 'mesh_visualization.gif'
            imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, quality=95, loop=0)
            print(f"Saved mesh visualization to {output_path}")
        else:
            print("No frames were generated. Mesh visualization failed.")
