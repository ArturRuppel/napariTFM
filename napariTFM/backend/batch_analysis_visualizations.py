from pathlib import Path

import imageio
import numpy as np
from matplotlib import pyplot as plt


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
        self.viz_folder = self.base_folder / "visualizations"
        self.viz_folder.mkdir(exist_ok=True)

    def save_bead_overlay(self, bead_stack: np.ndarray, reference_image: np.ndarray, fps: int = 10) -> None:
        """
        Create and save a GIF of bead-reference overlay.

        Parameters
        ----------
        bead_stack : np.ndarray
            Stack of bead images
        reference_image : np.ndarray
            Reference image
        fps : int, optional
            Frames per second for the GIF
        """
        # Normalize reference image
        reference = reference_image.astype(float)
        ref_min, ref_max = reference.min(), reference.max()
        if ref_max > ref_min:
            reference = (reference - ref_min) / (ref_max - ref_min)

        # Create overlay frames
        frames = []
        for bead_frame in bead_stack:
            # Normalize bead frame
            bead = bead_frame.astype(float)
            bead_min, bead_max = bead.min(), bead.max()
            if bead_max > bead_min:
                bead = (bead - bead_min) / (bead_max - bead_min)

            # Create RGB overlay (magenta reference, green beads)
            overlay = np.zeros((*bead.shape, 3))
            overlay[..., 0] = reference  # Red channel (for magenta)
            overlay[..., 1] = bead  # Green channel
            overlay[..., 2] = reference  # Blue channel (for magenta)

            # Convert to uint8 for GIF
            overlay_uint8 = (overlay * 255).astype(np.uint8)
            frames.append(overlay_uint8)

        # Save as GIF with looping enabled
        output_path = self.viz_folder / 'bead_overlay.gif'
        imageio.mimsave(str(output_path), frames, fps=fps, loop=0)

    def _create_colormap_legend(self, vmin: float, vmax: float, cmap: str,
                                label: str) -> np.ndarray:
        """Create a colormap legend image."""
        fig, ax = plt.subplots(figsize=(6, 0.5))
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
                cmap=cmap
            ),
            cax=ax,
            orientation='horizontal',
            label=label
        )

        # Render to numpy array
        fig.canvas.draw()
        legend_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        legend_img = legend_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return legend_img

    def save_displacement_visualization(self, displacement_results: dict, fps: int = 10) -> None:
        """
        Create and save a GIF of displacement magnitudes with vectors.

        Parameters
        ----------
        displacement_results : dict
            Dictionary containing displacement analysis results including flows
        fps : int, optional
            Frames per second for the GIF
        """
        flows = displacement_results['flows']
        params = displacement_results.get('parameters', {})

        # Get visualization parameters from the parameters dictionary
        d_max = params.get('d_max', 10.0)  # µm
        vector_stride = params.get('vector_stride', 20)
        arrow_scale = params.get('arrow_scale', 1.0)

        frames = []
        for flow in flows:
            # Create figure with two subplots - one for vectors, one for colorbar
            fig, (ax_map, ax_cbar) = plt.subplots(2, 1, figsize=(8, 10),
                                                  gridspec_kw={'height_ratios': [20, 1]})  # Increased ratio for thinner colorbar

            # Set font properties for better visibility
            plt.rcParams.update({'font.size': 18, 'text.color': 'black'})

            # Calculate and display magnitude
            magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))  # Convert to µm
            im = ax_map.imshow(magnitude, cmap='viridis', vmin=0, vmax=d_max)

            # Add vectors
            h, w = flow.shape[:2]
            y_points = np.arange(vector_stride // 2, h - vector_stride // 2, vector_stride)
            x_points = np.arange(vector_stride // 2, w - vector_stride // 2, vector_stride)
            Y, X = np.meshgrid(y_points, x_points, indexing='ij')

            # Get flow components and scale them by arrow_scale parameter
            U = flow[Y, X, 0] * arrow_scale
            V = flow[Y, X, 1] * arrow_scale

            # Calculate magnitudes for vector filtering
            magnitudes = np.sqrt(U ** 2 + V ** 2)
            mask = magnitudes > -1

            # Plot filtered vectors
            ax_map.quiver(X[mask], Y[mask], U[mask], V[mask],
                          magnitudes[mask], cmap='viridis',
                          scale=1.0, scale_units='xy',
                          angles='xy', width=0.003)

            # Add colorbar with improved styling
            cbar = plt.colorbar(im, cax=ax_cbar, orientation='horizontal',
                                label='Displacement (µm)')

            # Style the colorbar
            cbar.ax.tick_params(labelsize=16, labelcolor='black')
            cbar.set_label('Displacement (µm)', size=20, color='black')

            # Remove axes and adjust layout
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            plt.tight_layout()

            # Convert figure to image
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

            plt.close(fig)

        # Save as GIF with looping enabled
        output_path = self.viz_folder / 'displacement_map.gif'
        imageio.mimsave(
            str(output_path),
            frames,
            fps=fps,
            optimize=False,  # Disable optimization to prevent per-frame palette
            palettesize=256,  # Use maximum palette size
            loop=0  # Enable infinite looping
        )

    def save_force_visualization(self, force_results: dict, fps: int = 10) -> None:
        """
        Create and save a GIF of force magnitudes with vectors.

        Parameters
        ----------
        force_results : dict
            Dictionary containing force analysis results including tx and ty components
        fps : int, optional
            Frames per second for the GIF
        """
        # Extract force components and parameters
        tx = force_results['tx']
        ty = force_results['ty']
        params = force_results.get('parameters', {})

        # Get visualization parameters
        f_max = params.get('f_max', 1000.0)  # Pa
        vector_stride = params.get('vector_stride', 20)
        arrow_scale = params.get('arrow_scale', 1.0)

        frames = []
        for frame_idx in range(len(tx)):
            # Create figure with two subplots - one for vectors, one for colorbar
            fig, (ax_map, ax_cbar) = plt.subplots(2, 1, figsize=(8, 10),
                                                  gridspec_kw={'height_ratios': [20, 1]})

            # Set font properties for better visibility
            plt.rcParams.update({'font.size': 18, 'text.color': 'black'})

            # Calculate and display magnitude
            force_magnitude = np.sqrt(tx[frame_idx] ** 2 + ty[frame_idx] ** 2)
            im = ax_map.imshow(force_magnitude, cmap='inferno', vmin=0, vmax=f_max)

            # Add vectors
            h, w = tx[frame_idx].shape
            y_points = np.arange(vector_stride // 2, h - vector_stride // 2, vector_stride)
            x_points = np.arange(vector_stride // 2, w - vector_stride // 2, vector_stride)
            Y, X = np.meshgrid(y_points, x_points, indexing='ij')

            # Get force components at grid points
            U = tx[frame_idx][Y, X] * arrow_scale / 1000  # Scale down for visualization
            V = ty[frame_idx][Y, X] * arrow_scale / 1000

            # Calculate magnitudes for vector filtering and coloring
            magnitudes = np.sqrt(tx[frame_idx][Y, X] ** 2 + ty[frame_idx][Y, X] ** 2)  # Original magnitudes in Pa
            mask = magnitudes > -1

            # Get colors for vectors based on magnitude
            colors = plt.cm.inferno(magnitudes[mask] / f_max)  # Normalize by f_max

            # Plot filtered vectors
            ax_map.quiver(X[mask], Y[mask], U[mask], V[mask],
                          color=colors,
                          scale=1.0, scale_units='xy',
                          angles='xy', width=0.003)

            # Add colorbar with improved styling
            cbar = plt.colorbar(im, cax=ax_cbar, orientation='horizontal',
                                label='Force (Pa)')

            # Style the colorbar
            cbar.ax.tick_params(labelsize=16, labelcolor='black')
            cbar.set_label('Force (Pa)', size=20, color='black')

            # Remove axes and adjust layout
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            plt.tight_layout()

            # Convert figure to image
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

            plt.close(fig)

        # Save as GIF
        output_path = self.viz_folder / 'force_map.gif'
        imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, loop=0)

    def save_force_cell_overlay(self, force_results: dict, cell_images: np.ndarray, fps: int = 10) -> None:
        """
        Create and save a GIF of force vectors overlaid on the cell images.

        Parameters
        ----------
        force_results : dict
            Dictionary containing force analysis results including tx and ty components
        cell_images : np.ndarray
            Stack of cell images with shape (n_frames, height, width)
        fps : int, optional
            Frames per second for the GIF
        """
        # Extract force components and parameters
        tx = force_results['tx']
        ty = force_results['ty']
        params = force_results.get('parameters', {})

        # Get visualization parameters
        f_max = params.get('f_max', 1000.0)  # Pa
        vector_stride = params.get('vector_stride', 20)
        arrow_scale = params.get('arrow_scale', 1.0)

        # Ensure cell_images has at least as many frames as traction forces
        n_frames = min(len(tx), len(cell_images))
        cell_images = cell_images[:n_frames]

        frames = []
        for frame_idx in range(n_frames):
            # Create figure with two subplots - one for vectors, one for colorbar
            fig, (ax_map, ax_cbar) = plt.subplots(2, 1, figsize=(8, 10),
                                                  gridspec_kw={'height_ratios': [20, 1]})

            # Set font properties for better visibility
            plt.rcParams.update({'font.size': 18, 'text.color': 'black'})

            # Normalize and invert cell image
            cell_img = cell_images[frame_idx].astype(float)
            cell_img = (cell_img - cell_img.min()) / (cell_img.max() - cell_img.min())
            cell_img = 1 - cell_img

            # Display cell image
            ax_map.imshow(cell_img, cmap='gray')

            # Calculate force magnitude for colormap
            force_magnitude = np.sqrt(tx[frame_idx] ** 2 + ty[frame_idx] ** 2)

            # Create a dummy mappable for the colorbar
            dummy_mappable = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=f_max))

            # Calculate scaling factors between force grid and cell image
            scale_y = cell_img.shape[0] / tx[frame_idx].shape[0]
            scale_x = cell_img.shape[1] / tx[frame_idx].shape[1]

            # Add vectors using the same stride as in save_force_visualization
            h, w = tx[frame_idx].shape
            y_points = np.arange(vector_stride // 2, h - vector_stride // 2, vector_stride)
            x_points = np.arange(vector_stride // 2, w - vector_stride // 2, vector_stride)
            Y, X = np.meshgrid(y_points, x_points, indexing='ij')

            # Scale grid points to match cell image dimensions
            Y_scaled = Y * scale_y
            X_scaled = X * scale_x

            # Get force components and scale them
            U = tx[frame_idx][Y, X] * arrow_scale / 1000  # Scale down for visualization
            V = ty[frame_idx][Y, X] * arrow_scale / 1000

            # Scale the vectors by the same factor as the grid
            U_scaled = U * scale_x
            V_scaled = V * scale_y

            # Calculate magnitudes for vector filtering and coloring
            magnitudes = np.sqrt(tx[frame_idx][Y, X] ** 2 + ty[frame_idx][Y, X] ** 2)  # Original magnitudes in Pa
            mask = magnitudes > f_max * 0.01  # Filter threshold in Pa

            # Get colors for vectors based on magnitude
            colors = plt.cm.inferno(magnitudes[mask] / f_max)  # Normalize by f_max without scaling

            # Plot filtered vectors with scaled grid and vectors
            ax_map.quiver(X_scaled[mask], Y_scaled[mask], U_scaled[mask], V_scaled[mask],
                          color=colors,
                          scale=1.0, scale_units='xy',
                          angles='xy', width=0.003)

            # Add colorbar with improved styling
            cbar = plt.colorbar(dummy_mappable, cax=ax_cbar, orientation='horizontal',
                                label='Force (Pa)')

            # Style the colorbar
            cbar.ax.tick_params(labelsize=16, labelcolor='black')
            cbar.set_label('Force (Pa)', size=20, color='black')

            # Remove axes and adjust layout
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            plt.tight_layout()

            # Convert figure to image
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

            plt.close(fig)

        # Save as GIF
        output_path = self.viz_folder / 'force_cell_overlay.gif'
        imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, loop=0)

    def save_stress_visualization(self, stress_results: dict, fps: int = 10) -> None:
        """
        Create and save GIFs of stress tensor components.

        Parameters
        ----------
        stress_results : dict
            Dictionary containing stress tensor results and parameters
        fps : int, optional
            Frames per second for the GIF
        """
        # Extract stress tensor and parameters
        stress_tensor = stress_results['stress_tensor']
        params = stress_results.get('parameters', {})
        max_stress = params.get('max_stress', 10.0)  # mN/m

        # Define components and their corresponding visualization parameters
        components = {
            'sigma_xx': {
                'data': stress_tensor[..., 0, 0] * 1e3,  # Convert to mN/m
                'label': 'Normal Stress XX (mN/m)',
                'enabled': params.get('save_sigma_xx', False)
            },
            'sigma_yy': {
                'data': stress_tensor[..., 1, 1] * 1e3,  # Convert to mN/m
                'label': 'Normal Stress YY (mN/m)',
                'enabled': params.get('save_sigma_yy', False)
            },
            'shear': {
                'data': stress_tensor[..., 0, 1] * 1e3,  # Convert to mN/m
                'label': 'Shear Stress (mN/m)',
                'enabled': params.get('save_shear', False)
            },
            'normal_stress': {
                'data': (stress_tensor[..., 0, 0] + stress_tensor[..., 1, 1]) * 0.5e3,  # Convert to mN/m
                'label': 'Average Normal Stress (mN/m)',
                'enabled': params.get('save_normal_stress', False)
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
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)

                plt.close(fig)

            if frames:  # Only save if we have frames
                # Save as GIF
                output_path = self.viz_folder / f'{component_name}.gif'
                imageio.mimsave(str(output_path), frames, fps=fps, optimize=False, loop=0)