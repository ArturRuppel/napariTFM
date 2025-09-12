#!/usr/bin/env python3
"""
Plotting script for doublet_example TFM analysis visualization.

Creates a 2x6 subplot figure showing:
- Top row: Cell images with forces overlaid (6 equally spaced time points)
- Bottom row: Average stress images from the same time points

Uses figsize=(9.5, 4.5) and typical fontsizes from validate_MSM.py.
Replicates the same plotting methods as BatchVisualizationSaver.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
import skimage.io as io


def load_tfm_data(data_dir):
    """Load TFM analysis results from data files."""
    data_dir = Path(data_dir)
    
    print("Loading TFM data...")
    
    # Load force field data
    force_data = np.load(data_dir / "traction_forces.npy", allow_pickle=True).item()
    force_field = force_data.force_field  # Shape: (t, y, x, 2)
    force_params = force_data.parameters
    
    # Load stress tensor data  
    stress_data = np.load(data_dir / "stress_results.npy", allow_pickle=True).item()
    stress_tensor = stress_data.stress_tensor  # Shape: (t, y, x, 2, 2)
    stress_params = stress_data.parameters
    
    # Load cell images
    cell_images = io.imread(data_dir / "preprocessed_cells.tif")
    
    # Load metrics for frame count
    metrics_df = pd.read_csv(data_dir / "metrics_results.csv")
    
    print(f"Force field shape: {force_field.shape}")
    print(f"Stress tensor shape: {stress_tensor.shape}")
    print(f"Cell images shape: {cell_images.shape}")
    print(f"Number of frames: {len(metrics_df)}")
    
    return {
        'force_field': force_field,
        'force_params': force_params,
        'stress_tensor': stress_tensor,
        'stress_params': stress_params,
        'cell_images': cell_images,
        'metrics_df': metrics_df
    }


def create_stress_plot(stress_tensor, frame_idx, stress_params):
    """
    Create average normal stress plot using the same method as BatchVisualizationSaver.save_stress_visualization.
    """
    # Extract stress components (same as batch visualizer)
    stress_xx = stress_tensor[frame_idx, :, :, 0, 0]
    stress_yy = stress_tensor[frame_idx, :, :, 1, 1]
    
    # Calculate average normal stress (same as batch visualizer)
    normal_stress = (stress_xx + stress_yy) * 0.5
    
    # Get max stress parameter
    max_stress = getattr(stress_params, 'max_stress', 1.0)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Plot using seismic colormap (diverging, same as batch visualizer)
    im = ax.imshow(normal_stress, cmap='seismic', vmin=-max_stress, vmax=max_stress)
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    return fig, normal_stress.max(), normal_stress.min()


def create_doublet_visualization(data_dir, output_path=None):
    """
    Create 2x6 visualization of doublet TFM analysis using actual data.
    """
    data_dir = Path(data_dir)
    
    # Load the actual TFM data
    data = load_tfm_data(data_dir / "TFM_data")
    
    force_field = data['force_field']
    force_params = data['force_params']
    stress_tensor = data['stress_tensor']
    stress_params = data['stress_params']
    cell_images = data['cell_images']
    metrics_df = data['metrics_df']
    
    num_frames = len(metrics_df)
    
    # Use specific frames: 15, 20, 25, 30, 35, 40
    time_indices = np.array([15, 20, 25, 30, 35, 40])
    
    # Filter out indices that exceed the available frames
    time_indices = time_indices[time_indices < num_frames]
    
    print(f"Selected time indices: {time_indices}")
    
    # Create the main figure with 3 rows
    fig = plt.figure(figsize=(9.5, 7.0))
    gs = gridspec.GridSpec(3, 8, figure=fig, 
                          width_ratios=[1, 1, 1, 1, 1, 1, 0.05, 0.08],  # Extra columns for colorbars
                          height_ratios=[1, 1, 0.8],  # Third row slightly smaller
                          hspace=0.2, wspace=0.1)
    
    fig.suptitle('Doublet TFM Analysis: Force and Stress Evolution', fontsize=12, y=0.95)
    
    # Track force and stress ranges for colorbar consistency
    force_max = 0
    stress_max = 0
    stress_min = 0
    
    # Top row: Force overlays on cell images
    force_quiver = None  # Store the last quiver plot for colorbar
    for i, time_idx in enumerate(time_indices):
        ax = fig.add_subplot(gs[0, i])
        
        # Use the actual batch visualizer method
        cell_img = cell_images[time_idx] if len(cell_images.shape) == 3 else cell_images
        
        # Crop 25 pixels from each side of cell image
        cell_img = cell_img[25:-25, 25:-25]
        
        # Get force components and crop 25 pixels from each side
        tx = force_field[time_idx, 25:-25, 25:-25, 0]
        ty = force_field[time_idx, 25:-25, 25:-25, 1]
        
        # Parameters from force_params
        downscale_factor = getattr(force_params, 'downscale_factor', 2)
        f_max = getattr(force_params, 'f_max', 6000.0)
        arrow_scale = getattr(force_params, 'force_arrow_scale', 1.0)
        vector_stride = getattr(force_params, 'force_vector_stride', 20)
        
        vector_scale = arrow_scale / f_max * 50 / downscale_factor
        vector_stride_scaled = vector_stride // downscale_factor
        vector_stride_scaled = int(vector_stride_scaled * 0.75) # more arrows look better here
        
        # Resize cell image to match force map dimensions
        h, w = tx.shape
        resized_cell = resize(cell_img.astype(float), (h, w), order=3, anti_aliasing=True)
        resized_cell = 1 - resized_cell  # Invert for contrast
        
        ax.imshow(resized_cell, cmap='gray')
        
        # Calculate and plot force vectors
        force_magnitude = np.sqrt(tx**2 + ty**2)
        
        y_points = np.arange(vector_stride_scaled // 2, h - vector_stride_scaled // 2, vector_stride_scaled)
        x_points = np.arange(vector_stride_scaled // 2, w - vector_stride_scaled // 2, vector_stride_scaled)
        Y, X = np.meshgrid(y_points, x_points, indexing='ij')
        
        sampled_magnitude = force_magnitude[Y, X]
        U = tx[Y, X] * vector_scale
        V = ty[Y, X] * vector_scale
        
        # Color quiver by magnitude using inferno colormap
        force_quiver = ax.quiver(X, Y, U, V, sampled_magnitude, cmap='inferno', scale=1.0, scale_units='xy', angles='xy', width=0.003)
        
        ax.set_title(f'Frame {time_idx}', fontsize=9)
        ax.axis('off')
        
        force_max = max(force_max, sampled_magnitude.max())
    
    # Bottom row: Average stress images
    for i, time_idx in enumerate(time_indices):
        ax = fig.add_subplot(gs[1, i])
        
        # Calculate average normal stress (same as batch visualizer) and crop 25 pixels from each side
        stress_xx = stress_tensor[time_idx, 25:-25, 25:-25, 0, 0]
        stress_yy = stress_tensor[time_idx, 25:-25, 25:-25, 1, 1]
        normal_stress = (stress_xx + stress_yy) * 0.5
        
        # Get max stress for colorbar
        max_stress = 12.5
        
        # Plot with seismic colormap (diverging)
        im = ax.imshow(normal_stress, cmap='seismic', vmin=-max_stress, vmax=max_stress)
        
        ax.axis('off')
        
        if i == 0:
            ax.set_ylabel('Avg Stress', fontsize=9)
        
        stress_max = max(stress_max, normal_stress.max())
        stress_min = min(stress_min, normal_stress.min())
    
    # Third row: Time-series plots
    # Calculate strain energy over time
    from napariTFM.backend.metrics_calculator import calculate_strain_energy_density, calculate_total_strain_energy
    
    # Load displacement data to calculate strain energy
    displacement_data = np.load(data_dir / "TFM_data" / "displacements.npy", allow_pickle=True).item()
    displacements_um = displacement_data.displacement_field  # Shape: (t, y, x, 2)
    pixel_size_um = getattr(displacement_data.parameters, 'pixel_size_um', 0.1)
    pixel_area_m2 = (pixel_size_um * 1e-6) ** 2
    
    # Create simple mask (exclude 25 pixel border to match cropping)
    mask = np.ones(force_field.shape[1:3], dtype=bool)
    mask[:25, :] = False
    mask[-25:, :] = False  
    mask[:, :25] = False
    mask[:, -25:] = False
    
    strain_energies = []
    avg_stresses = []
    
    for i in range(num_frames):
        # Convert displacement to meters and crop to match force field
        disp_frame_m = displacements_um[i, 25:-25, 25:-25] * 1e-6
        force_frame_pa = force_field[i, 25:-25, 25:-25]
        mask_cropped = mask[25:-25, 25:-25]
        
        # Calculate strain energy density and total strain energy
        sed_jm2 = calculate_strain_energy_density(disp_frame_m, force_frame_pa)
        total_se_j = calculate_total_strain_energy(sed_jm2, mask_cropped, pixel_area_m2)
        strain_energies.append(total_se_j)
        
        # Calculate average stress (spatial average of normal stress)
        stress_xx = stress_tensor[i, 25:-25, 25:-25, 0, 0]
        stress_yy = stress_tensor[i, 25:-25, 25:-25, 1, 1]
        avg_normal_stress = (stress_xx + stress_yy) * 0.5
        avg_stress = np.mean(avg_normal_stress[mask_cropped])
        avg_stresses.append(avg_stress)
    
    time_points = np.arange(num_frames)
    
    # Plot strain energy over time (left subplot spanning 3 columns)
    ax_se = fig.add_subplot(gs[2, 0:3])
    ax_se.plot(time_points, strain_energies, 'b-', linewidth=1.5)
    ax_se.scatter(time_indices, [strain_energies[i] for i in time_indices], 
                  c='red', s=30, zorder=5)
    ax_se.set_xlabel('Time (frames)', fontsize=9)
    ax_se.set_ylabel('Strain Energy (J)', fontsize=9)
    ax_se.set_title('Strain Energy Over Time', fontsize=9)
    ax_se.tick_params(labelsize=8)
    ax_se.grid(True, alpha=0.3)
    
    # Plot average stress over time (right subplot spanning 3 columns)
    ax_stress = fig.add_subplot(gs[2, 3:6])
    ax_stress.plot(time_points, avg_stresses, 'g-', linewidth=1.5)
    ax_stress.scatter(time_indices, [avg_stresses[i] for i in time_indices], 
                      c='red', s=30, zorder=5)
    ax_stress.set_xlabel('Time (frames)', fontsize=9)
    ax_stress.set_ylabel('Average Stress (mN/m)', fontsize=9)
    ax_stress.set_title('Average Stress Over Time', fontsize=9)
    ax_stress.tick_params(labelsize=8)
    ax_stress.grid(True, alpha=0.3)
    
    # Add row labels (same font size as validate_MSM.py)
    fig.text(0.02, 0.8, 'Forces on\nCell Images', rotation=90, va='center', ha='center', fontsize=9)
    fig.text(0.02, 0.55, 'Average\nStress', rotation=90, va='center', ha='center', fontsize=9)
    fig.text(0.02, 0.25, 'Time Series', rotation=90, va='center', ha='center', fontsize=9)
    
    # Add colorbars with proper height using gridspace approach from validate_TFM.py
    
    # Create a gridspace with padding for force colorbar (top 20% and bottom 20% empty)
    force_gs = gridspec.GridSpecFromSubplotSpec(5, 1, gs[0, 6], height_ratios=[0.2, 0.6, 0.2, 0, 0])
    force_cax = fig.add_subplot(force_gs[1, 0])
    
    force_norm = plt.Normalize(vmin=0, vmax=force_max)
    force_sm = plt.cm.ScalarMappable(cmap='inferno', norm=force_norm)
    force_cbar = plt.colorbar(force_sm, cax=force_cax)
    force_cbar.set_label('Force Magnitude (Pa)', fontsize=8)
    force_cbar.ax.tick_params(labelsize=7)
    
    # Create a gridspace with padding for stress colorbar (top 20% and bottom 20% empty)
    stress_gs = gridspec.GridSpecFromSubplotSpec(5, 1, gs[1, 6], height_ratios=[0.2, 0.6, 0.2, 0, 0])
    stress_cax = fig.add_subplot(stress_gs[1, 0])
    stress_cbar = plt.colorbar(im, cax=stress_cax)
    stress_cbar.set_label('Stress (mN/m)', fontsize=8)
    stress_cbar.ax.tick_params(labelsize=7)
    
    # Save the figure
    if output_path is None:
        output_path = data_dir / "doublet_analysis_visualization.png"
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close(fig)
    
    # Print summary statistics
    print("\nVisualization Summary:")
    print(f"  Total frames analyzed: {num_frames}")
    print(f"  Time points visualized: {len(time_indices)}")
    print(f"  Max force magnitude: {force_max:.1f} Pa")
    print(f"  Stress range: [{stress_min:.3f}, {stress_max:.3f}] mN/m")
    print(f"  Output file: {output_path}")
    
    return output_path


def main():
    """Main function to create the doublet visualization."""
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("DOUBLET TFM ANALYSIS VISUALIZATION")
    print("="*60)
    
    try:
        output_path = create_doublet_visualization(script_dir)
        print("\n✓ Visualization created successfully!")
        print(f"  Output: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())