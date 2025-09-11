#!/usr/bin/env python3
"""
Validation script for MSM (Monolayer Stress Microscopy) analysis.

This script loads and visualizes ground truth stress data for MSM validation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add the parent directory to path to import napariTFM modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from napariTFM.services.msm_service import MSMService
from napariTFM.backend.parameter_dataclasses import MSMParameters


def create_square_plate_problem(size=100, edge_traction=4000, buffer=5):
    """
    Create a square plate problem with uniform edge tractions and buffer zone

    Args:
        size: Size of the square plate in pixels
        edge_traction: Magnitude of edge traction in Pa
        buffer: Size of buffer zone around the domain in pixels

    Returns:
        tx, ty: Traction field components
        mask: Boolean mask defining the plate
        traction_scale: Scaling factor for analytical solution
    """
    # Create domain with buffer
    total_size = size + 2 * buffer
    mask = np.zeros((total_size, total_size), dtype=bool)
    mask[buffer:-buffer, buffer:-buffer] = True  # Active domain

    # Initialize traction fields
    tx = np.zeros((total_size, total_size))
    ty = np.zeros((total_size, total_size))

    # Apply tractions in a balanced way
    # For x-direction: equal and opposite forces on left and right edges
    tx[buffer:-buffer, buffer:buffer + 1] = edge_traction  # Left edge
    tx[buffer:-buffer, -(buffer + 1):-buffer] = -edge_traction  # Right edge

    # For y-direction: equal and opposite forces on top and bottom edges
    ty[buffer:buffer + 1, buffer:-buffer] = edge_traction  # Top edge
    ty[-(buffer + 1):-buffer, buffer:-buffer] = -edge_traction  # Bottom edge

    # Store original traction magnitudes for scaling
    tx_max = np.max(np.abs(tx))
    ty_max = np.max(np.abs(ty))
    traction_scale = max(tx_max, ty_max)

    return tx, ty, mask, traction_scale


def calculate_square_plate_analytical_stress(traction_scale, mask, params):
    """Calculate analytical stress solution for square plate."""
    # Create analytical solution in mN/m (to match MSM output units)
    # MSM outputs: stress [mN/m] = stress [Pa] × downscale_factor × pixel_size [µm] × 1e-3
    analytical_stress_mNm = traction_scale * params.downscale_factor * params.pixel_size * 1e-3
    
    # For a square plate under balanced edge loading, stress is uniform
    stress_xx = np.zeros_like(mask, dtype=float)
    stress_yy = np.zeros_like(mask, dtype=float)
    
    # Apply uniform stress inside the domain
    stress_xx[mask] = analytical_stress_mNm
    stress_yy[mask] = analytical_stress_mNm
    
    # Calculate normal stress
    stress_normal = (stress_xx + stress_yy) / 2
    
    return stress_xx, stress_yy, stress_normal


def load_ground_truth_stress(folder_path):
    """Load ground truth stress components from .npy files."""
    stress_xx = np.load(os.path.join(folder_path, 'stress_xx.npy'))
    stress_yy = np.load(os.path.join(folder_path, 'stress_yy.npy'))
    return stress_xx, stress_yy


def load_ground_truth_traction(folder_path):
    """Load ground truth traction components from .npy files."""
    trac_x = np.load(os.path.join(folder_path, 'traction_x.npy'))
    trac_y = np.load(os.path.join(folder_path, 'traction_y.npy'))
    return trac_x, trac_y


def get_msm_parameters():
    """Get MSM parameters for validation."""
    return MSMParameters(
        # Mesh parameters
        density_factor=0.01,
        mesh_algorithm='Delaunay',
        use_optimization=False,
        
        # Material parameters  
        poisson_ratio_cells=0.5,
        young_modulus=1000.0,  # Pa
        
        # Scaling parameters
        pixel_size=0.3,  # µm
        downscale_factor=1
    )


def calculate_msm_stress(trac_x, trac_y, params):
    """Calculate stress field using MSMService with given parameters."""
    # Debug: Print data statistics
    print(f"  Input data ranges:")
    print(f"    Traction X: [{np.nanmin(trac_x):.6f}, {np.nanmax(trac_x):.6f}] Pa")
    print(f"    Traction Y: [{np.nanmin(trac_y):.6f}, {np.nanmax(trac_y):.6f}] Pa")
    trac_magnitude = np.sqrt(trac_x**2 + trac_y**2)
    print(f"    Traction magnitude: [{np.nanmin(trac_magnitude):.6f}, {np.nanmax(trac_magnitude):.6f}] Pa")
    
    # Create mask from non-NaN values (this is where the cells are)
    mask = ~np.isnan(trac_x) & ~np.isnan(trac_y)
    print(f"    Valid data coverage: {np.sum(mask)}/{mask.size} pixels ({100*np.sum(mask)/mask.size:.1f}%)")
    
    # Replace NaN values with zeros for MSM calculation
    trac_x_clean = np.nan_to_num(trac_x, nan=0.0)
    trac_y_clean = np.nan_to_num(trac_y, nan=0.0)
    
    try:
        # Initialize MSM service
        print(f"  Initializing MSM service with parameters:")
        print(f"    Young's modulus: {params.young_modulus} Pa")
        print(f"    Poisson ratio: {params.poisson_ratio_cells}")
        print(f"    Density factor: {params.density_factor}")
        print(f"    Pixel size: {params.pixel_size} µm")
        
        service = MSMService(params)
        
        # Prepare force field in the format expected by MSMService (H, W, 2)
        force_field = np.stack([trac_x_clean, trac_y_clean], axis=-1)
        
        # Prepare masks (add time dimension as service expects 3D: T, H, W)
        masks = mask[np.newaxis, ...]  # Shape: (1, H, W)
        
        print(f"  Running MSM calculation using service...")
        print(f"    Force field shape: {force_field.shape}")
        print(f"    Masks shape: {masks.shape}")
        
        # Calculate stresses using the service - this returns a generator
        stress_generator = service.calculate_stresses(force_field, masks)
        
        # Get the final result from the generator
        try:
            # Process through the generator to get final result
            result = None
            for intermediate_result, frame, total_frames in stress_generator:
                result = intermediate_result
                print(f"    Processing frame {frame}/{total_frames}")
        except StopIteration as e:
            # The final result is returned via StopIteration.value
            result = e.value
        
        if result is None:
            raise ValueError("MSM calculation did not return a valid result")
        
        # Extract stress components from MSMResult
        # result.stress_tensor has shape (1, H, W, 2, 2) for single frame
        stress_tensor = result.stress_tensor[0]  # Remove time dimension: (H, W, 2, 2)
        
        stress_xx = stress_tensor[:, :, 0, 0]
        stress_yy = stress_tensor[:, :, 1, 1]
        
        # Calculate normal stress (average of normal components)
        stress_normal = (stress_xx + stress_yy) / 2
        
        # Debug: Print output statistics
        print(f"  Output stress ranges (in {result.physical_scale['stress_units']}):")
        print(f"    Stress XX: [{np.nanmin(stress_xx):.6f}, {np.nanmax(stress_xx):.6f}]")
        print(f"    Stress YY: [{np.nanmin(stress_yy):.6f}, {np.nanmax(stress_yy):.6f}]")
        print(f"    Stress Normal: [{np.nanmin(stress_normal):.6f}, {np.nanmax(stress_normal):.6f}]")
        print(f"    Units: {result.physical_scale['stress_units']}")
        
        return stress_xx, stress_yy, stress_normal, result.condition_number, result.residual
        
    except Exception as e:
        print(f"  Error in MSM calculation: {e}")
        import traceback
        traceback.print_exc()
        raise e

def calculate_correlation_metrics(calculated, ground_truth, mask=None):
    """Calculate correlation metrics between calculated and ground truth stress fields (no RMSE)."""
    if mask is None:
        mask = np.ones_like(calculated, dtype=bool)
    
    # Apply mask and filter out NaN values
    valid_mask = mask & ~np.isnan(calculated) & ~np.isnan(ground_truth)
    
    if not np.any(valid_mask):
        return {'correlation': 0, 'max_error': 0}
    
    calc_valid = calculated[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    # Maximum error (kept for reference)
    max_error = np.max(np.abs(calc_valid - gt_valid))
    
    # Correlation coefficient
    if np.std(calc_valid) > 0 and np.std(gt_valid) > 0:
        correlation = np.corrcoef(calc_valid, gt_valid)[0, 1]
    else:
        correlation = 0
    
    return {
        'correlation': correlation,
        'max_error': max_error
    }


def plot_stress_validation_comparison(gt_stress_xx, gt_stress_yy, gt_stress_normal,
                                    calc_stress_xx, calc_stress_yy, calc_stress_normal, 
                                    xx_errors, yy_errors, normal_errors, vmax=5.1):
    """Plot 2x4 stress validation: Ground Truth vs Calculated for σ_xx, σ_yy, σ_normal with correlation bar plot."""
    fig = plt.figure(figsize=(9.5, 4.5))  # DIN A4 compatible width
    
    # Use 5 rows to create vertical padding for correlation plot, 5 cols for spacing
    gs = gridspec.GridSpec(5, 5, figure=fig, 
                          width_ratios=[0.25, 0.25, 0.25, 0.11, 0.14],
                          height_ratios=[0.1, 0.4, 0.1, 0.4, 0.1],
                          wspace=0.0, hspace=0.1)
    
    fig.suptitle('MSM Stress Field Validation', fontsize=12, y=0.9)
    
    # Create axes using gridspec - maps use rows 1 and 3, cols 0-2
    axes = []
    # Top row (row 1)
    top_axes = []
    for j in range(3):  # Only first 3 columns for maps
        ax = fig.add_subplot(gs[1, j])
        top_axes.append(ax)
    axes.append(top_axes)
    
    # Bottom row (row 3) 
    bottom_axes = []
    for j in range(3):  # Only first 3 columns for maps
        ax = fig.add_subplot(gs[3, j])
        bottom_axes.append(ax)
    axes.append(bottom_axes)
    
    # Data arrays and titles
    gt_data = [gt_stress_xx, gt_stress_yy, gt_stress_normal]
    calc_data = [calc_stress_xx, calc_stress_yy, calc_stress_normal]
    titles = ['σ_xx', 'σ_yy', 'σ_normal']
    errors = [xx_errors, yy_errors, normal_errors]
    
    # Plot stress fields for each component
    for i in range(3):
        # Use the passed vmax parameter for colorbar scales
        vmin = -vmax
        
        # Ground truth (top row)
        im_gt = axes[0][i].imshow(gt_data[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0][i].set_title(f'{titles[i]} Stress', fontsize=9)
        axes[0][i].axis('off')
        divider_gt = make_axes_locatable(axes[0][i])
        cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
        cbar_gt = plt.colorbar(im_gt, cax=cax_gt)
        if i == 2:  # Only add label to rightmost column
            cbar_gt.set_label('Stress (mN/m)', fontsize=7)
        cbar_gt.ax.tick_params(labelsize=6)
        
        # Calculated (bottom row)
        im_calc = axes[1][i].imshow(calc_data[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1][i].set_title('', fontsize=8)  # Remove individual title
        axes[1][i].axis('off')
        divider_calc = make_axes_locatable(axes[1][i])
        cax_calc = divider_calc.append_axes("right", size="5%", pad=0.05)
        cbar_calc = plt.colorbar(im_calc, cax=cax_calc)
        if i == 2:  # Only add label to rightmost column
            cbar_calc.set_label('Stress (mN/m)', fontsize=7)
        cbar_calc.ax.tick_params(labelsize=6)
    
    # Create correlation subplot spanning middle 80% vertical space (rows 1-3), column 4
    ax_corr = fig.add_subplot(gs[1:4, 4])
    
    stress_types = ['σ_xx', 'σ_yy', 'σ_normal']
    correlations = [xx_errors['correlation'], yy_errors['correlation'], normal_errors['correlation']]
    bars = ax_corr.bar(stress_types, correlations, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax_corr.set_title('Correlation between\nCalculated and\nGround Truth Data', fontsize=9)
    ax_corr.set_ylabel('Correlation Coefficient', fontsize=8)
    ax_corr.set_ylim(0, 1.1)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.tick_params(labelsize=7)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        ax_corr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{corr:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Add row labels on the left side
    fig.text(0.135, 0.675, 'Ground Truth', rotation=90, va='center', ha='center', fontsize=9)
    fig.text(0.135, 0.32, 'Calculated', rotation=90, va='center', ha='center', fontsize=9)
    return fig


def calculate_average_stress(stress_xx, stress_yy, mask=None):
    """Calculate average stress over the map."""
    if mask is None:
        mask = ~np.isnan(stress_xx) & ~np.isnan(stress_yy)
    
    # Calculate average of normal stresses (hydrostatic stress)
    avg_stress = (stress_xx + stress_yy) / 2
    
    # Calculate mean over valid region
    if np.any(mask):
        mean_avg_stress = np.mean(avg_stress[mask])
    else:
        mean_avg_stress = 0
    
    return mean_avg_stress


def plot_average_stress_comparison(gt_stress_xx, gt_stress_yy, gt_stress_normal,
                                 calc_stress_xx, calc_stress_yy, calc_stress_normal, mask, y_max=2.5):
    """Create average stress comparison figure: GT vs calculated for stress components."""
    stress_components = ['σ_xx', 'σ_yy', 'σ_normal']
    
    # Calculate average stress values for each component
    gt_values = []
    calc_values = []
    
    for gt_data, calc_data in [(gt_stress_xx, calc_stress_xx), 
                               (gt_stress_yy, calc_stress_yy), 
                               (gt_stress_normal, calc_stress_normal)]:
        if np.any(mask):
            gt_avg = np.mean(gt_data[mask])
            calc_avg = np.mean(calc_data[mask])
        else:
            gt_avg = calc_avg = 0
        gt_values.append(gt_avg)
        calc_values.append(calc_avg)
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))  # DIN A4 compatible
    
    x = np.arange(len(stress_components))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gt_values, width, label='Ground Truth', 
                  color='#1f77b4', alpha=0.7)
    bars2 = ax.bar(x + width/2, calc_values, width, label='Calculated', 
                  color='#ff7f0e', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 1e-5:
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.1 if height > 0 else height * 0.9,
                       f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                       fontsize=6, rotation=45)
    
    ax.set_xlabel('Stress Component', fontsize=8)
    ax.set_ylabel('Average Stress (mN/m)', fontsize=8)
    ax.set_title('Average Stress Comparison\nGround Truth vs Calculated', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(stress_components)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=6)
    
    # Set y-axis limits
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    return fig


def plot_normalized_average_stress(gt_stress_xx, gt_stress_yy, gt_stress_normal,
                                 calc_stress_xx, calc_stress_yy, calc_stress_normal, mask, y_max=1.5):
    """Create normalized average stress plot: calculated/ground_truth ratio for stress components."""
    stress_components = ['σ_xx', 'σ_yy', 'σ_normal']
    
    # Calculate normalized average stresses (calculated/ground_truth)
    normalized_values = []
    for gt_data, calc_data in [(gt_stress_xx, calc_stress_xx), 
                               (gt_stress_yy, calc_stress_yy), 
                               (gt_stress_normal, calc_stress_normal)]:
        if np.any(mask):
            gt_avg = np.mean(gt_data[mask])
            calc_avg = np.mean(calc_data[mask])
            if abs(gt_avg) > 1e-6:
                normalized_values.append(calc_avg / gt_avg)
            else:
                normalized_values.append(0)
        else:
            normalized_values.append(0)
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))  # DIN A4 compatible
    
    # Create bar plot
    bars = ax.bar(stress_components, normalized_values, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, normalized_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height >= 0 else height - 0.02,
               f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
               fontsize=6, fontweight='bold')
    
    # Add horizontal reference line at y=1 (perfect match)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label='Perfect Match (Calc/GT = 1.0)')
    
    ax.set_xlabel('Stress Component', fontsize=8)
    ax.set_ylabel('Normalized Average Stress\n(Calculated / Ground Truth)', fontsize=8)
    ax.set_title('Normalized Average Stress\nCalculated / Ground Truth', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=6)
    
    # Set y-axis limits using the passed parameter
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    return fig


def validate_file_based_msm():
    """Validate MSM using file-based ground truth data."""
    benchmark_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("FILE-BASED MSM VALIDATION")
    print("="*60)
    
    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory {benchmark_dir} not found!")
        return None
    
    # Load ground truth data
    print(f"\nLoading data from: {benchmark_dir}")
    
    try:
        gt_stress_xx, gt_stress_yy = load_ground_truth_stress(str(benchmark_dir))
        print(f"✓ Loaded ground truth stress: σ_xx {gt_stress_xx.shape}, σ_yy {gt_stress_yy.shape}")
        
        trac_x, trac_y = load_ground_truth_traction(str(benchmark_dir))
        print(f"✓ Loaded traction forces: T_x {trac_x.shape}, T_y {trac_y.shape}")
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return None
    
    # Convert ground truth from N/m to mN/m for consistency with MSM service output
    gt_stress_xx = gt_stress_xx * 1000  # Convert N/m to mN/m
    gt_stress_yy = gt_stress_yy * 1000  # Convert N/m to mN/m
    
    # Calculate ground truth normal stress
    gt_stress_normal = (gt_stress_xx + gt_stress_yy) / 2
    
    # Get MSM parameters
    params = get_msm_parameters()
    print(f"\nMSM Parameters:")
    print(f"  Young's modulus: {params.young_modulus} Pa")
    print(f"  Poisson ratio: {params.poisson_ratio_cells}")
    print(f"  Density factor: {params.density_factor}")
    print(f"  Mesh algorithm: {params.mesh_algorithm}")
    print(f"  Use optimization: {params.use_optimization}")
    print(f"  Pixel size: {params.pixel_size} µm")
    
    # Run MSM calculation
    print(f"\nRunning MSM calculation...")
    try:
        calc_stress_xx, calc_stress_yy, calc_stress_normal, condition_number, residual = calculate_msm_stress(trac_x, trac_y, params)
        print(f"✓ MSM calculation completed successfully")
        print(f"  Condition number: {condition_number:.2e}")
        print(f"  Residual norm: {residual:.2e}")
        
    except Exception as e:
        print(f"Error during MSM calculation: {e}")
        return
    
    # Calculate error metrics for each stress component
    print(f"\nCalculating error metrics...")
    
    # Create mask from ground truth data (where data is not NaN)
    gt_mask = ~np.isnan(gt_stress_xx) & ~np.isnan(gt_stress_yy)
    print(f"  Ground truth mask coverage: {np.sum(gt_mask)}/{gt_mask.size} pixels ({100*np.sum(gt_mask)/gt_mask.size:.1f}%)")
    
    xx_metrics = calculate_correlation_metrics(calc_stress_xx, gt_stress_xx, gt_mask)
    yy_metrics = calculate_correlation_metrics(calc_stress_yy, gt_stress_yy, gt_mask)
    normal_metrics = calculate_correlation_metrics(calc_stress_normal, gt_stress_normal, gt_mask)
    
    print(f"✓ Correlation metrics calculated")
    
    # Print correlation summary
    print(f"\nCORRELATION METRICS SUMMARY:")
    print(f"  σ_xx - Correlation: {xx_metrics['correlation']:.3f}")
    print(f"  σ_yy - Correlation: {yy_metrics['correlation']:.3f}")
    print(f"  σ_normal - Correlation: {normal_metrics['correlation']:.3f}")
    
    # Create visualizations
    print(f"\nCreating validation plots...")
    
    # Main validation plot (2x4 grid)
    validation_fig = plot_stress_validation_comparison(
        gt_stress_xx, gt_stress_yy, gt_stress_normal,
        calc_stress_xx, calc_stress_yy, calc_stress_normal,
        xx_metrics, yy_metrics, normal_metrics
    )
    validation_output_path = Path(__file__).parent / "msm_validation_comparison.png"
    validation_fig.savefig(validation_output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved validation comparison: {validation_output_path}")
    plt.close(validation_fig)
    
    # Average stress comparison plot
    avg_stress_fig = plot_average_stress_comparison(
        gt_stress_xx, gt_stress_yy, gt_stress_normal,
        calc_stress_xx, calc_stress_yy, calc_stress_normal, gt_mask
    )
    avg_stress_path = Path(__file__).parent / "average_stress_comparison.png"
    avg_stress_fig.savefig(avg_stress_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved average stress comparison plot: {avg_stress_path}")
    plt.close(avg_stress_fig)
    
    # Normalized average stress plot
    normalized_avg_stress_fig = plot_normalized_average_stress(
        gt_stress_xx, gt_stress_yy, gt_stress_normal,
        calc_stress_xx, calc_stress_yy, calc_stress_normal, gt_mask
    )
    normalized_avg_stress_path = Path(__file__).parent / "normalized_average_stress.png"
    normalized_avg_stress_fig.savefig(normalized_avg_stress_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved normalized average stress plot: {normalized_avg_stress_path}")
    plt.close(normalized_avg_stress_fig)
    
    print(f"\n" + "="*60)
    print("MSM VALIDATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {validation_output_path}")
    print(f"  - {avg_stress_path}")
    print(f"  - {normalized_avg_stress_path}")
    
    # Data statistics summary
    print(f"\nDATA STATISTICS:")
    print(f"  Ground truth stress range: σ_xx [{gt_stress_xx.min():.3f}, {gt_stress_xx.max():.3f}] mN/m")
    print(f"  Calculated stress range: σ_xx [{calc_stress_xx.min():.3f}, {calc_stress_xx.max():.3f}] mN/m")
    print(f"  Traction magnitude max: {np.sqrt(trac_x**2 + trac_y**2).max():.3f} Pa")
    print(f"  Mask coverage: {np.sum(gt_mask)}/{gt_mask.size} pixels ({100*np.sum(gt_mask)/gt_mask.size:.1f}%)")
    
    # Calculate and display average stress values for each component
    print(f"\nAVERAGE STRESS VALUES:")
    for name, gt_data, calc_data in [('σ_xx', gt_stress_xx, calc_stress_xx), 
                                     ('σ_yy', gt_stress_yy, calc_stress_yy), 
                                     ('σ_normal', gt_stress_normal, calc_stress_normal)]:
        if np.any(gt_mask):
            gt_avg = np.mean(gt_data[gt_mask])
            calc_avg = np.mean(calc_data[gt_mask])
            if abs(gt_avg) > 1e-6:
                normalized = calc_avg / gt_avg
                print(f"  {name}: GT={gt_avg:.3f}, Calc={calc_avg:.3f}, Normalized={normalized:.3f}")
            else:
                print(f"  {name}: GT={gt_avg:.3f}, Calc={calc_avg:.3f}, Normalized=N/A")
        else:
            print(f"  {name}: No valid data")
    
    return {
        'gt_stress_xx': gt_stress_xx,
        'gt_stress_yy': gt_stress_yy, 
        'gt_stress_normal': gt_stress_normal,
        'calc_stress_xx': calc_stress_xx,
        'calc_stress_yy': calc_stress_yy,
        'calc_stress_normal': calc_stress_normal,
        'gt_mask': gt_mask,
        'xx_metrics': xx_metrics,
        'yy_metrics': yy_metrics,
        'normal_metrics': normal_metrics
    }


def validate_square_plate_msm():
    """Validate MSM using square plate analytical solution."""
    print("\n" + "="*60)
    print("SQUARE PLATE MSM VALIDATION")
    print("="*60)
    
    # Square plate parameters
    size = 50  # pixels
    edge_traction = 1000  # Pa
    pixelsize = 1e-6  # meters (1 µm)
    buffer = 5  # pixels
    
    print(f"\nSquare plate parameters:")
    print(f"  Size: {size} pixels")
    print(f"  Edge traction: {edge_traction} Pa")
    print(f"  Pixel size: {pixelsize*1e6} µm")
    print(f"  Buffer: {buffer} pixels")
    
    # Create the test problem
    trac_x, trac_y, mask, traction_scale = create_square_plate_problem(
        size=size,
        edge_traction=edge_traction,
        buffer=buffer
    )
    print(f"✓ Created square plate problem: {trac_x.shape}")
    print(f"  Traction scale: {traction_scale} Pa")
    
    # Get MSM parameters for square plate
    params = MSMParameters(
        # Mesh parameters
        density_factor=0.01,
        mesh_algorithm='Frontal-Del.',
        use_optimization=False,
        
        # Material parameters  
        poisson_ratio_cells=0.5,
        young_modulus=1000.0,  # Pa
        
        # Scaling parameters
        pixel_size=pixelsize * 1e6,  # Convert to microns
        downscale_factor=1
    )
    
    print(f"\nMSM Parameters:")
    print(f"  Young's modulus: {params.young_modulus} Pa")
    print(f"  Poisson ratio: {params.poisson_ratio_cells}")
    print(f"  Density factor: {params.density_factor}")
    print(f"  Pixel size: {params.pixel_size} µm")
    
    # Run MSM calculation using the working approach from original square plate file
    print(f"\nRunning MSM calculation...")
    try:
        service = MSMService(params)
        
        # Use the working MSM calculation approach
        # Prepare force field in the format expected by MSMService (H, W, 2)
        force_field = np.stack([trac_x, trac_y], axis=-1)
        
        # Prepare masks (add time dimension as service expects 3D: T, H, W)
        masks = mask[np.newaxis, ...]  # Shape: (1, H, W)
        
        # Calculate stresses using the service - this returns a generator
        stress_generator = service.calculate_stresses(force_field, masks)
        
        # Get the final result from the generator
        try:
            # Process through the generator to get final result
            result = None
            for intermediate_result, frame, total_frames in stress_generator:
                result = intermediate_result
                print(f"    Processing frame {frame}/{total_frames}")
        except StopIteration as e:
            # The final result is returned via StopIteration.value
            result = e.value
        
        if result is None:
            raise ValueError("MSM calculation did not return a valid result")
        
        # Extract stress tensor from MSMResult
        # result.stress_tensor has shape (1, H, W, 2, 2) for single frame
        stress_tensor = result.stress_tensor[0]  # Remove time dimension: (H, W, 2, 2)
        
        calc_stress_xx = stress_tensor[:, :, 0, 0]
        calc_stress_yy = stress_tensor[:, :, 1, 1]
        calc_stress_normal = (calc_stress_xx + calc_stress_yy) / 2
        
        print(f"✓ MSM calculation completed successfully")
        print(f"  Condition number: {result.condition_number:.2e}")
        print(f"  Residual norm: {result.residual:.2e}")
        
    except Exception as e:
        print(f"Error during MSM calculation: {e}")
        return None
    
    # Calculate analytical solution
    print(f"\nCalculating analytical solution...")
    gt_stress_xx, gt_stress_yy, gt_stress_normal = calculate_square_plate_analytical_stress(traction_scale, mask, params)
    print(f"✓ Analytical solution calculated")
    
    # Calculate error metrics for each stress component
    print(f"\nCalculating error metrics...")
    # For square plate, calculate correlation over full domain (including borders)
    # to capture the transition from 0 outside to uniform stress inside
    xx_metrics = calculate_correlation_metrics(calc_stress_xx, gt_stress_xx, mask=None)
    yy_metrics = calculate_correlation_metrics(calc_stress_yy, gt_stress_yy, mask=None)
    normal_metrics = calculate_correlation_metrics(calc_stress_normal, gt_stress_normal, mask=None)
    
    print(f"✓ Correlation metrics calculated")
    
    # Print correlation summary
    print(f"\nCORRELATION METRICS SUMMARY:")
    print(f"  σ_xx - Correlation: {xx_metrics['correlation']:.3f}")
    print(f"  σ_yy - Correlation: {yy_metrics['correlation']:.3f}")
    print(f"  σ_normal - Correlation: {normal_metrics['correlation']:.3f}")
    
    # Create visualizations with prefix for square plate
    print(f"\nCreating validation plots...")
    
    # Main validation plot (2x4 grid)
    validation_fig = plot_stress_validation_comparison(
        gt_stress_xx, gt_stress_yy, gt_stress_normal,
        calc_stress_xx, calc_stress_yy, calc_stress_normal,
        xx_metrics, yy_metrics, normal_metrics, vmax=1.5
    )
    validation_output_path = Path(__file__).parent / "square_plate_validation_comparison.png"
    validation_fig.savefig(validation_output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved validation comparison: {validation_output_path}")
    plt.close(validation_fig)
    
    # Average stress comparison plot
    avg_stress_fig = plot_average_stress_comparison(
        gt_stress_xx, gt_stress_yy, gt_stress_normal,
        calc_stress_xx, calc_stress_yy, calc_stress_normal, mask, y_max=1.5
    )
    avg_stress_path = Path(__file__).parent / "square_plate_average_stress_comparison.png"
    avg_stress_fig.savefig(avg_stress_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved average stress comparison plot: {avg_stress_path}")
    plt.close(avg_stress_fig)
    
    # Normalized average stress plot
    normalized_avg_stress_fig = plot_normalized_average_stress(
        gt_stress_xx, gt_stress_yy, gt_stress_normal,
        calc_stress_xx, calc_stress_yy, calc_stress_normal, mask, y_max=1.5
    )
    normalized_avg_stress_path = Path(__file__).parent / "square_plate_normalized_average_stress.png"
    normalized_avg_stress_fig.savefig(normalized_avg_stress_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved normalized average stress plot: {normalized_avg_stress_path}")
    plt.close(normalized_avg_stress_fig)
    
    print(f"\nGenerated files:")
    print(f"  - {validation_output_path}")
    print(f"  - {avg_stress_path}")
    print(f"  - {normalized_avg_stress_path}")
    
    # Data statistics summary
    print(f"\nDATA STATISTICS:")
    print(f"  Analytical stress range: σ_xx [{gt_stress_xx.min():.3f}, {gt_stress_xx.max():.3f}] mN/m")
    print(f"  Calculated stress range: σ_xx [{calc_stress_xx.min():.3f}, {calc_stress_xx.max():.3f}] mN/m")
    print(f"  Traction magnitude max: {np.sqrt(trac_x**2 + trac_y**2).max():.3f} Pa")
    print(f"  Mask coverage: {np.sum(mask)}/{mask.size} pixels ({100*np.sum(mask)/mask.size:.1f}%)")
    
    # Calculate and display average stress values for each component
    print(f"\nAVERAGE STRESS VALUES:")
    for name, gt_data, calc_data in [('σ_xx', gt_stress_xx, calc_stress_xx), 
                                     ('σ_yy', gt_stress_yy, calc_stress_yy), 
                                     ('σ_normal', gt_stress_normal, calc_stress_normal)]:
        if np.any(mask):
            gt_avg = np.mean(gt_data[mask])
            calc_avg = np.mean(calc_data[mask])
            if abs(gt_avg) > 1e-6:
                normalized = calc_avg / gt_avg
                print(f"  {name}: GT={gt_avg:.3f}, Calc={calc_avg:.3f}, Normalized={normalized:.3f}")
            else:
                print(f"  {name}: GT={gt_avg:.3f}, Calc={calc_avg:.3f}, Normalized=N/A")
        else:
            print(f"  {name}: No valid data")
    
    return {
        'gt_stress_xx': gt_stress_xx,
        'gt_stress_yy': gt_stress_yy, 
        'gt_stress_normal': gt_stress_normal,
        'calc_stress_xx': calc_stress_xx,
        'calc_stress_yy': calc_stress_yy,
        'calc_stress_normal': calc_stress_normal,
        'mask': mask,
        'xx_metrics': xx_metrics,
        'yy_metrics': yy_metrics,
        'normal_metrics': normal_metrics
    }


def main():
    """Main MSM validation function - runs both file-based and square plate validations."""
    print("="*60)
    print("MSM STRESS FIELD VALIDATION SUITE")
    print("="*60)
    
    # Run file-based validation
    file_results = validate_file_based_msm()
    
    # Run square plate validation
    square_results = validate_square_plate_msm()
    
    print("\n" + "="*60)
    print("ALL VALIDATIONS COMPLETE")
    print("="*60)
    
    if file_results:
        print("\n✓ File-based validation completed successfully")
        print(f"  File-based correlations: σ_xx={file_results['xx_metrics']['correlation']:.3f}, "
              f"σ_yy={file_results['yy_metrics']['correlation']:.3f}, "
              f"σ_normal={file_results['normal_metrics']['correlation']:.3f}")
    else:
        print("\n✗ File-based validation failed (files not found)")
    
    if square_results:
        print("✓ Square plate validation completed successfully")
        print(f"  Square plate correlations: σ_xx={square_results['xx_metrics']['correlation']:.3f}, "
              f"σ_yy={square_results['yy_metrics']['correlation']:.3f}, "
              f"σ_normal={square_results['normal_metrics']['correlation']:.3f}")
    else:
        print("✗ Square plate validation failed")


if __name__ == "__main__":
    main()
