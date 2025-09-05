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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add the parent directory to path to import napariTFM modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from napariTFM.services.msm_service import MSMService
from napariTFM.backend.parameter_dataclasses import MSMParameters


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
        density_factor=0.005,  # Good balance between accuracy and computation time
        mesh_algorithm='Frontal-Del.',
        use_optimization=True,
        
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

def calculate_error_metrics(calculated, ground_truth, mask=None):
    """Calculate error metrics between calculated and ground truth stress fields."""
    if mask is None:
        mask = np.ones_like(calculated, dtype=bool)
    
    # Apply mask and filter out NaN values
    valid_mask = mask & ~np.isnan(calculated) & ~np.isnan(ground_truth)
    
    if not np.any(valid_mask):
        return {'rmse': 0, 'max_error': 0, 'correlation': 0, 'relative_error': 0}
    
    calc_valid = calculated[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    # Root mean square error
    rmse = np.sqrt(np.mean((calc_valid - gt_valid)**2))
    
    # Maximum error
    max_error = np.max(np.abs(calc_valid - gt_valid))
    
    # Correlation coefficient
    if np.std(calc_valid) > 0 and np.std(gt_valid) > 0:
        correlation = np.corrcoef(calc_valid, gt_valid)[0, 1]
    else:
        correlation = 0
    
    # Relative error (RMSE normalized by ground truth standard deviation)
    if np.std(gt_valid) > 0:
        relative_error = rmse / np.std(gt_valid)
    else:
        relative_error = 0
    
    return {
        'rmse': rmse,
        'max_error': max_error,
        'correlation': correlation,
        'relative_error': relative_error
    }


def plot_stress_validation_comparison(gt_stress_xx, gt_stress_yy, gt_stress_normal,
                                    calc_stress_xx, calc_stress_yy, calc_stress_normal, 
                                    xx_errors, yy_errors, normal_errors):
    """Plot 2x3 stress validation: Ground Truth vs Calculated for σ_xx, σ_yy, σ_normal."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle('MSM Stress Field Validation: Calculated vs Ground Truth', fontsize=11, color='black', y=0.95)
    
    # Data arrays and titles
    gt_data = [gt_stress_xx, gt_stress_yy, gt_stress_normal]
    calc_data = [calc_stress_xx, calc_stress_yy, calc_stress_normal]
    titles = ['σ_xx', 'σ_yy', 'σ_normal']
    errors = [xx_errors, yy_errors, normal_errors]
    
    # Determine common colorbar scales for each stress type
    for i in range(3):
        vmax = 5
        vmin = -vmax
        
        # Top row: Ground truth
        im_gt = axes[0, i].imshow(gt_data[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].set_title(f'{titles[i]} (Ground Truth)', fontsize=10, pad=5)
        
        divider_gt = make_axes_locatable(axes[0, i])
        cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
        cbar_gt = plt.colorbar(im_gt, cax=cax_gt)
        cbar_gt.set_label('Stress (mN/m)', fontsize=8)
        cbar_gt.ax.tick_params(labelsize=6)
        
        # Bottom row: Calculated
        im_calc = axes[1, i].imshow(calc_data[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        
        # Add error metrics to title
        error = errors[i]
        axes[1, i].set_title(f'{titles[i]} (Calculated)\nRMSE: {error["relative_error"]*100:.1f}% | '
                           f'Corr: {error["correlation"]:.3f}', fontsize=9, pad=5)
        
        divider_calc = make_axes_locatable(axes[1, i])
        cax_calc = divider_calc.append_axes("right", size="5%", pad=0.05)
        cbar_calc = plt.colorbar(im_calc, cax=cax_calc)
        cbar_calc.set_label('Stress (mN/m)', fontsize=8)
        cbar_calc.ax.tick_params(labelsize=6)
    
    # Add row labels
    axes[0, 0].text(-0.15, 0.5, 'Ground Truth', transform=axes[0, 0].transAxes, 
                    fontsize=10, color='black', rotation=90, va='center', ha='center')
    axes[1, 0].text(-0.15, 0.5, 'Calculated', transform=axes[1, 0].transAxes, 
                    fontsize=10, color='black', rotation=90, va='center', ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_error_comparison(xx_errors, yy_errors, normal_errors):
    """Create bar plot comparing error metrics across stress components."""
    stress_types = ['σ_xx', 'σ_yy', 'σ_normal']
    relative_errors = [xx_errors['relative_error'] * 100, 
                      yy_errors['relative_error'] * 100,
                      normal_errors['relative_error'] * 100]
    correlations = [xx_errors['correlation'], 
                   yy_errors['correlation'],
                   normal_errors['correlation']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle('MSM Validation Error Metrics', fontsize=11, color='black', y=0.95)
    
    # Plot 1: Relative RMSE
    colors = plt.cm.Set1.colors[:3]
    bars1 = ax1.bar(stress_types, relative_errors, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, error in zip(bars1, relative_errors):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{error:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Relative RMSE (%)', fontsize=9)
    ax1.set_title('Normalized Error', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(relative_errors) * 1.2)
    
    # Plot 2: Correlation
    bars2 = ax2.bar(stress_types, correlations, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, corr in zip(bars2, correlations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('Correlation Coefficient', fontsize=9)
    ax2.set_title('Correlation with Ground Truth', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    """Main MSM validation function."""
    benchmark_dir = Path(__file__).parent
    
    print("="*60)
    print("MSM STRESS FIELD VALIDATION")
    print("="*60)
    
    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory {benchmark_dir} not found!")
        return
    
    # Load ground truth data
    print(f"\nLoading data from: {benchmark_dir}")
    
    try:
        gt_stress_xx, gt_stress_yy = load_ground_truth_stress(str(benchmark_dir))
        print(f"✓ Loaded ground truth stress: σ_xx {gt_stress_xx.shape}, σ_yy {gt_stress_yy.shape}")
        
        trac_x, trac_y = load_ground_truth_traction(str(benchmark_dir))
        print(f"✓ Loaded traction forces: T_x {trac_x.shape}, T_y {trac_y.shape}")
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return
    
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
    
    xx_errors = calculate_error_metrics(calc_stress_xx, gt_stress_xx, gt_mask)
    yy_errors = calculate_error_metrics(calc_stress_yy, gt_stress_yy, gt_mask)
    normal_errors = calculate_error_metrics(calc_stress_normal, gt_stress_normal, gt_mask)
    
    print(f"✓ Error metrics calculated")
    
    # Print error summary
    print(f"\nERROR METRICS SUMMARY:")
    print(f"  σ_xx - RMSE: {xx_errors['relative_error']*100:.2f}%, Correlation: {xx_errors['correlation']:.3f}")
    print(f"  σ_yy - RMSE: {yy_errors['relative_error']*100:.2f}%, Correlation: {yy_errors['correlation']:.3f}")
    print(f"  σ_normal - RMSE: {normal_errors['relative_error']*100:.2f}%, Correlation: {normal_errors['correlation']:.3f}")
    
    # Create visualizations
    print(f"\nCreating validation plots...")
    
    # Main validation plot (2x3 grid)
    validation_fig = plot_stress_validation_comparison(
        gt_stress_xx, gt_stress_yy, gt_stress_normal,
        calc_stress_xx, calc_stress_yy, calc_stress_normal,
        xx_errors, yy_errors, normal_errors
    )
    validation_output_path = Path(__file__).parent / "msm_validation_comparison.png"
    validation_fig.savefig(validation_output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved validation comparison: {validation_output_path}")
    plt.close(validation_fig)
    
    # Error metrics plot
    error_fig = plot_error_comparison(xx_errors, yy_errors, normal_errors)
    error_output_path = Path(__file__).parent / "msm_error_metrics.png"
    error_fig.savefig(error_output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error metrics plot: {error_output_path}")
    plt.close(error_fig)
    
    print(f"\n" + "="*60)
    print("MSM VALIDATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {validation_output_path}")
    print(f"  - {error_output_path}")
    
    # Data statistics summary
    print(f"\nDATA STATISTICS:")
    print(f"  Ground truth stress range: σ_xx [{gt_stress_xx.min():.3f}, {gt_stress_xx.max():.3f}] mN/m")
    print(f"  Calculated stress range: σ_xx [{calc_stress_xx.min():.3f}, {calc_stress_xx.max():.3f}] mN/m")
    print(f"  Traction magnitude max: {np.sqrt(trac_x**2 + trac_y**2).max():.3f} Pa")
    print(f"  Mask coverage: {np.sum(gt_mask)}/{gt_mask.size} pixels ({100*np.sum(gt_mask)/gt_mask.size:.1f}%)")


if __name__ == "__main__":
    main()
