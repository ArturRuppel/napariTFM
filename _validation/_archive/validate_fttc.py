#!/usr/bin/env python3
"""
Validation script for FTTC (Fourier Transform Traction Cytometry) analysis.

This script compares calculated traction forces from displacement fields
with ground truth traction forces for low, mid, and high scenarios.
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

from napariTFM.backend.fttc import FTTC
from napariTFM.backend.parameter_dataclasses import FTTCParameters


def load_displacement_data(folder_path):
    """Load displacement components from .npy files."""
    disp_x = np.load(os.path.join(folder_path, 'displacement_x.npy'))
    disp_y = np.load(os.path.join(folder_path, 'displacement_y.npy'))
    return disp_x, disp_y


def load_ground_truth_traction(folder_path):
    """Load ground truth traction components from .npy files."""
    trac_x = np.load(os.path.join(folder_path, 'traction_x.npy'))
    trac_y = np.load(os.path.join(folder_path, 'traction_y.npy'))
    return trac_x, trac_y


def get_scenario_parameters(scenario_name):
    """Get FTTC parameters for each scenario."""
    base_params = FTTCParameters()
    
    # Scenario-specific parameter modifications
    scenario_configs = {
        'low': {
            'young_modulus': 20000,  # Pa
            'poisson_ratio_substrate': 0.5,
            'lanczos_exp': 1,
            'auto_gcv': True,
            'pixel_size': 0.1,  # µm
            'downscale_factor': 1
        },
        'mid': {
            'young_modulus': 20000,  # Pa
            'poisson_ratio_substrate': 0.5,
            'lanczos_exp': 1,
            'auto_gcv': True,
            'pixel_size': 0.1,  # µm
            'downscale_factor': 1
        },
        'high': {
            'young_modulus': 20000,  # Pa
            'poisson_ratio_substrate': 0.5,
            'lanczos_exp': 1,
            'auto_gcv': True,
            'pixel_size': 0.1,  # µm
            'downscale_factor': 1
        }
    }
    
    # Update parameters for the specific scenario
    if scenario_name in scenario_configs:
        config = scenario_configs[scenario_name]
        for param, value in config.items():
            setattr(base_params, param, value)
    
    return base_params


def calculate_traction_field(disp_x, disp_y, params):
    """Calculate traction field using FTTC with given parameters."""
    fttc = FTTC(params)
    
    # Stack displacements in the format expected by FTTC (H, W, 2)
    displacements = np.stack([disp_x, disp_y], axis=-1)
    
    # Calculate traction forces - use None for regularization if auto_gcv is enabled
    regularization = None if params.auto_gcv else params.regularization
    traction_coords, traction_values = fttc.calculate_traction(
        displacements, 
        params.pixel_size,
        params.downscale_factor,
        regularization
    )
    
    return traction_coords, traction_values


def plot_combined_traction_comparison(all_results):
    """Plot all scenarios in one figure with magnitude+vector plots only."""
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    fig.suptitle('Traction Force Validation: Calculated vs Ground Truth', fontsize=11, color='black')
    
    scenarios = ['low', 'mid', 'high']
    column_titles = ['Low Force', 'Medium Force', 'High Force']
    
    for i, scenario in enumerate(scenarios):
        if scenario not in all_results:
            continue
            
        calculated_trac = all_results[scenario]['calculated']
        ground_truth = all_results[scenario]['ground_truth']
        
        # Calculate magnitudes
        calc_magnitude = np.sqrt(calculated_trac[:,:,0]**2 + calculated_trac[:,:,1]**2)
        gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
        
        # Determine common colorbar scale (use max of both fields)
        vmax = max(np.max(calc_magnitude), np.max(gt_magnitude))
        vmin = 0  # Magnitude is always >= 0
        
        # Create coordinate grids for vector plotting (subsample for visibility)
        h, w = calculated_trac.shape[:2]
        step = max(h//30, w//30, 10)  # Adjust step size based on image size
        y, x = np.mgrid[0:h:step, 0:w:step]
        
        # Top row: Calculated traction fields
        im_calc = axes[0, i].imshow(calc_magnitude, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, i].quiver(x, y, calculated_trac[::step, ::step, 0], 
                         -calculated_trac[::step, ::step, 1], 
                         color='white', scale_units='xy', scale=0.01*vmax, alpha=0.4)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Create colorbar with same height as plot
        divider_calc = make_axes_locatable(axes[0, i])
        cax_calc = divider_calc.append_axes("right", size="5%", pad=0.05)
        cbar_calc = plt.colorbar(im_calc, cax=cax_calc)
        cbar_calc.set_label('Magnitude (Pa)', fontsize=8, color='black')
        cbar_calc.ax.tick_params(labelsize=6, colors='black')
        
        # Bottom row: Ground truth traction fields
        im_gt = axes[1, i].imshow(gt_magnitude, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, i].quiver(x, y, ground_truth[::step, ::step, 0], 
                         -ground_truth[::step, ::step, 1], 
                         color='white', scale_units='xy', scale=0.01*vmax, alpha=0.4)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        
        # Create colorbar with same height as plot (same scale as calculated)
        divider_gt = make_axes_locatable(axes[1, i])
        cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
        cbar_gt = plt.colorbar(im_gt, cax=cax_gt)
        cbar_gt.set_label('Magnitude (Pa)', fontsize=8, color='black')
        cbar_gt.ax.tick_params(labelsize=6, colors='black')
        
        # Add column title to top row
        if i < len(column_titles):
            axes[0, i].set_title(column_titles[i], fontsize=10, color='black', pad=20)
    
    # Add row titles
    axes[0, 0].text(-0.1, 0.5, 'Calculated', transform=axes[0, 0].transAxes, 
                    fontsize=10, color='black', rotation=90, va='center', ha='right')
    axes[1, 0].text(-0.1, 0.5, 'Ground Truth', transform=axes[1, 0].transAxes, 
                    fontsize=10, color='black', rotation=90, va='center', ha='right')
    
    # Set tick label properties for all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=6, colors='black')
    
    plt.tight_layout()
    return fig


def calculate_error_metrics(calculated_trac, ground_truth):
    """Calculate normalized error metrics between calculated and ground truth traction fields."""
    # Calculate differences
    diff_x = calculated_trac[:,:,0] - ground_truth[:,:,0]
    diff_y = calculated_trac[:,:,1] - ground_truth[:,:,1]
    
    # Calculate maximum traction magnitude in ground truth for normalization
    gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
    max_gt_traction = np.max(gt_magnitude)
    
    # Root mean square error (normalized)
    rmse_total = np.sqrt(np.mean(diff_x**2 + diff_y**2))
    normalized_rmse = rmse_total / max_gt_traction if max_gt_traction > 0 else 0
    
    return {
        'normalized_rmse': normalized_rmse,
        'max_gt_traction': max_gt_traction
    }


def plot_error_comparison(all_results):
    """Create bar plot comparing normalized error metrics across scenarios."""
    scenarios = list(all_results.keys())
    
    # Extract normalized error metrics
    normalized_errors = [all_results[s]['errors']['normalized_rmse'] for s in scenarios]
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    
    # Create bar plot with tab10 colors
    colors = plt.cm.tab10.colors[:3]  # First three colors of tab10
    bars = ax.bar(scenarios, normalized_errors, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, error) in enumerate(zip(bars, normalized_errors)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
               f'{error:.3f}', ha='center', va='bottom', fontsize=6, color='black')
    
    ax.set_xlabel('Scenario', fontsize=8, color='black')
    ax.set_ylabel('Normalized RMSE', fontsize=8, color='black')
    ax.set_title('Normalized Traction Error Across Scenarios', fontsize=10, color='black')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(normalized_errors) * 1.2)
    
    # Set tick label properties
    ax.tick_params(labelsize=6, colors='black')
    
    plt.tight_layout()
    return fig


def validate_scenario(scenario_folder):
    """Validate FTTC analysis for a single scenario."""
    scenario_name = os.path.basename(scenario_folder)
    print(f"\nValidating scenario: {scenario_name}")
    
    # Get scenario-specific parameters
    params = get_scenario_parameters(scenario_name)
    regularization_info = "auto-GCV" if params.auto_gcv else f"{params.regularization}"
    print(f"  Using parameters: E={params.young_modulus} Pa, nu={params.poisson_ratio_substrate}, "
          f"regularization={regularization_info}, pixel_size={params.pixel_size} µm")
    
    # Load displacement data (input for FTTC)
    disp_x, disp_y = load_displacement_data(scenario_folder)
    print(f"  Displacement field shapes: x={disp_x.shape}, y={disp_y.shape}")
    
    # Calculate traction field using FTTC
    _, calculated_trac = calculate_traction_field(disp_x, disp_y, params)
    print(f"  Calculated traction field shape: {calculated_trac.shape}")
    
    # Transpose to match ground truth format (H, W, 2)
    if calculated_trac.shape[0] == 2:
        calculated_trac = np.transpose(calculated_trac, (1, 2, 0))
        print(f"  Transposed traction field shape: {calculated_trac.shape}")
    
    # Load ground truth traction
    gt_trac_x, gt_trac_y = load_ground_truth_traction(scenario_folder)
    ground_truth = np.stack([gt_trac_x, gt_trac_y], axis=-1)
    print(f"  Ground truth traction shape: {ground_truth.shape}")
    
    # Calculate error metrics
    errors = calculate_error_metrics(calculated_trac, ground_truth)
    
    print("  Error Metrics:")
    print(f"    Normalized RMSE: {errors['normalized_rmse']:.4f} "
          f"(fraction of max GT traction: {errors['max_gt_traction']:.3f} Pa)")
    
    return calculated_trac, ground_truth, errors


def main():
    """Main validation function."""
    base_dir = Path(__file__).parent / 'benchmark_displacements_forces'
    scenarios = ['low', 'mid', 'high']
    
    all_results = {}
    
    print("Starting FTTC traction force validation...")
    
    for scenario in scenarios:
        scenario_path = base_dir / scenario
        if scenario_path.exists():
            calculated, ground_truth, errors = validate_scenario(str(scenario_path))
            all_results[scenario] = {
                'calculated': calculated,
                'ground_truth': ground_truth,
                'errors': errors
            }
        else:
            print(f"Warning: Scenario folder {scenario_path} not found")
    
    # Create and save combined traction plot
    if all_results:
        print("\nCreating combined traction comparison plot...")
        combined_fig = plot_combined_traction_comparison(all_results)
        combined_output_path = Path(__file__).parent / "fttc_validation_combined.png"
        combined_fig.savefig(combined_output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved combined plot: {combined_output_path}")
        plt.close(combined_fig)
        
        # Create and save error comparison plot
        print("Creating error comparison plot...")
        error_fig = plot_error_comparison(all_results)
        error_output_path = Path(__file__).parent / "fttc_error_comparison.png"
        error_fig.savefig(error_output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved error plot: {error_output_path}")
        plt.close(error_fig)
    
    # Summary
    print("\n" + "="*50)
    print("FTTC VALIDATION SUMMARY")
    print("="*50)
    
    for scenario, results in all_results.items():
        errors = results['errors']
        print(f"\n{scenario.upper()} scenario:")
        print(f"  Normalized RMSE: {errors['normalized_rmse']:.4f}")
        print(f"  Max GT traction: {errors['max_gt_traction']:.3f} Pa")


if __name__ == "__main__":
    main()
