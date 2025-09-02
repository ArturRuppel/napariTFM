#!/usr/bin/env python3
"""
Validation script for displacement analysis.

This script compares calculated displacement fields from the backend
with ground truth displacement fields for low, mid, and high scenarios.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# Add the parent directory to path to import napariTFM modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from napariTFM.backend.displacement_analysis import DisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters


def load_tif_image(filepath):
    """Load a TIF image as numpy array."""
    return np.array(Image.open(filepath))


def load_ground_truth_displacement(folder_path, pixel_size_um=0.1):
    """Load ground truth displacement components from .npy files and convert to pixels."""
    disp_x = np.load(os.path.join(folder_path, 'displacement_x.npy'))
    disp_y = np.load(os.path.join(folder_path, 'displacement_y.npy'))
    
    # Convert from microns to pixels
    disp_x_pixels = disp_x / pixel_size_um
    disp_y_pixels = disp_y / pixel_size_um
    
    return np.stack([disp_x_pixels, disp_y_pixels], axis=-1)


def get_scenario_parameters(scenario_name):
    """Get displacement analysis parameters for each scenario."""
    # Default parameters
    base_params = DisplacementParameters()
    
    # Scenario-specific parameter modifications
    scenario_configs = {
        'low': {
            'tau': 0.25,
            'lambda_': 0.1,
            'theta': 0.3,
            'nscales': 3,
            'warps': 3,
            'epsilon': 0.01,
            'inner_iterations': 15,
            'outer_iterations': 5
        },
        'mid': {
            'tau': 0.2,
            'lambda_': 0.08,
            'theta': 0.25,
            'nscales': 4,
            'warps': 4,
            'epsilon': 0.008,
            'inner_iterations': 18,
            'outer_iterations': 6
        },
        'high': {
            'tau': 0.15,
            'lambda_': 0.06,
            'theta': 0.2,
            'nscales': 5,
            'warps': 5,
            'epsilon': 0.005,
            'inner_iterations': 20,
            'outer_iterations': 8
        }
    }
    
    # Update parameters for the specific scenario
    if scenario_name in scenario_configs:
        config = scenario_configs[scenario_name]
        for param, value in config.items():
            setattr(base_params, param, value)
    
    return base_params


def calculate_displacement_field(reference_img, deformed_img, params=None):
    """Calculate displacement field using DisplacementAnalyzer with custom parameters."""
    analyzer = DisplacementAnalyzer(params)
    flow = analyzer.calculate_flow(reference_img, deformed_img)
    return flow


def plot_combined_displacement_comparison(all_results):
    """Plot all scenarios in one figure with magnitude+vector plots only."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Displacement Field Validation: Calculated vs Ground Truth', fontsize=16)
    
    scenarios = ['low', 'mid', 'high']
    
    for i, scenario in enumerate(scenarios):
        if scenario not in all_results:
            continue
            
        calculated_flow = all_results[scenario]['calculated']
        ground_truth = all_results[scenario]['ground_truth']
        
        # Calculate magnitudes
        calc_magnitude = np.sqrt(calculated_flow[:,:,0]**2 + calculated_flow[:,:,1]**2)
        gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
        
        # Create coordinate grids for vector plotting (subsample for visibility)
        h, w = calculated_flow.shape[:2]
        step = max(h//20, w//20, 10)  # Adjust step size based on image size
        y, x = np.mgrid[0:h:step, 0:w:step]
        
        # Top row: Calculated displacement fields
        im_calc = axes[0, i].imshow(calc_magnitude, cmap='viridis')
        axes[0, i].quiver(x, y, calculated_flow[::step, ::step, 0], 
                         calculated_flow[::step, ::step, 1], 
                         color='white', scale_units='xy', scale=0.5, alpha=0.8)
        axes[0, i].set_title(f'Calculated: {scenario.upper()}')
        axes[0, i].set_xlabel('X (pixels)')
        axes[0, i].set_ylabel('Y (pixels)')
        
        # Create colorbar with same height as plot
        divider_calc = make_axes_locatable(axes[0, i])
        cax_calc = divider_calc.append_axes("right", size="5%", pad=0.05)
        cbar_calc = plt.colorbar(im_calc, cax=cax_calc, label='Magnitude (pixels)')
        
        # Bottom row: Ground truth displacement fields
        im_gt = axes[1, i].imshow(gt_magnitude, cmap='viridis')
        axes[1, i].quiver(x, y, ground_truth[::step, ::step, 0], 
                         ground_truth[::step, ::step, 1], 
                         color='white', scale_units='xy', scale=0.5, alpha=0.8)
        axes[1, i].set_title(f'Ground Truth: {scenario.upper()}')
        axes[1, i].set_xlabel('X (pixels)')
        axes[1, i].set_ylabel('Y (pixels)')
        
        # Create colorbar with same height as plot
        divider_gt = make_axes_locatable(axes[1, i])
        cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
        cbar_gt = plt.colorbar(im_gt, cax=cax_gt, label='Magnitude (pixels)')
    
    plt.tight_layout()
    return fig


def calculate_error_metrics(calculated_flow, ground_truth):
    """Calculate normalized error metrics between calculated and ground truth displacement fields."""
    # Calculate differences
    diff_x = calculated_flow[:,:,0] - ground_truth[:,:,0]
    diff_y = calculated_flow[:,:,1] - ground_truth[:,:,1]
    
    # Calculate maximum displacement magnitude in ground truth for normalization
    gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
    max_gt_displacement = np.max(gt_magnitude)
    
    # Root mean square error (normalized)
    rmse_total = np.sqrt(np.mean(diff_x**2 + diff_y**2))
    normalized_rmse = rmse_total / max_gt_displacement if max_gt_displacement > 0 else 0
    
    return {
        'normalized_rmse': normalized_rmse,
        'max_gt_displacement': max_gt_displacement
    }


def plot_error_comparison(all_results):
    """Create bar plot comparing normalized error metrics across scenarios."""
    scenarios = list(all_results.keys())
    
    # Extract normalized error metrics
    normalized_errors = [all_results[s]['errors']['normalized_rmse'] for s in scenarios]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create bar plot
    colors = ['blue', 'orange', 'green']
    bars = ax.bar(scenarios, normalized_errors, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, error) in enumerate(zip(bars, normalized_errors)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
               f'{error:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Normalized RMSE (fraction of max GT displacement)', fontsize=12)
    ax.set_title('Normalized Displacement Error Across Scenarios', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(normalized_errors) * 1.2)
    
    # Convert scenario names to uppercase for labels
    ax.set_xticklabels([s.upper() for s in scenarios])
    
    plt.tight_layout()
    return fig


def validate_scenario(scenario_folder):
    """Validate displacement analysis for a single scenario."""
    scenario_name = os.path.basename(scenario_folder)
    print(f"\nValidating scenario: {scenario_name}")
    
    # Get scenario-specific parameters
    params = get_scenario_parameters(scenario_name)
    print(f"  Using parameters: tau={params.tau}, lambda_={params.lambda_}, "
          f"nscales={params.nscales}, warps={params.warps}")
    
    # Load images
    reference_path = os.path.join(scenario_folder, 'reference.tif')
    deformed_path = os.path.join(scenario_folder, 'deformed.tif')
    
    reference_img = load_tif_image(reference_path)
    deformed_img = load_tif_image(deformed_path)
    
    print(f"  Reference image shape: {reference_img.shape}")
    print(f"  Deformed image shape: {deformed_img.shape}")
    
    # Calculate displacement field with custom parameters
    calculated_flow = calculate_displacement_field(reference_img, deformed_img, params)
    print(f"  Calculated flow shape: {calculated_flow.shape}")
    
    # Load ground truth
    ground_truth = load_ground_truth_displacement(scenario_folder)
    print(f"  Ground truth shape: {ground_truth.shape}")
    
    # Calculate error metrics
    errors = calculate_error_metrics(calculated_flow, ground_truth)
    
    print("  Error Metrics:")
    print(f"    Normalized RMSE: {errors['normalized_rmse']:.4f} "
          f"(fraction of max GT displacement: {errors['max_gt_displacement']:.3f})")
    
    return calculated_flow, ground_truth, errors


def main():
    """Main validation function."""
    base_dir = Path(__file__).parent / 'benchmark_displacements_forces'
    scenarios = ['low', 'mid', 'high']
    
    all_results = {}
    
    print("Starting displacement field validation...")
    
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
    
    # Create and save combined displacement plot
    if all_results:
        print("\nCreating combined displacement comparison plot...")
        combined_fig = plot_combined_displacement_comparison(all_results)
        combined_output_path = Path(__file__).parent / "displacement_validation_combined.png"
        combined_fig.savefig(combined_output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved combined plot: {combined_output_path}")
        plt.close(combined_fig)
        
        # Create and save error comparison plot
        print("Creating error comparison plot...")
        error_fig = plot_error_comparison(all_results)
        error_output_path = Path(__file__).parent / "error_comparison.png"
        error_fig.savefig(error_output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved error plot: {error_output_path}")
        plt.close(error_fig)
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    for scenario, results in all_results.items():
        errors = results['errors']
        print(f"\n{scenario.upper()} scenario:")
        print(f"  Normalized RMSE: {errors['normalized_rmse']:.4f}")
        print(f"  Max GT displacement: {errors['max_gt_displacement']:.3f} pixels")


if __name__ == "__main__":
    main()