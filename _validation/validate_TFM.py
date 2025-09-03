#!/usr/bin/env python3
"""
Comprehensive TFM validation script.

This script validates both displacement analysis and FTTC (Fourier Transform Traction Cytometry)
by comparing calculated results with ground truth data for low, mid, and high scenarios.
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
from napariTFM.backend.preprocessing import ImageProcessor
from napariTFM.backend.fttc import FTTC
from napariTFM.backend.parameter_dataclasses import FTTCParameters


def load_tif_image(filepath):
    """Load a TIF image as numpy array."""
    return np.array(Image.open(filepath))


def preprocess_image(image):
    """Apply preprocessing steps to the image."""
    processor = ImageProcessor()
    
    # Apply intensity scaling with thresholds 85 and 99.9
    processed, _ = processor.apply_intensity_scaling(image, 80, 99.9)
    
    # Apply Gaussian blur with sigma = 1
    processed = processor.apply_gaussian_filter(processed, sigma=1)
    
    # Apply rolling ball background subtraction with radius = 0 (effectively no-op)
    processed = processor.apply_rolling_ball(processed, radius=0)
    
    return processed


def load_ground_truth_displacement(folder_path, pixel_size_um=0.1):
    """Load ground truth displacement components from .npy files and convert to pixels."""
    disp_x = np.load(os.path.join(folder_path, 'displacement_x.npy'))
    disp_y = np.load(os.path.join(folder_path, 'displacement_y.npy'))
    
    # Convert from microns to pixels
    disp_x_pixels = disp_x / pixel_size_um
    disp_y_pixels = disp_y / pixel_size_um
    
    return np.stack([disp_x_pixels, disp_y_pixels], axis=-1)


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


def get_displacement_parameters(scenario_name):
    """Get displacement analysis parameters for each scenario."""
    # Default parameters
    base_params = DisplacementParameters()
    
    # Scenario-specific parameter modifications
    scenario_configs = {
        'low': {
            'tau': 0.15,
            'lambda_': 0.1,
            'theta': 0.1,
            'nscales': 10,
            'warps': 10,
            'epsilon': 0.01,
            'inner_iterations': 20,
            'outer_iterations': 10
        },
        'mid': {
            'tau': 0.15,
            'lambda_': 0.1,
            'theta': 0.1,
            'nscales': 10,
            'warps': 10,
            'epsilon': 0.01,
            'inner_iterations': 20,
            'outer_iterations': 10
        },
        'high': {
            'tau': 0.15,
            'lambda_': 0.1,
            'theta': 0.1,
            'nscales': 10,
            'warps': 10,
            'epsilon': 0.01,
            'inner_iterations': 20,
            'outer_iterations': 10
        }
    }
    
    # Update parameters for the specific scenario
    if scenario_name in scenario_configs:
        config = scenario_configs[scenario_name]
        for param, value in config.items():
            setattr(base_params, param, value)
    
    return base_params


def get_fttc_parameters(scenario_name):
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


def calculate_displacement_field(reference_img, deformed_img, params=None):
    """Calculate displacement field using DisplacementAnalyzer with custom parameters."""
    analyzer = DisplacementAnalyzer(params)
    flow = analyzer.calculate_flow(reference_img, deformed_img)
    return flow


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


def plot_combined_comparison(all_results, analysis_type="displacement"):
    """Plot all scenarios in one figure with magnitude+vector plots only."""
    if analysis_type == "displacement":
        fig, axes = plt.subplots(2, 3, figsize=(7, 4))
        fig.suptitle('Displacement Field Validation: Calculated vs Ground Truth', fontsize=11, color='black', y=0.875)
        column_titles = ['Low Displacement', 'Medium Displacement', 'High Displacement']
        unit_label = 'Magnitude (pixels)'
        colormap = 'viridis'
        row_titles = ['Calculated', 'Ground Truth']
    else:
        fig, axes = plt.subplots(3, 3, figsize=(7, 6))
        fig.suptitle('Traction Force Validation: 3-Way Comparison', fontsize=11, color='black', y=0.875)
        column_titles = ['Low Force', 'Medium Force', 'High Force']
        unit_label = 'Magnitude (Pa)'
        colormap = 'inferno'
        row_titles = ['From Ground Truth Disp.', 'From Calculated Disp.', 'Ground Truth Traction']
    
    scenarios = ['low', 'mid', 'high']
    
    for i, scenario in enumerate(scenarios):
        if scenario not in all_results:
            continue
            
        calculated_data = all_results[scenario]['calculated']
        ground_truth = all_results[scenario]['ground_truth']
        
        if analysis_type == "traction" and 'pipeline' in all_results[scenario]:
            pipeline_data = all_results[scenario]['pipeline']
            
            # Calculate magnitudes for all three
            calc_magnitude = np.sqrt(calculated_data[:,:,0]**2 + calculated_data[:,:,1]**2)
            pipeline_magnitude = np.sqrt(pipeline_data[:,:,0]**2 + pipeline_data[:,:,1]**2)
            gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
            
            # Determine common colorbar scale
            vmax = max(np.max(calc_magnitude), np.max(pipeline_magnitude), np.max(gt_magnitude))
            vmin = 0
            
            # Create coordinate grids for vector plotting
            h, w = calculated_data.shape[:2]
            step = max(h//30, w//30, 10)
            y, x = np.mgrid[0:h:step, 0:w:step]
            
            # Row 0: Traction from ground truth displacement
            im_calc = axes[0, i].imshow(calc_magnitude, cmap=colormap, vmin=vmin, vmax=vmax)
            axes[0, i].quiver(x, y, calculated_data[::step, ::step, 0], 
                             -calculated_data[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.4)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            divider_calc = make_axes_locatable(axes[0, i])
            cax_calc = divider_calc.append_axes("right", size="5%", pad=0.05)
            cbar_calc = plt.colorbar(im_calc, cax=cax_calc)
            cbar_calc.set_label(unit_label, fontsize=8, color='black')
            cbar_calc.ax.tick_params(labelsize=6, colors='black')
            
            # Row 1: Traction from calculated displacement (pipeline)
            im_pipeline = axes[1, i].imshow(pipeline_magnitude, cmap=colormap, vmin=vmin, vmax=vmax)
            axes[1, i].quiver(x, y, pipeline_data[::step, ::step, 0], 
                             -pipeline_data[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.4)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            
            divider_pipeline = make_axes_locatable(axes[1, i])
            cax_pipeline = divider_pipeline.append_axes("right", size="5%", pad=0.05)
            cbar_pipeline = plt.colorbar(im_pipeline, cax=cax_pipeline)
            cbar_pipeline.set_label(unit_label, fontsize=8, color='black')
            cbar_pipeline.ax.tick_params(labelsize=6, colors='black')
            
            # Row 2: Ground truth traction
            im_gt = axes[2, i].imshow(gt_magnitude, cmap=colormap, vmin=vmin, vmax=vmax)
            axes[2, i].quiver(x, y, ground_truth[::step, ::step, 0], 
                             -ground_truth[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.4)
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            
            divider_gt = make_axes_locatable(axes[2, i])
            cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
            cbar_gt = plt.colorbar(im_gt, cax=cax_gt)
            cbar_gt.set_label(unit_label, fontsize=8, color='black')
            cbar_gt.ax.tick_params(labelsize=6, colors='black')
            
        else:
            # Original 2-row layout for displacement or if no pipeline data
            calc_magnitude = np.sqrt(calculated_data[:,:,0]**2 + calculated_data[:,:,1]**2)
            gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
            
            vmax = max(np.max(calc_magnitude), np.max(gt_magnitude))
            vmin = 0
            
            h, w = calculated_data.shape[:2]
            step = max(h//30, w//30, 10)
            y, x = np.mgrid[0:h:step, 0:w:step]
            
            # Top row: Calculated fields
            im_calc = axes[0, i].imshow(calc_magnitude, cmap=colormap, vmin=vmin, vmax=vmax)
            axes[0, i].quiver(x, y, calculated_data[::step, ::step, 0], 
                             -calculated_data[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.4)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            divider_calc = make_axes_locatable(axes[0, i])
            cax_calc = divider_calc.append_axes("right", size="5%", pad=0.05)
            cbar_calc = plt.colorbar(im_calc, cax=cax_calc)
            cbar_calc.set_label(unit_label, fontsize=8, color='black')
            cbar_calc.ax.tick_params(labelsize=6, colors='black')
            
            # Bottom row: Ground truth fields
            im_gt = axes[1, i].imshow(gt_magnitude, cmap=colormap, vmin=vmin, vmax=vmax)
            axes[1, i].quiver(x, y, ground_truth[::step, ::step, 0], 
                             -ground_truth[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.4)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            
            divider_gt = make_axes_locatable(axes[1, i])
            cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
            cbar_gt = plt.colorbar(im_gt, cax=cax_gt)
            cbar_gt.set_label(unit_label, fontsize=8, color='black')
            cbar_gt.ax.tick_params(labelsize=6, colors='black')
        
        # Add column title to top row
        if i < len(column_titles):
            axes[0, i].set_title(column_titles[i], fontsize=10, color='black', pad=5)
    
    # Add row titles
    for j, title in enumerate(row_titles):
        axes[j, 0].text(-0.1, 0.5, title, transform=axes[j, 0].transAxes, 
                        fontsize=9, color='black', rotation=90, va='center', ha='right')
    
    # Set tick label properties for all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=6, colors='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for suptitle
    return fig


def calculate_error_metrics(calculated_data, ground_truth, analysis_type="displacement"):
    """Calculate normalized error metrics between calculated and ground truth fields."""
    # Calculate differences
    diff_x = calculated_data[:,:,0] - ground_truth[:,:,0]
    diff_y = calculated_data[:,:,1] - ground_truth[:,:,1]
    
    # Calculate maximum magnitude in ground truth for normalization
    gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
    max_gt_magnitude = np.max(gt_magnitude)
    
    # Root mean square error (normalized)
    rmse_total = np.sqrt(np.mean(diff_x**2 + diff_y**2))
    normalized_rmse = rmse_total / max_gt_magnitude if max_gt_magnitude > 0 else 0
    
    return {
        'normalized_rmse': normalized_rmse,
        'max_gt_magnitude': max_gt_magnitude
    }


def plot_error_comparison(all_results, analysis_type="displacement"):
    """Create bar plot comparing normalized error metrics across scenarios."""
    scenarios = list(all_results.keys())
    
    if analysis_type == "traction" and any('pipeline_errors' in all_results[s] for s in scenarios):
        # Create grouped bar plot for FTTC with pipeline comparison
        gt_errors = [all_results[s]['errors']['normalized_rmse'] * 100 for s in scenarios]
        pipeline_errors = [all_results[s]['pipeline_errors']['normalized_rmse'] * 100 
                         if 'pipeline_errors' in all_results[s] and all_results[s]['pipeline_errors'] 
                         else 0 for s in scenarios]
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gt_errors, width, label='From Ground Truth Displacement', 
                      color=plt.cm.tab10.colors[0], alpha=0.7)
        bars2 = ax.bar(x + width/2, pipeline_errors, width, label='From Calculated Displacement', 
                      color=plt.cm.tab10.colors[1], alpha=0.7)
        
        # Add value labels on bars
        for bar, error in zip(bars1, gt_errors):
            if error > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{error:.1f}%', ha='center', va='bottom', fontsize=6, color='black')
        
        for bar, error in zip(bars2, pipeline_errors):
            if error > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{error:.1f}%', ha='center', va='bottom', fontsize=6, color='black')
        
        ax.set_xlabel('Scenario', fontsize=8, color='black')
        ax.set_ylabel('Normalized RMSE (%)', fontsize=8, color='black')
        ax.set_title('FTTC Error Comparison', fontsize=9, color='black')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')
        
        max_error = max(max(gt_errors), max(pipeline_errors)) if pipeline_errors else max(gt_errors)
        ax.set_ylim(0, max_error * 1.2)
        
    else:
        # Original single bar plot for displacement or simple FTTC
        normalized_errors = [all_results[s]['errors']['normalized_rmse'] * 100 for s in scenarios]
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        colors = plt.cm.tab10.colors[:3]
        bars = ax.bar(scenarios, normalized_errors, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, normalized_errors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{error:.1f}%', ha='center', va='bottom', fontsize=6, color='black')
        
        ax.set_xlabel('Scenario', fontsize=8, color='black')
        ax.set_ylabel('Normalized RMSE (%)', fontsize=8, color='black')
        
        if analysis_type == "displacement":
            title = 'Normalized Displacement Error'
        else:
            title = 'Normalized Traction Error'
        ax.set_title(title, fontsize=10, color='black')
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(normalized_errors) * 1.2)
    
    # Set tick label properties
    ax.tick_params(labelsize=6, colors='black')
    
    plt.tight_layout()
    return fig


def validate_displacement_scenario(scenario_folder):
    """Validate displacement analysis for a single scenario."""
    scenario_name = os.path.basename(scenario_folder)
    print(f"\n--- Validating displacement for scenario: {scenario_name} ---")
    
    # Get scenario-specific parameters
    params = get_displacement_parameters(scenario_name)
    print(f"  Using parameters: tau={params.tau}, lambda_={params.lambda_}, "
          f"nscales={params.nscales}, warps={params.warps}")
    
    # Load images
    reference_path = os.path.join(scenario_folder, 'reference.tif')
    deformed_path = os.path.join(scenario_folder, 'deformed.tif')
    
    reference_img = load_tif_image(reference_path)
    deformed_img = load_tif_image(deformed_path)
    
    # Apply preprocessing
    reference_img = preprocess_image(reference_img)
    deformed_img = preprocess_image(deformed_img)
    
    print(f"  Reference image shape: {reference_img.shape}")
    print(f"  Deformed image shape: {deformed_img.shape}")
    
    # Calculate displacement field with custom parameters
    calculated_flow = calculate_displacement_field(reference_img, deformed_img, params)
    print(f"  Calculated flow shape: {calculated_flow.shape}")
    
    # Load ground truth
    ground_truth = load_ground_truth_displacement(scenario_folder)
    print(f"  Ground truth shape: {ground_truth.shape}")
    
    # Calculate error metrics
    errors = calculate_error_metrics(calculated_flow, ground_truth, "displacement")
    
    print("  Displacement Error Metrics:")
    print(f"    Normalized RMSE: {errors['normalized_rmse']*100:.2f}% "
          f"(fraction of max GT displacement: {errors['max_gt_magnitude']:.3f})")
    
    return calculated_flow, ground_truth, errors


def validate_fttc_scenario(scenario_folder, displacement_flow=None):
    """Validate FTTC analysis for a single scenario."""
    scenario_name = os.path.basename(scenario_folder)
    print(f"\n--- Validating FTTC for scenario: {scenario_name} ---")
    
    # Get scenario-specific parameters
    params = get_fttc_parameters(scenario_name)
    regularization_info = "auto-GCV" if params.auto_gcv else f"{params.regularization}"
    print(f"  Using parameters: E={params.young_modulus} Pa, nu={params.poisson_ratio_substrate}, "
          f"regularization={regularization_info}, pixel_size={params.pixel_size} µm")
    
    # Load displacement data (input for FTTC from ground truth)
    disp_x, disp_y = load_displacement_data(scenario_folder)
    print(f"  Ground truth displacement field shapes: x={disp_x.shape}, y={disp_y.shape}")
    
    # Calculate traction field using FTTC from ground truth displacement
    _, calculated_trac = calculate_traction_field(disp_x, disp_y, params)
    print(f"  Calculated traction field shape: {calculated_trac.shape}")
    
    # Transpose to match ground truth format (H, W, 2)
    if calculated_trac.shape[0] == 2:
        calculated_trac = np.transpose(calculated_trac, (1, 2, 0))
        print(f"  Transposed traction field shape: {calculated_trac.shape}")
    
    # Calculate pipeline traction (from calculated displacement if provided)
    pipeline_trac = None
    if displacement_flow is not None:
        print(f"  Calculating pipeline traction from displacement analysis results...")
        # Convert displacement flow from pixels to microns for FTTC input
        disp_x_pipeline = displacement_flow[:,:,0] * params.pixel_size
        disp_y_pipeline = displacement_flow[:,:,1] * params.pixel_size
        
        _, pipeline_trac = calculate_traction_field(disp_x_pipeline, disp_y_pipeline, params)
        if pipeline_trac.shape[0] == 2:
            pipeline_trac = np.transpose(pipeline_trac, (1, 2, 0))
        print(f"  Pipeline traction field shape: {pipeline_trac.shape}")
    
    # Load ground truth traction
    gt_trac_x, gt_trac_y = load_ground_truth_traction(scenario_folder)
    ground_truth = np.stack([gt_trac_x, gt_trac_y], axis=-1)
    print(f"  Ground truth traction shape: {ground_truth.shape}")
    
    # Calculate error metrics
    errors = calculate_error_metrics(calculated_trac, ground_truth, "traction")
    
    # Calculate pipeline error metrics if available
    pipeline_errors = None
    if pipeline_trac is not None:
        pipeline_errors = calculate_error_metrics(pipeline_trac, ground_truth, "traction")
        print("  Pipeline FTTC Error Metrics (from calculated displacement):")
        print(f"    Normalized RMSE: {pipeline_errors['normalized_rmse']*100:.2f}%")
    
    print("  FTTC Error Metrics (from ground truth displacement):")
    print(f"    Normalized RMSE: {errors['normalized_rmse']*100:.2f}% "
          f"(fraction of max GT traction: {errors['max_gt_magnitude']:.3f} Pa)")
    
    return calculated_trac, ground_truth, errors, pipeline_trac, pipeline_errors


def main():
    """Main validation function."""
    base_dir = Path(__file__).parent / 'benchmark_displacements_forces'
    scenarios = ['low', 'mid', 'high']
    
    displacement_results = {}
    fttc_results = {}
    
    print("="*60)
    print("COMPREHENSIVE TFM VALIDATION")
    print("="*60)
    
    for scenario in scenarios:
        scenario_path = base_dir / scenario
        if scenario_path.exists():
            # Validate displacement analysis
            displacement_flow, disp_ground_truth, disp_errors = validate_displacement_scenario(str(scenario_path))
            displacement_results[scenario] = {
                'calculated': displacement_flow,
                'ground_truth': disp_ground_truth,
                'errors': disp_errors
            }
            
            # Validate FTTC analysis (pass displacement flow for pipeline validation)
            trac_from_gt, trac_ground_truth, trac_errors, pipeline_trac, pipeline_errors = validate_fttc_scenario(str(scenario_path), displacement_flow)
            
            fttc_results[scenario] = {
                'calculated': trac_from_gt,
                'ground_truth': trac_ground_truth,
                'errors': trac_errors
            }
            
            # Add pipeline results if available
            if pipeline_trac is not None:
                fttc_results[scenario]['pipeline'] = pipeline_trac
                fttc_results[scenario]['pipeline_errors'] = pipeline_errors
        else:
            print(f"Warning: Scenario folder {scenario_path} not found")
    
    # Create and save displacement plots
    if displacement_results:
        print("\n--- Creating displacement validation plots ---")
        combined_fig = plot_combined_comparison(displacement_results, "displacement")
        combined_output_path = Path(__file__).parent / "displacement_validation_combined.png"
        combined_fig.savefig(combined_output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved combined displacement plot: {combined_output_path}")
        plt.close(combined_fig)
        
        error_fig = plot_error_comparison(displacement_results, "displacement")
        error_output_path = Path(__file__).parent / "displacement_error_comparison.png"
        error_fig.savefig(error_output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved displacement error plot: {error_output_path}")
        plt.close(error_fig)
    
    # Create and save FTTC plots
    if fttc_results:
        print("\n--- Creating FTTC validation plots ---")
        combined_fig = plot_combined_comparison(fttc_results, "traction")
        combined_output_path = Path(__file__).parent / "fttc_validation_combined.png"
        combined_fig.savefig(combined_output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved combined FTTC plot: {combined_output_path}")
        plt.close(combined_fig)
        
        error_fig = plot_error_comparison(fttc_results, "traction")
        error_output_path = Path(__file__).parent / "fttc_error_comparison.png"
        error_fig.savefig(error_output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved FTTC error plot: {error_output_path}")
        plt.close(error_fig)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print("\nDISPLACEMENT ANALYSIS:")
    for scenario, results in displacement_results.items():
        errors = results['errors']
        print(f"  {scenario.upper()} scenario:")
        print(f"    Normalized RMSE: {errors['normalized_rmse']*100:.2f}%")
        print(f"    Max GT displacement: {errors['max_gt_magnitude']:.3f} pixels")
    
    print("\nFTTC ANALYSIS:")
    for scenario, results in fttc_results.items():
        errors = results['errors']
        print(f"  {scenario.upper()} scenario:")
        print(f"    From ground truth displacement - RMSE: {errors['normalized_rmse']*100:.2f}%")
        if 'pipeline_errors' in results and results['pipeline_errors']:
            pipeline_errors = results['pipeline_errors']
            print(f"    From calc. displacement - RMSE: {pipeline_errors['normalized_rmse']*100:.2f}%")
        print(f"    Max GT traction: {errors['max_gt_magnitude']:.3f} Pa")


if __name__ == "__main__":
    main()
