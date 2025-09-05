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
from napariTFM.backend.metrics_calculator import calculate_strain_energy_density, calculate_total_strain_energy


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
            # 'auto_gcv': True,
            'auto_gcv': False,
            'regularization': 1e-17,
            'pixel_size': 0.1,  # µm
            'downscale_factor': 1
        },
        'mid': {
            'young_modulus': 20000,  # Pa
            'poisson_ratio_substrate': 0.5,
            'lanczos_exp': 1,
            # 'auto_gcv': True,
            'auto_gcv': False,
            'regularization': 1e-17,
            'pixel_size': 0.1,  # µm
            'downscale_factor': 1
        },
        'high': {
            'young_modulus': 20000,  # Pa
            'poisson_ratio_substrate': 0.5,
            'lanczos_exp': 1,
            # 'auto_gcv': True,
            'auto_gcv': False,
            'regularization': 1e-17,
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


def calculate_correlation_metrics(calculated_data, ground_truth_data):
    """Calculate correlation between calculated and ground truth fields."""
    # Flatten the data and remove invalid values
    calc_flat = calculated_data.flatten()
    gt_flat = ground_truth_data.flatten()
    
    # Create mask for valid (non-NaN, non-zero) values
    valid_mask = ~np.isnan(calc_flat) & ~np.isnan(gt_flat) & (calc_flat != 0) & (gt_flat != 0)
    
    if np.sum(valid_mask) > 1:
        correlation = np.corrcoef(calc_flat[valid_mask], gt_flat[valid_mask])[0, 1]
    else:
        correlation = 0
    
    return correlation


def calculate_strain_energy_metrics(displacement_data, calculated_traction, ground_truth_traction, pixel_size_um=0.1):
    """Calculate strain energy metrics for TFM validation."""
    # Convert displacement from pixels to meters
    displacement_m = displacement_data * (pixel_size_um * 1e-6)
    
    # Create a simple mask (non-zero regions)
    mask = np.logical_and(
        np.sqrt(calculated_traction[:,:,0]**2 + calculated_traction[:,:,1]**2) > 0,
        np.sqrt(ground_truth_traction[:,:,0]**2 + ground_truth_traction[:,:,1]**2) > 0
    )
    
    # Calculate pixel area in m²
    pixel_area_m2 = (pixel_size_um * 1e-6) ** 2
    
    # Calculate strain energy density for both calculated and ground truth
    sed_calculated = calculate_strain_energy_density(displacement_m, calculated_traction)
    sed_gt = calculate_strain_energy_density(displacement_m, ground_truth_traction)
    
    # Calculate total strain energies
    total_se_calculated = calculate_total_strain_energy(sed_calculated, mask, pixel_area_m2)
    total_se_gt = calculate_total_strain_energy(sed_gt, mask, pixel_area_m2)
    
    return {
        'total_se_calculated': total_se_calculated,
        'total_se_gt': total_se_gt,
        'mask_coverage': np.sum(mask) / mask.size
    }


def plot_displacement(displacement_results):
    scenarios = ['low', 'mid', 'high']
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4))  # DIN A4 compatible width
    fig.suptitle('Displacement Field Validation', fontsize=10, y=0.95)
    
    # Plot displacement fields for each scenario
    for i, scenario in enumerate(scenarios):
        if scenario in displacement_results:
            calculated = displacement_results[scenario]['calculated']
            ground_truth = displacement_results[scenario]['ground_truth']
            
            # Calculate magnitudes
            calc_magnitude = np.sqrt(calculated[:,:,0]**2 + calculated[:,:,1]**2)
            gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
            vmax = max(np.max(calc_magnitude), np.max(gt_magnitude))
            
            # Create coordinate grids for vector plotting
            h, w = calculated.shape[:2]
            step = max(h//20, w//20, 8)  # Adjust step size for better visibility
            y, x = np.mgrid[0:h:step, 0:w:step]
            
            # Ground truth (top row)
            im1 = axes[0, i].imshow(gt_magnitude, cmap='viridis', vmin=0, vmax=vmax)
            axes[0, i].quiver(x, y, ground_truth[::step, ::step, 0], 
                             -ground_truth[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.6)
            axes[0, i].set_title(f'{scenario.upper()}\nGround Truth', fontsize=8)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[0, i].tick_params(labelsize=6)
            divider1 = make_axes_locatable(axes[0, i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.ax.tick_params(labelsize=6)
            
            # Calculated (bottom row)
            im2 = axes[1, i].imshow(calc_magnitude, cmap='viridis', vmin=0, vmax=vmax)
            axes[1, i].quiver(x, y, calculated[::step, ::step, 0], 
                             -calculated[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.6)
            axes[1, i].set_title('Calculated', fontsize=8)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            axes[1, i].tick_params(labelsize=6)
            divider2 = make_axes_locatable(axes[1, i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.ax.tick_params(labelsize=6)
    
    # Create a single subplot spanning both rows for correlation
    gs = axes[0, 3].get_gridspec()
    # Remove the individual subplots
    axes[0, 3].remove()
    axes[1, 3].remove()
    # Create a subplot spanning both rows
    ax_corr = fig.add_subplot(gs[:, 3])
    
    correlations = [displacement_results[s]['displacement_correlation'] if s in displacement_results else 0 for s in scenarios]
    bars = ax_corr.bar(scenarios, correlations, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax_corr.set_title('Correlation between\nCalculated and\nGround Truth Data', fontsize=8, pad=15)
    ax_corr.set_ylabel('Correlation Coefficient', fontsize=8)
    ax_corr.set_ylim(0, 1.1)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.tick_params(labelsize=6)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        ax_corr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{corr:.3f}', ha='center', va='bottom', fontsize=6)
    
    # Adjust the correlation subplot to be more compact vertically
    pos = ax_corr.get_position()
    # Make it smaller vertically and center it better
    new_height = pos.height * 0.6
    new_y = pos.y0 + (pos.height - new_height) * 0.5
    ax_corr.set_position([pos.x0, new_y, pos.width, new_height])
    
    plt.tight_layout()
    return fig


def plot_traction(fttc_results):
    scenarios = ['low', 'mid', 'high']
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4))  # DIN A4 compatible width
    fig.suptitle('Traction Force Validation', fontsize=10, y=0.95)
    
    # Plot traction fields for each scenario
    for i, scenario in enumerate(scenarios):
        if scenario in fttc_results:
            calculated = fttc_results[scenario]['calculated']
            ground_truth = fttc_results[scenario]['ground_truth']
            
            # Calculate magnitudes
            if calculated is not None:
                calc_magnitude = np.sqrt(calculated[:,:,0]**2 + calculated[:,:,1]**2)
            else:
                calc_magnitude = np.zeros_like(ground_truth[:,:,0])
            gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
            vmax = max(np.max(calc_magnitude), np.max(gt_magnitude)) if np.max(calc_magnitude) > 0 else np.max(gt_magnitude)
            
            # Create coordinate grids for vector plotting
            h, w = ground_truth.shape[:2]
            step = max(h//20, w//20, 8)  # Adjust step size for better visibility
            y, x = np.mgrid[0:h:step, 0:w:step]
            
            # Ground truth (top row)
            im1 = axes[0, i].imshow(gt_magnitude, cmap='inferno', vmin=0, vmax=vmax)
            axes[0, i].quiver(x, y, ground_truth[::step, ::step, 0], 
                             -ground_truth[::step, ::step, 1], 
                             color='white', scale_units='xy', scale=0.01*vmax, alpha=0.6)
            axes[0, i].set_title(f'{scenario.upper()}\nGround Truth', fontsize=8)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[0, i].tick_params(labelsize=6)
            divider1 = make_axes_locatable(axes[0, i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.ax.tick_params(labelsize=6)
            
            # Calculated (bottom row)
            im2 = axes[1, i].imshow(calc_magnitude, cmap='inferno', vmin=0, vmax=vmax)
            if calculated is not None:
                axes[1, i].quiver(x, y, calculated[::step, ::step, 0], 
                                 -calculated[::step, ::step, 1], 
                                 color='white', scale_units='xy', scale=0.01*vmax, alpha=0.6)
            axes[1, i].set_title('Calculated', fontsize=8)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            axes[1, i].tick_params(labelsize=6)
            divider2 = make_axes_locatable(axes[1, i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.ax.tick_params(labelsize=6)
    
    # Create a single subplot spanning both rows for correlation
    gs = axes[0, 3].get_gridspec()
    # Remove the individual subplots
    axes[0, 3].remove()
    axes[1, 3].remove()
    # Create a subplot spanning both rows
    ax_corr = fig.add_subplot(gs[:, 3])
    
    correlations = [fttc_results[s]['traction_correlation'] if s in fttc_results else 0 for s in scenarios]
    bars = ax_corr.bar(scenarios, correlations, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax_corr.set_title('Correlation between\nCalculated and\nGround Truth Data', fontsize=8, pad=15)
    ax_corr.set_ylabel('Correlation Coefficient', fontsize=8)
    ax_corr.set_ylim(0, 1.1)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.tick_params(labelsize=6)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        ax_corr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{corr:.3f}', ha='center', va='bottom', fontsize=6)
    
    # Adjust the correlation subplot to be more compact vertically
    pos = ax_corr.get_position()
    # Make it smaller vertically and center it better
    new_height = pos.height * 0.6
    new_y = pos.y0 + (pos.height - new_height) * 0.5
    ax_corr.set_position([pos.x0, new_y, pos.width, new_height])
    
    plt.tight_layout()
    return fig


def plot_strain_energy_comparison(fttc_results):
    """Create strain energy comparison figure: GT vs calculated."""
    scenarios = ['low', 'mid', 'high']
    
    # Extract strain energy values
    se_gt_values = [fttc_results[s]['strain_energy_gt'] if s in fttc_results and fttc_results[s]['strain_energy_gt'] > 0 else 1e-20 for s in scenarios]
    se_calc_values = [fttc_results[s]['strain_energy_calc'] if s in fttc_results and fttc_results[s]['strain_energy_calc'] > 0 else 1e-20 for s in scenarios]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))  # DIN A4 compatible
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, se_gt_values, width, label='Ground Truth', 
                  color='#1f77b4', alpha=0.7)
    bars2 = ax.bar(x + width/2, se_calc_values, width, label='Calculated', 
                  color='#ff7f0e', alpha=0.7)
    
    # Add value labels on bars (in scientific notation)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1e-19:
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.2,
                       f'{height:.1e}', ha='center', va='bottom', fontsize=6, rotation=45)
    
    ax.set_xlabel('Scenario', fontsize=8)
    ax.set_ylabel('Strain Energy (J)', fontsize=8)
    ax.set_title('Strain Energy Comparison: Ground Truth vs Calculated', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    return fig


def plot_normalized_strain_energy(fttc_results):
    """Create normalized strain energy plot: calculated/ground_truth ratio."""
    scenarios = ['low', 'mid', 'high']
    
    # Calculate normalized strain energies (calculated/ground_truth)
    normalized_values = []
    for scenario in scenarios:
        if scenario in fttc_results:
            se_gt = fttc_results[scenario]['strain_energy_gt']
            se_calc = fttc_results[scenario]['strain_energy_calc']
            if se_gt > 0:
                normalized_values.append(se_calc / se_gt)
            else:
                normalized_values.append(0)
        else:
            normalized_values.append(0)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))  # DIN A4 compatible
    
    # Create bar plot
    bars = ax.bar(scenarios, normalized_values, 
                  color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, normalized_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{value:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    # Add horizontal reference line at y=1 (perfect match)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label='Perfect Match (Calc/GT = 1.0)')
    
    ax.set_xlabel('Scenario', fontsize=8)
    ax.set_ylabel('Normalized Strain Energy\n(Calculated / Ground Truth)', fontsize=8)
    ax.set_title('Normalized Strain Energy: Calculated vs Ground Truth', fontsize=10)
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=6)
    
    # Set y-axis limits to show values clearly
    y_max = max(normalized_values) if normalized_values else 1
    ax.set_ylim(0, max(1.2, y_max * 1.1))
    
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
    
    # Calculate displacement correlation
    displacement_correlation = calculate_correlation_metrics(calculated_flow, ground_truth)
    
    print("  Displacement Metrics:")
    print(f"    Displacement Correlation: {displacement_correlation:.3f}")
    
    return calculated_flow, ground_truth, displacement_correlation


def validate_fttc_scenario(scenario_folder, displacement_flow=None):
    """Validate FTTC analysis for a single scenario."""
    scenario_name = os.path.basename(scenario_folder)
    print(f"\n--- Validating FTTC for scenario: {scenario_name} ---")
    
    # Get scenario-specific parameters
    params = get_fttc_parameters(scenario_name)
    regularization_info = "auto-GCV" if params.auto_gcv else f"{params.regularization}"
    print(f"  Using parameters: E={params.young_modulus} Pa, nu={params.poisson_ratio_substrate}, "
          f"regularization={regularization_info}, pixel_size={params.pixel_size} µm")
    
    # Load ground truth traction
    gt_trac_x, gt_trac_y = load_ground_truth_traction(scenario_folder)
    ground_truth = np.stack([gt_trac_x, gt_trac_y], axis=-1)
    print(f"  Ground truth traction shape: {ground_truth.shape}")
    
    # Calculate traction from calculated displacement only
    calculated_trac = None
    traction_correlation = 0
    strain_energy_gt = 0
    strain_energy_calc = 0
    
    if displacement_flow is not None:
        print("  Calculating traction from calculated displacement...")
        # Convert displacement flow from pixels to microns for FTTC input
        disp_x_pipeline = displacement_flow[:,:,0] * params.pixel_size
        disp_y_pipeline = displacement_flow[:,:,1] * params.pixel_size
        
        _, calculated_trac = calculate_traction_field(disp_x_pipeline, disp_y_pipeline, params)
        if calculated_trac.shape[0] == 2:
            calculated_trac = np.transpose(calculated_trac, (1, 2, 0))
        print(f"  Calculated traction field shape: {calculated_trac.shape}")
        
        # Calculate correlation
        traction_correlation = calculate_correlation_metrics(calculated_trac, ground_truth)
        
        # Calculate strain energies
        # Load ground truth displacement for strain energy calculation
        gt_disp_x, gt_disp_y = load_displacement_data(scenario_folder)
        gt_displacement = np.stack([gt_disp_x, gt_disp_y], axis=-1)
        
        # GT strain energy
        se_metrics_gt = calculate_strain_energy_metrics(
            gt_displacement, ground_truth, ground_truth, params.pixel_size
        )
        strain_energy_gt = se_metrics_gt['total_se_calculated']
        
        # Calculated strain energy
        disp_for_se = displacement_flow * params.pixel_size
        se_metrics_calc = calculate_strain_energy_metrics(
            disp_for_se, calculated_trac, ground_truth, params.pixel_size
        )
        strain_energy_calc = se_metrics_calc['total_se_calculated']
        
        print("  FTTC Metrics:")
        print(f"    Traction Correlation: {traction_correlation:.3f}")
        print(f"    Strain Energy - GT: {strain_energy_gt:.2e} J")
        print(f"    Strain Energy - Calc: {strain_energy_calc:.2e} J")
    
    return {
        'calculated_trac': calculated_trac,
        'ground_truth_trac': ground_truth,
        'traction_correlation': traction_correlation,
        'strain_energy_gt': strain_energy_gt,
        'strain_energy_calc': strain_energy_calc
    }


def main():
    """Main validation function."""
    base_dir = Path(__file__).parent
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
            displacement_flow, disp_ground_truth, displacement_correlation = validate_displacement_scenario(str(scenario_path))
            displacement_results[scenario] = {
                'calculated': displacement_flow,
                'ground_truth': disp_ground_truth,
                'displacement_correlation': displacement_correlation
            }
            
            # Validate FTTC analysis (pass displacement flow for traction calculation)
            trac_results = validate_fttc_scenario(str(scenario_path), displacement_flow)
            
            fttc_results[scenario] = {
                'calculated': trac_results['calculated_trac'],
                'ground_truth': trac_results['ground_truth_trac'],
                'traction_correlation': trac_results['traction_correlation'],
                'strain_energy_gt': trac_results['strain_energy_gt'],
                'strain_energy_calc': trac_results['strain_energy_calc']
            }
        else:
            print(f"Warning: Scenario folder {scenario_path} not found")
    
    # Create and save displacement plots
    if displacement_results:
        print("\n--- Creating validation plots ---")
        
        # Create consolidated displacement plot
        disp_fig = plot_displacement(displacement_results)
        disp_path = Path(__file__).parent / "displacement.png"
        disp_fig.savefig(disp_path, dpi=300, bbox_inches='tight')
        print(f"  Saved consolidated displacement plot: {disp_path}")
        plt.close(disp_fig)
        
        # Create consolidated traction plot
        trac_fig = plot_traction(fttc_results)
        trac_path = Path(__file__).parent / "traction.png"
        trac_fig.savefig(trac_path, dpi=300, bbox_inches='tight')
        print(f"  Saved consolidated traction plot: {trac_path}")
        plt.close(trac_fig)
        
        # Create strain energy comparison plot
        strain_energy_fig = plot_strain_energy_comparison(fttc_results)
        strain_energy_path = Path(__file__).parent / "strain_energy_comparison.png"
        strain_energy_fig.savefig(strain_energy_path, dpi=300, bbox_inches='tight')
        print(f"  Saved strain energy comparison plot: {strain_energy_path}")
        plt.close(strain_energy_fig)
        
        # Create normalized strain energy plot
        normalized_se_fig = plot_normalized_strain_energy(fttc_results)
        normalized_se_path = Path(__file__).parent / "normalized_strain_energy.png"
        normalized_se_fig.savefig(normalized_se_path, dpi=300, bbox_inches='tight')
        print(f"  Saved normalized strain energy plot: {normalized_se_path}")
        plt.close(normalized_se_fig)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    # Add normalized strain energy summary
    print("\nNORMALIZED STRAIN ENERGY RATIOS (Calculated/Ground Truth):")
    for scenario in displacement_results.keys():
        if scenario in fttc_results:
            trac_results = fttc_results[scenario]
            if trac_results['strain_energy_gt'] > 0:
                normalized_se = trac_results['strain_energy_calc'] / trac_results['strain_energy_gt']
                print(f"  {scenario.upper()}: {normalized_se:.3f}")
            else:
                print(f"  {scenario.upper()}: N/A (GT = 0)")
    print("  (Values close to 1.0 indicate good agreement)\n")
    
    print("\nDISPLACEMENT ANALYSIS:")
    for scenario, results in displacement_results.items():
        print(f"  {scenario.upper()} scenario: Correlation = {results['displacement_correlation']:.3f}")
    
    print("\nTFM ANALYSIS SUMMARY:")
    for scenario, results in displacement_results.items():
        print(f"  {scenario.upper()} scenario:")
        print(f"    Displacement Correlation: {results['displacement_correlation']:.3f}")
        if scenario in fttc_results:
            trac_results = fttc_results[scenario]
            print(f"    Traction Correlation: {trac_results['traction_correlation']:.3f}")
            print(f"    Strain Energy - GT: {trac_results['strain_energy_gt']:.2e} J")
            print(f"    Strain Energy - Calc: {trac_results['strain_energy_calc']:.2e} J")
            # Calculate and display normalized strain energy
            if trac_results['strain_energy_gt'] > 0:
                normalized_se = trac_results['strain_energy_calc'] / trac_results['strain_energy_gt']
                print(f"    Normalized Strain Energy (Calc/GT): {normalized_se:.3f}")
            else:
                print(f"    Normalized Strain Energy (Calc/GT): N/A (GT = 0)")


if __name__ == "__main__":
    main()
