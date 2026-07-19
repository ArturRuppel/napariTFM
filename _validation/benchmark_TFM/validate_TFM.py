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
import tifffile
try:
    import tomllib
except ImportError:
    import toml as tomllib

matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# Add the parent directory to path to import napariTFM modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.ndimage import gaussian_filter

from napariTFM.backend.displacement_analysis import DisplacementAnalyzer
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.fttc import FTTC
from napariTFM.backend.parameter_dataclasses import FTTCParameters
# from napariTFM.backend.metrics_calculator import calculate_strain_energy_density, calculate_total_strain_energy


def load_tif_image(filepath):
    """Load a TIF image as numpy array."""
    return np.array(Image.open(filepath))


def preprocess_image(image):
    """Contrast-scale (80/99.9 percentile clip -> [0, 1]) then lightly blur.

    Inlined from the old backend ``ImageProcessor``, which was removed together
    with the preprocessing stage; this keeps the benchmark's historical
    preprocessing behaviour self-contained.
    """
    img = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(img, (80, 99.9))
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return gaussian_filter(img, sigma=1)


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
            'tau': 0.1,
            'lambda_': 0.1,
            'theta': 0.15,
            'nscales': 50,
            'scale_step': 0.95,
            'warps': 10,
            'epsilon': 0.01,
            'inner_iterations': 25,
            'outer_iterations': 15
        },
        'mid': {
            'tau': 0.1,
            'lambda_': 0.3,
            'theta': 0.15,
            'nscales': 50,
            'scale_step': 0.95,
            'warps': 10,
            'epsilon': 0.01,
            'inner_iterations': 25,
            'outer_iterations': 15
        },
        'high': {
            'tau': 0.1,
            'lambda_': 0.5,
            'theta': 0.15,
            'nscales': 50,
            'scale_step': 0.95,
            'warps': 10,
            'epsilon': 0.01,
            'inner_iterations': 25,
            'outer_iterations': 15
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
            'regularization': 1e-6,
            'pixel_size': 0.1,  # µm
            'downscale_factor': 1
        },
        'mid': {
            'young_modulus': 20000,  # Pa
            'poisson_ratio_substrate': 0.5,
            'regularization': 1e-6,
            'pixel_size': 0.1,  # µm
            'downscale_factor': 1
        },
        'high': {
            'young_modulus': 20000,  # Pa
            'poisson_ratio_substrate': 0.5,
            'regularization': 1e-6,
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
    
    # Calculate traction forces
    regularization = params.regularization
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


def calculate_dtm_metric(calculated_traction, ground_truth_traction):
    """
    Calculate Deviation of Traction Magnitude (DTM) as defined by Sabass et al. 2008.

    DTM = (1/N) * Σ [ (‖Calc‖ - ‖GT‖) / ‖GT‖ ]

    Only calculated in regions where GT magnitude > 0.

    Dimensionless metric:
    - DTM = 0: perfect match
    - DTM < 0: underestimation (e.g., -0.2 means 20% underestimation on average)
    - DTM > 0: overestimation (e.g., +0.2 means 20% overestimation on average)
    """
    # Calculate magnitudes
    calc_magnitude = np.sqrt(calculated_traction[:,:,0]**2 + calculated_traction[:,:,1]**2)
    gt_magnitude = np.sqrt(ground_truth_traction[:,:,0]**2 + ground_truth_traction[:,:,1]**2)

    # Valid where GT > 0 and no NaN values
    valid_mask = (gt_magnitude > 0) & ~np.isnan(calc_magnitude) & ~np.isnan(gt_magnitude)

    if np.sum(valid_mask) > 0:
        # Relative deviation per pixel: (calc - gt) / gt
        relative_deviations = (calc_magnitude[valid_mask] - gt_magnitude[valid_mask]) / gt_magnitude[valid_mask]
        dtm = np.mean(relative_deviations)
    else:
        dtm = np.nan

    return dtm


def calculate_displacement_relative_error(calculated, ground_truth, min_threshold_fraction=0.01):
    """
    Calculate per-pixel signed relative error for displacement fields.

    Relative error = (|calc| - |GT|) / |GT|

    Args:
        calculated: Calculated displacement field (H x W x 2)
        ground_truth: Ground truth displacement field (H x W x 2)
        min_threshold_fraction: Minimum GT magnitude threshold as fraction of max (to avoid division by ~0)

    Returns:
        gt_magnitudes: 1D array of valid GT magnitudes
        relative_errors: 1D array of corresponding signed relative errors
    """
    # Calculate magnitudes
    calc_magnitude = np.sqrt(calculated[:,:,0]**2 + calculated[:,:,1]**2)
    gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)

    # Filter out very small GT magnitudes to avoid division issues
    max_gt = np.max(gt_magnitude)
    threshold = min_threshold_fraction * max_gt
    valid_mask = (gt_magnitude > threshold) & ~np.isnan(calc_magnitude) & ~np.isnan(gt_magnitude)

    gt_valid = gt_magnitude[valid_mask]
    calc_valid = calc_magnitude[valid_mask]

    # Signed relative error: positive = overestimation, negative = underestimation
    relative_errors = (calc_valid - gt_valid) / gt_valid

    return gt_valid, relative_errors


def parse_adhesion_config(config_path):
    """
    Parse dipole config to extract adhesion site locations and radii.

    Returns list of dicts with keys: 'center' (y, x in pixels), 'radius' (in pixels)
    """
    try:
        # Python 3.11+ tomllib requires binary mode
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    except (AttributeError, TypeError):
        # Older toml library requires text mode
        import toml
        with open(config_path, 'r') as f:
            config = toml.load(f)

    pixel_size_um = config['image']['spacing_xy']
    image_size = config['simulation']['n_points_xy']
    center_px = image_size // 2  # Image center in pixels

    adhesions = []

    for _, value in config.get('adheasion', {}).items():
        if value.get('type') == 'dipole':
            # Dipole parameters
            d_um = value['d']  # Distance between adhesion sites in µm
            phi_deg = value['phi']  # Angle in degrees
            a_um = value['a']  # Radius of adhesion site in µm
            pos_um = value.get('pos', [0, 0])  # Center position in µm

            # Convert to pixels
            d_px = d_um / pixel_size_um
            a_px = a_um / pixel_size_um
            pos_px = [p / pixel_size_um for p in pos_um]

            # Calculate positions of the two adhesion sites
            phi_rad = np.radians(phi_deg)
            offset_x = (d_px / 2) * np.cos(phi_rad)
            offset_y = (d_px / 2) * np.sin(phi_rad)

            # Site 1 and Site 2 positions (relative to image center)
            site1_x = center_px + pos_px[0] + offset_x
            site1_y = center_px + pos_px[1] + offset_y
            site2_x = center_px + pos_px[0] - offset_x
            site2_y = center_px + pos_px[1] - offset_y

            adhesions.append({'center': (site1_y, site1_x), 'radius': a_px})
            adhesions.append({'center': (site2_y, site2_x), 'radius': a_px})

    return adhesions, pixel_size_um


# def debug_plot_adhesion_masks(ground_truth_traction, adhesions, ring_width_px=15, save_path=None):
#     """
#     DEBUG: Plot GT force map with adhesion and surrounding masks overlaid.
#     Comment out calls to this function when not debugging.
#     """
#     gt_magnitude = np.sqrt(ground_truth_traction[:,:,0]**2 + ground_truth_traction[:,:,1]**2)

#     h, w = gt_magnitude.shape
#     y_coords, x_coords = np.ogrid[:h, :w]

#     # Create masks
#     adhesion_mask = np.zeros((h, w), dtype=bool)
#     ring_mask = np.zeros((h, w), dtype=bool)

#     for adhesion in adhesions:
#         cy, cx = adhesion['center']
#         radius = adhesion['radius']

#         distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)

#         adhesion_mask |= (distances <= radius)
#         ring_mask |= (distances > radius) & (distances <= radius + ring_width_px)

#     # Create figure with 3 subplots
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))

#     # Plot 1: GT magnitude with adhesion centers marked
#     im0 = axes[0].imshow(gt_magnitude * 1e-3, cmap='inferno')
#     for adhesion in adhesions:
#         cy, cx = adhesion['center']
#         radius = adhesion['radius']
#         circle = plt.Circle((cx, cy), radius, fill=False, color='cyan', linewidth=2)
#         axes[0].add_patch(circle)
#         axes[0].plot(cx, cy, 'c+', markersize=10, markeredgewidth=2)
#     axes[0].set_title('GT Magnitude + Adhesion Circles', fontsize=10)
#     plt.colorbar(im0, ax=axes[0], label='kPa')

#     # Plot 2: GT magnitude with adhesion mask overlay
#     gt_with_adhesion = gt_magnitude.copy() * 1e-3
#     im1 = axes[1].imshow(gt_with_adhesion, cmap='inferno')
#     # Overlay adhesion mask in semi-transparent cyan
#     adhesion_overlay = np.ma.masked_where(~adhesion_mask, np.ones_like(gt_magnitude))
#     axes[1].imshow(adhesion_overlay, cmap='cool', alpha=0.5, vmin=0, vmax=1)
#     axes[1].set_title(f'Adhesion Mask (radius={adhesions[0]["radius"]:.1f}px)', fontsize=10)
#     plt.colorbar(im1, ax=axes[1], label='kPa')

#     # Plot 3: GT magnitude with ring mask overlay
#     gt_with_ring = gt_magnitude.copy() * 1e-3
#     im2 = axes[2].imshow(gt_with_ring, cmap='inferno')
#     # Overlay ring mask in semi-transparent green
#     ring_overlay = np.ma.masked_where(~ring_mask, np.ones_like(gt_magnitude))
#     axes[2].imshow(ring_overlay, cmap='summer', alpha=0.5, vmin=0, vmax=1)
#     axes[2].set_title(f'Surrounding Ring Mask (width={ring_width_px}px)', fontsize=10)
#     plt.colorbar(im2, ax=axes[2], label='kPa')

#     plt.suptitle('DEBUG: Adhesion Region Masks', fontsize=12)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"  DEBUG: Saved adhesion mask plot to {save_path}")

#     plt.close(fig)
#     return fig


def calculate_dtms_metric(calculated_traction, adhesions, ring_width_px=15):
    """
    Calculate Deviation of Traction Magnitude in the Surrounding (DTMS) as defined by Sabass et al. 2008.

    DTMS = (1/N) * Σ [ ‖Reconstructed traction in surrounding‖ / ‖Reconstructed traction on adhesion‖ ]

    Measures how well edges/contours are reconstructed. Traction should decay rapidly outside adhesion.

    DTMS is expected to lie between 0 (optimal) and 1 (worst).
    """
    calc_magnitude = np.sqrt(calculated_traction[:,:,0]**2 + calculated_traction[:,:,1]**2)

    h, w = calc_magnitude.shape
    y_coords, x_coords = np.ogrid[:h, :w]

    dtms_values = []

    for adhesion in adhesions:
        cy, cx = adhesion['center']
        radius = adhesion['radius']

        # Distance from adhesion center
        distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)

        # Adhesion mask (inside the adhesion)
        adhesion_mask = distances <= radius

        # Surrounding ring mask
        ring_inner = radius
        ring_outer = radius + ring_width_px
        ring_mask = (distances > ring_inner) & (distances <= ring_outer)

        # Calculate mean magnitudes
        if np.sum(adhesion_mask) > 0 and np.sum(ring_mask) > 0:
            mag_on_adhesion = np.mean(calc_magnitude[adhesion_mask])
            mag_in_surrounding = np.mean(calc_magnitude[ring_mask])

            if mag_on_adhesion > 0:
                dtms_values.append(mag_in_surrounding / mag_on_adhesion)

    if len(dtms_values) > 0:
        return np.mean(dtms_values)
    else:
        return np.nan


def calculate_dta_metric(calculated_traction, ground_truth_traction, adhesions):
    """
    Calculate Deviation of Traction Angle (DTA) as defined by Sabass et al. 2008.

    DTA = arccos( (1/N) * Σ [ (Reconstructed · Real) / (‖Reconstructed‖ * ‖Real‖) ] )

    Returns angular deviation in degrees between reconstructed and real traction vectors.
    """
    cosine_values = []

    h, w = calculated_traction.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]

    for adhesion in adhesions:
        cy, cx = adhesion['center']
        radius = adhesion['radius']

        # Distance from adhesion center
        distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)

        # Adhesion mask
        adhesion_mask = distances <= radius

        if np.sum(adhesion_mask) > 0:
            # Get mean vectors on this adhesion
            calc_vec = np.array([
                np.mean(calculated_traction[:,:,0][adhesion_mask]),
                np.mean(calculated_traction[:,:,1][adhesion_mask])
            ])
            gt_vec = np.array([
                np.mean(ground_truth_traction[:,:,0][adhesion_mask]),
                np.mean(ground_truth_traction[:,:,1][adhesion_mask])
            ])

            calc_norm = np.linalg.norm(calc_vec)
            gt_norm = np.linalg.norm(gt_vec)

            if calc_norm > 0 and gt_norm > 0:
                cosine = np.dot(calc_vec, gt_vec) / (calc_norm * gt_norm)
                # Clamp to [-1, 1] to handle numerical errors
                cosine = np.clip(cosine, -1, 1)
                cosine_values.append(cosine)

    if len(cosine_values) > 0:
        mean_cosine = np.mean(cosine_values)
        dta_rad = np.arccos(mean_cosine)
        dta_deg = np.degrees(dta_rad)
        return dta_deg
    else:
        return np.nan


# def calculate_strain_energy_metrics(displacement_data, calculated_traction, ground_truth_traction, pixel_size_um=0.1):
#     """Calculate strain energy metrics for TFM validation."""
#     # Convert displacement from pixels to meters
#     displacement_m = displacement_data * (pixel_size_um * 1e-6)
#
#     # Calculate pixel area in m²
#     pixel_area_m2 = (pixel_size_um * 1e-6) ** 2
#
#     # Calculate strain energy density for both calculated and ground truth
#     sed_calculated = calculate_strain_energy_density(displacement_m, calculated_traction)
#     sed_gt = calculate_strain_energy_density(displacement_m, ground_truth_traction)
#
#     # create simple square mask, excluding borders because of artifacts
#     mask = np.zeros_like(sed_calculated)
#     mask [10:-10, 10:-10] = 1
#
#     # Calculate total strain energies
#     total_se_calculated = calculate_total_strain_energy(sed_calculated, mask, pixel_area_m2)
#     total_se_gt = calculate_total_strain_energy(sed_gt, mask, pixel_area_m2)
#
#     return {
#         'total_se_calculated': total_se_calculated,
#         'total_se_gt': total_se_gt,
#         'mask_coverage': np.sum(mask) / mask.size
#     }


def plot_displacement(displacement_results):
    scenarios = ['low', 'mid', 'high']
    vmax_values = {'low': 0.3, 'mid': 3, 'high': 30}
    fig = plt.figure(figsize=(7.8, 3.8))

    # 2 rows x 3 cols for the maps only
    gs = gridspec.GridSpec(2, 3, figure=fig,
                          width_ratios=[1, 1, 1],
                          height_ratios=[1, 1],
                          wspace=0.15, hspace=0.15)

    fig.suptitle('Displacement Field Validation', fontsize=12, y=0.98)

    # Create axes for maps
    axes = [[fig.add_subplot(gs[row, col]) for col in range(3)] for row in range(2)]

    # Plot displacement fields for each scenario
    for i, scenario in enumerate(scenarios):
        if scenario in displacement_results:
            calculated = displacement_results[scenario]['calculated']
            ground_truth = displacement_results[scenario]['ground_truth']

            # Calculate magnitudes
            calc_magnitude = np.sqrt(calculated[:,:,0]**2 + calculated[:,:,1]**2)
            gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
            vmax = vmax_values[scenario]

            # Create coordinate grids for vector plotting
            h, w = calculated.shape[:2]
            step = max(h//40, w//40, 8)
            y, x = np.mgrid[0:h:step, 0:w:step]

            # Ground truth (top row)
            im1 = axes[0][i].imshow(gt_magnitude, cmap='viridis', vmin=0, vmax=vmax)
            axes[0][i].quiver(x, y, ground_truth[::step, ::step, 0],
                             -ground_truth[::step, ::step, 1],
                             color='white', scale_units='xy', scale=0.02*vmax, alpha=0.6)
            scenario_title = f'{scenario.capitalize()} Displacement'
            axes[0][i].set_title(scenario_title, fontsize=9)
            axes[0][i].set_xticks([])
            axes[0][i].set_yticks([])
            divider1 = make_axes_locatable(axes[0][i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            if i == 2:
                cbar1.set_label('Magnitude (pixels)', fontsize=7)
            cbar1.ax.tick_params(labelsize=6)

            # Calculated (bottom row)
            im2 = axes[1][i].imshow(calc_magnitude, cmap='viridis', vmin=0, vmax=vmax)
            axes[1][i].quiver(x, y, calculated[::step, ::step, 0],
                             -calculated[::step, ::step, 1],
                             color='white', scale_units='xy', scale=0.02*vmax, alpha=0.6)
            axes[1][i].set_title('', fontsize=8)
            axes[1][i].set_xticks([])
            axes[1][i].set_yticks([])
            divider2 = make_axes_locatable(axes[1][i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            if i == 2:
                cbar2.set_label('Magnitude (pixels)', fontsize=7)
            cbar2.ax.tick_params(labelsize=6)

    # Add row labels on the left side
    fig.text(0.12, 0.72, 'Ground Truth', rotation=90, va='center', ha='center', fontsize=9)
    fig.text(0.12, 0.28, 'Calculated', rotation=90, va='center', ha='center', fontsize=9)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    return fig


def plot_traction(fttc_results):
    scenarios = ['low', 'mid', 'high']
    vmax_values = {'low': 0.2, 'mid': 2, 'high': 20}
    fig = plt.figure(figsize=(7.8, 3.8))

    # 2 rows x 3 cols for the maps only
    gs = gridspec.GridSpec(2, 3, figure=fig,
                          width_ratios=[1, 1, 1],
                          height_ratios=[1, 1],
                          wspace=0.15, hspace=0.15)

    fig.suptitle('Traction Force Validation', fontsize=12)

    # Create axes for maps
    axes = [[fig.add_subplot(gs[row, col]) for col in range(3)] for row in range(2)]

    # Plot traction fields for each scenario
    for i, scenario in enumerate(scenarios):
        if scenario in fttc_results:
            calculated = fttc_results[scenario]['calculated'] * 1e-3  # convert to kPa
            ground_truth = fttc_results[scenario]['ground_truth'] * 1e-3

            # Calculate magnitudes
            if calculated is not None:
                calc_magnitude = np.sqrt(calculated[:,:,0]**2 + calculated[:,:,1]**2)
            else:
                calc_magnitude = np.zeros_like(ground_truth[:,:,0])
            gt_magnitude = np.sqrt(ground_truth[:,:,0]**2 + ground_truth[:,:,1]**2)
            vmax = vmax_values[scenario]

            # Create coordinate grids for vector plotting
            h, w = ground_truth.shape[:2]
            step = max(h//40, w//40, 8)
            y, x = np.mgrid[0:h:step, 0:w:step]

            # Ground truth (top row)
            im1 = axes[0][i].imshow(gt_magnitude, cmap='inferno', vmin=0, vmax=vmax)
            axes[0][i].quiver(x, y, ground_truth[::step, ::step, 0],
                             -ground_truth[::step, ::step, 1],
                             color='white', scale_units='xy', scale=0.02*vmax, alpha=0.6)
            scenario_title = f'{scenario.capitalize()} Traction'
            axes[0][i].set_title(scenario_title, fontsize=9)
            axes[0][i].set_xticks([])
            axes[0][i].set_yticks([])
            divider1 = make_axes_locatable(axes[0][i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            if i == 2:
                cbar1.set_label('Magnitude (kPa)', fontsize=7)
            cbar1.ax.tick_params(labelsize=6)

            # Calculated (bottom row)
            im2 = axes[1][i].imshow(calc_magnitude, cmap='inferno', vmin=0, vmax=vmax)
            if calculated is not None:
                axes[1][i].quiver(x, y, calculated[::step, ::step, 0],
                                 -calculated[::step, ::step, 1],
                                 color='white', scale_units='xy', scale=0.02*vmax, alpha=0.6)
            axes[1][i].set_title('', fontsize=8)
            axes[1][i].set_xticks([])
            axes[1][i].set_yticks([])
            divider2 = make_axes_locatable(axes[1][i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            if i == 2:
                cbar2.set_label('Magnitude (kPa)', fontsize=7)
            cbar2.ax.tick_params(labelsize=6)

    # Add row labels on the left side
    fig.text(0.12, 0.72, 'Ground Truth', rotation=90, va='center', ha='center', fontsize=9)
    fig.text(0.12, 0.28, 'Calculated', rotation=90, va='center', ha='center', fontsize=9)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    return fig


def plot_displacement_metrics(displacement_results):
    """
    Create combined displacement metrics figure:
    - Left: Correlation bar chart
    - Right: Displacement error vs GT magnitude
    """
    scenarios = ['low', 'mid', 'high']
    colors = {'low': '#1f77b4', 'mid': '#ff7f0e', 'high': '#2ca02c'}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    fig.suptitle('Displacement Analysis Metrics', fontsize=12, y=0.875)

    # --- Left panel: Correlation bar chart ---
    ax_corr = axes[0]
    correlations = [displacement_results[s]['displacement_correlation'] if s in displacement_results else 0 for s in scenarios]
    bar_colors = [colors[s] for s in scenarios]

    bars_corr = ax_corr.bar(scenarios, correlations, color=bar_colors, alpha=0.7)
    ax_corr.set_title('Displacement Correlation', fontsize=10)
    ax_corr.set_ylabel('Correlation Coefficient', fontsize=9)
    ax_corr.set_ylim(0, 1.15)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.tick_params(labelsize=8)
    for bar, corr in zip(bars_corr, correlations):
        ax_corr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{corr:.3f}', ha='center', va='bottom', fontsize=8)

    # --- Right panel: Displacement error vs magnitude ---
    ax_err = axes[1]

    # Collect all GT magnitudes to determine global bin edges
    all_gt_mags = []
    for scenario in scenarios:
        if scenario in displacement_results:
            gt_mags, _ = calculate_displacement_relative_error(
                displacement_results[scenario]['calculated'],
                displacement_results[scenario]['ground_truth']
            )
            all_gt_mags.extend(gt_mags)

    if all_gt_mags:
        # Create logarithmic bins spanning all scenarios with floor of 0.01
        all_gt_mags = np.array(all_gt_mags)
        min_mag = 0.01  # Floor for all scenarios
        max_mag = np.max(all_gt_mags)
        n_bins = 20
        bin_edges = np.logspace(np.log10(min_mag), np.log10(max_mag), n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log scale

        # Plot each scenario
        for scenario in scenarios:
            if scenario not in displacement_results:
                continue

            gt_mags, rel_errors = calculate_displacement_relative_error(
                displacement_results[scenario]['calculated'],
                displacement_results[scenario]['ground_truth']
            )

            # Bin the data
            bin_means = []
            bin_stds = []
            valid_centers = []

            for i in range(len(bin_edges) - 1):
                mask = (gt_mags >= bin_edges[i]) & (gt_mags < bin_edges[i + 1])
                if np.sum(mask) >= 10:  # Require at least 10 points per bin
                    bin_means.append(np.mean(rel_errors[mask]))
                    bin_stds.append(np.std(rel_errors[mask]))
                    valid_centers.append(bin_centers[i])

            if not valid_centers:
                continue

            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)
            valid_centers = np.array(valid_centers)

            # Plot mean line with std ribbon
            ax_err.plot(valid_centers, bin_means, '-', color=colors[scenario],
                        label=f'{scenario.capitalize()}', linewidth=2)
            ax_err.fill_between(valid_centers, bin_means - bin_stds, bin_means + bin_stds,
                               color=colors[scenario], alpha=0.2)

        # Add reference line at y=0 (perfect estimation)
        ax_err.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax_err.set_xscale('log')
    ax_err.set_xlabel('GT Displacement Magnitude (pixels)', fontsize=9)
    ax_err.set_ylabel('Relative Error\n(|calc| - |GT|) / |GT|', fontsize=9)
    ax_err.set_title('Displacement Error vs GT Magnitude', fontsize=10)
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.tick_params(labelsize=8)

    # Add annotation explaining the metric
    ax_err.text(0.02, 0.98, '+ = overestimation\n− = underestimation',
                transform=ax_err.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_traction_metrics(fttc_results):
    """
    Create combined traction metrics figure:
    - First column: Correlation bar chart
    - Remaining columns: Sabass metrics (DTM, DTMS, DTA)
    """
    scenarios = ['low', 'mid', 'high']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.7))
    fig.suptitle('Traction Force Analysis Metrics', fontsize=12, y=0.8)

    # --- First panel: Correlation bar chart ---
    ax_corr = axes[0]
    correlations = [fttc_results[s]['traction_correlation'] if s in fttc_results else 0 for s in scenarios]

    bars_corr = ax_corr.bar(scenarios, correlations, color=colors, alpha=0.7)
    ax_corr.set_title('Traction \n Correlation', fontsize=9)
    ax_corr.set_ylabel('Correlation Coefficient', fontsize=8)
    ax_corr.set_ylim(0, 1.15)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.tick_params(labelsize=7)
    for bar, corr in zip(bars_corr, correlations):
        ax_corr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{corr:.3f}', ha='center', va='bottom', fontsize=6)

    # --- Sabass metrics panels ---
    dtm_values = [fttc_results[s]['traction_dtm'] if s in fttc_results else np.nan for s in scenarios]
    dtms_values = [fttc_results[s]['traction_dtms'] if s in fttc_results else np.nan for s in scenarios]
    dta_values = [fttc_results[s]['traction_dta'] if s in fttc_results else np.nan for s in scenarios]

    # DTM plot
    ax_dtm = axes[1]
    bars_dtm = ax_dtm.bar(scenarios, dtm_values, color=colors, alpha=0.7)
    # ax_dtm.set_title('DTM', fontsize=9, y=1.2)
    ax_dtm.text(0.5, 1.05, 'Deviation of \n Traction Magnitude', transform=ax_dtm.transAxes,
                fontsize=9, ha='center', va='bottom')
    ax_dtm.set_ylabel('DTM', fontsize=8)
    max_abs_dtm = max([abs(v) for v in dtm_values if not np.isnan(v)], default=0.5) * 1.3
    ax_dtm.set_ylim(-max_abs_dtm, max_abs_dtm)
    ax_dtm.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_dtm.grid(True, alpha=0.3)
    ax_dtm.tick_params(labelsize=7)
    for bar, dtm in zip(bars_dtm, dtm_values):
        if not np.isnan(dtm):
            va = 'bottom' if dtm >= 0 else 'top'
            offset = max_abs_dtm * 0.05 if dtm >= 0 else -max_abs_dtm * 0.05
            ax_dtm.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                       f'{dtm:+.3f}', ha='center', va=va, fontsize=6)
    ax_dtm.set_xlabel('- = underest., + = overest.', fontsize=7)

    # DTMS plot
    ax_dtms = axes[2]
    bars_dtms = ax_dtms.bar(scenarios, dtms_values, color=colors, alpha=0.7)
    # ax_dtms.set_title('DTMS', fontsize=9, y=1.2)
    ax_dtms.text(0.5, 1.05, 'Traction in \n Surrounding', transform=ax_dtms.transAxes,
                 fontsize=9, ha='center', va='bottom')
    ax_dtms.set_ylabel('DTMS', fontsize=8)
    max_dtms = max([v for v in dtms_values if not np.isnan(v)], default=1) * 1.2
    ax_dtms.set_ylim(0, max(1.0, max_dtms))
    ax_dtms.grid(True, alpha=0.3)
    ax_dtms.tick_params(labelsize=7)
    for bar, dtms in zip(bars_dtms, dtms_values):
        if not np.isnan(dtms):
            ax_dtms.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{dtms:.3f}', ha='center', va='bottom', fontsize=6)
    ax_dtms.set_xlabel('0 = optimal, 1 = worst', fontsize=7)

    # DTA plot
    ax_dta = axes[3]
    bars_dta = ax_dta.bar(scenarios, dta_values, color=colors, alpha=0.7)
    # ax_dta.set_title('DTA', fontsize=9, y=1.2)
    ax_dta.text(0.5, 1.05, 'Deviation of \n Traction Angle', transform=ax_dta.transAxes,
                fontsize=9, ha='center', va='bottom')
    ax_dta.set_ylabel('DTA (°)', fontsize=8)
    max_dta = max([v for v in dta_values if not np.isnan(v)], default=90) * 1.2
    ax_dta.set_ylim(0, min(90, max_dta))
    ax_dta.grid(True, alpha=0.3)
    ax_dta.tick_params(labelsize=7)
    for bar, dta in zip(bars_dta, dta_values):
        if not np.isnan(dta):
            ax_dta.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{dta:.1f}°', ha='center', va='bottom', fontsize=6)
    ax_dta.set_xlabel('0° = perfect alignment', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# def plot_sabass_metrics(fttc_results):
#     """
#     Create plot for Sabass et al. 2008 metrics: DTM, DTMS, DTA.

#     These metrics are computed per-adhesion and provide insight into:
#     - DTM: Magnitude accuracy (under/overestimation)
#     - DTMS: Edge/contour reconstruction quality
#     - DTA: Angular accuracy of traction vectors
#     """
#     scenarios = ['low', 'mid', 'high']

#     fig, axes = plt.subplots(1, 3, figsize=(9, 3))
#     fig.suptitle('Sabass et al. 2008 Metrics', fontsize=12)

#     dtm_values = [fttc_results[s]['traction_dtm'] if s in fttc_results else np.nan for s in scenarios]
#     dtms_values = [fttc_results[s]['traction_dtms'] if s in fttc_results else np.nan for s in scenarios]
#     dta_values = [fttc_results[s]['traction_dta'] if s in fttc_results else np.nan for s in scenarios]
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

#     # DTM plot
#     ax_dtm = axes[0]
#     bars_dtm = ax_dtm.bar(scenarios, dtm_values, color=colors, alpha=0.7)
#     ax_dtm.set_title('DTM\n(Deviation of Traction Magnitude)', fontsize=9)
#     ax_dtm.set_ylabel('DTM', fontsize=8)
#     max_abs_dtm = max([abs(v) for v in dtm_values if not np.isnan(v)], default=0.5) * 1.3
#     ax_dtm.set_ylim(-max_abs_dtm, max_abs_dtm)
#     ax_dtm.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
#     ax_dtm.grid(True, alpha=0.3)
#     ax_dtm.tick_params(labelsize=7)
#     for bar, dtm in zip(bars_dtm, dtm_values):
#         if not np.isnan(dtm):
#             va = 'bottom' if dtm >= 0 else 'top'
#             offset = max_abs_dtm * 0.05 if dtm >= 0 else -max_abs_dtm * 0.05
#             ax_dtm.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
#                        f'{dtm:+.3f}', ha='center', va=va, fontsize=7)
#     ax_dtm.set_xlabel('0 = perfect, - = underest., + = overest.', fontsize=7)

#     # DTMS plot
#     ax_dtms = axes[1]
#     bars_dtms = ax_dtms.bar(scenarios, dtms_values, color=colors, alpha=0.7)
#     ax_dtms.set_title('DTMS\n(Traction in Surrounding)', fontsize=9)
#     ax_dtms.set_ylabel('DTMS', fontsize=8)
#     max_dtms = max([v for v in dtms_values if not np.isnan(v)], default=1) * 1.2
#     ax_dtms.set_ylim(0, max(1.0, max_dtms))
#     ax_dtms.grid(True, alpha=0.3)
#     ax_dtms.tick_params(labelsize=7)
#     for bar, dtms in zip(bars_dtms, dtms_values):
#         if not np.isnan(dtms):
#             ax_dtms.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
#                         f'{dtms:.3f}', ha='center', va='bottom', fontsize=7)
#     ax_dtms.set_xlabel('0 = optimal, 1 = worst', fontsize=7)

#     # DTA plot
#     ax_dta = axes[2]
#     bars_dta = ax_dta.bar(scenarios, dta_values, color=colors, alpha=0.7)
#     ax_dta.set_title('DTA\n(Deviation of Traction Angle)', fontsize=9)
#     ax_dta.set_ylabel('DTA (°)', fontsize=8)
#     max_dta = max([v for v in dta_values if not np.isnan(v)], default=90) * 1.2
#     ax_dta.set_ylim(0, min(90, max_dta))
#     ax_dta.grid(True, alpha=0.3)
#     ax_dta.tick_params(labelsize=7)
#     for bar, dta in zip(bars_dta, dta_values):
#         if not np.isnan(dta):
#             ax_dta.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
#                        f'{dta:.1f}°', ha='center', va='bottom', fontsize=7)
#     ax_dta.set_xlabel('0° = perfect alignment', fontsize=7)

#     plt.tight_layout()
#     return fig


# def plot_strain_energy_comparison(fttc_results):
#     """Create strain energy comparison figure: GT vs calculated."""
#     scenarios = ['low', 'mid', 'high']
#
#     # Extract strain energy values
#     se_gt_values = [fttc_results[s]['strain_energy_gt'] if s in fttc_results and fttc_results[s]['strain_energy_gt'] > 0 else 1e-20 for s in scenarios]
#     se_calc_values = [fttc_results[s]['strain_energy_calc'] if s in fttc_results and fttc_results[s]['strain_energy_calc'] > 0 else 1e-20 for s in scenarios]
#
#     fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
#
#     x = np.arange(len(scenarios))
#     width = 0.35
#
#     bars1 = ax.bar(x - width/2, se_gt_values, width, label='Ground Truth',
#                   color='#1f77b4', alpha=0.7)
#     bars2 = ax.bar(x + width/2, se_calc_values, width, label='Calculated',
#                   color='#ff7f0e', alpha=0.7)
#
#     # Add value labels on bars (in scientific notation)
#     for bars in [bars1, bars2]:
#         for bar in bars:
#             height = bar.get_height()
#             if height > 1e-19:
#                 ax.text(bar.get_x() + bar.get_width()/2, height * 1.2,
#                        f'{height:.1e}', ha='center', va='bottom', fontsize=6, rotation=45)
#
#     ax.set_xlabel('Scenario', fontsize=8)
#     ax.set_ylabel('Strain Energy (J)', fontsize=8)
#     ax.set_title('Strain Energy Comparison\nGround Truth vs Calculated', fontsize=10)
#     ax.set_xticks(x)
#     ax.set_xticklabels([s.upper() for s in scenarios])
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.3, axis='y')
#     ax.set_yscale('log')
#     ax.set_ylim(0, 1e-13)
#     ax.tick_params(labelsize=6)
#
#     plt.tight_layout()
#     return fig


# def plot_displacement_error_vs_magnitude(displacement_results):
#     """
#     Create binned line plot of displacement relative error vs GT magnitude.

#     X-axis: GT displacement magnitude (log scale, floor at 0.01)
#     Y-axis: Signed relative error = (|calc| - |GT|) / |GT|
#     Shows mean ± std as line with shaded ribbon for each scenario.
#     """
#     scenarios = ['low', 'mid', 'high']
#     colors = {'low': '#1f77b4', 'mid': '#ff7f0e', 'high': '#2ca02c'}

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4))

#     # Collect all GT magnitudes to determine global bin edges
#     all_gt_mags = []
#     for scenario in scenarios:
#         if scenario in displacement_results:
#             gt_mags, _ = calculate_displacement_relative_error(
#                 displacement_results[scenario]['calculated'],
#                 displacement_results[scenario]['ground_truth']
#             )
#             all_gt_mags.extend(gt_mags)

#     if not all_gt_mags:
#         return fig

#     # Create logarithmic bins spanning all scenarios with floor of 0.01
#     all_gt_mags = np.array(all_gt_mags)
#     min_mag = 0.01  # Floor for all scenarios
#     max_mag = np.max(all_gt_mags)
#     n_bins = 20
#     bin_edges = np.logspace(np.log10(min_mag), np.log10(max_mag), n_bins + 1)
#     bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log scale

#     # Plot each scenario
#     for scenario in scenarios:
#         if scenario not in displacement_results:
#             continue

#         gt_mags, rel_errors = calculate_displacement_relative_error(
#             displacement_results[scenario]['calculated'],
#             displacement_results[scenario]['ground_truth']
#         )

#         # Bin the data
#         bin_means = []
#         bin_stds = []
#         valid_centers = []

#         for i in range(len(bin_edges) - 1):
#             mask = (gt_mags >= bin_edges[i]) & (gt_mags < bin_edges[i + 1])
#             if np.sum(mask) >= 10:  # Require at least 10 points per bin
#                 bin_means.append(np.mean(rel_errors[mask]))
#                 bin_stds.append(np.std(rel_errors[mask]))
#                 valid_centers.append(bin_centers[i])

#         if not valid_centers:
#             continue

#         bin_means = np.array(bin_means)
#         bin_stds = np.array(bin_stds)
#         valid_centers = np.array(valid_centers)

#         # Plot mean line with std ribbon
#         ax.plot(valid_centers, bin_means, '-', color=colors[scenario],
#                 label=f'{scenario.capitalize()}', linewidth=2)
#         ax.fill_between(valid_centers, bin_means - bin_stds, bin_means + bin_stds,
#                        color=colors[scenario], alpha=0.2)

#     # Add reference line at y=0 (perfect estimation)
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

#     ax.set_xscale('log')
#     ax.set_xlabel('GT Displacement Magnitude (pixels)', fontsize=10)
#     ax.set_ylabel('Relative Error\n(|calc| - |GT|) / |GT|', fontsize=10)
#     ax.set_title('Displacement Error vs Ground Truth Magnitude', fontsize=11)
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)
#     ax.tick_params(labelsize=8)

#     # Add annotation explaining the metric
#     ax.text(0.02, 0.98, '+ = overestimation\n− = underestimation',
#             transform=ax.transAxes, fontsize=8, va='top', ha='left',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#     plt.tight_layout()
#     return fig


# def plot_normalized_strain_energy(fttc_results):
#     """Create normalized strain energy plot: calculated/ground_truth ratio."""
#     scenarios = ['low', 'mid', 'high']
#
#     # Calculate normalized strain energies (calculated/ground_truth)
#     normalized_values = []
#     for scenario in scenarios:
#         if scenario in fttc_results:
#             se_gt = fttc_results[scenario]['strain_energy_gt']
#             se_calc = fttc_results[scenario]['strain_energy_calc']
#             if se_gt > 0:
#                 normalized_values.append(se_calc / se_gt)
#             else:
#                 normalized_values.append(0)
#         else:
#             normalized_values.append(0)
#
#     fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
#
#     # Create bar plot
#     bars = ax.bar(scenarios, normalized_values,
#                   color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.7)
#
#     # Add value labels on bars
#     for bar, value in zip(bars, normalized_values):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
#                f'{value:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
#
#     # Add horizontal reference line at y=1 (perfect match)
#     ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2,
#                label='Perfect Match (Calc/GT = 1.0)')
#
#     ax.set_xlabel('Scenario', fontsize=8)
#     ax.set_ylabel('Normalized Strain Energy\n(Calculated / Ground Truth)', fontsize=8)
#     ax.set_title('Normalized Strain Energy\nCalculated / Ground Truth', fontsize=10)
#     ax.set_xticklabels([s.upper() for s in scenarios])
#     ax.grid(True, alpha=0.3, axis='y')
#     ax.legend(fontsize=8)
#     ax.tick_params(labelsize=6)
#
#     # Set y-axis limits to show values clearly
#     y_max = max(normalized_values) if normalized_values else 1
#     ax.set_ylim(0, max(1.2, y_max * 1.1))
#
#     plt.tight_layout()
#     return fig


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

    # Load ground truth displacement
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
    regularization_info = f"{params.regularization}"
    print(f"  Using parameters: E={params.young_modulus} Pa, nu={params.poisson_ratio_substrate}, "
          f"regularization={regularization_info}, pixel_size={params.pixel_size} µm")

    # Load ground truth traction
    gt_trac_x, gt_trac_y = load_ground_truth_traction(scenario_folder)
    ground_truth = np.stack([gt_trac_x, gt_trac_y], axis=-1)
    print(f"  Ground truth traction shape: {ground_truth.shape}")

    # Parse adhesion config for DTMS and DTA metrics
    config_path = os.path.join(scenario_folder, 'dipole_config.toml')
    adhesions = []
    if os.path.exists(config_path):
        adhesions, _ = parse_adhesion_config(config_path)
        print(f"  Found {len(adhesions)} adhesion sites in config")

        # DEBUG: Plot adhesion masks - uncomment when debugging
        # debug_save_path = os.path.join(scenario_folder, f'debug_adhesion_masks.png')
        # debug_plot_adhesion_masks(ground_truth, adhesions, ring_width_px=15, save_path=debug_save_path)

    # Calculate traction from calculated displacement only
    calculated_trac = None
    traction_correlation = 0
    traction_dtm = np.nan
    traction_dtms = np.nan
    traction_dta = np.nan
    # strain_energy_gt = 0
    # strain_energy_calc = 0

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

        # Calculate DTM (Deviation of Traction Magnitude) per Sabass et al. 2008
        traction_dtm = calculate_dtm_metric(calculated_trac, ground_truth)

        # Calculate DTMS and DTA if adhesion regions are defined
        if len(adhesions) > 0:
            traction_dtms = calculate_dtms_metric(calculated_trac, adhesions)
            traction_dta = calculate_dta_metric(calculated_trac, ground_truth, adhesions)

        # # Calculate strain energies
        # # Load ground truth displacement for strain energy calculation
        # gt_disp_x, gt_disp_y = load_displacement_data(scenario_folder)
        # gt_displacement = np.stack([gt_disp_x, gt_disp_y], axis=-1)
        #
        # # GT strain energy
        # se_metrics_gt = calculate_strain_energy_metrics(
        #     gt_displacement, ground_truth, ground_truth, params.pixel_size
        # )
        # strain_energy_gt = se_metrics_gt['total_se_calculated']
        #
        # # Calculated strain energy
        # disp_for_se = displacement_flow * params.pixel_size
        # se_metrics_calc = calculate_strain_energy_metrics(
        #     disp_for_se, calculated_trac, ground_truth, params.pixel_size
        # )
        # strain_energy_calc = se_metrics_calc['total_se_calculated']

        print("  FTTC Metrics:")
        print(f"    Traction Correlation: {traction_correlation:.3f}")
        print(f"    Traction DTM: {traction_dtm:+.3f}")
        print(f"    Traction DTMS: {traction_dtms:.3f}")
        print(f"    Traction DTA: {traction_dta:.1f}°")
        # print(f"    Strain Energy - GT: {strain_energy_gt:.2e} J")
        # print(f"    Strain Energy - Calc: {strain_energy_calc:.2e} J")

    return {
        'calculated_trac': calculated_trac,
        'ground_truth_trac': ground_truth,
        'traction_correlation': traction_correlation,
        'traction_dtm': traction_dtm,
        'traction_dtms': traction_dtms,
        'traction_dta': traction_dta,
        # 'strain_energy_gt': strain_energy_gt,
        # 'strain_energy_calc': strain_energy_calc
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

            # Validate FTTC analysis
            trac_results = validate_fttc_scenario(str(scenario_path), displacement_flow)

            fttc_results[scenario] = {
                'calculated': trac_results['calculated_trac'],
                'ground_truth': trac_results['ground_truth_trac'],
                'traction_correlation': trac_results['traction_correlation'],
                'traction_dtm': trac_results['traction_dtm'],
                'traction_dtms': trac_results['traction_dtms'],
                'traction_dta': trac_results['traction_dta'],
                # 'strain_energy_gt': trac_results['strain_energy_gt'],
                # 'strain_energy_calc': trac_results['strain_energy_calc']
            }
        else:
            print(f"Warning: Scenario folder {scenario_path} not found")
    
    # Create and save displacement plots
    if displacement_results:
        print("\n--- Creating validation plots ---")
        
        # Create displacement plot
        disp_fig = plot_displacement(displacement_results)
        disp_path = Path(__file__).parent / "displacement.png"
        disp_fig.savefig(disp_path, dpi=300, bbox_inches='tight')
        print(f"  Saved displacement plot: {disp_path}")
        plt.close(disp_fig)
        
        # Create traction plot
        trac_fig = plot_traction(fttc_results)
        trac_path = Path(__file__).parent / "traction.png"
        trac_fig.savefig(trac_path, dpi=300, bbox_inches='tight')
        print(f"  Saved traction plot: {trac_path}")
        plt.close(trac_fig)

        # Create displacement metrics plot (correlation + error vs magnitude)
        disp_metrics_fig = plot_displacement_metrics(displacement_results)
        disp_metrics_path = Path(__file__).parent / "displacement_metrics.png"
        disp_metrics_fig.savefig(disp_metrics_path, dpi=300, bbox_inches='tight')
        print(f"  Saved displacement metrics plot: {disp_metrics_path}")
        plt.close(disp_metrics_fig)

        # Create traction metrics plot (correlation + Sabass metrics)
        trac_metrics_fig = plot_traction_metrics(fttc_results)
        trac_metrics_path = Path(__file__).parent / "traction_metrics.png"
        trac_metrics_fig.savefig(trac_metrics_path, dpi=300, bbox_inches='tight')
        print(f"  Saved traction metrics plot: {trac_metrics_path}")
        plt.close(trac_metrics_fig)

        # # Create strain energy comparison plot
        # strain_energy_fig = plot_strain_energy_comparison(fttc_results)
        # strain_energy_path = Path(__file__).parent / "strain_energy_comparison.png"
        # strain_energy_fig.savefig(strain_energy_path, dpi=300, bbox_inches='tight')
        # print(f"  Saved strain energy comparison plot: {strain_energy_path}")
        # plt.close(strain_energy_fig)
        #
        # # Create normalized strain energy plot
        # normalized_se_fig = plot_normalized_strain_energy(fttc_results)
        # normalized_se_path = Path(__file__).parent / "normalized_strain_energy.png"
        # normalized_se_fig.savefig(normalized_se_path, dpi=300, bbox_inches='tight')
        # print(f"  Saved normalized strain energy plot: {normalized_se_path}")
        # plt.close(normalized_se_fig)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    # # Add normalized strain energy summary
    # print("\nNORMALIZED STRAIN ENERGY RATIOS (Calculated/Ground Truth):")
    # for scenario in displacement_results.keys():
    #     if scenario in fttc_results:
    #         trac_results = fttc_results[scenario]
    #         if trac_results['strain_energy_gt'] > 0:
    #             normalized_se = trac_results['strain_energy_calc'] / trac_results['strain_energy_gt']
    #             print(f"  {scenario.upper()}: {normalized_se:.3f}")
    #         else:
    #             print(f"  {scenario.upper()}: N/A (GT = 0)")
    # print("  (Values close to 1.0 indicate good agreement)")

    # Add DTM summary
    print("\nDEVIATION OF TRACTION MAGNITUDE (DTM):")
    for scenario in displacement_results.keys():
        if scenario in fttc_results:
            trac_results = fttc_results[scenario]
            dtm = trac_results['traction_dtm']
            if not np.isnan(dtm):
                print(f"  {scenario.upper()}: {dtm:+.3f} ({dtm*100:+.1f}%)")
            else:
                print(f"  {scenario.upper()}: N/A")
    print("  (0 = perfect, negative = underestimation, positive = overestimation)")

    # Add DTMS summary
    print("\nDEVIATION OF TRACTION MAGNITUDE IN SURROUNDING (DTMS):")
    for scenario in displacement_results.keys():
        if scenario in fttc_results:
            trac_results = fttc_results[scenario]
            dtms = trac_results['traction_dtms']
            if not np.isnan(dtms):
                print(f"  {scenario.upper()}: {dtms:.3f}")
            else:
                print(f"  {scenario.upper()}: N/A")
    print("  (0 = optimal edge reconstruction, 1 = worst)")

    # Add DTA summary
    print("\nDEVIATION OF TRACTION ANGLE (DTA):")
    for scenario in displacement_results.keys():
        if scenario in fttc_results:
            trac_results = fttc_results[scenario]
            dta = trac_results['traction_dta']
            if not np.isnan(dta):
                print(f"  {scenario.upper()}: {dta:.1f}°")
            else:
                print(f"  {scenario.upper()}: N/A")
    print("  (0° = perfect angular alignment)\n")
    
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
            print(f"    Traction DTM: {trac_results['traction_dtm']:+.3f}")
            print(f"    Traction DTMS: {trac_results['traction_dtms']:.3f}")
            print(f"    Traction DTA: {trac_results['traction_dta']:.1f}°")
            # print(f"    Strain Energy - GT: {trac_results['strain_energy_gt']:.2e} J")
            # print(f"    Strain Energy - Calc: {trac_results['strain_energy_calc']:.2e} J")
            # # Calculate and display normalized strain energy
            # if trac_results['strain_energy_gt'] > 0:
            #     normalized_se = trac_results['strain_energy_calc'] / trac_results['strain_energy_gt']
            #     print(f"    Normalized Strain Energy (Calc/GT): {normalized_se:.3f}")
            # else:
            #     print(f"    Normalized Strain Energy (Calc/GT): N/A (GT = 0)")


if __name__ == "__main__":
    main()
