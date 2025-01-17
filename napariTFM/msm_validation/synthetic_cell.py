from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napariTFM.msm import MonolayerStressMicroscopy


def plot_validation_results_common_scale(t_x, t_y, sigma_xx_true, sigma_yy_true,
                                         sigma_xx_calc, sigma_yy_calc, mask):
    """Plot comparison between true and calculated stress fields with common scale"""
    fig_stress = plt.figure(figsize=(12, 10))
    plt.suptitle('Stress Tensor Validation (Synthetic Cell) - Common Scale', fontsize=14)

    ax_s1 = plt.subplot(221)
    ax_s2 = plt.subplot(222)
    ax_s3 = plt.subplot(223)
    ax_s4 = plt.subplot(224)

    # Get common scale for stress components
    vmax_xx = np.nanpercentile(np.abs(sigma_xx_true[mask]), 99)
    vmax_yy = np.nanpercentile(np.abs(sigma_yy_true[mask]), 99)

    def plot_component(ax, data, title, vmax):
        masked_data = np.copy(data)
        masked_data[~mask] = np.nan
        im = ax.imshow(masked_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        return im

    # Plot stress components
    im5 = plot_component(ax_s1, sigma_xx_true, 'True σxx (Warped)', vmax_xx)
    plt.colorbar(im5, ax=ax_s1)

    im6 = plot_component(ax_s2, sigma_xx_calc, 'Calculated σxx', vmax_xx)
    plt.colorbar(im6, ax=ax_s2)

    im7 = plot_component(ax_s3, sigma_yy_true, 'True σyy (Warped)', vmax_yy)
    plt.colorbar(im7, ax=ax_s3)

    im8 = plot_component(ax_s4, sigma_yy_calc, 'Calculated σyy', vmax_yy)
    plt.colorbar(im8, ax=ax_s4)

    plt.tight_layout()
    return fig_stress


def plot_mesh(nodes, elements, mask):
    """Plot triangular mesh overlaid on mask"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot mask as background
    ax.imshow(mask, cmap='gray', alpha=0.3)

    # Plot triangular elements
    for element in elements:
        vertices = nodes[element]
        ax.plot([vertices[0][0], vertices[1][0]], [vertices[0][1], vertices[1][1]], 'b-', linewidth=0.5)
        ax.plot([vertices[1][0], vertices[2][0]], [vertices[1][1], vertices[2][1]], 'b-', linewidth=0.5)
        ax.plot([vertices[2][0], vertices[0][0]], [vertices[2][1], vertices[0][1]], 'b-', linewidth=0.5)

    ax.set_aspect('equal')
    return fig


def calculate_metrics(true, calc, mask):
    """Calculate error metrics between true and calculated fields"""
    valid_mask = mask & ~np.isnan(true) & ~np.isnan(calc)
    rmse = np.sqrt(np.mean((true[valid_mask] - calc[valid_mask]) ** 2))
    max_error = np.max(np.abs(true[valid_mask] - calc[valid_mask]))
    correlation = np.corrcoef(true[valid_mask].flatten(), calc[valid_mask].flatten())[0, 1]
    rel_error = rmse / np.std(true[valid_mask])
    return rmse, max_error, correlation, rel_error


def validate_msm_synthetic_cell(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, msm):
    """Validate MSM implementation using synthetic cell data"""
    # Plot mesh
    mesh_fig = plot_mesh(msm.nodes, msm.elements, mask)
    plt.title('Triangular Mesh (Synthetic Cell)')
    plt.show()

    # Calculate stress tensor and get numerical metrics
    stress_tensor_calc, condition_number, residual = msm.calculate_stress_field(t_x, t_y)
    sigma_xx_calc = stress_tensor_calc[:, :, 0, 0]
    sigma_yy_calc = stress_tensor_calc[:, :, 1, 1]

    # Print numerical quality metrics
    print("\nNumerical Quality Metrics:")
    print("-" * 50)
    print(f"Condition Number: {condition_number:.2e}")
    print(f"Residual Norm: {residual:.2e}")

    # Create visualizations
    fig_stress_common = plot_validation_results_common_scale(
        t_x, t_y, sigma_xx_true, sigma_yy_true,
        sigma_xx_calc, sigma_yy_calc, mask
    )
    plt.show()

    # Print validation metrics
    print("\nValidation Metrics (Synthetic Cell):")
    print("-" * 50)

    all_metrics = {}  # Dictionary to store all metrics

    for comp, true, calc in [
        ('σxx', sigma_xx_true, sigma_xx_calc),
        ('σyy', sigma_yy_true, sigma_yy_calc)
    ]:
        rmse, max_error, corr, rel_error = calculate_metrics(true, calc, mask)
        print(f"\n{comp}:")
        print(f"RMSE: {rmse:.2e}")
        print(f"Max Error: {max_error:.2e}")
        print(f"Correlation: {corr:.4f}")
        print(f"Relative Error: {rel_error:.4f}")

        # Store metrics
        all_metrics[comp] = {
            'rmse': rmse,
            'max_error': max_error,
            'correlation': corr,
            'relative_error': rel_error
        }

    # Add numerical metrics to dictionary
    all_metrics['numerical'] = {
        'condition_number': condition_number,
        'residual': residual
    }

    # Save metrics to file
    metrics_file = synthetic_cell_dir / 'validation_metrics.npz'
    np.savez(metrics_file, **all_metrics)
    print(f"\nMetrics saved to: {metrics_file}")

    return all_metrics


if __name__ == "__main__":
    # Get the current directory and construct path to synthetic_cell
    current_dir = Path(__file__).parent
    synthetic_cell_dir = current_dir.parent / 'benchmarks' / 'synthetic_cell_physical_units'

    # Load data
    t_x = np.load(synthetic_cell_dir / 'Traction_x_warped.npy')
    t_y = np.load(synthetic_cell_dir / 'Traction_y_warped.npy')
    sigma_xx_true = np.load(synthetic_cell_dir / 'Stress_xx_warped.npy')
    sigma_yy_true = np.load(synthetic_cell_dir / 'Stress_yy_warped.npy')

    # Create mask based on stress magnitude
    mask = np.abs(sigma_xx_true) > 0

    # Initialize MSM calculator
    msm = MonolayerStressMicroscopy(
        mask=mask,
        pixelsize=0.3*1e-6,
        density_factor=0.0025,  # Finer mesh
        algorithm=6,  # MeshAdapt algorithm
        use_optimization=False,  # Enable Netgen optimization
        youngs_modulus=1
    )

    # Run validation and get metrics
    metrics = validate_msm_synthetic_cell(
        t_x, t_y,
        sigma_xx_true, sigma_yy_true,
        mask,
        msm
    )
