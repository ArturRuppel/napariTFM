from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napariTFM.msm import MonolayerStressMicroscopy
from napariTFM.msm_pyTFM import MonolayerStressMicroscopy as MSM_pyTFM


def calculate_correlation(true, calc, mask):
    """Calculate correlation coefficient between true and calculated fields"""
    valid_mask = mask & ~np.isnan(true) & ~np.isnan(calc)
    return np.corrcoef(true[valid_mask].flatten(), calc[valid_mask].flatten())[0, 1]


def plot_validation_results_comparison(t_x, t_y, sigma_xx_true, sigma_yy_true,
                                       sigma_xx_calc, sigma_yy_calc,
                                       sigma_xx_calc_pytfm, sigma_yy_calc_pytfm, mask):
    """Plot comparison between true and calculated stress fields from both implementations"""
    fig_stress = plt.figure(figsize=(15, 10))
    plt.suptitle('Stress Tensor Validation (Synthetic Cell) - Common Scale', fontsize=14)

    ax_s1 = plt.subplot(231)
    ax_s2 = plt.subplot(232)
    ax_s3 = plt.subplot(233)
    ax_s4 = plt.subplot(234)
    ax_s5 = plt.subplot(235)
    ax_s6 = plt.subplot(236)

    # Get common scale for stress components
    vmax_xx = np.nanpercentile(np.abs(sigma_xx_true[mask]), 99)
    vmax_yy = np.nanpercentile(np.abs(sigma_yy_true[mask]), 99)

    # Calculate correlations
    corr_xx_msm = calculate_correlation(sigma_xx_true, sigma_xx_calc, mask)
    corr_xx_pytfm = calculate_correlation(sigma_xx_true, sigma_xx_calc_pytfm, mask)
    corr_yy_msm = calculate_correlation(sigma_yy_true, sigma_yy_calc, mask)
    corr_yy_pytfm = calculate_correlation(sigma_yy_true, sigma_yy_calc_pytfm, mask)

    def plot_component(ax, data, title, vmax, corr=None):
        masked_data = np.copy(data)
        masked_data[~mask] = np.nan
        im = ax.imshow(masked_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if corr is not None:
            title = f'{title}\nr = {corr:.3f}'
        ax.set_title(title)
        return im

    # Plot σxx components
    im1 = plot_component(ax_s1, sigma_xx_true, 'True σxx (Warped)', vmax_xx)
    plt.colorbar(im1, ax=ax_s1)

    im2 = plot_component(ax_s2, sigma_xx_calc, 'napariTFM σxx', vmax_xx, corr_xx_msm)
    plt.colorbar(im2, ax=ax_s2)

    im3 = plot_component(ax_s3, sigma_xx_calc_pytfm, 'pyTFM σxx', vmax_xx, corr_xx_pytfm)
    plt.colorbar(im3, ax=ax_s3)

    # Plot σyy components
    im4 = plot_component(ax_s4, sigma_yy_true, 'True σyy (Warped)', vmax_yy)
    plt.colorbar(im4, ax=ax_s4)

    im5 = plot_component(ax_s5, sigma_yy_calc, 'napariTFM σyy', vmax_yy, corr_yy_msm)
    plt.colorbar(im5, ax=ax_s5)

    im6 = plot_component(ax_s6, sigma_yy_calc_pytfm, 'pyTFM σyy', vmax_yy, corr_yy_pytfm)
    plt.colorbar(im6, ax=ax_s6)

    plt.tight_layout()
    return fig_stress


def calculate_metrics(true, calc, mask):
    """Calculate error metrics between true and calculated fields"""
    valid_mask = mask & ~np.isnan(true) & ~np.isnan(calc)
    rmse = np.sqrt(np.mean((true[valid_mask] - calc[valid_mask]) ** 2))
    max_error = np.max(np.abs(true[valid_mask] - calc[valid_mask]))
    correlation = np.corrcoef(true[valid_mask].flatten(), calc[valid_mask].flatten())[0, 1]
    rel_error = rmse / np.std(true[valid_mask])
    return rmse, max_error, correlation, rel_error


def validate_msm_synthetic_cell(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, msm, msm_pytfm):
    """Validate both MSM implementations using synthetic cell data"""
    # Calculate stress tensor with MSM implementation
    stress_tensor_calc, condition_number, residual = msm.calculate_stress_field(t_x, t_y)
    sigma_xx_calc = stress_tensor_calc[:, :, 0, 0]
    sigma_yy_calc = stress_tensor_calc[:, :, 1, 1]

    # Calculate stress tensor with pyTFM implementation
    stress_tensor_pytfm = msm_pytfm.calculate_stress_field(t_x, t_y, mask)
    sigma_xx_calc_pytfm = stress_tensor_pytfm[:, :, 0, 0]
    sigma_yy_calc_pytfm = stress_tensor_pytfm[:, :, 1, 1]

    # Print numerical quality metrics for MSM
    print("\nNumerical Quality Metrics (MSM):")
    print("-" * 50)
    print(f"Condition Number: {condition_number:.2e}")
    print(f"Residual Norm: {residual:.2e}")

    # Create visualizations
    fig_stress = plot_validation_results_comparison(
        t_x, t_y, sigma_xx_true, sigma_yy_true,
        sigma_xx_calc, sigma_yy_calc,
        sigma_xx_calc_pytfm, sigma_yy_calc_pytfm, mask
    )
    plt.show()

    # Print validation metrics
    print("\nValidation Metrics (Synthetic Cell):")
    print("-" * 50)

    all_metrics = {}  # Dictionary to store all metrics

    # Compare both implementations
    for implementation, xx_calc, yy_calc in [
        ('MSM', sigma_xx_calc, sigma_yy_calc),
        ('pyTFM', sigma_xx_calc_pytfm, sigma_yy_calc_pytfm)
    ]:
        print(f"\n{implementation} Implementation:")
        implementation_metrics = {}

        for comp, true, calc in [
            ('σxx', sigma_xx_true, xx_calc),
            ('σyy', sigma_yy_true, yy_calc)
        ]:
            rmse, max_error, corr, rel_error = calculate_metrics(true, calc, mask)
            print(f"\n{comp}:")
            print(f"RMSE: {rmse:.2e}")
            print(f"Max Error: {max_error:.2e}")
            print(f"Correlation: {corr:.4f}")
            print(f"Relative Error: {rel_error:.4f}")

            implementation_metrics[comp] = {
                'rmse': rmse,
                'max_error': max_error,
                'correlation': corr,
                'relative_error': rel_error
            }

        all_metrics[implementation] = implementation_metrics

    # Add numerical metrics to dictionary
    all_metrics['numerical'] = {
        'condition_number': condition_number,
        'residual': residual
    }

    # Save metrics to file
    metrics_file = synthetic_cell_dir / 'validation_metrics_comparison.npz'
    np.savez(metrics_file, **all_metrics)
    print(f"\nMetrics saved to: {metrics_file}")

    return all_metrics


if __name__ == "__main__":
    # Get the current directory and construct path to synthetic_cell
    current_dir = Path(__file__).parent
    synthetic_cell_dir = current_dir.parent / 'benchmarks' / 'synthetic_cell'

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
        pixelsize=0.3 * 1e-6,
        density_factor=0.005,  # Finer mesh
        algorithm=6,  # MeshAdapt algorithm
        use_optimization=False,  # Enable Netgen optimization
        youngs_modulus=1
    )

    # Initialize pyTFM MSM calculator
    msm_pytfm = MSM_pyTFM(pixelsize=0.3, youngs_modulus=1)

    # Run validation and get metrics
    metrics = validate_msm_synthetic_cell(
        t_x, t_y,
        sigma_xx_true, sigma_yy_true,
        mask,
        msm,
        msm_pytfm
    )