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


def plot_validation_results_individual_scale(t_x, t_y, sigma_xx_true, sigma_yy_true,
                                             sigma_xx_calc, sigma_yy_calc, mask):
    """Plot comparison between true and calculated stress fields with individual scales"""
    fig_stress = plt.figure(figsize=(12, 10))
    plt.suptitle('Stress Tensor Validation (Synthetic Cell) - Individual Scales', fontsize=14)

    ax_s1 = plt.subplot(221)
    ax_s2 = plt.subplot(222)
    ax_s3 = plt.subplot(223)
    ax_s4 = plt.subplot(224)

    def plot_component(ax, data, title):
        masked_data = np.copy(data)
        masked_data[~mask] = np.nan
        vmax = np.nanpercentile(np.abs(masked_data), 99)
        im = ax.imshow(masked_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        return im

    # Plot stress components
    im5 = plot_component(ax_s1, sigma_xx_true, 'True σxx (Warped)')
    plt.colorbar(im5, ax=ax_s1)

    im6 = plot_component(ax_s2, sigma_xx_calc, 'Calculated σxx')
    plt.colorbar(im6, ax=ax_s2)

    im7 = plot_component(ax_s3, sigma_yy_true, 'True σyy (Warped)')
    plt.colorbar(im7, ax=ax_s3)

    im8 = plot_component(ax_s4, sigma_yy_calc, 'Calculated σyy')
    plt.colorbar(im8, ax=ax_s4)

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


def validate_msm_synthetic_cell(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, pixelsize=1.0):
    """Validate MSM implementation using synthetic cell data"""
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

    # Initialize MSM calculator
    msm = MonolayerStressMicroscopy(
        mask=mask,
        pixelsize=pixelsize,
        density_factor=0.01,  # Finer mesh
        algorithm=4,  # MeshAdapt algorithm
        use_optimization=True,  # Enable Netgen optimization
        youngs_modulus=1
    )

    # Plot mesh
    # mesh_fig = plot_mesh(msm.nodes, msm.elements, mask)
    # plt.title('Triangular Mesh (Synthetic Cell)')
    # plt.savefig("mesh.svg")
    # plt.show()

    # Calculate stress tensor
    stress_tensor_calc = msm.calculate_stress_field(t_x, t_y)
    sigma_xx_calc = stress_tensor_calc[:, :, 0, 0]
    sigma_yy_calc = stress_tensor_calc[:, :, 1, 1]

    # # Create visualizations
    # fig_stress_common = plot_validation_results_common_scale(
    #     t_x, t_y, sigma_xx_true, sigma_yy_true,
    #     sigma_xx_calc, sigma_yy_calc, mask
    # )
    # plt.show()
    plt.figure()
    fig_stress_individual = plot_validation_results_individual_scale(
        t_x, t_y, sigma_xx_true, sigma_yy_true,
        sigma_xx_calc, sigma_yy_calc, mask
    )
    plt.show()

    # Print metrics
    print("\nValidation Metrics (Synthetic Cell):")
    print("-" * 50)

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

    # return stress_tensor_calc, fig_stress_common, fig_stress_individual, mesh_fig
    return


if __name__ == "__main__":
    # Get the current directory and construct path to synthetic_cell
    current_dir = Path(__file__).parent
    synthetic_cell_dir = current_dir.parent / 'benchmarks' / 'synthetic_cell_unitless'

    # Load data
    t_x = np.load(synthetic_cell_dir / 'Traction_x_warped.npy')
    t_y = np.load(synthetic_cell_dir / 'Traction_y_warped.npy')
    sigma_xx_true = np.load(synthetic_cell_dir / 'Stress_xx_warped.npy')
    sigma_yy_true = np.load(synthetic_cell_dir / 'Stress_yy_warped.npy')

    # Create mask based on stress magnitude
    mask = np.abs(sigma_xx_true) > 0

    # Run validation
    validate_msm_synthetic_cell(
        t_x, t_y,
        sigma_xx_true, sigma_yy_true,
        mask,
        pixelsize=1
    )