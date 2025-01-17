from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napariTFM.msm import MonolayerStressMicroscopy
from napariTFM.msm_validation.synthetic_cell import calculate_metrics


def analyze_density_impact(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, pixelsize=1.0):
    """Analyze the impact of density_factor on solution accuracy"""
    # Define density factors to test
    density_factors = np.linspace(0.005, 0.05, 10)

    # Initialize results storage
    results = {
        'density_factors': density_factors,
        'correlations_xx': [],
        'correlations_yy': [],
        'rmse_xx': [],
        'rmse_yy': [],
        'mesh_sizes': []
    }

    for density in density_factors:
        print(f"\nAnalyzing density factor: {density:.3f}")

        # Initialize MSM calculator
        msm = MonolayerStressMicroscopy(
            mask=mask,
            pixelsize=pixelsize,
            density_factor=density,
            algorithm=6,
            use_optimization=True
        )

        # Store mesh size
        results['mesh_sizes'].append(len(msm.nodes))

        # Calculate stress tensor
        stress_tensor_calc = msm.calculate_stress_field(t_x, t_y)
        sigma_xx_calc = stress_tensor_calc[:, :, 0, 0]
        sigma_yy_calc = stress_tensor_calc[:, :, 1, 1]

        # Calculate metrics
        for comp, true, calc, corr_list, rmse_list in [
            ('σxx', sigma_xx_true, sigma_xx_calc, results['correlations_xx'], results['rmse_xx']),
            ('σyy', sigma_yy_true, sigma_yy_calc, results['correlations_yy'], results['rmse_yy'])
        ]:
            rmse, _, corr, _ = calculate_metrics(true, calc, mask)
            corr_list.append(corr)
            rmse_list.append(rmse)
            print(f"{comp} - Correlation: {corr:.4f}, RMSE: {rmse:.2e}")

    return results


def plot_density_analysis(results):
    """Plot the results of density factor analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot correlations
    ax1.plot(results['density_factors'], results['correlations_xx'], 'b.-', label='σxx')
    ax1.plot(results['density_factors'], results['correlations_yy'], 'r.-', label='σyy')
    ax1.set_xlabel('Density Factor')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Correlation vs Density Factor')
    ax1.legend()
    ax1.grid(True)

    # Plot RMSE
    ax2.plot(results['density_factors'], results['rmse_xx'], 'b.-', label='σxx')
    ax2.plot(results['density_factors'], results['rmse_yy'], 'r.-', label='σyy')
    ax2.set_xlabel('Density Factor')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs Density Factor')
    ax2.legend()
    ax2.grid(True)

    # Plot mesh size
    ax3.plot(results['density_factors'], results['mesh_sizes'], 'k.-')
    ax3.set_xlabel('Density Factor')
    ax3.set_ylabel('Number of Mesh Nodes')
    ax3.set_title('Mesh Size vs Density Factor')
    ax3.grid(True)

    # Plot RMSE vs mesh size
    ax4.plot(results['mesh_sizes'], results['rmse_xx'], 'b.-', label='σxx')
    ax4.plot(results['mesh_sizes'], results['rmse_yy'], 'r.-', label='σyy')
    ax4.set_xlabel('Number of Mesh Nodes')
    ax4.set_ylabel('RMSE')
    ax4.set_title('RMSE vs Mesh Size')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Get the current directory and construct path to synthetic_cell
    current_dir = Path(__file__).parent
    synthetic_cell_dir = current_dir.parent / 'benchmarks' / 'synthetic_cell_downscaled'

    # Load data
    t_x = np.load(synthetic_cell_dir / 'Traction_x_warped.npy')
    t_y = np.load(synthetic_cell_dir / 'Traction_y_warped.npy')
    sigma_xx_true = np.load(synthetic_cell_dir / 'Stress_xx_warped.npy')
    sigma_yy_true = np.load(synthetic_cell_dir / 'Stress_yy_warped.npy')

    # Create mask based on stress magnitude
    mask = np.abs(sigma_xx_true) > 0

    # Run density factor analysis
    results = analyze_density_impact(
        t_x, t_y,
        sigma_xx_true, sigma_yy_true,
        mask,
        pixelsize=1 * 1e6  # assuming unit pixels
    )

    # Plot results
    fig = plot_density_analysis(results)
    plt.savefig("density_analysis.svg")
    plt.show()