from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napariTFM.msm import MonolayerStressMicroscopy
from napariTFM.msm_validation.msm_validation_synthetic_cell import calculate_metrics


def analyze_algorithm_impact(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, pixelsize=1.0):
    """Analyze the impact of different algorithms and optimization settings"""
    # Define algorithms to test (1-6 are valid options)
    algorithms = range(1, 7)

    # Initialize results storage
    results = {
        'algorithms': list(algorithms),
        'correlations_xx_opt': [],
        'correlations_yy_opt': [],
        'rmse_xx_opt': [],
        'rmse_yy_opt': [],
        'correlations_xx_no_opt': [],
        'correlations_yy_no_opt': [],
        'rmse_xx_no_opt': [],
        'rmse_yy_no_opt': [],
        'computation_times_opt': [],
        'computation_times_no_opt': []
    }

    for alg in algorithms:
        print(f"\nAnalyzing algorithm: {alg}")

        # Test with optimization
        print("With optimization:")
        msm_opt = MonolayerStressMicroscopy(
            mask=mask,
            pixelsize=pixelsize,
            density_factor=0.01,  # Fixed density as requested
            algorithm=alg,
            use_optimization=True
        )

        # Calculate stress tensor with optimization
        stress_tensor_opt = msm_opt.calculate_stress_field(t_x, t_y)
        sigma_xx_opt = stress_tensor_opt[:, :, 0, 0]
        sigma_yy_opt = stress_tensor_opt[:, :, 1, 1]

        # Calculate metrics for optimized version
        for comp, true, calc, corr_list, rmse_list in [
            ('σxx', sigma_xx_true, sigma_xx_opt, results['correlations_xx_opt'], results['rmse_xx_opt']),
            ('σyy', sigma_yy_true, sigma_yy_opt, results['correlations_yy_opt'], results['rmse_yy_opt'])
        ]:
            rmse, _, corr, _ = calculate_metrics(true, calc, mask)
            corr_list.append(corr)
            rmse_list.append(rmse)
            print(f"{comp} - Correlation: {corr:.4f}, RMSE: {rmse:.2e}")

        # Test without optimization
        print("Without optimization:")
        msm_no_opt = MonolayerStressMicroscopy(
            mask=mask,
            pixelsize=pixelsize,
            density_factor=0.01,  # Fixed density as requested
            algorithm=alg,
            use_optimization=False
        )

        # Calculate stress tensor without optimization
        stress_tensor_no_opt = msm_no_opt.calculate_stress_field(t_x, t_y)
        sigma_xx_no_opt = stress_tensor_no_opt[:, :, 0, 0]
        sigma_yy_no_opt = stress_tensor_no_opt[:, :, 1, 1]

        # Calculate metrics for non-optimized version
        for comp, true, calc, corr_list, rmse_list in [
            ('σxx', sigma_xx_true, sigma_xx_no_opt, results['correlations_xx_no_opt'], results['rmse_xx_no_opt']),
            ('σyy', sigma_yy_true, sigma_yy_no_opt, results['correlations_yy_no_opt'], results['rmse_yy_no_opt'])
        ]:
            rmse, _, corr, _ = calculate_metrics(true, calc, mask)
            corr_list.append(corr)
            rmse_list.append(rmse)
            print(f"{comp} - Correlation: {corr:.4f}, RMSE: {rmse:.2e}")

    return results


def plot_algorithm_analysis(results):
    """Plot the results of algorithm analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    x = np.array(results['algorithms'])
    width = 0.35  # Width of bars

    # Plot correlations for σxx
    ax1.bar(x - width / 2, results['correlations_xx_opt'], width, label='With optimization', color='blue', alpha=0.7)
    ax1.bar(x + width / 2, results['correlations_xx_no_opt'], width, label='Without optimization', color='lightblue', alpha=0.7)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Correlation')
    ax1.set_title('σxx Correlation vs Algorithm')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(x)

    # Plot correlations for σyy
    ax2.bar(x - width / 2, results['correlations_yy_opt'], width, label='With optimization', color='red', alpha=0.7)
    ax2.bar(x + width / 2, results['correlations_yy_no_opt'], width, label='Without optimization', color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Correlation')
    ax2.set_title('σyy Correlation vs Algorithm')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(x)

    # Plot RMSE for σxx
    ax3.bar(x - width / 2, results['rmse_xx_opt'], width, label='With optimization', color='blue', alpha=0.7)
    ax3.bar(x + width / 2, results['rmse_xx_no_opt'], width, label='Without optimization', color='lightblue', alpha=0.7)
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('RMSE')
    ax3.set_title('σxx RMSE vs Algorithm')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xticks(x)

    # Plot RMSE for σyy
    ax4.bar(x - width / 2, results['rmse_yy_opt'], width, label='With optimization', color='red', alpha=0.7)
    ax4.bar(x + width / 2, results['rmse_yy_no_opt'], width, label='Without optimization', color='lightcoral', alpha=0.7)
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('RMSE')
    ax4.set_title('σyy RMSE vs Algorithm')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xticks(x)

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

    # Run algorithm analysis
    results = analyze_algorithm_impact(
        t_x, t_y,
        sigma_xx_true, sigma_yy_true,
        mask,
        pixelsize=1 * 1e6  # assuming unit pixels
    )

    # Plot results
    fig = plot_algorithm_analysis(results)
    plt.savefig("algorithm_analysis.svg")
    plt.show()