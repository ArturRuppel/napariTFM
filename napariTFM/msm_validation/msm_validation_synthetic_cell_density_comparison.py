from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from napariTFM.msm import MonolayerStressMicroscopy


def run_msm_simulation(t_x, t_y, mask, density_factor, pixelsize=0.2):
    """Run MSM simulation with given density factor"""
    msm = MonolayerStressMicroscopy(
        mask=mask,
        pixelsize=pixelsize,
        density_factor=density_factor,
        algorithm=4,
        use_optimization=True,
        youngs_modulus=100
    )

    stress_tensor_calc = msm.calculate_stress_field(t_x, t_y)
    return stress_tensor_calc[:, :, 0, 0], stress_tensor_calc[:, :, 1, 1]


def plot_comparison_results(sigma_xx_true, sigma_xx_results, density_factors, mask):
    """Plot ground truth and calculated sigma_xx for different density factors, each with its own scale"""
    fig = plt.figure(figsize=(15, 4))
    plt.suptitle('Comparison of σxx for Different Mesh Densities', fontsize=14)

    # Print header for percentile values
    print("\n99th Percentile Values:")
    print("-" * 50)

    # Plot ground truth
    ax1 = plt.subplot(141)
    masked_data = np.copy(sigma_xx_true)
    masked_data[~mask] = np.nan
    vmax_true = np.nanpercentile(np.abs(masked_data), 99)
    print(f"Ground Truth σxx: {vmax_true:.2e}")
    im1 = ax1.imshow(masked_data, cmap='RdBu_r', vmin=-vmax_true, vmax=vmax_true)
    ax1.set_title('Ground Truth σxx')
    plt.colorbar(im1, ax=ax1)

    # Plot results for each density factor
    for idx, (sigma_xx_calc, density) in enumerate(zip(sigma_xx_results, density_factors)):
        ax = plt.subplot(1, 4, idx + 2)
        masked_calc = np.copy(sigma_xx_calc)
        masked_calc[~mask] = np.nan
        vmax_calc = np.nanpercentile(np.abs(masked_calc), 99)
        print(f"Density {density}: {vmax_calc:.2e}")
        im = ax.imshow(masked_calc, cmap='RdBu_r', vmin=-vmax_calc, vmax=vmax_calc)
        ax.set_title(f'σxx (density={density})')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def main():
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

    # Define density factors to test
    density_factors = [0.005, 0.01, 0.02]

    # Run simulations for each density factor
    sigma_xx_results = []
    sigma_yy_results = []

    for density in density_factors:
        print(f"\nRunning simulation with density factor: {density}")
        sigma_xx, sigma_yy = run_msm_simulation(t_x, t_y, mask, density)
        sigma_xx_results.append(sigma_xx)
        sigma_yy_results.append(sigma_yy)

    # Plot results
    plot_comparison_results(sigma_xx_true, sigma_xx_results, density_factors, mask)
    plt.show()

    # Print average stress values
    print("\nAverage Stress Values:")
    print("-" * 50)
    print(f"Ground Truth:")
    print(f"Average σxx: {np.nanmean(sigma_xx_true[mask]):.2e}")
    print(f"Average σyy: {np.nanmean(sigma_yy_true[mask]):.2e}")

    for density, sigma_xx, sigma_yy in zip(density_factors, sigma_xx_results, sigma_yy_results):
        print(f"\nDensity factor {density}:")
        print(f"Average σxx: {np.nanmean(sigma_xx[mask]):.2e}")
        print(f"Average σyy: {np.nanmean(sigma_yy[mask]):.2e}")


if __name__ == "__main__":
    main()