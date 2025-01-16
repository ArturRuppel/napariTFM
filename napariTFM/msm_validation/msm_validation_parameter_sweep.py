from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from napariTFM.msm import MonolayerStressMicroscopy
from napariTFM.msm_validation.msm_validation_synthetic_cell import calculate_metrics
from typing import Dict



def analyze_mesh_quality(points: np.ndarray, triangles: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics for the generated mesh"""
    # Get triangle vertices
    v0 = points[triangles[:, 0]]
    v1 = points[triangles[:, 1]]
    v2 = points[triangles[:, 2]]

    # Compute edge vectors
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # Compute edge lengths
    lengths = np.stack([
        np.linalg.norm(e0, axis=1),
        np.linalg.norm(e1, axis=1),
        np.linalg.norm(e2, axis=1)
    ]).T

    # Compute angles
    angles = []
    for i in range(3):
        e_prev = -e2 if i == 0 else -e0 if i == 1 else -e1
        e_next = e0 if i == 0 else e1 if i == 1 else e2
        cos_angle = np.sum(e_prev * e_next, axis=1) / (
                np.linalg.norm(e_prev, axis=1) * np.linalg.norm(e_next, axis=1)
        )
        angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    angles = np.concatenate(angles) * 180 / np.pi

    # Compute areas
    areas = 0.5 * np.abs(
        (v0[:, 0] * v1[:, 1] + v1[:, 0] * v2[:, 1] + v2[:, 0] * v0[:, 1]) -
        (v1[:, 0] * v0[:, 1] + v2[:, 0] * v1[:, 1] + v0[:, 0] * v2[:, 1])
    )

    # Quality metrics
    aspect_ratios = np.max(lengths, axis=1) / np.min(lengths, axis=1)
    s = np.sum(lengths, axis=1) / 2
    r_in = 2 * areas / (s * 2)
    r_out = np.prod(lengths, axis=1) / (4 * areas)
    quality = 2 * r_in / r_out

    return {
        "min_angle": np.min(angles),
        "mean_angle": np.mean(angles),
        "min_quality": np.min(quality),
        "mean_quality": np.mean(quality),
        "max_aspect_ratio": np.max(aspect_ratios),
        "mean_aspect_ratio": np.mean(aspect_ratios),
        "n_elements": len(triangles)
    }


def run_parameter_sweep(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, pixelsize=1.0):
    """Run a comprehensive parameter sweep over density factors and algorithms"""
    # Define parameters to sweep
    density_factors = np.linspace(0.005, 0.05, 10)
    algorithms = range(1, 7)
    optimization_options = [True, False]

    # Initialize results list
    results = []

    total_combinations = len(density_factors) * len(algorithms) * len(optimization_options)
    counter = 0

    for density in density_factors:
        for alg in algorithms:
            for use_optimization in optimization_options:
                counter += 1
                print(f"\nProgress: {counter}/{total_combinations}")
                print(f"Testing - Density: {density:.3f}, Algorithm: {alg}, Optimization: {use_optimization}")

                # Initialize MSM calculator
                msm = MonolayerStressMicroscopy(
                    mask=mask,
                    pixelsize=pixelsize,
                    density_factor=density,
                    algorithm=alg,
                    use_optimization=use_optimization
                )

                # Calculate stress tensor
                stress_tensor_calc = msm.calculate_stress_field(t_x, t_y)
                sigma_xx_calc = stress_tensor_calc[:, :, 0, 0]
                sigma_yy_calc = stress_tensor_calc[:, :, 1, 1]

                # Get mesh quality metrics
                mesh_quality = analyze_mesh_quality(msm.nodes, msm.elements)

                # Calculate metrics for both components
                rmse_xx, _, corr_xx, _ = calculate_metrics(sigma_xx_true, sigma_xx_calc, mask)
                rmse_yy, _, corr_yy, _ = calculate_metrics(sigma_yy_true, sigma_yy_calc, mask)

                # Store results
                result = {
                    'density_factor': density,
                    'algorithm': alg,
                    'optimization': use_optimization,
                    'n_nodes': len(msm.nodes),
                    'correlation_xx': corr_xx,
                    'correlation_yy': corr_yy,
                    'rmse_xx': rmse_xx,
                    'rmse_yy': rmse_yy,
                    **mesh_quality  # Unpack mesh quality metrics
                }

                results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df


def plot_results(df):
    """Create comprehensive visualizations of the parameter sweep results"""
    # Set up the plotting style
    sns.set_theme(style="whitegrid")  # Using seaborn's theme instead of plt.style

    # Create figure for correlation analysis
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Correlation Analysis', fontsize=16)

    # Plot correlation vs density for different algorithms
    for opt in [True, False]:
        opt_label = 'With optimization' if opt else 'Without optimization'
        for alg in sorted(df['algorithm'].unique()):
            mask = (df['optimization'] == opt) & (df['algorithm'] == alg)
            data = df[mask]
            ax1.plot(data['density_factor'], data['correlation_xx'],
                     label=f'Alg {alg} ({opt_label})', alpha=0.7, marker='o')
    ax1.set_xlabel('Density Factor')
    ax1.set_ylabel('Correlation σxx')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    # Plot correlation vs algorithm with boxplots
    sns.boxplot(data=df, x='algorithm', y='correlation_xx', hue='optimization', ax=ax2)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Correlation σxx')
    ax2.set_title('Distribution of Correlations by Algorithm')

    # Plot correlation vs mesh quality
    sns.scatterplot(data=df, x='mean_quality', y='correlation_xx',
                    hue='algorithm', style='optimization', ax=ax3)
    ax3.set_xlabel('Mean Mesh Quality')
    ax3.set_ylabel('Correlation σxx')
    ax3.set_title('Correlation vs Mesh Quality')

    # Plot correlation vs number of nodes
    sns.scatterplot(data=df, x='n_nodes', y='correlation_xx',
                    hue='algorithm', style='optimization', ax=ax4)
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Correlation σxx')
    ax4.set_title('Correlation vs Mesh Size')

    plt.tight_layout()
    plt.show()

    # Create figure for mesh quality analysis
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('Mesh Quality Analysis', fontsize=16)

    # Plot mesh quality metrics vs density
    sns.boxplot(data=df, x=pd.qcut(df['density_factor'], 10), y='mean_quality', ax=ax1)
    ax1.set_xlabel('Density Factor')
    ax1.set_ylabel('Mean Mesh Quality')
    ax1.set_title('Mesh Quality vs Density')
    ax1.tick_params(axis='x', rotation=45)

    # Plot angle distribution
    sns.boxplot(data=df, x='algorithm', y='mean_angle', hue='optimization', ax=ax2)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Mean Angle (degrees)')
    ax2.set_title('Angle Distribution by Algorithm')

    # Plot aspect ratio distribution
    sns.boxplot(data=df, x='algorithm', y='mean_aspect_ratio', hue='optimization', ax=ax3)
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Mean Aspect Ratio')
    ax3.set_title('Aspect Ratio Distribution by Algorithm')

    # Plot number of elements vs density
    sns.scatterplot(data=df, x='density_factor', y='n_elements',
                    hue='algorithm', style='optimization', ax=ax4)
    ax4.set_xlabel('Density Factor')
    ax4.set_ylabel('Number of Elements')
    ax4.set_title('Mesh Size vs Density')

    plt.tight_layout()
    plt.show()

    # Save figures
    fig1.savefig('correlation_analysis.svg', bbox_inches='tight')
    fig2.savefig('mesh_quality_analysis.svg', bbox_inches='tight')

    return fig1, fig2


if __name__ == "__main__":
    # Define paths
    results_file = Path('parameter_sweep_results.csv')

    if results_file.exists():
        print("Loading existing results from file...")
        results_df = pd.read_csv(results_file)
    else:
        print("Results file not found. Running parameter sweep...")
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

        # Run parameter sweep
        results_df = run_parameter_sweep(
            t_x, t_y,
            sigma_xx_true, sigma_yy_true,
            mask,
            pixelsize=0.3 * 1e-6  # assuming unit pixels
        )

        # Save results to CSV
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

    # Plot results
    fig1, fig2 = plot_results(results_df)
