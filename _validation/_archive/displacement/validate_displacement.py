import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import sys
from dataclasses import dataclass

# Add the root directory to Python path to import backend
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from backend.displacement_analysis import DisplacementAnalyzer
from backend.parameter_dataclasses import DisplacementParameters


def load_benchmark_data(benchmark_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the benchmark displacement fields."""
    disp_x = np.load(benchmark_dir / "benchmark" / "displacement_x.npy")
    disp_y = np.load(benchmark_dir / "benchmark" / "displacement_y.npy")
    return disp_x, disp_y


def load_images(benchmark_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the reference and deformed images."""
    reference = tifffile.imread(benchmark_dir / "benchmark" / "reference.tif")
    deformed = tifffile.imread(benchmark_dir / "benchmark" / "deformed.tif")
    return reference, deformed


def calculate_error_metrics(true_field: np.ndarray,
                            calculated_field: np.ndarray) -> dict:
    """Calculate error metrics between true and calculated displacement fields."""
    error = true_field - calculated_field
    metrics = {
        'mae': np.mean(np.abs(error)),
        'rmse': np.sqrt(np.mean(error ** 2)),
        'max_error': np.max(np.abs(error)),
        'min_error': np.min(np.abs(error))
    }
    return metrics


def plot_displacement_comparison(true_x: np.ndarray, true_y: np.ndarray,
                                 calc_x: np.ndarray, calc_y: np.ndarray,
                                 save_path: Path = None):
    """Create comparison plots of true vs calculated displacement fields."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Calculate global min and max for consistent colorbar
    vmin_x = min(np.min(true_x), np.min(calc_x))
    vmax_x = max(np.max(true_x), np.max(calc_x))
    vmin_y = min(np.min(true_y), np.min(calc_y))
    vmax_y = max(np.max(true_y), np.max(calc_y))

    # Plot X displacement
    im0 = axes[0, 0].imshow(true_x, vmin=vmin_x, vmax=vmax_x)
    axes[0, 0].set_title('True X Displacement')
    plt.colorbar(im0, ax=axes[0, 0], label='µm')

    im1 = axes[0, 1].imshow(calc_x, vmin=vmin_x, vmax=vmax_x)
    axes[0, 1].set_title('Calculated X Displacement')
    plt.colorbar(im1, ax=axes[0, 1], label='µm')

    # Plot Y displacement
    im2 = axes[1, 0].imshow(true_y, vmin=vmin_y, vmax=vmax_y)
    axes[1, 0].set_title('True Y Displacement')
    plt.colorbar(im2, ax=axes[1, 0], label='µm')

    im3 = axes[1, 1].imshow(calc_y, vmin=vmin_y, vmax=vmax_y)
    axes[1, 1].set_title('Calculated Y Displacement')
    plt.colorbar(im3, ax=axes[1, 1], label='µm')

    plt.tight_layout()

    # Add parameter information as text
    param_text = f"Parameters:\nτ={params.tau}\nλ={params.lambda_}\nθ={params.theta}\n"
    param_text += f"scales={params.nscales}\nwarps={params.warps}\nε={params.epsilon}\n"
    param_text += f"inner_iter={params.inner_iterations}\nouter_iter={params.outer_iterations}\n"
    param_text += f"scale_step={params.scale_step}\nmedian_filter={params.median_filtering}"

    plt.figtext(1.02, 0.5, param_text, fontsize=10, va='center')

    if save_path:
        # Adjust figure size to accommodate parameters text
        plt.gcf().set_size_inches(18, 15)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def main():
    # Setup paths
    current_dir = Path(__file__).parent  # in _validation/displacement
    validation_dir = current_dir.parent  # up to _validation
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Load benchmark data
    true_disp_x, true_disp_y = load_benchmark_data(validation_dir)
    reference, deformed = load_images(validation_dir)
    pixel_size = 0.1  # µm

    # Initialize displacement analyzer with explicit parameters
    global params  # Make params accessible to plotting function
    params = DisplacementParameters(
        tau=0.1,
        lambda_=0.20,
        theta=0.15,
        nscales=4,
        warps=4,
        epsilon=0.005,
        inner_iterations=10,
        outer_iterations=3,
        scale_step=0.6,
        median_filtering=5,
        downscale_factor=1,
        pixel_size=pixel_size,
        frame_interval=1,
        d_max=1,
        disp_vector_stride=20,
        disp_arrow_scale=1
    )
    analyzer = DisplacementAnalyzer(params)

    # Calculate displacement field
    flow = analyzer.calculate_flow(reference, deformed) * pixel_size
    calc_disp_x, calc_disp_y = flow[..., 0], flow[..., 1]

    # Calculate error metrics
    x_metrics = calculate_error_metrics(true_disp_x, calc_disp_x)
    y_metrics = calculate_error_metrics(true_disp_y, calc_disp_y)

    # Print metrics
    print("\nX Displacement Metrics:")
    for metric, value in x_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nY Displacement Metrics:")
    for metric, value in y_metrics.items():
        print(f"{metric}: {value:.6f}")

    # Create and save comparison plots
    plot_displacement_comparison(
        true_disp_x, true_disp_y,
        calc_disp_x, calc_disp_y,
        save_path=results_dir / "displacement_comparison.png"
    )


if __name__ == "__main__":
    main()