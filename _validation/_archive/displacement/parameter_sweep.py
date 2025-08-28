import numpy as np
from itertools import product
from dataclasses import replace
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tifffile
import sys

# Add the root directory to Python path to import backend
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from backend.displacement_analysis import DisplacementAnalyzer
from backend.parameter_dataclasses import DisplacementParameters


def load_benchmark_data(benchmark_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the benchmark displacement fields."""
    # Modified path to look directly in benchmark directory
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


def parameter_sweep(reference, deformed, true_disp_x, true_disp_y, base_params, param_ranges, analyzer_class):
    """
    Perform parameter sweep and return results DataFrame.
    """
    results = []

    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(product(*param_values))

    # Run analysis for each combination
    for combo in tqdm(combinations, desc="Testing parameter combinations"):
        # Create parameter dictionary
        param_dict = dict(zip(param_names, combo))

        # Create new parameters instance
        current_params = replace(base_params, **param_dict)

        # Initialize analyzer and calculate flow
        analyzer = analyzer_class(current_params)
        flow = analyzer.calculate_flow(reference, deformed) * current_params.pixel_size
        calc_x, calc_y = flow[..., 0], flow[..., 1]

        # Calculate metrics
        x_metrics = calculate_error_metrics(true_disp_x, calc_x)
        y_metrics = calculate_error_metrics(true_disp_y, calc_y)

        # Combine results
        result = {
            **param_dict,
            'x_mae': x_metrics['mae'],
            'x_rmse': x_metrics['rmse'],
            'y_mae': y_metrics['mae'],
            'y_rmse': y_metrics['rmse'],
            'total_mae': (x_metrics['mae'] + y_metrics['mae']) / 2,
            'total_rmse': (x_metrics['rmse'] + y_metrics['rmse']) / 2
        }
        results.append(result)

    return pd.DataFrame(results)


def analyze_results(df):
    """Analyze and print parameter sweep results."""
    # Find best parameters based on total MAE
    best_params = df.loc[df['total_mae'].idxmin()]

    print("\nBest parameters found:")
    for param in df.columns:
        if param not in ['x_mae', 'x_rmse', 'y_mae', 'y_rmse', 'total_mae', 'total_rmse']:
            print(f"{param}: {best_params[param]}")

    print(f"\nBest metrics achieved:")
    print(f"Total MAE: {best_params['total_mae']:.6f}")
    print(f"Total RMSE: {best_params['total_rmse']:.6f}")

    # Parameter sensitivity analysis
    print("\nParameter sensitivity (std of metrics):")
    for param in df.columns:
        if param not in ['x_mae', 'x_rmse', 'y_mae', 'y_rmse', 'total_mae', 'total_rmse']:
            sensitivity = df.groupby(param)['total_mae'].mean().std()
            print(f"{param}: {sensitivity:.6f}")

    # Save results
    df.to_csv('parameter_sweep_results.csv', index=False)


def main():
    # Setup paths
    current_dir = Path(__file__).parent  # in _validation/displacement
    validation_dir = current_dir.parent  # up to _validation

    # Print paths to debug
    print("Current directory:", current_dir)
    print("Validation directory:", validation_dir)
    print("Expected benchmark path:", validation_dir / "benchmark" / "displacement_x.npy")

    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Load data
    true_disp_x, true_disp_y = load_benchmark_data(validation_dir)
    reference, deformed = load_images(validation_dir)
    pixel_size = 0.1  # µm

    # Set up base parameters
    base_params = DisplacementParameters(
        tau=0.25,
        lambda_=0.4,
        theta=0.1,
        nscales=3,
        warps=3,
        epsilon=0.01,
        inner_iterations=15,
        outer_iterations=5,
        scale_step=0.5,
        median_filtering=0,
        downscale_factor=1,
        pixel_size=pixel_size,
        frame_interval=1,
        d_max=1,
        disp_vector_stride=20,
        disp_arrow_scale=1
    )

    # Define parameter ranges for sweep
    param_ranges = {
        'tau': [0.05, 0.1],
        'lambda_': [0.1, 0.2],
        'theta': [0.15, 0.2, 0.25],
        'nscales': [4, 6],
        'warps': [4, 5, 6],
        'epsilon': [0.005],
        'inner_iterations': [5, 10],
        'outer_iterations': [2, 3],
        'scale_step': [0.6, 0.7, 0.8],
    }

    # Run parameter sweep
    results_df = parameter_sweep(
        reference, deformed,
        true_disp_x, true_disp_y,
        base_params, param_ranges,
        DisplacementAnalyzer
    )

    # Analyze and save results
    analyze_results(results_df)


if __name__ == "__main__":
    main()