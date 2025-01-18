from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
from napariTFM.msm import MonolayerStressMicroscopy
from napariTFM.mesh_generator import MeshGenerator, MeshParameters


def calculate_metrics(true, calc, mask):
    """Calculate error metrics between true and calculated fields"""
    valid_mask = mask & ~np.isnan(true) & ~np.isnan(calc)
    rmse = np.sqrt(np.mean((true[valid_mask] - calc[valid_mask]) ** 2))
    max_error = np.max(np.abs(true[valid_mask] - calc[valid_mask]))
    correlation = np.corrcoef(true[valid_mask].flatten(), calc[valid_mask].flatten())[0, 1]
    rel_error = rmse / np.std(true[valid_mask])
    return {
        'rmse': rmse,
        'max_error': max_error,
        'correlation': correlation,
        'relative_error': rel_error
    }


def analyze_mesh_quality(points, triangles, mask, density_factor, algorithm, use_optimization):
    """Compute quality metrics for the generated mesh using MeshGenerator's method"""
    mesh_params = MeshParameters(
        mask=mask,
        density_factor=density_factor,
        algorithm=algorithm,
        use_optimization=use_optimization
    )
    mesh_gen = MeshGenerator(mesh_params)
    return mesh_gen.analyze_mesh_quality(points, triangles)


def run_msm_analysis(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, params):
    """Run MSM analysis with given parameters and return results"""
    try:
        # Initialize MSM calculator with current parameter set
        msm = MonolayerStressMicroscopy(
            mask=mask,
            pixelsize=params['pixelsize'],
            density_factor=params['density_factor'],
            algorithm=params['algorithm'],
            use_optimization=params['use_optimization'],
            youngs_modulus=params['youngs_modulus']
        )

        # Calculate stress tensor and get numerical metrics
        stress_tensor_calc, condition_number, residual = msm.calculate_stress_field(t_x, t_y)

        # Extract components
        sigma_xx_calc = stress_tensor_calc[:, :, 0, 0]
        sigma_yy_calc = stress_tensor_calc[:, :, 1, 1]

        # Calculate metrics
        xx_metrics = calculate_metrics(sigma_xx_true, sigma_xx_calc, mask)
        yy_metrics = calculate_metrics(sigma_yy_true, sigma_yy_calc, mask)

        # Use MeshGenerator's mesh quality analysis via helper function
        mesh_metrics = analyze_mesh_quality(
            msm.nodes,
            msm.elements,
            mask=mask,
            density_factor=params['density_factor'],
            algorithm=params['algorithm'],
            use_optimization=params['use_optimization']
        )

        # Add numerical quality metrics
        numerical_metrics = {
            'condition_number': condition_number,
            'residual_norm': residual
        }

        return {
            'success': True,
            'stress_tensor': stress_tensor_calc,
            'nodes': msm.nodes,
            'elements': msm.elements,
            'mesh_metrics': mesh_metrics,
            'xx_metrics': xx_metrics,
            'yy_metrics': yy_metrics,
            'numerical_metrics': numerical_metrics,
            'parameters': params
        }

    except Exception as e:
        print(f"Error during analysis: {str(e)}")  # Added error printing for debugging
        return {
            'success': False,
            'error': str(e),
            'parameters': params
        }


def parameter_sweep():
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

    # Define parameter ranges to sweep
    param_combinations = [
        {
            'pixelsize': 0.3e-6,  # Fixed parameter
            'density_factor': df,
            'algorithm': algo,
            'use_optimization': opt,
            'youngs_modulus': E
        }
        for df in [0.005, 0.01, 0.05]  # Mesh density factors
        for algo in [1, 5, 6, 7, 8, 9]  # Different meshing algorithms
        for opt in [True, False]  # Optimization settings
        for E in [1]  # Young's modulus values
    ]

    # Initialize results dictionary
    results = {
        'metadata': {
            'data_shape': t_x.shape,
            'mask_sum': np.sum(mask),
            'timestamp': np.datetime64('now')
        },
        'parameter_combinations': param_combinations,
        'runs': []
    }

    # Run parameter sweep with progress bar
    for params in tqdm(param_combinations, desc="Running parameter sweep"):
        result = run_msm_analysis(t_x, t_y, sigma_xx_true, sigma_yy_true, mask, params)
        results['runs'].append(result)

    # Save results
    output_dir = current_dir / 'parameter_sweep_results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'msm_parameter_sweep.pickle'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    successful_runs = [run for run in results['runs'] if run['success']]
    print(f"\nCompleted {len(successful_runs)}/{len(param_combinations)} runs successfully")

    if successful_runs:
        best_xx_rmse = min(successful_runs, key=lambda x: x['xx_metrics']['rmse'])
        best_mesh_quality = max(successful_runs, key=lambda x: x['mesh_metrics']['mean_quality'])
        best_condition = min(successful_runs, key=lambda x: x['numerical_metrics']['condition_number'])
        best_residual = min(successful_runs, key=lambda x: x['numerical_metrics']['residual_norm'])
        best_min_angle = max(successful_runs, key=lambda x: x['mesh_metrics']['min_angle'])

        print("\nBest XX RMSE parameters:")
        print(best_xx_rmse['parameters'])
        print(f"RMSE: {best_xx_rmse['xx_metrics']['rmse']:.2e}")

        print("\nBest mesh quality parameters:")
        print(best_mesh_quality['parameters'])
        print(f"Mean quality: {best_mesh_quality['mesh_metrics']['mean_quality']:.4f}")
        print(f"Min angle: {best_mesh_quality['mesh_metrics']['min_angle']:.2f}°")
        print(f"Mean aspect ratio: {best_mesh_quality['mesh_metrics']['mean_aspect_ratio']:.4f}")

        print("\nBest minimum angle parameters:")
        print(best_min_angle['parameters'])
        print(f"Min angle: {best_min_angle['mesh_metrics']['min_angle']:.2f}°")

        print("\nBest condition number parameters:")
        print(best_condition['parameters'])
        print(f"Condition number: {best_condition['numerical_metrics']['condition_number']:.2e}")

        print("\nBest residual norm parameters:")
        print(best_residual['parameters'])
        print(f"Residual norm: {best_residual['numerical_metrics']['residual_norm']:.2e}")


if __name__ == "__main__":
    parameter_sweep()