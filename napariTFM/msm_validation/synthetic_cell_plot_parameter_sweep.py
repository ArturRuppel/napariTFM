import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_ground_truth(benchmark_dir):
    """Load ground truth stress fields"""
    stress_xx = np.load(benchmark_dir / 'Stress_xx_warped.npy')
    stress_yy = np.load(benchmark_dir / 'Stress_yy_warped.npy')
    return stress_xx, stress_yy


def plot_selected_cases(results, ground_truth, save_path=None):
    """Plot three cases with different mesh densities and their stress fields compared to ground truth"""
    stress_xx_gt, stress_yy_gt = ground_truth

    # Find the best case based on correlation
    successful_runs = [run for run in results['runs'] if run['success']]
    for run in successful_runs:
        run['avg_corr'] = (run['xx_metrics']['correlation'] + run['yy_metrics']['correlation']) / 2

    sorted_runs = sorted(successful_runs, key=lambda x: x['avg_corr'], reverse=True)
    best_case = sorted_runs[0]

    # Find cases with same parameters but different density factors
    same_param_runs = [
        run for run in successful_runs
        if (run['parameters']['algorithm'] == best_case['parameters']['algorithm'] and
            run['parameters']['use_optimization'] == best_case['parameters']['use_optimization'] and
            run['parameters']['youngs_modulus'] == best_case['parameters']['youngs_modulus'])
    ]

    # Sort by density factor and select three cases
    same_param_runs.sort(key=lambda x: x['parameters']['density_factor'])
    cases = [same_param_runs[0], same_param_runs[len(same_param_runs) // 2], best_case]
    titles = ['Coarse Mesh', 'Medium Mesh', 'Fine Mesh']

    # Create figure with 4x4 grid
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle('Mesh Refinement Comparison', fontsize=16)

    # Fixed stress limits
    stress_lim = 0.004

    # Print debug information
    print(f"Number of cases: {len(cases)}")
    for idx, case in enumerate(cases):
        print(f"Case {idx} density factor: {case['parameters']['density_factor']}")

    # First column: empty plot and meshes
    plt.subplot(4, 3, 1).set_visible(False)  # Empty plot

    # Plot meshes (first column, rows 2-4)
    for idx in range(3):
        ax_mesh = plt.subplot(4, 3, 3 * idx + 4)  # This gives indices 4, 7, 10
        case = cases[idx]
        title = titles[idx]

        for element in case['elements']:
            vertices = case['nodes'][element]
            ax_mesh.plot([vertices[0][0], vertices[1][0]],
                         [-vertices[0][1], -vertices[1][1]], 'b-', linewidth=0.5)
            ax_mesh.plot([vertices[1][0], vertices[2][0]],
                         [-vertices[1][1], -vertices[2][1]], 'b-', linewidth=0.5)
            ax_mesh.plot([vertices[2][0], vertices[0][0]],
                         [-vertices[2][1], -vertices[0][1]], 'b-', linewidth=0.5)
        ax_mesh.set_aspect('equal')
        ax_mesh.set_title(f'{title}\nMesh Quality: {case["mesh_metrics"]["mean_quality"]:.3f}')

        # Parameters text
        params = case['parameters']
        param_text = f'df={params["density_factor"]}\n'
        param_text += f'algo={params["algorithm"]}\n'
        param_text += f'opt={params["use_optimization"]}\n'
        param_text += f'E={params["youngs_modulus"]}'
        ax_mesh.text(0.02, 0.98, param_text, transform=ax_mesh.transAxes,
                     verticalalignment='top', fontsize=8)

    # Second column: σxx ground truth and cases
    ax_gt_xx = plt.subplot(4, 3, 2)
    im_gt_xx = ax_gt_xx.imshow(stress_xx_gt, cmap='RdBu_r',
                               vmin=-stress_lim, vmax=stress_lim)
    plt.colorbar(im_gt_xx, ax=ax_gt_xx)
    ax_gt_xx.set_title('Ground Truth σxx')

    for idx in range(3):
        ax_xx = plt.subplot(4, 3, 3 * idx + 5)  # This gives indices 5, 8, 11
        case = cases[idx]
        im_xx = ax_xx.imshow(case['stress_tensor'][:, :, 0, 0],
                             cmap='RdBu_r', vmin=-stress_lim, vmax=stress_lim)
        plt.colorbar(im_xx, ax=ax_xx)
        ax_xx.set_title(f'σxx (Correlation: {case["xx_metrics"]["correlation"]:.3f})')

    # Third column: σyy ground truth and cases
    ax_gt_yy = plt.subplot(4, 3, 3)
    im_gt_yy = ax_gt_yy.imshow(stress_yy_gt, cmap='RdBu_r',
                               vmin=-stress_lim, vmax=stress_lim)
    plt.colorbar(im_gt_yy, ax=ax_gt_yy)
    ax_gt_yy.set_title('Ground Truth σyy')

    for idx in range(3):
        ax_yy = plt.subplot(4, 3, 3 * idx + 6)  # This gives indices 6, 9, 12
        case = cases[idx]
        im_yy = ax_yy.imshow(case['stress_tensor'][:, :, 1, 1],
                             cmap='RdBu_r', vmin=-stress_lim, vmax=stress_lim)
        plt.colorbar(im_yy, ax=ax_yy)
        ax_yy.set_title(f'σyy (Correlation: {case["yy_metrics"]["correlation"]:.3f})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'selected_cases.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_mesh_quality_vs_density(results, save_path=None):
    """Plot mesh quality vs density factor with algorithm and optimization encoded in color and marker"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Prepare data
    data = {
        'Algorithm': [],
        'Optimization': [],
        'Density Factor': [],
        'Mean Quality': []
    }

    for run in successful_runs:
        data['Algorithm'].append(run['parameters']['algorithm'])
        data['Optimization'].append('On' if run['parameters']['use_optimization'] else 'Off')
        data['Density Factor'].append(run['parameters']['density_factor'])
        data['Mean Quality'].append(run['mesh_metrics']['mean_quality'])

    plt.figure(figsize=(10, 6))

    # Create unique markers for optimization states
    markers = {'On': 'o', 'Off': 's'}

    # Plot for each algorithm and optimization combination
    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                plt.scatter(
                    [df for df, m in zip(data['Density Factor'], mask) if m],
                    [q for q, m in zip(data['Mean Quality'], mask) if m],
                    label=f'{algo} (Opt {opt})',
                    marker=markers[opt]
                )

    plt.xlabel('Density Factor')
    plt.ylabel('Mean Mesh Quality')
    plt.title('Mesh Quality vs Density Factor')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path / 'mesh_quality_vs_density.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_mesh_density_vs_correlation(results, save_path=None):
    """Plot mesh density vs solution correlation for different algorithms"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Prepare data
    data = {
        'Algorithm': [],
        'Optimization': [],
        'Density Factor': [],
        'XX Correlation': [],
        'YY Correlation': []
    }

    for run in successful_runs:
        data['Algorithm'].append(run['parameters']['algorithm'])
        data['Optimization'].append('On' if run['parameters']['use_optimization'] else 'Off')
        data['Density Factor'].append(run['parameters']['density_factor'])
        data['XX Correlation'].append(run['xx_metrics']['correlation'])
        data['YY Correlation'].append(run['yy_metrics']['correlation'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    markers = {'On': 'o', 'Off': 's'}

    # Plot density vs XX correlation
    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                ax1.scatter(
                    [df for df, m in zip(data['Density Factor'], mask) if m],
                    [c for c, m in zip(data['XX Correlation'], mask) if m],
                    label=f'{algo} (Opt {opt})',
                    marker=markers[opt]
                )

    ax1.set_xlabel('Density Factor')
    ax1.set_ylabel('XX Correlation')
    ax1.set_title('Mesh Density vs XX Correlation')
    ax1.legend()
    ax1.grid(True)

    # Plot density vs YY correlation
    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                ax2.scatter(
                    [df for df, m in zip(data['Density Factor'], mask) if m],
                    [c for c, m in zip(data['YY Correlation'], mask) if m],
                    label=f'{algo} (Opt {opt})',
                    marker=markers[opt]
                )

    ax2.set_xlabel('Density Factor')
    ax2.set_ylabel('YY Correlation')
    ax2.set_title('Mesh Density vs YY Correlation')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'density_vs_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_optimization_impact(results, save_path=None):
    """Plot the impact of optimization on solution quality"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Group runs by parameters excluding optimization
    param_groups = {}
    for run in successful_runs:
        key = (run['parameters']['algorithm'],
               run['parameters']['density_factor'],
               run['parameters']['youngs_modulus'])
        if key not in param_groups:
            param_groups[key] = {'opt_on': None, 'opt_off': None}

        if run['parameters']['use_optimization']:
            param_groups[key]['opt_on'] = run
        else:
            param_groups[key]['opt_off'] = run

    # Prepare data for plotting
    data = {
        'Algorithm': [],
        'Density Factor': [],
        'XX Improvement': [],
        'YY Improvement': []
    }

    for params, runs in param_groups.items():
        if runs['opt_on'] and runs['opt_off']:  # Only compare when we have both cases
            data['Algorithm'].append(params[0])
            data['Density Factor'].append(params[1])
            data['XX Improvement'].append(
                runs['opt_on']['xx_metrics']['correlation'] -
                runs['opt_off']['xx_metrics']['correlation']
            )
            data['YY Improvement'].append(
                runs['opt_on']['yy_metrics']['correlation'] -
                runs['opt_off']['yy_metrics']['correlation']
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot XX improvement
    for algo in set(data['Algorithm']):
        mask = [a == algo for a in data['Algorithm']]
        ax1.scatter(
            [df for df, m in zip(data['Density Factor'], mask) if m],
            [imp for imp, m in zip(data['XX Improvement'], mask) if m],
            label=algo
        )

    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Density Factor')
    ax1.set_ylabel('XX Correlation Improvement')
    ax1.set_title('Impact of Optimization on XX Stress')
    ax1.legend()
    ax1.grid(True)

    # Plot YY improvement
    for algo in set(data['Algorithm']):
        mask = [a == algo for a in data['Algorithm']]
        ax2.scatter(
            [df for df, m in zip(data['Density Factor'], mask) if m],
            [imp for imp, m in zip(data['YY Improvement'], mask) if m],
            label=algo
        )

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Density Factor')
    ax2.set_ylabel('YY Correlation Improvement')
    ax2.set_title('Impact of Optimization on YY Stress')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'optimization_impact.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_youngs_impact(results, save_path=None):
    """Plot the impact of Young's modulus on solution quality"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Prepare data
    data = {
        'Algorithm': [],
        'Optimization': [],
        'Young\'s Modulus': [],
        'XX Correlation': [],
        'YY Correlation': []
    }

    for run in successful_runs:
        data['Algorithm'].append(run['parameters']['algorithm'])
        data['Optimization'].append('On' if run['parameters']['use_optimization'] else 'Off')
        data['Young\'s Modulus'].append(run['parameters']['youngs_modulus'])
        data['XX Correlation'].append(run['xx_metrics']['correlation'])
        data['YY Correlation'].append(run['yy_metrics']['correlation'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    markers = {'On': 'o', 'Off': 's'}

    # Plot XX correlation
    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                ax1.scatter(
                    [e for e, m in zip(data['Young\'s Modulus'], mask) if m],
                    [c for c, m in zip(data['XX Correlation'], mask) if m],
                    label=f'{algo} (Opt {opt})',
                    marker=markers[opt]
                )

    ax1.set_xlabel('Young\'s Modulus')
    ax1.set_ylabel('XX Correlation')
    ax1.set_title('Impact on XX Stress Component')
    ax1.legend()
    ax1.grid(True)

    # Plot YY correlation
    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                ax2.scatter(
                    [e for e, m in zip(data['Young\'s Modulus'], mask) if m],
                    [c for c, m in zip(data['YY Correlation'], mask) if m],
                    label=f'{algo} (Opt {opt})',
                    marker=markers[opt]
                )

    ax2.set_xlabel('Young\'s Modulus')
    ax2.set_ylabel('YY Correlation')
    ax2.set_title('Impact on YY Stress Component')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'youngs_impact.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_solution_ranking(results, save_path=None):
    """Plot ranking of solutions based on correlation metrics"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Calculate combined metric (average of XX and YY correlations)
    for run in successful_runs:
        run['combined_correlation'] = (run['xx_metrics']['correlation'] +
                                       run['yy_metrics']['correlation']) / 2

    # Sort runs by combined correlation
    sorted_runs = sorted(successful_runs,
                         key=lambda x: x['combined_correlation'],
                         reverse=True)

    # Prepare data for plotting
    ranks = range(1, len(sorted_runs) + 1)
    correlations = [run['combined_correlation'] for run in sorted_runs]

    # Create parameter strings for labels
    labels = [f"df={run['parameters']['density_factor']}, "
              f"algo={run['parameters']['algorithm']}, "
              f"opt={run['parameters']['use_optimization']}, "
              f"E={run['parameters']['youngs_modulus']}"
              for run in sorted_runs]

    plt.figure(figsize=(15, 8))
    plt.plot(ranks, correlations, 'o-')
    plt.xlabel('Rank')
    plt.ylabel('Average Correlation')
    plt.title('Solution Ranking by Average Correlation')

    # Add parameter details for top 5 and bottom 5 cases
    for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
        plt.annotate(labels[i], (ranks[i], correlations[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, rotation=45, ha='left')

    plt.grid(True)
    if save_path:
        plt.savefig(save_path / 'solution_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_quality_vs_solution(results, save_path=None):
    """Plot mesh quality metrics vs solution correlation with algorithm and optimization encoding"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Prepare data
    data = {
        'Algorithm': [],
        'Optimization': [],
        'Mean Quality': [],
        'XX Correlation': [],
        'YY Correlation': []
    }

    for run in successful_runs:
        data['Algorithm'].append(run['parameters']['algorithm'])
        data['Optimization'].append('On' if run['parameters']['use_optimization'] else 'Off')
        data['Mean Quality'].append(run['mesh_metrics']['mean_quality'])
        data['XX Correlation'].append(run['xx_metrics']['correlation'])
        data['YY Correlation'].append(run['yy_metrics']['correlation'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    markers = {'On': 'o', 'Off': 's'}

    # Plot mesh quality vs XX correlation
    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                ax1.scatter(
                    [q for q, m in zip(data['Mean Quality'], mask) if m],
                    [c for c, m in zip(data['XX Correlation'], mask) if m],
                    label=f'{algo} (Opt {opt})',
                    marker=markers[opt]
                )

    ax1.set_xlabel('Mean Mesh Quality')
    ax1.set_ylabel('XX Correlation')
    ax1.set_title('Mesh Quality vs XX Correlation')
    ax1.legend()
    ax1.grid(True)

    # Plot mesh quality vs YY correlation
    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                ax2.scatter(
                    [q for q, m in zip(data['Mean Quality'], mask) if m],
                    [c for c, m in zip(data['YY Correlation'], mask) if m],
                    label=f'{algo} (Opt {opt})',
                    marker=markers[opt]
                )

    ax2.set_xlabel('Mean Mesh Quality')
    ax2.set_ylabel('YY Correlation')
    ax2.set_title('Mesh Quality vs YY Correlation')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'quality_vs_solution.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Load results
    current_dir = Path(__file__).parent
    benchmark_dir = current_dir.parent / 'benchmarks/synthetic_cell_physical_units'

    with open(current_dir / 'parameter_sweep_results' / 'msm_parameter_sweep.pickle', 'rb') as f:
        results = pickle.load(f)

    # Load ground truth
    ground_truth = load_ground_truth(benchmark_dir)

    # Create output directory for plots
    plot_dir = current_dir / 'parameter_sweep_plots'
    plot_dir.mkdir(exist_ok=True)

    # Generate all plots
    plot_selected_cases(results, ground_truth, plot_dir)
    # plot_mesh_quality_vs_density(results, plot_dir)
    # plot_mesh_density_vs_correlation(results, plot_dir)
    # plot_optimization_impact(results, plot_dir)
    # plot_youngs_impact(results, plot_dir)
    # plot_solution_ranking(results, plot_dir)
    # plot_quality_vs_solution(results, plot_dir)


if __name__ == "__main__":
    main()
