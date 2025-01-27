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

    # Find the best algorithm based on average correlation
    successful_runs = [run for run in results['runs'] if run['success']]
    for run in successful_runs:
        run['avg_corr'] = (run['xx_metrics']['correlation'] + run['yy_metrics']['correlation']) / 2

    # Group by algorithm and find the best one
    algo_performances = {}
    for run in successful_runs:
        algo = run['parameters']['algorithm']
        if algo not in algo_performances:
            algo_performances[algo] = []
        algo_performances[algo].append(run['avg_corr'])

    best_algo = max(algo_performances.items(), key=lambda x: np.mean(x[1]))[0]

    # Get all cases with the best algorithm
    best_algo_runs = [
        run for run in successful_runs
        if run['parameters']['algorithm'] == best_algo
    ]

    # Define the densities we want and their labels
    density_mapping = {
        0.05: ('Coarse Mesh', 2),  # index 2 means it will appear last
        0.01: ('Medium Mesh', 1),  # index 1 means it will appear in the middle
        0.005: ('Fine Mesh', 0)  # index 0 means it will appear first
    }

    # Sort cases into the correct slots
    cases = [None] * 3
    for run in best_algo_runs:
        df = run['parameters']['density_factor']
        if df in density_mapping:
            label, idx = density_mapping[df]
            cases[idx] = (run, label)

    # Ensure we found all three densities
    if None in cases:
        raise ValueError("Could not find all required density factors in the results")

    # Unpack the cases and titles
    cases, titles = zip(*cases)

    # Create figure with 4x4 grid
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle(f'Mesh Refinement Comparison (Algorithm {best_algo})', fontsize=16)

    # Fixed stress limits
    stress_lim = 0.004

    # First column: empty plot and meshes
    plt.subplot(4, 3, 1).set_visible(False)  # Empty plot

    # Plot meshes (first column, rows 2-4)
    for idx in range(3):
        ax_mesh = plt.subplot(4, 3, 3 * idx + 4)
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
        ax_xx = plt.subplot(4, 3, 3 * idx + 5)
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
        ax_yy = plt.subplot(4, 3, 3 * idx + 6)
        case = cases[idx]
        im_yy = ax_yy.imshow(case['stress_tensor'][:, :, 1, 1],
                             cmap='RdBu_r', vmin=-stress_lim, vmax=stress_lim)
        plt.colorbar(im_yy, ax=ax_yy)
        ax_yy.set_title(f'σyy (Correlation: {case["yy_metrics"]["correlation"]:.3f})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'selected_cases.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_residual_correlation(results, save_path=None):
    """Plot residual vs correlation"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Prepare data
    data = {
        'Algorithm': [],
        'Optimization': [],
        'Residual': [],
        'Average Correlation': []
    }

    for run in successful_runs:
        data['Algorithm'].append(run['parameters']['algorithm'])
        data['Optimization'].append('On' if run['parameters']['use_optimization'] else 'Off')
        data['Residual'].append(run['numerical_metrics']['residual_norm'])
        data['Average Correlation'].append(
            (run['xx_metrics']['correlation'] + run['yy_metrics']['correlation']) / 2
        )

    plt.figure(figsize=(10, 6))
    markers = {'On': 'o', 'Off': 's'}

    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                plt.scatter(
                    [r for r, m in zip(data['Residual'], mask) if m],
                    [c for c, m in zip(data['Average Correlation'], mask) if m],
                    label=f'Algorithm {algo} (Opt {opt})',
                    marker=markers[opt]
                )

    plt.xscale('log')
    plt.xlabel('Residual Norm')
    plt.ylabel('Average Correlation')
    plt.title('Solution Residual vs Correlation')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path / 'residual_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_condition_density(results, save_path=None):
    """Plot condition number vs mesh density"""
    successful_runs = [run for run in results['runs'] if run['success']]

    # Prepare data
    data = {
        'Algorithm': [],
        'Optimization': [],
        'Density Factor': [],
        'Condition Number': []
    }

    for run in successful_runs:
        data['Algorithm'].append(run['parameters']['algorithm'])
        data['Optimization'].append('On' if run['parameters']['use_optimization'] else 'Off')
        data['Density Factor'].append(run['parameters']['density_factor'])
        data['Condition Number'].append(run['numerical_metrics']['condition_number'])

    plt.figure(figsize=(10, 6))
    markers = {'On': 'o', 'Off': 's'}

    for algo in set(data['Algorithm']):
        for opt in ['On', 'Off']:
            mask = [(a == algo and o == opt) for a, o in zip(data['Algorithm'], data['Optimization'])]
            if any(mask):
                plt.scatter(
                    [df for df, m in zip(data['Density Factor'], mask) if m],
                    [cn for cn, m in zip(data['Condition Number'], mask) if m],
                    label=f'Algorithm {algo} (Opt {opt})',
                    marker=markers[opt]
                )

    plt.yscale('log')
    plt.xlabel('Density Factor')
    plt.ylabel('Condition Number')
    plt.title('Mesh Density vs Condition Number')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path / 'condition_density.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_solution_ranking(results, save_path=None):
    """Plot ranking of solutions based on correlation metrics with detailed parameter labels"""
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
    labels = []
    for run in sorted_runs:
        label = (f"df={run['parameters']['density_factor']}, "
                 f"algo={run['parameters']['algorithm']}, "
                 f"opt={run['parameters']['use_optimization']}, "
                 f"E={run['parameters']['youngs_modulus']}, "
                 f"corr={run['combined_correlation']:.3f}, "
                 f"res={run['numerical_metrics']['residual_norm']:.2e}, "
                 f"cond={run['numerical_metrics']['condition_number']:.2e}")
        labels.append(label)

    plt.figure(figsize=(15, 10))
    plt.plot(ranks, correlations, 'o-')
    plt.xlabel('Rank')
    plt.ylabel('Average Correlation')
    plt.title('Solution Ranking by Average Correlation')

    # Add parameter details for all points with smaller font and angled text
    for i in range(len(ranks)):
        plt.annotate(labels[i], (ranks[i], correlations[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=6, rotation=45, ha='left')

    plt.grid(True)
    if save_path:
        plt.savefig(save_path / 'solution_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Load results
    current_dir = Path(__file__).parent
    benchmark_dir = current_dir.parent / 'benchmarks/synthetic_cell'

    with open(current_dir / 'parameter_sweep_results' / 'msm_parameter_sweep.pickle', 'rb') as f:
        results = pickle.load(f)

    # Load ground truth
    ground_truth = load_ground_truth(benchmark_dir)

    # Create output directory for plots
    plot_dir = current_dir / 'parameter_sweep_plots'
    plot_dir.mkdir(exist_ok=True)

    # Generate all plots
    plot_selected_cases(results, ground_truth, plot_dir)
    plot_residual_correlation(results, plot_dir)
    plot_condition_density(results, plot_dir)
    plot_solution_ranking(results, plot_dir)


if __name__ == "__main__":
    main()