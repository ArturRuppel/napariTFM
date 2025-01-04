import numpy as np
import matplotlib.pyplot as plt
from msm import MonolayerStressMicroscopy


def plot_validation_results(t_x_true, t_y_true, sigma_xx_true, sigma_yy_true,
                            sigma_xx_calc, sigma_yy_calc, mask):
    """Plot comparison between true and calculated stress fields"""

    # Create stress tensor figure (2x2 grid)
    fig_stress = plt.figure(figsize=(12, 10))
    plt.suptitle('Stress Tensor Validation', fontsize=14)

    # Create subplots for each stress component
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
    # σxx
    im5 = plot_component(ax_s1, sigma_xx_true, 'True σxx (Pa)', vmax_xx)
    plt.colorbar(im5, ax=ax_s1)

    im6 = plot_component(ax_s2, sigma_xx_calc, 'Calculated σxx (Pa)', vmax_xx)
    plt.colorbar(im6, ax=ax_s2)

    # σyy
    im7 = plot_component(ax_s3, sigma_yy_true, 'True σyy (Pa)', vmax_yy)
    plt.colorbar(im7, ax=ax_s3)

    im8 = plot_component(ax_s4, sigma_yy_calc, 'Calculated σyy (Pa)', vmax_yy)
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


def pad_array(array, padding_width=5):
    """Pad array with zeros around the border"""
    return np.pad(array, padding_width, mode='constant', constant_values=0)


def validate_msm_with_fem_data(t_x_true, t_y_true, sigma_xx_true, sigma_yy_true, mask, pixelsize=0.8e-6, padding_width=5):
    """
    Validate MSM implementation using FEM simulation data with padded arrays
    All quantities in SI units
    """
    # Pad all input arrays
    t_x_padded = pad_array(t_x_true, padding_width)
    t_y_padded = pad_array(t_y_true, padding_width)
    sigma_xx_padded = pad_array(sigma_xx_true, padding_width)
    sigma_yy_padded = pad_array(sigma_yy_true, padding_width)
    mask_padded = pad_array(mask, padding_width)

    from scipy.ndimage import binary_dilation
    kernel = np.ones((4, 4), np.uint8)
    dilated = binary_dilation(mask_padded, kernel, iterations=1)
    mask_padded = dilated

    # Initialize MSM calculator
    msm = MonolayerStressMicroscopy(sigma=0.5, pixelsize=pixelsize * 1e6, base_refinement=0.8, boundary_refinement=3.0, gradient_refinement=1.0)

    # Generate and plot mesh using the built-in method
    nodes, elements = msm.mesh_generator.generate_mesh(mask_padded)
    mesh_fig = msm.mesh_generator.plot_mesh(nodes, elements, mask_padded)
    plt.title('Triangular Mesh with Padded Domain')
    mesh_fig.tight_layout()
    plt.show()

    # Calculate stress tensor from true tractions (Forward MSM)
    stress_tensor_calc = msm.calculate_stress_field(t_x_padded, t_y_padded, mask_padded) / pixelsize

    # Remove padding for comparison and visualization
    slice_obj = slice(padding_width, -padding_width)
    sigma_xx_calc = stress_tensor_calc[slice_obj, slice_obj, 0, 0]
    sigma_yy_calc = stress_tensor_calc[slice_obj, slice_obj, 1, 1]

    # Create visualizations
    fig_stress = plot_validation_results(
        t_x_true, t_y_true,
        sigma_xx_true, sigma_yy_true,
        sigma_xx_calc, sigma_yy_calc,
        mask
    )

    # Print metrics for forward calculation
    print("\nValidation Metrics:")
    print("-" * 50)

    # Forward MSM metrics (Traction -> Stress)
    print("\nForward MSM (Traction -> Stress):")
    for comp, true, calc in [
        ('σxx', sigma_xx_true, sigma_xx_calc),
        ('σyy', sigma_yy_true, sigma_yy_calc)
    ]:
        rmse, max_error, corr, rel_error = calculate_metrics(true, calc, mask)
        print(f"\n{comp}:")
        print(f"RMSE: {rmse:.2e} Pa")
        print(f"Max Error: {max_error:.2e} Pa")
        print(f"Correlation: {corr:.4f}")
        print(f"Relative Error: {rel_error:.4f}")

    return stress_tensor_calc[slice_obj, slice_obj], fig_stress, mesh_fig


if __name__ == "__main__":
    import pickle

    # Load the FEM simulation data
    folder = "C:/Users/aruppel/Desktop/test_MSM"
    doublet_FEM_simulation = pickle.load(open(folder + "/FEM_doublets.dat", "rb"))

    # Extract the ground truth data
    t_x_true = doublet_FEM_simulation["feedback0.0"]["t_x"][:, :, 0]
    t_y_true = doublet_FEM_simulation["feedback0.0"]["t_y"][:, :, 0]
    sigma_xx_true = doublet_FEM_simulation["feedback0.0"]["sigma_xx"][:, :, 0]
    sigma_yy_true = doublet_FEM_simulation["feedback0.0"]["sigma_yy"][:, :, 0]

    # Create mask based on stress data
    mask = sigma_xx_true > 0.00006

    # Run the validation
    results = validate_msm_with_fem_data(
        t_x_true, t_y_true,
        sigma_xx_true, sigma_yy_true,
        mask,
        pixelsize=0.8e-6,  # 0.8 µm pixel size
        padding_width=5
    )

    plt.show()