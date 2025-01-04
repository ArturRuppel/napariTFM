import numpy as np
import matplotlib.pyplot as plt
from inverse_msm import calculate_traction_from_stress
from napariTFM._archive.msm_optimized import MonolayerStressMicroscopy


def create_force_dipole_field(shape=(200, 200), center=None, orientation=np.pi / 4, strength=1.0, separation=20, pixelsize=0.8e-6):
    """
    Create analytical stress tensor field and traction forces for a force dipole.
    All quantities are in SI units:
    - strength: Force in Newtons
    - separation: Distance in meters
    - pixelsize: Meters per pixel
    Returns stress tensor in Pa and traction forces in Pa
    """
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)

    # Convert separation from meters to pixels
    separation_px = separation / pixelsize

    # Create coordinate grid in meters
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    x = x * pixelsize  # Convert to meters
    y = y * pixelsize  # Convert to meters

    # Calculate dipole positions in meters
    half_sep = separation / 2
    x1 = center[1] * pixelsize + half_sep * np.cos(orientation)
    y1 = center[0] * pixelsize + half_sep * np.sin(orientation)
    x2 = center[1] * pixelsize - half_sep * np.cos(orientation)
    y2 = center[0] * pixelsize - half_sep * np.sin(orientation)

    # Calculate distances and angles from each force point
    r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    theta1 = np.arctan2(y - y1, x - x1)
    theta2 = np.arctan2(y - y2, x - x2)

    # Set minimum distance for stress calculation
    min_distance = pixelsize  # One pixel width
    r1[r1 < min_distance] = min_distance
    r2[r2 < min_distance] = min_distance

    def point_force_stress(r, theta, sign):
        """Stress components for a point force in Pa"""
        factor = sign * strength / (2 * np.pi * r ** 2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        sigma_rr = factor * 2 * cos_theta
        sigma_rt = factor * sin_theta

        sigma_xx = sigma_rr * cos_theta ** 2 - sigma_rt * sin_theta * cos_theta
        sigma_yy = sigma_rr * sin_theta ** 2 + sigma_rt * sin_theta * cos_theta
        sigma_xy = sigma_rr * sin_theta * cos_theta + sigma_rt * (cos_theta ** 2 - sin_theta ** 2)

        return sigma_xx, sigma_yy, sigma_xy

    # Combine contributions from both forces
    sigma_xx1, sigma_yy1, sigma_xy1 = point_force_stress(r1, theta1, 1.0)
    sigma_xx2, sigma_yy2, sigma_xy2 = point_force_stress(r2, theta2, -1.0)

    sigma_xx = sigma_xx1 + sigma_xx2
    sigma_yy = sigma_yy1 + sigma_yy2
    sigma_xy = sigma_xy1 + sigma_xy2

    # Create stress tensor (in Pa)
    stress_tensor = np.zeros((*shape, 2, 2))
    stress_tensor[..., 0, 0] = sigma_xx
    stress_tensor[..., 1, 1] = sigma_yy
    stress_tensor[..., 0, 1] = sigma_xy
    stress_tensor[..., 1, 0] = sigma_xy

    # Calculate analytical traction forces (in Pa) using divergence
    t_x = np.gradient(sigma_xx, pixelsize, axis=1) + np.gradient(sigma_xy, pixelsize, axis=0)
    t_y = np.gradient(sigma_xy, pixelsize, axis=1) + np.gradient(sigma_yy, pixelsize, axis=0)

    return stress_tensor, t_x, t_y



def plot_validation_results(t_x_true, t_y_true, t_x_calc, t_y_calc, stress_true, stress_calc, mask):
    """Plot comparison between true and calculated fields for both traction and stress"""

    # Create two figures - one for traction, one for stress
    fig_traction = plt.figure(figsize=(12, 10))
    plt.suptitle('Traction Force Validation', fontsize=14)

    # Traction plots (2x2 grid)
    ax_t1 = plt.subplot(221)
    ax_t2 = plt.subplot(222)
    ax_t3 = plt.subplot(223)
    ax_t4 = plt.subplot(224)

    # Get common scale for each traction component
    vmax_x = np.nanpercentile(np.abs(t_x_true[mask]), 99)
    vmax_y = np.nanpercentile(np.abs(t_y_true[mask]), 99)

    def plot_component(ax, data, title, vmax):
        masked_data = np.copy(data)
        masked_data[~mask] = np.nan
        im = ax.imshow(masked_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        return im

    # Plot traction components
    im1 = plot_component(ax_t1, t_x_true, 'True Tx (Pa)', vmax_x)
    plt.colorbar(im1, ax=ax_t1)

    im2 = plot_component(ax_t2, t_x_calc, 'Calculated Tx (Pa)', vmax_x)
    plt.colorbar(im2, ax=ax_t2)

    im3 = plot_component(ax_t3, t_y_true, 'True Ty (Pa)', vmax_y)
    plt.colorbar(im3, ax=ax_t3)

    im4 = plot_component(ax_t4, t_y_calc, 'Calculated Ty (Pa)', vmax_y)
    plt.colorbar(im4, ax=ax_t4)

    plt.tight_layout()

    # Create stress tensor figure (3x2 grid)
    fig_stress = plt.figure(figsize=(12, 15))
    plt.suptitle('Stress Tensor Validation', fontsize=14)

    # Create subplots for each stress component
    ax_s1 = plt.subplot(321)
    ax_s2 = plt.subplot(322)
    ax_s3 = plt.subplot(323)
    ax_s4 = plt.subplot(324)
    ax_s5 = plt.subplot(325)
    ax_s6 = plt.subplot(326)

    # Get common scale for stress components
    vmax_xx = np.nanpercentile(np.abs(stress_true[..., 0, 0][mask]), 99)
    vmax_yy = np.nanpercentile(np.abs(stress_true[..., 1, 1][mask]), 99)
    vmax_xy = np.nanpercentile(np.abs(stress_true[..., 0, 1][mask]), 99)

    # Plot stress components
    # σxx
    im5 = plot_component(ax_s1, stress_true[..., 0, 0], 'True σxx (Pa)', vmax_xx)
    plt.colorbar(im5, ax=ax_s1)

    im6 = plot_component(ax_s2, stress_calc[..., 0, 0], 'Calculated σxx (Pa)', vmax_xx)
    plt.colorbar(im6, ax=ax_s2)

    # σyy
    im7 = plot_component(ax_s3, stress_true[..., 1, 1], 'True σyy (Pa)', vmax_yy)
    plt.colorbar(im7, ax=ax_s3)

    im8 = plot_component(ax_s4, stress_calc[..., 1, 1], 'Calculated σyy (Pa)', vmax_yy)
    plt.colorbar(im8, ax=ax_s4)

    # σxy
    im9 = plot_component(ax_s5, stress_true[..., 0, 1], 'True σxy (Pa)', vmax_xy)
    plt.colorbar(im9, ax=ax_s5)

    im10 = plot_component(ax_s6, stress_calc[..., 0, 1], 'Calculated σxy (Pa)', vmax_xy)
    plt.colorbar(im10, ax=ax_s6)

    plt.tight_layout()

    return fig_traction, fig_stress


def validate_msm(shape=(200, 200), pixelsize=0.8e-6):
    """
    Validate MSM implementation using analytical solution
    All quantities in SI units
    """
    # Create analytical solution
    stress_tensor_true, t_x_true, t_y_true = create_force_dipole_field(
        shape=shape,
        orientation=np.pi / 4,
        strength=1e-9,  # 1 nN force
        separation=40e-6,  # 20 µm separation
        pixelsize=pixelsize
    )

    # Create mask (circular region around dipole)
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    center = (shape[0] // 2, shape[1] // 2)
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 < (shape[0] // 3) ** 2

    # Calculate traction forces using inverse MSM
    t_x_calc, t_y_calc = calculate_traction_from_stress(stress_tensor_true, mask, pixelsize * 1e6)

    # Calculate stress tensor from calculated tractions
    msm = MonolayerStressMicroscopy(pixelsize=pixelsize * 1e6)  # Convert to µm for the function
    stress_tensor_calc = msm.calculate_stress_field(t_x_calc, t_y_calc, mask)
    stress_tensor_calc = stress_tensor_calc / (pixelsize)  # Convert back to Pa

    # Calculate error metrics for both traction and stress
    def calculate_metrics(true, calc, mask):
        valid_mask = mask & ~np.isnan(true) & ~np.isnan(calc)
        rmse = np.sqrt(np.mean((true[valid_mask] - calc[valid_mask]) ** 2))
        max_error = np.max(np.abs(true[valid_mask] - calc[valid_mask]))
        correlation = np.corrcoef(true[valid_mask].flatten(), calc[valid_mask].flatten())[0, 1]
        rel_error = rmse / np.std(true[valid_mask])
        return rmse, max_error, correlation, rel_error

    # Print metrics
    print("\nValidation Metrics:")
    print("-" * 50)

    # Traction force metrics
    print("\nTraction Forces (Pa):")
    for comp, t_true, t_calc in [('X', t_x_true, t_x_calc), ('Y', t_y_true, t_y_calc)]:
        rmse, max_error, corr, rel_error = calculate_metrics(t_true, t_calc, mask)
        print(f"\n{comp}-component:")
        print(f"RMSE: {rmse:.2e} Pa")
        print(f"Max Error: {max_error:.2e} Pa")
        print(f"Correlation: {corr:.4f}")
        print(f"Relative Error: {rel_error:.4f}")

    # Stress tensor metrics
    print("\nStress Components (Pa):")
    components = [('xx', (0, 0)), ('yy', (1, 1)), ('xy', (0, 1))]
    for comp, idx in components:
        rmse, max_error, corr, rel_error = calculate_metrics(
            stress_tensor_true[..., idx[0], idx[1]],
            stress_tensor_calc[..., idx[0], idx[1]],
            mask
        )
        print(f"\nσ{comp}:")
        print(f"RMSE: {rmse:.2e} Pa")
        print(f"Max Error: {max_error:.2e} Pa")
        print(f"Correlation: {corr:.4f}")
        print(f"Relative Error: {rel_error:.4f}")

    # Create visualizations
    fig_traction, fig_stress = plot_validation_results(
        t_x_true, t_y_true, t_x_calc, t_y_calc,
        stress_tensor_true, stress_tensor_calc,
        mask
    )
    plt.show()

    return stress_tensor_true, t_x_true, t_y_true, t_x_calc, t_y_calc, stress_tensor_calc, mask


if __name__ == "__main__":
    results = validate_msm()