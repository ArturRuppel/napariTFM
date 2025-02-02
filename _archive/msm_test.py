import numpy as np
import matplotlib.pyplot as plt
from napariTFM._archive.msm_optimized import MonolayerStressMicroscopy
from inverse_msm import calculate_traction_from_stress


def create_synthetic_traction_field(shape=(100, 100), center_distance=30, spot_radius=8, force_magnitude=500):
    """
    Create synthetic traction field with 4 spots arranged in a square pattern,
    with forces pointing towards the center. Modified for smoother force distribution.
    """
    # Initialize arrays
    t_x = np.zeros(shape)
    t_y = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)

    # Calculate center point
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # Define spot centers
    spots = [
        (center_y - center_distance, center_x - center_distance),  # Top left
        (center_y - center_distance, center_x + center_distance),  # Top right
        (center_y + center_distance, center_x - center_distance),  # Bottom left
        (center_y + center_distance, center_x + center_distance)  # Bottom right
    ]

    # Create coordinate grids for entire field
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    # Generate smoother force spots
    for spot_y, spot_x in spots:
        # Calculate radial distance from spot center
        r = np.sqrt((x - spot_x) ** 2 + (y - spot_y) ** 2)

        # Create smoother gaussian force profile with wider extent
        force_profile = force_magnitude * np.exp(-0.5 * (r / spot_radius) ** 2)

        # Calculate direction vectors towards center for all points
        dx = center_x - x
        dy = center_y - y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Avoid division by zero
        distance[distance == 0] = 1

        # Normalize direction vectors
        dx = dx / distance
        dy = dy / distance

        # Add forces weighted by gaussian profile
        t_x += dx * force_profile
        t_y += dy * force_profile

        # Update mask
        mask = mask | (r <= spot_radius * 3)  # Wider mask

    # Add a buffer to the mask
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, iterations=2)

    return t_x, t_y, mask


def calculate_detailed_metrics(t_x_orig, t_y_orig, t_x_calc, t_y_calc, mask):
    """Calculate comprehensive error metrics"""
    # Calculate magnitudes
    t_mag_orig = np.sqrt(t_x_orig ** 2 + t_y_orig ** 2)
    t_mag_calc = np.sqrt(t_x_calc ** 2 + t_y_calc ** 2)

    # Mask the data
    t_mag_orig_masked = t_mag_orig[mask]
    t_mag_calc_masked = t_mag_calc[mask]
    t_x_orig_masked = t_x_orig[mask]
    t_y_orig_masked = t_y_orig[mask]
    t_x_calc_masked = t_x_calc[mask]
    t_y_calc_masked = t_y_calc[mask]

    # Magnitude metrics
    rmse = np.sqrt(np.mean((t_mag_calc_masked - t_mag_orig_masked) ** 2))
    nrmse = rmse / np.mean(t_mag_orig_masked)
    correlation = np.corrcoef(t_mag_orig_masked, t_mag_calc_masked)[0, 1]

    # Direction metrics
    angle_orig = np.arctan2(t_y_orig_masked, t_x_orig_masked)
    angle_calc = np.arctan2(t_y_calc_masked, t_x_calc_masked)
    angle_diff = np.abs(np.rad2deg(angle_orig - angle_calc))
    mean_angle_error = np.mean(np.minimum(angle_diff, 360 - angle_diff))

    # Component-wise metrics
    rmse_x = np.sqrt(np.mean((t_x_calc_masked - t_x_orig_masked) ** 2))
    rmse_y = np.sqrt(np.mean((t_y_calc_masked - t_y_orig_masked) ** 2))

    return {
        'RMSE': rmse,
        'NRMSE': nrmse,
        'Correlation': correlation,
        'Mean Angular Error (degrees)': mean_angle_error,
        'RMSE X': rmse_x,
        'RMSE Y': rmse_y
    }


def plot_stress_tensor(stress_tensor, mask, title):
    """Plot stress tensor components"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    components = [
        (0, 0, 'σxx'),
        (0, 1, 'σxy'),
        (1, 0, 'σyx'),
        (1, 1, 'σyy')
    ]

    for (i, j, label), ax in zip(components, axes.flat):
        data = stress_tensor[:, :, i, j]
        vmax = np.percentile(np.abs(data[mask]), 95)
        im = ax.imshow(data, cmap='RdBu', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(label)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_comparison(t_x_orig, t_y_orig, t_x_calc, t_y_calc, mask, title):
    """Create comparison plot with magnitude difference"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Calculate magnitudes
    t_mag_orig = np.sqrt(t_x_orig ** 2 + t_y_orig ** 2)
    t_mag_calc = np.sqrt(t_x_calc ** 2 + t_y_calc ** 2)

    # Use same colorscale for magnitude plots
    vmax = np.percentile(t_mag_orig[mask], 95)

    # Plot original tractions
    im1 = axes[0, 0].imshow(t_mag_orig, cmap='viridis', vmax=vmax)
    plt.colorbar(im1, ax=axes[0, 0], label='Traction magnitude (Pa)')

    # Add quiver plot
    spacing = 5
    y, x = np.mgrid[:t_x_orig.shape[0]:spacing, :t_x_orig.shape[1]:spacing]
    scale = np.max(t_mag_orig) * 2

    axes[0, 0].quiver(x, y,
                      t_x_orig[::spacing, ::spacing],
                      -t_y_orig[::spacing, ::spacing],
                      color='white', scale=scale)
    axes[0, 0].set_title('Original Traction Field')

    # Plot calculated tractions
    im2 = axes[0, 1].imshow(t_mag_calc, cmap='viridis', vmax=vmax)
    plt.colorbar(im2, ax=axes[0, 1], label='Traction magnitude (Pa)')

    axes[0, 1].quiver(x, y,
                      t_x_calc[::spacing, ::spacing],
                      -t_y_calc[::spacing, ::spacing],
                      color='white', scale=10*scale)
    axes[0, 1].set_title('Recalculated Traction Field')

    # Plot magnitude difference
    diff = t_mag_calc - t_mag_orig
    vmax_diff = np.percentile(np.abs(diff[mask]), 95)
    im3 = axes[1, 0].imshow(diff, cmap='RdBu', vmin=-vmax_diff, vmax=vmax_diff)
    plt.colorbar(im3, ax=axes[1, 0], label='Magnitude Difference (Pa)')
    axes[1, 0].set_title('Magnitude Difference')

    # Plot angle difference
    angle_orig = np.arctan2(t_y_orig, t_x_orig)
    angle_calc = np.arctan2(t_y_calc, t_x_calc)
    angle_diff = np.rad2deg(np.arctan2(np.sin(angle_orig - angle_calc),
                                       np.cos(angle_orig - angle_calc)))
    im4 = axes[1, 1].imshow(angle_diff, cmap='RdBu', vmin=-180, vmax=180)
    plt.colorbar(im4, ax=axes[1, 1], label='Angle Difference (degrees)')
    axes[1, 1].set_title('Direction Difference')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    shape = (200, 200)
    pixelsize = 0.8  # microns per pixel

    # Create synthetic traction field
    print("Generating synthetic traction field...")
    t_x_orig, t_y_orig, mask = create_synthetic_traction_field(
        shape=shape,
        center_distance=50,
        spot_radius=12,  # Increased for smoother field
        force_magnitude=500
    )

    # Initialize MSM calculator
    print("Calculating stress tensor...")
    msm = MonolayerStressMicroscopy(pixelsize=pixelsize)

    # Calculate stress field
    stress_tensor = msm.calculate_stress_field(t_x_orig, t_y_orig, mask)
    stress_tensor = stress_tensor / (pixelsize * 1e-6)  # Convert to Pa

    # Plot stress tensor components
    plot_stress_tensor(stress_tensor, mask, "Stress Tensor Components")

    # Calculate traction forces from stress tensor
    print("Recalculating traction forces...")
    t_x_calc, t_y_calc = calculate_traction_from_stress(stress_tensor, mask, pixelsize)

    # Plot detailed comparison
    plot_comparison(t_x_orig, t_y_orig, t_x_calc, t_y_calc, mask,
                    "Synthetic Traction Field Validation")

    # Calculate and print detailed metrics
    metrics = calculate_detailed_metrics(t_x_orig, t_y_orig, t_x_calc, t_y_calc, mask)
    print("\nValidation Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")