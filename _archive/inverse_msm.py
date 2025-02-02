import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def calculate_traction_from_stress(stress_tensor, mask, pixelsize):
    """
    Calculate traction forces from stress tensor using the equilibrium equation:
    t = div(σ) where div(σ) is the divergence of the stress tensor

    Args:
        stress_tensor: 4D array (height, width, 2, 2) containing stress components
        mask: Boolean mask of valid area
        pixelsize: Pixel size in microns

    Returns:
        tuple: (t_x, t_y) traction force components in Pascal
    """
    # Convert pixelsize to meters
    dx = dy = pixelsize * 1e-6

    # Extract stress components
    sigma_xx = stress_tensor[:, :, 0, 0]
    sigma_yy = stress_tensor[:, :, 1, 1]
    sigma_xy = stress_tensor[:, :, 0, 1]

    # Apply small amount of smoothing to reduce numerical noise
    sigma_xx = gaussian_filter1d(gaussian_filter1d(sigma_xx, sigma=1, axis=0), sigma=1, axis=1)
    sigma_yy = gaussian_filter1d(gaussian_filter1d(sigma_yy, sigma=1, axis=0), sigma=1, axis=1)
    sigma_xy = gaussian_filter1d(gaussian_filter1d(sigma_xy, sigma=1, axis=0), sigma=1, axis=1)

    # Calculate derivatives using central differences
    def central_diff_x(f):
        """Central difference in x-direction"""
        diff = np.zeros_like(f)
        diff[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
        diff[:, 0] = (f[:, 1] - f[:, 0]) / dx  # Forward difference at left boundary
        diff[:, -1] = (f[:, -1] - f[:, -2]) / dx  # Backward difference at right boundary
        return diff

    def central_diff_y(f):
        """Central difference in y-direction"""
        diff = np.zeros_like(f)
        diff[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
        diff[0, :] = (f[1, :] - f[0, :]) / dy  # Forward difference at top boundary
        diff[-1, :] = (f[-1, :] - f[-2, :]) / dy  # Backward difference at bottom boundary
        return diff

    # Calculate divergence components
    t_x = central_diff_x(sigma_xx) + central_diff_y(sigma_xy)
    t_y = central_diff_x(sigma_xy) + central_diff_y(sigma_yy)

    # Apply mask
    t_x[~mask] = 0
    t_y[~mask] = 0

    return t_x, t_y


def plot_traction_comparison(t_x_orig, t_y_orig, t_x_calc, t_y_calc, mask):
    """
    Plot original and calculated traction fields side by side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate magnitudes
    traction_mag_orig = np.sqrt(t_x_orig ** 2 + t_y_orig ** 2)
    traction_mag_calc = np.sqrt(t_x_calc ** 2 + t_y_calc ** 2)

    # Use same scale for both plots
    vmax = np.percentile(traction_mag_orig[mask], 99.9) * 1.2

    # Original tractions
    im1 = ax1.imshow(traction_mag_orig, vmax=vmax)
    plt.colorbar(im1, ax=ax1, label='Traction (Pa)')

    # Add quiver plot
    spacing = 10
    y, x = np.mgrid[:t_x_orig.shape[0]:spacing, :t_x_orig.shape[1]:spacing]
    scale = np.percentile(traction_mag_orig[mask], 95) * 20

    ax1.quiver(x, y,
               t_x_orig[::spacing, ::spacing],
               -t_y_orig[::spacing, ::spacing],
               color='white',
               scale=scale)
    ax1.set_title('Original Traction Field')

    # Recalculated tractions
    im2 = ax2.imshow(traction_mag_calc, vmax=vmax)
    plt.colorbar(im2, ax=ax2, label='Traction (Pa)')

    ax2.quiver(x, y,
               t_x_calc[::spacing, ::spacing],
               -t_y_calc[::spacing, ::spacing],
               color='white',
               scale=scale)
    ax2.set_title('Recalculated Traction Field')

    plt.tight_layout()
    plt.show()


def calculate_error_metrics(t_x_orig, t_y_orig, t_x_calc, t_y_calc, mask):
    """
    Calculate error metrics between original and calculated traction fields
    """
    # Calculate magnitudes
    t_mag_orig = np.sqrt(t_x_orig ** 2 + t_y_orig ** 2)
    t_mag_calc = np.sqrt(t_x_calc ** 2 + t_y_calc ** 2)

    # Calculate metrics only for masked region
    t_mag_orig_masked = t_mag_orig[mask]
    t_mag_calc_masked = t_mag_calc[mask]

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((t_mag_calc_masked - t_mag_orig_masked) ** 2))

    # Normalized RMSE
    nrmse = rmse / np.mean(t_mag_orig_masked)

    # Pearson correlation coefficient
    correlation = np.corrcoef(t_mag_orig_masked, t_mag_calc_masked)[0, 1]

    # Direction error (in degrees)
    angle_orig = np.arctan2(t_y_orig[mask], t_x_orig[mask])
    angle_calc = np.arctan2(t_y_calc[mask], t_x_calc[mask])
    angle_diff = np.abs(np.rad2deg(angle_orig - angle_calc))
    mean_angle_error = np.mean(np.minimum(angle_diff, 360 - angle_diff))

    return {
        'RMSE': rmse,
        'NRMSE': nrmse,
        'Correlation': correlation,
        'Mean Angular Error': mean_angle_error
    }

