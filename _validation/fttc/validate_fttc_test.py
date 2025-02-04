import sys
from pathlib import Path
from typing import Dict, Tuple
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

from backend.fttc import FTTC
from backend.parameter_dataclasses import FTTCParameters

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from scipy.ndimage import uniform_filter
from typing import Dict


def calculate_advanced_metrics(calculated_forces: np.ndarray,
                               true_forces: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive error metrics for TFM force fields with robust NaN handling.

    Args:
        calculated_forces: Force field to evaluate (2 x H x W)
        true_forces: Ground truth force field (2 x H x W)

    Returns:
        Dictionary containing various error metrics
    """
    metrics = {}

    # Calculate force magnitudes
    calc_mag = np.sqrt(calculated_forces[0] ** 2 + calculated_forces[1] ** 2)
    true_mag = np.sqrt(true_forces[0] ** 2 + true_forces[1] ** 2)

    # Create valid data mask
    valid_mask = ~np.isnan(calc_mag) & ~np.isnan(true_mag)

    # === Gradient-based metrics ===
    # Compute gradients for both components
    grad_calc_x = np.gradient(np.nan_to_num(calculated_forces, 0), axis=2)
    grad_calc_y = np.gradient(np.nan_to_num(calculated_forces, 0), axis=1)
    grad_true_x = np.gradient(np.nan_to_num(true_forces, 0), axis=2)
    grad_true_y = np.gradient(np.nan_to_num(true_forces, 0), axis=1)

    # Combine x and y gradients and compute error
    grad_error = np.sqrt(np.mean(
        (grad_calc_x[:, valid_mask] - grad_true_x[:, valid_mask]) ** 2 +
        (grad_calc_y[:, valid_mask] - grad_true_y[:, valid_mask]) ** 2
    ))
    metrics['gradient_error'] = grad_error

    # === Frequency analysis ===
    def get_frequency_metrics(field1, field2, valid_mask):
        # Apply mask and prepare data
        f1 = np.nan_to_num(field1)[valid_mask]
        f2 = np.nan_to_num(field2)[valid_mask]

        # Compute 1D FFT (simpler but still informative)
        fft1 = np.fft.fft(f1)
        fft2 = np.fft.fft(f2)

        # Split into low and high frequency components
        n = len(fft1)
        low_freq_idx = slice(1, n // 4)  # Skip DC component
        high_freq_idx = slice(n // 4, n // 2)  # Only up to Nyquist frequency

        # Calculate errors in frequency bands
        low_freq_error = np.mean(np.abs(fft1[low_freq_idx] - fft2[low_freq_idx]) ** 2)
        high_freq_error = np.mean(np.abs(fft1[high_freq_idx] - fft2[high_freq_idx]) ** 2)

        return np.sqrt(low_freq_error), np.sqrt(high_freq_error)

    # Calculate frequency metrics for both components
    low_x, high_x = get_frequency_metrics(calculated_forces[0], true_forces[0], valid_mask)
    low_y, high_y = get_frequency_metrics(calculated_forces[1], true_forces[1], valid_mask)

    metrics['high_freq_error'] = (high_x + high_y) / 2
    metrics['low_freq_error'] = (low_x + low_y) / 2
    metrics['freq_ratio'] = metrics['high_freq_error'] / metrics['low_freq_error']

    # === Local variance analysis ===
    window_size = 5

    def compute_local_variance(field, mask):
        field = np.nan_to_num(field)
        local_mean = uniform_filter(field, size=window_size)
        local_mean2 = uniform_filter(field ** 2, size=window_size)
        variance = local_mean2 - local_mean ** 2
        return variance[mask]

    # Calculate local variance for both fields
    calc_var = compute_local_variance(calc_mag, valid_mask)
    true_var = compute_local_variance(true_mag, valid_mask)

    metrics['local_var_error'] = np.sqrt(np.mean((calc_var - true_var) ** 2))

    # === Peak region analysis ===
    peak_threshold = np.nanpercentile(true_mag, 90)
    peak_mask = (true_mag > peak_threshold) & valid_mask

    if np.any(peak_mask):
        # Error specifically in high-force regions
        peak_error = np.sqrt(np.mean(
            (calc_mag[peak_mask] - true_mag[peak_mask]) ** 2
        ))
        metrics['peak_region_error'] = peak_error

        # Local variance specifically around peaks
        peak_var_calc = compute_local_variance(calc_mag, peak_mask)
        peak_var_true = compute_local_variance(true_mag, peak_mask)
        metrics['peak_local_var_error'] = np.sqrt(np.mean((peak_var_calc - peak_var_true) ** 2))

    return metrics


def print_metric_comparison(simple_metrics: Dict[str, float],
                            fttc_metrics: Dict[str, float]):
    """Print a formatted comparison of metrics between methods."""
    print("\nDetailed Metric Comparison:")
    print("-" * 60)
    print(f"{'Metric':<30} {'Simple':<15} {'FTTC':<15}")
    print("-" * 60)

    for key in sorted(simple_metrics.keys()):
        simple_val = simple_metrics[key]
        fttc_val = fttc_metrics[key]

        # Normalize very large/small numbers for better readability
        magnitude = max(abs(simple_val), abs(fttc_val))
        if magnitude < 0.01 or magnitude > 1000:
            simple_str = f"{simple_val:,.2e}"
            fttc_str = f"{fttc_val:,.2e}"
        else:
            simple_str = f"{simple_val:.3f}"
            fttc_str = f"{fttc_val:.3f}"

        print(f"{key:<30} {simple_str:<15} {fttc_str:<15}")

    print("-" * 60)

def calculate_traction_stresses_simple(u, v, E, nu, pixelsize, alpha):
    '''This function takes a displacement field u and v, a gel rigidity E, it's poisson ratio nu, the size of a pixel in '''
    M, N = u.shape

    # pad displacement map with zeros until it has a shape of 2^n by 2^n because fourier transform is faster
    n = 2
    while (2 ** n < M) or (2 ** n < N):
        n = n + 1

    M2 = 2 ** n
    N2 = M2

    u_padded = np.zeros((M2, N2))
    v_padded = np.zeros((M2, N2))

    u_padded[:u.shape[0], :u.shape[1]] = u
    v_padded[:v.shape[0], :v.shape[1]] = v

    u_fft = fft2(u_padded)
    v_fft = fft2(v_padded)

    # remove component related to translation
    u_fft[0, 0] = 0
    v_fft[0, 0] = 0

    Kx1 = (2 * np.pi / pixelsize) / N2 * np.arange(int(N2 / 2))
    Kx2 = -(2 * np.pi / pixelsize) / N2 * (N2 - np.arange(int(N2 / 2), N2))
    Ky1 = (2 * np.pi / pixelsize) / M2 * np.arange(int(M2 / 2))
    Ky2 = -(2 * np.pi / pixelsize) / M2 * (M2 - np.arange(int(M2 / 2), M2))

    Kx = np.concatenate((Kx1, Kx2))
    Ky = np.concatenate((Ky1, Ky2))

    kx, ky = np.meshgrid(Kx, Ky)
    k = np.sqrt(kx ** 2 + ky ** 2)
    t_xt = np.zeros((M2, N2), dtype=complex)
    t_yt = np.zeros((M2, N2), dtype=complex)

    for i in np.arange(M2):
        for j in np.arange(N2):
            if i == M2 / 2 or j == N2 / 2:  # Nyquist frequency
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

            elif ~((i == 1) and (j == 1)):
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)
                Gt[0, 1] = - nu * kx[i, j] * ky[i, j]
                Gt[1, 0] = - nu * kx[i, j] * ky[i, j]

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

    t_x = ifft2(t_xt)
    t_y = ifft2(t_yt)
    traction_x = np.real(t_x)
    traction_y = np.real(t_y)

    return traction_x[0:M, 0:N], traction_y[0:M, 0:N]


def dilate_non_nan_region(data: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Dilate the non-NaN region of the data by adding a border of zeros.

    Args:
        data: Input array with NaN values
        width: Width of the border to add in pixels

    Returns:
        Array with dilated non-NaN region (filled with zeros in the dilated area)
    """
    # Create a mask of non-NaN values
    non_nan_mask = ~np.isnan(data)

    # Dilate the mask
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    dilated_mask = ndimage.binary_dilation(non_nan_mask,
                                           structure=structure,
                                           iterations=width)

    # Create output array
    result = np.copy(data)
    # Set newly dilated region (difference between dilated and original mask) to zero
    result[dilated_mask & ~non_nan_mask] = 0

    return result


def mask_calculated_forces(calculated_forces: np.ndarray, true_forces: np.ndarray) -> np.ndarray:
    """
    Mask calculated forces with NaN where true forces are NaN.

    Args:
        calculated_forces: (2 x H x W) array of calculated forces
        true_forces: (2 x H x W) array of true forces

    Returns:
        (2 x H x W) array of masked calculated forces
    """
    # Create a copy to avoid modifying the original array
    masked_forces = calculated_forces.copy()

    # Create mask where either component of true forces is NaN
    nan_mask = np.isnan(true_forces[0]) | np.isnan(true_forces[1])

    # Apply mask to both components of calculated forces
    masked_forces[0][nan_mask] = np.nan
    masked_forces[1][nan_mask] = np.nan

    return masked_forces


def calculate_error_metrics(calculated_forces: np.ndarray, true_forces: np.ndarray) -> dict:
    """Calculate error metrics between calculated and true forces."""
    # Calculate various error metrics (ignoring NaN values)
    rmse = np.sqrt(np.nanmean((calculated_forces - true_forces) ** 2))

    max_error = np.nanmax(np.abs(calculated_forces - true_forces))

    # For relative L2 error, we need to handle NaNs carefully
    diff = calculated_forces - true_forces
    rel_l2_error = np.sqrt(np.nansum(diff ** 2)) / np.sqrt(np.nansum(true_forces ** 2))

    # Calculate correlation coefficient for force magnitude (ignoring NaNs)
    calc_mag = np.sqrt(calculated_forces[0] ** 2 + calculated_forces[1] ** 2)
    true_mag = np.sqrt(true_forces[0] ** 2 + true_forces[1] ** 2)
    valid_mask = ~np.isnan(calc_mag) & ~np.isnan(true_mag)
    correlation = np.corrcoef(calc_mag[valid_mask].flatten(), true_mag[valid_mask].flatten())[0, 1]

    return {
        'rmse': rmse,
        'max_error': max_error,
        'relative_l2_error': rel_l2_error,
        'correlation': correlation
    }


def plot_three_way_comparison(coords: Tuple[np.ndarray, np.ndarray],
                              true_forces: np.ndarray,
                              simple_forces: np.ndarray,
                              fttc_forces: np.ndarray,
                              save_path: Optional[Path] = None):
    """Plot comparison between true forces and both calculation methods."""
    x, y = coords

    # Calculate force magnitudes
    true_mag = np.sqrt(true_forces[0] ** 2 + true_forces[1] ** 2)
    simple_mag = np.sqrt(simple_forces[0] ** 2 + simple_forces[1] ** 2)
    fttc_mag = np.sqrt(fttc_forces[0] ** 2 + fttc_forces[1] ** 2)

    # Set up common colormap range (ignoring NaNs)
    vmin = min(np.nanmin(true_mag), np.nanmin(simple_mag), np.nanmin(fttc_mag))
    vmax = max(np.nanmax(true_mag), np.nanmax(simple_mag), np.nanmax(fttc_mag))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Calculate extent for imshow
    extent = [x.min(), x.max(), y.min(), y.max()]

    # Create downsampled coordinate grids for quiver
    step = 4
    x_quiver = x[::step, ::step]
    y_quiver = y[::-step, ::step]

    # Plot true forces
    im1 = ax1.imshow(true_mag,
                     origin='upper',
                     extent=extent,
                     vmin=vmin,
                     vmax=vmax,
                     aspect='equal')

    true_x = true_forces[0][::step, ::step]
    true_y = -true_forces[1][::step, ::step]
    mask = ~np.isnan(true_x) & ~np.isnan(true_y)
    # q1 = ax1.quiver(x_quiver[mask], y_quiver[mask],
    #                 true_x[mask], true_y[mask])
    ax1.set_title('True Forces')
    plt.colorbar(im1, ax=ax1, label='Force magnitude (Pa)')

    # Plot simple calculation forces
    im2 = ax2.imshow(simple_mag,
                     origin='upper',
                     extent=extent,
                     vmin=vmin,
                     vmax=vmax,
                     aspect='equal')

    simple_x = simple_forces[0][::step, ::step]
    simple_y = -simple_forces[1][::step, ::step]
    mask = ~np.isnan(simple_x) & ~np.isnan(simple_y)
    # q2 = ax2.quiver(x_quiver[mask], y_quiver[mask],
    #                 simple_x[mask], simple_y[mask])
    ax2.set_title('Simple FTTC Forces')
    plt.colorbar(im2, ax=ax2, label='Force magnitude (Pa)')

    # Plot FTTC class forces
    im3 = ax3.imshow(fttc_mag,
                     origin='upper',
                     extent=extent,
                     vmin=vmin,
                     vmax=vmax,
                     aspect='equal')

    fttc_x = fttc_forces[0][::step, ::step]
    fttc_y = -fttc_forces[1][::step, ::step]
    mask = ~np.isnan(fttc_x) & ~np.isnan(fttc_y)
    # q3 = ax3.quiver(x_quiver[mask], y_quiver[mask],
    #                 fttc_x[mask], fttc_y[mask])
    ax3.set_title('FTTC Class Forces')
    plt.colorbar(im3, ax=ax3, label='Force magnitude (Pa)')

    # Add quiver scale
    # qk = plt.quiverkey(q1, 0.9, 0.9, 100, '100 Pa',
    #                    labelpos='E', coordinates='figure')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    def calculate_psd_error(field1: np.ndarray, field2: np.ndarray) -> Tuple[float, float]:
        """Calculate high and low frequency errors between two fields using PSD"""
        # Compute 2D FFT
        fft1 = np.fft.fft2(np.nan_to_num(field1))
        fft2 = np.fft.fft2(np.nan_to_num(field2))

        # Compute power spectrum
        psd1 = np.abs(fft1) ** 2
        psd2 = np.abs(fft2) ** 2

        # Separate high and low frequencies
        h, w = psd1.shape
        center_h, center_w = h // 2, w // 2
        radius = min(center_h, center_w)

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

        # Consider frequencies up to 1/3 of Nyquist as "low"
        low_freq_mask = dist_from_center <= radius / 3
        high_freq_mask = dist_from_center > radius / 3

        # Calculate errors in different frequency bands
        low_freq_error = np.sqrt(np.mean((psd1[low_freq_mask] - psd2[low_freq_mask]) ** 2))
        high_freq_error = np.sqrt(np.mean((psd1[high_freq_mask] - psd2[high_freq_mask]) ** 2))

        return low_freq_error, high_freq_error

        # Calculate PSD errors for both components
        low_freq_x, high_freq_x = calculate_psd_error(calculated_forces[0], true_forces[0])
        low_freq_y, high_freq_y = calculate_psd_error(calculated_forces[1], true_forces[1])

        metrics['low_frequency_error'] = (low_freq_x + low_freq_y) / 2
        metrics['high_frequency_error'] = (high_freq_x + high_freq_y) / 2
        metrics['high_to_low_freq_ratio'] = metrics['high_frequency_error'] / (metrics['low_frequency_error'] + 1e-10)

        # === Peak analysis ===
        # Find peaks and analyze their surroundings
        peak_threshold = np.nanpercentile(true_mag, 90)  # Top 10% of forces
        peak_mask = true_mag > peak_threshold

        if np.any(peak_mask):
            # Error specifically around peaks
            metrics['peak_region_rmse'] = np.sqrt(np.nanmean(
                (calculated_forces[:, peak_mask] - true_forces[:, peak_mask]) ** 2
            ))

            # Analyze ringing artifacts around peaks
            kernel_size = 5
            smoothed_calc = gaussian_filter(calc_mag, kernel_size / 2)
            smoothed_true = gaussian_filter(true_mag, kernel_size / 2)

            oscillation_calc = np.abs(calc_mag - smoothed_calc)
            oscillation_true = np.abs(true_mag - smoothed_true)

            metrics['peak_ringing_error'] = np.sqrt(np.nanmean(
                (oscillation_calc[peak_mask] - oscillation_true[peak_mask]) ** 2
            ))

        return metrics


def load_and_upscale_data(data_dir: Path, scale_factor: int = 1, dilation_width: int = 2):
    """Load and upscale synthetic data by the specified factor."""
    # Load original data
    dx = np.load(data_dir / "displacement_x.npy")
    dy = np.load(data_dir / "displacement_y.npy")
    fx = np.load(data_dir / "traction_x.npy")
    fy = np.load(data_dir / "traction_y.npy")

    # Create masks for valid (non-NaN) regions
    dx_mask = ~np.isnan(dx)
    dy_mask = ~np.isnan(dy)
    fx_mask = ~np.isnan(fx)
    fy_mask = ~np.isnan(fy)

    # Fill NaNs temporarily for interpolation
    dx = np.nan_to_num(dx, 0)
    dy = np.nan_to_num(dy, 0)
    fx = np.nan_to_num(fx, 0)
    fy = np.nan_to_num(fy, 0)

    # Upscale the data
    dx_up = zoom(dx, scale_factor, order=1)
    dy_up = zoom(dy, scale_factor, order=1)
    fx_up = zoom(fx, scale_factor, order=1)
    fy_up = zoom(fy, scale_factor, order=1)

    # Upscale the masks
    dx_mask_up = zoom(dx_mask.astype(float), scale_factor, order=0).astype(bool)
    dy_mask_up = zoom(dy_mask.astype(float), scale_factor, order=0).astype(bool)
    fx_mask_up = zoom(fx_mask.astype(float), scale_factor, order=0).astype(bool)
    fy_mask_up = zoom(fy_mask.astype(float), scale_factor, order=0).astype(bool)

    # Reapply masks (set invalid regions to NaN)
    dx_up[~dx_mask_up] = np.nan
    dy_up[~dy_mask_up] = np.nan
    fx_up[~fx_mask_up] = np.nan
    fy_up[~fy_mask_up] = np.nan

    # Dilate non-NaN regions if requested
    if dilation_width > 0:
        dx_up = dilate_non_nan_region(dx_up, dilation_width)
        dy_up = dilate_non_nan_region(dy_up, dilation_width)
        fx_up = dilate_non_nan_region(fx_up, dilation_width)
        fy_up = dilate_non_nan_region(fy_up, dilation_width)

    # Combine into arrays with correct shapes
    displacements = np.stack([dx_up, dy_up], axis=-1)  # H x W x 2
    true_forces = np.stack([fx_up, fy_up], axis=0)  # 2 x H x W

    return displacements, true_forces


def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent.parent.absolute()
    data_dir = base_dir / "_validation" / "benchmarks" / "synthetic_cell"

    print(f"Looking for synthetic data in: {data_dir}")
    print(f"Project root: {base_dir}")

    try:
        # Load and upscale synthetic data
        print("\nLoading and upscaling data by factor of 8...")
        displacements, true_forces = load_and_upscale_data(data_dir, scale_factor=4, dilation_width=2)
        print(f"Upscaled displacement shape: {displacements.shape}")

        # Set up parameters for both methods
        E = 10000  # Young's modulus (10 kPa)
        nu = 0.5  # Poisson ratio
        pixel_size = 0.3 / 4
        alpha = 1e-20  # Regularization parameter

        # Calculate forces using simple method
        print("\nCalculating traction forces using simple method...")
        import time
        simple_start_time = time.time()
        simple_fx, simple_fy = calculate_traction_stresses_simple(
            displacements[..., 0],
            displacements[..., 1],
            E, nu, pixel_size, alpha
        )
        simple_end_time = time.time()
        simple_execution_time = simple_end_time - simple_start_time

        simple_forces = np.stack([simple_fx, simple_fy], axis=0)
        simple_forces = mask_calculated_forces(simple_forces, true_forces)

        # Set up FTTC parameters
        params = FTTCParameters(
            young_modulus=E,
            poisson_ratio_substrate=nu,
            lanczos_exp=0,
            gel_height=None,
            pixel_size=pixel_size,
            auto_gcv=False,  # Explicitly disable auto GCV
            regularization=alpha,
            downscale_factor=1
        )

        # Create FTTC calculator and calculate forces
        print("\nCalculating traction forces using FTTC class...")
        calculator = FTTC(params)

        fttc_start_time = time.time()
        (x, y), fttc_forces = calculator.calculate_traction(
            displacements=displacements,
            pixel_size=params.pixel_size,
            downscale_factor=params.downscale_factor,
            regularization=params.regularization  # Explicitly pass regularization
        )
        fttc_end_time = time.time()
        fttc_execution_time = fttc_end_time - fttc_start_time

        fttc_forces = mask_calculated_forces(fttc_forces, true_forces)

        # Calculate and print metrics
        print("\nComputing validation metrics...")
        simple_metrics = calculate_error_metrics(simple_forces, true_forces)
        fttc_metrics = calculate_error_metrics(fttc_forces, true_forces)

        print("\nSimple FTTC Results:")
        print("-" * 50)
        print(f"Execution time: {simple_execution_time:.3f} seconds")
        print(f"RMSE: {simple_metrics['rmse']:.2f} Pa")
        print(f"Maximum Error: {simple_metrics['max_error']:.2f} Pa")
        print(f"Relative L2 Error: {simple_metrics['relative_l2_error']:.3f}")
        print(f"Correlation Coefficient: {simple_metrics['correlation']:.3f}")

        print("\nFTTC Class Results:")
        print("-" * 50)
        print(f"Execution time: {fttc_execution_time:.3f} seconds")
        print(f"RMSE: {fttc_metrics['rmse']:.2f} Pa")
        print(f"Maximum Error: {fttc_metrics['max_error']:.2f} Pa")
        print(f"Relative L2 Error: {fttc_metrics['relative_l2_error']:.3f}")
        print(f"Correlation Coefficient: {fttc_metrics['correlation']:.3f}")

        # After calculating forces
        advanced_simple_metrics = calculate_advanced_metrics(
            simple_forces, true_forces)
        advanced_fttc_metrics = calculate_advanced_metrics(
            fttc_forces, true_forces)

        # Print detailed comparison
        print_metric_comparison(advanced_simple_metrics, advanced_fttc_metrics)

        print("\nSpeed Comparison:")
        print("-" * 50)
        speed_ratio = fttc_execution_time / simple_execution_time
        print(f"FTTC Class is {speed_ratio:.1f}x slower than Simple FTTC")

        # Plot comparison
        print("\nGenerating validation plots...")
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        plot_three_way_comparison(
            (x, y), true_forces, simple_forces, fttc_forces,
            save_path=output_dir / "fttc_validation_comparison_upscaled.png"
        )

        print(f"\nValidation complete. Plots saved to: {output_dir}")

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

