import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import sys
from scipy import ndimage

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.fttc import FTTC
from backend.parameter_dataclasses import FTTCParameters
from scipy.fft import fft2, ifft2
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


def load_synthetic_data(data_dir: Path, dilation_width: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic cell data for validation.

    Args:
        data_dir: Path to directory containing synthetic data
        dilation_width: Width of the border to add around non-NaN regions

    Returns:
        Tuple containing:
        - displacements: (H x W x 2) array
        - true_forces: (2 x H x W) array
    """
    try:
        # Load displacement field
        dx = np.load(data_dir / "displacement_x.npy")
        dy = np.load(data_dir / "displacement_y.npy")

        # Load ground truth traction forces
        fx = np.load(data_dir / "traction_x.npy")
        fy = np.load(data_dir / "traction_y.npy")

        # Dilate the non-NaN regions in force components
        fx = dilate_non_nan_region(fx, dilation_width)
        fy = dilate_non_nan_region(fy, dilation_width)

        # Apply the same dilation to displacement components
        dx = dilate_non_nan_region(dx, dilation_width)
        dy = dilate_non_nan_region(dy, dilation_width)

        # Combine into arrays with correct shapes
        displacements = np.stack([dx, dy], axis=-1)  # H x W x 2
        true_forces = np.stack([fx, fy], axis=0)  # 2 x H x W

        return displacements, true_forces

    except FileNotFoundError as e:
        print(f"Error loading synthetic data: {e}")
        print(f"Looked in directory: {data_dir}")
        print("Please ensure all required .npy files are present:")
        print("- displacement_x.npy")
        print("- displacement_y.npy")
        print("- traction_x.npy")
        print("- traction_y.npy")
        raise


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
    q1 = ax1.quiver(x_quiver[mask], y_quiver[mask],
                    true_x[mask], true_y[mask])
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
    q2 = ax2.quiver(x_quiver[mask], y_quiver[mask],
                    simple_x[mask], simple_y[mask])
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
    q3 = ax3.quiver(x_quiver[mask], y_quiver[mask],
                    fttc_x[mask], fttc_y[mask])
    ax3.set_title('FTTC Class Forces')
    plt.colorbar(im3, ax=ax3, label='Force magnitude (Pa)')

    # Add quiver scale
    qk = plt.quiverkey(q1, 0.9, 0.9, 100, '100 Pa',
                       labelpos='E', coordinates='figure')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent.parent.absolute()
    data_dir = base_dir / "_validation" / "benchmarks" / "synthetic_cell"

    print(f"Looking for synthetic data in: {data_dir}")
    print(f"Project root: {base_dir}")

    try:
        # Load synthetic data with dilation
        displacements, true_forces = load_synthetic_data(data_dir, dilation_width=2)

        # Set up parameters for both methods
        E = 10000  # Young's modulus (10 kPa)
        nu = 0.5  # Poisson ratio
        pixel_size = 0.3  # 0.1 μm/pixel
        alpha = 1e-15  # Regularization parameter

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
            auto_gcv=False,
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
            regularization=None if params.auto_gcv else params.regularization
        )
        fttc_end_time = time.time()
        fttc_execution_time = fttc_end_time - fttc_start_time

        fttc_forces = mask_calculated_forces(fttc_forces, true_forces)

        # Calculate error metrics for both methods
        print("\nComputing validation metrics...")
        simple_metrics = calculate_error_metrics(simple_forces, true_forces)
        fttc_metrics = calculate_error_metrics(fttc_forces, true_forces)

        # Print results
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

        # Print speed comparison
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
            save_path=output_dir / "fttc_validation_comparison.png"
        )

        print(f"\nValidation complete. Plots saved to: {output_dir}")

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
