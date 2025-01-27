"""
Traction Force Microscopy Analysis
Author: Artur Ruppel
Modified version with improved structure and visualization
"""

import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as colors

from napariTFM.fttc import FTTC


def calculate_traction_stresses(u, v, E, nu, pixelsize, alpha):
    '''
    Calculate traction stresses from displacement fields.

    Parameters:
    -----------
    u, v : ndarray
        Displacement fields in x and y directions
    E : float
        Young's modulus of the gel
    nu : float
        Poisson ratio of the gel
    pixelsize : float
        Size of each pixel in meters
    alpha : float
        Regularization parameter

    Returns:
    --------
    traction_x, traction_y : ndarray
        Calculated traction forces in x and y directions
    '''
    M, N = u.shape

    # pad displacement map with zeros until it has a shape of 2^n by 2^n
    n = 2
    while (2 ** n < M) or (2 ** n < N):
        n = n + 1

    M2 = N2 = 2 ** n

    u_padded = np.zeros((M2, N2))
    v_padded = np.zeros((M2, N2))

    u_padded[:u.shape[0], :u.shape[1]] = u
    v_padded[:v.shape[0], :v.shape[1]] = v

    u_fft = fft2(u_padded)
    v_fft = fft2(v_padded)

    # remove component related to translation
    u_fft[0, 0] = v_fft[0, 0] = 0

    # Calculate wave vectors
    Kx = np.concatenate([
        (2 * np.pi / pixelsize) / N2 * np.arange(int(N2 / 2)),
        -(2 * np.pi / pixelsize) / N2 * (N2 - np.arange(int(N2 / 2), N2))
    ])

    Ky = np.concatenate([
        (2 * np.pi / pixelsize) / M2 * np.arange(int(M2 / 2)),
        -(2 * np.pi / pixelsize) / M2 * (M2 - np.arange(int(M2 / 2), M2))
    ])

    kx, ky = np.meshgrid(Kx, Ky)
    k = np.sqrt(kx ** 2 + ky ** 2)
    t_xt = np.zeros((M2, N2), dtype=complex)
    t_yt = np.zeros((M2, N2), dtype=complex)

    for i in range(M2):
        for j in range(N2):
            if i == M2 / 2 or j == N2 / 2:  # Nyquist frequency
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)
            elif not ((i == 1) and (j == 1)):
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)
                Gt[0, 1] = Gt[1, 0] = - nu * kx[i, j] * ky[i, j]

            if (i == M2 / 2 or j == N2 / 2) or not ((i == 1) and (j == 1)):
                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j], t_yt[i, j] = Tt

    t_x = ifft2(t_xt)
    t_y = ifft2(t_yt)

    return np.real(t_x[0:M, 0:N]), np.real(t_y[0:M, 0:N])

def process_traction_forces(d_x, d_y, pixelsize, downsamplerate=8, E=19960, nu=0.5, alpha=1e-19):
    """
    Process displacement fields to calculate traction forces.

    Parameters:
    -----------
    d_x, d_y : ndarray
        Displacement fields in x and y directions
    pixelsize : float
        Size of each pixel in meters
    downsamplerate : int
        Factor by which to downsample the displacement field
    E : float
        Young's modulus of the gel
    nu : float
        Poisson ratio of the gel
    alpha : float
        Regularization parameter

    Returns:
    --------
    t_x, t_y : ndarray
        Calculated traction forces in x and y directions
    """
    pixelsize *= 1e-6  # Convert to meters
    forcemap_pixelsize = pixelsize * downsamplerate

    if len(d_x.shape) == 2:
        d_x = d_x[np.newaxis, ...]
        d_y = d_y[np.newaxis, ...]

    no_frames = d_x.shape[0]
    t_x = np.zeros(d_x.shape)
    t_y = np.zeros(d_y.shape)

    for frame in range(no_frames):
        print(f"Calculating traction forces for frame: {frame}")
        t_x[frame], t_y[frame] = calculate_traction_stresses(
            d_x[frame] * pixelsize,
            d_y[frame] * pixelsize,
            E, nu, forcemap_pixelsize, alpha
        )

    return t_x, t_y


def plot_displacement_and_traction(d_x, d_y, t_x, t_y, t_x_ref=None, t_y_ref=None):
    """
    Plot displacement fields and traction forces.

    Parameters:
    -----------
    d_x, d_y : ndarray
        Displacement fields
    t_x, t_y : ndarray
        Calculated traction forces
    t_x_ref, t_y_ref : ndarray, optional
        Reference traction forces for comparison
    """
    fig, axes = plt.subplots(2, 2 if t_x_ref is not None else 1, figsize=(15, 15))
    axes = axes.ravel()

    # Plot displacement field
    displacement_magnitude = np.sqrt(d_x ** 2 + d_y ** 2)
    im1 = axes[0].imshow(displacement_magnitude, cmap='viridis')
    axes[0].set_title('Displacement Magnitude')
    plt.colorbar(im1, ax=axes[0], label='Displacement (μm)')

    # Calculate traction magnitudes
    traction_magnitude = np.sqrt(t_x ** 2 + t_y ** 2)

    # If reference data is provided, find global min and max for consistent scaling
    if t_x_ref is not None:
        traction_ref_magnitude = np.sqrt(t_x_ref ** 2 + t_y_ref ** 2)
        vmin = min(np.min(traction_magnitude), np.min(traction_ref_magnitude))
        vmax = max(np.max(traction_magnitude), np.max(traction_ref_magnitude))
    else:
        vmin = np.min(traction_magnitude)
        vmax = np.max(traction_magnitude)

    # Plot calculated traction forces
    im2 = axes[1].imshow(traction_magnitude, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Calculated Traction Forces')
    plt.colorbar(im2, ax=axes[1], label='Force (N/m²)')

    if t_x_ref is not None:
        # Plot reference traction forces with same scale
        im3 = axes[2].imshow(traction_ref_magnitude, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title('Reference Traction Forces')
        plt.colorbar(im3, ax=axes[2], label='Force (N/m²)')

        # Plot difference
        diff_magnitude = traction_magnitude - traction_ref_magnitude
        # Use symmetric scaling for difference plot
        max_diff = max(abs(np.min(diff_magnitude)), abs(np.max(diff_magnitude)))
        im4 = axes[3].imshow(diff_magnitude, cmap='RdBu', vmin=-max_diff, vmax=max_diff)
        axes[3].set_title('Difference (Calculated - Reference)')
        plt.colorbar(im4, ax=axes[3], label='Force Difference (N/m²)')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Set paths and parameters
    current_dir = Path(__file__).parent

    # Load displacement fields and reference traction forces
    d_x = np.load(current_dir / "d_x.npy")
    d_y = np.load(current_dir / "d_y.npy")
    t_x_ref = np.load(current_dir / "t_x.npy")
    t_y_ref = np.load(current_dir / "t_y.npy")

    # Common parameters
    pixelsize = 0.1  # micrometers
    downsamplerate = 4
    E = 10000  # Young's modulus in Pa
    nu = 0.5  # Poisson ratio

    # 1. Calculate forces using the original method
    t_x_orig, t_y_orig = process_traction_forces(
        d_x, d_y,
        pixelsize=pixelsize,
        downsamplerate=downsamplerate,
        E=E,
        nu=nu
    )
    t_x_orig = np.squeeze(t_x_orig)
    t_y_orig = np.squeeze(t_y_orig)

    # 2. Calculate forces using the FTTC method
    fttc = FTTC(
        E=E,  # Young's modulus in Pa
        nu=nu  # Poisson ratio
    )

    # Calculate forces with clear parameters
    (x, y), forces_fttc = fttc.calculate_traction(displacements=(d_x * pixelsize, d_y * pixelsize), pixel_size=pixelsize, downsample_factor=downsamplerate, regularization=0.00000001)

    # Reshape forces to match original dimensions
    t_x_fttc = forces_fttc[0].reshape(d_x.shape)
    t_y_fttc = forces_fttc[1].reshape(d_y.shape)

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate force magnitudes
    traction_ref = np.sqrt(t_x_ref ** 2 + t_y_ref ** 2)
    traction_orig = np.sqrt(t_x_orig ** 2 + t_y_orig ** 2)
    traction_fttc = np.sqrt(t_x_fttc ** 2 + t_y_fttc ** 2)

    # Find global scale for consistent visualization
    vmin = min(np.min(traction_ref), np.min(traction_orig), np.min(traction_fttc))
    vmax = max(np.max(traction_ref), np.max(traction_orig), np.max(traction_fttc))

    # Plot all results
    titles = ['Reference Forces', 'Original Method', 'FTTC Method']
    tractions = [traction_ref, traction_orig, traction_fttc]

    for ax, title, traction in zip(axes, titles, tractions):
        im = ax.imshow(traction, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')

    # Add colorbar
    # plt.colorbar(im, ax=axes.ravel().tolist(), label='Force (N/m²)')

    plt.tight_layout()
    plt.show()

    # Print error metrics for both methods
    error_orig = np.mean(np.abs(traction_orig))/np.mean(np.abs(traction_ref))
    error_fttc = np.mean(np.abs(traction_fttc))/np.mean(np.abs(traction_ref))

    print("\nOriginal Method Metrics:")
    print(f"Magnitude ratios: {error_orig}")

    print("\nFTTC Method Metrics:")
    print(f"Magnitude ratios: {error_fttc}")

    # Save results
    np.save(current_dir / "t_x_original.npy", t_x_orig)
    np.save(current_dir / "t_y_original.npy", t_y_orig)
    np.save(current_dir / "t_x_fttc.npy", t_x_fttc)
    np.save(current_dir / "t_y_fttc.npy", t_y_fttc)