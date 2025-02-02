import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from napariTFM.fttc import FTTC


def load_data(benchmark_path):
    """Load displacement and traction data from numpy files"""
    d_x = np.load(benchmark_path / "d_x.npy")  # in microns
    d_y = np.load(benchmark_path / "d_y.npy")  # in microns
    t_x = np.load(benchmark_path / "t_x.npy")
    t_y = np.load(benchmark_path / "t_y.npy")
    return d_x, d_y, t_x, t_y


def calculate_magnitude(x, y):
    """Calculate magnitude from x and y components"""
    return np.sqrt(x ** 2 + y ** 2)


def calculate_traction(d_x, d_y, pixelsize, E, nu, lanczos_exp, gel_height=None, regularization=None):
    """Calculate traction forces using FTTC"""
    E_scaled = E * (pixelsize ** 2)

    if gel_height is not None:
        gel_height = gel_height / pixelsize

    fttc = FTTC(E=E_scaled, nu=nu, lanczos_exp=lanczos_exp, gel_height=gel_height)

    ny, nx = d_x.shape
    x = np.arange(nx)
    y = np.arange(ny)
    xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
    pos0 = np.array([xgrid.flatten(), ygrid.flatten()])
    vec0 = np.array([d_x.flatten(), d_y.flatten()])

    if regularization is None:
        regularization = fttc._find_regularization(pos0, vec0)

    (_, _), f = fttc._perform_tfm(pos0, vec0, regularization)
    f_pascal = f / (pixelsize ** 2)

    return f_pascal[0].reshape(d_x.shape), f_pascal[1].reshape(d_y.shape)


def plot_displacements(d_mag_01_pix, d_mag_10_pix, d_mag_01_micron, d_mag_10_micron):
    """Plot displacement magnitudes in pixels and microns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # Plot displacements in pixels
    im1 = ax1.imshow(d_mag_01_pix)
    ax1.set_title('Displacement Magnitude (0.1 µm/pix)\nin pixels')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(d_mag_10_pix)
    ax2.set_title('Displacement Magnitude (1.0 µm/pix)\nin pixels')
    plt.colorbar(im2, ax=ax2)

    # Plot displacements in microns
    im3 = ax3.imshow(d_mag_01_micron)
    ax3.set_title('Displacement Magnitude (0.1 µm/pix)\nin microns')
    plt.colorbar(im3, ax=ax3)

    im4 = ax4.imshow(d_mag_10_micron)
    ax4.set_title('Displacement Magnitude (1.0 µm/pix)\nin microns')
    plt.colorbar(im4, ax=ax4)

    plt.tight_layout()
    return fig


def plot_tractions(t_mag_ref_01, t_mag_calc_01, t_mag_ref_10, t_mag_calc_10):
    """Plot reference and calculated traction magnitudes"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # Plot 0.1 µm/pixel results
    im1 = ax1.imshow(t_mag_ref_01)
    ax1.set_title('Reference Traction Magnitude\n(0.1 µm/pix) [Pa]')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(t_mag_calc_01)
    ax2.set_title('Calculated Traction Magnitude\n(0.1 µm/pix) [Pa]')
    plt.colorbar(im2, ax=ax2)

    # Plot 1.0 µm/pixel results
    im3 = ax3.imshow(t_mag_ref_10)
    ax3.set_title('Reference Traction Magnitude\n(1.0 µm/pix) [Pa]')
    plt.colorbar(im3, ax=ax3)

    im4 = ax4.imshow(t_mag_calc_10)
    ax4.set_title('Calculated Traction Magnitude\n(1.0 µm/pix) [Pa]')
    plt.colorbar(im4, ax=ax4)

    plt.tight_layout()
    return fig


def main():
    # Parameters
    E = 10000  # Young's modulus in Pascal
    nu = 0.45  # Poisson ratio
    lanczos_exp = 1
    gel_height = None
    regularization = None

    # Set up paths
    current_dir = Path(__file__).parent
    benchmark_path_01 = current_dir.parent / "force_quadrupole" / "pix0.1"
    benchmark_path_10 = current_dir.parent / "force_quadrupole" / "pix1.0"

    # Load both datasets (displacements in microns)
    d_x_01, d_y_01, t_x_01, t_y_01 = load_data(benchmark_path_01)
    d_x_10, d_y_10, t_x_10, t_y_10 = load_data(benchmark_path_10)

    # Calculate displacement magnitudes in microns
    d_mag_01_micron = calculate_magnitude(d_x_01, d_y_01)
    d_mag_10_micron = calculate_magnitude(d_x_10, d_y_10)

    # Convert displacements to pixels
    d_x_01_pix = d_x_01 / 0.1  # Convert 0.1 µm/pix data to pixels
    d_y_01_pix = d_y_01 / 0.1
    d_mag_01_pix = d_mag_01_micron / 0.1

    d_x_10_pix = d_x_10 / 1.0  # Convert 1.0 µm/pix data to pixels
    d_y_10_pix = d_y_10 / 1.0
    d_mag_10_pix = d_mag_10_micron / 1.0

    # Plot displacement magnitudes
    fig_displacements = plot_displacements(d_mag_01_pix, d_mag_10_pix,
                                           d_mag_01_micron, d_mag_10_micron)
    fig_displacements.savefig('displacement_magnitudes.png')

    try:
        # Calculate traction forces for 0.1 µm/pixel
        calc_t_x_01, calc_t_y_01 = calculate_traction(
            d_x_01_pix, d_y_01_pix,  # Input in pixels
            pixelsize=0.1e-6,
            E=E, nu=nu,
            lanczos_exp=lanczos_exp,
            gel_height=gel_height,
            regularization=regularization
        )

        # Calculate traction forces for 1.0 µm/pixel
        calc_t_x_10, calc_t_y_10 = calculate_traction(
            d_x_10_pix, d_y_10_pix,  # Input in pixels
            pixelsize=1.0e-6,
            E=E, nu=nu,
            lanczos_exp=lanczos_exp,
            gel_height=gel_height,
            regularization=regularization
        )

        # Calculate traction magnitudes
        t_mag_ref_01 = calculate_magnitude(t_x_01, t_y_01)
        t_mag_calc_01 = calculate_magnitude(calc_t_x_01, calc_t_y_01)
        t_mag_ref_10 = calculate_magnitude(t_x_10, t_y_10)
        t_mag_calc_10 = calculate_magnitude(calc_t_x_10, calc_t_y_10)

        # Plot traction magnitudes
        fig_tractions = plot_tractions(t_mag_ref_01, t_mag_calc_01,
                                       t_mag_ref_10, t_mag_calc_10)
        fig_tractions.savefig('traction_magnitudes.png')

        plt.show()

    except Exception as e:
        print(f"Error during calculation: {str(e)}")


if __name__ == "__main__":
    main()