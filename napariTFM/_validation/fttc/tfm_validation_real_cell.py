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

    # Slice the data to get only the spatial dimensions (x, y)
    # Assuming the first dimension is time, we'll take the first timepoint
    d_x = d_x[0]
    d_y = d_y[0]
    t_x = t_x[0]
    t_y = t_y[0]

    return d_x, d_y, t_x, t_y


def calculate_magnitude(x, y):
    """Calculate magnitude from x and y components"""
    return np.sqrt(x ** 2 + y ** 2)


def calculate_traction(d_x, d_y, pixelsize, E, nu, lanczos_exp, gel_height=None, regularization=None):
    """Calculate traction forces using FTTC"""
    E_scaled = E

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
    f_pascal = f

    return f_pascal[0].reshape(d_x.shape), f_pascal[1].reshape(d_y.shape)


def plot_results(d_mag, t_mag_ref, t_mag_calc):
    """Plot displacement and traction magnitudes"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot displacement magnitude
    im1 = ax1.imshow(d_mag)
    ax1.set_title('Displacement Magnitude\nin microns')
    plt.colorbar(im1, ax=ax1)

    # Plot traction magnitudes
    im2 = ax2.imshow(t_mag_ref)
    ax2.set_title('Reference Traction Magnitude [Pa]')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(t_mag_calc)
    ax3.set_title('Calculated Traction Magnitude [Pa]')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    return fig


def main():
    # Parameters
    E = 6000  # Young's modulus in Pascal
    nu = 0.45  # Poisson ratio
    pixelsize = 0.1e-6  # 0.1 µm/pixel
    lanczos_exp = 1
    gel_height = None
    regularization = None

    # Set up paths
    current_dir = Path(__file__).parent
    benchmark_path = current_dir.parent / "force_quadrupole/real_cell"

    # Load dataset (displacements in microns)
    d_x, d_y, t_x, t_y = load_data(benchmark_path)

    # Calculate displacement magnitude in microns
    d_mag = calculate_magnitude(d_x, d_y)

    # Convert displacements to pixels
    d_x_pix = d_x / 0.4  # Convert to pixels
    d_y_pix = d_y / 0.4

    try:
        # Calculate traction forces
        calc_t_x, calc_t_y = calculate_traction(
            d_x_pix, d_y_pix,  # Input in pixels
            pixelsize=pixelsize,
            E=E, nu=nu,
            lanczos_exp=lanczos_exp,
            gel_height=gel_height,
            regularization=regularization
        )

        # Calculate traction magnitudes
        t_mag_ref = calculate_magnitude(t_x, t_y)
        t_mag_calc = calculate_magnitude(calc_t_x, calc_t_y)

        # Plot results
        fig = plot_results(d_mag, t_mag_ref, t_mag_calc)
        fig.savefig('traction_analysis.png')

        plt.show()

    except Exception as e:
        print(f"Error during calculation: {str(e)}")


if __name__ == "__main__":
    main()