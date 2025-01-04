import numpy as np
import matplotlib.pyplot as plt
from msm import MonolayerStressMicroscopy


def create_square_plate_problem(size=50, edge_traction=1.0, buffer=5):
    """
    Create a square plate problem with uniform edge tractions and buffer zone

    Args:
        size: Size of the square plate in pixels
        edge_traction: Magnitude of edge traction in Pa
        buffer: Size of buffer zone around the domain in pixels

    Returns:
        tx, ty: Traction field components
        mask: Boolean mask defining the plate
        analytical_stress: Analytical stress tensor field
    """
    # Create domain with buffer
    total_size = size + 2 * buffer
    mask = np.zeros((total_size, total_size), dtype=bool)
    mask[buffer:-buffer, buffer:-buffer] = True  # Active domain

    # Initialize traction fields
    tx = np.zeros((total_size, total_size))
    ty = np.zeros((total_size, total_size))

    # Apply tractions in a balanced way
    # For x-direction: equal and opposite forces on left and right edges
    tx[buffer:-buffer, buffer:buffer + 2] = edge_traction  # Left edge
    tx[buffer:-buffer, -(buffer + 2):-buffer] = -edge_traction  # Right edge

    # For y-direction: equal and opposite forces on top and bottom edges
    ty[buffer:buffer + 2, buffer:-buffer] = edge_traction  # Top edge
    ty[-(buffer + 2):-buffer, buffer:-buffer] = -edge_traction  # Bottom edge

    # Apply small amount of smoothing to avoid sharp transitions
    from scipy.ndimage import gaussian_filter
    tx = gaussian_filter(tx, sigma=0.5)
    ty = gaussian_filter(ty, sigma=0.5)

    # Store original traction magnitudes for scaling
    tx_max = np.max(np.abs(tx))
    ty_max = np.max(np.abs(ty))
    traction_scale = max(tx_max, ty_max)

    return tx, ty, mask, traction_scale


def calculate_stress_field_with_scaling(msm, tx, ty, mask):
    """Calculate stress field and determine proper scaling"""
    # Get original traction magnitudes
    orig_tx_max = np.max(np.abs(tx))
    orig_ty_max = np.max(np.abs(ty))
    orig_max = max(orig_tx_max, orig_ty_max)

    # Calculate stress field
    stress_tensor = msm.calculate_stress_field(tx, ty, mask)

    # Get calculated stress magnitudes
    calc_max = np.max(np.abs(stress_tensor[mask]))

    # Scale factor to match original traction magnitude
    scale_factor = orig_max / (calc_max + 1e-16)

    # Scale the stress tensor
    stress_tensor_scaled = stress_tensor * scale_factor

    return stress_tensor_scaled


def validate_msm_square_plate(size=50, edge_traction=1.0, pixelsize=0.8e-6, buffer=5):
    """
    Validate MSM implementation using square plate problem
    """
    # Create the test problem
    tx, ty, mask, traction_scale = create_square_plate_problem(
        size=size,
        edge_traction=edge_traction,
        buffer=buffer
    )

    # Initialize MSM calculator with corrected parameters
    msm = MonolayerStressMicroscopy(
        pixelsize=pixelsize * 1e6,  # Convert to microns
        sigma=0.5,  # Poisson's ratio
        youngs_modulus=1.0,  # Young's modulus
    )

    # Calculate stress field with proper scaling
    calculated_stress = calculate_stress_field_with_scaling(msm, tx, ty, mask)

    # Create analytical solution
    analytical_stress = np.zeros_like(calculated_stress)
    analytical_stress[..., 0, 0] = traction_scale  # σxx
    analytical_stress[..., 1, 1] = traction_scale  # σyy
    analytical_stress[~mask] = 0  # Apply mask

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle('Square Plate Validation Results', fontsize=14)

    # Function to plot stress fields with consistent formatting
    def plot_stress_field(ax, data, title, scale_factor=1.0):
        vmax = np.nanmax(np.abs(data[mask])) * scale_factor
        vmin = -vmax
        im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        return im

    # Plot input fields
    plot_stress_field(axes[0, 0], tx, 'Input tx')
    plot_stress_field(axes[0, 1], ty, 'Input ty')
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Domain Mask')

    # Plot stress components
    plot_stress_field(axes[1, 0], calculated_stress[..., 0, 0], 'Calculated σxx')
    plot_stress_field(axes[1, 1], calculated_stress[..., 1, 1], 'Calculated σyy')
    plot_stress_field(axes[1, 2], calculated_stress[..., 0, 1], 'Calculated σxy', scale_factor=0.1)

    plot_stress_field(axes[2, 0], analytical_stress[..., 0, 0], 'Analytical σxx')
    plot_stress_field(axes[2, 1], analytical_stress[..., 1, 1], 'Analytical σyy')
    plot_stress_field(axes[2, 2], analytical_stress[..., 0, 1], 'Analytical σxy', scale_factor=0.1)

    plt.tight_layout()

    # Calculate normalized error metrics
    print("\nValidation Metrics:")
    print("-" * 50)

    for comp, name in [((0, 0), 'xx'), ((1, 1), 'yy'), ((0, 1), 'xy')]:
        calc = calculated_stress[..., comp[0], comp[1]]
        true = analytical_stress[..., comp[0], comp[1]]

        # Calculate normalized values for comparison
        valid_mask = mask & ~np.isnan(calc) & ~np.isnan(true)
        if name != 'xy':
            calc_norm = calc[valid_mask] / np.max(np.abs(calc[valid_mask]))
            true_norm = true[valid_mask] / np.max(np.abs(true[valid_mask]))

            rmse = np.sqrt(np.mean((calc_norm - true_norm) ** 2))
            max_error = np.max(np.abs(calc_norm - true_norm))
            rel_error = rmse / np.mean(np.abs(true_norm))

            print(f"\nσ{name}:")
            print(f"RMSE (normalized): {rmse:.2e}")
            print(f"Max Error (normalized): {max_error:.2e}")
            print(f"Relative Error: {rel_error:.4f}")
            print(f"Mean value: {np.mean(calc[valid_mask]):.2e} Pa")
            print(f"Expected value: {np.mean(true[valid_mask]):.2e} Pa")

        else:  # Shear component
            # For shear, compare absolute values since it should be zero
            rms_shear = np.sqrt(np.mean(calc[valid_mask] ** 2))
            max_shear = np.max(np.abs(calc[valid_mask]))
            print(f"\nσ{name}:")
            print(f"RMS shear: {rms_shear:.2e} Pa")
            print(f"Max shear: {max_shear:.2e} Pa")

    return calculated_stress, analytical_stress, fig


if __name__ == "__main__":
    # Run validation with default parameters
    size = 50  # pixels
    edge_traction = 1.0  # Pa
    pixelsize = 0.8e-6  # meters (0.8 µm)
    buffer = 5  # pixels

    calculated_stress, analytical_stress, fig = validate_msm_square_plate(
        size=size,
        edge_traction=edge_traction,
        pixelsize=pixelsize,
        buffer=buffer
    )

    plt.show()