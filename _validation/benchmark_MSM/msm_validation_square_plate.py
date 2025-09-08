import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues

from napariTFM.services.msm_service import MSMService
from napariTFM.backend.parameter_dataclasses import MSMParameters


def create_square_plate_problem(size=100, edge_traction=1000, buffer=5):
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
    tx[buffer:-buffer, buffer:buffer + 1] = edge_traction  # Left edge
    tx[buffer:-buffer, -(buffer + 1):-buffer] = -edge_traction  # Right edge

    # For y-direction: equal and opposite forces on top and bottom edges
    ty[buffer:buffer + 1, buffer:-buffer] = edge_traction  # Top edge
    ty[-(buffer + 1):-buffer, buffer:-buffer] = -edge_traction  # Bottom edge

    # Store original traction magnitudes for scaling
    tx_max = np.max(np.abs(tx))
    ty_max = np.max(np.abs(ty))
    traction_scale = max(tx_max, ty_max)

    return tx, ty, mask, traction_scale


def calculate_stress_field_with_msm_service(service, tx, ty, mask):
    """Calculate stress field using MSM service"""
    # Prepare force field in the format expected by MSMService (H, W, 2)
    force_field = np.stack([tx, ty], axis=-1)
    
    # Prepare masks (add time dimension as service expects 3D: T, H, W)
    masks = mask[np.newaxis, ...]  # Shape: (1, H, W)
    
    # Calculate stresses using the service - this returns a generator
    stress_generator = service.calculate_stresses(force_field, masks)
    
    # Get the final result from the generator
    try:
        # Process through the generator to get final result
        result = None
        for intermediate_result, frame, total_frames in stress_generator:
            result = intermediate_result
    except StopIteration as e:
        # The final result is returned via StopIteration.value
        result = e.value
    
    if result is None:
        raise ValueError("MSM calculation did not return a valid result")
    
    # Extract stress tensor from MSMResult
    # result.stress_tensor has shape (1, H, W, 2, 2) for single frame
    stress_tensor = result.stress_tensor[0]  # Remove time dimension: (H, W, 2, 2)
    
    return stress_tensor


def validate_msm_square_plate(size=100, edge_traction=1000, pixelsize=1e-6, buffer=5):
    """
    Validate MSM implementation using square plate problem
    """
    # Create the test problem
    tx, ty, mask, traction_scale = create_square_plate_problem(
        size=size,
        edge_traction=edge_traction,
        buffer=buffer
    )

    # Initialize MSM service with parameters
    params = MSMParameters(
        # Mesh parameters
        density_factor=0.005,
        mesh_algorithm='Frontal-Del.',
        use_optimization=False,
        
        # Material parameters  
        poisson_ratio_cells=0.5,
        young_modulus=1000.0,  # Pa
        
        # Scaling parameters
        pixel_size=pixelsize * 1e6,  # Convert to microns
        downscale_factor=1
    )
    
    service = MSMService(params)

    # Calculate stress field using MSM service
    calculated_stress = calculate_stress_field_with_msm_service(service, tx, ty, mask)

    # Create analytical solution in mN/m (to match MSM output units)
    # MSM outputs: stress [mN/m] = stress [Pa] × downscale_factor × pixel_size [µm] × 1e-3
    analytical_stress_mNm = traction_scale * params.downscale_factor * params.pixel_size * 1e-3
    
    analytical_stress = np.zeros_like(calculated_stress)
    analytical_stress[..., 0, 0] = analytical_stress_mNm  # σxx
    analytical_stress[..., 1, 1] = analytical_stress_mNm  # σyy
    analytical_stress[~mask] = 0  # Apply mask

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle('Square Plate Validation Results', fontsize=14)

    # Function to plot stress fields with consistent formatting
    def plot_stress_field(ax, data, title, scale_factor=1.0):
        vmax = 1.5
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
    plt.savefig("square.png", dpi=300, bbox_inches='tight')

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
            print(f"Mean value: {np.mean(calc[valid_mask]):.2e} mN/m")
            print(f"Expected value: {np.mean(true[valid_mask]):.2e} mN/m")

        else:  # Shear component
            # For shear, compare absolute values since it should be zero
            rms_shear = np.sqrt(np.mean(calc[valid_mask] ** 2))
            max_shear = np.max(np.abs(calc[valid_mask]))
            print(f"\nσ{name}:")
            print(f"RMS shear: {rms_shear:.2e} mN/m")
            print(f"Max shear: {max_shear:.2e} mN/m")

    return calculated_stress, analytical_stress, fig


if __name__ == "__main__":
    # Run validation with default parameters
    size = 50  # pixels
    edge_traction = 1000  # Pa
    pixelsize = 1e-6  # meters (1 µm)
    buffer = 5  # pixels

    calculated_stress, analytical_stress, fig = validate_msm_square_plate(
        size=size,
        edge_traction=edge_traction,
        pixelsize=pixelsize,
        buffer=buffer
    )

    plt.show()