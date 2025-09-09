import numpy as np
from scipy.ndimage import center_of_mass


def calculate_strain_energy_density(displacement_frame_m: np.ndarray, force_frame_pa: np.ndarray) -> np.ndarray:
    """
    Calculate strain energy density.

    Args:
        displacement_frame_m: Displacement field (y, x, 2) in meters.
        force_frame_pa: Traction force field (y, x, 2) in Pascals (N/m^2).

    Returns:
        2D NumPy array (y, x) of strain energy density in J/m^2.
    """
    if displacement_frame_m.shape != force_frame_pa.shape:
        raise ValueError("Displacement and force frames must have the same shape.")
    if displacement_frame_m.ndim != 3 or displacement_frame_m.shape[-1] != 2:
        raise ValueError("Displacement frame must be of shape (y, x, 2).")

    # Strain Energy Density (SED) = 0.5 * (u_x * T_x + u_y * T_y)
    # u (m), T (N/m^2) -> u*T (N/m = J/m^2)
    sed = 0.5 * np.sum(displacement_frame_m * force_frame_pa, axis=-1)
    return sed


def calculate_total_strain_energy(strain_energy_density_frame_jm2: np.ndarray,
                                  mask_frame: np.ndarray,
                                  pixel_area_m2: float) -> float:
    """
    Calculate total strain energy over a masked area.

    Args:
        strain_energy_density_frame_jm2: 2D array of SED (J/m^2).
        mask_frame: 2D binary array (y, x) where True indicates the region of interest.
        pixel_area_m2: Area of a single pixel in square meters (m^2).

    Returns:
        Total strain energy in Joules (J).
    """
    if strain_energy_density_frame_jm2.shape != mask_frame.shape:
        raise ValueError("Strain energy density and mask frames must have the same shape.")
    if not np.isscalar(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be a positive scalar.")

    total_se = np.sum(strain_energy_density_frame_jm2[mask_frame] * pixel_area_m2)
    return float(total_se)


def calculate_moment_tensor(force_frame_pa: np.ndarray,
                            mask_frame: np.ndarray,
                            pixel_positions_m: np.ndarray,
                            pixel_area_m2: float) -> np.ndarray:
    """
    Calculate the moment tensor.

    Args:
        force_frame_pa: Traction force field (y, x, 2) in Pa (N/m^2).
        mask_frame: 2D binary array (y, x).
        pixel_positions_m: Pixel positions (y, x, 2) relative to an origin, in meters.
                           Order is (x_pos, y_pos) for each pixel.
        pixel_area_m2: Area of a single pixel in m^2.

    Returns:
        2x2 NumPy array representing the moment tensor in N.m.
    """
    if force_frame_pa.shape != pixel_positions_m.shape or force_frame_pa.shape[:2] != mask_frame.shape:
        raise ValueError("Force, pixel positions, and mask frames have incompatible shapes.")
    if not np.isscalar(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be a positive scalar.")

    moment_tensor = np.zeros((2, 2))


    # Consider only pixels within the mask
    masked_forces = np.zeros_like(force_frame_pa)
    masked_forces[:,:,0] = force_frame_pa[:,:,0] * mask_frame
    masked_forces[:,:,1] = force_frame_pa[:,:,1] * mask_frame

    masked_positions = np.zeros_like(pixel_positions_m)
    masked_positions[:,:,0] = pixel_positions_m[:,:,0] * mask_frame
    masked_positions[:,:,1] = pixel_positions_m[:,:,1] * mask_frame

    
    
    # Calculate each component
    m00_terms = masked_positions[:,:,0] * masked_forces[:,:,0]  # x * Fx
    m01_terms = masked_positions[:,:,0] * masked_forces[:,:,1]  # x * Fy
    m10_terms = masked_positions[:,:,1] * masked_forces[:,:,0]  # y * Fx
    m11_terms = masked_positions[:,:,1] * masked_forces[:,:,1]  # y * Fy
    

    moment_tensor[0, 0] = np.sum(m00_terms)
    moment_tensor[0, 1] = np.sum(m01_terms) 
    moment_tensor[1, 0] = np.sum(m10_terms)
    moment_tensor[1, 1] = np.sum(m11_terms)

    return moment_tensor * pixel_area_m2


def calculate_polarization(moment_tensor: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate polarization index and eigenvalues from the moment tensor.

    Args:
        moment_tensor: 2x2 moment tensor (N.m).

    Returns:
        Tuple (polarization_index, lambda1, lambda2). Eigenvalues in N.m.
    """
    eigenvalues = np.linalg.eigvals(moment_tensor)
    lambda1 = np.max(eigenvalues)  # Principal eigenvalue (more positive or less negative)
    lambda2 = np.min(eigenvalues)  # Secondary eigenvalue

    # Handle edge cases where sum is close to zero
    denominator = lambda1 + lambda2
    if np.abs(denominator) < 1e-15:  # Near zero sum
        polarization_index = 0.0
    else:
        polarization_index = np.abs((lambda1 - lambda2)) / np.abs(denominator)
    
    # Clamp to valid range [-1, 1] for safety
    polarization_index = np.clip(polarization_index, -1.0, 1.0)

    return float(polarization_index), float(lambda1), float(lambda2)