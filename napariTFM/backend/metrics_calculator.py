import numpy as np


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

    total_se = np.sum(strain_energy_density_frame_jm2 * mask_frame * pixel_area_m2)
    return float(total_se)


def calculate_moment_tensor(force_frame_pa: np.ndarray,
                            mask_frame: np.ndarray,
                            pixel_positions_m: np.ndarray,
                            pixel_area_m2: float) -> np.ndarray:
    """
    Calculate the (symmetric) contractile moment tensor.

    M_ij = 1/2 * integral( x_i * f_j + x_j * f_i ) dA

    The raw first moment of the traction field, integral(x_i * f_j) dA, is not
    symmetric: its antisymmetric part is the net TORQUE the cell exerts on the
    substrate. A cell in mechanical equilibrium exerts none, so that part is
    measurement error (finite grid, regularization, truncation of the traction
    field at the mask edge) and is dropped here rather than fed to an eigenvalue
    solver. This is the standard contractile moment tensor of Butler et al.
    (Am J Physiol Cell Physiol 282:C595, 2002).

    Dropping it is not cosmetic. The eigenvalues of a non-symmetric 2x2 matrix
    are complex whenever the torque is large enough relative to the anisotropy,
    at which point "the principal axes of contraction" are not defined at all and
    downstream code comparing them silently compares complex numbers.

    Args:
        force_frame_pa: Traction force field (y, x, 2) in Pa (N/m^2).
        mask_frame: 2D binary array (y, x). Weights the integration domain; pass
                    an array of ones to integrate over the whole field.
        pixel_positions_m: Pixel positions (y, x, 2) relative to an origin, in meters.
                           Order is (x_pos, y_pos) for each pixel.
        pixel_area_m2: Area of a single pixel in m^2.

    Returns:
        2x2 symmetric NumPy array representing the moment tensor in N.m.
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

    # Symmetrize: 0.5 * (M + M.T). Diagonal untouched, off-diagonals averaged.
    moment_tensor = 0.5 * (moment_tensor + moment_tensor.T)

    return moment_tensor * pixel_area_m2


def moment_tensor_torque(force_frame_pa: np.ndarray,
                         mask_frame: np.ndarray,
                         pixel_positions_m: np.ndarray,
                         pixel_area_m2: float) -> float:
    """Net torque about the position origin, in N.m — the part symmetrization drops.

    Returned separately so that "how much did symmetrization change" is
    measurable rather than assumed. For a cell in equilibrium this should be
    small next to the trace of the symmetric tensor; a large value means the
    traction field is not force-balanced over the chosen integration domain
    (typically because the domain cuts through it).
    """
    fx = force_frame_pa[:, :, 0] * mask_frame
    fy = force_frame_pa[:, :, 1] * mask_frame
    x = pixel_positions_m[:, :, 0] * mask_frame
    y = pixel_positions_m[:, :, 1] * mask_frame
    return float((np.sum(x * fy) - np.sum(y * fx)) * pixel_area_m2)


def calculate_polarization(moment_tensor: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate polarization index and eigenvalues from the moment tensor.

    Uses ``eigvalsh``, the symmetric solver: the contractile moment tensor is
    symmetric by construction (see :func:`calculate_moment_tensor`), so its
    eigenvalues are real and its principal axes orthogonal. ``eigvals`` would
    return a complex pair for any residual asymmetry, and ``np.max`` on complex
    values compares them lexicographically — a silently meaningless answer.

    Args:
        moment_tensor: 2x2 symmetric moment tensor (N.m).

    Returns:
        Tuple (polarization_index, lambda1, lambda2). Eigenvalues in N.m.
    """
    if not np.allclose(moment_tensor, moment_tensor.T, rtol=1e-6, atol=0.0):
        raise ValueError(
            "moment_tensor is not symmetric; use calculate_moment_tensor, which "
            "symmetrizes, rather than a raw first moment of the traction field.")
    eigenvalues = np.linalg.eigvalsh(moment_tensor)  # real, ascending
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