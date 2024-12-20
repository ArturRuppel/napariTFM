"""
Force calculation module for Traction Force Microscopy (TFM).
Includes validation for different correction methods based on experimental length scales.

This implementation is based on and significantly refactors code from the pyTFM package
(https://github.com/fabrylab/pyTFM), licensed under GNU General Public License v3.0.
The code has been restructured into a class-based implementation with additional
features including proper regularization, and validation based on experimental length scales.

Original paper references:
- Butler et al. Traction fields, moments, and strain energy that cells exert on their surroundings (2002)
- Sabass et al. High resolution traction force microscopy based on experimental and computational advances (2008)
- Trepat et al. Physical forces during collective cell migration (2009)

Code references:
- pyTFM by Fabry Lab (https://github.com/fabrylab/pyTFM)
"""

from enum import Enum
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from warnings import warn

logger = logging.getLogger(__name__)


class CorrectionMethod(Enum):
    """Enumeration of available thickness correction methods."""
    NONE = "none"  # Infinite thickness assumption
    FINITE = "finite"  # Full finite thickness correction
    PURE_SHEAR = "pure_shear"  # Pure shear approximation for thin substrates


@dataclass
class ValidationResult:
    """Contains validation results for correction method selection."""
    is_valid: bool
    warnings: list[str]
    errors: list[str]


class TractionForceCalculator:
    def __init__(
        self,
        young_modulus: float,
        pixel_size: float,
        characteristic_length: float,  # Added parameter
        poisson_ratio: float = 0.49,
        regularization: float = 1e-6,
        gel_height: Optional[float] = None,
        correction_method: CorrectionMethod = CorrectionMethod.NONE,
        filter_sigma: Optional[float] = None
    ):
        """
        Initialize the TractionForceCalculator.

        Parameters
        ----------
        young_modulus : float
            Young's modulus of the substrate (Pa)
        pixel_size : float
            Size of a pixel in physical units (meters)
        characteristic_length : float
            Characteristic length scale of the experiment (meters)
            This could be cell size, typical displacement distance, etc.
        poisson_ratio : float, optional
            Poisson's ratio of the substrate, default 0.49
        regularization : float, optional
            Tikhonov regularization parameter, default 1e-6
        gel_height : float, optional
            Height of the gel in physical units (meters)
        correction_method : CorrectionMethod, optional
            Method for thickness correction
        filter_sigma : float, optional
            Standard deviation for Gaussian smoothing
        """
        self.young_modulus = young_modulus
        self.pixel_size = pixel_size
        self.characteristic_length = characteristic_length
        self.poisson_ratio = poisson_ratio
        self.regularization = regularization
        self.gel_height = gel_height
        self.correction_method = CorrectionMethod(correction_method)
        self.filter_sigma = filter_sigma

        # Computed properties
        self.shear_modulus = young_modulus / (2 * (1 + poisson_ratio))

        # Results storage
        self.latest_results: Optional[Dict[str, np.ndarray]] = None

        # Validate parameters and correction method
        self._validate_parameters()
        validation = self.validate_correction_method()

        # Log warnings and raise errors if any
        for warning in validation.warnings:
            warn(warning)
        if validation.errors:
            raise ValueError("\n".join(validation.errors))

    def validate_correction_method(self) -> ValidationResult:
        """
        Validate the selected correction method based on experimental parameters.

        Returns
        -------
        ValidationResult
            Contains validation status, warnings, and errors
        """
        warnings = []
        errors = []
        is_valid = True

        if self.gel_height is None:
            if self.correction_method != CorrectionMethod.NONE:
                errors.append("Gel height must be provided for finite thickness or pure shear corrections")
                is_valid = False
        else:
            # Calculate dimensionless ratios
            height_to_length_ratio = self.gel_height / self.characteristic_length

            if self.correction_method == CorrectionMethod.PURE_SHEAR:
                if height_to_length_ratio > 0.1:
                    warnings.append(
                        f"Pure shear approximation may be inaccurate: gel height ({self.gel_height*1e6:.1f}µm) "
                        f"is not much smaller than characteristic length ({self.characteristic_length*1e6:.1f}µm)"
                    )

                if self.gel_height < 0.5e-6:  # 0.5 µm threshold
                    warnings.append(
                        "Gel height is extremely small (<0.5µm). Bulk gel properties may not apply, "
                        "and surface effects might dominate"
                    )

            elif self.correction_method == CorrectionMethod.FINITE:
                if height_to_length_ratio < 0.1:
                    warnings.append(
                        "Consider using pure shear approximation: gel is very thin compared to "
                        "characteristic length scale"
                    )

            elif self.correction_method == CorrectionMethod.NONE:
                if height_to_length_ratio < 1.0:
                    warnings.append(
                        "Infinite thickness assumption may be inaccurate: gel height is smaller than "
                        "characteristic length scale"
                    )

        return ValidationResult(is_valid=is_valid, warnings=warnings, errors=errors)

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.young_modulus <= 0:
            raise ValueError("Young's modulus must be positive")
        if self.pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
        if self.characteristic_length <= 0:
            raise ValueError("Characteristic length must be positive")
        if not 0 <= self.poisson_ratio < 0.5:
            raise ValueError("Poisson's ratio must be in [0, 0.5)")
        if self.regularization < 0:
            raise ValueError("Regularization parameter must be non-negative")

    def _prepare_displacement_fields(
            self,
            u: np.ndarray,
            v: np.ndarray,
            pad_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare displacement fields for FFT with proper padding."""
        # Create padded arrays
        u_pad = np.zeros((pad_size, pad_size))
        v_pad = np.zeros((pad_size, pad_size))

        # Get original dimensions
        M, N = u.shape

        # Remove mean and convert to meters (like in original code)
        u_shift = (u - np.mean(u)) * self.pixel_size
        v_shift = (v - np.mean(v)) * self.pixel_size

        # Place the data in the corner like in original code
        u_pad[pad_size - M:pad_size, pad_size - N:pad_size] = u_shift
        v_pad[pad_size - M:pad_size, pad_size - N:pad_size] = v_shift

        return u_pad, v_pad

    def _calculate_wave_vectors(self, pad_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate wave vectors for Fourier space operations using original method."""
        # First half: 0 to N/2
        kx1 = np.array([list(range(0, int(pad_size / 2), 1)), ] * int(pad_size), dtype=float)
        # Second half: -N/2 to -1
        kx2 = np.array([list(range(-int(pad_size / 2), 0, 1)), ] * int(pad_size), dtype=float)

        # Combine and scale properly
        kx = np.append(kx1, kx2, axis=1) * 2 * np.pi / (self.pixel_size * pad_size)
        ky = np.transpose(kx)
        k = np.sqrt(kx ** 2 + ky ** 2)

        return kx, ky, k

    def calculate_forces(
            self,
            u: np.ndarray,
            v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate traction forces from displacement fields."""
        if u.shape != v.shape:
            raise ValueError("Displacement fields u and v must have the same shape")

        # Get original dimensions
        M, N = u.shape

        # Calculate optimal padding size
        max_size = max(M, N)
        pad_size = 2 ** int(np.ceil(np.log2(max_size)))
        if pad_size % 2 != 0:
            pad_size += 1

        # Prepare displacement fields
        u_pad, v_pad = self._prepare_displacement_fields(u, v, pad_size)

        # Transform to Fourier space
        u_ft = fft2(u_pad)
        v_ft = fft2(v_pad)

        # Remove translation components
        u_ft[0, 0] = 0
        v_ft[0, 0] = 0

        # Calculate wave vectors
        kx, ky, k = self._calculate_wave_vectors(pad_size)

        # Calculate forces based on correction method
        if self.correction_method == CorrectionMethod.PURE_SHEAR:
            tx_ft, ty_ft = self._calculate_pure_shear(u_ft, v_ft)
        else:
            try:
                if self.correction_method == CorrectionMethod.FINITE:
                    tx_ft, ty_ft = self._apply_finite_thickness_correction(u_ft, v_ft, kx, ky, k)
                else:
                    tx_ft, ty_ft = self._apply_infinite_thickness_solution(u_ft, v_ft, kx, ky, k)

                # Check for NaN values and fall back to infinite thickness if needed
                if np.any(np.isnan(tx_ft)) or np.any(np.isnan(ty_ft)):
                    tx_ft, ty_ft = self._apply_infinite_thickness_solution(u_ft, v_ft, kx, ky, k)
            except Exception as e:
                # Fall back to infinite thickness solution if any calculation fails
                logger.warning(f"Falling back to infinite thickness solution due to: {str(e)}")
                tx_ft, ty_ft = self._apply_infinite_thickness_solution(u_ft, v_ft, kx, ky, k)

        # Transform back to real space
        tx = np.real(ifft2(tx_ft))
        ty = np.real(ifft2(ty_ft))

        # Cut back to original size (from corner like in original code)
        tx = tx[pad_size - M:pad_size, pad_size - N:pad_size]
        ty = ty[pad_size - M:pad_size, pad_size - N:pad_size]

        # Apply smoothing if requested
        if self.filter_sigma is not None:
            tx = gaussian_filter(tx, sigma=self.filter_sigma)
            ty = gaussian_filter(ty, sigma=self.filter_sigma)

        # Store results
        self.latest_results = {
            'tx': tx,
            'ty': ty,
            'magnitude': self.calculate_magnitude(tx, ty)
        }

        return tx, ty



    def _get_optimal_pad_size(self, M: int, N: int) -> int:
        """Calculate optimal padding size for FFT."""
        target_size = max(M, N)
        return 2 ** int(np.ceil(np.log2(target_size)))


    def _calculate_pure_shear(
            self,
            u_fft: np.ndarray,
            v_fft: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate forces using pure shear approximation."""
        tx_fft = self.shear_modulus * u_fft / self.gel_height
        ty_fft = self.shear_modulus * v_fft / self.gel_height

        # Zero out DC component
        tx_fft[0, 0] = 0
        ty_fft[0, 0] = 0

        return tx_fft, ty_fft

    def _calculate_regularized_forces(
            self,
            u_fft: np.ndarray,
            v_fft: np.ndarray,
            kx: np.ndarray,
            ky: np.ndarray,
            k: np.ndarray,
            pad_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate forces using regularization and optional finite thickness correction."""
        tx_fft = np.zeros((pad_size, pad_size), dtype=complex)
        ty_fft = np.zeros((pad_size, pad_size), dtype=complex)

        for i in range(pad_size):
            for j in range(pad_size):
                if i == 0 and j == 0:
                    continue

                if self.correction_method == CorrectionMethod.FINITE:
                    tx_fft[i, j], ty_fft[i, j] = self._apply_finite_thickness_correction(
                        u_fft[i, j], v_fft[i, j], kx[i, j], ky[i, j], k[i, j]
                    )
                else:
                    tx_fft[i, j], ty_fft[i, j] = self._apply_infinite_thickness_solution(
                        u_fft[i, j], v_fft[i, j], kx[i, j], ky[i, j], k[i, j]
                    )

        return tx_fft, ty_fft

    def _apply_finite_thickness_correction(
            self,
            u_fft: complex,
            v_fft: complex,
            kx: float,
            ky: float,
            k: float
    ) -> Tuple[complex, complex]:
        """Apply finite thickness correction to force calculation."""
        kh = k * self.gel_height

        # Fall back to infinite thickness for numerical stability
        if kh > 100:
            return self._apply_infinite_thickness_solution(u_fft, v_fft, kx, ky, k)

        c = np.cosh(kh)
        s = np.sinh(kh)
        s_c = np.tanh(kh)

        gamma = ((3 - 4 * self.poisson_ratio) +
                 (((1 - 2 * self.poisson_ratio) ** 2) / (c ** 2)) +
                 ((kh ** 2) / (c ** 2))) / \
                ((3 - 4 * self.poisson_ratio) * s_c + kh / (c ** 2))

        factor1 = v_fft * kx - u_fft * ky
        factor2 = u_fft * kx + v_fft * ky

        tx = ((-self.young_modulus * ky * c) /
              (2 * k * s * (1 + self.poisson_ratio))) * factor1 + \
             ((self.young_modulus * kx) /
              (2 * k * (1 - self.poisson_ratio ** 2))) * gamma * factor2

        ty = ((self.young_modulus * kx * c) /
              (2 * k * s * (1 + self.poisson_ratio))) * factor1 + \
             ((self.young_modulus * ky) /
              (2 * k * (1 - self.poisson_ratio ** 2))) * gamma * factor2

        return tx, ty

    def _apply_infinite_thickness_solution(
            self,
            u_fft: np.ndarray,
            v_fft: np.ndarray,
            kx: np.ndarray,
            ky: np.ndarray,
            k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply infinite thickness solution using fast vectorized operations with proper regularization.
        Based on the FFTTC method but including correct Tikhonov regularization.
        """
        # Calculate angle between k and kx
        alpha = np.arctan2(ky, kx)
        alpha[0, 0] = np.pi / 2  # Handle division by zero at origin

        # Calculate Green's function components
        denom = self.young_modulus * k ** 3
        # Add small number to prevent division by zero
        denom = denom + np.finfo(float).eps

        Gxx = 2 * (1 + self.poisson_ratio) / denom * \
              ((1 - self.poisson_ratio) * k ** 2 + self.poisson_ratio * ky ** 2)
        Gyy = 2 * (1 + self.poisson_ratio) / denom * \
              ((1 - self.poisson_ratio) * k ** 2 + self.poisson_ratio * kx ** 2)
        Gxy = 2 * (1 + self.poisson_ratio) / denom * \
              (self.poisson_ratio * kx * ky)

        # Zero out Nyquist frequencies in coupling terms
        M, N = u_fft.shape
        Gxy[:, N // 2] = 0
        Gxy[M // 2, :] = 0

        # Calculate determinant for inverse
        det = Gxx * Gyy - Gxy * Gxy + self.regularization

        # Calculate inverse with regularization (derived from 2x2 matrix inversion formula)
        # [Gxx  Gxy] ^-1     1    [ Gyy  -Gxy]
        # [Gxy  Gyy]    = ------- [-Gxy   Gxx]
        #                   det

        Kxx = Gyy / det
        Kyy = Gxx / det
        Kxy = -Gxy / det

        # Set zero frequency components to zero
        Kxx[0, 0] = 0
        Kyy[0, 0] = 0
        Kxy[0, 0] = 0

        # Calculate forces in Fourier space using vectorized operations
        tx_fft = Kxx * u_fft + Kxy * v_fft
        ty_fft = Kxy * u_fft + Kyy * v_fft

        return tx_fft, ty_fft

    @staticmethod
    def calculate_magnitude(tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
        """Calculate magnitude of traction forces."""
        return np.sqrt(tx ** 2 + ty ** 2)

    def calculate_strain_energy(self) -> Optional[float]:
        """
        Calculate total strain energy density from latest results.

        Returns
        -------
        float or None
            Total strain energy density (Joules/m²) if results exist,
            None otherwise
        """
        if self.latest_results is None:
            return None

        tx = self.latest_results['tx']
        ty = self.latest_results['ty']

        # Calculate strain energy density
        strain_energy = 0.5 * (tx ** 2 + ty ** 2) / self.young_modulus

        # Return total energy per unit area
        return np.sum(strain_energy) * self.pixel_size ** 2

    def get_statistics(self) -> Optional[Dict[str, float]]:
        """
        Get statistical information about the latest calculation.

        Returns
        -------
        Dict[str, float] or None
            Dictionary containing statistics if results exist,
            None otherwise
        """
        if self.latest_results is None:
            return None

        magnitude = self.latest_results['magnitude']
        return {
            'mean_force': float(np.mean(magnitude)),
            'max_force': float(np.max(magnitude)),
            'std_force': float(np.std(magnitude)),
            'total_strain_energy': float(self.calculate_strain_energy())
        }