"""
Traction Force Microscopy (TFM) force calculation module implementing the FTTC method 
with Generalized Cross-Validation (GCV) for regularization parameter optimization.

Core TFM functions and GCV implementation is based on:
- DirectMethod package (https://github.com/usschwarz/DirectMethod) - MIT License
- Blumberg & Schwarz, Comparison of direct and inverse methods for 2.5D traction force microscopy (2022)
  https://doi.org/10.1371/journal.pone.0262773


Gel height correction implementation adapted from:
- pyTFM package (https://github.com/fabrylab/pyTFM) - GNU GPL v3.0 License

Paper references:
- Butler et al. Traction fields, moments, and strain energy that cells exert on 
  their surroundings (2002)
- Sabass et al. High resolution traction force microscopy based on experimental and 
  computational advances (2008)
- Trepat et al. Physical forces during collective cell migration (2009)
- P. C. Hansen, Regularization Tools Version 4.0 for Matlab 7.3 (gcv.m and gcvfun.m)
- Golub, G. H., Heath, M., & Wahba, G. Generalized Cross-Validation as a Method for 
  Choosing a Good Ridge Parameter (2012)
"""

import time
from typing import Tuple, Optional

from scipy import optimize

from napariTFM.fttc_numba_functions import *


class FTTC:
    def __init__(self, E: float, nu: float, mesh_size: int = 1, lanczos_exp: int = 1, gel_height: float = float('inf')):
        self.E = E
        self.nu = nu
        self.mesh_size = mesh_size
        self.lanczos_exp = lanczos_exp
        self.gel_height = gel_height  # Added gel height parameter
        self.timing_stats = {}
        self.detailed_timing = {}

    def calculate_traction(self, x: np.ndarray, y: np.ndarray,
                           u_data: np.ndarray, v_data: np.ndarray,
                           dx: float, set_lam: Optional[float] = None) -> Tuple:
        """
        Calculate traction forces from displacement data

        Args:
            x: x coordinates
            y: y coordinates
            u_data: x displacement data
            v_data: y displacement data
            dx: pixel size
            set_lam: optional regularization parameter

        Returns:
            Tuple containing:
            - (x, y) coordinates
            - traction magnitude
            - traction vectors
            - reconstructed displacement
            - original displacement
            - energy
            - total force
            - Fourier transform of traction
            - Fourier transform of reconstructed displacement
        """
        # Create grid using original coordinates
        xpix, ypix = np.meshgrid(x, y, indexing='xy')
        pos0 = np.array([xpix.flatten(), ypix.flatten()])

        # Scale only the displacement vectors, not the coordinates
        pix_per_mu = self.mesh_size / dx
        vec0 = pix_per_mu * np.array([u_data.flatten(), v_data.flatten()])

        if set_lam is None:
            lam = self._find_regularization(pos0, vec0)
        else:
            lam = set_lam

        return self._perform_tfm(pos0, vec0, pix_per_mu, lam)

    def _calculate_greens_function(self, kx: np.ndarray, ky: np.ndarray):
        """Calculate Green's function in Fourier space with gel height correction"""
        # Scale k-vectors by pixel size to get physical units
        V = 2 * (1 + self.nu) / self.E
        kx_sq = kx ** 2
        ky_sq = ky ** 2
        kabs = np.sqrt(kx_sq + ky_sq)
        kabs_sq = kx_sq + ky_sq

        # Standard Green's function components
        GFt_std = V * kabs ** (-3) * np.array([
            [kabs_sq - self.nu * kx_sq, -self.nu * kx * ky],
            [-self.nu * kx * ky, kabs_sq - self.nu * ky_sq]
        ])
        GFt_std[:, :, 0, 0] = 0.0

        # Apply gel height correction if finite (not None and not infinity)
        if self.gel_height is not None:
            kh = kabs * self.gel_height
            mask_standard = kh > 100  # Use standard for large kh

            # Calculate finite thickness components
            c = np.cosh(kh)
            s = np.sinh(kh)

            # Correction factor
            gamma = ((3 - 4 * self.nu) +
                     (((1 - 2 * self.nu) ** 2) / (c ** 2)) +
                     ((kh ** 2) / (c ** 2))) / \
                    ((3 - 4 * self.nu) * np.tanh(kh) + kh / (c ** 2))

            # Apply correction while handling numerical stability
            gamma[mask_standard] = 1.0
            gamma[0, 0] = 1.0  # Handle k=0 case

            # Apply correction to all components
            GFt = np.empty_like(GFt_std)
            for i in range(2):
                for j in range(2):
                    GFt[i, j] = GFt_std[i, j] * gamma

            return GFt

        return GFt_std

    def _perform_tfm(self, pos0: np.ndarray, vec0: np.ndarray,
                     pix_per_mu: float, lam: float,
                     i_max: Optional[int] = None,
                     j_max: Optional[int] = None) -> Tuple:
        """Core TFM calculation with shape matching to input"""

        # Store original input dimensions
        original_shape = (int(np.sqrt(vec0.shape[1])), int(np.sqrt(vec0.shape[1])))

        # Original TFM calculation
        grid_mat, u, i_max, j_max, i_bound_size, j_bound_size = self._interp_vec2grid(
            pos0, vec0, i_max=i_max, j_max=j_max)

        kx, ky, lanczosx, lanczosy = self._calculate_fourier_modes(i_max, j_max)
        GFt = self._calculate_greens_function(kx, ky)

        G_inv = calculate_traction_2d(GFt, lam ** 2)
        G_inv_xx = G_inv[0, 0]
        G_inv_xy = G_inv[0, 1]
        G_inv_yy = G_inv[1, 1]

        Ftfx, Ftfy = self._reg_fourier_TFM_L2(u, G_inv_xx, G_inv_xy, G_inv_yy)
        Ftf = np.array([Ftfx, Ftfy])

        urec, Fturec = self._reconstruct_displacement(GFt, Ftfx, Ftfy, lanczosx, lanczosy)

        pos, vec, fnorm, f, energy, force = self._calculate_stress_field(
            Ftfx, Ftfy, lanczosx, lanczosy, grid_mat, u, i_max, j_max,
            i_bound_size, j_bound_size, pix_per_mu)

        x = np.reshape(pos[0], (i_max, j_max)).T / pix_per_mu
        y = np.reshape(pos[1], (i_max, j_max)).T / pix_per_mu

        # Pad output arrays to match input dimensions
        def pad_to_shape(arr, target_shape):
            if isinstance(arr, np.ndarray):
                current_shape = arr.shape
                if len(current_shape) == 2:
                    pad_width = [(0, target_shape[0] - current_shape[0]),
                                 (0, target_shape[1] - current_shape[1])]
                    return np.pad(arr, pad_width, mode='constant', constant_values=0)
                elif len(current_shape) == 3:  # For arrays like f with components
                    pad_width = [(0, 0),  # Don't pad the components dimension
                                 (0, target_shape[0] - current_shape[1]),
                                 (0, target_shape[1] - current_shape[2])]
                    return np.pad(arr, pad_width, mode='constant', constant_values=0)
            return arr

        # Pad relevant outputs
        fnorm = pad_to_shape(fnorm, original_shape)
        f = pad_to_shape(f, original_shape)
        urec = pad_to_shape(urec, original_shape)

        # Create padded coordinate grids to match dimensions
        x_full = np.linspace(x[0, 0], x[-1, -1], original_shape[1])
        y_full = np.linspace(y[0, 0], y[-1, -1], original_shape[0])
        x, y = np.meshgrid(x_full, y_full)

        return (x, y), fnorm, f, urec, u, energy, force, Ftf, Fturec

    @staticmethod
    def _gcvfun(lmbda, s2, beta, delta0, mn):
        """Auxiliary routine for GCV calculation"""
        f = (lmbda ** 2) / (s2 + lmbda ** 2)
        G = (np.linalg.norm(f * beta) ** 2 + delta0) / (mn + np.sum(f)) ** 2
        return G

    def _gcv_blockdiag(self, U: np.ndarray, s: np.ndarray, b: np.ndarray,
                       lambdarange: np.ndarray, plot: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Calculate generalized cross validation"""
        npoints = lambdarange.size
        beta = blkmul_adj(U, b)

        reg_param = np.copy(lambdarange)
        G = np.zeros(npoints)
        s2 = s ** 2

        for i in range(npoints):
            G[i] = self._gcvfun(reg_param[i], s2, beta[:s.size], 0., 0)

        minGi = G.argmin(0)
        reg_min = optimize.fmin(
            self._gcvfun,
            x0=reg_param[np.max([minGi, 0])],
            args=(s2, beta[:s.size], 0., 0),
            disp=0,
        )[0]
        minG = self._gcvfun(reg_min, s2, beta[:s.size], 0., 0)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(reg_param, G, "-")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel(r"$G(\lambda)$")
            plt.plot([reg_min], [minG], "*r")
            plt.plot([reg_min, reg_min], [minG / 1000, minG], ":r")
            plt.title(r"GCV function, minimum at $\lambda = %.2e$" % reg_min)
            plt.show()

        return float(reg_min), float(minG), G, reg_param

    def _find_regularization(self, pos0: np.ndarray, vec0: np.ndarray) -> float:
        """Find optimal regularization parameter using GCV"""
        lamguess = 0.2 / self.E
        lamlow = np.log10(lamguess) - 5.0
        lamhigh = np.log10(lamguess) + 5.0
        lambdarange = np.logspace(lamlow, lamhigh, 50)

        blockU, s, b = self._svd_block(pos0, vec0)
        reg_min, _, _, _ = self._gcv_blockdiag(blockU, s, b, lambdarange, plot=False)
        return reg_min

    def _svd_block(self, pos: np.ndarray, vec: np.ndarray):
        """Prepare SVD representation of the FTTC problem"""
        grid_mat, u, i_max, j_max, _, _ = self._interp_vec2grid(pos, vec)
        kx, ky, _, _ = self._calculate_fourier_modes(i_max, j_max)
        GFt = self._calculate_greens_function(kx, ky)

        Ftu = np.fft.fft2(u).reshape(2, -1).T
        shape = GFt[0, 0].shape

        U_h = np.empty((shape[0] * shape[1], 2, 2), dtype=np.complex128)
        s_h = np.empty((shape[0] * shape[1], 2))
        for i in range(shape[0]):
            for j in range(shape[1]):
                idx = i * shape[1] + j
                U_h[idx, :], s_h[idx, :], _ = np.linalg.svd(GFt[:, :, i, j])

        return U_h, s_h.flatten(), Ftu.flatten()

    def _interp_vec2grid(self, pos: np.ndarray, vec: np.ndarray,
                         i_max: Optional[int] = None, j_max: Optional[int] = None):
        """Highly optimized interpolation using KD-tree based approach"""
        from scipy.spatial import cKDTree

        # Calculate grid dimensions
        max_corner = np.array([np.max(pos[0]), np.max(pos[1])])
        min_corner = np.array([np.min(pos[0]), np.min(pos[1])])

        if i_max is None and j_max is None:
            i_max = np.round((max_corner[0] - min_corner[0]) / self.mesh_size)
            j_max = np.round((max_corner[1] - min_corner[1]) / self.mesh_size)
            i_max -= np.int64(np.mod(i_max, 2))
            j_max -= np.int64(np.mod(j_max, 2))

        i_max, j_max = np.int64(i_max), np.int64(j_max)

        # Create target grid points
        x = min_corner[0] + np.arange(0.5, i_max, 1) * self.mesh_size
        y = min_corner[1] + np.arange(0.5, j_max, 1) * self.mesh_size
        X, Y = np.meshgrid(x, y)
        grid_mat = np.array([X, Y])

        # Prepare source points and values
        source_points = np.column_stack((pos[0].ravel(), pos[1].ravel()))
        values = np.column_stack((vec[0].ravel(), vec[1].ravel()))

        # Remove NaN values
        valid_mask = ~np.isnan(values).any(axis=1)
        valid_points = source_points[valid_mask]
        valid_values = values[valid_mask]

        # Create KD-tree for efficient nearest neighbor search
        tree = cKDTree(valid_points)

        # Prepare target points
        target_points = np.column_stack((X.ravel(), Y.ravel()))

        # Initialize output arrays
        u = np.empty((2, *X.shape), dtype=np.float64)

        # Find k nearest neighbors for each target point
        k = min(12, len(valid_points))  # Adjust k based on available points
        distances, indices = tree.query(target_points, k=k)

        # Convert distances to weights using inverse distance weighting
        weights = 1.0 / (distances + np.finfo(float).eps)  # Add eps to avoid division by zero
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        normalized_weights = weights / weights_sum

        # Compute weighted average of values
        weighted_values = np.sum(valid_values[indices] * normalized_weights[..., np.newaxis], axis=1)

        # Reshape results
        u[0] = weighted_values[:, 0].reshape(X.shape)
        u[1] = weighted_values[:, 1].reshape(X.shape)

        # Handle edge cases where interpolation might fail
        if np.any(np.isnan(u)):
            # Fall back to nearest neighbor for any remaining NaN values
            nan_mask = np.isnan(u[0]) | np.isnan(u[1])
            if np.any(nan_mask):
                flat_indices = nan_mask.ravel()
                _, nearest_indices = tree.query(target_points[flat_indices], k=1)
                u[0].ravel()[flat_indices] = valid_values[nearest_indices, 0]
                u[1].ravel()[flat_indices] = valid_values[nearest_indices, 1]

        return grid_mat, u, i_max, j_max, 0, 0


    def _calculate_fourier_modes(self, i_max: int, j_max: int):
        """Calculate Fourier modes and Lanczos filter"""
        kx_vec = 2. * np.pi / i_max / self.mesh_size * np.append(
            np.arange(0, (i_max // 2)), np.arange(-i_max // 2, 0))
        ky_vec = 2. * np.pi / j_max / self.mesh_size * np.append(
            np.arange(0, (j_max // 2)), np.arange(-j_max // 2, 0))
        kx, ky = np.meshgrid(kx_vec, ky_vec)

        lanczosx = np.sinc(kx * self.mesh_size / np.pi) ** self.lanczos_exp
        lanczosy = np.sinc(ky * self.mesh_size / np.pi) ** self.lanczos_exp

        kx[0, 0] = 1
        ky[0, 0] = 1
        return kx, ky, lanczosx, lanczosy

    def _reg_fourier_TFM_L2(self, u: np.ndarray, Ginv_xx: np.ndarray,
                            Ginv_xy: np.ndarray, Ginv_yy: np.ndarray):
        """Calculate Fourier transformed traction field"""
        Ftux = np.fft.fft2(u[0])
        Ftuy = np.fft.fft2(u[1])
        Ftfx = Ginv_xx * Ftux + Ginv_xy * Ftuy
        Ftfy = Ginv_xy * Ftux + Ginv_yy * Ftuy
        return Ftfx, Ftfy

    def _reconstruct_displacement(self, GFt: np.ndarray, Ftfx: np.ndarray,
                                  Ftfy: np.ndarray, lanczosx: np.ndarray,
                                  lanczosy: np.ndarray):
        """Reconstruct displacement field from traction"""
        Ftux_rec = GFt[0, 0] * Ftfx + GFt[0, 1] * Ftfy
        Ftuy_rec = GFt[1, 0] * Ftfx + GFt[1, 1] * Ftfy
        ux_rec = np.fft.ifft2(lanczosx * Ftux_rec)
        uy_rec = np.fft.ifft2(lanczosy * Ftuy_rec)
        urec = np.array([np.real(ux_rec), np.real(uy_rec)])
        Fturec = np.array([Ftux_rec, Ftuy_rec])
        return urec, Fturec

    def _calculate_stress_field(self, Ftfx: np.ndarray, Ftfy: np.ndarray,
                                lanczosx: np.ndarray, lanczosy: np.ndarray,
                                grid_mat: np.ndarray, u: np.ndarray,
                                i_max: int, j_max: int,
                                i_bound_size: int, j_bound_size: int,
                                pix_per_mu: float):
        """Calculate stress field and related quantities"""
        fx = np.fft.ifft2(lanczosx * Ftfx)
        fy = np.fft.ifft2(lanczosy * Ftfy)

        pos = np.array([
            np.reshape(grid_mat[0], (i_max * j_max)),
            np.reshape(grid_mat[1], (i_max * j_max)),
        ])
        vec = np.array([
            np.reshape(u[0], (i_max * j_max)),
            np.reshape(u[1], (i_max * j_max))
        ])

        f = np.array([np.real(fx), np.real(fy)])
        fnorm = (f[0] ** 2 + f[1] ** 2) ** 0.5

        if j_bound_size > 0 and i_bound_size > 0:
            u_slice = u[:, j_bound_size:-j_bound_size, i_bound_size:-i_bound_size]
            f_slice = f[:, j_bound_size:-j_bound_size, i_bound_size:-i_bound_size]
            fnorm_slice = fnorm[j_bound_size:-j_bound_size, i_bound_size:-i_bound_size]
            energy = self._calculate_energy(u_slice, f_slice, pix_per_mu)
            force = self._calculate_total_force(fnorm_slice, pix_per_mu)
        else:
            energy = self._calculate_energy(u, f, pix_per_mu)
            force = self._calculate_total_force(fnorm, pix_per_mu)

        # # Swap both position and force components before returning
        # pos = np.array([pos[1], pos[0]])  # Swap position components
        # f = np.array([f[1], f[0]])  # Swap force components

        return pos, vec, fnorm, f, energy, force

    def _calculate_energy(self, u: np.ndarray, f: np.ndarray, pix_per_mu: float) -> float:
        """Calculate energy stored in the traction profile"""
        l = self.mesh_size / pix_per_mu * 1e-6  # nodal distance in m^2
        energy = 0.5 * l ** 2 * np.sum(u * f) * 1e-6 / pix_per_mu
        return energy

    def _calculate_total_force(self, fnorm: np.ndarray, pix_per_mu: float) -> float:
        """Calculate L2 norm of the force field"""
        unit_factor = self.mesh_size / pix_per_mu * 1e-6
        total_force = unit_factor ** 2 * np.sum(fnorm)
        return total_force


