"""
Generalized Cross-Validation (GCV) module for optimizing regularization parameters in TFM analysis.

This implementation is based on:
- DirectMethod package (https://github.com/usschwarz/DirectMethod), licensed under MIT License
- P. C. Hansen, Regularization Tools Version 4.0 for Matlab 7.3 files gcv.m and gcvfun.m

Original paper references:
- Blumberg & Schwarz, Comparison of direct and inverse methods for 2.5D traction force microscopy (2022)
  https://doi.org/10.1371/journal.pone.0262773
- Golub, G. H., Heath, M., & Wahba, G. Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter (2012)

Code references:
- DirectMethod by Schwarz Lab (https://github.com/usschwarz/DirectMethod) - MIT License
- P. C. Hansen, Regularization Tools Version 4.0
"""

import numpy as np
from scipy.interpolate import griddata, SmoothBivariateSpline
from scipy.optimize import fmin


class GCVAnalysis:
    def __init__(self, E: float = 1e5, nu: float = 0.5, mesh_size: int = 4,
                 alpha: float = 1.2, omega: float = 1.5):
        self.E = E
        self.nu = nu
        self.mesh_size = mesh_size
        self.alpha = alpha  # MGCV parameter for increased regularization
        self.omega = omega  # Stabilization parameter

    def calculate_gcv(self, u_data: np.ndarray, v_data: np.ndarray,
                      lamlow: float = -8, lamhigh: float = 2,
                      lamcount: int = 50, plot: bool = True) -> float:
        """Calculate optimal regularization parameter using GCV."""
        x = np.arange(u_data.shape[0])
        y = np.arange(u_data.shape[1])
        xg, yg = np.meshgrid(x, y, indexing='ij')
        pos0 = np.array([xg.flatten(), yg.flatten()])
        vec0 = np.array([u_data.flatten(), v_data.flatten()])

        # Estimate noise level from data
        noise_level = self._estimate_noise_level(u_data, v_data)

        lambdarange = np.logspace(lamlow, lamhigh, lamcount)
        blockU, s, b = self._svd_block(pos0, vec0)
        reg_min, minG, G, reg_param = self._gcv_blockdiag(blockU, s, b, lambdarange, noise_level, plot)

        # Convert to regularization parameter with stabilization
        kx, ky = self._calculate_wave_vectors(u_data.shape[0])
        kmax = np.sqrt(kx.max() ** 2 + ky.max() ** 2)
        regularization = self.omega * reg_min / (kmax ** 2)

        return regularization

    def _estimate_noise_level(self, u_data: np.ndarray, v_data: np.ndarray) -> float:
        """Estimate noise level from displacement data using local variations."""
        # Calculate local variations in displacement field
        du_dx = np.gradient(u_data, axis=0)
        du_dy = np.gradient(u_data, axis=1)
        dv_dx = np.gradient(v_data, axis=0)
        dv_dy = np.gradient(v_data, axis=1)

        # Estimate noise from median absolute deviation of gradients
        gradients = np.concatenate([du_dx.flatten(), du_dy.flatten(),
                                    dv_dx.flatten(), dv_dy.flatten()])
        mad = np.median(np.abs(gradients - np.median(gradients)))

        # Convert MAD to standard deviation (assuming Gaussian noise)
        noise_estimate = 1.4826 * mad
        return noise_estimate

    def _gcv_blockdiag(self, U: np.ndarray, s: np.ndarray, b: np.ndarray,
                       lambdarange: np.ndarray, noise_level: float, plot: bool) -> tuple:
        """Calculate GCV function and find minimum."""
        G = np.array([self._gcvfun(lam, s ** 2, b, noise_level) for lam in lambdarange])
        minGi = G.argmin()

        reg_min = fmin(self._gcvfun, x0=lambdarange[minGi],
                       args=(s ** 2, b, noise_level), disp=0)[0]
        minG = self._gcvfun(reg_min, s ** 2, b, noise_level)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(lambdarange, G, '-', label='Modified GCV curve')
            plt.plot([reg_min], [minG], '*r', label='Minimum')
            plt.plot([reg_min, reg_min], [minG / 1000, minG], ':r')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('λ')
            plt.ylabel('G(λ)')
            plt.title(f'Modified GCV function, minimum at λ = {reg_min:.2e}')
            plt.legend()
            plt.grid(True)
            plt.show()

        return reg_min, minG, G, lambdarange

    def _gcvfun(self, lmbda: float, s2: np.ndarray, beta: np.ndarray,
                noise_level: float) -> float:
        """Calculate modified GCV function value with noise consideration."""
        f = (lmbda ** 2) / (s2 + lmbda ** 2)

        # Modified denominator with alpha parameter for stronger regularization
        effective_dof = np.sum(f)
        modified_dof = effective_dof ** self.alpha

        # Include noise level in the calculation
        residual_norm = np.linalg.norm(f * beta) ** 2
        noise_term = noise_level * np.sum(f * s2 / (s2 + lmbda ** 2))

        G = (residual_norm + noise_term) / modified_dof ** 2
        return G

    # [Previous methods remain unchanged]
    def _calculate_wave_vectors(self, N: int) -> tuple:
        """Calculate wave vectors."""
        k = 2.0 * np.pi * np.fft.fftfreq(N)
        kx, ky = np.meshgrid(k, k)
        return kx, ky

    def _svd_block(self, pos: np.ndarray, vec: np.ndarray):
        """Calculate SVD of the system matrix."""
        grid_mat, u, i_max, j_max = self._interp_vec2grid(pos, vec)
        kx, ky = self._calculate_wave_vectors(i_max)
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

    def _calculate_greens_function(self, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
        """Calculate Green's function in Fourier space."""
        V = 2 * (1 + self.nu) / self.E
        kx_sq = kx ** 2
        ky_sq = ky ** 2
        kabs = np.sqrt(kx_sq + ky_sq)
        kabs_sq = kx_sq + ky_sq

        GFt = V * kabs ** (-3) * np.array([
            [kabs_sq - self.nu * kx_sq, -self.nu * kx * ky],
            [-self.nu * kx * ky, kabs_sq - self.nu * ky_sq]
        ])
        GFt[:, :, 0, 0] = 0.0
        return GFt

    def _interp_vec2grid(self, pos: np.ndarray, vec: np.ndarray) -> tuple:
        """Interpolate vector field to regular grid."""
        max_corner = np.array([pos[0].max(), pos[1].max()])
        min_corner = np.array([pos[0].min(), pos[1].min()])

        i_max = int(np.round((max_corner[0] - min_corner[0]) / self.mesh_size))
        j_max = int(np.round((max_corner[1] - min_corner[1]) / self.mesh_size))
        i_max -= int(np.mod(i_max, 2))
        j_max -= int(np.mod(j_max, 2))

        X, Y = np.meshgrid(
            min_corner[0] + np.arange(0.5, i_max, 1) * self.mesh_size,
            min_corner[1] + np.arange(0.5, j_max, 1) * self.mesh_size
        )

        grid_mat = np.array([X, Y])
        u_center = np.array([
            griddata((pos[0], pos[1]), vec[0], (X, Y), method='cubic'),
            griddata((pos[0], pos[1]), vec[1], (X, Y), method='cubic')
        ])

        u_center = self._extrapolate_u(grid_mat, u_center)
        return grid_mat, u_center, i_max, j_max

    def _extrapolate_u(self, grid_mat: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Extrapolate nan values in displacement field."""
        grid_positions = np.array([grid_mat[0].flatten(), grid_mat[1].flatten()]).T
        displacements = np.array([u[0].flatten(), u[1].flatten()]).T
        mask = np.ones(displacements.shape[0], dtype=bool)
        mask[~np.isnan(displacements[:, 0])] = False

        valid_pos = grid_positions[~mask]
        invalid_pos = grid_positions[mask]
        valid_dis = displacements[~mask]

        try:
            sbs_0 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1],
                                          valid_dis[:, 0], kx=3, ky=3)
            sbs_1 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1],
                                          valid_dis[:, 1], kx=3, ky=3)
        except:
            sbs_0 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1],
                                          valid_dis[:, 0], kx=1, ky=1)
            sbs_1 = SmoothBivariateSpline(valid_pos[:, 0], valid_pos[:, 1],
                                          valid_dis[:, 1], kx=1, ky=1)

        u[0][np.isnan(u[0])] = sbs_0.ev(invalid_pos[:, 0], invalid_pos[:, 1])
        u[1][np.isnan(u[1])] = sbs_1.ev(invalid_pos[:, 0], invalid_pos[:, 1])
        return u


if __name__ == "__main__":
    u_data = np.load("C:/Users/aruppel/Desktop/test/d_x.npy")[0, :, :]
    v_data = np.load("C:/Users/aruppel/Desktop/test/d_y.npy")[0, :, :]
    gcv = GCVAnalysis(E=1e5, nu=0.5, alpha=1.2, omega=4)
    reg_param = gcv.calculate_gcv(u_data, v_data, plot=True)
    print(reg_param)