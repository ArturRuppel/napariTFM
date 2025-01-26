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

from typing import Tuple, Optional

from scipy import optimize

from napariTFM.backend.fttc_numba_functions import *


class FTTC:
    def __init__(self, E: float, nu: float, lanczos_exp: int = 1, gel_height: float = float('inf')):
        """
        Initialize FTTC calculator.

        Args:
            E: Young's modulus in Pa
            nu: Poisson ratio
            lanczos_exp: Lanczos filter exponent
            gel_height: Gel height for correction (default: infinity)
        """
        self.E = E
        self.nu = nu
        self.lanczos_exp = lanczos_exp
        self.gel_height = gel_height
    def calculate_traction(self, displacements: Tuple[np.ndarray, np.ndarray],
                           pixel_size: float,
                           downscale_factor: int = 1,
                           regularization: float = None) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Calculate traction forces from displacement field measurements using Fourier Transform
        Traction Cytometry (FTTC).

        Parameters
        ----------
        displacements : np.ndarray (shape: H x W x 2)
            - dx: x-direction displacements (shape: H x W, displacements[..., 0])
            - dy: y-direction displacements (shape: H x W, displacements[..., 1])
            Units: micrometers (μm)
            The displacement fields should represent how far each point in the gel
            has moved from its original position.

        pixel_size : float
            Physical size of each pixel in the displacement field.
            Units: micrometers (μm)
            Example: for a 100x objective with 0.1 μm/pixel, use 0.1

        downscale_factor : int
            Factor representing the spatial downsampling that was already applied
            to the displacement field data before being passed to this function.
            This is used to correctly scale the pixel size for force calculations.

        regularization : float, optional (default=None)
            Tikhonov regularization parameter (λ) for the inverse problem.
            Units: dimensionless
            If None, will be automatically determined using Generalized Cross-Validation (GCV).
            Typical values range from 1e-6 to 1e-3.
            Higher values give smoother force fields but may underestimate peak forces.

        Returns
        -------
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
            Returns ((x, y), forces) where:

            x, y : np.ndarray
                2D coordinate grids (shape: H x W each) giving the physical position
                corresponding to each point in the force field.
                Units: micrometers (μm)
                These can be used for plotting or further analysis.

            forces : np.ndarray
                Calculated traction forces (shape: 2 x H x W)
                - forces[0]: x-direction forces
                - forces[1]: y-direction forces
                Units: N/m² (Pascals)
                These represent the forces exerted by the cell on the substrate
                at each point.

        Notes
        -----
        The calculation involves several steps:
        1. Fourier transform of the displacement field
        2. Application of the Green's function to calculate forces
        3. Inverse Fourier transform to get the final force field

        The relationship between forces and displacements is given by the
        Boussinesq solution in Fourier space, modified by the Tikhonov
        regularization parameter to handle noise in the measurements.

        Examples
        --------
        >> fttc = FTTC(E=10000, nu=0.5)  # Initialize with gel properties
        >> # dx and dy are already in micrometers
        >> worker = fttc.calculate_traction(
        ...     displacements=(dx, dy),
        ...     pixelsize=0.1,  # 0.1 μm per pixel
        ...     downscale_factor=4  # if data was previously downsampled by factor of 4
        ... )
        >> # Set up callbacks
        >> def handle_result(result):
        ...     (x, y), forces = result
        ...     force_magnitude = np.sqrt(forces[0]**2 + forces[1]**2)
        ...     # Process the results here
        >> worker.returned.connect(handle_result)
        >> worker.start()
        """
        d_x = displacements[..., 0]
        d_y = displacements[..., 1]


        # Create coordinate grid
        x = np.arange(d_x.shape[1])
        y = np.arange(d_x.shape[0])

        # Create position array in pixel coordinates
        pos = np.array([np.ones(len(y))[:, None] * x,
                        y[:, None] * np.ones(len(x))])

        # Create displacement vector array, already in physical units
        vec = np.array([d_x.flatten(), d_y.flatten()])

        # Convert pixel coordinates to physical units inside _perform_tfm
        forcemap_pixel_size = pixel_size * downscale_factor

        # Calculate forces
        if regularization is None:
            regularization = self._find_regularization(pos, vec, forcemap_pixel_size)

        return self._perform_tfm(pos, vec, forcemap_pixel_size, regularization)

    def _perform_tfm(self, pos: np.ndarray, vec: np.ndarray,
                     pixelsize: float, regularization: float,
                     i_max: Optional[int] = None,
                     j_max: Optional[int] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Core TFM calculation.

        Args:
            pos: Position array in pixel coordinates (2, N)
            vec: Displacement vector array in physical units (2, N)
            pixelsize: Effective pixel size in meters (including any downsampling)
            regularization: Regularization parameter (lambda)
            i_max, j_max: Optional output grid dimensions

        Returns:
            Tuple containing:
            - (x, y) coordinate grids in physical units
            - forces array in N/m²
        """

        # Store original input dimensions
        original_shape = (int(np.sqrt(vec.shape[1])), int(np.sqrt(vec.shape[1])))

        # Interpolate to regular grid
        grid_mat, u, i_max, j_max, i_bound_size, j_bound_size = self._interp_vec2grid(
            pos, vec, i_max=i_max, j_max=j_max)

        # Calculate in Fourier space using physical units
        kx, ky, lanczosx, lanczosy = self._calculate_fourier_modes(i_max, j_max, pixelsize)
        GFt = self._calculate_greens_function(kx, ky)

        G_inv = calculate_traction_2d(GFt, regularization ** 2)
        G_inv_xx = G_inv[0, 0]
        G_inv_xy = G_inv[0, 1]
        G_inv_yy = G_inv[1, 1]

        Ftfx, Ftfy = self._reg_fourier_TFM_L2(u, G_inv_xx, G_inv_xy, G_inv_yy)

        # Calculate final forces
        pos, vec, f = self._calculate_stress_field(
            Ftfx, Ftfy, lanczosx, lanczosy, grid_mat, u, i_max, j_max)

        # Convert output coordinates to physical units
        x = np.reshape(pos[0], (i_max, j_max)).T * pixelsize
        y = np.reshape(pos[1], (i_max, j_max)).T * pixelsize

        # Pad outputs to match input dimensions if needed
        f = self._pad_to_shape(f, original_shape)

        # Create full coordinate grids to match dimensions
        x_full = np.linspace(x[0, 0], x[-1, -1], original_shape[1])
        y_full = np.linspace(y[0, 0], y[-1, -1], original_shape[0])
        x, y = np.meshgrid(x_full, y_full)

        return (x, y), f

    @staticmethod
    def _pad_to_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Pad array to match target shape."""
        if isinstance(arr, np.ndarray):
            current_shape = arr.shape
            if len(current_shape) == 2:
                pad_width = [(0, target_shape[0] - current_shape[0]),
                             (0, target_shape[1] - current_shape[1])]
                return np.pad(arr, pad_width, mode='constant', constant_values=0)
            elif len(current_shape) == 3:
                pad_width = [(0, 0),
                             (0, target_shape[0] - current_shape[1]),
                             (0, target_shape[1] - current_shape[2])]
                return np.pad(arr, pad_width, mode='constant', constant_values=0)
        return arr

    def _calculate_greens_function(self, kx: np.ndarray, ky: np.ndarray):
        """Calculate Green's function in Fourier space with gel height correction"""
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

        return float(reg_min), float(minG), G, reg_param

    def _find_regularization(self, pos0: np.ndarray, vec0: np.ndarray, forcemap_pixel_size: float) -> float:
        """Find optimal regularization parameter using GCV

        Args:
            pos0: Position array in pixel coordinates
            vec0: Displacement vector array
             forcemap_pixel_size: Pixel size in micrometers

        Returns:
            float: Optimal regularization parameter
        """
        lamguess = 0.2 / self.E
        lamlow = np.log10(lamguess) - 5.0
        lamhigh = np.log10(lamguess) + 5.0
        lambdarange = np.logspace(lamlow, lamhigh, 50)

        blockU, s, b = self._svd_block(pos0, vec0, forcemap_pixel_size)
        reg_min, _, _, _ = self._gcv_blockdiag(blockU, s, b, lambdarange, plot=False)
        return reg_min

    def _svd_block(self, pos: np.ndarray, vec: np.ndarray, forcemap_pixel_size: float):
        """Prepare SVD representation of the FTTC problem

        Args:
            pos: Position array
            vec: Displacement vector array
             forcemap_pixel_size: Pixel size in micrometers
        """
        grid_mat, u, i_max, j_max, _, _ = self._interp_vec2grid(pos, vec)
        kx, ky, _, _ = self._calculate_fourier_modes(i_max, j_max, forcemap_pixel_size)
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
            i_max = np.round((max_corner[0] - min_corner[0]))
            j_max = np.round((max_corner[1] - min_corner[1]))
            i_max -= np.int64(np.mod(i_max, 2))
            j_max -= np.int64(np.mod(j_max, 2))

        i_max, j_max = np.int64(i_max), np.int64(j_max)

        # Create target grid points
        x = min_corner[0] + np.arange(0.5, i_max, 1)
        y = min_corner[1] + np.arange(0.5, j_max, 1)
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

    def _calculate_fourier_modes(self, i_max: int, j_max: int, forcemap_pixel_size: float):
        """Calculate Fourier modes and Lanczos filter"""
        kx_vec = 2. * np.pi / i_max / forcemap_pixel_size * np.append(
            np.arange(0, (i_max // 2)), np.arange(-i_max // 2, 0))
        ky_vec = 2. * np.pi / j_max / forcemap_pixel_size * np.append(
            np.arange(0, (j_max // 2)), np.arange(-j_max // 2, 0))
        kx, ky = np.meshgrid(kx_vec, ky_vec)

        lanczosx = np.sinc(kx / np.pi) ** self.lanczos_exp
        lanczosy = np.sinc(ky / np.pi) ** self.lanczos_exp

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


    def _calculate_stress_field(self, Ftfx: np.ndarray, Ftfy: np.ndarray,
                                lanczosx: np.ndarray, lanczosy: np.ndarray,
                                grid_mat: np.ndarray, u: np.ndarray,
                                i_max: int, j_max: int):
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


        return pos, vec, f




