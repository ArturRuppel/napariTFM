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

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

from scipy import optimize
import numpy as np

from napariTFM.backend.fttc_numba_functions import calculate_traction_2d, blkmul_adj
from napariTFM.backend.parameter_dataclasses import FTTCParameters
from napariTFM.backend.parameter_validation import validate_fttc_parameters


@dataclass
class FTTCResult:
    """Results from FTTC force calculation."""
    force_field: np.ndarray
    original_shape: tuple
    force_shape: tuple
    parameters: FTTCParameters
    physical_scale: dict


def validate_displacement_field(displacement_field: np.ndarray) -> Tuple[bool, str]:
    """Validate displacement field data format and values."""
    if displacement_field is None:
        return False, "No displacement field data provided"

    if not isinstance(displacement_field, np.ndarray):
        return False, "Displacement field must be a numpy array"

    if displacement_field.ndim not in (3, 4):
        return False, "Displacement field must be 3D (y,x,2) or 4D (t,y,x,2)"

    if displacement_field.shape[-1] != 2:
        return False, f"Last dimension must be 2 (x,y components), got {displacement_field.shape[-1]}"

    if np.all(np.isnan(displacement_field)):
        return False, "Displacement field contains only NaN values"

    return True, ""


def _mask_frame_for_grid(mask: np.ndarray, frame: int, hw: Tuple[int, int]) -> np.ndarray:
    """Pick the frame's 2D mask and resize (nearest) it to the force grid ``hw``.

    ``mask`` may be 2D (shared across frames) or 3D ``(T, H, W)``; masks are an
    external input at bead resolution, so they are resampled to the (downscaled)
    force grid here, matching how the stress stage fits its mask to the force field.
    """
    m = mask[frame] if mask.ndim > 2 else mask
    m = np.asarray(m) > 0
    if m.shape != tuple(hw):
        from skimage.transform import resize
        m = resize(m.astype(float), hw, order=0, mode="edge",
                   anti_aliasing=False) > 0.5
    return m


def calculate_force_field(
        displacement_field: np.ndarray,
        params: FTTCParameters,
        mask: Optional[np.ndarray] = None,
) -> Generator[Tuple[np.ndarray, int, int], None, FTTCResult]:
    """Calculate traction forces from displacement field data.

    Three inversions, selected by dial in priority order:

    * ``l1_sparsity > 0`` → the sparse group-L1 solver
      (:mod:`napariTFM.backend.forward_l1`). Regularizes with an L1 sparsity prior
      (thresholds rather than spreads); needs no mask, and a ``mask`` if supplied is
      used as a *soft* support (``fwd_mask_strength`` sets an off-mask L2 penalty in
      the objective, ramped over a collar — no hard cliff). The recommended default
      (best in-cell accuracy + peak recovery on the force benchmark).
    * else ``fwd_mask_strength > 0`` with a ``mask`` → the confined forward solver
      (:mod:`napariTFM.backend.forward_tfm`), L2 + smoothness confined to the mask,
      sharing ``regularization`` as its Tikhonov λ.
    * else → plain FTTC (regularized Fourier inversion + Lanczos). Its Tikhonov λ is
      the manual ``regularization``, or — if ``bayesian_l2`` — chosen automatically by
      Bayesian evidence maximization (:mod:`napariTFM.backend.bayesian_l2`; BL2 when a
      ``mask`` gives the cell exterior for the noise estimate, ABL2 otherwise), or — if
      ``auto_gcv`` — by GCV. ``bayesian_l2`` takes precedence over ``auto_gcv``.

    ``mask`` is consumed on the L1 and confined paths, and on the plain-FTTC path only
    for the BL2 noise estimate.
    """
    is_valid, error_msg = validate_fttc_parameters(params)
    if not is_valid:
        raise ValueError(error_msg)

    is_valid, error_msg = validate_displacement_field(displacement_field)
    if not is_valid:
        raise ValueError(error_msg)

    if displacement_field.ndim == 3:
        displacement_field = displacement_field[np.newaxis, ...]

    total_frames = displacement_field.shape[0]
    force_shape = displacement_field.shape[1:4]
    force_stack = np.zeros((total_frames, *force_shape), dtype=np.float32)
    use_l1 = params.l1_sparsity > 0
    use_forward = (not use_l1) and params.fwd_mask_strength > 0 and mask is not None
    calculator = None if (use_l1 or use_forward) else FTTC(params)

    for frame in range(total_frames):
        if use_l1:
            from napariTFM.backend.forward_l1 import l1_traction_frame
            m = None if mask is None else _mask_frame_for_grid(mask, frame, force_shape[:2])
            traction = l1_traction_frame(displacement_field[frame], params, mask=m)
            force_stack[frame, ..., 0] = traction[0]
            force_stack[frame, ..., 1] = traction[1]
        elif use_forward:
            from napariTFM.backend.forward_tfm import forward_traction_frame
            m = None if mask is None else _mask_frame_for_grid(mask, frame, force_shape[:2])
            traction = forward_traction_frame(displacement_field[frame], params, mask=m)
            force_stack[frame, ..., 0] = traction[0]
            force_stack[frame, ..., 1] = traction[1]
        else:
            use_bayes = getattr(params, "bayesian_l2", False)
            noise_var = None
            if use_bayes:
                # BL2: measure the noise level from the displacement — from the cell-free
                # exterior when a mask is loaded, else a global high-pass estimate. Only
                # if that fails (None) does the solver infer the noise itself (ABL2).
                from napariTFM.backend.bayesian_l2 import estimate_noise_variance
                m = (_mask_frame_for_grid(mask, frame, force_shape[:2])
                     if mask is not None else None)
                noise_var = estimate_noise_variance(displacement_field[frame], m)
            # Pass the Bayesian kwargs only when actually selecting the Bayesian path, so
            # the common (manual / GCV) call keeps its original signature.
            extra = {"bayesian": True, "noise_var": noise_var} if use_bayes else {}
            result = calculator.calculate_traction(
                displacements=displacement_field[frame],
                pixel_size=params.pixel_size,
                downscale_factor=params.downscale_factor,
                regularization=None if params.auto_gcv else params.regularization,
                **extra,
            )

            force_stack[frame, ..., 0] = result[1][0]
            force_stack[frame, ..., 1] = result[1][1]

        yield force_stack[frame].copy(), frame + 1, total_frames

    physical_scale = {
        'pixel_size': params.pixel_size,
        'grid_spacing': params.pixel_size * params.downscale_factor,
        'time_interval': params.frame_interval,
        'force_units': 'Pa',
        'grid_spacing_units': 'µm',
        'time_interval_units': 'min',
    }

    return FTTCResult(
        force_field=force_stack,
        original_shape=displacement_field.shape[1:3],
        force_shape=force_stack.shape[1:3],
        parameters=params,
        physical_scale=physical_scale,
    )


def find_optimal_regularization(displacement_field: np.ndarray, params: FTTCParameters) -> float:
    """Find the GCV regularization parameter for one displacement frame."""
    is_valid, error_msg = validate_fttc_parameters(params)
    if not is_valid:
        raise ValueError(error_msg)

    is_valid, error_msg = validate_displacement_field(displacement_field)
    if not is_valid:
        raise ValueError(error_msg)

    if displacement_field.ndim != 3:
        raise ValueError("Optimal regularization requires one 3D displacement frame")

    shape = displacement_field.shape[:-1]
    pos = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy'))
    vec = np.array([displacement_field[..., 0], displacement_field[..., 1]])
    return FTTC(params)._find_regularization(
        pos, vec, params.pixel_size * params.downscale_factor, shape[1], shape[0])


class FTTC:
    def __init__(self, params: FTTCParameters):
        """Initialize FTTC calculator with substrate properties and calculation parameters.

        Args:
            params (FTTCParameters): Configuration object containing:
                - young_modulus (float): Young's modulus of the substrate in Pascals (Pa)
                - poisson_ratio_substrate (float): Poisson ratio of the substrate
                - lanczos_exp (float): Lanczos filter exponent for noise reduction
                - gel_height (float, optional): Gel height in micrometers for finite thickness
                    correction. Use None for infinite thickness.

        Example:
            >>> fttc = FTTC(FTTCParameters(
            ...     young_modulus=10000,  # 10 kPa
            ...     poisson_ratio_substrate=0.5,
            ...     lanczos_exp=2,
            ...     gel_height=None  # infinite thickness
            ... ))
        """
        self.E = params.young_modulus
        self.nu = params.poisson_ratio_substrate
        self.lanczos_exp = params.lanczos_exp
        # 0 is the UI/config sentinel for "infinite thickness" (ParameterManager
        # flattens gel_height=None to 0 when serializing to a plain dict/YAML,
        # e.g. for batch configs, but never converts it back). A gel height of
        # exactly 0 is otherwise physically meaningless, and the finite-
        # thickness correction below divides by tanh(kh), which is 0 when
        # kh == 0 -- so leaving it as 0 here would produce all-NaN forces.
        self.gel_height = params.gel_height if params.gel_height else None


    def calculate_traction(self, displacements: Tuple[np.ndarray, np.ndarray],
                           pixel_size: float,
                           downscale_factor: int = 1,
                           regularization: float = None,
                           bayesian: bool = False,
                           noise_var: Optional[float] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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

        bayesian : bool, optional (default=False)
            If True, choose λ by Bayesian evidence maximization
            (:meth:`_bayesian_regularization`) instead of the manual value or GCV. Takes
            precedence over ``regularization``.

        noise_var : float, optional (default=None)
            Per-component real-space displacement noise variance for the Bayesian path.
            Supplied (BL2) pins the noise level; None (ABL2) infers it from the data.
            Ignored unless ``bayesian`` is True.

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

        The output force field will have exactly the same dimensions as the
        input displacement field to ensure dimensional consistency.

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

        # Preserve exact input dimensions throughout the calculation
        input_height, input_width = d_x.shape

        # Create coordinate grid matching input dimensions exactly
        x = np.arange(input_width)
        y = np.arange(input_height)

        # Create position array in pixel coordinates
        pos = np.array([np.ones(len(y))[:, None] * x,
                        y[:, None] * np.ones(len(x))])

        # Create displacement vector array, already in physical units
        vec = np.array([d_x.flatten(), d_y.flatten()])

        # Convert pixel coordinates to physical units inside _perform_tfm
        forcemap_pixel_size = pixel_size * downscale_factor

        # Calculate forces with exact input dimensions preserved
        if bayesian:
            regularization = self._bayesian_regularization(
                pos, vec, forcemap_pixel_size, input_width, input_height,
                noise_var=noise_var)
        elif regularization is None:
            regularization = self._find_regularization(pos, vec, forcemap_pixel_size,
                                                       input_width, input_height)

        return self._perform_tfm(pos, vec, forcemap_pixel_size, regularization,
                                 i_max=input_width, j_max=input_height)

    def _perform_tfm(self, pos: np.ndarray, vec: np.ndarray,
                     pixelsize: float, regularization: float,
                     i_max: Optional[int] = None,
                     j_max: Optional[int] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Core TFM calculation with exact dimension preservation.

        Performs the complete Fourier Transform Traction Cytometry calculation
        while preserving the exact input dimensions throughout the pipeline.

        Args:
            pos (np.ndarray): Position array in pixel coordinates (2, N)
            vec (np.ndarray): Displacement vector array in physical units (2, N)
            pixelsize (float): Effective pixel size in meters (including any downsampling)
            regularization (float): Regularization parameter (lambda)
            i_max (int, optional): Exact output grid width dimension
            j_max (int, optional): Exact output grid height dimension

        Returns:
            Tuple containing:
            - (x, y) coordinate grids in physical units, shape (j_max, i_max) each
            - forces array in N/m², shape (2, j_max, i_max)

        Note:
            When i_max and j_max are specified, the output will have exactly
            these dimensions, preserving the input displacement field size.
            This prevents dimension mismatches in downstream processing.
        """
        # Interpolate to regular grid using exact input dimensions
        grid_mat, u, i_max, j_max, _, _ = self._interp_vec2grid(
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

        # Create coordinate grids that exactly match the input dimensions
        # No padding or cropping - dimensions are preserved exactly
        x_full = np.linspace(x[0, 0], x[-1, -1], i_max)
        y_full = np.linspace(y[0, 0], y[-1, -1], j_max)
        x, y = np.meshgrid(x_full, y_full)

        return (x, y), f

    def _calculate_greens_function(self, kx: np.ndarray, ky: np.ndarray):
        """Calculate Green's function in Fourier space with optional gel height correction.

        Implements the Boussinesq solution modified for finite gel thickness when applicable.

        Args:
            kx (np.ndarray): x-component of wave vectors
            ky (np.ndarray): y-component of wave vectors

        Returns:
            np.ndarray: Green's function tensor in Fourier space.
                Shape: 2 × 2 × H × W complex array
        """
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
                       lambdarange: np.ndarray) -> float:
        """Return the GCV-optimal regularization for the block-diagonal system."""
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

        return float(reg_min)

    def _find_regularization(self, pos0: np.ndarray, vec0: np.ndarray,
                             forcemap_pixel_size: float, input_width: int,
                             input_height: int) -> float:
        """Find optimal regularization parameter using Generalized Cross-Validation (GCV).

        Implements the method from Golub, Heath, & Wahba (2012) to automatically
        determine the Tikhonov regularization parameter while preserving exact
        input dimensions.

        Args:
            pos0 (np.ndarray): Position array in pixel coordinates (2 × N)
            vec0 (np.ndarray): Displacement vector array (2 × N)
            forcemap_pixel_size (float): Pixel size in micrometers
            input_width (int): Exact width of input displacement field
            input_height (int): Exact height of input displacement field

        Returns:
            float: Optimal regularization parameter λ that minimizes the GCV function

        Note:
            The search range is centered around λ = 0.2/E with a span of ±5 orders
            of magnitude, where E is Young's modulus. The calculation preserves
            the exact input dimensions throughout the optimization process.
        """
        lamguess = 0.2 / self.E
        lamlow = np.log10(lamguess) - 5.0
        lamhigh = np.log10(lamguess) + 5.0
        lambdarange = np.logspace(lamlow, lamhigh, 50)

        blockU, s, b = self._svd_block(pos0, vec0, forcemap_pixel_size,
                                       i_max=input_width, j_max=input_height)
        reg_min = self._gcv_blockdiag(blockU, s, b, lambdarange)
        # _gcvfun depends on lambda only via lambda**2, so the unconstrained
        # optimizer can settle on a negative root that is numerically identical
        # to its magnitude. The force path squares it, but the raw value is
        # stored as `regularization` and later hits math.log10() in the UI, which
        # rejects negatives. Return the magnitude — an exact, lossless fix.
        return abs(reg_min)

    def _bayesian_regularization(self, pos0: np.ndarray, vec0: np.ndarray,
                                 forcemap_pixel_size: float, input_width: int,
                                 input_height: int,
                                 noise_var: Optional[float] = None) -> float:
        """Find the Tikhonov λ by Bayesian evidence maximization (Huang et al. 2019).

        The Bayesian alternative to :meth:`_find_regularization`'s GCV: it maximizes the
        marginal likelihood of the displacement over the same Fourier-SVD blocks GCV
        uses (:meth:`_svd_block`), so it is a drop-in swap for the λ search. See
        :mod:`napariTFM.backend.bayesian_l2`.

        Args:
            pos0, vec0, forcemap_pixel_size, input_width, input_height: as for
                :meth:`_find_regularization`.
            noise_var: per-component real-space displacement noise variance ``σ_r²``. When
                supplied, the noise level is pinned from it (**BL2**); when ``None``, both
                the prior width and noise level are inferred from the evidence (**ABL2**).

        Returns:
            The optimal ``regularization`` λ (already square-rooted, so the force path's
            ``λ**2`` reproduces the Bayesian ridge ``α/β``).
        """
        from napariTFM.backend.bayesian_l2 import evidence_optimal_lambda

        blockU, s, b = self._svd_block(pos0, vec0, forcemap_pixel_size,
                                       i_max=input_width, j_max=input_height)
        data_coef = blkmul_adj(blockU, b)
        # Unnormalized FFT (Parseval): white real-space noise of per-component variance
        # σ_r² has per-Fourier-coefficient variance N·σ_r² with N = i_max·j_max grid
        # points, and the per-mode SVD rotation U is unitary so preserves it.
        noise_var_fourier = (None if noise_var is None
                             else float(input_width) * float(input_height) * noise_var)
        lam_ridge, _ = evidence_optimal_lambda(s, data_coef,
                                               noise_var_fourier=noise_var_fourier)
        # Stored as `regularization`; the force path squares it, so return sqrt(ridge).
        return float(np.sqrt(lam_ridge))

    def _svd_block(self, pos: np.ndarray, vec: np.ndarray, forcemap_pixel_size: float,
                   i_max: int = None, j_max: int = None):
        """Prepare SVD representation of the FTTC problem with exact dimensions.

        Creates the singular value decomposition representation needed for
        Generalized Cross-Validation regularization parameter optimization.
        Preserves exact input dimensions throughout the calculation.

        Args:
            pos (np.ndarray): Position array in pixel coordinates (2 × N)
            vec (np.ndarray): Displacement vector array (2 × N)
            forcemap_pixel_size (float): Pixel size in micrometers
            i_max (int, optional): Exact width dimension to preserve
            j_max (int, optional): Exact height dimension to preserve

        Returns:
            Tuple containing:
            - U_h (np.ndarray): Block diagonal matrix of left singular vectors
            - s_h (np.ndarray): Flattened array of singular values
            - Ftu (np.ndarray): Flattened Fourier transform of displacement field

        Note:
            When i_max and j_max are provided, these exact dimensions are
            preserved throughout the SVD calculation to ensure consistency
            with the input displacement field dimensions.
        """
        grid_mat, u, i_max, j_max, _, _ = self._interp_vec2grid(pos, vec, i_max=i_max, j_max=j_max)
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
        """Interpolate scattered displacement data to a regular grid using KD-tree.

        Implements efficient nearest-neighbor interpolation with inverse distance
        weighting for smooth results.

        Args:
            pos (np.ndarray): Position array in pixel coordinates (2 × N)
            vec (np.ndarray): Vector values to interpolate (2 × N)
            i_max (int, optional): Output grid x-dimension
            j_max (int, optional): Output grid y-dimension

        Returns:
            Tuple containing:
            - grid_mat (np.ndarray): Regular grid coordinates (2 × H × W)
            - u (np.ndarray): Interpolated values on grid (2 × H × W)
            - i_max (int): Actual x-dimension used
            - j_max (int): Actual y-dimension used
            - i_bound_size (int): Always 0 (kept for compatibility)
            - j_bound_size (int): Always 0 (kept for compatibility)

        Note:
            If i_max/j_max not provided, determines dimensions from data extent.
            Uses adaptive number of neighbors (up to 12) for interpolation.
        """
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
        # Angular wavenumbers on the FFT grid. np.fft.fftfreq gives the correct
        # frequency ordering for *both* even and odd lengths; the previous
        # hand-rolled `arange(0, n//2), arange(-n//2, 0)` matched fftfreq only for
        # even n and mislabelled the Nyquist-adjacent bin for odd n (e.g. n=5 →
        # [0,1,-3,-2,-1] instead of [0,1,2,-2,-1]), silently corrupting the
        # Green's function / Lanczos filter on odd-sized grids.
        kx_vec = 2. * np.pi / forcemap_pixel_size * np.fft.fftfreq(int(i_max))
        ky_vec = 2. * np.pi / forcemap_pixel_size * np.fft.fftfreq(int(j_max))
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
