from dataclasses import dataclass, fields
from typing import Optional, Type, TypeVar

import numpy as np

_T = TypeVar("_T")


@dataclass
class DisplacementParameters:
    """Parameters for displacement analysis.

    Three interchangeable backends selected by ``disp_method``; each has a trusted
    CPU reference (openpiv / scikit-image) and a torch GPU port used when available
    and selected by the shared ``disp_device``. FFD is GPU-only. See
    napariTFM/backend/{piv,ilk,ffd}_displacement.py.
    """
    # Method + shared device selector.
    disp_method: str = "PIV"      # "PIV" | "Lucas-Kanade" | "FFD"
    disp_device: str = "auto"     # "auto" | "cuda" | "cpu" (shared by all methods)

    # PIV (multi-pass FFT cross-correlation): openpiv CPU / torch GPU, same knobs.
    piv_window: int = 16          # final interrogation window (px)
    piv_overlap: float = 0.75     # window overlap fraction [0, 1)
    piv_passes: int = 8           # coarse->fine window-deformation passes

    # iLK (iterative Lucas-Kanade): scikit-image CPU / torch GPU, same knobs.
    ilk_radius: int = 7           # half-window of the local LK solve (px), the primary knob
    ilk_num_warp: int = 10        # coarse->fine warp iterations per pyramid level

    # FFD (grid-pyramid free-form deformation): GPU-only.
    ffd_level_spacing: float = 12.0   # finest control spacing (px) -- the bias-variance dial
    ffd_num_levels: int = 6           # DERIVED/display only: pyramid depth follows from
                                      # ffd_downscale + ffd_min_size (see pyramid_num_levels);
                                      # the backend ignores this field. Kept so recipes/UI can
                                      # surface the resulting depth.
    ffd_metric: str = "lncc"          # "lncc" | "mse" image-match objective
    ffd_num_iters: int = 50           # LBFGS iterations per pyramid level
    ffd_elastic: float = 0.0          # elastic (Navier strain-energy) regularization weight; 0 = off
    ffd_downscale: float = 2.0        # image-pyramid downscale factor per level
    ffd_min_size: int = 16            # coarsest pyramid level min dimension (px) -- with
                                      # ffd_downscale this sets the pyramid depth / capture range
    ffd_interp: str = "bicubic"       # warp interpolation: "bicubic" | "bilinear"
    ffd_early_stop: float = 0.0       # per-level LBFGS convergence tolerance; 0 = run full num_iters (current behaviour)

    # Confine the displacement measurement to the foreground mask + margin, when a
    # mask is supplied and disp_mask_confine is on: each frame is measured only
    # within the bounding box of its cell plus disp_mask_margin_um, and read as zero
    # outside. This both speeds the method (fewer pixels) and structurally excludes
    # the aperture-vignette border garbage, instead of relying on downstream masking.
    # Off by default (opt-in, like the fwd_* confinement). The margin is a physical
    # length: set it to your traction halo's decay length -- too small silently
    # clips the real substrate-displacement halo just outside the cell, so err
    # generous.
    disp_mask_confine: bool = False       # gate: confine the measurement to the mask
    disp_mask_margin_um: float = 20.0     # mask bounding-box margin (µm) when confining

    # Analysis parameters
    downscale_factor: int = 4
    pixel_size: float = 0.1
    frame_interval: float = 1

    # Visualization parameters
    d_max: float = 1
    disp_vector_stride: int = 20
    disp_arrow_scale: float = 1


@dataclass
class FTTCParameters:
    """Parameters for FTTC calculations"""
    # Material parameters
    young_modulus: float = 5000  # Pa
    poisson_ratio_substrate: float = 0.5
    gel_height: Optional[float] = None  # None for infinite thickness
    lanczos_exp: int = 1

    # Processing parameters
    regularization: float = 1e-4
    auto_gcv: bool = False
    pixel_size: float = 0.1  # in µm
    downscale_factor: int = 4

    # Traction inversion is selected by the mask-confinement dial, not a separate
    # method flag: fwd_mask_strength == 0 runs plain FTTC (regularized Fourier
    # inversion + Lanczos + GCV, using `regularization` above); > 0 (with a mask)
    # kicks off the confined forward solver (napariTFM.backend.forward_tfm), which
    # reuses that same `regularization` as its Tikhonov λ. The fwd_* fields below
    # are only read on that confined (> 0) path.
    fwd_mask_strength: float = 0.0        # 0..100 log-scaled mask confinement dial (0 = off → FTTC)
    fwd_smoothness: float = 0.05          # gradient-smoothness weight on the traction field.
    #                                       This is the PRIMARY regularizer of the iterative
    #                                       (confined) solve — it replaces the coarse B-spline
    #                                       basis the photometric one-shot used as its smoother.
    #                                       Without it, confining forces to the mask removes the
    #                                       solver's off-mask escape valve and the in-mask field
    #                                       overfits the (delocalized) displacement into garbage.
    #                                       Non-dim data term ⇒ useful band ~0.01..0.3, roughly
    #                                       scale-independent. 0 = off (reproduces the artifacts).
    fwd_fit_margin_um: float = 1e6        # trust displacement only within mask+margin (µm)
    fwd_max_iter: int = 200               # max CG iterations (β>0 iterative path)
    fwd_cg_tol: float = 1e-8              # CG relative-residual tolerance (β>0 path)
    fwd_traction_scale: float = 1e-2      # non-dim traction scale T0 (rarely touched)
    fwd_device: str = "auto"              # "auto" | "cuda" | "cpu" (β>0 path; cuda ⇒ cupy)
    fwd_dtype: str = "float32"            # "float32" (default; complex128 is throttled on
    #                                       laptop GPUs) | "float64" (the QP is convex &
    #                                       well-conditioned, so float32 is ample)

    # Time parameters
    frame_interval: float = 1  # minutes

    # Visualization parameters
    force_vector_stride: int = 20
    force_arrow_scale: float = 1.0
    f_max: float = 500.0  # Pa


@dataclass
class StressParameters:
    """Parameters for stress calculations (BISM, Bayesian, mesh-free)."""
    # The Bayesian regularization hyperparameter Lambda, trading traction-fit
    # against the stress-norm prior. Stored as the actual value; the UI exposes
    # it as a base-10 exponent.
    bism_regularization: float = 1e-6

    # Scaling parameter
    pixel_size: float = 0.1  # in µm
    downscale_factor: int = 4

    # Time parameters
    frame_interval: float = 1  # minutes

    # Visualization parameters
    max_stress: float = 1


@dataclass
class UnifiedParameters:
    """Single source of truth for all parameters"""
    # General parameters
    pixel_size: float = 0.1  # µm
    frame_interval: float = 1.0  # min

    # Displacement parameters (PIV / iLK / FFD backends; see DisplacementParameters)
    disp_method: str = "PIV"  # "PIV" | "Lucas-Kanade" | "FFD"
    disp_device: str = "auto"  # "auto" | "cuda" | "cpu" (shared by all methods)
    piv_window: int = 16
    piv_overlap: float = 0.75
    piv_passes: int = 8
    ilk_radius: int = 7
    ilk_num_warp: int = 10
    ffd_level_spacing: float = 12.0
    ffd_num_levels: int = 6            # derived/display only (see DisplacementParameters)
    ffd_metric: str = "lncc"
    ffd_num_iters: int = 50
    ffd_elastic: float = 0.0
    ffd_downscale: float = 2.0
    ffd_min_size: int = 16
    ffd_interp: str = "bicubic"
    ffd_early_stop: float = 0.0
    disp_mask_confine: bool = False    # confine displacement measurement to the mask
    disp_mask_margin_um: float = 20.0  # mask bounding-box margin (µm) when confining
    downscale_factor: int = 4
    disp_vector_stride: int = 20
    disp_arrow_scale: float = 1.0
    d_max: float = 1.0  # µm

    # Force parameters
    young_modulus: float = 5000  # Pa
    poisson_ratio_substrate: float = 0.5
    gel_height: Optional[float] = None
    lanczos_exp: int = 1
    regularization: float = 1e-4
    auto_gcv: bool = False
    # Confined forward solver (see FTTCParameters / napariTFM.backend.forward_tfm);
    # gated by fwd_mask_strength, shares `regularization` as its Tikhonov λ.
    fwd_mask_strength: float = 0.0
    fwd_smoothness: float = 0.05
    fwd_fit_margin_um: float = 1e6
    fwd_max_iter: int = 200
    fwd_cg_tol: float = 1e-8
    fwd_traction_scale: float = 1e-2
    fwd_device: str = "auto"
    fwd_dtype: str = "float32"
    force_vector_stride: int = 20
    force_arrow_scale: float = 1.0
    f_max: float = 500.0  # Pa

    # Stress parameters (BISM, Bayesian, mesh-free)
    bism_regularization: float = 1e-6  # stored as value, UI shows 10^x
    max_stress: float = 1.0

    def _project(self, cls: Type[_T]) -> _T:
        """Build a per-stage parameter subset by field-name projection.

        Every field of each per-stage dataclass is a field of
        ``UnifiedParameters`` with the same name (enforced by
        ``tests/test_parameter_dataclasses.py``), so the subset is just the
        matching fields copied across. This replaces four hand-written
        constructors that had to be edited — and kept in lockstep with the
        default values — every time a parameter was added.
        """
        return cls(**{f.name: getattr(self, f.name) for f in fields(cls)})

    def to_displacement_parameters(self) -> DisplacementParameters:
        """Create DisplacementParameters from unified parameters"""
        return self._project(DisplacementParameters)

    def to_fttc_parameters(self) -> FTTCParameters:
        """Create FTTCParameters from unified parameters"""
        return self._project(FTTCParameters)

    def to_stress_parameters(self) -> StressParameters:
        """Create StressParameters from unified parameters"""
        return self._project(StressParameters)
