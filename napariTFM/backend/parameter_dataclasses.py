from dataclasses import dataclass, fields
from typing import Optional, Type, TypeVar

import numpy as np

_T = TypeVar("_T")


@dataclass
class DisplacementParameters:
    """Parameters for displacement analysis"""
    # PIV (multi-pass FFT cross-correlation) parameters. The backend has a
    # torch-free numpy core and is GPU-accelerated automatically when torch +
    # CUDA are available (see napariTFM/backend/piv_displacement.py).
    piv_window: int = 16          # final interrogation window (px)
    piv_overlap: float = 0.75     # window overlap fraction [0, 1)
    piv_passes: int = 8           # coarse->fine window-deformation passes
    piv_device: str = "auto"      # "auto" | "cuda" | "cpu"

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
    fwd_max_iter: int = 200               # L-BFGS iterations (β>0 iterative path)
    fwd_traction_scale: float = 1e-2      # non-dim traction scale T0 (rarely touched)
    fwd_device: str = "auto"              # "auto" | "cuda" | "cpu" (β>0 path)
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

    # Displacement parameters (PIV backend)
    piv_window: int = 16
    piv_overlap: float = 0.75
    piv_passes: int = 8
    piv_device: str = "auto"  # "auto" | "cuda" | "cpu"
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
