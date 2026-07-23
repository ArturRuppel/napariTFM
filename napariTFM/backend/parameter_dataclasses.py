from dataclasses import dataclass, fields
from typing import Optional, Type, TypeVar

import numpy as np

_T = TypeVar("_T")


@dataclass
class DisplacementParameters:
    """Parameters for displacement analysis.

    Three interchangeable backends selected by ``disp_method``. PIV is a single torch
    implementation run on CPU or CUDA; iLK has a scikit-image CPU path and a torch GPU
    port; FFD is GPU-only. The device is chosen by the shared ``disp_device``. See
    napariTFM/backend/{piv,ilk,ffd}_displacement.py.
    """
    # Method + shared device selector.
    disp_method: str = "PIV"      # "PIV" | "Lucas-Kanade" | "FFD"
    disp_device: str = "auto"     # "auto" | "cuda" | "cpu" (shared by all methods)

    # PIV (multi-pass FFT cross-correlation): one torch backend, CPU or CUDA.
    piv_window: int = 24          # final interrogation window (px); heuristic-sweep default (24-32)
    piv_overlap: float = 0.75     # window overlap fraction [0, 1)
    piv_passes: int = 8           # coarse->fine window-deformation passes
    piv_smooth: float = 1.0       # per-pass Gaussian sigma (sparse-grid cells) on the vector
                                  # field; regularizes noise. 0 = off (raw, sharper but rougher)

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
    # Where the downscale_factor coarsening happens relative to the measurement.
    # False (default): measure at full resolution, then block-average the vector
    # field down to the grid (accurate -- uses all bead texture). True: block-average
    # the *images* first and measure on 1/downscale_factor^2 the pixels (faster, and
    # on real data within ~0.06 px of the full-res result). Registration always runs
    # at full resolution regardless. No-op when downscale_factor == 1.
    disp_downscale_before: bool = False
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

    # Processing parameters
    # Traction-inversion method (the explicit selector, superseding the old sentinel ladder):
    #   "FTTC"        -> Fourier Tikhonov (manual λ, or per-frame GCV when auto_gcv)
    #   "Bayesian L2" -> real-space evidence-max reconstruction (napariTFM.backend.bayesian_l2)
    #   "Elastic net" -> group-L1 (+optional l2_ridge, +mask soft-support) (forward_l1)
    #   "auto"        -> infer from the numeric flags exactly as the pre-selector code did
    #                    (l1_sparsity>0 -> Elastic net; bayesian_l2 -> Bayesian L2;
    #                     else FTTC). See napariTFM.backend.fttc.
    force_method: str = "auto"
    regularization: float = 1e-4        # manual Tikhonov λ (the override for Bayesian L2)
    # GCV auto-λ on the plain-FTTC (Fourier) path: pick the Tikhonov λ per frame by
    # Generalized Cross-Validation instead of the manual value. The one-shot button fills
    # `regularization` (same Fourier operator); this flag re-picks it every frame. Distinct
    # from Bayesian L2, which is a different (real-space) solver. See napariTFM.backend.fttc.
    auto_gcv: bool = False
    # Bayesian evidence-maximizing choice of the Tikhonov λ (Huang et al. 2019), the
    # automatic, noise-robust selector on the plain-FTTC path. Takes precedence over the
    # manual λ. BL2 (noise measured from the cell exterior) when a mask is loaded, ABL2
    # (noise inferred) otherwise. See napariTFM.backend.bayesian_l2.
    bayesian_l2: bool = False
    # Frozen Bayesian-L2 ridge λ (= α/β), estimated on one representative frame (the auto-λ
    # button) and reused across every frame for comparability -- the forward operator is identical
    # across frames, so one λ transfers exactly. None re-infers per frame, which drifts λ frame to
    # frame and breaks cross-frame comparison. See napariTFM.backend.bayesian_l2.
    bayesian_lambda: Optional[float] = None
    # Bayesian-L2 λ mode: True re-infers λ every frame (parameter-free, the default); False
    # reuses the frozen `bayesian_lambda` for comparability. The freeze button sets both.
    bayesian_per_frame: bool = True
    pixel_size: float = 0.1  # in µm
    downscale_factor: int = 4

    # Post-hoc force mask clipping. When fwd_mask_strength > 0 and a mask is supplied,
    # the selected inversion runs normally and then every force vector outside
    # mask + fwd_mask_reach is set to zero. No soft support, smoothing, or solver
    # dispatch is attached to these fields.
    fwd_mask_strength: float = 0.0        # 0 disables clipping; >0 enables post-hoc clipping
    fwd_mask_reach: float = 2.0           # force-grid px radius added around the mask before clipping
    fwd_fit_margin_um: float = 1e6        # trust displacement only within mask+margin (µm)
    fwd_max_iter: int = 200               # max CG iterations (β>0 iterative path)
    fwd_cg_tol: float = 1e-8              # CG relative-residual tolerance (β>0 path)
    fwd_traction_scale: float = 1e-2      # non-dim traction scale T0 (rarely touched)
    fwd_device: str = "auto"              # "auto" | "cuda" | "cpu" (β>0 path; cuda ⇒ cupy)
    fwd_dtype: str = "float32"            # "float32" (default; complex128 is throttled on
    #                                       laptop GPUs) | "float64" (the QP is convex &
    #                                       well-conditioned, so float32 is ample)

    # Sparse (group-L1) inversion (napariTFM.backend.forward_l1). Selected when
    # l1_sparsity > 0, ahead of the plain-FTTC path. Regularizes with an
    # L1 sparsity prior instead of L2: it thresholds rather than spreads, so it wins
    # in-cell accuracy and peak recovery over FTTC and needs NO mask. Mask clipping,
    # when enabled, happens after the solver.
    l1_sparsity: float = 0.05  # 0..1 fraction of λ₁_max (0 = off → FTTC); heuristic-sweep default:
    #                            flat basin 0.02..0.11, lean low ("err low"); ↑ with noise
    l2_ridge: float = 0.0      # elastic-net L2 ridge, fraction of median per-mode curvature; 0 = pure L1
    l1_max_iter: int = 400     # FISTA iteration budget

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
    piv_window: int = 24          # heuristic-sweep default (24-32)
    piv_overlap: float = 0.75
    piv_passes: int = 8
    piv_smooth: float = 1.0       # per-pass Gaussian sigma on the vector field (0 = off)
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
    disp_downscale_before: bool = False  # bin images before measuring (fast) vs bin the field after (accurate)
    disp_vector_stride: int = 20
    disp_arrow_scale: float = 1.0
    d_max: float = 1.0  # µm

    # Force parameters
    young_modulus: float = 5000  # Pa
    poisson_ratio_substrate: float = 0.5
    gel_height: Optional[float] = None
    # Traction-inversion method selector (see FTTCParameters for the value contract). "auto"
    # infers from the numeric flags for backward compatibility; the UI writes a concrete value.
    force_method: str = "auto"
    regularization: float = 1e-4
    # GCV auto-λ on the Fourier FTTC path (see FTTCParameters); per-frame λ selection that
    # fills the same `regularization` the manual slider sets.
    auto_gcv: bool = False
    # Bayesian evidence-max λ selection on the plain-FTTC path (see FTTCParameters /
    # napariTFM.backend.bayesian_l2); precedence over the manual λ.
    bayesian_l2: bool = False
    # Frozen Bayesian-L2 ridge λ estimated once and reused across frames (see FTTCParameters).
    bayesian_lambda: Optional[float] = None
    # Bayesian-L2 per-frame vs frozen λ (see FTTCParameters).
    bayesian_per_frame: bool = True
    # Post-hoc force mask clipping (see FTTCParameters).
    fwd_mask_strength: float = 0.0
    fwd_mask_reach: float = 2.0        # force-grid px radius added around the mask before clipping
    fwd_fit_margin_um: float = 1e6
    fwd_max_iter: int = 200
    fwd_cg_tol: float = 1e-8
    fwd_traction_scale: float = 1e-2
    fwd_device: str = "auto"
    fwd_dtype: str = "float32"
    # Sparse (group-L1) inversion (napariTFM.backend.forward_l1); selected when
    # l1_sparsity > 0, ahead of the confined/FTTC paths. See FTTCParameters.
    l1_sparsity: float = 0.05     # heuristic-sweep default (group-L1 is the recommended engine)
    l2_ridge: float = 0.0
    l1_max_iter: int = 400
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
