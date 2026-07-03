from dataclasses import dataclass, fields
from typing import Optional, Type, TypeVar

import numpy as np

_T = TypeVar("_T")


@dataclass
class PreprocessingParameters:
    """Parameters for image preprocessing"""
    min_intensity_percentile: float = 0.0
    max_intensity_percentile: float = 100
    gaussian_sigma: float = 0.0
    cell_min_intensity_percentile: float = 0.0
    cell_max_intensity_percentile: float = 100
    cell_gaussian_sigma: float = 0.0
    registration_mode: str = 'translation'


@dataclass
class DisplacementParameters:
    """Parameters for displacement analysis"""
    # Farneback optical flow parameters. Names are retained for compatibility
    # with existing saved configs: nscales=levels, inner_iterations=iterations,
    # median_filtering=window size.
    nscales: int = 10
    inner_iterations: int = 10
    median_filtering: int = 9

    # Farneback internals (defaults match OpenCV's typical values; previously
    # hardcoded as DisplacementAnalyzer constants). use_gaussian_window selects
    # the OPTFLOW_FARNEBACK_GAUSSIAN flag (Gaussian vs. box windowing).
    pyr_scale: float = 0.5
    poly_n: int = 5
    poly_sigma: float = 1.2
    # Gaussian (not box) windowing by default: a box window makes the flow
    # piecewise-constant over its winsize footprint, tiling sparse-bead fields
    # into blocks. The Gaussian window weights the neighbourhood smoothly.
    use_gaussian_window: bool = True

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

    # Preprocessing parameters
    min_intensity_percentile: float = 0.0
    max_intensity_percentile: float = 100.0
    gaussian_sigma: float = 0.0
    cell_min_intensity_percentile: float = 0.0
    cell_max_intensity_percentile: float = 100.0
    cell_gaussian_sigma: float = 0.0
    registration_mode: str = 'translation'

    # Displacement parameters
    nscales: int = 10
    inner_iterations: int = 10
    median_filtering: int = 9
    pyr_scale: float = 0.5
    poly_n: int = 5
    poly_sigma: float = 1.2
    use_gaussian_window: bool = True  # Gaussian window avoids box-window block tiling
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

    def to_preprocessing_parameters(self) -> PreprocessingParameters:
        """Create PreprocessingParameters from unified parameters"""
        return self._project(PreprocessingParameters)

    def to_displacement_parameters(self) -> DisplacementParameters:
        """Create DisplacementParameters from unified parameters"""
        return self._project(DisplacementParameters)

    def to_fttc_parameters(self) -> FTTCParameters:
        """Create FTTCParameters from unified parameters"""
        return self._project(FTTCParameters)

    def to_stress_parameters(self) -> StressParameters:
        """Create StressParameters from unified parameters"""
        return self._project(StressParameters)
