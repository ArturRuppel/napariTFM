from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PreprocessingParameters:
    """Parameters for image preprocessing"""
    rolling_ball_radius: int = 0
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
    use_gaussian_window: bool = False

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
class MeshParameters:
    """Parameters for mesh generation"""
    mask: np.ndarray
    density_factor: float = 0.01
    mesh_algorithm: int = 2
    use_optimization: bool = True


# MSM stress is linear in Young's modulus, so its absolute value does not affect
# the computed stress *distribution*. The cell monolayer modulus is therefore
# fixed to a constant rather than exposed as a tunable parameter.
MSM_YOUNG_MODULUS: float = 1.0


@dataclass
class MSMParameters:
    """Parameters for stress calculations (MSM or BISM engine)."""
    # Engine selector: "MSM" (FEM, material params + mesh) or "BISM" (Bayesian,
    # mesh-free, no material params). The mesh/material fields below are read
    # only by the MSM engine.
    stress_method: str = "MSM"

    # Mesh parameters
    density_factor: float = 0.01
    mesh_algorithm: str = 'Frontal-Del.'
    use_optimization: bool = True

    # Material parameters
    poisson_ratio_cells: float = 0.5
    young_modulus: float = MSM_YOUNG_MODULUS

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
    rolling_ball_radius: int = 0
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
    use_gaussian_window: bool = False
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

    # Stress parameters
    stress_method: str = "MSM"   # "MSM" (FEM) or "BISM" (Bayesian, mesh-free)
    density_factor: float = 0.01
    mesh_algorithm: str = 'Frontal-Del.'
    use_optimization: bool = True
    poisson_ratio_cells: float = 0.5
    max_stress: float = 1.0

    def to_preprocessing_parameters(self) -> PreprocessingParameters:
        """Create PreprocessingParameters from unified parameters"""
        return PreprocessingParameters(
            rolling_ball_radius=self.rolling_ball_radius,  # Add new parameter
            min_intensity_percentile=self.min_intensity_percentile,
            max_intensity_percentile=self.max_intensity_percentile,
            gaussian_sigma=self.gaussian_sigma,
            cell_min_intensity_percentile=self.cell_min_intensity_percentile,
            cell_max_intensity_percentile=self.cell_max_intensity_percentile,
            cell_gaussian_sigma=self.cell_gaussian_sigma,
            registration_mode=self.registration_mode
        )

    def to_displacement_parameters(self) -> DisplacementParameters:
        """Create DisplacementParameters from unified parameters"""
        return DisplacementParameters(
            nscales=self.nscales,
            inner_iterations=self.inner_iterations,
            median_filtering=self.median_filtering,
            pyr_scale=self.pyr_scale,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            use_gaussian_window=self.use_gaussian_window,
            downscale_factor=self.downscale_factor,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval,
            d_max=self.d_max,
            disp_vector_stride=self.disp_vector_stride,
            disp_arrow_scale=self.disp_arrow_scale
        )

    def to_fttc_parameters(self) -> FTTCParameters:
        """Create FTTCParameters from unified parameters"""
        return FTTCParameters(
            young_modulus=self.young_modulus,
            poisson_ratio_substrate=self.poisson_ratio_substrate,
            gel_height=self.gel_height,
            lanczos_exp=self.lanczos_exp,
            regularization=self.regularization,
            auto_gcv=self.auto_gcv,
            downscale_factor=self.downscale_factor,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval,
            force_vector_stride=self.force_vector_stride,
            force_arrow_scale=self.force_arrow_scale,
            f_max=self.f_max
        )

    def to_msm_parameters(self) -> MSMParameters:
        """Create MSMParameters from unified parameters"""
        return MSMParameters(
            stress_method=self.stress_method,
            density_factor=self.density_factor,
            mesh_algorithm=self.mesh_algorithm,
            use_optimization=self.use_optimization,
            poisson_ratio_cells=self.poisson_ratio_cells,
            young_modulus=MSM_YOUNG_MODULUS,
            pixel_size=self.pixel_size,
            downscale_factor=self.downscale_factor,
            frame_interval=self.frame_interval,
            max_stress=self.max_stress
        )
