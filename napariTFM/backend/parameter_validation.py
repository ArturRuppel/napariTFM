from typing import Tuple

from napariTFM.backend.parameter_dataclasses import (
    FTTCParameters,
    PreprocessingParameters,
)


def validate_preprocessing_parameters(params: PreprocessingParameters) -> Tuple[bool, str]:
    if not 0 <= params.min_intensity_percentile < params.max_intensity_percentile <= 100:
        return False, "Invalid intensity percentile range"

    if not 0 <= params.cell_min_intensity_percentile < params.cell_max_intensity_percentile <= 100:
        return False, "Invalid cell intensity percentile range"

    if params.gaussian_sigma < 0:
        return False, "Gaussian sigma must be non-negative"

    if params.cell_gaussian_sigma < 0:
        return False, "Cell gaussian sigma must be non-negative"

    if params.registration_mode not in ['translation', 'rigid', 'no registration']:
        return False, f"Invalid registration mode: {params.registration_mode}"

    return True, ""


def validate_fttc_parameters(params: FTTCParameters) -> Tuple[bool, str]:
    if params.young_modulus <= 0:
        return False, "Young's modulus must be positive"

    if not 0 <= params.poisson_ratio_substrate <= 0.5:
        return False, "Poisson ratio must be between 0 and 0.5"

    if params.gel_height is not None and params.gel_height < 0:
        return False, "Gel height must be non-negative or None (infinite)"

    if params.lanczos_exp < 0:
        return False, "Lanczos exponent must be non-negative"

    if params.regularization <= 0:
        return False, "Regularization parameter must be positive"

    if params.force_vector_stride < 1:
        return False, "Vector stride must be at least 1"

    if params.force_arrow_scale <= 0:
        return False, "Arrow scale must be positive"

    if params.f_max <= 0:
        return False, "Maximum force must be positive"

    if params.frame_interval <= 0:
        return False, "Frame interval must be positive"

    if params.pixel_size <= 0:
        return False, "Pixel size must be positive"

    if params.downscale_factor < 1:
        return False, "Downscale factor must be at least 1"

    return True, ""
