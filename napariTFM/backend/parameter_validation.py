from typing import Tuple

from napariTFM.backend.parameter_dataclasses import FTTCParameters


def validate_fttc_parameters(params: FTTCParameters) -> Tuple[bool, str]:
    """Validate only the parameters the traction solve actually consumes.

    This is the pre-compute gate for ``calculate_force_field`` /
    ``find_optimal_regularization``, so it checks compute-critical inputs only.
    Two deliberate exclusions:

    * **Visualization-only knobs** (``force_arrow_scale``, ``f_max``,
      ``force_vector_stride``) never enter the solve — they drive arrow rendering
      and the colormap ceiling — so a bad value there must not block a force
      computation. They are clamped/validated at the rendering layer.
    * **``regularization``** is checked only when it is used. Under
      ``auto_gcv=True`` the manual value is ignored (GCV supplies lambda), so
      requiring it > 0 would spuriously fail an otherwise-valid auto-GCV run.
    """
    if params.young_modulus <= 0:
        return False, "Young's modulus must be positive"

    if not 0 <= params.poisson_ratio_substrate <= 0.5:
        return False, "Poisson ratio must be between 0 and 0.5"

    if params.gel_height is not None and params.gel_height < 0:
        return False, "Gel height must be non-negative or None (infinite)"

    if params.lanczos_exp < 0:
        return False, "Lanczos exponent must be non-negative"

    if not params.auto_gcv and params.regularization <= 0:
        return False, "Regularization parameter must be positive"

    if params.frame_interval <= 0:
        return False, "Frame interval must be positive"

    if params.pixel_size <= 0:
        return False, "Pixel size must be positive"

    if params.downscale_factor < 1:
        return False, "Downscale factor must be at least 1"

    return True, ""
