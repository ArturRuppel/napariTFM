from typing import Tuple

from napariTFM.backend.parameter_dataclasses import FTTCParameters


def validate_fttc_parameters(params: FTTCParameters) -> Tuple[bool, str]:
    """Validate only the parameters the traction solve actually consumes.

    This is the pre-compute gate for ``calculate_force_field`` /
    ``find_bayesian_regularization``, so it checks compute-critical inputs only.
    Two deliberate exclusions:

    * **Visualization-only knobs** (``force_arrow_scale``, ``f_max``,
      ``force_vector_stride``) never enter the solve — they drive arrow rendering
      and the colormap ceiling — so a bad value there must not block a force
      computation. They are clamped/validated at the rendering layer.
    * **``regularization``** is checked only when it is used. Under ``bayesian_l2=True``
      (evidence maximization) the manual value is ignored, so requiring it > 0 would
      spuriously fail an otherwise-valid automatic run.
    """
    if params.young_modulus <= 0:
        return False, "Young's modulus must be positive"

    if not 0 <= params.poisson_ratio_substrate <= 0.5:
        return False, "Poisson ratio must be between 0 and 0.5"

    if params.gel_height is not None and params.gel_height < 0:
        return False, "Gel height must be non-negative or None (infinite)"

    # The manual λ is only consumed when the automatic selector is off. Under
    # bayesian_l2 (evidence maximization) the manual value is ignored, so requiring
    # it > 0 would spuriously fail an otherwise-valid automatic run.
    if not getattr(params, "bayesian_l2", False) and params.regularization <= 0:
        return False, "Regularization parameter must be positive"

    if params.frame_interval <= 0:
        return False, "Frame interval must be positive"

    if params.pixel_size <= 0:
        return False, "Pixel size must be positive"

    if params.downscale_factor < 1:
        return False, "Downscale factor must be at least 1"

    # Confined forward solver (fwd_mask_strength > 0). Only gate the fields its
    # solve consumes; fwd_mask_strength is a 0..100 dial and fwd_fit_margin_um is
    # visualization-adjacent (it only weights the data term), so both are clamped
    # in the solver rather than gated. λ is the shared `regularization`, already
    # checked above.
    if params.fwd_mask_strength > 0:
        if getattr(params, "fwd_mask_softness", 0.0) < 0:
            return False, "Mask softness (σ) must be non-negative"
        if params.fwd_max_iter < 1:
            return False, "Forward max iterations must be at least 1"
        if str(params.fwd_device) not in ("auto", "cuda", "cpu"):
            return False, "fwd_device must be 'auto', 'cuda', or 'cpu'"
        if str(params.fwd_dtype) not in ("float64", "float32"):
            return False, "fwd_dtype must be 'float64' or 'float32'"

    # Sparse group-L1 solver (l1_sparsity > 0). The dial is a fraction of λ₁_max, so it
    # must lie in (0, 1]; it shares fwd_device/fwd_dtype with the confined solver.
    if getattr(params, "l1_sparsity", 0.0) > 0:
        if not 0.0 < params.l1_sparsity <= 1.0:
            return False, "l1_sparsity must be in (0, 1]"
        if params.l1_max_iter < 1:
            return False, "l1_max_iter must be at least 1"
        if str(params.fwd_device) not in ("auto", "cuda", "cpu"):
            return False, "fwd_device must be 'auto', 'cuda', or 'cpu'"
        if str(params.fwd_dtype) not in ("float64", "float32"):
            return False, "fwd_dtype must be 'float64' or 'float32'"

    return True, ""
