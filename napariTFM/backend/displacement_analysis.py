from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import numpy as np

from napariTFM.backend._displacement_base import (
    BaseDisplacementAnalyzer,
    autotune_displacement_parameters,
    resolve_gpu_device,
)
from napariTFM.backend.parameter_dataclasses import DisplacementParameters
from napariTFM.backend.ffd_displacement import FFDDisplacementAnalyzer
from napariTFM.backend.ilk_displacement import ILKDisplacementAnalyzer
from napariTFM.backend.piv_displacement import PIVDisplacementAnalyzer
from napariTFM.backend.registration import (
    apply_drift,
    drift_fingerprint,
    estimate_drift,
    load_drift_csv,
    mask_region,
    mask_weight,
    save_drift_csv,
    valid_region,
)

# Map the UI method label (stored verbatim in DisplacementParameters.disp_method) to
# its analyzer. Each analyzer honours DisplacementParameters.disp_device the same way:
# a trusted CPU reference by default, the torch GPU port when available/selected.
_ANALYZERS = {
    "PIV": PIVDisplacementAnalyzer,
    "Lucas-Kanade": ILKDisplacementAnalyzer,
    "FFD": FFDDisplacementAnalyzer,
}


def _bin_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a 2D image (or weight) over ``factor x factor`` tiles.

    The image-side analogue of ``BaseDisplacementAnalyzer.downscale_flow``: same
    trimming (remainder rows/cols beyond the last whole tile are dropped) so a crop
    binned here lines up tile-for-tile with the coarse output grid. ``factor <= 1``
    returns the input unchanged.
    """
    if factor <= 1:
        return image
    h, w = image.shape[:2]
    nh, nw = h // factor, w // factor
    trimmed = image[:nh * factor, :nw * factor]
    return trimmed.reshape(nh, factor, nw, factor).mean(axis=(1, 3)).astype(np.float32)


def build_analyzer(params: DisplacementParameters):
    """Construct the displacement analyzer selected by ``params.disp_method``.

    Raises ``ValueError`` on an unknown method label. FFD raises ``RuntimeError`` at
    construction when no CUDA GPU is available (it is GPU-only).
    """
    try:
        cls = _ANALYZERS[params.disp_method]
    except KeyError:
        raise ValueError(
            f"Unknown displacement method {params.disp_method!r}; "
            f"expected one of {sorted(_ANALYZERS)}."
        )
    return cls(params)


@dataclass
class DisplacementResult:
    """Results from displacement field calculation."""

    displacement_field: np.ndarray  # Shape (t, y, x, 2) for time series, units in µm
    original_shape: tuple  # Original image shape (y, x)
    displacement_field_shape: tuple  # Displacement field shape (y, x)
    parameters: DisplacementParameters
    physical_scale: dict  # Dictionary containing physical scaling information
    # Per-frame bulk translation (stage drift) of each target frame relative to
    # the anchor (first frame), shape (t, 2) in pixels, ordered [u_x, u_y] to
    # match displacement_field's last axis. Estimated by the registration step
    # that removed it before measurement; retained so the same shift registers
    # the cell channel into the anchor frame for overlays.
    drift_pixels: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


def validate_displacement_image(image: np.ndarray) -> Tuple[bool, str]:
    """Validate input image data for displacement analysis."""
    if image is None:
        return False, "No image data provided"

    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"

    if image.ndim not in (2, 3):
        return False, "Image must be 2D or 3D (time series)"

    if np.all(np.isnan(image)):
        return False, "Image contains only NaN values"

    return True, ""


def _registration_ops(anchor: np.ndarray, params: DisplacementParameters):
    """Resolve the per-frame registration ops ``(estimate, apply)`` for this run's device.

    Reads the shared ``disp_device`` knob through the exact same
    :func:`resolve_gpu_device` the displacement backends use, so ``"cpu"``/``"auto"``/
    ``"cuda"`` mean the same thing here as for the method (and ``"cuda"`` on a machine
    with no GPU raises identically, rather than silently degrading). On a resolved
    CUDA device both the drift estimate and the drift-undoing resample run on the GPU
    via :mod:`napariTFM.backend._registration_torch`: phase cross-correlation that is
    bit-identical to the scikit-image reference (so results are unchanged) but ~17x
    faster at 2048^2 by caching the fixed anchor's spectrum, plus a ``grid_sample``
    resample. Otherwise it returns the scikit-image reference functions.

    ``estimate(image) -> (u_x, u_y)`` is bound to ``anchor``; ``apply(image, drift)``
    resamples into the anchor frame. Both take/return numpy, matching the CPU path.
    """
    device = resolve_gpu_device(str(params.disp_device), method="registration")
    if device is not None:
        from napariTFM.backend._registration_torch import (
            TorchDriftEstimator, apply_drift_torch,
        )
        estimator = TorchDriftEstimator(anchor, device=device)
        return (estimator,
                lambda image, drift: apply_drift_torch(image, drift, device=device, order=3))
    return (lambda image: estimate_drift(anchor, image), apply_drift)


def calculate_displacement_field(
    reference: np.ndarray,
    target: np.ndarray,
    params: DisplacementParameters,
    analyzer: Optional[BaseDisplacementAnalyzer] = None,
    drift_cache: Optional[Union[str, Path]] = None,
    mask: Optional[np.ndarray] = None,
    tune_parameters: bool = False,
) -> Generator[Tuple[np.ndarray, int, int], None, DisplacementResult]:
    """Calculate the displacement field between a reference image and target image(s).

    Dispatches to the backend selected by ``params.disp_method`` (PIV, Lucas-Kanade,
    or FFD) via :func:`build_analyzer`. By default, the backend measures the
    reference/target pair directly so a uniform displacement component remains in
    the returned field. If ``params.disp_remove_stage_drift`` is true, it first
    rigidly registers the reference and every target frame to a common anchor (the
    first target frame) with parameter-free phase cross-correlation, so the
    displacement method sees deformation after bulk stage drift removal. In that
    opt-in mode the per-frame drift is recorded in ``drift_pixels`` and reused to
    register the cell channel for overlays.

    Registering by a shift fabricates a border strip (edge replication), so the
    measurement is confined to the interior region every registered frame still
    fills with real, co-observed data (:func:`registration.valid_region`); the
    excluded border is returned as zero displacement. The output field keeps the
    full frame shape regardless.

    ``mask`` (optional, ``(t, y, x)`` or ``(y, x)``, foreground = ``> 0``) confines
    each frame's measurement to its cell + ``params.disp_mask_margin_um`` when
    ``params.disp_mask_confine`` is set. Two layers: the method runs on the mask's
    bounding box plus margin (:func:`registration.mask_region`) -- fewer pixels, so
    faster -- and the output is zeroed outside the *cell footprint grown by the
    margin* (:func:`registration.mask_weight`), the literal mask rather than its
    rectangle. For FFD that same foreground weight also masks the LNCC/MSE loss, so
    the fit ignores the empty, vignette-corrupted background instead of letting the
    box corners pull the control grid. Off by default; when off, or when no mask is
    given, behaviour is identical to the full-frame path.

    ``drift_cache`` (optional) is used only when ``disp_remove_stage_drift`` is
    true. It is a CSV path the estimated drift is persisted to and read back from.
    Registration drift depends only on the reference and the bead frames, not on the
    displacement method or its knobs, so a tune-loop that re-runs the same folder
    can skip re-estimation. The cache is keyed by a content fingerprint of the
    inputs (see :func:`registration.drift_fingerprint`): any change to the images
    fails safe to a fresh estimate. The resample still runs every frame, so the
    cache saves the estimate, not the whole registration step.

    ``tune_parameters`` is reserved for the explicit UI "tune current frame"
    action. When true, each frame's crop runs a small candidate grid that tunes
    only the selected backend's declared convergence and smoothing parameters,
    then returns the chosen field and diagnostics. Normal preview/run calls leave
    it false so parameters are exactly the visible manual values.

    Yields per-frame displacement fields in physical units with 1-based frame
    progress, and returns a complete :class:`DisplacementResult` when exhausted.
    """
    is_valid, error_msg = validate_displacement_image(reference)
    if not is_valid:
        raise ValueError(f"Invalid reference image: {error_msg}")

    is_valid, error_msg = validate_displacement_image(target)
    if not is_valid:
        raise ValueError(f"Invalid target image: {error_msg}")

    if target.ndim == 2:
        target = target[np.newaxis, ...]

    analyzer = analyzer or build_analyzer(params)
    total_frames = target.shape[0]

    if params.downscale_factor > 1:
        displacement_field_shape = (
            total_frames,
            target.shape[1] // params.downscale_factor,
            target.shape[2] // params.downscale_factor,
            2,
        )
    else:
        displacement_field_shape = (total_frames, target.shape[1], target.shape[2], 2)

    displacement_field_stack = np.zeros(displacement_field_shape, dtype=np.float32)
    drift_pixels = np.zeros((total_frames, 2), dtype=np.float32)
    tuning_frames = []

    frame_shape = (target.shape[1], target.shape[2])
    remove_stage_drift = bool(getattr(params, "disp_remove_stage_drift", False))

    if remove_stage_drift:
        # Register everything to the first target frame (the anchor) so bulk stage
        # drift is gone before the displacement method runs. Estimate every bulk
        # drift first (reference + all frames): the whole set is needed up front to
        # fix the common measurement region. drift_pixels[frame] is that frame's
        # bulk (u_x, u_y) drift relative to the anchor, reused to move the cell
        # channel into the same frame downstream (batch_analysis._cells_for_overlay).
        # FTTC nulls the DC mode, so the shared anchor leaves traction unchanged.
        anchor = target[0]
        # GPU phase-correlation + grid_sample resample when the device is CUDA
        # (drift bit-identical to the scikit-image reference), else the CPU
        # reference. Bound to the fixed anchor so its spectrum is computed once for
        # the whole stack.
        estimate, apply = _registration_ops(anchor, params)

        # Reuse a persisted drift estimate when the inputs are unchanged (a
        # tune-loop re-running the same folder), else estimate and persist. The
        # fingerprint keys the cache to the exact reference + bead frames, so a
        # swapped/edited dataset fails safe to recomputation rather than silently
        # reusing stale drift.
        cached = None
        fingerprint = None
        if drift_cache is not None:
            fingerprint = drift_fingerprint(reference, target)
            cached = load_drift_csv(drift_cache, total_frames, fingerprint)

        if cached is not None:
            ref_drift, drift_pixels = cached
        else:
            ref_drift = estimate(reference)
            for frame in range(total_frames):
                drift_pixels[frame] = estimate(target[frame])
            if drift_cache is not None:
                save_drift_csv(drift_cache, ref_drift, drift_pixels, fingerprint)

        # Registering by a shift fabricates a border strip (edge replication). Crop
        # every registered frame to the interior box that ALL frames still fill with
        # real, co-observed data, so no method measures a fabricated border. The
        # measurement runs on the crop and is re-embedded into a full-size zero
        # field, so the excluded border reads as no motion and the output shape is
        # unchanged.
        valid = valid_region(ref_drift, drift_pixels, frame_shape)
        reference_registered_full = apply(reference, ref_drift)
    else:
        ref_drift = np.zeros(2, dtype=np.float32)
        valid = (0, frame_shape[0], 0, frame_shape[1])
        reference_registered_full = np.asarray(reference)
    # Mask confinement (opt-in via disp_mask_confine, and only if a mask is given):
    # margin in px from the µm knob. A mask whose frame axis matches is indexed per
    # frame; a 2D mask is reused for every frame.
    confine = mask is not None and getattr(params, "disp_mask_confine", False)
    margin_px = (params.disp_mask_margin_um / params.pixel_size
                 if params.pixel_size else float("inf"))
    # The reference is constant across frames; register it once in drift-removal
    # mode, then slice each frame's own box out of it (the box follows a migrating
    # cell frame-to-frame). In direct mode this is the unmodified reference.

    # Downsample-before vs downsample-after: with the flag on (and an actual factor),
    # bin the registered images and measure on 1/df^2 the pixels -- faster, and on
    # real data within ~0.06 px of the full-res-then-bin path. Registration already
    # ran at full resolution, so the coarsening only touches the measurement.
    df = int(params.downscale_factor)
    downscale_before = df > 1 and getattr(params, "disp_downscale_before", False)

    for frame in range(total_frames):
        if remove_stage_drift:
            frame_registered_full = apply(target[frame], drift_pixels[frame])
        else:
            frame_registered_full = target[frame]
        mask_frame = (mask[frame] if mask.ndim == 3 else mask) if confine else None
        r0, r1, c0, c1 = mask_region(
            mask_frame, drift_pixels[frame], margin_px, valid, frame_shape
        )
        # Foreground weight over the crop: confines FFD's loss to the cell + margin
        # (its loss ignores the rest -- the bounding box's empty, vignette-corrupted
        # corners no longer pull the control grid) and zeros the output outside it,
        # so the result is the literal mask region, not just the bounding box.
        weight = (mask_weight(mask_frame, margin_px, (r0, r1, c0, c1))
                  if mask_frame is not None else None)
        ref_crop = reference_registered_full[r0:r1, c0:c1]
        mov_crop = frame_registered_full[r0:r1, c0:c1]

        if downscale_before and min(ref_crop.shape[:2]) >= df:
            # Bin the images first; the coarse field IS the output grid (nothing to
            # downscale afterwards). calculate_flow returns coarse-px displacement, so
            # scale by df to express it in full-res px -- matching the other path's
            # units -- then drop it onto the coarse grid at the crop's floored origin.
            w_small = _bin_image(weight, df) if weight is not None else None
            ref_small = _bin_image(ref_crop, df)
            mov_small = _bin_image(mov_crop, df)
            if tune_parameters:
                field_small, _, diag = autotune_displacement_parameters(
                    analyzer, ref_small, mov_small, weight=w_small,
                    smoothing_selector=getattr(params, "disp_tune_selector", "L-curve"),
                )
                diag.update({"frame": frame + 1, "downscale_before": True})
                tuning_frames.append(diag)
            else:
                field_small = analyzer.calculate_flow(ref_small, mov_small, weight=w_small)
            field_small = field_small * df
            if w_small is not None:
                field_small = field_small * w_small[..., None]
            coarse = np.zeros(
                (target.shape[1] // df, target.shape[2] // df, 2), dtype=np.float32
            )
            rs, cs = r0 // df, c0 // df
            hs = min(field_small.shape[0], coarse.shape[0] - rs)
            ws = min(field_small.shape[1], coarse.shape[1] - cs)
            coarse[rs:rs + hs, cs:cs + ws] = field_small[:hs, :ws]
            displacement_field_stack[frame] = coarse * params.pixel_size
            yield displacement_field_stack[frame].copy(), frame + 1, total_frames
            continue

        if tune_parameters:
            field_crop, _, diag = autotune_displacement_parameters(
                analyzer, ref_crop, mov_crop, weight=weight,
                smoothing_selector=getattr(params, "disp_tune_selector", "L-curve"),
            )
            diag.update({"frame": frame + 1, "downscale_before": False})
            tuning_frames.append(diag)
        else:
            field_crop = analyzer.calculate_flow(ref_crop, mov_crop, weight=weight)
        if weight is not None:
            field_crop = field_crop * weight[..., None]

        displacement_field_pixels = np.zeros(
            (target.shape[1], target.shape[2], 2), dtype=np.float32
        )
        displacement_field_pixels[r0:r1, c0:c1] = field_crop

        if params.downscale_factor > 1:
            displacement_field_pixels = analyzer.downscale_flow(
                displacement_field_pixels,
                params.downscale_factor,
            )

        displacement_field_stack[frame] = displacement_field_pixels * params.pixel_size

        yield displacement_field_stack[frame].copy(), frame + 1, total_frames

    physical_scale = {
        "pixel_size": params.pixel_size,
        "grid_spacing": params.pixel_size * params.downscale_factor,
        "time_interval": params.frame_interval,
        "displacement_units": "µm",
        "grid_spacing_units": "µm",
        "time_interval_units": "min",
    }

    metadata = {}
    if tuning_frames:
        metadata["parameter_tuning"] = {
            "mode": "current_frame_button" if total_frames == 1 else "per_frame",
            "frames": tuning_frames,
        }

    return DisplacementResult(
        displacement_field=displacement_field_stack,
        original_shape=reference.shape,
        displacement_field_shape=displacement_field_stack.shape[1:3],
        parameters=params,
        physical_scale=physical_scale,
        drift_pixels=drift_pixels,
        metadata=metadata,
    )
