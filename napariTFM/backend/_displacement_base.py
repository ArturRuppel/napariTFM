"""Shared plumbing for the displacement backends (PIV, iLK, FFD).

Each backend is a thin analyzer with the same two-method interface the pipeline
expects: ``calculate_flow(reference, moving) -> (H, W, 2) float32`` in **pixels**
(``[..., 0] = u_x`` columns, ``[..., 1] = u_y`` rows; positive = right/down), and
``downscale_flow(flow, factor)``. Device story: PIV is a single torch
implementation run on CPU or CUDA; iLK has a scikit-image CPU path with a
numerically-equivalent torch GPU port; FFD is GPU-only. torch is a core
dependency. The device is chosen by the shared ``disp_device``.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from napariTFM.backend.parameter_dataclasses import DisplacementParameters

logger = logging.getLogger(__name__)


def resolve_gpu_device(request: str, *, method: str):
    """Resolve the shared device request to a torch CUDA device, or ``None`` for CPU.

    ``request`` is ``"auto"`` (GPU if torch+CUDA are present, else the CPU
    reference), ``"cuda"`` (require a CUDA device), or ``"cpu"`` (force the CPU
    reference). Returns a ``torch.device`` when the GPU port should run, or
    ``None`` when the CPU reference should. ``method`` only sharpens the error
    messages. Never imports torch on the ``"cpu"`` path, so a torch-free install
    stays torch-free by default.
    """
    request = (request or "auto").lower()
    if request == "cpu":
        return None
    if request == "cuda":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                f"device='cuda' needs PyTorch, which is not installed. Install the "
                f"optional GPU extra with `pip install napariTFM[gpu]`, or set "
                f"device='auto'/'cpu' to use the CPU {method} implementation."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' but no CUDA device is available; use device='auto' or 'cpu'."
            )
        return torch.device("cuda")
    # "auto": GPU when torch + CUDA are both present, else the CPU reference.
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except ImportError:
        pass
    return None


class BaseDisplacementAnalyzer:
    """Common interface + shared field post-processing for the displacement backends."""

    algorithm_name = "displacement"
    smoothing_param_name: str | None = None
    smoothing_candidates: tuple[float, ...] = ()
    convergence_param_name: str | None = None
    convergence_candidates: tuple[float, ...] = ()

    def __init__(self, params: DisplacementParameters | None = None):
        self.params = params or DisplacementParameters()

    def calculate_flow(self, reference: np.ndarray, moving: np.ndarray,
                       weight: np.ndarray | None = None) -> np.ndarray:
        """Estimate the dense displacement field for one frame pair.

        ``weight`` (optional, ``(H, W)`` in ``[0, 1]``) confines the fit to a
        foreground region: only FFD honours it (it masks its loss); PIV and iLK,
        being local estimators, accept and ignore it (their confinement is the
        upstream crop). ``None`` = fit the whole input, the default for all methods.
        """
        raise NotImplementedError

    @staticmethod
    def _pack(u_xy: np.ndarray, H: int, W: int) -> np.ndarray:
        """Pack a ``(2, H, W)`` ``[u_x, u_y]`` field into the pipeline's ``(H, W, 2)``
        float32 layout, zeroing a non-finite (degenerate/blank input) field with a warning."""
        flow = np.stack([u_xy[0], u_xy[1]], axis=-1).astype(np.float32, copy=False)
        if not np.isfinite(flow).all():
            warnings.warn(
                "Displacement backend produced a non-finite field for one frame pair "
                "(degenerate/blank input?); returning zeros for it.",
                RuntimeWarning,
            )
            flow = np.zeros((H, W, 2), dtype=np.float32)
        return flow

    def downscale_flow(self, flow: np.ndarray, factor: int) -> np.ndarray:
        """Downscale the dense flow by block-mean averaging over ``factor x factor`` tiles.

        Preserves vector magnitudes (mean, not decimation); ``factor <= 1`` returns the
        input unchanged. Remainder rows/cols beyond the last whole tile are dropped.
        """
        if factor <= 1:
            return flow
        h, w = flow.shape[:2]
        new_h, new_w = h // factor, w // factor
        trimmed = flow[:new_h * factor, :new_w * factor]
        return (
            trimmed.reshape(new_h, factor, new_w, factor, 2)
            .mean(axis=(1, 3))
            .astype(np.float64)
        )


def _resample_flow_to_shape(flow: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a ``(H, W, 2)`` field to image resolution for GT-free scoring."""
    flow = np.asarray(flow, dtype=np.float64)
    if flow.shape[:2] == tuple(shape):
        return flow
    from scipy import ndimage

    h, w = flow.shape[:2]
    target_h, target_w = int(shape[0]), int(shape[1])
    out = np.empty((target_h, target_w, 2), dtype=np.float64)
    zoom = (target_h / max(h, 1), target_w / max(w, 1))
    out[..., 0] = ndimage.zoom(flow[..., 0], zoom, order=1)
    out[..., 1] = ndimage.zoom(flow[..., 1], zoom, order=1)
    return out


def _warp_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample ``image`` at pixel coordinates ``x + flow`` with edge padding."""
    from scipy import ndimage

    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = [yy + flow[..., 1], xx + flow[..., 0]]
    return ndimage.map_coordinates(image, coords, order=1, mode="nearest")


def displacement_candidate_diagnostics(
    reference: np.ndarray,
    moving: np.ndarray,
    field: np.ndarray,
    *,
    score_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """GT-free candidate score: symmetric warp residual and field roughness.

    ``field`` may be returned on a backend-native grid; it is first resampled to
    the image resolution for both metrics. The residual is RMS(ref warped by
    ``-0.5*u`` minus moving warped by ``+0.5*u``), normalized by ``std(ref)``.
    Roughness is ``mean(|grad ux|^2 + |grad uy|^2)``. Lower is better for both.
    """
    ref = np.nan_to_num(np.asarray(reference, dtype=np.float64), nan=0.0)
    dfm = np.nan_to_num(np.asarray(moving, dtype=np.float64), nan=0.0)
    u = _resample_flow_to_shape(field, ref.shape)
    warped_ref = _warp_image(ref, -0.5 * u)
    warped_dfm = _warp_image(dfm, 0.5 * u)
    denom = max(float(np.std(ref)), 1e-12)
    sq = (warped_ref - warped_dfm) ** 2
    if score_weight is None:
        residual = float(np.sqrt(np.mean(sq)) / denom)
        score_pixels = int(sq.size)
    else:
        w = np.nan_to_num(np.asarray(score_weight, dtype=np.float64), nan=0.0)
        if w.shape != sq.shape:
            raise ValueError(
                f"score_weight shape {w.shape} does not match image shape {sq.shape}"
            )
        total = float(np.sum(w))
        if total <= 0:
            raise ValueError("score_weight must contain at least one positive pixel")
        residual = float(np.sqrt(np.sum(w * sq) / total) / denom)
        score_pixels = int(np.count_nonzero(w > 0))

    ux, uy = u[..., 0], u[..., 1]
    ux_y, ux_x = np.gradient(ux)
    uy_y, uy_x = np.gradient(uy)
    roughness = float(np.mean(ux_x ** 2 + ux_y ** 2 + uy_x ** 2 + uy_y ** 2))
    return {"residual": residual, "roughness": roughness, "score_pixels": score_pixels}


def pick_lcurve_corner(
    candidates,
    residuals,
    roughnesses,
    *,
    residual_tolerance: float = 0.05,
    kink_tolerance: float = 0.02,
) -> tuple[int, str]:
    """Pick an L-curve corner, falling back to residual plateau when degenerate.

    The primary selector uses the maximum perpendicular distance from each
    ``(log10 roughness, log10 residual)`` point to the chord through the two
    endpoints. If the curve has fewer than three finite points, has no usable
    chord, chooses an endpoint, or has no meaningful kink, the documented fallback
    picks the largest smoothing candidate whose residual is within 5% of the
    minimum residual.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    roughnesses = np.asarray(roughnesses, dtype=np.float64)
    valid = np.isfinite(residuals) & np.isfinite(roughnesses)
    if int(valid.sum()) < 3:
        return _fallback_residual_plateau(candidates, residuals, residual_tolerance)

    idxs = np.nonzero(valid)[0]
    x = np.log10(np.clip(roughnesses[valid], 1e-30, None))
    y = np.log10(np.clip(residuals[valid], 1e-30, None))
    pts = np.column_stack([x, y])
    chord = pts[-1] - pts[0]
    chord_len = float(np.linalg.norm(chord))
    scale = float(np.linalg.norm(np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)))
    if chord_len < 1e-12 or scale < 1e-12:
        return _fallback_residual_plateau(candidates, residuals, residual_tolerance)

    offsets = pts - pts[0]
    distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / chord_len
    local = int(np.argmax(distances))
    if local == 0 or local == len(idxs) - 1:
        return _fallback_residual_plateau(candidates, residuals, residual_tolerance)
    if float(distances[local]) < kink_tolerance * max(scale, 1e-12):
        return _fallback_residual_plateau(candidates, residuals, residual_tolerance)
    return int(idxs[local]), "lcurve_corner"


def _fallback_residual_plateau(candidates, residuals, tolerance=0.05) -> tuple[int, str]:
    residuals = np.asarray(residuals, dtype=np.float64)
    valid = np.isfinite(residuals)
    if not np.any(valid):
        return 0, "fallback_first_candidate_no_finite_residual"
    min_res = float(np.nanmin(residuals[valid]))
    ok = np.nonzero(valid & (residuals <= min_res * (1.0 + tolerance)))[0]
    if len(ok) == 0:
        return int(np.nanargmin(residuals)), "fallback_min_residual"
    return int(ok[-1]), "fallback_largest_candidate_within_5pct_min_residual"


def pick_residual_plateau(candidates, residuals, *, tolerance: float = 0.05) -> tuple[int, str]:
    """Pick the cheapest convergence setting on the residual plateau."""
    residuals = np.asarray(residuals, dtype=np.float64)
    valid = np.isfinite(residuals)
    if not np.any(valid):
        return 0, "fallback_first_candidate_no_finite_residual"
    min_res = float(np.nanmin(residuals[valid]))
    ok = np.nonzero(valid & (residuals <= min_res * (1.0 + tolerance)))[0]
    if len(ok) == 0:
        return int(np.nanargmin(residuals)), "fallback_min_residual"
    return int(ok[0]), "residual_plateau_first_within_5pct_min"


def pick_min_residual(candidates, residuals) -> tuple[int, str]:
    """Pick the candidate with the lowest validation residual."""
    residuals = np.asarray(residuals, dtype=np.float64)
    valid = np.isfinite(residuals)
    if not np.any(valid):
        return 0, "fallback_first_candidate_no_finite_residual"
    return int(np.nanargmin(residuals)), "min_validation_residual"


def _masked_holdout_weights(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split a mask weight into train and validation weights."""
    if weight is None:
        raise ValueError(
            "Masked holdout CV tuning needs a displacement mask. Enable mask "
            "confinement and provide a mask, or use the L-curve selector."
        )
    w = np.nan_to_num(np.asarray(weight, dtype=np.float32), nan=0.0)
    if w.ndim != 2:
        raise ValueError("Masked holdout CV tuning needs a 2D displacement mask weight")
    mask = w > 0
    if int(mask.sum()) < 16:
        raise ValueError("Masked holdout CV tuning needs at least 16 masked pixels")

    h, wdim = w.shape
    yy, xx = np.indices((h, wdim))
    block = max(4, min(h, wdim) // 16)
    holdout = mask & (((yy // block + xx // block) % 4) == 0)
    if int(holdout.sum()) < 4 or int((mask & ~holdout).sum()) < 4:
        holdout = mask & (((yy + xx) % 5) == 0)
    if int(holdout.sum()) < 4 or int((mask & ~holdout).sum()) < 4:
        raise ValueError("Masked holdout CV tuning could not create a stable mask split")

    train = w.copy()
    train[holdout] = 0.0
    validation = np.zeros_like(w, dtype=np.float32)
    validation[holdout] = w[holdout]
    return train, validation


def normalize_tuning_selector(selector: str | None) -> str:
    value = (selector or "L-curve").strip().lower().replace("_", " ").replace("-", " ")
    if value in {"l curve", "lcurve"}:
        return "lcurve"
    if value in {"masked holdout cv", "mask holdout cv", "holdout cv", "masked cv"}:
        return "masked_cv"
    raise ValueError(f"Unknown displacement tuning selector {selector!r}")


def autotune_parameter(
    analyzer: BaseDisplacementAnalyzer,
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    param_name: str,
    candidates,
    selector: str,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, object, dict]:
    """Run one full independent solve per candidate and pick a parameter value."""
    if not candidates:
        raise ValueError(f"No candidates provided for {param_name}")

    solve_weight = weight
    score_weight = None
    if selector == "masked_cv":
        solve_weight, score_weight = _masked_holdout_weights(weight)

    original = getattr(analyzer.params, param_name)
    fields = []
    rows = []
    try:
        for value in candidates:
            setattr(analyzer.params, param_name, value)
            field = analyzer.calculate_flow(reference, moving, weight=solve_weight)
            diag = displacement_candidate_diagnostics(
                reference, moving, field, score_weight=score_weight,
            )
            fields.append(field)
            rows.append({
                "value": float(value),
                "residual": diag["residual"],
                "roughness": diag["roughness"],
                "score_pixels": diag["score_pixels"],
            })
    finally:
        setattr(analyzer.params, param_name, original)

    residuals = [row["residual"] for row in rows]
    roughnesses = [row["roughness"] for row in rows]
    if selector == "lcurve":
        idx, reason = pick_lcurve_corner(candidates, residuals, roughnesses)
    elif selector == "plateau":
        idx, reason = pick_residual_plateau(candidates, residuals)
    elif selector == "masked_cv":
        idx, reason = pick_min_residual(candidates, residuals)
    else:
        raise ValueError(f"Unknown tuning selector {selector!r}")

    chosen = candidates[idx]
    diagnostics = {
        "param_name": param_name,
        "selector": selector,
        "chosen": float(chosen),
        "selection": reason,
        "candidates": rows,
    }
    logger.info(
        "%s auto-tune %s=%s via %s (%s)",
        analyzer.algorithm_name, param_name, chosen, selector, reason,
    )
    return fields[idx], chosen, diagnostics


def autotune_displacement_parameters(
    analyzer: BaseDisplacementAnalyzer,
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    smoothing_selector: str | None = None,
) -> tuple[np.ndarray, dict[str, object], dict]:
    """Tune this analyzer's current-frame convergence and smoothing knobs.

    This is intentionally a one-shot helper for an explicit UI button. It is
    expensive: each candidate value is a full solve, because PIV and FFD apply
    their regularization/convergence knobs inside the solver rather than as a
    post-processing filter.
    """
    chosen: dict[str, object] = {}
    smoothing_selector = normalize_tuning_selector(smoothing_selector)
    diagnostics = {"algorithm": analyzer.algorithm_name, "steps": []}

    conv_name = getattr(analyzer, "convergence_param_name", None)
    conv_candidates = tuple(getattr(analyzer, "convergence_candidates", ()) or ())
    if conv_name and conv_candidates:
        field, value, diag = autotune_parameter(
            analyzer, reference, moving,
            param_name=conv_name, candidates=conv_candidates,
            selector="plateau", weight=weight,
        )
        setattr(analyzer.params, conv_name, value)
        chosen[conv_name] = value
        diagnostics["steps"].append(diag)
    else:
        field = analyzer.calculate_flow(reference, moving, weight=weight)

    smooth_name = getattr(analyzer, "smoothing_param_name", None)
    smooth_candidates = tuple(getattr(analyzer, "smoothing_candidates", ()) or ())
    if smooth_name and smooth_candidates:
        field, value, diag = autotune_parameter(
            analyzer, reference, moving,
            param_name=smooth_name, candidates=smooth_candidates,
            selector=smoothing_selector, weight=weight,
        )
        setattr(analyzer.params, smooth_name, value)
        chosen[smooth_name] = value
        diagnostics["steps"].append(diag)
    elif not chosen:
        diagnostics["selection"] = "no_tunable_parameters"

    diagnostics["chosen"] = {name: float(value) for name, value in chosen.items()}
    return field, chosen, diagnostics
