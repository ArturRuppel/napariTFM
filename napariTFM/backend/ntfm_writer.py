"""Shared writer for one experiment's ``.ntfm`` container.

Both paths that persist results — the batch orchestrator (:class:`BatchAnalysis`)
and an interactive per-stage run — funnel through here, so a stage saved live in
napari and the same stage saved by a headless batch produce an identical
container at the identical canonical path. The single writer is what keeps the
experiments-list row dots and the per-stage section dots reading one truth.

Pure compute/IO: no Qt, no napari. Errors propagate; each caller decides whether
to surface them (the batch re-raises so a folder reads ``error``; the interactive
path shows a message without crashing the viewer).
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from napariTFM.backend.stress import process_mask_data
from napariTFM.backend.parameter_dataclasses import UnifiedParameters
from napariTFM.utilities import ntfm


def mask_on_grid(mask_data, force_result, displacement_result) -> Optional[np.ndarray]:
    """Resize a raw external mask onto the analysis grid, or return ``None``.

    The mask shares the force/displacement grid; ``process_mask_data`` resizes it
    exactly as the stress stage does. A mask that cannot be aligned is dropped (``None``)
    rather than aborting the write — the results are still worth persisting.
    """
    if mask_data is None:
        return None

    field = None
    if force_result is not None:
        field = force_result.force_field
    elif displacement_result is not None:
        field = displacement_result.displacement_field
    if field is None:
        return None

    try:
        # ``field`` is (nt, ny, nx, 2); process_mask_data resizes to its grid.
        mask_stack, _ = process_mask_data(mask_data, field)
        return mask_stack.astype(np.int64)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Could not align mask to analysis grid: {str(e)}")
        return None


def _zeroed_off_mask(field: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
    """Copy of ``field`` with every pixel zeroed where ``grid_mask == 0``.

    ``grid_mask`` may carry a single frame (``nt == 1``) while ``field`` has
    several; broadcast it rather than require an exact shape match, mirroring
    the 2D-mask broadcast ``arrays_to_tidy`` already does for the ``mask`` column.
    """
    mask = grid_mask
    if mask.shape[0] == 1 and field.shape[0] > 1:
        mask = np.broadcast_to(mask[0], (field.shape[0],) + mask.shape[1:])
    out = np.array(field, copy=True)
    out[mask == 0] = 0.0
    return out


def _apply_mask_on_save(grid_mask, displacement_result, force_result, stress_result):
    """Off-mask results, ``apply_mask_on_save`` (ROADMAP §4 backlog).

    Off-cell substrate signal is noise; zeroing ``u_x/u_y/F_x/F_y`` (and stress,
    if present) wherever the mask is background compresses the written
    container substantially. The ``mask`` column already records which pixels
    were zeroed, so the loss is self-documenting. Returns lightweight shims
    (not mutated copies of the originals) carrying only the two attributes
    :func:`dataframe_from_results` reads, so the caller's result objects —
    still used for visualization before this point — are left untouched.
    """
    if displacement_result is not None:
        displacement_result = SimpleNamespace(
            displacement_field=_zeroed_off_mask(displacement_result.displacement_field, grid_mask),
            physical_scale=displacement_result.physical_scale,
        )
    if force_result is not None:
        force_result = SimpleNamespace(
            force_field=_zeroed_off_mask(force_result.force_field, grid_mask),
            physical_scale=force_result.physical_scale,
        )
    if stress_result is not None:
        stress_result = SimpleNamespace(
            stress_tensor=_zeroed_off_mask(stress_result.stress_tensor, grid_mask),
            physical_scale=stress_result.physical_scale,
        )
    return displacement_result, force_result, stress_result


def config_from_parameters(parameters: Optional[dict]) -> dict:
    """Resolved per-experiment config: ``asdict`` of the effective parameters.

    Reconstructing :class:`UnifiedParameters` and dumping it keeps one source of
    truth for field mapping — unknown keys from older presets are ignored and
    missing keys fall back to defaults.
    """
    valid = {f.name for f in fields(UnifiedParameters)}
    raw = parameters or {}
    return asdict(UnifiedParameters(**{k: v for k, v in raw.items() if k in valid}))


def write_experiment_ntfm(
    ntfm_path,
    *,
    parameters: Optional[dict] = None,
    displacement_result=None,
    force_result=None,
    stress_result=None,
    mask: Optional[np.ndarray] = None,
    folder=None,
    input_files: Optional[dict] = None,
    labels: Optional[dict] = None,
    apply_mask_on_save: bool = False,
) -> Optional[Path]:
    """Write one experiment's results to ``.ntfm``; the sole persisted artifact.

    ``apply_mask_on_save`` (opt-in, default off — batch-only, see
    :func:`_apply_mask_on_save`) zeroes off-cell pixels before write; the
    interactive per-stage save never sets it, so live results are unaffected.

    Returns the written :class:`Path`, or ``None`` when there is nothing to
    persist (all three results absent). Raises on a real write failure so the
    caller can report it rather than silently losing the only data artifact.
    """
    if displacement_result is None and force_result is None and stress_result is None:
        return None

    grid_mask = mask_on_grid(mask, force_result, displacement_result)

    if apply_mask_on_save and grid_mask is not None:
        displacement_result, force_result, stress_result = _apply_mask_on_save(
            grid_mask, displacement_result, force_result, stress_result
        )

    config = config_from_parameters(parameters)
    inputs = {
        "folder": str(folder) if folder is not None else "",
        "files": dict(input_files or {}),
    }

    ntfm_path = Path(ntfm_path)
    ntfm_path.parent.mkdir(parents=True, exist_ok=True)
    # merge_existing defaults to True: a force- or stress-only second write
    # preserves previously-saved displacement/force data from the existing container.
    ntfm.results_to_ntfm(
        ntfm_path,
        config=config,
        displacement_result=displacement_result,
        force_result=force_result,
        stress_result=stress_result,
        mask=grid_mask,
        inputs=inputs,
        labels=labels or {},
    )
    return ntfm_path
