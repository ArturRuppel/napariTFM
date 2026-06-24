"""Aggregator: reduce a ``.ntfm`` series to a tidy summary table and export a
``.iris`` document (ROADMAP §5).

A single ``.ntfm`` is per-sample ``(t, y, x)`` — far too granular for figures or
statistics. The aggregator is a **reduction layer**: it groups each ``.ntfm``'s
tidy table by ``(region = mask label, frame = t)`` and computes per-region,
per-frame derived scalar metrics, then stacks every experiment into one long
summary table whose grain is::

    one row per (experiment_id, region_id, frame)

This is the "regions-by-frame" analog of Iris's ``cells_by_frame`` sample. Each
mask label ``1..N`` is a distinct cell, so ``region_id`` is a real per-cell
identifier, not a connected-component artifact.

**Metric set** = exactly what ``metrics_calculator`` already computes (no new
physics): ``total_strain_energy``, ``polarization_index``, ``lambda1``,
``lambda2``.

**Conventions.**
- Displacement is stored in µm and converted to metres before energy/force math.
- The pixel area is ``grid_spacing²`` in m², with ``grid_spacing`` recovered from
  the table's physical ``x[µm]``/``y[µm]`` columns (no config-key coupling).
- The moment tensor is computed with pixel positions **centred on each region's
  geometric centroid**, so it measures the force dipole/spread within the region
  independent of the region's absolute position on the grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from napariTFM.backend import metrics_calculator as mc
from napariTFM.utilities import ntfm

# Derived scalar metric columns, in display order.
METRIC_COLUMNS = [
    "total_strain_energy",
    "polarization_index",
    "lambda1",
    "lambda2",
]

# Grain identifiers.
ID_COLUMNS = ["experiment_id", "region_id", "frame"]


def _present(field: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return the field, or None if it is missing / entirely NaN (unrun stage)."""
    if field is None or np.isnan(field).all():
        return None
    return field


def _grid_spacing_um(df: pd.DataFrame) -> float:
    """Recover the physical grid spacing (µm) from the tidy table's positions."""
    for column in ("x[µm]", "y[µm]"):
        values = np.sort(df[column].unique())
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(diffs.min())
    return 1.0


def _region_positions_m(region_mask: np.ndarray, spacing_m: float) -> np.ndarray:
    """Pixel positions (ny, nx, 2) in metres, centred on the region centroid.

    ``[..., 0]`` is x (from columns), ``[..., 1]`` is y (from rows), matching
    ``calculate_moment_tensor``'s expected order.
    """
    ny, nx = region_mask.shape
    row_idx, col_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ys, xs = np.where(region_mask)
    cy, cx = ys.mean(), xs.mean()
    positions = np.zeros((ny, nx, 2))
    positions[..., 0] = (col_idx - cx) * spacing_m
    positions[..., 1] = (row_idx - cy) * spacing_m
    return positions


def _region_frame_metrics(
    displacement_frame: Optional[np.ndarray],
    force_frame: Optional[np.ndarray],
    region_mask: np.ndarray,
    spacing_m: float,
) -> dict:
    """Compute the four derived scalars for one region in one frame.

    Missing displacement *or* force yields NaN strain energy; missing force
    yields NaN polarization metrics — an unrun stage never invents numbers.
    """
    pixel_area_m2 = spacing_m ** 2
    metrics = {name: float("nan") for name in METRIC_COLUMNS}

    if displacement_frame is not None and force_frame is not None:
        disp_m = displacement_frame * 1e-6
        sed = mc.calculate_strain_energy_density(disp_m, force_frame)
        metrics["total_strain_energy"] = mc.calculate_total_strain_energy(
            sed, region_mask, pixel_area_m2
        )

    if force_frame is not None:
        positions = _region_positions_m(region_mask, spacing_m)
        moment = mc.calculate_moment_tensor(
            force_frame, region_mask, positions, pixel_area_m2
        )
        pi, l1, l2 = mc.calculate_polarization(moment)
        metrics["polarization_index"] = pi
        metrics["lambda1"] = l1
        metrics["lambda2"] = l2

    return metrics


def summarize_ntfm(path) -> pd.DataFrame:
    """Reduce one ``.ntfm`` to a per-``(region_id, frame)`` summary table.

    ``experiment_id`` is the container's file stem. Rows are emitted for every
    ``(region, frame)`` where the region (a non-zero mask label) is present.
    """
    path = Path(path)
    df, _metadata = ntfm.read_ntfm(path)
    arrays = ntfm.tidy_to_arrays(df)

    mask = arrays["mask"]
    if mask is None:
        return pd.DataFrame(columns=ID_COLUMNS + METRIC_COLUMNS)

    # The writer always emits every measure column (all-NaN when its stage was
    # not run), so an all-NaN field round-trips as an array, not None. Treat it
    # as absent so unrun stages produce NaN metrics rather than 0.
    displacement = _present(arrays["displacement_field"])
    force = _present(arrays["force_field"])
    spacing_m = _grid_spacing_um(df) * 1e-6
    experiment_id = path.stem

    nt = mask.shape[0]
    region_ids = [int(r) for r in np.unique(mask) if r != 0]

    records = []
    for region in region_ids:
        for frame in range(nt):
            region_mask = (mask[frame] == region).astype(float)
            if region_mask.sum() == 0:
                continue  # region not present in this frame
            metrics = _region_frame_metrics(
                None if displacement is None else displacement[frame],
                None if force is None else force[frame],
                region_mask,
                spacing_m,
            )
            records.append(
                {
                    "experiment_id": experiment_id,
                    "region_id": region,
                    "frame": frame,
                    **metrics,
                }
            )

    return pd.DataFrame(records, columns=ID_COLUMNS + METRIC_COLUMNS)
