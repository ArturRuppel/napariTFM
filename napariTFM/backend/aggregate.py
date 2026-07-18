"""Cross-experiment aggregator: reduce a ``.ntfm`` series to one tidy table.

A single ``.ntfm`` is per-sample ``(t, y, x)`` — far too granular for figures or
statistics. This aggregator is a **reduction layer**: it groups each container's
tidy table by ``(region = mask label, frame = t)``, computes per-region,
per-frame derived scalars, then stacks every experiment into one long summary
table whose grain is::

    one row per (experiment_id, region_id, frame)

**Design (decoupled, "Option A").** The aggregator stops at a tidy ``summary.csv``
plus a ``provenance.json`` audit sidecar and a ``schema.json`` column-typing
sidecar. Statistics and plotting are downstream concerns (Iris, seaborn, R). It
deliberately does *not* emit a ``.iris`` container or premade grammar-of-graphics
specs: the previous aggregator (``utilities/iris.py``, deleted in ``6adbd0e``)
coupled itself to a moving external spec and rotted. Tidy CSV + a stable schema is
all any downstream stats tool needs.

**Metric set** = exactly what :mod:`napariTFM.backend.metrics_calculator` computes
(no new physics): ``total_strain_energy``, ``polarization_index``, ``lambda1``,
``lambda2``.

**Improvements ported from ITASC's ``itasc-aggregate``:**

- *Identity uniqueness validation* — refuse to pool when two containers map to the
  same ``experiment_id``, naming the colliding sources, instead of silently
  merging their rows into one indistinguishable grain.
- *Deterministic stable ``id``* — ``experiment_id|region_id|frame`` per row, so a
  re-aggregate keeps row identity regardless of order (downstream annotation can
  be joined back).
- *Ready / not-ready partitioning* — a header-only check (no pixel decode) that
  greys out containers that cannot contribute a usable row, reporting why.
- *Materialized-view output* — ``summary.csv`` is rewritten whole every pool,
  never appended, so a re-run never accumulates stale rows.
- *Run-level ``provenance.json``* — an audit of what the pool *attempted* (every
  contributing source + identity + row count, and every skipped source + reason),
  not just what it wrote.

**Conventions** (unchanged from the original reduction):

- Displacement is stored in µm and converted to metres before energy/force math.
- Pixel area is ``grid_spacing²`` in m², with ``grid_spacing`` recovered from the
  table's physical ``x[µm]``/``y[µm]`` columns (no config-key coupling).
- The moment tensor uses pixel positions centred on each region's centroid, so it
  measures the force dipole within the region independent of absolute position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json

import numpy as np
import pandas as pd

from napariTFM.backend import metrics_calculator as mc
from napariTFM.utilities import ntfm

try:  # header-only series inspection for the ready/not-ready partition
    import tifffile
except Exception:  # pragma: no cover - tifffile is a hard dep in practice
    tifffile = None


# Derived scalar metric columns, in display order.
METRIC_COLUMNS = [
    "total_strain_energy",
    "polarization_index",
    "lambda1",
    "lambda2",
]

# Grain identifiers.
ID_COLUMNS = ["experiment_id", "region_id", "frame"]

# The deterministic per-row identity column (leads the table).
ROW_ID_COLUMN = "id"

# Container series names (see ``ntfm`` module header): mask is the region source,
# traction is the force stage. A container needs both to yield a usable metric.
_MASK_SERIES = "mask"
_FORCE_SERIES = "traction"

# Human-readable label per identifier column.
_ID_LABELS = {
    "experiment_id": "Experiment",
    "region_id": "Cell",
    "frame": "Frame",
}

# Numeric metric metadata: (human label, physical unit). λ are moment-tensor
# eigenvalues in N·m; the polarization index is dimensionless.
_METRIC_META = {
    "total_strain_energy": ("Total strain energy", "J"),
    "polarization_index": ("Polarization index", ""),
    "lambda1": ("λ1", "N·m"),
    "lambda2": ("λ2", "N·m"),
}


# ---------------------------------------------------------------------------
# Per-region-per-frame reduction (revived from the deleted iris.py)
# ---------------------------------------------------------------------------

def _present(field_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return the field, or None if it is missing / entirely NaN (unrun stage)."""
    if field_arr is None or np.isnan(field_arr).all():
        return None
    return field_arr


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
    """Pixel positions ``(ny, nx, 2)`` in metres, centred on the region centroid.

    ``[..., 0]`` is x (from columns), ``[..., 1]`` is y (from rows), matching
    :func:`metrics_calculator.calculate_moment_tensor`'s expected order.
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


def _experiment_id(path: Path, metadata: dict) -> str:
    """Stable, human-meaningful id for a container.

    Every batch container is named ``TFM_results.ome.tif`` (``RESULTS_FILENAME``),
    so the filename stem is useless as an identifier — it is identical for every
    experiment. The source *folder* is the real identity; it is recorded in the
    container's own metadata (``inputs.folder``). Fall back to the parent-directory
    name, then the stem, if that provenance is absent.
    """
    folder = (metadata.get("inputs") or {}).get("folder")
    if folder:
        name = Path(str(folder)).name
        if name:
            return name
    parent = path.parent.name
    if parent:
        return parent
    return path.stem


@dataclass
class _Loaded:
    """One container read exactly once: everything the aggregator needs from it."""

    path: Path
    experiment_id: str
    labels: Dict[str, object]
    df: pd.DataFrame
    arrays: Dict[str, Optional[np.ndarray]]


def _load(path) -> _Loaded:
    """Read a container once and resolve its id + design tags (the single read)."""
    path = Path(path)
    df, metadata = ntfm.read_ntfm(path)
    return _Loaded(
        path=path,
        experiment_id=_experiment_id(path, metadata),
        labels=dict(metadata.get("labels") or {}),
        df=df,
        arrays=ntfm.tidy_to_arrays(df),
    )


def _summarize_loaded(loaded: _Loaded, experiment_id: Optional[str] = None) -> pd.DataFrame:
    """Reduce an already-loaded container to a per-``(region_id, frame)`` table."""
    exp_id = experiment_id if experiment_id is not None else loaded.experiment_id
    columns = ID_COLUMNS + METRIC_COLUMNS
    df, arrays = loaded.df, loaded.arrays

    mask = arrays["mask"]
    if mask is None:
        return pd.DataFrame(columns=columns)

    # The writer always emits every measure column (all-NaN when its stage was
    # not run), so an all-NaN field round-trips as an array, not None. Treat it
    # as absent so unrun stages produce NaN metrics rather than 0.
    displacement = _present(arrays["displacement_field"])
    force = _present(arrays["force_field"])
    spacing_m = _grid_spacing_um(df) * 1e-6

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
                    "experiment_id": exp_id,
                    "region_id": region,
                    "frame": frame,
                    **metrics,
                }
            )

    return pd.DataFrame(records, columns=columns)


def summarize_ntfm(path, *, experiment_id: Optional[str] = None) -> pd.DataFrame:
    """Reduce one ``.ntfm`` to a per-``(region_id, frame)`` summary table.

    ``experiment_id`` defaults to the container's source-folder name (see
    :func:`_experiment_id`). Rows are emitted for every ``(region, frame)`` where
    the region (a non-zero mask label) is present. A container with no mask yields
    an empty (correctly-columned) frame.
    """
    return _summarize_loaded(_load(path), experiment_id=experiment_id)


# ---------------------------------------------------------------------------
# Ready / not-ready partition (header-only, no pixel decode)
# ---------------------------------------------------------------------------

def _series_names(path) -> set:
    """Series present in a container, read from the OME-XML header only.

    Empty set for a missing/unreadable container, so callers treat "no output"
    and "can't tell" identically (mirrors ``ntfm.populated_measures``).
    """
    if tifffile is None:
        return set()
    try:
        with tifffile.TiffFile(str(path)) as tf:
            return {s.name for s in tf.series}
    except Exception:
        return set()


def readiness(path) -> Tuple[bool, str]:
    """Can this container contribute a usable (non-all-NaN) row?

    Needs a ``mask`` series (to define regions) and a ``traction`` series (force —
    without it every metric is NaN). Returns ``(ready, reason)`` where ``reason``
    is a short skip explanation when not ready, else the empty string.
    """
    names = _series_names(path)
    if _MASK_SERIES not in names:
        return False, "no mask"
    if _FORCE_SERIES not in names:
        return False, "no force"
    return True, ""


def partition_ready(paths: Iterable) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Split containers into ``(ready, [(skipped_path, reason), …])``."""
    ready: List[Path] = []
    skipped: List[Tuple[Path, str]] = []
    for raw in paths:
        p = Path(raw)
        ok, reason = readiness(p)
        (ready.append(p) if ok else skipped.append((p, reason)))
    return ready, skipped


# ---------------------------------------------------------------------------
# Series reduction: stack + promote labels + stable id
# ---------------------------------------------------------------------------

def _require_unique_ids(id_by_path: Dict[Path, str]) -> None:
    """Refuse to pool when two containers map to the same ``experiment_id``.

    A duplicate id makes the pooled ``(experiment_id, region_id, frame)`` grain —
    and the stable row ``id`` — ambiguous, silently merging distinct experiments.
    Raise an actionable error naming every colliding source (ITASC's
    ``_require_unique_identity``).
    """
    by_id: Dict[str, List[Path]] = {}
    for path, exp_id in id_by_path.items():
        by_id.setdefault(exp_id, []).append(path)
    collisions = {k: v for k, v in by_id.items() if len(v) > 1}
    if collisions:
        lines = [
            f"  {exp_id!r}: " + ", ".join(str(p) for p in sorted(paths))
            for exp_id, paths in sorted(collisions.items())
        ]
        raise ValueError(
            "Cannot aggregate: these experiments share an identifier "
            "(their rows would collide). Give the source folders distinct "
            "names:\n" + "\n".join(lines)
        )


def _assign_row_id(table: pd.DataFrame) -> None:
    """Insert the deterministic ``id`` column in place (leads the frame).

    ``experiment_id|region_id|frame`` — stable across re-aggregation regardless of
    row order, so downstream annotation can be joined back reliably.
    """
    if table.empty:
        table.insert(0, ROW_ID_COLUMN, pd.Series(dtype=object))
        return
    ids = (
        table["experiment_id"].astype(str)
        + "|"
        + table["region_id"].astype(str)
        + "|"
        + table["frame"].astype(str)
    )
    table.insert(0, ROW_ID_COLUMN, ids.to_numpy())


def _reduce_series(
    paths: Iterable,
    labels: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[pd.DataFrame, Dict[Path, str], Dict[str, Dict[str, object]]]:
    """Read each container once and pool: returns ``(table, id_by_path, tags_by_id)``.

    The single-read core behind :func:`build_summary_table` and
    :func:`pool_experiments` — both need the pooled table *and* the per-experiment
    identity/tag mapping, so reading each (multi-MB) container more than once is
    wasteful. ``labels`` overrides each container's on-disk tags when supplied.
    """
    loaded = [_load(p) for p in paths]

    id_by_path: Dict[Path, str] = {ld.path: ld.experiment_id for ld in loaded}
    _require_unique_ids(id_by_path)

    tags_by_id: Dict[str, Dict[str, object]] = {}
    for ld in loaded:
        tags_by_id[ld.experiment_id] = (
            dict((labels or {}).get(ld.experiment_id, {}))
            if labels is not None
            else dict(ld.labels)
        )

    frames = [_summarize_loaded(ld) for ld in loaded]
    if frames:
        table = pd.concat(frames, ignore_index=True)
    else:
        table = pd.DataFrame(columns=ID_COLUMNS + METRIC_COLUMNS)

    # Promote label columns, ordered by first appearance across all experiments.
    label_keys: List[str] = []
    for ld in loaded:
        for key in tags_by_id[ld.experiment_id]:
            if key not in label_keys:
                label_keys.append(key)
    for key in label_keys:
        table[key] = [
            tags_by_id.get(exp, {}).get(key) for exp in table["experiment_id"]
        ]

    _assign_row_id(table)
    return table, id_by_path, tags_by_id


def build_summary_table(
    paths: Iterable,
    labels: Optional[Dict[str, Dict[str, object]]] = None,
) -> pd.DataFrame:
    """Stack a ``.ntfm`` series into one summary table; promote label columns.

    Each path is reduced to the ``(experiment_id, region_id, frame)`` grain.
    ``labels`` maps ``experiment_id -> {key: value}``; when ``None``, each
    container's own on-disk ``labels`` are used (the batch run persists the
    experiments-table tags there). Every distinct key becomes one categorical
    column, ordered by first appearance; an experiment missing a key gets
    ``None``. A deterministic ``id`` column leads the table.

    Raises ``ValueError`` if two containers map to the same ``experiment_id``.
    """
    return _reduce_series(paths, labels=labels)[0]


# ---------------------------------------------------------------------------
# Schema typing (data/schema.json) — stable, self-contained column typing
# ---------------------------------------------------------------------------

def build_schema(table: pd.DataFrame) -> list:
    """Type every summary-table column: ``identifier`` / ``numeric`` / ``categorical``.

    One entry per column, in column order::

        {"name", "type", "label"[, "unit"][, "levels"]}

    ``numeric`` metric columns carry a physical ``unit``; ``categorical`` label
    columns carry sorted ``levels``. The row ``id`` and grain keys are identifiers.
    """
    schema = []
    for name in table.columns:
        if name == ROW_ID_COLUMN:
            schema.append({"name": name, "type": "identifier", "label": "Row id"})
        elif name in ID_COLUMNS:
            schema.append(
                {"name": name, "type": "identifier", "label": _ID_LABELS[name]}
            )
        elif name in _METRIC_META:
            label, unit = _METRIC_META[name]
            schema.append(
                {"name": name, "type": "numeric", "label": label, "unit": unit}
            )
        else:  # a promoted experiment-design label column
            levels = sorted(
                {str(v) for v in table[name].tolist() if v is not None and pd.notna(v)}
            )
            schema.append(
                {
                    "name": name,
                    "type": "categorical",
                    "label": name.replace("_", " ").capitalize(),
                    "levels": levels,
                }
            )
    return schema


# ---------------------------------------------------------------------------
# The pool: materialized tidy CSV + provenance.json + schema.json
# ---------------------------------------------------------------------------

# Output artifact names (beside each other in the pool's out_dir).
SUMMARY_FILENAME = "summary.csv"
SCHEMA_FILENAME = "schema.json"
PROVENANCE_FILENAME = "provenance.json"


@dataclass
class AggregateResult:
    """What a pool produced and attempted."""

    summary_path: Optional[Path]
    schema_path: Optional[Path]
    provenance_path: Optional[Path]
    n_rows: int
    ready: List[Path] = field(default_factory=list)
    skipped: List[Tuple[Path, str]] = field(default_factory=list)


def pool_experiments(
    paths: Iterable,
    out_dir,
    *,
    labels: Optional[Dict[str, Dict[str, object]]] = None,
) -> AggregateResult:
    """Pool a ``.ntfm`` series into ``out_dir``: ``summary.csv`` + sidecars.

    Only *ready* containers (mask + force present) contribute; the rest are
    skipped and named in the result (and in ``provenance.json``). The summary is
    a materialized view — the CSV is rewritten whole, never appended.

    Returns an :class:`AggregateResult`. Raises ``ValueError`` if two ready
    containers share an ``experiment_id``.
    """
    out_dir = Path(out_dir)
    ready, skipped = partition_ready(paths)

    if not ready:
        # Nothing to pool: still leave a provenance breadcrumb of what was seen.
        out_dir.mkdir(parents=True, exist_ok=True)
        provenance_path = out_dir / PROVENANCE_FILENAME
        _write_provenance(
            provenance_path,
            table=None,
            experiments=[],
            id_by_path={},
            tags_by_id={},
            skipped=skipped,
            metric_columns=METRIC_COLUMNS,
        )
        return AggregateResult(
            summary_path=None,
            schema_path=None,
            provenance_path=provenance_path,
            n_rows=0,
            ready=[],
            skipped=skipped,
        )

    table, id_by_path, tags_by_id = _reduce_series(ready, labels=labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_FILENAME
    schema_path = out_dir / SCHEMA_FILENAME
    provenance_path = out_dir / PROVENANCE_FILENAME

    table.to_csv(summary_path, index=False)
    schema_path.write_text(json.dumps(build_schema(table), indent=2), encoding="utf-8")

    # Per-experiment row counts for the provenance audit.
    counts = table.groupby("experiment_id").size().to_dict() if not table.empty else {}
    _write_provenance(
        provenance_path,
        table=table,
        experiments=ready,
        id_by_path=id_by_path,
        tags_by_id=tags_by_id,
        skipped=skipped,
        metric_columns=METRIC_COLUMNS,
        counts=counts,
    )

    return AggregateResult(
        summary_path=summary_path,
        schema_path=schema_path,
        provenance_path=provenance_path,
        n_rows=int(len(table)),
        ready=list(ready),
        skipped=skipped,
    )


def _write_provenance(
    provenance_path: Path,
    *,
    table: Optional[pd.DataFrame],
    experiments: List[Path],
    id_by_path: Dict[Path, str],
    tags_by_id: Dict[str, Dict[str, object]],
    skipped: List[Tuple[Path, str]],
    metric_columns: List[str],
    counts: Optional[Dict[str, int]] = None,
) -> None:
    """Write the run-level audit sidecar: what the pool attempted, not just wrote."""
    counts = counts or {}
    experiments_record = []
    for path in experiments:
        exp_id = id_by_path.get(path, _experiment_id(path, {}))
        experiments_record.append(
            {
                "experiment_id": exp_id,
                "source": str(path),
                "labels": tags_by_id.get(exp_id, {}),
                "n_rows": int(counts.get(exp_id, 0)),
            }
        )
    payload = {
        "tool": "napariTFM",
        "package_version": ntfm.package_version(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "grain": list(ID_COLUMNS),
        "metrics": list(metric_columns),
        "experiments": experiments_record,
        "skipped": [{"source": str(p), "reason": reason} for p, reason in skipped],
        "table": (
            None
            if table is None
            else {
                "path": SUMMARY_FILENAME,
                "n_rows": int(len(table)),
                "n_cols": int(table.shape[1]),
            }
        ),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
