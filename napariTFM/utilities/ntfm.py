"""Native ``.ntfm`` container and the tidy long-format table (ROADMAP §1).

A single tidy, long-format table is the canonical data representation: one row
per ``(t, y, x)`` grid sample, every value in physical units with the unit
carried inline in the column name. The table is self-contained for analysis;
the only thing in the sidecar is run-level provenance plus the resolved config.

``.ntfm`` is a zip::

    experiment.ntfm
    ├── samples.parquet   # the tidy table
    └── metadata.json     # provenance + resolved config

One converter handles both directions so import and export share a code path.
``arrays -> tidy -> arrays`` is lossless for every column below; array
reconstruction is trivial because ``row``/``col`` are explicit integer columns.
"""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FORMAT_VERSION = "1.0"

# Identifiers (Iris vocabulary) — locate every sample.
ID_COLUMNS = ["t[min]", "y[µm]", "x[µm]", "row", "col"]

# Measures — the full 2D field at each sample. NaN where a measure is absent
# (e.g. off-mask stress nodes). MSM is symmetric, so a single shear column.
MEASURE_COLUMNS = [
    "u_x[µm]",
    "u_y[µm]",
    "F_x[Pa]",
    "F_y[Pa]",
    "sigma_xx[mN/m]",
    "sigma_yy[mN/m]",
    "sigma_shear[mN/m]",
    "mask",
]

COLUMNS = ID_COLUMNS + MEASURE_COLUMNS

# Names of the entries inside the .ntfm zip.
_SAMPLES_NAME = "samples.parquet"
_METADATA_NAME = "metadata.json"


# ---------------------------------------------------------------------------
# arrays -> tidy
# ---------------------------------------------------------------------------

def arrays_to_tidy(
    *,
    displacement_field: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    stress_tensor: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    grid_spacing: float,
    frame_interval: float,
) -> pd.DataFrame:
    """Build the canonical tidy table from co-registered grid arrays.

    All arrays share one downscaled analysis grid ``(ny, nx)`` over ``nt``
    frames. At least one measure must be provided to fix the grid shape; absent
    measures become NaN columns.

    Args:
        displacement_field: ``(nt, ny, nx, 2)`` displacement (µm), ``[..., 0]``
            is x, ``[..., 1]`` is y.
        force_field: ``(nt, ny, nx, 2)`` traction stress (Pa), same component order.
        stress_tensor: ``(nt, ny, nx, 2, 2)`` Cauchy stress (mN/m).
        mask: ``(ny, nx)`` or ``(nt, ny, nx)`` integer region labels
            (0 = background). A single frame is broadcast to every ``t``.
        grid_spacing: physical sample spacing in µm (``pixel_size * downscale_factor``).
        frame_interval: time between frames in min.

    Returns:
        A DataFrame with exactly ``COLUMNS``, sorted by ``(t, row, col)``.
    """
    nt, ny, nx = _infer_grid_shape(
        displacement_field=displacement_field,
        force_field=force_field,
        stress_tensor=stress_tensor,
        mask=mask,
    )

    # Dense (t, row, col) index grid in row-major order.
    t_idx, row_idx, col_idx = np.meshgrid(
        np.arange(nt), np.arange(ny), np.arange(nx), indexing="ij"
    )
    t_idx = t_idx.ravel()
    row_idx = row_idx.ravel()
    col_idx = col_idx.ravel()

    data: Dict[str, np.ndarray] = {
        "t[min]": t_idx * float(frame_interval),
        "y[µm]": row_idx * float(grid_spacing),
        "x[µm]": col_idx * float(grid_spacing),
        "row": row_idx.astype(np.int64),
        "col": col_idx.astype(np.int64),
    }

    nan = np.full(t_idx.shape, np.nan)
    if displacement_field is not None:
        data["u_x[µm]"] = displacement_field[..., 0].ravel()
        data["u_y[µm]"] = displacement_field[..., 1].ravel()
    else:
        data["u_x[µm]"] = nan.copy()
        data["u_y[µm]"] = nan.copy()

    if force_field is not None:
        data["F_x[Pa]"] = force_field[..., 0].ravel()
        data["F_y[Pa]"] = force_field[..., 1].ravel()
    else:
        data["F_x[Pa]"] = nan.copy()
        data["F_y[Pa]"] = nan.copy()

    if stress_tensor is not None:
        data["sigma_xx[mN/m]"] = stress_tensor[..., 0, 0].ravel()
        data["sigma_yy[mN/m]"] = stress_tensor[..., 1, 1].ravel()
        data["sigma_shear[mN/m]"] = stress_tensor[..., 0, 1].ravel()
    else:
        data["sigma_xx[mN/m]"] = nan.copy()
        data["sigma_yy[mN/m]"] = nan.copy()
        data["sigma_shear[mN/m]"] = nan.copy()

    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = np.broadcast_to(mask, (nt, ny, nx))
        data["mask"] = mask.astype(np.int64).ravel()
    else:
        data["mask"] = np.zeros(t_idx.shape, dtype=np.int64)

    return pd.DataFrame(data, columns=COLUMNS)


def _infer_grid_shape(**arrays) -> tuple:
    """Resolve ``(nt, ny, nx)`` from whichever arrays are present, checking agreement."""
    nt = ny = nx = None

    def _check(name, candidate):
        nonlocal nt, ny, nx
        if candidate is None:
            return
        cnt, cny, cnx = candidate
        if nt is None:
            nt, ny, nx = cnt, cny, cnx
        elif (nt, ny, nx) != (cnt, cny, cnx):
            raise ValueError(
                f"Inconsistent grid shape from {name}: expected "
                f"(nt, ny, nx)=({nt}, {ny}, {nx}), got ({cnt}, {cny}, {cnx})"
            )

    disp = arrays.get("displacement_field")
    force = arrays.get("force_field")
    stress = arrays.get("stress_tensor")
    mask = arrays.get("mask")

    if disp is not None:
        if disp.ndim != 4 or disp.shape[-1] != 2:
            raise ValueError(f"displacement_field must be (nt, ny, nx, 2), got {disp.shape}")
        _check("displacement_field", disp.shape[:3])
    if force is not None:
        if force.ndim != 4 or force.shape[-1] != 2:
            raise ValueError(f"force_field must be (nt, ny, nx, 2), got {force.shape}")
        _check("force_field", force.shape[:3])
    if stress is not None:
        if stress.ndim != 5 or stress.shape[-2:] != (2, 2):
            raise ValueError(f"stress_tensor must be (nt, ny, nx, 2, 2), got {stress.shape}")
        _check("stress_tensor", stress.shape[:3])
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            _check("mask", m.shape)
        elif m.ndim != 2:
            raise ValueError(f"mask must be (ny, nx) or (nt, ny, nx), got {m.shape}")
        # 2D mask only constrains (ny, nx); it is broadcast over t later.

    if nt is None:
        raise ValueError("At least one of displacement/force/stress (or a 3D mask) is required")
    return nt, ny, nx


# ---------------------------------------------------------------------------
# tidy -> arrays
# ---------------------------------------------------------------------------

def tidy_to_arrays(df: pd.DataFrame) -> Dict[str, Optional[np.ndarray]]:
    """Reconstruct the grid arrays from a tidy table (inverse of ``arrays_to_tidy``).

    The reconstruction is order-independent: samples are placed by their explicit
    ``row``/``col`` integer indices and their position in the sorted unique ``t``.
    A measure absent from the table maps to ``None``.

    Returns:
        Dict with keys ``displacement_field``, ``force_field``, ``stress_tensor``,
        ``mask`` — each a numpy array or ``None`` if its columns are missing.
    """
    unique_t = np.sort(df["t[min]"].unique())
    nt = len(unique_t)
    ny = int(df["row"].max()) + 1
    nx = int(df["col"].max()) + 1

    t_pos = np.searchsorted(unique_t, df["t[min]"].to_numpy())
    rows = df["row"].to_numpy().astype(np.int64)
    cols = df["col"].to_numpy().astype(np.int64)

    def _scatter(column: str, dtype=float, fill=np.nan):
        arr = np.full((nt, ny, nx), fill, dtype=dtype)
        arr[t_pos, rows, cols] = df[column].to_numpy()
        return arr

    out: Dict[str, Optional[np.ndarray]] = {
        "displacement_field": None,
        "force_field": None,
        "stress_tensor": None,
        "mask": None,
    }

    if {"u_x[µm]", "u_y[µm]"}.issubset(df.columns):
        field = np.empty((nt, ny, nx, 2))
        field[..., 0] = _scatter("u_x[µm]")
        field[..., 1] = _scatter("u_y[µm]")
        out["displacement_field"] = field

    if {"F_x[Pa]", "F_y[Pa]"}.issubset(df.columns):
        field = np.empty((nt, ny, nx, 2))
        field[..., 0] = _scatter("F_x[Pa]")
        field[..., 1] = _scatter("F_y[Pa]")
        out["force_field"] = field

    if {"sigma_xx[mN/m]", "sigma_yy[mN/m]", "sigma_shear[mN/m]"}.issubset(df.columns):
        tensor = np.zeros((nt, ny, nx, 2, 2))
        xx = _scatter("sigma_xx[mN/m]")
        yy = _scatter("sigma_yy[mN/m]")
        shear = _scatter("sigma_shear[mN/m]")
        tensor[..., 0, 0] = xx
        tensor[..., 1, 1] = yy
        tensor[..., 0, 1] = shear
        tensor[..., 1, 0] = shear  # symmetric
        out["stress_tensor"] = tensor

    if "mask" in df.columns:
        out["mask"] = _scatter("mask", dtype=np.int64, fill=0)

    return out


# ---------------------------------------------------------------------------
# Metadata + provenance
# ---------------------------------------------------------------------------

def package_version() -> str:
    """Human-readable napariTFM version, or ``'unknown'`` if not installed."""
    try:
        from importlib.metadata import version

        return version("napariTFM")
    except Exception:
        return "unknown"


def git_provenance(repo_path: Optional[Path] = None) -> Dict[str, object]:
    """Capture ``git_commit`` / ``git_dirty`` for reproducibility.

    Returns ``git_commit=None`` when ``repo_path`` is not a git worktree.
    """
    repo_path = Path(repo_path) if repo_path else Path(__file__).resolve().parent

    def _git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def build_metadata(
    *,
    config: Dict,
    inputs: Optional[Dict] = None,
    labels: Optional[Dict] = None,
    repo_path: Optional[Path] = None,
) -> Dict:
    """Assemble the ``metadata.json`` payload: provenance + resolved config.

    ``config`` is the resolved per-experiment run config (e.g. ``asdict`` of the
    effective ``UnifiedParameters``) — the source of truth for parameters. The
    grid descriptor is intentionally *not* stored: spacing is derivable from
    ``config`` and the table carries ``row``/``col``.

    ``labels`` are the free-form experiment-design tags (``{condition,
    replicate, position, …}``) the §5 aggregator groups by, supplied
    per-experiment in the batch config (ROADMAP §4).
    """
    metadata = {
        "format_version": FORMAT_VERSION,
        "package_version": package_version(),
        "config": config,
        "inputs": inputs or {},
        "labels": labels or {},
    }
    metadata.update(git_provenance(repo_path))
    return metadata


# ---------------------------------------------------------------------------
# Container I/O
# ---------------------------------------------------------------------------

def write_ntfm(path, df: pd.DataFrame, metadata: Dict) -> Path:
    """Write a ``.ntfm`` zip (samples.parquet + metadata.json)."""
    path = Path(path)

    parquet_buf = io.BytesIO()
    df.to_parquet(parquet_buf, engine="pyarrow", index=False)

    metadata_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_SAMPLES_NAME, parquet_buf.getvalue())
        zf.writestr(_METADATA_NAME, metadata_bytes)
    return path


def read_ntfm(path) -> tuple:
    """Read a ``.ntfm`` zip, returning ``(df, metadata)``."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(_SAMPLES_NAME) as f:
            df = pd.read_parquet(io.BytesIO(f.read()), engine="pyarrow")
        with zf.open(_METADATA_NAME) as f:
            metadata = json.loads(f.read().decode("utf-8"))
    return df, metadata


# One representative column per measured stage. The writer always emits every
# measure column (all-NaN when the stage wasn't run), so presence is judged by
# "the column exists and is not entirely NaN".
_STAGE_MEASURE_COLUMN = {
    "displacement": "u_x[µm]",
    "force": "F_x[Pa]",
    "stress": "sigma_xx[mN/m]",
}

# All measure columns belonging to each stage.
# Keys match _STAGE_MEASURE_COLUMN; used by merge_tidy_preserving.
_STAGE_COLUMNS = {
    "displacement": ["u_x[µm]", "u_y[µm]"],
    "force": ["F_x[Pa]", "F_y[Pa]"],
    "stress": ["sigma_xx[mN/m]", "sigma_yy[mN/m]", "sigma_shear[mN/m]"],
}

# Key columns used when aligning rows between two tidy tables during merge.
_MERGE_KEYS = ["t[min]", "row", "col"]


_populated_measures_cache: dict = {}


def populated_measures(path) -> set:
    """Return which measured stages have real (non-all-NaN) data in a ``.ntfm``.

    The result is a subset of ``{"displacement", "force", "stress"}``. A missing
    or unreadable container yields the empty set, so callers can treat "no
    output yet" and "can't tell" identically. This is the per-output truth the
    experiments-list stage-status reads (P3); ``mask`` is an input, not a stage.

    Results are cached by ``(path, mtime_ns, size)`` so repeated calls on an
    unchanged file are O(1).  The parquet column statistics (row-group metadata)
    are read instead of the full table, reducing per-call I/O from ~200 ms to
    ~1 ms.  The cache self-invalidates when the file is rewritten (new mtime/size).
    """
    import pyarrow.parquet as pq

    path = Path(path)

    # Stat the file; propagate missing-file as empty set.
    try:
        st = path.stat()
    except OSError:
        return set()

    cache_key = (str(path), st.st_mtime_ns, st.st_size)

    # Cache hit: return a copy so callers cannot mutate the stored frozenset.
    cached = _populated_measures_cache.get(cache_key)
    if cached is not None:
        return set(cached)

    # Compute via parquet column statistics (no full-table read).
    try:
        with zipfile.ZipFile(path, "r") as zf:
            data = zf.read(_SAMPLES_NAME)
        pf = pq.ParquetFile(io.BytesIO(data))
        md = pf.metadata
        names = pf.schema_arrow.names
        present = set()
        for stage, column in _STAGE_MEASURE_COLUMN.items():
            if column not in names:
                continue
            ci = names.index(column)
            nulls = 0
            have_stats = True
            for rg in range(md.num_row_groups):
                st_col = md.row_group(rg).column(ci).statistics
                if st_col is None:
                    have_stats = False
                    break
                nulls += st_col.null_count
            if have_stats:
                if nulls < md.num_rows:  # at least one non-null value
                    present.add(stage)
            else:
                present.add(stage)  # stats unavailable -> conservatively present
    except Exception:
        return set()

    # Guard against unbounded cache growth.
    if len(_populated_measures_cache) >= 512:
        _populated_measures_cache.clear()

    _populated_measures_cache[cache_key] = frozenset(present)
    return set(present)


# ---------------------------------------------------------------------------
# Tidy-table merge (preserve prior-run stages on re-write)
# ---------------------------------------------------------------------------

def merge_tidy_preserving(new_df: pd.DataFrame, old_df: pd.DataFrame) -> pd.DataFrame:
    """Merge two tidy tables, preserving prior-run stages absent from the new run.

    When a pipeline stage is re-run with some upstream stages absent (e.g. a
    force-only resume where ``displacement_result`` is ``None``),
    ``arrays_to_tidy`` fills all-NaN placeholders for the missing stages.  A
    naive overwrite would PERMANENTLY ERASE previously-saved data.

    This helper fills each stage column-group that is all-NaN (or absent) in
    ``new_df`` but PRESENT and NOT all-NaN (or not all-zero for ``mask``) in
    ``old_df`` with values from ``old_df``.  Stages already populated in
    ``new_df`` are never touched — the new run always wins.

    Merge is performed via a left-join on ``(t[min], row, col)`` so
    row-ordering differences between the two tables do not matter.

    Grid compatibility: the two tables must share the same set of unique
    ``t[min]`` values (within a small float tolerance) and the same ``row``/
    ``col`` max extents.  On mismatch a warning is printed and ``new_df`` is
    returned unchanged (new data wins, no crash).

    Stage column groups (displacement / force / stress) use
    ``_STAGE_MEASURE_COLUMN`` as the representative-column presence indicator
    and ``_STAGE_COLUMNS`` for the full list of columns to copy.  ``mask``
    uses an all-zero / absence check.

    Args:
        new_df: The tidy table produced by the current run.
        old_df: The tidy table read from the existing on-disk container.

    Returns:
        A tidy DataFrame with ``COLUMNS`` column order, possibly enriched with
        preserved stages from ``old_df``.
    """
    # --- Grid compatibility check ---
    new_t = np.sort(new_df["t[min]"].unique())
    old_t = np.sort(old_df["t[min]"].unique())
    if len(new_t) != len(old_t) or not np.allclose(new_t, old_t, atol=1e-9):
        print(
            "merge_tidy_preserving: t[min] values differ between new and existing "
            "containers — skipping merge, new data wins."
        )
        return new_df

    new_row_max = int(new_df["row"].max())
    old_row_max = int(old_df["row"].max())
    new_col_max = int(new_df["col"].max())
    old_col_max = int(old_df["col"].max())
    if new_row_max != old_row_max or new_col_max != old_col_max:
        print(
            f"merge_tidy_preserving: grid extents differ "
            f"(new row/col max {new_row_max}/{new_col_max}, "
            f"existing {old_row_max}/{old_col_max}) — skipping merge, new data wins."
        )
        return new_df

    # Align old_df rows to new_df's ordering via a left-join on the mesh keys.
    old_aligned = new_df[_MERGE_KEYS].merge(
        old_df, on=_MERGE_KEYS, how="left", suffixes=("", "_old")
    )

    result_df = new_df.copy()

    # Fill measure stages (displacement / force / stress).
    for stage, rep_col in _STAGE_MEASURE_COLUMN.items():
        stage_cols = _STAGE_COLUMNS[stage]
        # Stage absent or all-NaN in new_df?
        new_absent = (rep_col not in new_df.columns) or new_df[rep_col].isna().all()
        # Stage present (non-NaN) in old_df?
        old_present = (rep_col in old_df.columns) and (not old_df[rep_col].isna().all())
        if new_absent and old_present:
            for col in stage_cols:
                if col in old_aligned.columns:
                    result_df[col] = old_aligned[col].to_numpy()

    # Fill mask (all-zero / absence check).
    new_mask_absent = ("mask" not in new_df.columns) or (new_df["mask"] == 0).all()
    old_mask_present = ("mask" in old_df.columns) and not (old_df["mask"] == 0).all()
    if new_mask_absent and old_mask_present and "mask" in old_aligned.columns:
        result_df["mask"] = old_aligned["mask"].to_numpy()

    # Preserve COLUMNS ordering, include only columns present.
    ordered_cols = [c for c in COLUMNS if c in result_df.columns]
    return result_df[ordered_cols]


# ---------------------------------------------------------------------------
# CSV export (lossy, additional)
# ---------------------------------------------------------------------------

def write_csv(path, df: pd.DataFrame, drop_displacement: bool = False) -> Path:
    """Export the tidy table to CSV (human/Excel portability, lossy on dtypes).

    Set ``drop_displacement`` to omit ``u_x``/``u_y`` when only results are wanted.
    """
    path = Path(path)
    if drop_displacement:
        df = df.drop(columns=["u_x[µm]", "u_y[µm]"], errors="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _csv_export_columns(pf) -> list:
    """Pick the CSV columns from a parquet file: ids always, measures when present.

    Identifier columns are always emitted. A measure column is kept only when it
    carries real data (so a stage that never ran — its column is all-NaN — is
    dropped, and an all-zero ``mask`` from a no-mask run is dropped too), giving
    the "stress / mask when present" behaviour. Presence is judged from row-group
    statistics (no full-table read); when stats are unavailable the column is
    conservatively kept. Returned in canonical ``COLUMNS`` order.
    """
    md = pf.metadata
    names = pf.schema_arrow.names
    keep: list = []
    for col in COLUMNS:
        if col not in names:
            continue
        if col in ID_COLUMNS:
            keep.append(col)
            continue
        ci = names.index(col)
        if col == "mask":
            # Keep mask only when some sample is non-zero (a real mask exists).
            present = False
            for rg in range(md.num_row_groups):
                st = md.row_group(rg).column(ci).statistics
                if st is None or not st.has_min_max:
                    present = True  # can't tell -> keep
                    break
                if st.min != 0 or st.max != 0:
                    present = True
                    break
            if present:
                keep.append(col)
            continue
        # Measure column: keep when at least one value is non-null.
        nulls = 0
        have_stats = True
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(ci).statistics
            if st is None:
                have_stats = False
                break
            nulls += st.null_count
        if not have_stats or nulls < md.num_rows:
            keep.append(col)
    return keep


def export_ntfm_to_csv(ntfm_path, csv_path, *, chunk_rows: int = 200_000) -> Path:
    """Stream a ``.ntfm``'s tidy table to a full per-pixel CSV dump.

    Writes every ``(t, y, x)`` grid sample's measures — ``u_x, u_y, F_x, F_y``
    (plus ``sigma_*`` and ``mask`` *when present*) — for every frame. The parquet
    inside the container is read and the CSV written in row-batches, so the whole
    CSV text is never materialised in memory at once (these dumps can be large).

    Columns absent from the data are omitted (see :func:`_csv_export_columns`).
    Raises ``FileNotFoundError`` when no container exists yet.
    """
    import pyarrow.parquet as pq

    ntfm_path = Path(ntfm_path)
    csv_path = Path(csv_path)
    if not ntfm_path.exists():
        raise FileNotFoundError(f"No .ntfm container to export at {ntfm_path}")

    with zipfile.ZipFile(ntfm_path, "r") as zf:
        data = zf.read(_SAMPLES_NAME)
    pf = pq.ParquetFile(io.BytesIO(data))

    keep = _csv_export_columns(pf)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        for batch in pf.iter_batches(batch_size=chunk_rows, columns=keep):
            frame = batch.to_pandas()
            frame = frame[[c for c in keep if c in frame.columns]]
            frame.to_csv(fh, header=first, index=False)
            first = False
        if first:  # no rows at all — still emit a header row for the schema
            pd.DataFrame(columns=keep).to_csv(fh, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Result-dataclass adapter (bridges the analysis pipeline to the converter)
# ---------------------------------------------------------------------------

def dataframe_from_results(
    *,
    displacement_result=None,
    force_result=None,
    stress_result=None,
    mask: Optional[np.ndarray] = None,
    grid_spacing: Optional[float] = None,
    frame_interval: Optional[float] = None,
) -> pd.DataFrame:
    """Build the tidy table from ``DisplacementResult`` / ``FTTCResult`` / ``MSMResult``.

    ``grid_spacing`` (µm) and ``frame_interval`` (min) default to the values
    carried in the results' ``physical_scale`` — every stage reports the same
    ``grid_spacing = pixel_size * downscale_factor``, so any present result fixes them.
    """
    disp = getattr(displacement_result, "displacement_field", None)
    force = getattr(force_result, "force_field", None)
    stress = getattr(stress_result, "stress_tensor", None)

    if grid_spacing is None or frame_interval is None:
        scale = _first_physical_scale(displacement_result, force_result, stress_result)
        if grid_spacing is None:
            grid_spacing = scale["grid_spacing"]
        if frame_interval is None:
            frame_interval = scale["time_interval"]

    return arrays_to_tidy(
        displacement_field=disp,
        force_field=force,
        stress_tensor=stress,
        mask=mask,
        grid_spacing=grid_spacing,
        frame_interval=frame_interval,
    )


def _first_physical_scale(*results) -> Dict:
    for result in results:
        scale = getattr(result, "physical_scale", None)
        if scale:
            return scale
    raise ValueError("No result provided to infer grid_spacing / frame_interval from")


def results_to_ntfm(
    path,
    *,
    config: Dict,
    displacement_result=None,
    force_result=None,
    stress_result=None,
    mask: Optional[np.ndarray] = None,
    inputs: Optional[Dict] = None,
    labels: Optional[Dict] = None,
    repo_path: Optional[Path] = None,
    merge_existing: bool = True,
) -> Path:
    """End-to-end: write one experiment's results to a ``.ntfm`` container.

    ``config`` is the resolved per-experiment run config (``asdict`` of the
    effective ``UnifiedParameters``). ``labels`` are the per-experiment
    design tags the §5 aggregator groups by.

    When ``merge_existing`` is ``True`` (the default) and ``path`` already
    exists, the existing container is read and any stage that is absent/all-NaN
    in the new data but present in the existing container is preserved via
    ``merge_tidy_preserving``.  Set ``merge_existing=False`` to force a pure
    overwrite (previously-saved stages will be erased).
    """
    df = dataframe_from_results(
        displacement_result=displacement_result,
        force_result=force_result,
        stress_result=stress_result,
        mask=mask,
    )

    if merge_existing and Path(path).exists():
        try:
            old_df, _ = read_ntfm(path)
            df = merge_tidy_preserving(df, old_df)
        except Exception as e:
            print(
                f"results_to_ntfm: could not read existing container for merge "
                f"({path!r}): {e!r} — writing new data without merging."
            )

    metadata = build_metadata(
        config=config, inputs=inputs, labels=labels, repo_path=repo_path
    )
    return write_ntfm(path, df, metadata)
