"""Native TFM container and the tidy long-format table (ROADMAP §1).

The canonical *on-disk* form is a single multi-series **OME-TIFF**
(``<experiment>.ome.tif``) that Fiji/ImageJ open on a double-click via bundled
Bio-Formats — data and provenance in one self-describing file::

    experiment.ome.tif            # one OME-TIFF, up to four named series
    ├── series "displacement"     # (T, C, Y, X) float32 — u_x[µm], u_y[µm]
    ├── series "traction"         # (T, C, Y, X) float32 — F_x[Pa], F_y[Pa]
    ├── series "stress"           # (T, C, Y, X) float32 — sigma_xx/yy/shear[mN/m]
    ├── series "mask"             # (T, Y, X) uint16 — region labels (binarised 0/1)
    └── OME-XML                   # channel names+units, PhysicalSizeX/Y (µm),
                                  # TimeIncrement (min), and the provenance dict
                                  # (config / labels / git) as a JSON Description

A series is written only when its stage ran and is not entirely NaN, so the set
of present series *is* the per-stage truth (no all-NaN placeholders on disk).
Off-mask NaN inside a written series is preserved. Pixels are float32 — beyond
any TFM measurement precision; ``read_ntfm`` casts measures back to float64.

The tidy long-format table (one row per ``(t, y, x)`` sample, units inline in
the column name) remains the *in-memory* interchange: ``read_ntfm`` returns it,
``write_ntfm`` accepts it. ``arrays -> tidy -> arrays`` is lossless (modulo the
float32 storage step); reconstruction is trivial because ``row``/``col`` are
explicit integer columns. Coordinate frame is image convention: ``x`` = column
(+x right), ``y`` = row (**+y down**, no flip to physical y-up).
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FORMAT_VERSION = "2.0"

# Identifiers (Iris vocabulary) — locate every sample.
ID_COLUMNS = ["t[min]", "y[µm]", "x[µm]", "row", "col"]

# Measures — the full 2D field at each sample. NaN where a measure is absent
# (e.g. off-mask stress nodes). The stress tensor is symmetric, so a single shear column.
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

# OME-TIFF series layout: series name -> the tidy measure columns it carries,
# in channel order. The canonical write order is displacement, traction, stress.
_SERIES_COLUMNS = {
    "displacement": ["u_x[µm]", "u_y[µm]"],
    "traction": ["F_x[Pa]", "F_y[Pa]"],
    "stress": ["sigma_xx[mN/m]", "sigma_yy[mN/m]", "sigma_shear[mN/m]"],
}
_MASK_SERIES = "mask"

# populated_measures speaks the stage vocabulary; the traction series is the
# "force" stage. mask is an input, not a stage, so it is excluded.
_SERIES_TO_STAGE = {"displacement": "displacement", "traction": "force", "stress": "stress"}


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

def _grid_step(values: np.ndarray, default: float = 1.0) -> float:
    """Smallest positive spacing between sorted unique ``values`` (else default)."""
    uniq = np.sort(np.unique(values))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    return float(diffs.min()) if diffs.size else float(default)


def _measure_series_array(df: pd.DataFrame, columns: list) -> Optional[np.ndarray]:
    """Stack a series' tidy columns into ``(T, C, Y, X)`` float32, or None.

    Returns None when the stage is absent (columns missing) or never ran (every
    value NaN), so only real stages become on-disk series.
    """
    if not set(columns).issubset(df.columns):
        return None
    arrays = ntfm_tidy_to_series_channels(df, columns)
    if arrays is None or np.isnan(arrays).all():
        return None
    return arrays.astype(np.float32)


def ntfm_tidy_to_series_channels(df: pd.DataFrame, columns: list) -> Optional[np.ndarray]:
    """Scatter the given tidy columns onto the ``(T, C, Y, X)`` grid."""
    unique_t = np.sort(df["t[min]"].unique())
    nt = len(unique_t)
    ny = int(df["row"].max()) + 1
    nx = int(df["col"].max()) + 1
    t_pos = np.searchsorted(unique_t, df["t[min]"].to_numpy())
    rows = df["row"].to_numpy().astype(np.int64)
    cols = df["col"].to_numpy().astype(np.int64)
    out = np.full((nt, len(columns), ny, nx), np.nan, dtype=float)
    for ci, col in enumerate(columns):
        out[t_pos, ci, rows, cols] = df[col].to_numpy()
    return out


def write_ntfm(path, df: pd.DataFrame, metadata: Dict) -> Path:
    """Write the container as a single multi-series OME-TIFF.

    One series per populated stage (``displacement`` / ``traction`` / ``stress``,
    each ``(T, C, Y, X)`` float32) plus a ``mask`` series (``(T, Y, X)`` uint16)
    when a real mask is present. Channel names carry units; the first series also
    carries ``PhysicalSizeX/Y`` (µm), ``TimeIncrement`` (min) and the full
    ``metadata`` dict as a JSON ``Description`` — so the file is self-describing
    and Fiji-openable.
    """
    path = Path(path)

    grid_spacing = _grid_step(df["x[µm]"].to_numpy())
    frame_interval = _grid_step(df["t[min]"].to_numpy())
    description = json.dumps(metadata, ensure_ascii=False, default=str)

    # Build the ordered list of (name, channel-names, array) to write.
    series: list = []
    for name, columns in _SERIES_COLUMNS.items():
        arr = _measure_series_array(df, columns)
        if arr is not None:
            series.append((name, columns, arr))

    if "mask" in df.columns and (df["mask"].to_numpy() != 0).any():
        mask_arr = ntfm_tidy_to_series_channels(df, ["mask"])[:, 0, :, :]
        series.append((_MASK_SERIES, [_MASK_SERIES], np.nan_to_num(mask_arr).astype(np.uint16)))

    if not series:
        raise ValueError("write_ntfm: nothing to write (no populated stage or mask)")

    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, ome=True) as tw:
        for idx, (name, channel_names, arr) in enumerate(series):
            axes = "TYX" if arr.ndim == 3 else "TCYX"
            md: Dict = {"axes": axes, "Name": name}
            if arr.ndim == 4:
                md["Channel"] = {"Name": list(channel_names)}
            if idx == 0:  # physical scale + provenance ride on the first series
                md.update(
                    PhysicalSizeX=grid_spacing,
                    PhysicalSizeXUnit="µm",
                    PhysicalSizeY=grid_spacing,
                    PhysicalSizeYUnit="µm",
                    TimeIncrement=frame_interval,
                    TimeIncrementUnit="min",
                    Description=description,
                )
            tw.write(arr, photometric="minisblack", compression="zlib", metadata=md)
    return path


def _parse_ome(xml: str) -> tuple:
    """Pull ``(metadata, grid_spacing, frame_interval)`` from the OME-XML.

    The provenance dict is the first ``Image`` element's JSON ``Description``;
    spacing/interval come from that image's ``Pixels`` attributes.
    """
    root = ET.fromstring(xml)
    ns = {"o": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    def find(elem, tag):
        return elem.find(f"o:{tag}", ns) if ns else elem.find(tag)

    image = find(root, "Image")
    metadata: Dict = {}
    desc = find(image, "Description") if image is not None else None
    if desc is not None and desc.text:
        try:
            metadata = json.loads(desc.text)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    grid_spacing, frame_interval = 1.0, 1.0
    pixels = find(image, "Pixels") if image is not None else None
    if pixels is not None:
        grid_spacing = float(pixels.get("PhysicalSizeX") or 1.0)
        frame_interval = float(pixels.get("TimeIncrement") or 1.0)
    return metadata, grid_spacing, frame_interval


def read_ntfm(path) -> tuple:
    """Read an OME-TIFF container, returning ``(df, metadata)``.

    The full tidy schema (every ``COLUMNS`` entry) is reconstructed: stages whose
    series is absent are refilled as all-NaN columns and measures are cast back to
    float64, so the returned frame matches what :func:`write_ntfm` was given.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        by_name = {s.name: s for s in tf.series}
        metadata, grid_spacing, frame_interval = _parse_ome(tf.ome_metadata or "")

        def series_array(name):
            return by_name[name].asarray() if name in by_name else None

        disp = series_array("displacement")
        trac = series_array("traction")
        stress_ch = series_array("stress")
        mask_arr = series_array(_MASK_SERIES)

    # tifffile squeezes singleton axes (a one-frame stack reads back as
    # (C, Y, X)); reshape to canonical (T, C, Y, X) using the known channel count
    # — Y, X are always the trailing two axes.
    def _as_tcyx(arr, n_channels):
        if arr is None:
            return None
        ny, nx = arr.shape[-2], arr.shape[-1]
        return np.asarray(arr, dtype=np.float64).reshape(-1, n_channels, ny, nx)

    disp = _as_tcyx(disp, 2)
    trac = _as_tcyx(trac, 2)
    stress_ch = _as_tcyx(stress_ch, 3)

    displacement_field = np.moveaxis(disp, 1, -1) if disp is not None else None
    force_field = np.moveaxis(trac, 1, -1) if trac is not None else None

    stress_tensor = None
    if stress_ch is not None:
        nt, _, ny, nx = stress_ch.shape
        stress_tensor = np.empty((nt, ny, nx, 2, 2))
        stress_tensor[..., 0, 0] = stress_ch[:, 0]
        stress_tensor[..., 1, 1] = stress_ch[:, 1]
        stress_tensor[..., 0, 1] = stress_ch[:, 2]
        stress_tensor[..., 1, 0] = stress_ch[:, 2]

    mask = None
    if mask_arr is not None:
        ny, nx = mask_arr.shape[-2], mask_arr.shape[-1]
        mask = np.asarray(mask_arr).reshape(-1, ny, nx).astype(np.int64)

    df = arrays_to_tidy(
        displacement_field=displacement_field,
        force_field=force_field,
        stress_tensor=stress_tensor,
        mask=mask,
        grid_spacing=grid_spacing,
        frame_interval=frame_interval,
    )
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
    """Return which measured stages have real data in the container.

    The result is a subset of ``{"displacement", "force", "stress"}``. Because a
    series is written only when its stage ran with non-all-NaN data, *presence of
    the series is the per-stage truth* — no pixel read is needed, only the series
    list from the OME-XML header. A missing or unreadable container yields the
    empty set, so callers treat "no output yet" and "can't tell" identically.
    This is what the experiments-list stage-status reads (P3); ``mask`` is an
    input, not a stage, and is excluded.

    Results are cached by ``(path, mtime_ns, size)``; the cache self-invalidates
    when the file is rewritten (new mtime/size).
    """
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

    # Read series names from the header only (no pixel data).
    try:
        with tifffile.TiffFile(path) as tf:
            names = {s.name for s in tf.series}
        present = {
            stage for series, stage in _SERIES_TO_STAGE.items() if series in names
        }
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
    """Build the tidy table from ``DisplacementResult`` / ``FTTCResult`` / ``StressResult``.

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
