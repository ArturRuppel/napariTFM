"""Native TFM container and the tidy long-format table.

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

The pipeline's hot path is **array-native**: dense stage arrays go straight to
the container (:func:`write_series_ntfm`) and come straight back
(:func:`read_series_ntfm`); the on-disk merge (:func:`merge_arrays`) is a
per-stage dict merge. The tidy long-format table (one row per ``(t, y, x)``
sample, units inline in the column name) is a **lazy export/import adapter** for
callers that genuinely want a DataFrame: :func:`arrays_to_tidy` /
:func:`tidy_to_arrays` (and the ``write_ntfm`` / ``read_ntfm`` wrappers over the
array path). ``arrays -> tidy -> arrays`` is lossless (modulo the float32 storage
step); reconstruction is trivial because ``row``/``col`` are explicit integer
columns. Coordinate frame is image convention: ``x`` = column (+x right), ``y`` =
row (**+y down**, no flip to physical y-up).
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
    per-experiment in the batch config.
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


# Series name -> the arrays-dict key it is built from. The OME container stores
# dense per-stage arrays, so the write/read hot path is array-native; the tidy
# table (arrays_to_tidy / tidy_to_arrays) is only a lazy export/import adapter.
_SERIES_ARRAY_KEY = {
    "displacement": "displacement_field",
    "traction": "force_field",
    "stress": "stress_tensor",
}


def _stage_series_channels(name: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """A stage array as ``(T, C, Y, X)`` float32, or None if absent/all-NaN.

    displacement/traction are ``(T, Y, X, 2)`` (x, y channels); stress is
    ``(T, Y, X, 2, 2)`` stored as the three independent channels (xx, yy, shear).
    An all-NaN array means the stage never ran, so no series is written for it.
    """
    if arr is None:
        return None
    if name == "stress":
        stacked = np.stack([arr[..., 0, 0], arr[..., 1, 1], arr[..., 0, 1]], axis=1)
    else:
        stacked = np.moveaxis(arr, -1, 1)  # (T, Y, X, 2) -> (T, 2, Y, X)
    if np.isnan(stacked).all():
        return None
    return stacked.astype(np.float32)


def _mask_series(mask: Optional[np.ndarray], nt: int) -> Optional[np.ndarray]:
    """A mask as ``(T, Y, X)`` uint16, or None when no real mask is present.

    A 2D mask is broadcast across the ``nt`` frames; an all-zero mask means "no
    mask supplied" and yields None (no series written).
    """
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 2:
        m = np.broadcast_to(m, (nt, *m.shape))
    if not (m != 0).any():
        return None
    return np.nan_to_num(m).astype(np.uint16)


def write_series_ntfm(
    path,
    *,
    displacement_field: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    stress_tensor: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    grid_spacing: float,
    frame_interval: float,
    metadata: Dict,
) -> Path:
    """Write co-registered grid arrays to the multi-series OME-TIFF (array-native).

    One series per populated stage (``displacement`` / ``traction`` / ``stress``,
    each ``(T, C, Y, X)`` float32) plus a ``mask`` series (``(T, Y, X)`` uint16)
    when a real mask is present. Channel names carry units; the first series also
    carries ``PhysicalSizeX/Y`` (µm), ``TimeIncrement`` (min) and the full
    ``metadata`` dict as a JSON ``Description`` — so the file is self-describing
    and Fiji-openable. This is the sole writer; ``write_ntfm`` is a tidy adapter.
    """
    path = Path(path)
    arrays = {
        "displacement": displacement_field,
        "traction": force_field,
        "stress": stress_tensor,
    }

    series: list = []
    nt = None
    for name, columns in _SERIES_COLUMNS.items():
        arr = _stage_series_channels(name, arrays[name])
        if arr is not None:
            series.append((name, columns, arr))
            if nt is None:
                nt = arr.shape[0]

    if nt is not None:
        mask_arr = _mask_series(mask, nt)
        if mask_arr is not None:
            series.append((_MASK_SERIES, [_MASK_SERIES], mask_arr))

    if not series:
        raise ValueError("write_series_ntfm: nothing to write (no populated stage or mask)")

    description = json.dumps(metadata, ensure_ascii=False, default=str)
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


def write_ntfm(path, df: pd.DataFrame, metadata: Dict) -> Path:
    """Tidy-table entry point to the container (adapter over ``write_series_ntfm``).

    The canonical on-disk form is dense arrays, so this just converts the tidy
    table to arrays once and delegates. Kept for the tidy-table API surface (and
    its tests); the pipeline itself writes arrays directly via
    :func:`write_series_ntfm`.
    """
    arrays = tidy_to_arrays(df)
    return write_series_ntfm(
        path,
        displacement_field=arrays["displacement_field"],
        force_field=arrays["force_field"],
        stress_tensor=arrays["stress_tensor"],
        mask=arrays["mask"],
        grid_spacing=_grid_step(df["x[µm]"].to_numpy()),
        frame_interval=_grid_step(df["t[min]"].to_numpy()),
        metadata=metadata,
    )


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


def read_series_ntfm(path) -> tuple:
    """Read the OME-TIFF container to dense arrays (array-native; the hot path).

    Returns ``(arrays, grid_spacing, frame_interval, metadata)`` where ``arrays``
    is a dict with keys ``displacement_field`` / ``force_field`` /
    ``stress_tensor`` / ``mask`` (each an array or ``None`` when its series is
    absent). Measures are cast to float64; the mask to int64. This is what the
    pipeline (stage-resume, on-demand viewer load) reads directly, without ever
    materializing the tidy table.
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

    arrays = {
        "displacement_field": displacement_field,
        "force_field": force_field,
        "stress_tensor": stress_tensor,
        "mask": mask,
    }
    return arrays, grid_spacing, frame_interval, metadata


def read_ntfm(path) -> tuple:
    """Tidy-table view of the container, returning ``(df, metadata)``.

    Adapter over :func:`read_series_ntfm`: the full tidy schema (every
    ``COLUMNS`` entry) is reconstructed, so stages whose series is absent become
    all-NaN columns. Kept for the tidy-table API and its tests; the pipeline
    reads arrays directly via :func:`read_series_ntfm`.
    """
    arrays, grid_spacing, frame_interval, metadata = read_series_ntfm(path)
    df = arrays_to_tidy(
        **arrays,
        grid_spacing=grid_spacing,
        frame_interval=frame_interval,
    )
    return df, metadata


# The stage arrays that the merge preserves (mask is handled separately — it
# never survives from the old container).
_STAGE_ARRAY_KEYS = ("displacement_field", "force_field", "stress_tensor")

# Pipeline dependency chain in array-key terms (mirrors DataManager._DOWNSTREAM):
# a stage freshly present in a write makes every stage below it stale, so the
# merge must not resurrect those from disk. See merge_arrays / B-3.
_DOWNSTREAM_ARRAY_KEYS = {
    "displacement_field": ("force_field", "stress_tensor"),
    "force_field": ("stress_tensor",),
}


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
# Array merge (preserve prior-run stages on re-write)
# ---------------------------------------------------------------------------

def merge_arrays(new: Dict, old: Dict) -> Dict:
    """Merge two stage-array dicts, preserving prior stages absent from the new run.

    ``new`` / ``old`` are dicts keyed by ``displacement_field`` / ``force_field``
    / ``stress_tensor`` / ``mask`` (each an array or ``None``), as produced by
    :func:`read_series_ntfm`. When a stage is re-run with some upstream stages
    absent (e.g. a force-only resume where ``displacement_result`` is ``None``),
    a naive overwrite would PERMANENTLY ERASE previously-saved data. Each stage
    that is absent (``None``) or all-NaN in ``new`` but present (non-all-NaN) in
    ``old`` is filled from ``old``; stages already populated in ``new`` always
    win. This is a per-stage dict merge — the merge granularity is the stage,
    which is exactly what this expresses (no per-sample relational join).

    **Downstream invalidation (B-3).** A stage *present* in ``new`` was (re)computed
    this run, which makes every stage below it in the pipeline
    (:data:`_DOWNSTREAM_ARRAY_KEYS`) stale. Such a downstream stage, when absent
    from ``new``, is therefore *not* resurrected from ``old`` — otherwise a fresh
    displacement would be paired on disk with a force computed from the *prior*
    displacement. This is what distinguishes an interactive upstream re-run
    (displacement present, force absent → drop the stale force) from a legitimate
    force-only resume (displacement absent → preserve it). Only stages with no
    fresh upstream in this write are eligible for preservation.

    The ``mask`` is deliberately *not* preserved: unlike the measure stages
    (where all-NaN unambiguously means "not computed this run"), an absent/
    all-zero mask only ever means "no mask supplied", never a deliberately-empty
    region, so the new mask always wins.

    Grid compatibility: the present stages must share ``(T, Y, X)``. On mismatch
    a warning is printed and ``new`` is returned unchanged (new data wins, no
    crash).
    """
    def grid_of(arrays: Dict):
        for key in _STAGE_ARRAY_KEYS:
            arr = arrays.get(key)
            if arr is not None:
                return arr.shape[:3]  # (T, Y, X)
        return None

    new_grid, old_grid = grid_of(new), grid_of(old)
    if new_grid is not None and old_grid is not None and new_grid != old_grid:
        print(
            f"merge_arrays: grid extents differ (new {new_grid}, existing "
            f"{old_grid}) — skipping merge, new data wins."
        )
        return new

    def present(arr):
        return arr is not None and not np.all(np.isnan(arr))

    merged = dict(new)

    # Stages made stale by a freshly-written upstream stage: never resurrect these.
    stale_downstream = set()
    for key in _STAGE_ARRAY_KEYS:
        if present(merged.get(key)):
            stale_downstream.update(_DOWNSTREAM_ARRAY_KEYS.get(key, ()))

    for key in _STAGE_ARRAY_KEYS:
        if key in stale_downstream:
            continue
        if not present(merged.get(key)) and present(old.get(key)):
            merged[key] = old[key]
    return merged


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
    arrays, grid_spacing, frame_interval = _arrays_from_results(
        displacement_result=displacement_result,
        force_result=force_result,
        stress_result=stress_result,
        mask=mask,
        grid_spacing=grid_spacing,
        frame_interval=frame_interval,
    )
    return arrays_to_tidy(**arrays, grid_spacing=grid_spacing, frame_interval=frame_interval)


def _arrays_from_results(
    *,
    displacement_result=None,
    force_result=None,
    stress_result=None,
    mask: Optional[np.ndarray] = None,
    grid_spacing: Optional[float] = None,
    frame_interval: Optional[float] = None,
) -> tuple:
    """Pull stage arrays + grid_spacing/frame_interval out of the result objects.

    Returns ``(arrays, grid_spacing, frame_interval)`` where ``arrays`` is the
    ``{stage_key: array}`` dict :func:`write_series_ntfm` / :func:`merge_arrays`
    consume. Missing spacing/interval default to the results' ``physical_scale``.
    """
    arrays = {
        "displacement_field": getattr(displacement_result, "displacement_field", None),
        "force_field": getattr(force_result, "force_field", None),
        "stress_tensor": getattr(stress_result, "stress_tensor", None),
        "mask": mask,
    }
    if grid_spacing is None or frame_interval is None:
        scale = _first_physical_scale(displacement_result, force_result, stress_result)
        if grid_spacing is None:
            grid_spacing = scale["grid_spacing"]
        if frame_interval is None:
            frame_interval = scale["time_interval"]
    return arrays, grid_spacing, frame_interval


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

    Array-native: results become stage arrays, the merge (when ``path`` exists
    and ``merge_existing`` is ``True``, the default) preserves any prior stage
    absent from this write via :func:`merge_arrays`, and the arrays are written
    straight to the OME-TIFF — no tidy table is materialized on this path. Set
    ``merge_existing=False`` to force a pure overwrite (prior stages erased).
    """
    arrays, grid_spacing, frame_interval = _arrays_from_results(
        displacement_result=displacement_result,
        force_result=force_result,
        stress_result=stress_result,
        mask=mask,
    )

    if merge_existing and Path(path).exists():
        try:
            old_arrays, _, _, _ = read_series_ntfm(path)
            arrays = merge_arrays(arrays, old_arrays)
        except Exception as e:
            print(
                f"results_to_ntfm: could not read existing container for merge "
                f"({path!r}): {e!r} — writing new data without merging."
            )

    metadata = build_metadata(
        config=config, inputs=inputs, labels=labels, repo_path=repo_path
    )
    return write_series_ntfm(
        path,
        grid_spacing=grid_spacing,
        frame_interval=frame_interval,
        metadata=metadata,
        **arrays,
    )
