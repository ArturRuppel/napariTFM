"""Resolve where batch-derived output lands — the ``TFM_data/`` bucket.

All derived output goes into a bucket named ``TFM_data``, sitting *next to* the
raw input — never mixed with it. Two modes:

- **Processed root set** → the input-folder tree is *mirrored* under it (each
  folder reproduced relative to the *longest common parent* of all input
  folders; disconnected roots or a single folder fall back to folder
  *basenames* flat under the processed root, with a warning). The ``TFM_data``
  bucket then sits where the input data would land in that cloned tree, with
  the experiment's own basename nested under it so siblings can't collide.
- **Processed root empty → in-place**: ``TFM_data`` is created as a *sibling*
  of each input folder (not nested inside it), again namespaced by the
  experiment's basename.

The resolver is pure path arithmetic so it is fully unit-testable without a
filesystem.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: Name of the bucket holding derived output, sibling of the input data.
TFM_DATA_BUCKET = "TFM_data"

#: Filename of the sole data artifact written into each experiment's bucket.
RESULTS_FILENAME = "TFM_results.ome.tif"

#: Bucket holding the cross-experiment aggregate output (summary + sidecars).
AGGREGATE_BUCKET = "TFM_aggregate"


@dataclass
class OutputPlan:
    """Mapping from each input folder (original string) to its output directory."""

    output_dirs: Dict[str, Path]
    warnings: List[str] = field(default_factory=list)


def resolve_output_plan(
    input_folders: Sequence[str],
    processed_root: Optional[str] = None,
) -> OutputPlan:
    """Map every input folder to the directory that holds its derived output.

    Each output directory holds ``<experiment>.ntfm`` (the experiment name is the
    input folder's basename), ``figures/`` and ``batch.log``.

    Args:
        input_folders: the batch input folders, as originally supplied.
        processed_root: the configured processed root, or empty/``None`` for the
            in-place mode.

    Returns:
        An :class:`OutputPlan` keyed by the original folder strings, plus any
        warnings to surface to the user.
    """
    warnings: List[str] = []

    if not processed_root:
        # In-place: TFM_data sits as a sibling of each raw input folder,
        # namespaced by the folder's own basename so siblings can't collide.
        output_dirs = {
            original: Path(original).parent / TFM_DATA_BUCKET / Path(original).name
            for original in input_folders
        }
        return OutputPlan(output_dirs, warnings)

    root = Path(processed_root)
    base = _mirror_base(input_folders, warnings)

    output_dirs: Dict[str, Path] = {}
    for original in input_folders:
        folder = Path(original)
        mirror = root / folder.name if base is None else root / folder.relative_to(base)
        output_dirs[original] = mirror.parent / TFM_DATA_BUCKET / mirror.name

    if base is None:
        _warn_basename_collisions(input_folders, warnings)

    return OutputPlan(output_dirs, warnings)


def experiment_output_dir(
    experiment_path: str, processed_root: Optional[str] = None
) -> Path:
    """The single directory that holds one experiment's derived output.

    The one resolver every consumer must share: the batch writer, the
    experiments-list status dots, and the interactive persist path all call this
    so they can never disagree about *where* an experiment's ``.ntfm`` lives.
    """
    plan = resolve_output_plan([str(experiment_path)], processed_root)
    return plan.output_dirs[str(experiment_path)]


def experiment_ntfm_path(
    experiment_path: str, processed_root: Optional[str] = None
) -> Path:
    """The canonical container path for one experiment.

    The on-disk artifact is a single multi-series OME-TIFF (``RESULTS_FILENAME``);
    the helper name is kept for continuity with the rest of the pipeline.
    """
    out_dir = experiment_output_dir(experiment_path, processed_root)
    return out_dir / RESULTS_FILENAME


def aggregate_output_dir(
    input_folders: Sequence[str], processed_root: Optional[str] = None
) -> Path:
    """Directory that holds the pooled cross-experiment summary (+ sidecars).

    Processed-root set → ``<root>/TFM_aggregate``. In-place → ``TFM_aggregate``
    at the longest common parent of the input folders (a peer of their
    ``TFM_data`` buckets), falling back to the first folder's parent when there is
    no common parent (a single folder, or disconnected roots).
    """
    if processed_root:
        return Path(processed_root) / AGGREGATE_BUCKET
    folders = [Path(f) for f in input_folders]
    if not folders:
        return Path.cwd() / AGGREGATE_BUCKET
    base = _mirror_base([str(f) for f in folders], [])
    if base is None:
        return folders[0].parent / AGGREGATE_BUCKET
    return base / AGGREGATE_BUCKET


def _mirror_base(input_folders: Sequence[str], warnings: List[str]) -> Optional[Path]:
    """Longest common parent of all folders, or ``None`` to signal basename fallback."""
    if len(input_folders) < 2:
        # A single folder has no meaningful mirror tree — use its basename.
        return None
    try:
        return Path(os.path.commonpath([str(f) for f in input_folders]))
    except ValueError:
        warnings.append(
            "Input folders are on disconnected roots (no common parent); "
            "falling back to folder basenames flat under the processed root."
        )
        return None


def _warn_basename_collisions(
    input_folders: Sequence[str], warnings: List[str]
) -> None:
    """Surface basename clashes that would overwrite each other in fallback mode."""
    names = [Path(f).name for f in input_folders]
    duplicates = sorted({name for name, count in Counter(names).items() if count > 1})
    if duplicates:
        warnings.append(
            "Folder basenames collide under the processed root: "
            f"{', '.join(duplicates)}. Outputs would overwrite each other."
        )
